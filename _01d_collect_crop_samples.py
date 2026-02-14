# %% [markdown]
# # Training Data Collector (Asset-based Export)
# Exports S2 indices at all sample points as GEE assets for reliable processing

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import ee
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import time
from collections import defaultdict

import __config__
from src.gee_utils import (
    initialize_earth_engine,
    scale_s2, add_s2_indices,
    apply_cloud_score_plus_mask, load_training_crop_mask,
    asset_exists, delete_asset, ensure_folder_exists,
)

# Initialize
print("Initializing Earth Engine...")
initialize_earth_engine()
study_area_config = __config__.get_study_area()
study_area = study_area_config['geometry']

print(f"Study area: {study_area_config['name']}")
analysis_start = __config__.START_DATE
analysis_end = __config__.END_DATE
start_year = int(analysis_start[:4])
end_year = int(analysis_end[:4])
print(f"Analysis period: {analysis_start} to {analysis_end}")

# Asset configuration
ASSET_FOLDER = (
    f'projects/{__config__.PROJECT_ID}/assets/{__config__.CURRENT_STUDY_AREA}/'
    f'{__config__.TRAINING_ASSET_SUBFOLDER}'
)
print(f"Asset folder: {ASSET_FOLDER}")

# Setup output directories
output_dir = Path('output') / __config__.CURRENT_STUDY_AREA / 'training'
figures_dir = output_dir / 'figures'
timeseries_dir = output_dir / 'timeseries'
output_dir.mkdir(parents=True, exist_ok=True)
figures_dir.mkdir(parents=True, exist_ok=True)
timeseries_dir.mkdir(parents=True, exist_ok=True)


# %% Helper functions for asset management

def wait_for_tasks(task_ids, check_interval=30):
    """Wait for multiple export tasks to complete."""
    print(f"\n⏳ Waiting for {len(task_ids)} export tasks...")
    
    pending = set(task_ids)
    completed = set()
    failed = set()
    
    while pending:
        time.sleep(check_interval)
        
        for task_id in list(pending):
            status = ee.data.getTaskStatus(task_id)[0]
            state = status['state']
            
            if state == 'COMPLETED':
                pending.remove(task_id)
                completed.add(task_id)
                desc = status.get('description', task_id)
                print(f"  ✓ {desc} completed")
            elif state == 'FAILED':
                pending.remove(task_id)
                failed.add(task_id)
                desc = status.get('description', task_id)
                error = status.get('error_message', 'Unknown error')
                print(f"  ✗ {desc} failed: {error}")
            elif state == 'CANCELLED':
                pending.remove(task_id)
                failed.add(task_id)
                print(f"  ⚠️ Task {task_id} was cancelled")
        
        if pending:
            print(f"  ... {len(pending)} tasks still running, {len(completed)} completed")
    
    return completed, failed


# %% Load crop masks
print("\nLoading crop masks...")
t0 = time.time()
mask_config = __config__.TRAINING_CROP_MASK
mask_source = mask_config['source']
crop_mask = load_training_crop_mask(mask_config, verbose=True)
print(f"  ⏱️  Crop masks took {time.time()-t0:.1f}s")


# %% Sample points
print("\n" + "="*60)
print("STEP 1: SAMPLING POINTS")
print("="*60)

start_id = __config__.START_SAMPLE_ID
end_id = __config__.END_SAMPLE_ID
num_samples = end_id - start_id + 1
print(f"Sample ID range: {start_id:03d} to {end_id:03d} ({num_samples} total)")

def sample_points_from_mask(mask, num, seed, source):
    """Sample points from a crop mask."""
    sample_scale = 10
    oversample_factor = 50
    
    try:
        points = mask.selfMask().sample(
            region=study_area,
            scale=sample_scale,
            numPixels=num * oversample_factor,
            seed=seed,
            geometries=True
        ).limit(num)
        
        coords_list = points.aggregate_array('.geo').getInfo()
        result = [(i + start_id, g['coordinates'][0], g['coordinates'][1], source) 
                  for i, g in enumerate(coords_list)]
        if result:
            print(f"  ✓ Sampled {len(result)} {source} points")
            return result
    except Exception as e:
        print(f"  ⚠️  Sampling failed: {str(e)[:60]}...")
    return []

print(f"\nSampling from: {mask_source}")
t0_sample = time.time()
sampled_points = sample_points_from_mask(crop_mask, num_samples, 42, mask_source)
print(f"⏱️  Sampling took {time.time()-t0_sample:.1f}s")
print(f"Total sampled: {len(sampled_points)} points")

if len(sampled_points) < num_samples:
    print(f"⚠️  Only {len(sampled_points)}/{num_samples} points available")
    num_samples = len(sampled_points)

# Create FeatureCollection of all sample points
print("\nCreating point FeatureCollection...")
points_list = [
    ee.Feature(ee.Geometry.Point([lon, lat]), {'sample_id': sample_id})
    for sample_id, lon, lat, source in sampled_points
]
points_fc = ee.FeatureCollection(points_list)
print(f"  ✓ Created FeatureCollection with {len(points_list)} points")

# Store point metadata for later
point_metadata = {p[0]: {'lon': p[1], 'lat': p[2], 'source': p[3]} for p in sampled_points}

# Save point metadata to CSV (merge with existing)
points_df = pd.DataFrame([
    {'sample_id': p[0], 'lon': p[1], 'lat': p[2], 'source': p[3]}
    for p in sampled_points
])
points_csv = output_dir / 'sampled_points_coords.csv'
if points_csv.exists():
    df_existing = pd.read_csv(points_csv)
    new_ids = set(points_df['sample_id'])
    df_existing_keep = df_existing[~df_existing['sample_id'].isin(new_ids)]
    points_df = pd.concat([df_existing_keep, points_df], ignore_index=True)
    points_df = points_df.sort_values('sample_id').reset_index(drop=True)
points_df.to_csv(points_csv, index=False)
print(f"  ✓ Saved point coordinates to {points_csv} ({len(points_df)} total)")


# %% Load S2 imagery
print("\n" + "="*60)
print("STEP 2: LOADING SENTINEL-2 IMAGERY")
print("="*60)
t0 = time.time()

# Get bounding box of all points for efficient filtering
lons = [p[1] for p in sampled_points]
lats = [p[2] for p in sampled_points]
points_bbox = ee.Geometry.Rectangle([min(lons)-0.1, min(lats)-0.1, max(lons)+0.1, max(lats)+0.1])

# Load Cloud Score+
cs_plus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')\
    .filterBounds(points_bbox)\
    .filterDate(analysis_start, analysis_end)

# Load S2 with masking and indices
clear_threshold = getattr(__config__, 'CLOUD_SCORE_THRESHOLD', 0.60)

s2_base = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')\
    .filterBounds(points_bbox)\
    .filterDate(analysis_start, analysis_end)\
    .linkCollection(cs_plus, ['cs'])\
    .map(lambda img: apply_cloud_score_plus_mask(img, clear_threshold))\
    .map(scale_s2)\
    .map(add_s2_indices)

print(f"  ⏱️  S2 setup took {time.time()-t0:.1f}s")


# %% Export extractions as assets (one per year)
print("\n" + "="*60)
print("STEP 3: EXPORT EXTRACTIONS AS ASSETS")
print("="*60)

# Ensure asset folder exists
ensure_folder_exists(ASSET_FOLDER)

def build_year_extraction(year):
    """Build a FeatureCollection of all point extractions for one year."""
    year_start = f'{year}-01-01'
    year_end = f'{year}-12-31'
    
    year_col = s2_base.filterDate(year_start, year_end)
    
    def extract_from_image(img):
        """Extract indices at all points from a single image."""
        indices = img.select(['NDVI', 'NDWI', 'NDMI'])
        
        extracted = indices.reduceRegions(
            collection=points_fc,
            reducer=ee.Reducer.mean(),
            scale=10
        )
        
        timestamp = img.get('system:time_start')
        return extracted.map(lambda f: f.set('timestamp', timestamp))
    
    # Map over all images and flatten
    return year_col.map(extract_from_image).flatten()

# Check which years need export
years_to_export = []
years_existing = []

for year in range(start_year, end_year + 1):
    asset_id = f'{ASSET_FOLDER}/s2_extractions_{year}'
    if asset_exists(asset_id):
        years_existing.append(year)
    else:
        years_to_export.append(year)

if years_existing:
    print(f"\n✓ Found existing assets for years: {years_existing}")

print(f"\n📤 Missing years to export: {years_to_export if years_to_export else 'none'}")

# Ask user whether to proceed (always offer options)
print("\nOptions:")
print("  1. Export missing years only")
print("  2. Overwrite ALL years (replace existing assets)")
print("  3. Skip export (use existing assets only)")
print("  4. Cancel")

choice = input("\nChoice [1/2/3/4]: ").strip()

if choice == '4':
    print("Cancelled.")
    sys.exit(0)
elif choice == '3':
    years_to_export = []
    print("Skipping export, will use existing assets only.")
elif choice == '2':
    # Re-export all (overwrite=True will handle existing assets)
    years_to_export = list(range(start_year, end_year + 1))
    print(f"Will re-export all years: {years_to_export}")
else:
    # Default to missing only
    if not years_to_export:
        print("No missing years to export.")

if years_to_export:
    print(f"\nSubmitting {len(years_to_export)} export tasks...")
    task_ids = []
    
    for year in years_to_export:
        asset_id = f'{ASSET_FOLDER}/s2_extractions_{year}'
        
        # Build the extraction FeatureCollection
        year_fc = build_year_extraction(year)
        
        # Submit export task (overwrite=True handles existing assets)
        task = ee.batch.Export.table.toAsset(
            collection=year_fc,
            description=f's2_extractions_{year}',
            assetId=asset_id,
            overwrite=True
        )
        task.start()
        task_ids.append(task.id)
        print(f"  📤 Submitted: s2_extractions_{year}")
    
    # Wait for all tasks to complete
    completed, failed = wait_for_tasks(task_ids, check_interval=30)
    
    if failed:
        print(f"\n⚠️  {len(failed)} exports failed. Check GEE task manager for details.")
        print("You can re-run this script to retry failed years.")
    else:
        print(f"\n✓ All {len(completed)} exports completed successfully!")
else:
    print("\n✓ All years already exported, proceeding to load...")


# %% Load from assets and process
print("\n" + "="*60)
print("STEP 4: LOAD FROM ASSETS")
print("="*60)

all_extractions = []

for year in range(start_year, end_year + 1):
    asset_id = f'{ASSET_FOLDER}/s2_extractions_{year}'
    
    if not asset_exists(asset_id):
        print(f"  ⚠️  Skipping {year} - asset not found")
        continue
    
    print(f"  Loading {year}...", end=' ', flush=True)
    t0 = time.time()
    
    try:
        fc = ee.FeatureCollection(asset_id).filter(
            ee.Filter.notNull(['sample_id', 'timestamp', 'NDVI', 'NDWI', 'NDMI'])
        )
        
        # Use aggregate_array for efficient retrieval
        props = ['sample_id', 'timestamp', 'NDVI', 'NDWI', 'NDMI']
        server_dict = ee.Dictionary({prop: fc.aggregate_array(prop) for prop in props})
        arrays = server_dict.getInfo()
        
        lengths = {k: len(v) for k, v in arrays.items() if isinstance(v, list)}
        n_records = lengths.get('sample_id', 0)
        min_len = min(lengths.values()) if lengths else 0
        print(f"✓ {n_records:,} records ({time.time()-t0:.1f}s)")
        if lengths and len(set(lengths.values())) != 1:
            print(f"  ⚠️  Property array length mismatch: {lengths}. Using min length {min_len}.")

        for i in range(min_len):
            all_extractions.append({
                'sample_id': arrays.get('sample_id', [None] * min_len)[i],
                'timestamp': arrays.get('timestamp', [None] * min_len)[i],
                'NDVI': arrays.get('NDVI', [None] * min_len)[i],
                'NDWI': arrays.get('NDWI', [None] * min_len)[i],
                'NDMI': arrays.get('NDMI', [None] * min_len)[i]
            })
            
    except Exception as e:
        print(f"✗ Failed: {e}")

print(f"\n📊 Total records loaded: {len(all_extractions):,}")
print(f"   Avg per point: {len(all_extractions)/num_samples:.0f} observations")


# %% Organize data by point
print("\n" + "="*60)
print("STEP 5: ORGANIZE DATA BY POINT")
print("="*60)

point_data = defaultdict(list)
for rec in all_extractions:
    point_data[rec['sample_id']].append({
        'timestamp': rec['timestamp'],
        'NDVI': rec['NDVI'],
        'NDWI': rec['NDWI'],
        'NDMI': rec['NDMI']
    })

print(f"  Points with data: {len(point_data)}/{num_samples}")


# %% Analyze and save
print("\n" + "="*60)
print("STEP 6: ANALYZE POINTS & SAVE")
print("="*60)

sampled_results = []
skipped_no_data = 0
skipped_no_dry = 0

for sample_id in sorted(point_data.keys()):
    records = point_data[sample_id]
    meta = point_metadata.get(sample_id, {})
    lon = meta.get('lon', 0)
    lat = meta.get('lat', 0)
    source = meta.get('source', 'unknown')
    
    # Convert to DataFrame
    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.sort_values('date').drop(columns=['timestamp'])
    
    # Compute composite index
    df['composite_index'] = (df['NDVI'] - df['NDWI'] + df['NDMI']) / 3
    
    # Filter valid NDVI
    df_valid = df[df['NDVI'].notna()].copy()
    
    if len(df_valid) < 10:
        skipped_no_data += 1
        continue
    
    # Calculate dry season statistics
    dry_mask = df_valid['date'].dt.month.isin(__config__.PEAK_DRY_MONTHS)
    df_dry = df_valid[dry_mask]
    
    if len(df_dry) == 0:
        skipped_no_dry += 1
        continue
    
    mean_dry_ndvi = df_dry['NDVI'].mean()
    mean_dry_ndwi = df_dry['NDWI'].mean()
    mean_dry_ndmi = df_dry['NDMI'].mean()
    
    # Save time series
    csv_path = timeseries_dir / f"sample_{sample_id:03d}_s2.csv"
    df[['date', 'NDVI', 'NDWI', 'NDMI', 'composite_index']].to_csv(csv_path, index=False)
    
    # Generate figure
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # Dry season shading
    for year in range(start_year, end_year + 1):
        dry_start = pd.Timestamp(f'{year}-{__config__.DRY_SEASON_MONTHS[0]:02d}-01')
        dry_end = pd.Timestamp(f'{year}-{__config__.DRY_SEASON_MONTHS[-1]:02d}-30')
        ax.axvspan(dry_start, dry_end, alpha=0.15, color='orange',
                   label='Dry Season' if year == start_year else '')
    
    ax.plot(df['date'], df['NDVI'], 'o-', color='darkgreen', linewidth=1, markersize=2, alpha=0.4,
            label=f'NDVI (dry: {mean_dry_ndvi:.3f})')
    ax.plot(df['date'], -df['NDWI'], 's-', color='blue', linewidth=1, markersize=2, alpha=0.4,
            label=f'-NDWI (dry: {-mean_dry_ndwi:.3f})')
    ax.plot(df['date'], df['NDMI'], '^-', color='brown', linewidth=1, markersize=2, alpha=0.4,
            label=f'NDMI (dry: {mean_dry_ndmi:.3f})')
    ax.plot(df['date'], df['composite_index'], 'D-', color='purple', linewidth=2, markersize=4,
            alpha=0.9, label='Composite Index', zorder=10)
    
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Index Value')
    ax.set_title(f'Sample {sample_id:03d} - [{lon:.4f}, {lat:.4f}]')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.4, 1.0)
    
    fig.tight_layout()
    fig.savefig(figures_dir / f"sample_{sample_id:03d}.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    sampled_results.append({
        'sample_num': sample_id,
        'lon': lon,
        'lat': lat,
        'source': source,
        'dry_ndvi': mean_dry_ndvi,
        'dry_ndwi': mean_dry_ndwi,
        'dry_ndmi': mean_dry_ndmi,
        'n_s2': len(df),
        'n_s2_valid': len(df_valid)
    })

print(f"\n✓ Processed {len(sampled_results)} valid points")
print(f"  Skipped (insufficient data): {skipped_no_data}")
print(f"  Skipped (no dry season): {skipped_no_dry}")


# %% Save results
if sampled_results:
    df_results = pd.DataFrame(sampled_results)
    csv_path = output_dir / 'sampled_points_data.csv'
    
    # Merge with existing if present
    if csv_path.exists():
        df_existing = pd.read_csv(csv_path)
        print(f"\n📂 Found existing data with {len(df_existing)} samples")
        
        new_ids = set(df_results['sample_num'])
        df_existing_keep = df_existing[~df_existing['sample_num'].isin(new_ids)]
        df_results = pd.concat([df_existing_keep, df_results], ignore_index=True)
        df_results = df_results.sort_values('sample_num').reset_index(drop=True)
        print(f"  Total after merge: {len(df_results)}")
    
    df_results.to_csv(csv_path, index=False)
    print(f"\n✓ Saved to {csv_path}")
    print(f"\nSummary:")
    print(f"  Total samples: {len(df_results)}")
    print(f"  Mean dry NDVI: {df_results['dry_ndvi'].mean():.3f}")
else:
    print("\n⚠️  No valid results to save")

print("\n" + "="*60)
print("DONE")
print("="*60)

