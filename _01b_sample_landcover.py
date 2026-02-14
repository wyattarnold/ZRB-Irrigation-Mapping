# %% [markdown]
# # Sample Land Cover Points from Dynamic World
# Efficiently sample high-confidence points for non-crop land cover classes
# Includes mapping visualization with DW assets and crop mask

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import ee
import pandas as pd
import time
import __config__
from src.gee_utils import initialize_earth_engine

# Initialize
print("Initializing Earth Engine...")
initialize_earth_engine()

study_area_config = __config__.get_study_area()
study_area = study_area_config['geometry']
print(f"Study area: {study_area_config['name']}")

# Setup output directory
output_dir = Path('output') / __config__.CURRENT_STUDY_AREA / 'training'
output_dir.mkdir(parents=True, exist_ok=True)

# %% Configuration from config.py
TARGET_CLASSES = __config__.LANDCOVER['target_classes']
CLASS_NAMES = __config__.LANDCOVER['class_names']
SAMPLES_PER_CLASS_PER_YEAR = __config__.LANDCOVER['samples_per_class_per_year']
SCALE = __config__.LANDCOVER['scale']
YEARS = __config__.LANDCOVER['years']
ASSET_FOLDER = __config__.LANDCOVER['asset_folder']
ASSET_PREFIX = __config__.LANDCOVER['asset_prefix']

# %% Sample from exported yearly assets
print("\n" + "="*60)
print("SAMPLING LAND COVER POINTS FROM DYNAMIC WORLD")
print("="*60)
print(f"Classes: {list(CLASS_NAMES.values())}")
print(f"Samples per class per year: {SAMPLES_PER_CLASS_PER_YEAR}")
print(f"Scale: {SCALE}m")
print(f"Years: {YEARS}")
print(f"Asset folder: {ASSET_FOLDER}")
print(f"Total expected: ~{SAMPLES_PER_CLASS_PER_YEAR * len(YEARS) * len(TARGET_CLASSES)} samples")
print("="*60)

t0 = time.time()

# Grid parameters for spatial uniformity
GRID_SIZE_DEG = 0.30  # ~33km cells
MAX_PER_CELL = 200     # Max samples per grid cell per class
OVERSAMPLE_FACTOR = 20  # Oversample aggressively to ensure enough pixels

def sample_class_spatially_uniform(year, class_value, n_samples):
    """Sample a class with spatial uniformity using grid-based thinning.
    
    Strategy:
    1. Use stratifiedSample which better handles rare classes
    2. Assign each point to a spatial grid cell
    3. Limit points per grid cell to ensure spatial spread (skip if not enough samples)
    4. Return up to n_samples total
    """
    label_mode = ee.Image(f'{ASSET_FOLDER}/{ASSET_PREFIX}_{year}')
    
    # Mask to just this class
    class_mask = label_mode.eq(class_value)
    masked = label_mode.updateMask(class_mask)
    
    # Create grid ID band based on coordinates
    lon_lat = ee.Image.pixelLonLat()
    grid_x = lon_lat.select('longitude').divide(GRID_SIZE_DEG).floor().toInt()
    grid_y = lon_lat.select('latitude').divide(GRID_SIZE_DEG).floor().toInt()
    grid_id = grid_x.multiply(100000).add(grid_y).rename('grid_id')
    
    # Add grid ID to image
    image_with_grid = masked.addBands(grid_id)
    
    # Use stratifiedSample - better for rare classes than sample()
    # Request more than needed to account for spatial thinning
    target_pixels = n_samples * OVERSAMPLE_FACTOR
    samples = image_with_grid.stratifiedSample(
        numPoints=target_pixels,
        classBand='label',
        classValues=[class_value],
        classPoints=[target_pixels],
        region=study_area,
        scale=SCALE,
        seed=year * 100 + class_value,
        geometries=True,
        dropNulls=True
    )
    
    # Add random column for within-cell selection
    samples = samples.randomColumn('random', year * 100 + class_value)
    
    # Check how many samples we got - if barely enough, skip spatial thinning
    sample_count = samples.size()
    
    # Conditional: only apply grid thinning if we have plenty of samples
    # If sample_count < n_samples * 1.5, just take all and randomly select
    def apply_thinning():
        """Apply grid-based thinning when we have excess samples."""
        distinct_grids = samples.aggregate_array('grid_id').distinct()
        
        def thin_grid_cell(grid_id_obj):
            gid = ee.Number(grid_id_obj)
            cell_samples = samples.filter(ee.Filter.eq('grid_id', gid))
            return cell_samples.sort('random').limit(MAX_PER_CELL)
        
        return ee.FeatureCollection(distinct_grids.map(thin_grid_cell)).flatten()
    
    def skip_thinning():
        """Return all samples when class is rare."""
        return samples
    
    # Use all samples if rare, otherwise thin spatially
    threshold = ee.Number(n_samples).multiply(1.5)
    thinned = ee.Algorithms.If(sample_count.lt(threshold), skip_thinning(), apply_thinning())
    thinned = ee.FeatureCollection(thinned)
    
    # Final shuffle and limit to target
    final = thinned.randomColumn('final_random', year).sort('final_random').limit(n_samples)
    
    # Add year and ensure label is set
    return final.map(lambda f: f.set('year', year).set('label', class_value))

# Sample each year and class separately to stay under 5000 element limit
print("\nSampling by year and class (spatially uniform):")
print(f"  Grid cell size: {GRID_SIZE_DEG}° (~{GRID_SIZE_DEG * 111:.0f}km)")
print(f"  Max per cell: {MAX_PER_CELL}")
all_features = []
for year in YEARS:
    print(f"  {year}:", end=" ", flush=True)
    year_count = 0
    t1 = time.time()
    
    for class_val in TARGET_CLASSES:
        class_name = CLASS_NAMES.get(class_val, f'class_{class_val}')
        samples = sample_class_spatially_uniform(year, class_val, SAMPLES_PER_CLASS_PER_YEAR)
        
        # Get features (should be <= 1000, well under 5000 limit)
        class_features = samples.getInfo()['features']
        all_features.extend(class_features)
        year_count += len(class_features)
        print(f"{class_name[:3]}:{len(class_features)}", end=" ", flush=True)
    
    print(f"= {year_count} pts ({time.time()-t1:.1f}s)")

print(f"\nTotal: {len(all_features)} points")
features = all_features

# %% Convert to DataFrame
print("Processing results...", end=" ", flush=True)
t2 = time.time()

results = []
for f in features:
    coords = f['geometry']['coordinates']
    class_val = int(f['properties']['label'])
    results.append({
        'lon': coords[0],
        'lat': coords[1],
        'class_name': CLASS_NAMES.get(class_val, f'class_{class_val}'),
        'class_value': class_val,
        'year': f['properties']['year'],
    })

df = pd.DataFrame(results)
print(f"✓ ({time.time()-t2:.1f}s)")

# %% Summary and save
print(f"\n⏱️ Total time: {time.time()-t0:.1f}s")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Total samples: {len(df)}")
print("\nSamples by class:")
print(df.groupby('class_name').size())
print("\nSamples by year:")
print(df.groupby('year').size())

# Save to CSV
csv_path = output_dir / 'landcover_samples.csv'
df.to_csv(csv_path, index=False)
print(f"\n✓ Saved to {csv_path}")
