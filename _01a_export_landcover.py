# %% [markdown]
# # Export Yearly Land Cover Composites from Dynamic World
# Export yearly modal classification layers to GEE assets

# %%
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import ee
import time
import __config__
from src.gee_utils import initialize_earth_engine

# Initialize
print("Initializing Earth Engine...")
initialize_earth_engine()

study_area_config = __config__.get_study_area()
study_area = study_area_config['geometry']
print(f"Study area: {study_area_config['name']}")

# %% Configuration from __config__.py
YEARS = __config__.LANDCOVER['years']
ASSET_FOLDER = __config__.LANDCOVER['asset_folder']
ASSET_PREFIX = __config__.LANDCOVER['asset_prefix']
SCALE = __config__.LANDCOVER['scale']

# %% Export function
def export_yearly_classification(year):
    """Export yearly modal land cover classification to GEE asset.
    
    Parameters
    ----------
    year : int
        Year to process
    
    Returns
    -------
    tuple
        (ee.batch.Task, asset_name)
    """
    # Load Dynamic World for entire year
    dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1')\
        .filterBounds(study_area)\
        .filterDate(f'{year}-01-01', f'{year}-12-31')
    
    # Modal class for the year
    label_mode = dw.select('label').mode()\
        .set({
            'year': year,
            'study_area': __config__.CURRENT_STUDY_AREA,
            'source': 'GOOGLE/DYNAMICWORLD/V1'
        })
    
    # Asset name
    asset_name = f'{ASSET_FOLDER}/{ASSET_PREFIX}_{year}'
    
    # Create export task
    task = ee.batch.Export.image.toAsset(
        image=label_mode,
        description=f'{ASSET_PREFIX}_{year}',
        assetId=asset_name,
        region=study_area,
        scale=SCALE,
        maxPixels=1e13,
        pyramidingPolicy={'label': 'mode'}
    )
    
    return task, asset_name

# %% Main export loop
print("\n" + "="*60)
print("EXPORTING YEARLY LAND COVER CLASSIFICATIONS")
print("="*60)
print(f"Years: {YEARS}")
print(f"Scale: {SCALE}m")
print(f"Asset folder: {ASSET_FOLDER}")
print(f"Total exports: {len(YEARS)}")
print("="*60)

tasks = []
t0 = time.time()

for year in YEARS:
    print(f"  Submitting {year}...", end=" ", flush=True)
    
    try:
        task, asset_name = export_yearly_classification(year)
        task.start()
        tasks.append({
            'task': task,
            'year': year,
            'asset': asset_name
        })
        print(f"✓ {asset_name.split('/')[-1]}")
    except Exception as e:
        print(f"✗ Error: {str(e)[:50]}")

print(f"\n⏱️ Submitted {len(tasks)} tasks in {time.time()-t0:.1f}s")

# %% Monitor tasks
print("\n" + "="*60)
print("TASK STATUS")
print("="*60)
print("Tasks are running in the background on GEE servers.")
print("Monitor at: https://code.earthengine.google.com/tasks")

print("\nSubmitted tasks:")
for t in tasks:
    print(f"  {t['year']}: {t['task'].id}")

# %%
