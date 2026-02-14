# %% [markdown]
# # Clean Landcover Samples
# 1) Removes "grass" samples within 50m of pixels classified as crop
#    in at least 4 out of 6 years in dw_landcover_{2019..2024} assets.
# 2) Keeps only samples that match their class in >= N years
#    across dw_landcover assets (configurable threshold).
# Saves removed points to separate CSVs.

# %%
import sys
from pathlib import Path
import math

sys.path.insert(0, str(Path(__file__).parent))

import ee
import pandas as pd
import __config__
from src.gee_utils import initialize_earth_engine

# Initialize
print("Initializing Earth Engine...")
initialize_earth_engine()

study_area_config = __config__.get_study_area()
study_area = study_area_config["geometry"]
print(f"Study area: {study_area_config['name']}")

# Output directory
output_dir = Path("output") / __config__.CURRENT_STUDY_AREA / "training"
output_dir.mkdir(parents=True, exist_ok=True)

# Input/Output paths
input_csv = output_dir / "landcover_samples.csv"
cleaned_csv = output_dir / "landcover_samples_cleaned.csv"
removed_csv = output_dir / "landcover_samples_removed_grass_near_crop.csv"
removed_inconsistent_csv = output_dir / "landcover_samples_removed_inconsistent.csv"

if not input_csv.exists():
    raise FileNotFoundError(f"Missing input CSV: {input_csv}")

# Configuration
LANDCOVER = __config__.LANDCOVER
years = LANDCOVER["years"]
asset_folder = LANDCOVER["asset_folder"]
asset_prefix = LANDCOVER["asset_prefix"]
scale = LANDCOVER["scale"]

# Crop class from training mask config (Dynamic World crop class = 4)
crop_class = __config__.TRAINING_CROP_MASK.get("crop_class", 4)

# Find grass class value from config (fallback to 2)
class_names = LANDCOVER["class_names"]
try:
    grass_class_value = next(k for k, v in class_names.items() if v == "grass")
except StopIteration:
    grass_class_value = 2

# Parameters
min_crop_years = 4
buffer_m = LANDCOVER.get("grass_buffer_m", 50)
batch_size = 2000
min_consistent_years = LANDCOVER.get("consistency_threshold_years", len(years))

if min_consistent_years < 1:
    raise ValueError("LANDCOVER.consistency_threshold_years must be >= 1")
if min_consistent_years > len(years):
    print(
        f"Warning: consistency_threshold_years ({min_consistent_years}) exceeds number of years "
        f"({len(years)}). Using {len(years)} instead."
    )
    min_consistent_years = len(years)

print("\n" + "=" * 60)
print("CLEANING LANDCOVER SAMPLES")
print("=" * 60)
print(f"Input CSV: {input_csv}")
print(f"Years: {years}")
print(f"Crop class: {crop_class}")
print(f"Grass class value: {grass_class_value}")
print(f"Crop frequency threshold: >= {min_crop_years} years")
print(f"Buffer distance: {buffer_m}m")
print(f"Class consistency threshold: >= {min_consistent_years} years")
print("=" * 60)

# Load samples
df = pd.read_csv(input_csv)
if df.empty:
    raise ValueError("Input landcover_samples.csv is empty.")

required_cols = {"lon", "lat", "class_name", "class_value"}
missing_cols = required_cols - set(df.columns)
if missing_cols:
    raise ValueError(f"Missing required columns: {sorted(missing_cols)}")

# Filter grass samples
is_grass = (df["class_value"] == grass_class_value) | (df["class_name"].str.lower() == "grass")
df_grass = df[is_grass].copy()
print(f"Total samples: {len(df)}")
print(f"Grass samples: {len(df_grass)}")

if df_grass.empty:
    print("No grass samples found; saving original file as cleaned output.")
    df.to_csv(cleaned_csv, index=False)
    print(f"✓ Saved cleaned samples to {cleaned_csv}")
    sys.exit(0)

# Build crop frequency mask
print("\nBuilding crop frequency mask...")
images = []
for year in years:
    asset_id = f"{asset_folder}/{asset_prefix}_{year}"
    img = ee.Image(asset_id).eq(crop_class).rename("crop")
    images.append(img)

crop_count = ee.ImageCollection.fromImages(images).sum().rename("crop_count")
frequent_crop = crop_count.gte(min_crop_years).rename("frequent_crop")

# Buffer crop pixels by 50m
# Expand crop mask to include pixels within 50m of frequent crop pixels
near_crop = frequent_crop.focal_max(radius=buffer_m, units="meters").rename("near_crop").unmask(0)

# Sample near-crop mask at grass point locations
print("\nSampling near-crop mask at grass points...")

flags = {}
indices = df_grass.index.tolist()
num_batches = int(math.ceil(len(indices) / batch_size))

for b in range(num_batches):
    batch_idx = indices[b * batch_size:(b + 1) * batch_size]
    batch = df_grass.loc[batch_idx]

    features = []
    for idx, row in batch.iterrows():
        geom = ee.Geometry.Point([float(row["lon"]), float(row["lat"])])
        features.append(ee.Feature(geom, {"idx": int(idx)}))

    fc = ee.FeatureCollection(features)
    sampled = near_crop.sampleRegions(
        collection=fc,
        properties=["idx"],
        scale=scale,
        geometries=False
    )

    info = sampled.getInfo()
    for f in info["features"]:
        props = f.get("properties", {})
        idx = props.get("idx")
        if idx is not None:
            flags[int(idx)] = int(round(props.get("near_crop", 0)))

    print(f"  Batch {b + 1}/{num_batches}: {len(batch)} points")

# Default missing flags to 0 (not near crop)
near_crop_flags = df_grass.index.map(lambda i: flags.get(i, 0))

# Separate removed vs kept
remove_mask = near_crop_flags.astype(bool)
removed_grass = df_grass[remove_mask].copy()
kept_grass = df_grass[~remove_mask].copy()

# Combine cleaned dataset
cleaned_df = pd.concat([df[~is_grass], kept_grass], ignore_index=True)

# Consistency check across DW landcover years for each class
if cleaned_df.empty:
    print("No samples left after grass filtering; saving empty outputs.")
    cleaned_df.to_csv(cleaned_csv, index=False)
    removed_grass.to_csv(removed_csv, index=False)
    cleaned_df.iloc[0:0].to_csv(removed_inconsistent_csv, index=False)
    sys.exit(0)

print("\nChecking class consistency across DW landcover years...")

dw_images = []
band_names = []
for year in years:
    asset_id = f"{asset_folder}/{asset_prefix}_{year}"
    band_name = f"dw_{year}"
    dw_images.append(ee.Image(asset_id).rename(band_name))
    band_names.append(band_name)

dw_multi = ee.Image.cat(dw_images)

class_value_by_idx = cleaned_df["class_value"].to_dict()
match_counts = {}
indices = cleaned_df.index.tolist()
num_batches = int(math.ceil(len(indices) / batch_size))

for b in range(num_batches):
    batch_idx = indices[b * batch_size:(b + 1) * batch_size]
    batch = cleaned_df.loc[batch_idx]

    features = []
    for idx, row in batch.iterrows():
        geom = ee.Geometry.Point([float(row["lon"]), float(row["lat"])])
        features.append(ee.Feature(geom, {"idx": int(idx)}))

    fc = ee.FeatureCollection(features)
    sampled = dw_multi.sampleRegions(
        collection=fc,
        properties=["idx"],
        scale=scale,
        geometries=False
    )

    info = sampled.getInfo()
    for f in info["features"]:
        props = f.get("properties", {})
        idx = props.get("idx")
        if idx is None:
            continue
        target = int(class_value_by_idx.get(idx, -999))
        count = 0
        for band in band_names:
            val = props.get(band)
            if val is not None and int(round(val)) == target:
                count += 1
        match_counts[int(idx)] = count

    print(f"  Batch {b + 1}/{num_batches}: {len(batch)} points")

# Default missing counts to 0 (inconsistent)
consistency_counts = cleaned_df.index.map(lambda i: match_counts.get(i, 0))
consistent_mask = consistency_counts >= min_consistent_years

removed_inconsistent = cleaned_df[~consistent_mask].copy()
cleaned_df = cleaned_df[consistent_mask].copy()

# Save outputs
removed_grass.to_csv(removed_csv, index=False)
removed_inconsistent.to_csv(removed_inconsistent_csv, index=False)
cleaned_df.to_csv(cleaned_csv, index=False)

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Removed grass samples: {len(removed_grass)}")
print(f"Remaining grass samples: {len(kept_grass)}")
print(f"Removed inconsistent samples: {len(removed_inconsistent)}")
print(f"Cleaned total samples: {len(cleaned_df)}")
print(f"✓ Saved cleaned samples to {cleaned_csv}")
print(f"✓ Saved removed samples to {removed_csv}")
print(f"✓ Saved inconsistent samples to {removed_inconsistent_csv}")
