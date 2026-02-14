# %% [markdown]
# # Step 3b: Multi-Year Consolidation with Weighted Confidence
#
# This script creates consolidated classification maps by combining multiple years
# of classification results using weighted confidence scores.
#
# **Approach:**
# 1. Load classified assets for all years (from _03a_classification.py)
# 2. Aggregate classes: Rainfed (0,1) → 0, Irrigated (2,3,4) → 1, Others (5-11) → 2-8
# 3. Calculate confidence score for each pixel based on:
#    - Temporal consistency: frequency of classification across years (weighted)
#    - Spatial consistency: agreement with neighboring pixels
# 4. Create two consolidated outputs:
#    - Early-weighted: 2019 weighted highest → 2024 lowest
#    - Late-weighted: 2024 weighted highest → 2019 lowest
#
# **Outputs:**
# - consolidated_early_weighted: Prioritizes earlier years' classifications
# - consolidated_late_weighted: Prioritizes recent years' classifications
# - Each includes: classification band + confidence band
#
# Run AFTER _03a_classification.py has exported classification assets.

# %% [markdown]
## 1. Setup and Initialization

# %%
import sys
from pathlib import Path

import ee
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import __config__

from src.gee_utils import initialize_earth_engine, asset_exists, _retry_with_backoff
from src.classification_utils import wait_for_exports

# Configuration
CONSOL_CONFIG = __config__.CONSOLIDATION
EMB_CONFIG = __config__.EMBEDDINGS_CLASSIFICATION

# Output directory
OUTPUT_DIR = Path('output') / __config__.CURRENT_STUDY_AREA / CONSOL_CONFIG.get('output_subfolder', 'consolidation')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %%
# Initialize Earth Engine
print("Initializing Earth Engine...")
initialize_earth_engine()

study_area_config = __config__.get_study_area()
study_area = study_area_config['geometry']
study_area_name = study_area_config['name']

print("\n" + "="*70)
print("MULTI-YEAR CONSOLIDATION")
print("="*70)
print(f"\nStudy area: {study_area_name}")
print(f"Output directory: {OUTPUT_DIR}")

# %% [markdown]
## 2. Load Configuration and Check Assets

# %%
# Load configuration
years = CONSOL_CONFIG['years']
asset_folder = CONSOL_CONFIG['asset_folder']
asset_prefix = CONSOL_CONFIG['asset_prefix']
class_aggregation = CONSOL_CONFIG['class_aggregation']
aggregated_labels = CONSOL_CONFIG['aggregated_labels']
aggregated_palette = CONSOL_CONFIG['aggregated_palette']

early_weights = CONSOL_CONFIG['early_weights']
late_weights = CONSOL_CONFIG['late_weights']
combined_weights = CONSOL_CONFIG.get('combined_weights', {year: 1.0 for year in years})

use_spatial = CONSOL_CONFIG.get('use_spatial_confidence', True)
neighborhood_radius = CONSOL_CONFIG.get('neighborhood_radius', 1)
spatial_weight = CONSOL_CONFIG.get('spatial_weight', 0.3)
scale = CONSOL_CONFIG.get('scale', 10)
process_combined_only = CONSOL_CONFIG.get('process_combined_only', False)

print(f"\nConfiguration:")
print(f"  Years to consolidate: {years}")
print(f"  Asset folder: projects/{__config__.PROJECT_ID}/assets/{asset_folder}/")
print(f"  Asset prefix: {asset_prefix}")
print(f"  Processing mode: {'Combined-only' if process_combined_only else 'Full (early + late + combined)'}")
print(f"  Spatial confidence: {'Enabled' if use_spatial else 'Disabled'}")
if use_spatial:
    print(f"    Neighborhood radius: {neighborhood_radius} pixels")
    print(f"    Spatial weight: {spatial_weight}")
print(f"  Scale: {scale}m")

# Display class aggregation
print(f"\nClass Aggregation:")
for orig_class, agg_class in sorted(class_aggregation.items()):
    orig_label = EMB_CONFIG.get('cluster_labels', {}).get(orig_class, f'Class {orig_class}')
    agg_label = aggregated_labels.get(agg_class, f'Agg {agg_class}')
    print(f"  {orig_class} ({orig_label}) → {agg_class} ({agg_label})")

# Display temporal weights
print(f"\nTemporal Weights (Early-weighted):")
total_early = sum(early_weights.values())
for year in years:
    w = early_weights.get(year, 0)
    print(f"  {year}: {w:.1f} ({w/total_early*100:.1f}%)")

print(f"\nTemporal Weights (Late-weighted):")
total_late = sum(late_weights.values())
for year in years:
    w = late_weights.get(year, 0)
    print(f"  {year}: {w:.1f} ({w/total_late*100:.1f}%)")

print(f"\nTemporal Weights (Combined all-years):")
total_combined = sum(combined_weights.get(year, 0) for year in years)
for year in years:
    w = combined_weights.get(year, 0)
    pct = (w / total_combined * 100) if total_combined > 0 else 0
    print(f"  {year}: {w:.1f} ({pct:.1f}%)")

# %%
# Check that classification assets exist
print("\nChecking classification assets...")
missing_years = []
available_years = []

for year in years:
    asset_id = f"projects/{__config__.PROJECT_ID}/assets/{asset_folder}/{asset_prefix}_{year}"
    if asset_exists(asset_id):
        print(f"  {year}: ✓ Found")
        available_years.append(year)
    else:
        print(f"  {year}: ✗ Missing")
        missing_years.append(year)

if missing_years:
    print(f"\n⚠️  Missing assets for years: {missing_years}")
    print("   Run _03a_classification.py first to export classification assets.")
    
if len(available_years) < 2:
    print("\n❌ Need at least 2 years of classification data. Exiting.")
    sys.exit(1)

years = available_years
print(f"\n✓ Proceeding with {len(years)} years: {years}")

# %% [markdown]
## 3. Load Classification Assets and Apply Aggregation

# %%
print("\nLoading and aggregating classified images...")

# Build remapping lists for GEE
orig_classes = sorted(class_aggregation.keys())
agg_classes = [class_aggregation[c] for c in orig_classes]
n_aggregated_classes = len(set(agg_classes))

print(f"  Original classes: {orig_classes}")
print(f"  Aggregated to: {sorted(set(agg_classes))}")
print(f"  Number of aggregated classes: {n_aggregated_classes}")

# Load and remap each year's classification
classified_by_year = {}
aggregated_by_year = {}

for year in years:
    asset_id = f"projects/{__config__.PROJECT_ID}/assets/{asset_folder}/{asset_prefix}_{year}"
    classified = ee.Image(asset_id)
    classified_by_year[year] = classified
    
    # Apply class aggregation using remap
    aggregated = classified.remap(orig_classes, agg_classes).rename('classification')
    aggregated_by_year[year] = aggregated
    print(f"  Loaded and aggregated: {year}")

print(f"\n✓ Loaded {len(years)} years of classification data")

# %% [markdown]
## 4. Build Confidence Score Functions

# %%
def calculate_weighted_mode_and_confidence(images_by_year, weights_dict, n_classes, class_labels=None):
    """
    Calculate weighted mode classification, confidence, and per-class probabilities.
    
    For each pixel:
    1. For each class, sum the weights of years where pixel == that class
    2. The class with highest weighted sum is the mode
    3. Confidence = weighted_sum_of_mode / total_weight
    4. Per-class probabilities = weighted sum for each class
    
    Parameters
    ----------
    images_by_year : dict
        {year: ee.Image} with aggregated classification
    weights_dict : dict
        {year: weight} temporal weights
    n_classes : int
        Number of aggregated classes
    class_labels : dict, optional
        {class_id: label} for naming probability bands
    
    Returns
    -------
    tuple of (classification ee.Image, confidence ee.Image, list of per-class prob ee.Images)
    """
    years = sorted(images_by_year.keys())
    
    # Normalize weights
    total_weight = sum(weights_dict[y] for y in years)
    norm_weights = {y: weights_dict[y] / total_weight for y in years}
    
    # For each class, calculate weighted frequency (probability)
    class_probs = []
    for class_id in range(n_classes):
        # Create binary mask for this class in each year, multiply by weight
        weighted_masks = []
        for year in years:
            img = images_by_year[year]
            # Cast to float to ensure homogeneous types across years
            mask = img.eq(class_id).multiply(norm_weights[year]).toFloat()
            weighted_masks.append(mask)
        
        # Sum weighted masks across years
        # Generate band name from label if available
        if class_labels and class_id in class_labels:
            band_name = f'prob_{class_labels[class_id].lower().replace(" ", "_").replace(".", "")}'
        else:
            band_name = f'prob_class_{class_id}'
        
        class_prob = ee.ImageCollection(weighted_masks).sum().rename(band_name)
        class_probs.append(class_prob)
    
    # Stack all class scores into multi-band image for argmax
    score_stack = ee.Image.cat(class_probs)
    
    # Find class with maximum score (weighted mode)
    # argmax returns band index with max value
    classification = score_stack.toArray().arrayArgmax().arrayGet(0).rename('classification')
    
    # Confidence = max class score (already normalized to 0-1)
    confidence = score_stack.reduce(ee.Reducer.max()).rename('confidence')
    
    return classification, confidence, class_probs


def calculate_spatial_confidence(classification_image, neighborhood_radius):
    """
    Calculate spatial consistency score based on neighborhood agreement.
    
    For each pixel, calculates what fraction of neighboring pixels 
    have the same classification.
    
    Parameters
    ----------
    classification_image : ee.Image
        Classification image
    neighborhood_radius : int
        Radius in pixels for neighborhood kernel
    
    Returns
    -------
    ee.Image
        Spatial confidence (0-1)
    """
    # Create kernel for neighborhood
    kernel_size = 2 * neighborhood_radius + 1
    kernel = ee.Kernel.square(neighborhood_radius, 'pixels')
    
    # For each pixel, count how many neighbors have the same class
    # Using focal mode and comparing to original
    
    # Get the mode of the neighborhood
    neighborhood_mode = classification_image.focal_mode(
        radius=neighborhood_radius,
        kernelType='square',
        units='pixels'
    )
    
    # Count total neighbors (excluding self) - this is constant
    total_neighbors = kernel_size * kernel_size
    
    # Create binary: 1 where pixel matches its neighborhood mode
    matches_mode = classification_image.eq(neighborhood_mode)
    
    # Calculate fraction of neighbors with same class using focal mean
    # This gives the proportion of the neighborhood with the same class as center
    neighbor_agreement = classification_image.eq(classification_image).focal_mean(
        radius=neighborhood_radius,
        kernelType='square', 
        units='pixels'
    )
    
    # Actually, better approach: for each pixel, calculate entropy-like measure
    # Simplified: use focal_mode confidence
    # Alternative: calculate fraction of pixels in neighborhood matching center pixel's class
    
    # Expand classification to match itself in neighborhood
    center_class = classification_image
    
    # For each neighbor position, check if it matches center
    # Use neighborhoodToBands then compare
    neighborhood_bands = classification_image.neighborhoodToBands(kernel)
    
    # Count matches with center pixel
    n_bands = kernel_size * kernel_size
    matches = neighborhood_bands.eq(center_class)
    match_count = matches.reduce(ee.Reducer.sum())
    
    # Spatial confidence = fraction of neighbors matching center
    spatial_confidence = match_count.divide(n_bands).rename('spatial_confidence')
    
    return spatial_confidence


def combine_confidence_scores(temporal_confidence, spatial_confidence, spatial_weight):
    """
    Combine temporal and spatial confidence into final score.
    
    Parameters
    ----------
    temporal_confidence : ee.Image
        Temporal consistency score (0-1)
    spatial_confidence : ee.Image
        Spatial consistency score (0-1)
    spatial_weight : float
        Weight for spatial component (0-1)
    
    Returns
    -------
    ee.Image
        Combined confidence score (0-1)
    """
    temporal_weight = 1.0 - spatial_weight
    
    combined = (temporal_confidence.multiply(temporal_weight)
                .add(spatial_confidence.multiply(spatial_weight))
                .rename('confidence'))
    
    return combined

# %% [markdown]
## 5. Calculate Consolidated Classifications

# %%
print("\nCalculating consolidated classifications...")

early_consolidated = None
late_consolidated = None

if not process_combined_only:
    # Early-weighted consolidation
    print("\n--- Early-Weighted Consolidation ---")
    print("  (Prioritizes 2019 classifications)")

    early_class, early_temporal_conf, early_class_probs = calculate_weighted_mode_and_confidence(
        aggregated_by_year, early_weights, n_aggregated_classes, class_labels=aggregated_labels
    )

    if use_spatial:
        early_spatial_conf = calculate_spatial_confidence(early_class, neighborhood_radius)
        early_confidence = combine_confidence_scores(early_temporal_conf, early_spatial_conf, spatial_weight)
        print(f"  ✓ Calculated with spatial confidence (weight={spatial_weight})")
    else:
        early_confidence = early_temporal_conf
        print("  ✓ Calculated (temporal only)")

    # Combine into single image with bands:
    # Band 0: classification (winning class)
    # Band 1: confidence (winning class probability, 0-100)
    # Bands 2+: per-class probabilities (0-100)
    early_prob_bands = [prob.multiply(100).toByte() for prob in early_class_probs]
    early_consolidated = ee.Image.cat(
        [early_class.toByte(), early_confidence.multiply(100).toByte().rename('confidence')] + early_prob_bands
    ).set('weighting', 'early').set('years', years)

    print(f"  Output bands: classification, confidence, + {len(early_class_probs)} per-class probabilities")

    # Late-weighted consolidation
    print("\n--- Late-Weighted Consolidation ---")
    print("  (Prioritizes 2024 classifications)")

    late_class, late_temporal_conf, late_class_probs = calculate_weighted_mode_and_confidence(
        aggregated_by_year, late_weights, n_aggregated_classes, class_labels=aggregated_labels
    )

    if use_spatial:
        late_spatial_conf = calculate_spatial_confidence(late_class, neighborhood_radius)
        late_confidence = combine_confidence_scores(late_temporal_conf, late_spatial_conf, spatial_weight)
        print(f"  ✓ Calculated with spatial confidence (weight={spatial_weight})")
    else:
        late_confidence = late_temporal_conf
        print("  ✓ Calculated (temporal only)")

    # Combine into single image with bands:
    # Band 0: classification (winning class)
    # Band 1: confidence (winning class probability, 0-100)
    # Bands 2+: per-class probabilities (0-100)
    late_prob_bands = [prob.multiply(100).toByte() for prob in late_class_probs]
    late_consolidated = ee.Image.cat(
        [late_class.toByte(), late_confidence.multiply(100).toByte().rename('confidence')] + late_prob_bands
    ).set('weighting', 'late').set('years', years)

    print(f"  Output bands: classification, confidence, + {len(late_class_probs)} per-class probabilities")
else:
    print("\nSkipping early/late consolidations (process_combined_only=True)")

# Combined all-years consolidation (equal-weight by default)
print("\n--- Combined All-Years Consolidation ---")
print("  (Combines all available years)")

combined_class, combined_temporal_conf, combined_class_probs = calculate_weighted_mode_and_confidence(
    aggregated_by_year, combined_weights, n_aggregated_classes, class_labels=aggregated_labels
)

if use_spatial:
    combined_spatial_conf = calculate_spatial_confidence(combined_class, neighborhood_radius)
    combined_confidence = combine_confidence_scores(combined_temporal_conf, combined_spatial_conf, spatial_weight)
    print(f"  ✓ Calculated with spatial confidence (weight={spatial_weight})")
else:
    combined_confidence = combined_temporal_conf
    print("  ✓ Calculated (temporal only)")

combined_prob_bands = [prob.multiply(100).toByte() for prob in combined_class_probs]
combined_consolidated = ee.Image.cat(
    [combined_class.toByte(), combined_confidence.multiply(100).toByte().rename('confidence')] + combined_prob_bands
).set('weighting', 'combined').set('years', years)

print(f"  Output bands: classification, confidence, + {len(combined_class_probs)} per-class probabilities")

print("\n✓ Consolidation calculations complete")

# %% [markdown]
## 6. Sample Statistics (Optional Validation)

# %%
print("\nSampling pixel statistics for validation...")

# Sample some points to verify the consolidation
sample_points = ee.FeatureCollection.randomPoints(study_area, 1000, 42)

def sample_consolidated(image, name):
    """Sample consolidated image and return statistics."""
    sampled = image.sampleRegions(
        collection=sample_points,
        scale=scale,
        geometries=False
    )
    
    # Get class distribution
    class_counts = _retry_with_backoff(
        lambda: sampled.aggregate_histogram('classification').getInfo()
    )
    
    # Get confidence statistics
    confidence_stats = _retry_with_backoff(
        lambda: sampled.aggregate_stats('confidence').getInfo()
    )
    
    return class_counts, confidence_stats

if not process_combined_only:
    print("\nEarly-weighted sample statistics:")
    early_class_counts, early_conf_stats = sample_consolidated(early_consolidated, 'early')
    for class_id, count in sorted(early_class_counts.items(), key=lambda x: int(x[0])):
        label = aggregated_labels.get(int(class_id), f'Class {class_id}')
        print(f"  Class {class_id} ({label}): {count} samples")
    print(f"  Confidence: mean={early_conf_stats['mean']:.1f}, std={early_conf_stats.get('total_sd', 0):.1f}")

    print("\nLate-weighted sample statistics:")
    late_class_counts, late_conf_stats = sample_consolidated(late_consolidated, 'late')
    for class_id, count in sorted(late_class_counts.items(), key=lambda x: int(x[0])):
        label = aggregated_labels.get(int(class_id), f'Class {class_id}')
        print(f"  Class {class_id} ({label}): {count} samples")
    print(f"  Confidence: mean={late_conf_stats['mean']:.1f}, std={late_conf_stats.get('total_sd', 0):.1f}")

print(f"\nCombined all-years sample statistics:")
combined_class_counts, combined_conf_stats = sample_consolidated(combined_consolidated, 'combined')
for class_id, count in sorted(combined_class_counts.items(), key=lambda x: int(x[0])):
    label = aggregated_labels.get(int(class_id), f'Class {class_id}')
    print(f"  Class {class_id} ({label}): {count} samples")
print(f"  Confidence: mean={combined_conf_stats['mean']:.1f}, std={combined_conf_stats.get('total_sd', 0):.1f}")

if not process_combined_only:
    # Calculate agreement between early and late weighted
    print("\nAgreement between early and late weighted:")
    both_sampled = ee.Image.cat([
        early_consolidated.select('classification').rename('early_class'),
        late_consolidated.select('classification').rename('late_class')
    ]).sampleRegions(collection=sample_points, scale=scale, geometries=False)

    agreement_count = _retry_with_backoff(
        lambda: both_sampled.filter(
            ee.Filter.equals('early_class', None, 'late_class', None)
        ).size().getInfo()
    )
    total_sampled = _retry_with_backoff(lambda: both_sampled.size().getInfo())
    agreement_pct = agreement_count / total_sampled * 100 if total_sampled > 0 else 0
    print(f"  Agreement: {agreement_count}/{total_sampled} ({agreement_pct:.1f}%)")

# %% [markdown]
## 7. Export Consolidated Assets

# %%
print("\n" + "="*70)
print("EXPORTING CONSOLIDATED ASSETS")
print("="*70)

from src.gee_utils import ensure_folder_exists, delete_asset

export_tasks = []
project_id = __config__.PROJECT_ID
folder_path = f"projects/{project_id}/assets/{asset_folder}"

# Ensure folder exists
ensure_folder_exists(folder_path)

early_asset_id = None
late_asset_id = None
combined_asset_id = None

# Export early-weighted
if CONSOL_CONFIG.get('export_early_weighted', True) and not process_combined_only:
    early_asset_name = CONSOL_CONFIG.get('early_asset_name', 'consolidated_early_weighted')
    early_asset_id = f"{folder_path}/{early_asset_name}"
    
    if CONSOL_CONFIG.get('overwrite_assets', False) and asset_exists(early_asset_id):
        delete_asset(early_asset_id)
    
    early_task = ee.batch.Export.image.toAsset(
        image=early_consolidated,
        description=early_asset_name,
        assetId=early_asset_id,
        region=study_area,
        scale=scale,
        maxPixels=1e10,
    )
    early_task.start()
    export_tasks.append(('early_weighted', early_task))
    print(f"\n  Started export: {early_asset_id}")

# Export late-weighted
if CONSOL_CONFIG.get('export_late_weighted', True) and not process_combined_only:
    late_asset_name = CONSOL_CONFIG.get('late_asset_name', 'consolidated_late_weighted')
    late_asset_id = f"{folder_path}/{late_asset_name}"
    
    if CONSOL_CONFIG.get('overwrite_assets', False) and asset_exists(late_asset_id):
        delete_asset(late_asset_id)
    
    late_task = ee.batch.Export.image.toAsset(
        image=late_consolidated,
        description=late_asset_name,
        assetId=late_asset_id,
        region=study_area,
        scale=scale,
        maxPixels=1e10,
    )
    late_task.start()
    export_tasks.append(('late_weighted', late_task))
    print(f"  Started export: {late_asset_id}")

# Export combined all-years consolidated asset (multi-band)
if CONSOL_CONFIG.get('export_combined_classified', True):
    combined_asset_name = CONSOL_CONFIG.get('combined_asset_name', 'consolidated_all_years_classified')
    combined_asset_id = f"{folder_path}/{combined_asset_name}"

    if CONSOL_CONFIG.get('overwrite_assets', False) and asset_exists(combined_asset_id):
        delete_asset(combined_asset_id)

    combined_task = ee.batch.Export.image.toAsset(
        image=combined_consolidated,
        description=combined_asset_name,
        assetId=combined_asset_id,
        region=study_area,
        scale=scale,
        maxPixels=1e10,
    )
    combined_task.start()
    export_tasks.append(('combined_all_years', combined_task))
    print(f"  Started export: {combined_asset_id}")

# Wait for exports
if export_tasks:
    print(f"\nWaiting for {len(export_tasks)} export(s) to complete...")
    results = wait_for_exports(export_tasks, check_interval=30)
    
    successful = sum(1 for ok in results.values() if ok)
    print(f"\n✓ {successful}/{len(export_tasks)} exports completed successfully")

# %% [markdown]
## 8. Save Configuration Summary

# %%
# Save configuration summary
summary = {
    'study_area': study_area_name,
    'years_consolidated': years,
    'n_aggregated_classes': n_aggregated_classes,
    'class_aggregation': class_aggregation,
    'aggregated_labels': aggregated_labels,
    'early_weights': early_weights,
    'late_weights': late_weights,
    'combined_weights': combined_weights,
    'use_spatial_confidence': use_spatial,
    'neighborhood_radius': neighborhood_radius if use_spatial else None,
    'spatial_weight': spatial_weight if use_spatial else None,
    'scale': scale,
    'process_combined_only': process_combined_only,
    'early_asset': early_asset_id if (CONSOL_CONFIG.get('export_early_weighted', True) and not process_combined_only) else None,
    'late_asset': late_asset_id if (CONSOL_CONFIG.get('export_late_weighted', True) and not process_combined_only) else None,
    'combined_asset': combined_asset_id if CONSOL_CONFIG.get('export_combined_classified', True) else None,
}

summary_path = OUTPUT_DIR / 'consolidation_summary.json'
import json
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"\n✓ Configuration summary saved to {summary_path}")

# %% [markdown]
## 9. Final Summary

# %%
print("\n" + "="*70)
print("CONSOLIDATION COMPLETE")
print("="*70)
print(f"\nProject: {project_id}")
print(f"Study Area: {study_area_name}")
print(f"Processed: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nYears Consolidated: {years}")
print(f"Aggregated Classes: {n_aggregated_classes}")
for agg_class, label in sorted(aggregated_labels.items()):
    orig_classes_for_agg = [k for k, v in class_aggregation.items() if v == agg_class]
    print(f"  {agg_class}: {label} (from: {orig_classes_for_agg})")

print(f"\nConfidence Scoring:")
print(f"  Temporal weighting: Enabled")
print(f"  Spatial consistency: {'Enabled' if use_spatial else 'Disabled'}")
if use_spatial:
    print(f"    Neighborhood: {2*neighborhood_radius+1}x{2*neighborhood_radius+1} pixels")
    print(f"    Spatial weight: {spatial_weight:.0%}")

print(f"\nOutputs:")
if CONSOL_CONFIG.get('export_early_weighted', True) and not process_combined_only:
    print(f"  Early-weighted: {early_asset_id}")
    print(f"    Weights: {', '.join(f'{y}:{early_weights[y]:.0f}' for y in years)}")
if CONSOL_CONFIG.get('export_late_weighted', True) and not process_combined_only:
    print(f"  Late-weighted: {late_asset_id}")
    print(f"    Weights: {', '.join(f'{y}:{late_weights[y]:.0f}' for y in years)}")
if CONSOL_CONFIG.get('export_combined_classified', True):
    print(f"  Combined all-years classified: {combined_asset_id}")
    print(f"    Weights: {', '.join(f'{y}:{combined_weights.get(y, 0):.0f}' for y in years)}")

if not process_combined_only:
    print(f"\nBands in weighted consolidation assets (early/late):")
    print(f"  - classification: Aggregated class (0-{n_aggregated_classes-1})")
    print(f"  - confidence: Winning class confidence score (0-100)")
    for class_id, label in sorted(aggregated_labels.items()):
        band_name = f'prob_{label.lower().replace(" ", "_").replace(".", "")}'
        print(f"  - {band_name}: {label} probability (0-100)")

if CONSOL_CONFIG.get('export_combined_classified', True):
    print(f"\nBands in combined all-years classified asset:")
    print(f"  - classification: Aggregated class (0-{n_aggregated_classes-1})")
    print(f"  - confidence: Winning class confidence score (0-100)")
    for class_id, label in sorted(aggregated_labels.items()):
        band_name = f'prob_{label.lower().replace(" ", "_").replace(".", "")}'
        print(f"  - {band_name}: {label} probability (0-100)")

print(f"\nSummary saved to: {summary_path}")
print("\n" + "="*70)
