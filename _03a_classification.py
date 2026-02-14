# %% [markdown]
# # Zambezi River Basin - Cluster-Based Classification using Satellite Embeddings
#
# This script uses Google Earth Engine Satellite Embeddings V1 to classify agricultural
# fields based on NSE clustering results from temporal analysis.
#
# **Approach:**
# Based on AlphaEarth Foundations (Brown et al. 2025), satellite embeddings provide
# a 64-dimensional representation of each pixel that captures semantic information
# about land cover and surface characteristics. 
#
# **Key Design: Year-Specific Classifiers**
# We train a SEPARATE classifier for each year using only that year's samples
# and embeddings. Each classifier is applied ONLY to its corresponding year.
#
# We also compute:
# - Per-year accuracy + confusion matrices
# - A combined confusion matrix/accuracy across all years (aggregated tests)
#
# **Data:**
# - Google Satellite Embedding V1 (GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL)
# - DTW clustering results from _02b_clustering.py
# - Sampled points with coordinates
#
# **Author:** GEE Application
# **Date:** 2026-01-18
# **Project ID:** ee-warnold

# %% [markdown]
## 1. Setup and Initialization

# %%
# Import libraries
import ee
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))
import __config__

# Import custom utilities
from src.gee_utils import initialize_earth_engine, _retry_with_backoff, asset_exists
from src.classification_utils import (
    load_embeddings_for_year,
    load_unified_training_data,
    create_unified_training_features,
    train_unified_classifier,
    apply_classification,
    export_classified_to_asset,
    wait_for_exports,
    normalize_confusion_matrix,
    print_confusion_matrix,
    per_class_accuracy,
    create_class_mapping,
    remap_training_classes,
    export_training_samples_to_asset,
    load_training_samples_from_asset,
    split_train_test_balanced,
)

# Configuration
EMB_CONFIG = __config__.EMBEDDINGS_CLASSIFICATION
# Set up plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

# Output directory
OUTPUT_DIR = Path('output') / __config__.CURRENT_STUDY_AREA / EMB_CONFIG['output_subfolder']
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
ACCURACY_DIR = OUTPUT_DIR / 'accuracy'
ACCURACY_DIR.mkdir(parents=True, exist_ok=True)

# %%
# Initialize Earth Engine
initialize_earth_engine()

# Load configuration
study_area_config = __config__.get_study_area()

print("\n" + "="*70)
print("SATELLITE EMBEDDINGS CLASSIFICATION (UNIFIED)")
print("="*70)
print(f"\nConfiguration:")
print(f"  Study Area: {study_area_config['name']}")
print(f"  Embeddings Dataset: {EMB_CONFIG['embeddings_dataset']}")
print(f"  Classification Years: {EMB_CONFIG['classification_years']}")
print(f"  Classifier: {EMB_CONFIG['classifier_type']}")
print(f"  Use Landcover Samples: {EMB_CONFIG.get('use_landcover_samples', False)}")
print(f"  Training samples: auto-detect (load from asset if exists, else create & export)")
print(f"  Training samples asset folder: {EMB_CONFIG.get('training_samples_asset_folder', 'classification_training_samples')}")
print(f"  Training samples asset prefix: {EMB_CONFIG.get('training_samples_asset_prefix', 'training_samples')}")
print(f"\n  Approach: Unified classifier with crop clusters + landcover classes")

# %% [markdown]
## 2. Load Training Data (Crop Clusters + Landcover Samples)

# %%
# Load unified training data from both sources
df_training, class_info = load_unified_training_data(EMB_CONFIG)

# Optional: reorder crop cluster IDs using original IDs in desired new order
cluster_reorder = EMB_CONFIG.get('cluster_reorder')
if cluster_reorder:
    reorder_list = list(cluster_reorder)
    if len(set(reorder_list)) != len(reorder_list):
        raise ValueError("EMBEDDINGS_CLASSIFICATION.cluster_reorder contains duplicate IDs")

    crop_mask = df_training['source'] == 'crop'
    crop_ids = sorted(df_training.loc[crop_mask, 'cluster'].dropna().unique())
    missing = sorted(set(crop_ids) - set(reorder_list))
    extra = sorted(set(reorder_list) - set(crop_ids))
    if missing:
        raise ValueError(
            f"cluster_reorder is missing crop cluster IDs present in data: {missing}. "
            f"Found crop clusters: {crop_ids}"
        )
    if extra:
        print(f"  ⚠ cluster_reorder contains IDs not present in crop clusters: {extra}")

    reorder_map = {old_id: new_id for new_id, old_id in enumerate(reorder_list)}
    df_training.loc[crop_mask, 'cluster'] = (
        df_training.loc[crop_mask, 'cluster'].map(reorder_map).astype(int)
    )

    class_info['class_counts'] = df_training.groupby('cluster').size().to_dict()
    class_info['n_classes'] = df_training['cluster'].nunique()
    class_info['crop_clusters'] = sorted(df_training.loc[crop_mask, 'cluster'].unique())

    print(f"\n  ✓ Applied crop cluster reorder to training data (new_id -> old_id): { {v: k for k, v in reorder_map.items()} }")

# Get class configuration
cluster_labels = EMB_CONFIG.get('cluster_labels', {})
class_types = EMB_CONFIG.get('class_types', {})
n_classes = class_info['n_classes']

print(f"\n  Total classes: {n_classes}")
print(f"  Crop clusters: {class_info['crop_clusters']}")
print(f"  Landcover classes: {class_info['landcover_classes']}")
print(f"  Years available: {class_info['years']}")

# Filter to classification years
classification_years = EMB_CONFIG['classification_years']
df_training = df_training[df_training['year'].isin(classification_years)].copy()
print(f"\n  Filtered to classification years {classification_years}: {len(df_training)} samples")

# Show distribution by year
print(f"\n  Samples by year:")
for year in sorted(df_training['year'].unique()):
    year_count = len(df_training[df_training['year'] == year])
    print(f"    {year}: {year_count}")

# %% [markdown]
## 3. Define Study Area and Load Embeddings

# %%
# Get study area from config
study_area = study_area_config['geometry']
study_area_name = __config__.CURRENT_STUDY_AREA

print(f"\nStudy area: {study_area_config['name']}")
print(f"Center: {study_area_config['center']}")

# %%
# Load embeddings for all years
all_years = EMB_CONFIG['classification_years']
print(f"\nLoading embeddings for all years: {all_years}")

embeddings_by_year = {}
for year in all_years:
    embeddings_by_year[year] = load_embeddings_for_year(
        study_area, year, EMB_CONFIG['embeddings_dataset']
    )
    # Count tiles available for this year
    tile_count = _retry_with_backoff(lambda y=year: (ee.ImageCollection(EMB_CONFIG['embeddings_dataset'])
                 .filterBounds(study_area)
                 .filter(ee.Filter.calendarRange(y, y, 'year'))
                 .size().getInfo()))
    band_count = _retry_with_backoff(lambda y=year: embeddings_by_year[y].bandNames().size().getInfo())
    print(f"  Year {year}: {band_count} bands from {tile_count} tiles")

# Reference embeddings (for band names)
reference_embeddings = embeddings_by_year[all_years[0]]
band_names = _retry_with_backoff(lambda: reference_embeddings.bandNames().getInfo())
print(f"\n✓ Embeddings loaded: {len(band_names)} bands per year")

# %% [markdown]
## 4. Train Year-Specific Classifiers (Per Year)

# %%
print(f"\nTraining year-specific classifiers for {len(all_years)} years...")
balance_training = EMB_CONFIG.get('balance_training', False)
balance_method = EMB_CONFIG.get('balance_method', 'undersample')
TEST_FRACTION = 0.2
seed = EMB_CONFIG.get('gtb_seed', EMB_CONFIG.get('rf_seed', 42))

training_samples_asset_folder = EMB_CONFIG.get('training_samples_asset_folder', 'classification_training_samples')
training_samples_asset_prefix = EMB_CONFIG.get('training_samples_asset_prefix', 'training_samples')
training_samples_overwrite_assets = EMB_CONFIG.get('training_samples_overwrite_assets', False)
skip_training_sample_counts = EMB_CONFIG.get('skip_training_sample_counts', False)
training_export_tasks = []

cluster_palette = EMB_CONFIG['cluster_palette']

# Get unique class IDs from training data and create mapping
# Filter out NaN values that may occur from unmapped classes
unique_classes = sorted([c for c in df_training['cluster'].unique() if pd.notna(c)])
class_mapping = create_class_mapping(unique_classes)
n_classes = class_mapping['n_classes']
print(f"\n  Classes in training data: {unique_classes}")
print(f"  Total unique classes: {n_classes}")

# Remap classes to sequential indices (0 to n-1) for GEE training
df_training = remap_training_classes(df_training, class_mapping)
print(f"  Remapped classes: {sorted(df_training['cluster'].unique())}")

# Build labels mapping for sequential indices
# cluster_labels now maps sequential index -> display label
idx_cluster_labels = {}
for idx, class_id in class_mapping['to_class'].items():
    idx_cluster_labels[idx] = cluster_labels.get(class_id, f"Class {class_id}")

# For compatibility
n_clusters = n_classes

# Build palette for sequential indices (in order of unique_classes)
# class_palette is a LIST for GEE visualization (ordered by sequential index)
# type_palette is a DICT for matplotlib lookups (idx -> color)
class_palette = []
type_palette = {}  # Maps sequential index -> color
for idx, class_id in enumerate(unique_classes):
    color = cluster_palette.get(class_id, '#808080')  # Default gray for unmapped
    class_palette.append(color)
    type_palette[idx] = color


classifiers_by_year = {}
prob_classifiers_by_year = {}
metrics_by_year = {}
training_stats_by_year = {}
class_counts_by_year = {}
combined_confusion_matrix = np.zeros((n_clusters, n_clusters), dtype=int)
combined_test_samples = 0
combined_train_samples = 0
used_classifier_type = None

merged_accuracy_rows = []
merged_summary_rows = []

# Use ee.List for band names
ee_band_names = ee.List(band_names)

for year in all_years:
    print(f"\n" + "-" * 60)
    print(f"YEAR {year} - Training classifier")
    print("-" * 60)

    df_year = df_training[df_training["year"] == year].copy()
    if df_year.empty:
        print(f"  ⚠ No training samples for year {year}. Skipping classifier.")
        continue

    # Auto-detect: use asset if it exists unless overwrite is enabled
    training_asset_id = f"projects/ee-warnold/assets/{training_samples_asset_folder}/{training_samples_asset_prefix}_{year}"

    if asset_exists(training_asset_id) and not training_samples_overwrite_assets:
        print(f"  ✓ Loading training samples from existing asset")
        year_training_data = load_training_samples_from_asset(
            year,
            asset_folder=training_samples_asset_folder,
            asset_prefix=training_samples_asset_prefix,
        )
        class_counts = df_year['cluster'].value_counts().to_dict()
    else:
        if asset_exists(training_asset_id) and training_samples_overwrite_assets:
            print(f"  ↻ Recreating training samples (overwrite enabled)")
        print(f"  → Asset not found, creating training samples directly...")
        year_training_data, class_counts = create_unified_training_features(
            df_year, embeddings_by_year, [year],
            scale=EMB_CONFIG['scale'],
            balance=balance_training,
            balance_method=balance_method,
            balance_floor=EMB_CONFIG.get('balance_floor', 200),
            balance_cap=EMB_CONFIG.get('balance_cap', 800),
            batch_size=EMB_CONFIG.get('training_batch_size', 300),
            max_workers=EMB_CONFIG.get('training_max_workers', 4),
            count_batches=EMB_CONFIG.get('training_count_batches', False),
        )

        # Always export when creating directly
        print(f"  → Exporting training samples to asset...")
        export_task = export_training_samples_to_asset(
            year_training_data,
            year,
            asset_folder=training_samples_asset_folder,
            asset_prefix=training_samples_asset_prefix,
            overwrite=training_samples_overwrite_assets,
        )
        training_export_tasks.append(export_task)
        print(f"  ⏳ Waiting for training sample export to finish...")
        wait_for_exports([export_task], check_interval=30)
        print(f"  ✓ Training samples exported and ready")

    if skip_training_sample_counts:
        year_total_samples = None
    else:
        year_total_samples = _retry_with_backoff(lambda: year_training_data.size().getInfo())
        if year_total_samples == 0:
            print(f"  ⚠ No valid embeddings for year {year}. Skipping classifier.")
            continue

    class_counts_by_year[str(year)] = class_counts
    training_stats_by_year[str(year)] = {
        'total_samples': year_total_samples,
        'samples_in_year': len(df_year)
    }

    if year_total_samples is not None:
        print(f"  Total training samples: {year_total_samples}")
    else:
        print("  Total training samples: (skipped count)")
    print(f"  Splitting data into train/test sets ({int((1-TEST_FRACTION)*100)}/{int(TEST_FRACTION*100)} split)...")
    balanced_split = EMB_CONFIG.get('balanced_split', True)

    if balanced_split:
        print("  Using class-balanced split")
        class_ids = sorted([int(c) for c in class_counts.keys()])
        train_data, test_data = split_train_test_balanced(
            year_training_data,
            class_property='cluster',
            test_fraction=TEST_FRACTION,
            seed=seed,
            class_ids=class_ids,
        )
    else:
        year_with_random = year_training_data.randomColumn('random', seed)
        train_data = year_with_random.filter(ee.Filter.gte('random', TEST_FRACTION))
        test_data = year_with_random.filter(ee.Filter.lt('random', TEST_FRACTION))

    if skip_training_sample_counts:
        train_count = None
        test_count = None
        print("    Training samples: (skipped count)")
        print("    Test samples: (skipped count)")
    else:
        train_count = _retry_with_backoff(lambda: train_data.size().getInfo())
        test_count = _retry_with_backoff(lambda: test_data.size().getInfo())
        combined_train_samples += train_count
        combined_test_samples += test_count

        print(f"    Training samples: {train_count}")
        print(f"    Test samples: {test_count}")

    classifier_type = EMB_CONFIG.get('classifier_type', 'random_forest')
    print(f"\n  Training {classifier_type.upper().replace('_', ' ')} classifier for {year}...")

    if classifier_type == 'gradient_boost':
        print(f"    Trees: {EMB_CONFIG.get('gtb_num_trees', 100)}")
        print(f"    Shrinkage (learning rate): {EMB_CONFIG.get('gtb_shrinkage', 0.1)}")
        print(f"    Sampling rate: {EMB_CONFIG.get('gtb_sampling_rate', 0.8)}")
        print(f"    Max nodes: {EMB_CONFIG.get('gtb_max_nodes', 10)}")
        print(f"    Seed: {EMB_CONFIG.get('gtb_seed', 42)}")
    else:
        print(f"    Trees: {EMB_CONFIG['rf_num_trees']}")
        print(f"    Seed: {EMB_CONFIG['rf_seed']}")

    classifier, prob_classifier, train_cm, test_cm, this_classifier_type = train_unified_classifier(
        train_data, test_data, ee_band_names, EMB_CONFIG
    )

    if used_classifier_type is None:
        used_classifier_type = this_classifier_type

    classifiers_by_year[year] = classifier
    prob_classifiers_by_year[year] = prob_classifier

    train_accuracy_value = _retry_with_backoff(lambda: train_cm.accuracy().getInfo())
    test_accuracy_value = _retry_with_backoff(lambda: test_cm.accuracy().getInfo())
    print(f"\n  Training accuracy: {train_accuracy_value:.3f}")
    print(f"  Test accuracy: {test_accuracy_value:.3f}")

    print(f"\n  TEST Confusion Matrix (rows=actual, cols=predicted):")
    confusion_array = _retry_with_backoff(lambda: test_cm.array().getInfo())
    matrix, _ = normalize_confusion_matrix(confusion_array, n_clusters)
    print_confusion_matrix(matrix, idx_cluster_labels, n_clusters)

    accuracy_metrics = per_class_accuracy(matrix, idx_cluster_labels)
    print(f"\n  Per-class accuracy (TEST):")
    for row in accuracy_metrics:
        # Map idx back to original class ID for display
        orig_class_id = class_mapping['to_class'].get(row['cluster'], row['cluster'])
        if row['total'] > 0:
            print(f"    Class {orig_class_id} ({row['label']}): {row['accuracy']:.1%} ({row['correct']}/{row['total']})")
        else:
            print(f"    Class {orig_class_id} ({row['label']}): N/A (no test samples)")

    for row in accuracy_metrics:
        orig_class_id = class_mapping['to_class'].get(row['cluster'], row['cluster'])
        merged_accuracy_rows.append({
            'year': year,
            'class': orig_class_id,  # Store original class ID
            'class_idx': row['cluster'],  # Store sequential index
            'label': row['label'],
            'correct': row['correct'],
            'total': row['total'],
            'accuracy': row['accuracy']
        })

    metrics_by_year[str(year)] = {
        'train_accuracy': train_accuracy_value,
        'test_accuracy': test_accuracy_value,
        'train_samples': train_count,
        'test_samples': test_count
    }

    merged_summary_rows.append({
        'year': year,
        'split': 'train',
        'accuracy': train_accuracy_value,
        'samples': train_count
    })
    merged_summary_rows.append({
        'year': year,
        'split': 'test',
        'accuracy': test_accuracy_value,
        'samples': test_count
    })

    combined_confusion_matrix += matrix

    print(f"\nSaving confusion matrix and accuracy metrics for {year}...")

    cm_data = []
    for display_idx, row_values in enumerate(matrix):
        label = cluster_labels.get(display_idx, f'C{display_idx}')
        row_dict = {'class': display_idx, 'label': label}
        for col_idx, val in enumerate(row_values):
            col_label = cluster_labels.get(col_idx, f'C{col_idx}')
            row_dict[f'{col_idx}:{col_label}'] = int(val)
        cm_data.append(row_dict)

    df_cm = pd.DataFrame(cm_data)
    cm_path = ACCURACY_DIR / f'test_confusion_matrix_{year}.csv'
    df_cm.to_csv(cm_path, index=False)
    print(f"  ✓ Test confusion matrix saved to {cm_path}")

    df_acc = pd.DataFrame(accuracy_metrics).rename(columns={'cluster': 'class'})
    acc_path = ACCURACY_DIR / f'test_accuracy_per_class_{year}.csv'
    df_acc.to_csv(acc_path, index=False)
    print(f"  ✓ Per-class accuracy saved to {acc_path}")

    accuracy_summary = pd.DataFrame([{
        'split': 'train',
        'accuracy': train_accuracy_value,
        'samples': train_count
    }, {
        'split': 'test',
        'accuracy': test_accuracy_value,
        'samples': test_count
    }])
    summary_path = ACCURACY_DIR / f'accuracy_summary_{year}.csv'
    accuracy_summary.to_csv(summary_path, index=False)
    print(f"  ✓ Accuracy summary saved to {summary_path}")


# Combined (all years) confusion matrix/accuracy
print(f"\n" + "=" * 70)
print("COMBINED (ALL YEARS) CONFUSION MATRIX")
print("=" * 70)
print(f"\n  TEST Confusion Matrix (rows=actual, cols=predicted):")
print_confusion_matrix(combined_confusion_matrix, idx_cluster_labels, n_clusters)

combined_accuracy_metrics = per_class_accuracy(combined_confusion_matrix, idx_cluster_labels)
print(f"\n  Per-class accuracy (TEST, combined):")
for row in combined_accuracy_metrics:
    orig_class_id = class_mapping['to_class'].get(row['cluster'], row['cluster'])
    if row['total'] > 0:
        print(f"    Class {orig_class_id} ({row['label']}): {row['accuracy']:.1%} ({row['correct']}/{row['total']})")
    else:
        print(f"    Class {orig_class_id} ({row['label']}): N/A (no test samples)")

combined_total = int(np.sum(combined_confusion_matrix))
combined_correct = int(np.trace(combined_confusion_matrix))
combined_test_accuracy = combined_correct / combined_total if combined_total > 0 else 0

print(f"\n  Combined test accuracy: {combined_test_accuracy:.3f}")

print(f"\nSaving combined confusion matrix and accuracy metrics...")

combined_cm_data = []
for display_idx, row_values in enumerate(combined_confusion_matrix):
    label = cluster_labels.get(display_idx, f'C{display_idx}')
    row_dict = {'class': display_idx, 'label': label}
    for col_idx, val in enumerate(row_values):
        col_label = cluster_labels.get(col_idx, f'C{col_idx}')
        row_dict[f'{col_idx}:{col_label}'] = int(val)
    combined_cm_data.append(row_dict)

df_combined_cm = pd.DataFrame(combined_cm_data)
combined_cm_path = ACCURACY_DIR / 'test_confusion_matrix_combined.csv'
df_combined_cm.to_csv(combined_cm_path, index=False)
print(f"  ✓ Combined test confusion matrix saved to {combined_cm_path}")

df_combined_acc = pd.DataFrame(combined_accuracy_metrics).rename(columns={'cluster': 'class'})
combined_acc_path = ACCURACY_DIR / 'test_accuracy_per_class_combined.csv'
df_combined_acc.to_csv(combined_acc_path, index=False)
print(f"  ✓ Combined per-class accuracy saved to {combined_acc_path}")

combined_accuracy_summary = pd.DataFrame([{
    'split': 'test',
    'accuracy': combined_test_accuracy,
    'samples': combined_total
}])
combined_summary_path = ACCURACY_DIR / 'accuracy_summary_combined.csv'
combined_accuracy_summary.to_csv(combined_summary_path, index=False)
print(f"  ✓ Combined accuracy summary saved to {combined_summary_path}")

# Save merged per-year accuracy CSVs
if merged_accuracy_rows:
    df_merged_acc = pd.DataFrame(merged_accuracy_rows)
    merged_acc_path = ACCURACY_DIR / 'test_accuracy_per_class_merged.csv'
    df_merged_acc.to_csv(merged_acc_path, index=False)
    print(f"  ✓ Merged per-class accuracy saved to {merged_acc_path}")

if merged_summary_rows:
    merged_summary_rows.append({
        'year': 'combined',
        'split': 'test',
        'accuracy': combined_test_accuracy,
        'samples': combined_total
    })
    df_merged_summary = pd.DataFrame(merged_summary_rows)
    merged_summary_path = ACCURACY_DIR / 'accuracy_summary_merged.csv'
    df_merged_summary.to_csv(merged_summary_path, index=False)
    print(f"  ✓ Merged accuracy summary saved to {merged_summary_path}")

# %%
# Calculate AGGREGATED class type accuracy
# Maps individual classes to broader types (Irrigated, Rainfed, Urban, etc.)
print(f"\n" + "=" * 70)
print("AGGREGATED CLASS TYPE ACCURACY")
print("=" * 70)

# Build mapping from sequential index to class type
idx_to_type = {}
type_names = list(class_types.keys())
for type_name, class_ids in class_types.items():
    for class_id in class_ids:
        # Map original class ID to sequential index
        seq_idx = class_mapping['to_idx'].get(class_id)
        if seq_idx is not None:
            idx_to_type[seq_idx] = type_name

# Build aggregated confusion matrix (type x type)
n_types = len(type_names)
type_confusion_matrix = np.zeros((n_types, n_types), dtype=int)

for actual_idx in range(n_clusters):
    for pred_idx in range(n_clusters):
        count = combined_confusion_matrix[actual_idx, pred_idx]
        if count > 0:
            actual_type = idx_to_type.get(actual_idx)
            pred_type = idx_to_type.get(pred_idx)
            if actual_type and pred_type:
                actual_type_idx = type_names.index(actual_type)
                pred_type_idx = type_names.index(pred_type)
                type_confusion_matrix[actual_type_idx, pred_type_idx] += count

# Print aggregated confusion matrix
print(f"\n  Aggregated Confusion Matrix (rows=actual, cols=predicted):")
header = "          " + "  ".join([f"{t[:8]:>8}" for t in type_names])
print(header)
for i, type_name in enumerate(type_names):
    row_str = f"{type_name[:8]:>8}  " + "  ".join([f"{type_confusion_matrix[i, j]:>8}" for j in range(n_types)])
    print(row_str)

# Calculate per-type accuracy
print(f"\n  Per-type accuracy (TEST, combined):")
aggregated_accuracy_rows = []
for i, type_name in enumerate(type_names):
    total = int(np.sum(type_confusion_matrix[i, :]))
    correct = int(type_confusion_matrix[i, i])
    accuracy = correct / total if total > 0 else 0
    
    # Also calculate precision (of predictions, how many were correct)
    pred_total = int(np.sum(type_confusion_matrix[:, i]))
    precision = correct / pred_total if pred_total > 0 else 0
    
    if total > 0:
        print(f"    {type_name}: Recall={accuracy:.1%} ({correct}/{total}), Precision={precision:.1%}")
    else:
        print(f"    {type_name}: N/A (no test samples)")
    
    aggregated_accuracy_rows.append({
        'type': type_name,
        'recall': accuracy,
        'precision': precision,
        'correct': correct,
        'total_actual': total,
        'total_predicted': pred_total
    })

# Overall aggregated accuracy
agg_total = int(np.sum(type_confusion_matrix))
agg_correct = int(np.trace(type_confusion_matrix))
agg_accuracy = agg_correct / agg_total if agg_total > 0 else 0
print(f"\n  Overall aggregated accuracy: {agg_accuracy:.3f}")

# Save aggregated results
df_type_acc = pd.DataFrame(aggregated_accuracy_rows)
type_acc_path = ACCURACY_DIR / 'test_accuracy_by_type.csv'
df_type_acc.to_csv(type_acc_path, index=False)
print(f"\n  ✓ Aggregated type accuracy saved to {type_acc_path}")

# Save aggregated confusion matrix
type_cm_data = []
for i, type_name in enumerate(type_names):
    row_dict = {'type': type_name}
    for j, pred_type in enumerate(type_names):
        row_dict[pred_type] = int(type_confusion_matrix[i, j])
    type_cm_data.append(row_dict)
df_type_cm = pd.DataFrame(type_cm_data)
type_cm_path = ACCURACY_DIR / 'test_confusion_matrix_by_type.csv'
df_type_cm.to_csv(type_cm_path, index=False)
print(f"  ✓ Aggregated confusion matrix saved to {type_cm_path}")

# %% [markdown]
## 6. Apply Classification to All Years

# %%
# Apply year-specific classifiers to each year's embeddings
print(f"\nApplying year-specific classifiers to all years...")

classified_by_year = {}
max_prob_by_year = {}
prob_bands_by_year = {}
for year in all_years:
    if year not in classifiers_by_year:
        print(f"  Skipping year {year} (no classifier trained)")
        continue

    print(f"  Classifying year {year}...")

    classified, max_prob, prob_bands = apply_classification(
        embeddings_by_year[year],
        classifiers_by_year[year],
        prob_classifiers_by_year[year],
        study_area,
        crop_mask=None,
        confidence_threshold=None,
        null_value=None,
        n_clusters=n_clusters
    )
    classified_by_year[year] = classified
    max_prob_by_year[year] = max_prob
    prob_bands_by_year[year] = prob_bands

print("✓ All available years classified with year-specific classifiers")

# %% [markdown]
## 7. Export Classification to GEE Assets

# %%
export_mode = EMB_CONFIG.get('classification_export_mode', 'none')
asset_folder = EMB_CONFIG.get('asset_folder', 'irrigation_classification')
classified_overwrite_assets = EMB_CONFIG.get('classified_overwrite_assets', False)

if export_mode in ('batch',):
    print(f"\nExporting classified images to GEE assets...")
    print(f"  Asset folder: projects/{__config__.PROJECT_ID}/assets/{asset_folder}/")
    tasks = export_classified_to_asset(
        classified_by_year,
        study_area,
        scale=10,
        asset_folder=asset_folder,
        overwrite=classified_overwrite_assets,
    )

    print(f"\n{len(tasks)} export tasks started. Waiting for completion...")
    results = wait_for_exports(tasks, check_interval=30)

    successful_years = [y for y, ok in results.items() if ok]
    print(f"\n✓ {len(successful_years)} classification exports completed successfully")
else:
    print("\nClassification export: Disabled")

# %% [markdown]
## 8. Final Summary

# %%
# Final summary
print("\n" + "="*70)
print("CLASSIFICATION COMPLETE (UNIFIED)")
print("="*70)
print(f"\nProject: {__config__.PROJECT_ID}")
print(f"Study Area: {study_area_config['name']}")
print(f"Processed: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nMethod: Satellite Embeddings V1 + {used_classifier_type.replace('_', ' ').title()}")
print(f"Approach: Unified classifier with crop clusters + landcover classes")
print(f"Embeddings: {EMB_CONFIG['embeddings_dataset']}")
print(f"Years: {all_years}")
print(f"Classes: {n_classes}")
print(f"  Original class IDs: {class_mapping['unique_classes']}")
print(f"  Class labels: {idx_cluster_labels}")
print(f"\nTraining Settings:")
print(f"  Class balancing: {EMB_CONFIG.get('balance_training', False)}")
if EMB_CONFIG.get('balance_training', False):
    print(f"  Balance method: {EMB_CONFIG.get('balance_method', 'undersample')}")
print(f"  Landcover samples: {'Enabled' if EMB_CONFIG.get('use_landcover_samples', False) else 'Disabled'}")
print(f"\nData Split: {int((1-TEST_FRACTION)*100)}/{int(TEST_FRACTION*100)} train/test")
if skip_training_sample_counts:
    print("  Training samples (all years): (skipped count)")
    print("  Test samples (all years): (skipped count)")
else:
    print(f"  Training samples (all years): {combined_train_samples}")
    print(f"  Test samples (all years): {combined_test_samples}")
print(f"\nAccuracy (combined test sets):")
print(f"  Per-class:   {combined_test_accuracy:.3f}")
print(f"  Aggregated:  {agg_accuracy:.3f} (by type: {', '.join(type_names)})")
print(f"\nOutputs saved to: {OUTPUT_DIR}")
print("  - accuracy/test_confusion_matrix_<year>.csv")
print("  - accuracy/test_accuracy_per_class_<year>.csv")
print("  - accuracy/accuracy_summary_<year>.csv")
print("  - accuracy/test_confusion_matrix_combined.csv")
print("  - accuracy/test_accuracy_per_class_combined.csv")
print("  - accuracy/accuracy_summary_combined.csv")
print("  - accuracy/test_accuracy_per_class_merged.csv")
print("  - accuracy/accuracy_summary_merged.csv")
print("  - accuracy/test_accuracy_by_type.csv")
print("  - accuracy/test_confusion_matrix_by_type.csv")
print("\nNext step: Run _03c_area_statistics.py to calculate area statistics")
print("\n" + "="*70)
