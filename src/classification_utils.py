"""
Classification Utilities

Reusable utilities for embedding-based classification, masking, smoothing,
export workflows, and summary helpers.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time
from typing import Dict, Iterable, Tuple

import ee
import numpy as np
import pandas as pd

import __config__
from src.gee_utils import _retry_with_backoff, asset_exists, delete_asset, ensure_folder_exists


S2_COLLECTION = __config__.DATASETS.get("sentinel2", "COPERNICUS/S2_SR_HARMONIZED")


def normalize_confusion_matrix(confusion_array, n_clusters, class_mapping=None):
    """Normalize GEE confusion matrix to an n_clusters x n_clusters numpy array.

    Handles optional empty row/column 0 when classes are 1-indexed.
    
    Parameters
    ----------
    confusion_array : list
        Raw confusion matrix from GEE
    n_clusters : int
        Expected number of classes
    class_mapping : dict, optional
        Mapping from class ID to display index (for non-consecutive IDs)
    
    Returns
    -------
    matrix : np.ndarray
        Normalized confusion matrix
    has_empty_row0 : bool
        Whether row 0 was empty (1-indexed classes)
    """
    n_matrix_rows = len(confusion_array)
    has_empty_row0 = n_matrix_rows > n_clusters and sum(confusion_array[0]) == 0
    start_idx = 1 if has_empty_row0 else 0

    matrix = []
    for matrix_idx in range(start_idx, min(start_idx + n_clusters, n_matrix_rows)):
        row = confusion_array[matrix_idx]
        row_values = row[start_idx:start_idx + n_clusters] if has_empty_row0 else row[:n_clusters]
        matrix.append([int(v) for v in row_values])

    return np.array(matrix, dtype=int), has_empty_row0


def create_class_mapping(unique_classes):
    """Create bidirectional mapping for non-consecutive class IDs.
    
    Parameters
    ----------
    unique_classes : list
        List of unique class IDs (possibly non-consecutive)
    
    Returns
    -------
    dict with keys:
        'to_idx': {class_id -> sequential_index}
        'to_class': {sequential_index -> class_id}
        'n_classes': number of unique classes
    """
    unique_classes = sorted(unique_classes)
    return {
        'to_idx': {class_id: idx for idx, class_id in enumerate(unique_classes)},
        'to_class': {idx: class_id for idx, class_id in enumerate(unique_classes)},
        'n_classes': len(unique_classes),
        'unique_classes': unique_classes
    }


def remap_training_classes(df: pd.DataFrame, class_mapping: dict) -> pd.DataFrame:
    """Remap cluster column to sequential indices for GEE training.
    
    Parameters
    ----------
    df : pd.DataFrame
        Training data with 'cluster' column
    class_mapping : dict
        From create_class_mapping()
    
    Returns
    -------
    pd.DataFrame
        Copy with remapped cluster values (sequential 0 to n-1)
    """
    df_remapped = df.copy()
    df_remapped['cluster_orig'] = df_remapped['cluster']
    df_remapped['cluster'] = df_remapped['cluster'].map(class_mapping['to_idx'])
    return df_remapped


def print_confusion_matrix(matrix, cluster_labels, n_clusters):
    """Print a formatted confusion matrix."""
    header = "         " + "  ".join([f"{i:>6}" for i in range(n_clusters)])
    print(header)
    print("         " + "-" * (7 * n_clusters))
    for display_idx, row_values in enumerate(matrix):
        label = cluster_labels.get(display_idx, f"C{display_idx}")[:6]
        row_str = f"{display_idx} {label:>6} |" + "  ".join([f"{int(v):>6}" for v in row_values])
        print(row_str)


def per_class_accuracy(matrix, cluster_labels):
    """Compute per-class accuracy from a confusion matrix."""
    metrics = []
    for display_idx, row_values in enumerate(matrix):
        row_total = int(np.sum(row_values))
        correct = int(row_values[display_idx]) if display_idx < len(row_values) else 0
        acc = correct / row_total if row_total > 0 else 0
        label = cluster_labels.get(display_idx, f"C{display_idx}")
        metrics.append(
            {
                "cluster": display_idx,
                "label": label,
                "correct": correct,
                "total": row_total,
                "accuracy": acc,
            }
        )
    return metrics


def split_train_test_balanced(
    training_data: ee.FeatureCollection,
    class_property: str = "cluster",
    test_fraction: float = 0.2,
    seed: int = 42,
    class_ids: Iterable[int] | None = None,
    min_test: int = 1,
) -> Tuple[ee.FeatureCollection, ee.FeatureCollection]:
    """Split FeatureCollection into balanced train/test sets per class.

    Ensures each class contributes roughly the same fraction of samples to the
    test set by stratifying and slicing within each class.

    Parameters
    ----------
    training_data : ee.FeatureCollection
        Input training samples with class labels.
    class_property : str
        Property name for class label (default: 'cluster').
    test_fraction : float
        Fraction of samples per class to allocate to the test set.
    seed : int
        Random seed for reproducibility.
    class_ids : iterable of int, optional
        Explicit class IDs to split. If None, derived from training_data.
    min_test : int
        Minimum test samples per class when possible.

    Returns
    -------
    train_data, test_data : ee.FeatureCollection
        Balanced train and test splits.
    """
    if class_ids is None:
        class_ids = _retry_with_backoff(
            lambda: training_data.aggregate_array(class_property).distinct().getInfo()
        )
    class_ids = [int(c) for c in class_ids if c is not None]

    train_collections = []
    test_collections = []

    for class_id in class_ids:
        class_fc = training_data.filter(ee.Filter.eq(class_property, class_id))
        class_count = _retry_with_backoff(lambda: class_fc.size().getInfo())
        if class_count == 0:
            continue

        n_test = int(round(class_count * test_fraction))
        if class_count > 1:
            n_test = max(min_test, n_test)
        else:
            n_test = class_count
        n_test = min(n_test, class_count)

        class_fc = class_fc.randomColumn("random", seed + int(class_id))
        class_list = class_fc.sort("random").toList(class_count)

        test_list = class_list.slice(0, n_test)
        train_list = class_list.slice(n_test, class_count)

        test_collections.append(ee.FeatureCollection(test_list))
        train_collections.append(ee.FeatureCollection(train_list))

    if not train_collections:
        return ee.FeatureCollection([]), ee.FeatureCollection([])

    train_data = ee.FeatureCollection(train_collections).flatten()
    test_data = ee.FeatureCollection(test_collections).flatten()
    return train_data, test_data


def build_condensed_confusion(matrix, class_types):
    """Collapse cluster-level confusion matrix into class types."""
    cluster_to_type = {}
    for type_name, cluster_list in class_types.items():
        for c in cluster_list:
            cluster_to_type[c] = type_name

    type_names = list(class_types.keys())
    condensed_cm = {t1: {t2: 0 for t2 in type_names} for t1 in type_names}

    for actual_cluster, row_values in enumerate(matrix):
        actual_type = cluster_to_type.get(actual_cluster, "Unknown")
        if actual_type == "Unknown":
            continue
        for pred_cluster, count in enumerate(row_values):
            pred_type = cluster_to_type.get(pred_cluster, "Unknown")
            if pred_type != "Unknown":
                condensed_cm[actual_type][pred_type] += int(count)

    return condensed_cm, type_names


def print_condensed_confusion(condensed_cm, type_names):
    """Print a formatted condensed confusion matrix."""
    header = "         " + "  ".join([f"{t:>8}" for t in type_names])
    print(header)
    print("         " + "-" * (10 * len(type_names)))
    for actual_type in type_names:
        row_values = [condensed_cm[actual_type][pred_type] for pred_type in type_names]
        row_str = f"{actual_type:>8} |" + "  ".join([f"{v:>8}" for v in row_values])
        print(row_str)


def save_map(map_obj, filename, study_area_name="kafue_flats", subfolder="embeddings_classification"):
    """Save a geemap Map object to HTML.

    Args:
        map_obj: geemap Map object
        filename: output filename
        study_area_name: name of study area for organization
        subfolder: subfolder within study area
    """
    output_path = __config__.OUTPUT_DIR / study_area_name / subfolder / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    map_obj.to_html(str(output_path))
    print(f"✓ Map saved to {output_path} - open in browser to view")
    return map_obj


def load_embeddings_for_year(region, year, embeddings_dataset):
    """Load satellite embeddings for a specific year."""
    embeddings = (
        ee.ImageCollection(embeddings_dataset)
        .filterBounds(region)
        .filter(ee.Filter.calendarRange(year, year, "year"))
        .mosaic()
        .clip(region)
    )
    return embeddings


def load_unified_training_data(config_dict: dict) -> Tuple[pd.DataFrame, dict]:
    """
    Load and merge training data from crop clusters and landcover samples.
    
    Combines:
    - Crop clusters (from _02b_clustering.py): classes 0-8 (or as configured)
    - Landcover samples (from sample_landcover.py): classes 10-15 (configurable)
    
    Parameters
    ----------
    config_dict : dict
        EMBEDDINGS_CLASSIFICATION config with paths and class mappings
    
    Returns
    -------
    df_training : pd.DataFrame
        Merged training data with columns: lon, lat, cluster, year, source
    class_info : dict
        Summary of classes including counts and label mappings
    """
    # Paths from config
    sampled_points_path = Path(config_dict['sampled_points_path'])
    clustering_results_path = Path(config_dict['clustering_results_path'])
    landcover_samples_path = Path(config_dict.get('landcover_samples_path', ''))
    use_landcover = config_dict.get('use_landcover_samples', False)
    
    # Load crop clustering data
    print("\n📊 Loading training data...")
    df_points = pd.read_csv(sampled_points_path)
    df_clusters = pd.read_csv(clustering_results_path)
    print(f"  ✓ Crop samples: {len(df_points)} points, {len(df_clusters)} sample-years")
    
    # Merge coordinates with clustering results
    df_coords = df_points[['lon', 'lat', 'sample_num']].drop_duplicates(subset=['sample_num'])
    df_crop = df_clusters.merge(df_coords, on='sample_num', how='left')
    df_crop['source'] = 'crop'
    
    # Apply crop cluster assignments if provided
    crop_assignments = config_dict.get('crop_cluster_assignments', {})
    if crop_assignments:
        # Keep original cluster numbers for crop classes (0-8)
        df_crop['cluster'] = df_crop['cluster'].astype(int)
    
    df_crop = df_crop[['lon', 'lat', 'cluster', 'year', 'source', 'sample_num']].copy()
    
    # Load landcover samples if enabled
    df_all = df_crop
    
    if use_landcover and landcover_samples_path and Path(landcover_samples_path).exists():
        df_landcover = pd.read_csv(landcover_samples_path)
        print(f"  ✓ Landcover samples: {len(df_landcover)} points")
        
        # Map landcover class names to unified class numbers
        landcover_mapping = config_dict.get('landcover_class_mapping', {
            'urban': 10,
            'water': 11,
            'flooded_veg': 12,
            'trees': 13,
            'shrubs': 14,
            'grass': 15,
        })
        
        df_landcover['cluster'] = df_landcover['class_name'].map(landcover_mapping)
        df_landcover['source'] = 'landcover'
        df_landcover['sample_num'] = -1  # Placeholder for landcover samples
        
        # Ensure required columns exist
        df_landcover = df_landcover[['lon', 'lat', 'cluster', 'year', 'source', 'sample_num']].copy()
        
        # Combine datasets
        df_all = pd.concat([df_crop, df_landcover], ignore_index=True)
        print(f"  ✓ Combined: {len(df_all)} total sample-years")
    else:
        print("  ℹ Landcover samples: disabled or not found")
    
    # Build class info summary
    cluster_labels = config_dict.get('cluster_labels', {})
    class_counts = df_all.groupby('cluster').size().to_dict()
    
    class_info = {
        'n_classes': df_all['cluster'].nunique(),
        'class_counts': class_counts,
        'cluster_labels': cluster_labels,
        'years': sorted(df_all['year'].unique()),
        'crop_clusters': sorted(df_crop['cluster'].unique()),
        'landcover_classes': sorted([v for v in config_dict.get('landcover_class_mapping', {}).values()]) if use_landcover else [],
    }
    
    # Print summary by source
    print(f"\n  Class distribution:")
    for cluster_id in sorted(class_counts.keys()):
        label = cluster_labels.get(cluster_id, f"Class {cluster_id}")
        count = class_counts[cluster_id]
        source = 'crop' if cluster_id < 10 else 'landcover'
        print(f"    [{int(cluster_id):2d}] {label:<20}: {count:>5} samples ({source})")
    
    return df_all, class_info


def create_unified_training_features(
    df_training: pd.DataFrame,
    embeddings_by_year: Dict[int, ee.Image],
    years: Iterable[int],
    scale: int,
    balance: bool = False,
    balance_method: str = "undersample",
    balance_floor: int = 200,
    balance_cap: int = 800,
    batch_size: int = 500,
    max_workers: int = 15,
    count_batches: bool = True,
):
    """
    Create unified training features from ALL sample-years.

    Returns:
        ee.FeatureCollection, class_counts dict
    """
    if balance and balance_method != "none":
        print(f"\n  Class balancing enabled (method: {balance_method})")
        cluster_counts = df_training["cluster"].value_counts()
        print("  Original class distribution:")
        for cluster, count in cluster_counts.sort_index().items():
            print(f"    Cluster {cluster}: {count}")

        if balance_method == "undersample":
            min_count = cluster_counts.min()
            print(f"  Undersampling to {min_count} per class...")
            balanced_dfs = []
            for cluster in cluster_counts.index:
                cluster_df = df_training[df_training["cluster"] == cluster]
                if len(cluster_df) > min_count:
                    cluster_df = cluster_df.sample(n=min_count, random_state=42)
                balanced_dfs.append(cluster_df)
            df_training = pd.concat(balanced_dfs, ignore_index=True)

        elif balance_method == "oversample":
            max_count = cluster_counts.max()
            print(f"  Oversampling to {max_count} per class...")
            balanced_dfs = []
            for cluster in cluster_counts.index:
                cluster_df = df_training[df_training["cluster"] == cluster]
                if len(cluster_df) < max_count:
                    cluster_df = cluster_df.sample(n=max_count, replace=True, random_state=42)
                balanced_dfs.append(cluster_df)
            df_training = pd.concat(balanced_dfs, ignore_index=True)

        elif balance_method == "hybrid":
            print(f"  Hybrid balancing: floor={balance_floor}, cap={balance_cap}")
            balanced_dfs = []
            for cluster in cluster_counts.index:
                cluster_df = df_training[df_training["cluster"] == cluster]
                n = len(cluster_df)
                if n > balance_cap:
                    cluster_df = cluster_df.sample(n=balance_cap, random_state=42)
                    print(f"    Cluster {cluster}: {n} -> {balance_cap} (capped)")
                elif n < balance_floor:
                    cluster_df = cluster_df.sample(n=balance_floor, replace=True, random_state=42)
                    print(f"    Cluster {cluster}: {n} -> {balance_floor} (floored)")
                else:
                    print(f"    Cluster {cluster}: {n} (unchanged)")
                balanced_dfs.append(cluster_df)
            df_training = pd.concat(balanced_dfs, ignore_index=True)

        elif balance_method == "sqrt":
            sqrt_counts = {c: int(np.sqrt(n) * 10) for c, n in cluster_counts.items()}
            print("  Square root balancing (sqrt(n) * 10):")
            balanced_dfs = []
            for cluster, target in sqrt_counts.items():
                cluster_df = df_training[df_training["cluster"] == cluster]
                n = len(cluster_df)
                target = max(min(target, n), min(balance_floor, n))
                if n > target:
                    cluster_df = cluster_df.sample(n=target, random_state=42)
                elif n < target:
                    cluster_df = cluster_df.sample(n=target, replace=True, random_state=42)
                print(f"    Cluster {cluster}: {n} -> {len(cluster_df)}")
                balanced_dfs.append(cluster_df)
            df_training = pd.concat(balanced_dfs, ignore_index=True)

        elif balance_method == "log":
            log_counts = {c: int(np.log2(max(n, 2)) * 50) for c, n in cluster_counts.items()}
            print("  Logarithmic balancing (log2(n) * 50):")
            balanced_dfs = []
            for cluster, target in log_counts.items():
                cluster_df = df_training[df_training["cluster"] == cluster]
                n = len(cluster_df)
                target = max(min(target, n), min(balance_floor, n))
                if n > target:
                    cluster_df = cluster_df.sample(n=target, random_state=42)
                elif n < target:
                    cluster_df = cluster_df.sample(n=target, replace=True, random_state=42)
                print(f"    Cluster {cluster}: {n} -> {len(cluster_df)}")
                balanced_dfs.append(cluster_df)
            df_training = pd.concat(balanced_dfs, ignore_index=True)

        balanced_counts = df_training["cluster"].value_counts()
        total_original = cluster_counts.sum()
        total_balanced = balanced_counts.sum()
        print("\n  Balanced class distribution:")
        for cluster, count in balanced_counts.sort_index().items():
            print(f"    Cluster {cluster}: {count}")
        print(
            f"  Total samples: {total_original} -> {total_balanced} "
            f"({total_balanced / total_original * 100:.1f}% retained)"
        )

    all_features = []
    class_counts: Dict[int, int] = {}
    
    # Batch size and concurrency for sampleRegions
    BATCH_SIZE = int(batch_size)
    MAX_WORKERS = int(max_workers)  # Concurrent GEE requests (don't overwhelm the API)

    def process_batch(batch_data):
        """Process a single batch of points - called concurrently."""
        df_batch, year_emb, sc, batch_idx, n_batches = batch_data
        
        features = []
        for _, row in df_batch.iterrows():
            cluster_id = int(row["cluster"])
            point = ee.Geometry.Point([row["lon"], row["lat"]])
            sample_num = int(row["sample_num"]) if row["sample_num"] >= 0 else -1
            feature = ee.Feature(
                point,
                {
                    "cluster": cluster_id,
                    "sample_num": sample_num,
                    "year": int(row["year"]),
                },
            )
            features.append(feature)

        batch_fc = ee.FeatureCollection(features)
        
        sampled = year_emb.sampleRegions(
            collection=batch_fc,
            properties=["cluster", "sample_num", "year"],
            scale=sc,
            geometries=True,
        )
        
        if count_batches:
            batch_count = _retry_with_backoff(lambda s=sampled: s.size().getInfo())
        else:
            batch_count = None
        return sampled, batch_count, batch_idx, len(df_batch), n_batches

    for year in years:
        df_year = df_training[df_training["year"] == year].copy()

        if len(df_year) == 0:
            print(f"  Warning: No samples for year {year}")
            continue

        # Count classes from dataframe (no GEE call needed)
        for cluster_id in df_year["cluster"].values:
            class_counts[int(cluster_id)] = class_counts.get(int(cluster_id), 0) + 1

        year_embeddings = embeddings_by_year[year]
        year_sampled_count = 0
        
        # Prepare batches
        n_batches = (len(df_year) + BATCH_SIZE - 1) // BATCH_SIZE
        batch_tasks = []
        
        for batch_idx in range(n_batches):
            batch_start = batch_idx * BATCH_SIZE
            batch_end = min((batch_idx + 1) * BATCH_SIZE, len(df_year))
            df_batch = df_year.iloc[batch_start:batch_end]
            batch_tasks.append((df_batch, year_embeddings, scale, batch_idx, n_batches))
        
        # Process batches concurrently
        if n_batches > 1:
            print(f"  Year {year}: Processing {n_batches} batches concurrently (max {MAX_WORKERS} workers)...")
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_batch, task): task[3] for task in batch_tasks}
            
            for future in as_completed(futures):
                sampled, batch_count, batch_idx, batch_size, total_batches = future.result()
                if batch_count is not None:
                    year_sampled_count += batch_count
                all_features.append(sampled)
                
                if n_batches > 1:
                    if batch_count is not None:
                        print(f"    Batch {batch_idx + 1}/{total_batches}: {batch_size} points -> {batch_count} valid")
                    else:
                        print(f"    Batch {batch_idx + 1}/{total_batches}: {batch_size} points")

        if count_batches:
            print(f"  Year {year}: {len(df_year)} input samples -> {year_sampled_count} valid embeddings")
        else:
            print(f"  Year {year}: {len(df_year)} input samples (batch counts skipped)")

    unified_training = ee.FeatureCollection(all_features).flatten()
    return unified_training, class_counts


def train_unified_classifier(training_data, test_data, band_names, config_dict):
    """Train a classifier on unified multi-year training data."""
    classifier_type = config_dict.get("classifier_type", "random_forest")

    if classifier_type == "gradient_boost":
        classifier = ee.Classifier.smileGradientTreeBoost(
            numberOfTrees=config_dict.get("gtb_num_trees", 100),
            shrinkage=config_dict.get("gtb_shrinkage", 0.1),
            samplingRate=config_dict.get("gtb_sampling_rate", 0.8),
            maxNodes=config_dict.get("gtb_max_nodes", 10),
            seed=config_dict.get("gtb_seed", 42),
        ).train(
            features=training_data,
            classProperty="cluster",
            inputProperties=band_names,
        )
    else:
        classifier = ee.Classifier.smileRandomForest(
            numberOfTrees=config_dict["rf_num_trees"],
            seed=config_dict["rf_seed"],
            minLeafPopulation=config_dict["rf_min_leaf_population"],
            bagFraction=config_dict["rf_bag_fraction"],
        ).train(
            features=training_data,
            classProperty="cluster",
            inputProperties=band_names,
        )

    prob_classifier = classifier.setOutputMode("MULTIPROBABILITY")
    train_accuracy = classifier.confusionMatrix()
    test_classified = test_data.classify(classifier)
    test_accuracy = test_classified.errorMatrix("cluster", "classification")

    return classifier, prob_classifier, train_accuracy, test_accuracy, classifier_type


def apply_classification(
    embeddings,
    classifier,
    prob_classifier,
    region,
    crop_mask=None,
    confidence_threshold=None,
    null_value=99,
    n_clusters=6,
):
    """Apply trained classifier to embeddings image with confidence filtering."""
    classified = embeddings.classify(classifier)
    max_prob = None
    prob_bands = None

    if prob_classifier is not None:
        probabilities = embeddings.classify(prob_classifier)
        max_prob = probabilities.arrayReduce(ee.Reducer.max(), [0]).arrayGet([0])
        prob_band_names = [f"prob_{i}" for i in range(n_clusters)]
        prob_bands = probabilities.arrayFlatten([prob_band_names])

        if confidence_threshold is not None:
            confidence_mask = max_prob.gte(confidence_threshold)
            classified = classified.where(confidence_mask.eq(0), null_value)

    if crop_mask is not None:
        classified = classified.updateMask(crop_mask)
        if max_prob is not None:
            max_prob = max_prob.updateMask(crop_mask)
        if prob_bands is not None:
            prob_bands = prob_bands.updateMask(crop_mask)

    classified = classified.clip(region)
    if max_prob is not None:
        max_prob = max_prob.clip(region)
    if prob_bands is not None:
        prob_bands = prob_bands.clip(region)

    return classified, max_prob, prob_bands


def apply_focal_mode_smoothing(classified, kernel_size=3, kernel_type="square"):
    """Apply focal mode filter to smooth classification results."""
    if kernel_type == "circle":
        kernel = ee.Kernel.circle(radius=kernel_size // 2, units="pixels")
    else:
        kernel = ee.Kernel.square(radius=kernel_size // 2, units="pixels")

    smoothed = classified.focal_mode(kernel=kernel, iterations=1)
    smoothed = smoothed.updateMask(classified.mask())
    return smoothed


def analyze_probability_distribution(prob_image, region, sample_scale=100, num_samples=10000):
    """Analyze the distribution of maximum classification probabilities."""
    if prob_image is None:
        return None

    _ = prob_image.sample(
        region=region,
        scale=sample_scale,
        numPixels=num_samples,
        seed=42,
        geometries=False,
    )

    stats = prob_image.reduceRegion(
        reducer=
        ee.Reducer.percentile([5, 10, 25, 50, 75, 90, 95])
        .combine(ee.Reducer.mean(), "", True)
        .combine(ee.Reducer.stdDev(), "", True)
        .combine(ee.Reducer.minMax(), "", True),
        geometry=region,
        scale=sample_scale,
        maxPixels=1e9,
    )

    return stats.getInfo()


def get_seasonal_composites(region, year):
    """Get wet season NDVI, dry season NDVI, and true color composites for a year."""
    dry_start = f"{year}-05-01"
    dry_end = f"{year}-09-30"
    wet_start = f"{year-1}-11-01"
    wet_end = f"{year}-03-31"

    s2 = (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(region)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
    )

    def add_ndvi(img):
        ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
        return img.addBands(ndvi)

    dry_s2 = s2.filterDate(dry_start, dry_end).map(add_ndvi)
    dry_ndvi = dry_s2.select("NDVI").median().clip(region)

    wet_s2 = s2.filterDate(wet_start, wet_end).map(add_ndvi)
    wet_ndvi = wet_s2.select("NDVI").median().clip(region)

    true_color = dry_s2.select(["B4", "B3", "B2"]).median().clip(region)

    return {"wet_ndvi": wet_ndvi, "dry_ndvi": dry_ndvi, "true_color": true_color}


def calculate_cluster_areas_batch(
    classified_by_year,
    region,
    scale=250,
    bestEffort=True,
    batch_size=2,
    timeout_seconds=300,
):
    """Calculate area of each cluster class for all years in batches."""
    pixel_area = ee.Image.pixelArea().divide(1e6)
    all_year_areas = {}
    years = list(classified_by_year.keys())

    for i in range(0, len(years), batch_size):
        batch_years = years[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(years) + batch_size - 1) // batch_size

        print(f"  Processing batch {batch_num}/{total_batches}: years {batch_years}...")
        batch_start = time.time()

        area_computations = {}
        for year in batch_years:
            classified = classified_by_year[year]
            classified_renamed = classified.rename("cluster")
            stacked = pixel_area.addBands(classified_renamed)

            areas = stacked.reduceRegion(
                reducer=ee.Reducer.sum().group(groupField=1, groupName="cluster"),
                geometry=region,
                scale=scale,
                maxPixels=1e10,
                bestEffort=bestEffort,
            )
            area_computations[str(year)] = areas

        try:
            batch_result = ee.Dictionary(area_computations).getInfo()

            for year_str, stats in batch_result.items():
                year = int(year_str)
                year_areas = {}
                for group in stats.get("groups", []):
                    cluster_id = int(group["cluster"])
                    year_areas[cluster_id] = group["sum"]
                all_year_areas[year] = year_areas

            elapsed = time.time() - batch_start
            print(f"    ✓ Batch {batch_num} completed in {elapsed:.1f}s")

        except Exception as e:
            print(f"    ✗ Batch {batch_num} failed: {e}")
            print("    Trying year-by-year fallback...")

            for year in batch_years:
                try:
                    classified = classified_by_year[year]
                    classified_renamed = classified.rename("cluster")
                    stacked = pixel_area.addBands(classified_renamed)

                    result = stacked.reduceRegion(
                        reducer=ee.Reducer.sum().group(groupField=1, groupName="cluster"),
                        geometry=region,
                        scale=scale,
                        maxPixels=1e10,
                        bestEffort=bestEffort,
                    ).getInfo()

                    year_areas = {}
                    for group in result.get("groups", []):
                        cluster_id = int(group["cluster"])
                        year_areas[cluster_id] = group["sum"]
                    all_year_areas[year] = year_areas
                    print(f"      ✓ {year} completed")

                except Exception as e2:
                    print(f"      ✗ {year} failed: {e2}")

    return all_year_areas


def export_classified_to_asset(
    classified_by_year,
    region,
    scale=10,
    asset_folder="irrigation_classification",
    project_id=None,
    overwrite=False,
):
    """Export classified images as GEE assets."""
    project_id = project_id or __config__.PROJECT_ID
    folder_path = f"projects/{project_id}/assets/{asset_folder}"
    ensure_folder_exists(folder_path)

    tasks = []
    for year, classified in classified_by_year.items():
        asset_id = f"projects/{project_id}/assets/{asset_folder}/classified_{year}"
        if overwrite and asset_exists(asset_id):
            delete_asset(asset_id)

        task = ee.batch.Export.image.toAsset(
            image=classified.toByte(),
            description=f"classified_{year}",
            assetId=asset_id,
            region=region,
            scale=scale,
            maxPixels=1e10,
        )
        task.start()
        tasks.append((year, task))
        print(f"  Started asset export for {year}: {asset_id}")

    return tasks


def export_training_samples_to_asset(
    samples_fc,
    year,
    asset_folder="classification_training_samples",
    asset_prefix="training_samples",
    project_id=None,
    overwrite=False,
):
    """Export training samples FeatureCollection to a GEE asset."""
    project_id = project_id or __config__.PROJECT_ID
    asset_id = f"projects/{project_id}/assets/{asset_folder}/{asset_prefix}_{year}"
    folder_path = f"projects/{project_id}/assets/{asset_folder}"
    ensure_folder_exists(folder_path)
    if overwrite and asset_exists(asset_id):
        delete_asset(asset_id)

    task = ee.batch.Export.table.toAsset(
        collection=samples_fc,
        description=f"{asset_prefix}_{year}",
        assetId=asset_id,
    )
    task.start()
    print(f"  Started training sample export for {year}: {asset_id}")
    return year, task


def load_training_samples_from_asset(
    year,
    asset_folder="classification_training_samples",
    asset_prefix="training_samples",
    project_id=None,
):
    """Load training samples FeatureCollection from a GEE asset."""
    project_id = project_id or __config__.PROJECT_ID
    asset_id = f"projects/{project_id}/assets/{asset_folder}/{asset_prefix}_{year}"
    return ee.FeatureCollection(asset_id)


def wait_for_exports(tasks, check_interval=30):
    """Wait for batch export tasks to complete."""
    results = {}
    pending = list(tasks)
    while pending:
        time.sleep(check_interval)
        still_pending = []
        for year, task in pending:
            status = task.status()
            state = status["state"]
            if state == "COMPLETED":
                print(f"  ✓ {year} export completed")
                results[year] = True
            elif state == "FAILED":
                print(
                    f"  ✗ {year} export failed: {status.get('error_message', 'unknown error')}"
                )
                results[year] = False
            elif state in ("READY", "RUNNING"):
                still_pending.append((year, task))
        pending = still_pending
        if pending:
            print(f"  ... {len(pending)} exports still running")

    return results


def calculate_areas_from_assets(years, region, scale=250, asset_folder="irrigation_classification", asset_prefix="classified", project_id=None):
    """Calculate area statistics from exported GEE assets."""
    project_id = project_id or __config__.PROJECT_ID

    pixel_area = ee.Image.pixelArea().divide(1e6)
    all_year_areas = {}

    for year in years:
        asset_id = f"projects/{project_id}/assets/{asset_folder}/{asset_prefix}_{year}"

        try:
            classified = ee.Image(asset_id).rename("cluster")
            stacked = pixel_area.addBands(classified)

            result = stacked.reduceRegion(
                reducer=ee.Reducer.sum().group(groupField=1, groupName="cluster"),
                geometry=region,
                scale=scale,
                maxPixels=1e10,
                bestEffort=True,
            ).getInfo()

            year_areas = {}
            for group in result.get("groups", []):
                cluster_id = int(group["cluster"])
                year_areas[cluster_id] = group["sum"]
            all_year_areas[year] = year_areas
            print(f"  ✓ {year}: {sum(year_areas.values()):.1f} km²")

        except Exception as e:
            print(f"  ✗ {year}: Asset not found or error - {e}")

    return all_year_areas


def build_area_stats_feature_collection(classified_image, region, scale=250):
    """Create a FeatureCollection of per-class areas (km^2) from a classified image."""
    pixel_area = ee.Image.pixelArea().divide(1e6)
    classified_renamed = classified_image.rename("cluster")
    stacked = pixel_area.addBands(classified_renamed)

    groups = stacked.reduceRegion(
        reducer=ee.Reducer.sum().group(groupField=1, groupName="cluster"),
        geometry=region,
        scale=scale,
        maxPixels=1e10,
        bestEffort=True,
    ).get("groups")

    # Use region centroid as feature geometry (required for asset export)
    centroid = region.centroid(1)

    def to_feature(g):
        g = ee.Dictionary(g)
        return ee.Feature(centroid, {
            "cluster": g.get("cluster"),
            "area_km2": g.get("sum"),
        })

    return ee.FeatureCollection(ee.List(groups).map(to_feature))


def export_area_stats_to_asset(
    classified_by_year,
    region,
    scale=250,
    asset_folder="irrigation_classification",
    asset_prefix="area_stats",
    project_id=None,
    overwrite=False,
):
    """Export per-class area statistics as a table asset for each year."""
    project_id = project_id or __config__.PROJECT_ID
    folder_path = f"projects/{project_id}/assets/{asset_folder}"
    ensure_folder_exists(folder_path)
    tasks = []

    for year, classified in classified_by_year.items():
        fc = build_area_stats_feature_collection(classified, region, scale=scale)
        asset_id = f"projects/{project_id}/assets/{asset_folder}/{asset_prefix}_{year}"
        if overwrite and asset_exists(asset_id):
            delete_asset(asset_id)
        task = ee.batch.Export.table.toAsset(
            collection=fc,
            description=f"{asset_prefix}_{year}",
            assetId=asset_id,
        )
        task.start()
        tasks.append((year, task))
        print(f"  Started area stats export for {year}: {asset_id}")

    return tasks


def get_cluster_color(cid, palette, null_val, null_color):
    """Get display color for cluster ID (handles NULL)."""
    if cid == null_val:
        return null_color
    return palette[(cid - 1) % len(palette)]


def get_cluster_display_label(cid, labels, null_val):
    """Get display label for cluster ID (handles NULL)."""
    if cid == null_val:
        return "NULL"
    display_id = cid - 1
    label = labels.get(display_id, "")
    return f"{display_id}: {label}" if label else f"C{display_id}"


def export_to_asset(
    image,
    description,
    region,
    year,
    study_area_name,
    scale,
    asset_folder=None,
    project_id=None,
):
    """Export classified image to GEE Asset organized by study area or folder."""
    project_id = project_id or __config__.PROJECT_ID
    safe_area_name = study_area_name.replace(" ", "_").replace(",", "").replace("/", "_")
    folder_name = asset_folder or safe_area_name
    folder_name = folder_name.replace(" ", "_").replace(",", "").replace("/", "_")
    asset_id = f"projects/{project_id}/assets/{folder_name}/{description}_{year}"

    task = ee.batch.Export.image.toAsset(
        image=image,
        description=f"{description}_{year}",
        assetId=asset_id,
        region=region,
        scale=scale,
        maxPixels=1e10,
    )
    task.start()
    return task, asset_id
