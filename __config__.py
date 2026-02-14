"""
Configuration for Zambezi Irrigation Mapping Project

This file is organized by workflow stage:

1. GLOBAL SETTINGS - Used by all scripts (project ID, study area, dates, seasons)

2. STEP 1: TRAINING DATA COLLECTION
    - _01a_export_landcover.py
    - _01b_sample_landcover.py
    - _01c_clean_landcover_samples.py
    - _01d_collect_crop_samples.py

3. STEP 2: CLUSTERING
    - _02b_clustering.py
    - _02c_sample_map.py

4. STEP 3: CLASSIFICATION + CONSOLIDATION + STATS
    - _03a_classification.py
    - _03b_consolidation.py
    - _03c_area_statistics.py

Each section indicates which script(s) use those settings.
"""

from pathlib import Path

# ============================================================================
# ============================================================================
#                           GLOBAL SETTINGS
#                    (Used by ALL scripts in the pipeline)
# ============================================================================
# ============================================================================

# ----------------------------------------------------------------------------
# Project Identity
# Used by: all scripts via initialize_earth_engine(), gee_utils.py
# ----------------------------------------------------------------------------
PROJECT_ID = 'ee-warnold'
PROJECT_ROOT = Path(__file__).parent

# Optional Earth Engine high-volume endpoint (set to None to use default)
GEE_OPT_URL = None  # e.g., 'https://earthengine-highvolume.googleapis.com'

# Earth Engine query timeout (seconds) for client-side getInfo() calls
GEE_TIMEOUT_SEC = 60

# ----------------------------------------------------------------------------
# OUTPUT PATHS
# Used by: all scripts to determine output locations
# ----------------------------------------------------------------------------
OUTPUT_DIR = PROJECT_ROOT / 'output'

# ----------------------------------------------------------------------------
# Study Area Selection
# Used by: all main scripts (_01a through _03)
# ----------------------------------------------------------------------------
CURRENT_STUDY_AREA = 'kafue_flats'  # Options: 'kafue_flats', 'zambezi_basin'

def _get_study_area_geometries():
    """Get study area geometries. Requires ee to be imported."""
    import ee
    return {
        'kafue_flats': ee.Geometry.Polygon([[
            # full watershed box
            [25.29357450684133, -17.124202489229475],
            [28.77624052246633, -17.124202489229475],
            [28.77624052246633, -14.699658421068504],
            [25.29357450684133, -14.699658421068504],
            [25.29357450684133, -17.124202489229475]
            # smaller box for testing
            # [27.56953501135865, -16.180738994806912], 
            # [28.526718849249274, -16.180738994806912],
            # [28.526718849249274, -15.600905864806341],
            # [27.56953501135865, -15.600905864806341],
            # [27.56953501135865, -16.180738994806912]
        ]]),
        'zambezi_basin': ee.Geometry.Polygon([[
            [20.0, -8.0],   # Northwest corner
            [36.0, -8.0],   # Northeast corner
            [36.0, -20.0],  # Southeast corner
            [20.0, -20.0],  # Southwest corner
            [20.0, -8.0]    # Close polygon
        ]]),  # Approximate Zambezi River Basin boundary
    }

STUDY_AREAS = {
    'kafue_flats': {
        'name': 'Kafue Flats, Zambia',
        'geometry': None,  # Lazy loaded
        'center': [-15.891, 28.048],
        'zoom': 10
    },
    'zambezi_basin': {
        'name': 'Zambezi River Basin',
        'geometry': None,
        'center': [-15.0, 28.0],
        'zoom': 6
    }
}

# ----------------------------------------------------------------------------
# Temporal Parameters
# Used by: _01d_collect_crop_samples.py, _02b_clustering.py (date filtering, seasonal analysis)
# ----------------------------------------------------------------------------
START_DATE = '2019-01-01'  # Analysis start (S2 data availability)
END_DATE = '2024-12-31'    # Analysis end

# Seasonal definitions (Southern Hemisphere) - defined by month indices
DRY_SEASON_MONTHS = [5, 6, 7, 8, 9]   # May-September (winter)
PEAK_DRY_MONTHS = [8, 9]              # Aug-Sep (for NDVI statistics)
WET_SEASON_MONTHS = [11, 12, 1, 2, 3] # Nov-March (crosses year boundary)

# ----------------------------------------------------------------------------
# Data Sources
# Used by: gee_utils.py (all)
# ----------------------------------------------------------------------------
DATASETS = {
    'sentinel2': 'COPERNICUS/S2_SR_HARMONIZED',
    'chirps': 'UCSB-CHG/CHIRPS/DAILY',
}

# Cloud masking threshold used in sample extraction
CLOUD_SCORE_THRESHOLD = 0.60


# ============================================================================
# ============================================================================
#          (_01a, _01b, _01c_*.py for landcover; _01d_*.py for crops)
# ============================================================================
# ============================================================================
# Workflow:
#   Landcover: Export DW yearly → Sample points → Clean samples
#   Crops: Sample crop pixels → Extract S2 time series → Generate figures

# ----------------------------------------------------------------------------
# Sample Collection Settings
# ----------------------------------------------------------------------------
START_SAMPLE_ID = 3001   # First sample number (allows batch collection: 1-20, then 21-40)
END_SAMPLE_ID = 4000    # Last sample number (inclusive)

# ----------------------------------------------------------------------------
# Training Crop Mask - WHERE TO SAMPLE CROP POINTS
# Used by: _01d_collect_crop_samples.py (controls which pixels are sampled)
# ----------------------------------------------------------------------------
TRAINING_CROP_MASK = {
    # Only option: 'annual_dw_landcover'
    'source': 'annual_dw_landcover',
    
    # For 'annual_dw_landcover' (sample pixels classified as crop in ALL years):
    'asset_pattern': f'projects/{PROJECT_ID}/assets/classification/dw_landcover_{{year}}',
    'years': [2019, 2020, 2021, 2022, 2023, 2024],
    'crop_class': 4,  # Dynamic World crop class
}

# ----------------------------------------------------------------------------
# Training Asset Export Settings
# Used by: _01d_collect_crop_samples.py (asset export destination)
# ----------------------------------------------------------------------------
TRAINING_ASSET_SUBFOLDER = 'training'  # Asset path: projects/<id>/assets/<study_area>/<subfolder>

# ----------------------------------------------------------------------------
# Land Cover Sampling Settings
# Used by: _01a_export_landcover.py, _01b_sample_landcover.py, _01c_clean_landcover_samples.py
# ----------------------------------------------------------------------------
LANDCOVER = {
    # Target classes from Dynamic World (exclude crops=4, snow=8)
    'target_classes': [0, 1, 2, 3, 5, 6, 7],
    'class_names': {
        0: 'water', 1: 'trees', 2: 'grass', 3: 'flooded_veg',
        5: 'shrubs', 6: 'urban', 7: 'bare'
    },
    
    # Sampling parameters
    'samples_per_class_per_year': 2000,
    'scale': 10,  # Native Dynamic World resolution
    'years': [2019, 2020, 2021, 2022, 2023, 2024],
    # Grass cleaning parameters
    'grass_buffer_m': 100,
    # Minimum number of years a sample must match its class to be kept
    # (set to len(years) for strict consistency across all years)
    'consistency_threshold_years': 4,
    
    # Export settings
    'asset_folder': f'projects/{PROJECT_ID}/assets/classification',
    'asset_prefix': 'dw_landcover',  # Asset name: dw_landcover_YYYY
}


# ============================================================================
# ============================================================================
#                          STEP 2: CLUSTERING
#                          (_02b_clustering.py)
# ============================================================================
# ============================================================================
# Workflow: Load time series from Step 1 → DTW distance matrix →
#           Hierarchical clustering → Assign cluster labels

CLUSTERING = {
    # Indices to load from training data
    's2_indices': ['NDVI', 'NDWI', 'NDMI', 'composite_index'],
    
    # Which index to cluster on
    'clustering_index': 'composite_index',  # (NDVI - NDWI + NDMI) / 3
    
    # Calendar year settings (Jan 1 - Dec 31)
    'min_obs_per_year': 20,   # Minimum S2 observations required per year
    'use_dry_season_only': True,  # If True, only compare dry season patterns
    
    # DTW (Dynamic Time Warping) parameters
    'dtw_window': 15,           # Sakoe-Chiba band width (days)
    'dtw_normalize': False,     # Normalize by path length
    'use_precomputed_dtw': True,  # Load existing DTW matrix if available
    
    # Sample limiting (for faster iteration during development)
    'max_samples': None,  # None = load all samples
    
    # Hierarchical clustering
    'n_clusters': 5,            # Number of clusters
    'linkage_method': 'ward',    # Options: 'single', 'complete', 'average', 'ward'
    'min_cluster_size': 20,      # Reassign tiny clusters to nearest larger
    
    # Visualization
    'max_plot_columns': 3,
}


# ============================================================================
# ============================================================================
#                        STEP 3: CLASSIFICATION
#                        (_03a_classification.py)
# ============================================================================
# ============================================================================
# Workflow: Load cluster labels + landcover samples → Train classifier on
#           satellite embeddings → Apply to study area → Generate maps/stats

# ----------------------------------------------------------------------------
# Embeddings Classification Settings
# Used by: _03a_classification.py, src/classification_utils.py
# ----------------------------------------------------------------------------
# Based on AlphaEarth Foundations (Brown et al. 2025)
# - 256 embedding dimensions per pixel, annual composites at 10m resolution
#
# UNIFIED CLASSIFICATION: Combines crop clusters + landcover samples
# - Crop classes (0-7): From _02b_clustering.py → irrigated/rainfed labels
# - Landcover classes (10-16): From _01b_sample_landcover.py → non-crop types

EMBEDDINGS_CLASSIFICATION = {
    # --- Data Sources ---
    'embeddings_dataset': 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL',
    
    # Crop samples (from _01d + _02b)
    'clustering_results_path': f'output/{CURRENT_STUDY_AREA}/training/clustering/clustering_results.csv',
    'sampled_points_path': f'output/{CURRENT_STUDY_AREA}/training/sampled_points_data.csv',

    # Landcover samples (from _01b + _01c) - non-crop classes
    'landcover_samples_path': f'output/{CURRENT_STUDY_AREA}/training/landcover_samples_cleaned.csv',
    'use_landcover_samples': True,  # Set False to use only crop clusters
    
    # --- Classification Years ---
    # Year = calendar year containing the dry season (e.g., 2020 = Nov 2019 - Oct 2020)
    'classification_years': [2019, 2020, 2021, 2022, 2023, 2024],
    
    # --- Classifier Selection ---
    # Options: 'random_forest', 'gradient_boost'
    # Random Forest recommended for 20+ classes
    'classifier_type': 'gradient_boost',
    
    # Random Forest parameters (optimized for ~20 classes)
    'rf_num_trees': 300,           # More trees for more classes
    'rf_seed': 42,
    'rf_min_leaf_population': 20,  # Lower for fewer samples per class
    'rf_bag_fraction': 0.8,
    
    # Gradient Tree Boost parameters (alternative)
    'gtb_num_trees': 350,
    'gtb_shrinkage': 0.08,         # Lower learning rate for stability with many classes
    'gtb_sampling_rate': 0.6,
    'gtb_max_nodes': 20,           # More nodes for class complexity
    'gtb_seed': 42,
    
    # --- Training Data Balancing ---
    # Important for 20 classes with varying sample sizes
    'balance_training': True,
    'balance_method': 'hybrid',  # Options: 'none', 'undersample', 'oversample', 'hybrid', 'sqrt', 'log'
    'balance_floor': 250,         # Min samples per class (lower for sparse classes)
    'balance_cap': 2000,          # Max samples per class (avoid domination)
    'balanced_split': True,       # Stratify train/test split by class

    # --- Training Sample Extraction (GEE) ---
    'training_batch_size': 300,    # Smaller batches reduce aggregation load
    'training_max_workers': 4,     # Limit concurrency to avoid 429 errors
    'training_count_batches': False,  # Skip per-batch size() getInfo calls

    # --- Processing ---
    'scale': 10,  # Match embeddings native resolution

    # --- Training Samples (Embeddings) ---
    # Auto-detect: loads from asset if exists, else creates directly and exports
    'training_samples_asset_folder': f"{CURRENT_STUDY_AREA}/{TRAINING_ASSET_SUBFOLDER}",
    'training_samples_asset_prefix': 'training_samples',
    'training_samples_overwrite_assets': False,  # Overwrite if re-creating
    'skip_training_sample_counts': False,  # Skip size() getInfo calls to reduce timeouts
    
    # --- Export ---
    'asset_folder': f"{CURRENT_STUDY_AREA}/classification",
    'output_subfolder': 'embeddings_classification',
    'classification_export_mode': 'batch',  # Options: 'none', 'batch'
    'classified_overwrite_assets': True,

    # =========================================================================
    # CLASS DEFINITIONS - Unified numbering scheme
    # =========================================================================
    # Classes 0-7:   Crop clusters (from _02b_clustering.py)
    # Classes 10-16: Landcover types (from _01b_sample_landcover.py)
    #
    # This allows flexible assignment of crop clusters to irrigated/rainfed
    # while keeping landcover classes separate and stable.
    # =========================================================================

    # Optional: reorder crop cluster IDs using original IDs in desired new order.
    # Example: [3, 1, 0, 2, 4, 5, 6, 7] maps old 3->new 0, old 1->new 1, etc.
    # Set to None or [] to disable.
    'cluster_reorder': [],

    # --- Crop Cluster Assignments ---
    # Map _02b_clustering.py cluster numbers (0-7) to irrigation types
    # UPDATE THESE based on your clustering results analysis!
    'crop_cluster_assignments': {
        0: 'Rainfed',
        1: 'Rainfed',
        2: 'Irrigated',
        3: 'Irrigated',
        4: 'Irrigated',
    },
    
    # --- Landcover Class Mapping ---
    # Maps _01b_sample_landcover.py class names to unified class numbers
    # These start at 10 to avoid collision with crop clusters (0-7)
    'landcover_class_mapping': {
        'urban': 5,
        'water': 6,
        'flooded_veg': 7,
        'trees': 8,
        'shrubs': 9,
        'grass': 10,
        'bare': 11,
    },
    
    # --- Visualization ---
    # Palette indexed by original class ID (0-8 for crops, 10-15 for landcover)
    # The code maps these to sequential indices for GEE training
    'cluster_palette': {
        # Crop clusters (0-8)
        0: "#f4912e",   # Rainfed
        1: "#bd5e00",   # Rainfed
        2: "#4B0974",   # Irrigated
        3: "#67179C",   # Irrigated
        4: "#842FBD",   # Irrigated
        # Landcover classes (10-16)
        5: "#320000",  # Urban (gray)
        6: "#0055be",  # Water (blue)
        7: "#00a86b",  # Flooded vegetation (teal)
        8: "#1a6e1a",  # Trees (forest green)
        9: "#beda20",  # Shrubs (goldenrod)
        10: "#90ee90",  # Grass (light green)
        11: "#a59b8f",  # Bare (tan/brown)
    },
    
    # Human-readable labels for all classes
    'cluster_labels': {
        # Crop clusters
        0: 'Rainfed (0)',
        1: 'Rainfed (1)',
        2: 'Irrigated (2)',
        3: 'Irrigated (3)',
        4: 'Irrigated (4)',
        # Landcover classes
        5: 'Urban',
        6: 'Water',
        7: 'Flooded Veg.',
        8: 'Trees',
        9: 'Shrubs',
        10: 'Grass',
        11: 'Bare',
    },
    
    # Aggregated class types for area statistics
    'class_types': {
        'Irrigated': [2, 3, 4],  # UPDATE based on crop_cluster_assignments
        'Rainfed': [0, 1],  # UPDATE based on crop_cluster_assignments
        'Urban': [5],
        'Water': [6],
        'Native Veg.': [7, 8, 9, 10],
        'Bare': [11],
    },
}


# ============================================================================
# ============================================================================
#                        STEP 3c: AREA STATISTICS
#                        (_03c_area_statistics.py)
# ============================================================================
# ============================================================================
# Workflow: Load yearly + consolidated assets → Calculate per-class areas → Generate charts

AREA_STATISTICS = {
    # Consolidated assets to load (from Step 3b exports)
    'asset_folder': f"{CURRENT_STUDY_AREA}/classification",
    'consolidated_assets': {
        'early': {
            'asset_name': 'consolidated_2024_weighted',
            'label': 'Early (2019)',
            'enabled': True,
        },
        'late': {
            'asset_name': 'consolidated_2019_weighted',
            'label': 'Late (2024)',
            'enabled': True,
        },
        'combined': {
            'asset_name': 'consolidated_all_years_classified',
            'label': 'All-Years',
            'enabled': True,
        },
    },

    # Consolidated class-type grouping (class IDs from CONSOLIDATION.aggregated_labels)
    'class_types': {
        'Irrigated': [1],
        'Rainfed': [0],
        'Urban': [2],
        'Water': [3],
        'Native Veg.': [4, 5, 6, 7],
        'Bare': [8],
    },

    # Area calculation scale (meters) - larger = faster but less precise
    'scale': 10,  # 30m is a common compromise for area stats

    # Plot generation controls
    # If True, regenerate plots from existing CSV outputs without re-computing GEE statistics
    'regenerate_plots_only': False,
    # Slide-friendly 16:9 figure sizes (inches)
    'plot_figsize': [6.5, 3.5],
    'plot_heatmap_figsize': [12.0, 7.0],
    'plot_dpi': 300,

    # Export area stats to GEE assets
    'export_assets': False,
    'asset_prefix_stats': 'area_stats_consolidated',
    'overwrite_assets': True,
    'wait_for_exports': True,

    # Output
    'output_subfolder': 'embeddings_classification',
}


# ============================================================================
# ============================================================================
#                        STEP 3b: CONSOLIDATION
#                        (_03b_consolidation.py)
# ============================================================================
# ============================================================================
# Workflow: Load classified assets → Create weighted confidence scores →
#           Aggregate irrigated/rainfed classes → Export consolidated assets

CONSOLIDATION = {
    # --- Input Classification Assets ---
    'asset_folder': f"{CURRENT_STUDY_AREA}/classification",
    'asset_prefix': 'classified',  # Expects: classified_YYYY
    
    # Years to consolidate (should match years exported by _03a_classification.py)
    'years': [2019, 2020, 2021, 2022, 2023, 2024],
    
    # --- Class Aggregation ---
    # Combine individual classes into aggregated classes for the consolidated output
    # Original classes 0,1 (rainfed) → 0, classes 2,3,4 (irrigated) → 1
    # Landcover classes 5-11 are remapped to 2-8
    'class_aggregation': {
        # Rainfed clusters → Rainfed (0)
        0: 0,  # Rainfed (0) → Aggregated Rainfed
        1: 0,  # Rainfed (1) → Aggregated Rainfed
        # Irrigated clusters → Irrigated (1)
        2: 1,  # Irrigated (2) → Aggregated Irrigated
        3: 1,  # Irrigated (3) → Aggregated Irrigated
        4: 1,  # Irrigated (4) → Aggregated Irrigated
        # Landcover classes (keep separate, remap to sequential)
        5: 2,   # Urban → 2
        6: 3,   # Water → 3
        7: 4,   # Flooded Veg → 4
        8: 5,   # Trees → 5
        9: 6,   # Shrubs → 6
        10: 7,  # Grass → 7
        11: 8,  # Bare → 8
    },
    
    # Labels for aggregated classes
    'aggregated_labels': {
        0: 'Rainfed',
        1: 'Irrigated',
        2: 'Urban',
        3: 'Water',
        4: 'Flooded Veg.',
        5: 'Trees',
        6: 'Shrubs',
        7: 'Grass',
        8: 'Bare',
    },
    
    # Palette for aggregated classes
    'aggregated_palette': {
        0: '#f4912e',  # Rainfed (orange)
        1: '#67179C',  # Irrigated (purple)
        2: '#320000',  # Urban
        3: '#0055be',  # Water
        4: '#00a86b',  # Flooded Veg
        5: '#1a6e1a',  # Trees
        6: '#beda20',  # Shrubs
        7: '#90ee90',  # Grass
        8: '#a59b8f',  # Bare
    },
    
    # --- Temporal Weighting ---
    # Weights for each year (will be normalized to sum to 1.0)
    # Early-weighted: 2019 heaviest, decreasing to 2024
    'early_weights': {
        2019: 20.0,
        2020: 10.0,
        2021: 5.0,
        2022: 3.0,
        2023: 1.0,
        2024: 0.0,
    },
    # Late-weighted: 2024 heaviest, decreasing to 2019
    'late_weights': {
        2019: 0.0,
        2020: 1.0,
        2021: 3.0,
        2022: 5.0,
        2023: 10.0,
        2024: 20.0,
    },
    # Equal weighting across all years (used for all-years combined classified layer)
    'combined_weights': {
        2019: 1.0,
        2020: 1.0,
        2021: 1.0,
        2022: 1.0,
        2023: 1.0,
        2024: 1.0,
    },
    
    # --- Spatial Confidence (Neighborhood Analysis) ---
    # Include neighbor pixel consistency in confidence score
    'use_spatial_confidence': True,
    'neighborhood_radius': 1,  # Kernel radius in pixels (1 = 3x3 window)
    'spatial_weight': 0.2,     # Weight of spatial consistency vs temporal (0-1)
                               # Final score = (1-spatial_weight)*temporal + spatial_weight*spatial
    
    # --- Processing ---
    'scale': 10,  # Match classification resolution
    'process_combined_only': False,  # If True, skip early/late consolidation and export only combined
    
    # --- Export ---
    'export_early_weighted': True,
    'export_late_weighted': True,
    'export_combined_classified': True,
    'early_asset_name': 'consolidated_2024_weighted',
    'late_asset_name': 'consolidated_2019_weighted',
    'combined_asset_name': 'consolidated_all_years_classified',
    'overwrite_assets': True,
    
}


# ============================================================================
# ============================================================================
#                      STEP 3d: TRANSITION ANALYSIS
#                      (_03d_transition_analysis.py)
# ============================================================================
# ============================================================================

TRANSITION_ANALYSIS = {
    'years': CONSOLIDATION.get('years', [2019, 2020, 2021, 2022, 2023, 2024]),
    'asset_folder': EMBEDDINGS_CLASSIFICATION.get('asset_folder', f"{CURRENT_STUDY_AREA}/classification"),
    'asset_prefix': CONSOLIDATION.get('asset_prefix', 'classified'),
    'transition_schema': 'detailed',  # Options: 'detailed', 'aggregated'

    # Detailed schema defaults
    'class_labels': EMBEDDINGS_CLASSIFICATION.get('cluster_labels', {}),
    'class_palette': EMBEDDINGS_CLASSIFICATION.get('cluster_palette', {}),
    'class_types': EMBEDDINGS_CLASSIFICATION.get('class_types', {}),

    # Aggregated schema mapping (used when transition_schema='aggregated')
    'class_aggregation': CONSOLIDATION.get('class_aggregation', {}),
    'aggregated_labels': CONSOLIDATION.get('aggregated_labels', {}),
    'aggregated_palette': CONSOLIDATION.get('aggregated_palette', {}),

    'scale': 10,
    'min_flow_km2': 20.0,
    'generate_pairwise_flow_plots': False,
    'generate_multiyear_flow_plot': True,
    'multiyear_flow_width_in': 6.5,
    'generate_major_focus_flow_plots': True,
    'major_focus_categories': [],  # [] = generate for all major categories

    # Consolidated early/late transition (from _03b outputs)
    'compute_consolidated_early_late_transition': True,
    'consolidated_early_asset_name': CONSOLIDATION.get('early_asset_name', 'consolidated_2024_weighted'),
    'consolidated_late_asset_name': CONSOLIDATION.get('late_asset_name', 'consolidated_2019_weighted'),
    'consolidated_use_confidence_weight': True,
    'consolidated_use_probability_weight': False,
    'consolidated_min_confidence': 10,  # confidence is 0-100 in consolidated assets

    'output_subfolder': 'embeddings_classification/transitions',
}


# ============================================================================
# ============================================================================
#                         HELPER FUNCTIONS
# ============================================================================
# ============================================================================

def get_study_area(name=None):
    """Get study area configuration with lazy-loaded geometry.

    This function is required because ee.Geometry objects can only be created
    after ee.Initialize() is called. It lazy-loads the geometry on first access.

    Used by: all main scripts (_01a through _03)

    Parameters
    ----------
    name : str, optional
        Study area name. Defaults to CURRENT_STUDY_AREA.

    Returns
    -------
    dict
        Study area config with 'name', 'geometry', 'center', 'zoom'
    """
    if name is None:
        name = CURRENT_STUDY_AREA
    
    area = STUDY_AREAS[name].copy()
    
    # Lazy load geometry if not already loaded
    if area['geometry'] is None and name in ['kafue_flats', 'zambezi_basin']:
        geometries = _get_study_area_geometries()
        area['geometry'] = geometries[name]
    
    return area
