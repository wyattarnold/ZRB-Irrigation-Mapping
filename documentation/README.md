# Zambezi Irrigation Mapping

Classify agricultural fields as **irrigated** or **rainfed** using Google Earth Engine, multi-temporal spectral indices, DTW-based clustering, and satellite embeddings.

## Project Overview

This project uses a multi-step pipeline to map irrigation in the Zambezi River Basin (Zambia):

1. **Training Data Collection** - Sample landcover and crop pixels, extract time series
2. **Clustering** - Group crop pixels by irrigation pattern using DTW
3. **Classification** - Apply satellite embeddings + ML to classify the full region

**Study Area**: Kafue Flats, Zambia | **Period**: 2019-2024 | **Project ID**: `ee-warnold`

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STEP 1: TRAINING DATA COLLECTION                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ _01a_export_landcover.py  → Export yearly Dynamic World classifications    │
│ _01b_sample_landcover.py  → Sample non-crop landcover points (7 classes)   │
│ _01c_clean_landcover_samples.py → Clean + validate landcover samples       │
│ _01d_collect_crop_samples.py → Sample crop pixels, extract S2 time series  │
├─────────────────────────────────────────────────────────────────────────────┤
│                         STEP 2: CLUSTERING                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│ _02b_clustering.py → DTW-based hierarchical clustering on composite index  │
│ _02c_sample_map.py → Standalone sample map (landcover + clusters by year)  │
├─────────────────────────────────────────────────────────────────────────────┤
│                       STEP 3: CLASSIFICATION                               │
├─────────────────────────────────────────────────────────────────────────────┤
│ _03a_classification.py → Train on embeddings, classify full study area     │
│ _03b_consolidation.py → Multi-year weighted consolidation                  │
│ _03c_area_statistics.py → Area stats (yearly + consolidated, final step)  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# 1. Setup environment
conda activate gee
earthengine authenticate

# 2. Run pipeline (execute in order)
python _01a_export_landcover.py      # Export DW yearly composites
python _01b_sample_landcover.py      # Sample landcover points
python _01c_clean_landcover_samples.py  # Clean landcover samples
python _01d_collect_crop_samples.py  # Collect crop samples with time series
python _02b_clustering.py            # Cluster crop patterns
python _02c_sample_map.py            # Standalone sample map (optional)
python _03a_classification.py        # Classify using embeddings
python _03b_consolidation.py         # Multi-year consolidation
python _03c_area_statistics.py       # Final area stats (yearly + consolidated)
```

---

## Detailed Pipeline Steps

### Step 1a: Export Landcover (`_01a_export_landcover.py`)

Exports yearly modal land cover classification from Dynamic World to GEE assets.

- **Input**: Dynamic World imagery (2019-2024)
- **Output**: GEE assets `projects/{id}/assets/classification/dw_landcover_{year}`
- **Purpose**: Create consistent yearly landcover baselines for sampling

### Step 1b: Sample Landcover (`_01b_sample_landcover.py`)

Samples non-crop landcover points using spatially-uniform grid-based thinning.

- **Input**: Landcover assets from Step 1a
- **Output**: `output/{study_area}/training/landcover_samples.csv`
- **Classes**: Water, Trees, Grass, Flooded Veg, Shrubs, Urban, Bare (7 classes)
- **Strategy**: ~2000 samples/class/year with spatial uniformity

### Step 1c: Clean Landcover (`_01c_clean_landcover_samples.py`)

Cleans and validates landcover samples:
1. Removes grass samples within 100m of crop pixels
2. Filters samples that don't match their class across ≥N years

- **Input**: `landcover_samples.csv`
- **Output**: `landcover_samples_cleaned.csv`
- **Removed samples**: Saved to separate CSVs for review

### Step 1d: Collect Crop Samples (`_01d_collect_crop_samples.py`)

Samples crop pixels and extracts multi-year Sentinel-2 time series.

- **Input**: Crop mask (pixels classified as crop in ALL years)
- **Output**: 
  - `output/{study_area}/training/sampled_points_data.csv`
  - `output/{study_area}/training/timeseries/*.csv` (per-sample time series)
  - `output/{study_area}/training/figures/*.png` (NDVI/precip plots)
- **Indices**: NDVI, NDWI, NDMI, composite_index
- **Period**: Full analysis period (2019-2024)

### Step 2b: Clustering (`_02b_clustering.py`)

Clusters crop samples by irrigation pattern using Dynamic Time Warping (DTW).

- **Input**: Time series from Step 1d (`_01d_collect_crop_samples.py`)
- **Algorithm**: Hierarchical clustering with DTW distance on composite index
- **Output**:
  - `output/{study_area}/training/clustering/clustering_results.csv`
  - `output/{study_area}/training/clustering/dtw_matrix.npy`
  - Dendrograms, cluster pattern plots
- **Result**: Crop clusters labeled by irrigation type (irrigated vs rainfed)

### Step 2c: Sample Map (`_02c_sample_map.py`)

Builds a standalone interactive QA map for sampled training points.

- **Inputs**:
  - Cleaned landcover samples (`landcover_samples_cleaned.csv`)
  - Clustering results (`clustering_results.csv`)
  - Sample coordinates (`sampled_points_data.csv`)
  - Dynamic World yearly assets (`dw_landcover_{year}`)
- **Output**:
  - `output/{study_area}/training/samples_combined_map.html`
- **Includes**:
  - Year-specific clustered sample layers
  - Cleaned landcover sample layers
  - DW yearly layers + Sentinel-2 true color yearly layers
  - Legends for both landcover classes and cluster IDs

### Step 3a: Classification (`_03a_classification.py`)

Trains year-specific classifiers using Google Satellite Embeddings V1.

- **Input**: 
  - Clustering results (crop classes 0-4)
  - Landcover samples (classes 5-11)
  - Satellite embeddings (256 dimensions/pixel)
- **Classifier**: Gradient Boost or Random Forest
- **Output**: 
  - GEE assets: `{study_area}/classification/classified_{year}`
  - Accuracy metrics, confusion matrices
- **Key**: Separate classifier per year → avoids temporal leakage

### Step 3b: Consolidation (`_03b_consolidation.py`)

Creates consolidated multi-year classification with weighted confidence.

- **Input**: Yearly classification assets
- **Approach**:
  - Aggregates detailed classes (5 crop → 2: irrigated/rainfed)
  - Calculates temporal + spatial consistency scores
  - Creates early-weighted and late-weighted consolidations
- **Output**:
  - `consolidated_early_weighted` (prioritizes earlier years)
  - `consolidated_late_weighted` (prioritizes recent years)
  - `consolidated_all_years_classified` (all-years combined)
  - Each includes classification + confidence bands

### Step 3c (Final): Area Statistics (`_03c_area_statistics.py`)

Calculates per-class area statistics from yearly and consolidated outputs.

- **Input**:
  - Yearly classification assets from Step 3a
  - Consolidated assets from Step 3b (early, late, combined)
- **Output**:
  - Yearly area statistics table (km² and % by class and year)
  - Consolidated area statistics table (km² and % by class and variant)
  - Improved comparison plots (stacked bars + heatmap)
  - Optional: GEE table asset exports

---

## Configuration

All pipeline settings are centralized in `__config__.py`:

```python
# Key settings
CURRENT_STUDY_AREA = 'kafue_flats'
START_DATE = '2019-01-01'
END_DATE = '2024-12-31'
PROJECT_ID = 'ee-warnold'

# Seasonal definitions
DRY_SEASON_MONTHS = [5, 6, 7, 8, 9]   # May-September
WET_SEASON_MONTHS = [11, 12, 1, 2, 3] # Nov-March
```

See `__config__.py` for detailed documentation of each setting section.

---

## Output Structure

```
output/{study_area}/
├── training/
│   ├── sampled_points_data.csv      # Crop sample locations
│   ├── landcover_samples.csv        # Raw landcover samples
│   ├── landcover_samples_cleaned.csv # Cleaned landcover samples
│   ├── timeseries/                  # Per-sample S2 time series
│   ├── figures/                     # NDVI/precip plots
│   └── clustering/
│       ├── clustering_results.csv   # Cluster assignments
│       ├── dtw_matrix.npy           # DTW distance matrix
│       └── *.png                    # Dendrograms, patterns
├── embeddings_classification/
│   ├── accuracy/                    # Per-year accuracy metrics
│   └── *.html                       # Interactive maps
└── consolidation/
    └── *.html                       # Consolidated maps
```

---

## Data Sources

| Dataset | GEE Path | Resolution |
|---------|----------|------------|
| Sentinel-2 SR | `COPERNICUS/S2_SR_HARMONIZED` | 10m |
| Cloud Score+ | `GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED` | 10m |
| Satellite Embeddings | `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` | 10m |
| CHIRPS Precipitation | `UCSB-CHG/CHIRPS/DAILY` | 5km |
| Dynamic World | `GOOGLE/DYNAMICWORLD/V1` | 10m |

---

## Key Files

| File | Purpose |
|------|---------|
| `__config__.py` | Centralized configuration |
| `src/gee_utils.py` | GEE authentication, data loading, S2 processing |
| `src/clustering_utils.py` | DTW computation, gap-filling, clustering |
| `src/classification_utils.py` | Classifier training, export, area stats |

---

## Class Definitions

### Crop Classes (from clustering)
| Class | Label | Description |
|-------|-------|-------------|
| 0 | Rainfed (0) | Rainfed agriculture pattern |
| 1 | Rainfed (1) | Rainfed agriculture pattern |
| 2 | Irrigated (2) | Irrigated - sustained dry season NDVI |
| 3 | Irrigated (3) | Irrigated - sustained dry season NDVI |
| 4 | Irrigated (4) | Irrigated - sustained dry season NDVI |

### Landcover Classes (from Dynamic World)
| Class | Label | DW Class |
|-------|-------|----------|
| 5 | Urban | 6 |
| 6 | Water | 0 |
| 7 | Flooded Veg | 3 |
| 8 | Trees | 1 |
| 9 | Shrubs | 5 |
| 10 | Grass | 2 |
| 11 | Bare | 7 |

---

## References

- **Satellite Embeddings**: Brown et al. (2025) - AlphaEarth Foundations
- **Dynamic World**: Brown et al. (2022) - Near real-time land use classification
- **DTW**: Sakoe & Chiba (1978) - Dynamic programming algorithm optimization

---

**Need help?** See `__config__.py` for detailed parameter documentation.
