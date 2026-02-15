# Zambezi Irrigation Mapping

Classify agricultural fields as **irrigated** or **rainfed** using Google Earth Engine, multi-temporal spectral indices, DTW-based clustering, and satellite embeddings.

## Project Overview

This project uses a multi-step pipeline to map irrigation in the Zambezi River Basin (Zambia):

1. **Training Data Collection** - Sample landcover and crop pixels, extract time series
2. **Clustering** - Group crop pixels by irrigation pattern using DTW
3. **Classification** - Apply satellite embeddings + ML to classify the full region

**Study Area**: Kafue Flats, Zambia (~100,000 km²) | **Period**: 2019-2024 | **Resolution**: 10 m | **Project ID**: `ee-warnold`

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
│ _03c_area_statistics.py → Area stats (yearly + consolidated)               │
│ _03d_transition_analysis.py → Transition flows, matrices, alluvial plots   │
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
python _03d_transition_analysis.py   # Transition flows and alluvial plots
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
  - Satellite embeddings (64 dimensions/pixel, int8)
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

### Step 3c: Area Statistics (`_03c_area_statistics.py`)

Calculates per-class area statistics from yearly and consolidated outputs.

- **Input**:
  - Yearly classification assets from Step 3a
  - Consolidated assets from Step 3b (early, late, combined)
- **Output**:
  - Yearly area statistics table (km² and % by class and year)
  - Consolidated area statistics table (km² and % by class and variant)
  - Publication-quality comparison plots (stacked bars, line panels, consolidated summaries)

### Step 3d: Transition Analysis (`_03d_transition_analysis.py`)

Quantifies land cover transitions between consecutive years and between consolidated early/late products.

- **Input**:
  - Yearly classification assets from Step 3a
  - Consolidated assets from Step 3b (early-weighted, late-weighted)
- **Approach**:
  - Computes pixel-level transition matrices for each year pair (2019→2020, ..., 2023→2024)
  - Aggregates 12 detailed classes into 6 major types (Irrigated, Rainfed, Urban, Water, Native Veg., Bare)
  - Optionally weights transitions by pixel confidence from consolidation
- **Output**:
  - Transition edge tables (per-pair and all-pairs combined)
  - Transition matrices (detailed 12×12 and aggregated 6×6)
  - Left-to-right alluvial flow plots (detailed and major-class)
  - Per-class focus alluvial plots with km² labels
  - Consolidated early→late transition with confidence weighting

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
│   ├── class_areas_by_year.csv      # Yearly area statistics
│   ├── consolidated_class_areas.csv # Consolidated area statistics
│   ├── area_statistics_summary.json # Metadata and summary
│   ├── *.png                        # Area charts (yearly + consolidated)
│   └── transitions/
│       ├── transition_edges_*.csv   # Per-pair and aggregated edge tables
│       ├── transition_matrix_*.csv  # Transition matrices (raw + weighted)
│       ├── matrices/                # Per-pair 12×12 matrices
│       ├── matrices_major/          # Per-pair 6×6 major-class matrices
│       └── alluvial_*.png           # Alluvial flow plots
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

## Reference Method Note (Owusu et al., 2024)

This project's external-reference comparisons (HCCM and Budyko-style partitioning context) follow the framework described in:

- Owusu, A. et al. (2024), *A framework for disaggregating remote-sensing cropland into rainfed and irrigated classes at continental scale*.

Key points used as methodological reference:

- **High-confidence cropland mask construction (HCCM concept):** derive a confidence cropland mask from agreement among top-performing crop masks; Owusu et al. used a **66% agreement threshold** among the selected masks.
- **Rainfed vs irrigated decision rule:** apply Budyko partitioning to estimate green ET, then compute blue ET as:
  - `ET_blue = ET_actual - ET_green`
- **Irrigation identification:** cropland pixels are considered irrigated when blue ET exceeds the local threshold:
  - `max(mean blue ET over grassland in the sub-catchment, 0.01 mm)`
- **Irrigation intensity split:**
  - `ET_blue >= 100 mm` → formal irrigation
  - `ET_blue < 100 mm` → supplemental irrigation

In this repository, these concepts are used as literature reference for interpretation/benchmarking layers (e.g., Budkyo/HCCM comparisons in area statistics), while the core production workflow remains the project-specific clustering + embeddings pipeline.

---

## Key Files

| File | Purpose |
|------|---------|
| `__config__.py` | Centralized configuration |
| `src/gee_utils.py` | GEE authentication, data loading, S2 processing |
| `src/clustering_utils.py` | DTW computation, gap-filling, clustering |
| `src/classification_utils.py` | Classifier training, export, area stats |
| `_03d_transition_analysis.py` | Land cover transition flows and alluvials |

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

## Analysis and Results

This section presents the full analysis pipeline results for the Kafue Flats study area (~100,000 km², 25.3–28.8°E / 14.7–17.1°S) covering 2019–2024, including training data characteristics, clustering outcomes, classification accuracy, irrigated area estimates, multi-year consolidation, and land cover transition analysis.

### Training Data

#### Crop Samples

A total of **4,000 crop sample points** were collected from pixels consistently classified as cropland across all six years (2019–2024) by Dynamic World. For each sample, full Sentinel-2 time series were extracted, yielding four spectral indices (NDVI, NDWI, NDMI, and a composite index) at each observation date. This produced **24,000 sample-year records** (4,000 samples × 6 years), each with a minimum of 20 cloud-free observations per year (cloud score threshold: 60%).

The composite index — defined as $\frac{NDVI - NDWI + NDMI}{3}$ — combines vegetation vigor, surface water, and canopy moisture into a single signal that amplifies the dry-season separation between irrigated and rainfed fields.

#### Landcover Samples

Seven non-crop landcover classes were sampled from Dynamic World yearly composites, targeting ~2,000 samples per class per year. After cleaning (removing grass samples within 100 m of crop pixels and filtering for multi-year temporal consistency across ≥4 years), **47,632 landcover samples** were retained:

| Class | Samples |
|-------|---------|
| Water | 10,695 |
| Shrubs | 10,123 |
| Trees | 9,694 |
| Urban | 9,551 |
| Flooded Veg. | 3,685 |
| Grass | 2,036 |
| Bare | 1,848 |

The imbalance (fewer grass and bare samples) reflects the natural prevalence of these classes in the study area and the stringency of the consistency filter. Class balancing is applied during classifier training (see [Classification](#classification-with-satellite-embeddings)).

### DTW Clustering

Dynamic Time Warping (DTW) hierarchical clustering was applied to the dry-season (May–September) composite index time series to discover irrigation patterns without manual labeling. DTW measures shape similarity between time series while tolerating temporal shifts — critical for phenological analysis where the timing of green-up or senescence varies between years and fields.

Clustering is performed independently for each sample-year (i.e., the same field may be assigned to different clusters in different years), which captures temporal variability in irrigation practices — for example, a field that is irrigated in some years and left rainfed in others.

**Parameters:**
- Sakoe-Chiba window: 15 days
- Linkage: Ward (minimizes within-cluster variance)
- Number of clusters: $k = 5$
- Season: Dry season only (May–September ± DTW window buffer)

**Cluster Distribution** (across all 24,000 sample-year records):

| Cluster | Assignment | Sample-Years |
|---------|------------|--------------|
| 0 | Rainfed | 11,340 |
| 1 | Rainfed | 9,722 |
| 2 | Irrigated | 894 |
| 3 | Irrigated | 1,210 |
| 4 | Irrigated | 834 |

Rainfed clusters (0–1) are characterized by a steep decline in the composite index during the dry season as vegetation senesces following the cessation of wet-season rains. Irrigated clusters (2–4) maintain elevated composite index values throughout the dry season, reflecting sustained water availability from irrigation infrastructure. Clusters 2–4 differ in the magnitude and timing of their dry-season greenness, likely corresponding to different irrigation methods (pivot, drip, flood) or crop types.

The strong numerical imbalance (rainfed ≫ irrigated) is consistent with the known dominance of rainfed agriculture in the Kafue Flats region. This imbalance is addressed through hybrid class balancing during classifier training.

### Classification with Satellite Embeddings

Classification uses the **AlphaEarth Foundations (AEF)** satellite embedding dataset (Brown et al., 2025), which provides a 64-dimensional learned representation for every 10 m pixel globally. These embeddings are produced by a 480-million-parameter foundation model trained on over 3 billion observations from nine satellite sources (Sentinel-2, Sentinel-1, Landsat, DEM, GEDI, ERA5, and others) using self-supervised objectives including reconstruction, contrastive learning, uniformity regularization, and text-image alignment via Gemini (Brown et al., 2025). AEF achieved a **23.9% average error reduction** over next-best methods across 15 benchmark evaluations spanning classification, regression, and change detection tasks.

The embedding approach offers several advantages over hand-crafted spectral features:

1. **Spatial context**: Embeddings encode field boundaries, texture, and neighborhood patterns that individual band values cannot capture.
2. **Multi-source fusion**: Information from optical, SAR, elevation, and climate data is distilled into a single compact vector (64 bytes/pixel).
3. **Transfer learning**: Pre-trained representations generalize across regions and tasks without task-specific feature engineering.
4. **Efficiency**: Annual composites at 10 m resolution are pre-computed and served via Google Earth Engine (`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`).

#### Year-Specific Classifiers

A separate **Gradient Boosted Tree** classifier is trained for each year (2019–2024) to avoid temporal data leakage. Each classifier is trained on that year's embedding features using labels from DTW clustering (crop classes 0–4) and cleaned Dynamic World landcover samples (classes 5–11), producing a **12-class** unified scheme.

**Classifier Configuration:**

| Parameter | Value |
|-----------|-------|
| Algorithm | Gradient Tree Boosting |
| Number of trees | 350 |
| Learning rate (shrinkage) | 0.08 |
| Max leaf nodes | 20 |
| Sampling rate | 60% |
| Class balancing | Hybrid (floor: 250, cap: 2,000) |
| Train/test split | Stratified balanced (~80/20) |

Total training data per year is approximately **11,500 samples** (e.g., 9,159 train + 2,290 test for 2023), combining crop cluster labels and landcover samples. All classification and area mapping is performed at the native 10 m resolution of both the Sentinel-2 source data and the AEF embeddings.

### Classification Accuracy

#### Overall Accuracy by Year

| Year | Test Accuracy | Test Samples |
|------|--------------|--------------|
| 2019 | 86.1% | 2,338 |
| 2020 | 83.2% | 2,392 |
| 2021 | 81.4% | 2,370 |
| 2022 | 84.4% | 2,417 |
| 2023 | 82.6% | 2,290 |
| 2024 | 84.6% | 2,294 |
| **Mean** | **83.7%** | — |

Accuracy remains relatively stable across all six years (81.4–86.1%), with lower performance concentrated in rainfed/natural vegetation boundary classes.

#### Per-Class Accuracy (2023, representative year)

| Class | Label | Correct | Total | Accuracy |
|-------|-------|---------|-------|----------|
| 0 | Rainfed (0) | 342 | 400 | 85.5% |
| 1 | Rainfed (1) | 92 | 179 | 51.4% |
| 2 | Irrigated (2) | 49 | 50 | 98.0% |
| 3 | Irrigated (3) | 48 | 50 | 96.0% |
| 4 | Irrigated (4) | 47 | 50 | 94.0% |
| 5 | Urban | 254 | 295 | 86.1% |
| 6 | Water | 339 | 354 | 95.8% |
| 7 | Flooded Veg. | 91 | 113 | 80.5% |
| 8 | Trees | 265 | 303 | 87.5% |
| 9 | Shrubs | 267 | 356 | 75.0% |
| 10 | Grass | 47 | 72 | 65.3% |
| 11 | Bare | 50 | 68 | 73.5% |

**Key observations:**
- **Irrigated classes remain strong** (94–98%), indicating that embeddings still separate irrigated phenology reliably.
- **Rainfed performance is mixed**: Rainfed (0) remains strong (85.5%), while Rainfed (1) is substantially lower (51.4%), indicating overlap with neighboring vegetation/cropland signatures.
- **Water is highly separable** (95.8%), while shrub/grass classes remain the most challenging.
- **Most confusion concentrates among spectrally similar non-target natural classes** (Shrubs, Grass, Flooded Veg., Trees).

#### Aggregated Accuracy by Type

When detailed classes are aggregated into broader categories, irrigation classification performance is excellent:

| Type | Precision | Recall | Correct | Actual | Predicted |
|------|-----------|--------|---------|--------|-----------|
| **Irrigated** | **97.9%** | **95.3%** | 143 | 150 | 146 |
| **Rainfed** | **86.5%** | **91.6%** | 533 | 582 | 616 |
| Urban | 87.9% | 83.5% | 218 | 261 | 248 |
| Water | 97.1% | 96.6% | 372 | 385 | 383 |
| Native Veg. | 91.6% | 90.5% | 784 | 866 | 856 |
| Bare | 100.0% | 90.0% | 45 | 50 | 45 |

The irrigated type retains high performance (**97.9% precision**, **95.3% recall**), while rainfed performance is lower than previous runs (**86.5% precision**, **91.6% recall**) due to stronger confusion with natural vegetation classes in the latest outputs. Even with this reduction, irrigated recall remains substantially higher than the Budyko benchmark of Owusu et al. (2024) (59% recall at 300 m resolution).

### Irrigated Area Estimates

The following table summarizes estimated irrigated and rainfed crop areas across the study period (computed at 10 m resolution):

| Year | Irrigated (km²) | Rainfed (km²) | Urban (km²) | Water (km²) | Native Veg. (km²) | Bare (km²) |
|------|-----------------|---------------|-------------|-------------|-------------------|------------|
| 2019 | 442.4 | 13,200.2 | 4,063.5 | 3,929.9 | 78,134.1 | 300.9 |
| 2020 | 443.4 | 14,490.3 | 3,782.8 | 3,921.7 | 77,095.4 | 337.5 |
| 2021 | 525.1 | 13,552.4 | 3,701.0 | 4,033.4 | 77,955.2 | 303.9 |
| 2022 | 503.6 | 14,577.7 | 3,266.3 | 3,967.9 | 77,210.4 | 545.1 |
| 2023 | 523.9 | 13,317.6 | 3,288.7 | 3,922.5 | 78,631.6 | 386.8 |
| 2024 | 495.7 | 16,203.6 | 3,000.2 | 3,739.7 | 76,491.6 | 140.1 |

**Key findings:**

- **Irrigated area ranges from 442 to 525 km²** across the six-year period, with a six-year mean of approximately **489 km²**.
- **Peak irrigated extent occurs in 2021 and 2023** (525 and 524 km²), with a moderate decline in 2024 (496 km²).
- **Irrigated area increases from 2019 to 2024** (442 → 496 km²; +53 km²), while still showing interannual variability likely linked to climate and management effects.
- **Rainfed cropland** (13,200–16,204 km²) is roughly **25–37× larger** than irrigated area, confirming rainfed dominance. Irrigated cropland remains about **3.0–3.8%** of total cropland.
- **Urban and bare areas decline toward 2024**, while water remains relatively stable and native vegetation remains the dominant class by area.

### Precipitation Context

Wet-season precipitation from CHIRPS provides context for interpreting interannual variability in irrigated area:

| Water Year | Wet Season Precip. (mm) |
|------------|------------------------|
| 2017 | 890.6 |
| 2018 | 809.2 |
| 2019 | 424.5 |
| 2020 | 776.3 |
| 2021 | 788.0 |
| 2022 | 692.0 |
| 2023 | 747.4 |
| 2024 | 527.2 |

The **2019 wet season was severely dry** (424.5 mm, ~60% of the 8-year mean of 707 mm), followed by recovery in 2020–2021. The 2022 season was again below average (692 mm), while 2024 saw another significant decline (527.2 mm). Irrigated area remains comparatively high in 2021–2024 (496–525 km²), suggesting that rainfall alone does not explain annual irrigated extent. The relationship appears multi-factor: precipitation, carry-over soil moisture, operational irrigation capacity, and classification uncertainty all likely contribute.

### Multi-Year Consolidation

Year-to-year classification variability arises from weather-dependent crop responses, sensor noise, and residual cloud effects. To produce stable, high-confidence maps, multi-year consolidation combines the six annual classifications using weighted confidence scoring:

1. **Class aggregation**: The five crop clusters (0–4) are collapsed into two classes: Rainfed (0) and Irrigated (1). Landcover classes (5–11) are remapped to classes 2–8, yielding a final **9-class scheme**.

2. **Confidence scoring**: Each pixel receives a confidence score based on:
   - **Temporal consistency** (80% weight): frequency of a given classification across years, weighted by temporal weighting scheme
   - **Spatial consistency** (20% weight): agreement with neighboring pixels in a 3×3 window

3. **Temporal weighting variants**:
   - **Early-weighted**: prioritizes 2019 classifications (useful for baseline mapping)
   - **Late-weighted**: prioritizes 2024 classifications (useful for current-state mapping)
   - **Combined (equal-weight)**: all years weighted equally (most robust consensus)

Each variant produces a classification band and a confidence band, exported as GEE assets for downstream analysis. In total, the pipeline exports **9 GEE assets**: 6 yearly classifications (`classified_2019` through `classified_2024`) and 3 consolidated products (`consolidated_2024_weighted`, `consolidated_2019_weighted`, `consolidated_all_years_classified`). Final area statistics are computed from both yearly and consolidated assets at 10 m resolution.

#### Consolidated Area Estimates

| Variant | Irrigated (km²) | Rainfed (km²) | Urban (km²) | Water (km²) | Native Veg. (km²) | Bare (km²) |
|---------|-----------------|---------------|-------------|-------------|-------------------|------------|
| All-Years (equal) | 459.7 | 15,117.4 | 3,133.8 | 3,970.3 | 77,247.6 | 142.3 |
| Early (2019) | 442.4 | 13,200.2 | 4,063.5 | 3,929.9 | 78,134.1 | 300.9 |
| Late (2024) | 495.7 | 16,203.6 | 3,000.2 | 3,739.7 | 76,491.6 | 140.1 |

The **all-years consolidated** irrigated area (459.7 km²) is lower than the six-year yearly mean (489.0 km²), reflecting the conservative nature of consensus assignment. The **early** and **late** variants indicate irrigated expansion from 442.4 to 495.7 km² (+53.3 km², +12.0%).

### Transition Analysis

Transition analysis quantifies how land cover classes change between years, revealing which conversions are dominant and which classes are most stable. The analysis is performed by `_03d_transition_analysis.py` (see [Step 3d](#step-3d-transition-analysis-_03d_transition_analysispy) for methodology).

#### Consolidated Early → Late Transitions

The most policy-relevant transition analysis compares the early-weighted (2019) and late-weighted (2024) consolidations, which represent stable estimates of land cover at the start and end of the study period. Transitions are optionally weighted by pixel confidence (pixels with confidence < 10% are excluded):

| Transition | Raw (km²) | Weighted (km²) | Retention (%) |
|------------|-----------|----------------|---------------|
| Irrigated → Irrigated | 335.3 | 305.2 | 91.0 |
| Irrigated → Rainfed | 93.6 | 55.2 | — |
| Rainfed → Irrigated | 101.4 | 62.6 | — |
| Rainfed → Rainfed | 10,332.7 | 9,106.9 | 88.1 |
| Trees → Trees | 27,088.0 | 25,493.1 | 94.1 |
| Shrubs → Shrubs | 32,546.9 | 28,956.0 | 89.0 |
| Water → Water | 3,690.3 | 3,674.4 | 99.6 |
| Urban → Urban | 1,861.1 | 1,541.9 | 82.8 |

**Key transition findings:**

- **Irrigated areas show 91.0% weighted retention** between early and late periods, indicating strong persistence of irrigated land.
- **Net irrigated change is +53.3 km²**, from early (442.4 km²) to late (495.7 km²).
- **Water is the most stable class** (99.6% retention), while **urban areas show the most churn** (82.8% retention), consistent with edge-class variability.
- **The largest inter-class flows** occur between Rainfed ↔ Shrubs and Trees ↔ Shrubs, consistent with spectral overlap among cropland and natural vegetation classes.
- **Rainfed → Irrigated** (62.6 km² weighted) now exceeds **Irrigated → Rainfed** (55.2 km² weighted), matching the observed net irrigated increase.

### Comparison with Benchmark Methods

The table below compares this work with the Budyko water balance framework of Owusu et al. (2024), a continental-scale approach for disaggregating cropland into irrigated and rainfed classes:

| Dimension | Owusu et al. — Budyko (2024) | This Work — DTW + Embeddings |
|-----------|------------------------------|------------------------------|
| Core method | Physics-based water balance | Data-driven: time series clustering + ML |
| Spatial resolution | 300 m (limited by climate inputs) | **10 m** (Sentinel-2 native) |
| Spatial scale | Continental (all Africa) | Regional (Kafue Flats, Zambia) |
| Temporal granularity | Annual aggregates | **Sub-annual** time series |
| Training data | None needed (physics-based) | Auto-generated via DTW clustering |
| Multi-year coverage | Single year (2019) | **6 years** (2019–2024) with consolidation |
| Feature representation | 3 climate variables | **64-D learned embeddings** (Brown et al., 2025) |
| Overall accuracy | 73% (Cohen's κ = 0.48) | **83.7%** (mean across years) |
| Irrigated precision | 92% | **97.9%** |
| Irrigated recall | 59% | **95.3%** |

The two approaches are complementary: Budyko excels at continental-scale screening with zero training data, while DTW + embeddings provides field-level precision for targeted regional analysis. The approaches have different failure modes — Budyko fails when ET estimates are inaccurate, while DTW/embeddings may miss supplemental irrigation where irrigated and rainfed temporal profiles are similar.

### Relevance to Sustainable Development

This analysis contributes to monitoring several UN Sustainable Development Goals:

- **SDG 2.4** (Sustainable Agriculture): Quantifies the extent and temporal dynamics of irrigated vs. rainfed cropland, informing food security assessments and agricultural development planning.
- **SDG 6.4** (Water-Use Efficiency): Separates irrigated from rainfed agriculture at 10 m resolution, enabling estimation of agricultural water consumption by land-use type.
- **SDG 13.1** (Climate Resilience): Tracks expansion and contraction of irrigated area in response to precipitation variability, providing indicators of climate adaptation capacity.

As noted in the EO Compendium for SDGs (CEOS/GEO EO4SDG, 2020), Earth observation contributes to 34 indicators across 11 SDGs. The combination of Copernicus Sentinel-2 data, foundation model embeddings (Brown et al., 2025), and cloud-based processing (Google Earth Engine) demonstrates how open satellite data can support SDG monitoring in data-poor regions of sub-Saharan Africa, where FAO estimates of irrigated area (101,000 km² across Africa) may undercount actual extent by a factor of 4.5× (Owusu et al., 2024).

---

## References

- **Brown, C. F. et al. (2025)**. *AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data*. Google DeepMind / Google Research. — Foundation model providing the 64-D satellite embeddings used for classification. See `docs/references/brown_et_al_alphaearth-foundatio_2025.pdf`.
- **Owusu, A. et al. (2024)**. *A framework for disaggregating remote-sensing cropland into rainfed and irrigated classes at continental scale*. International Journal of Applied Earth Observation and Geoinformation, Elsevier. — Benchmark comparison using Budyko water balance. See `docs/references/owusu_et_al_a-framework-for-disa_2024.pdf`.
- **CEOS/GEO EO4SDG (2020)**. *Compendium of Earth Observation contributions to the SDG Targets and Indicators*. ESA, UNEP-WCMC. — Framework linking Earth observations to SDG indicators. See `docs/references/EO_Compendium-for-SDGs 2.pdf`.
- **Dynamic World**: Brown et al. (2022) - Near real-time land use classification
- **DTW**: Sakoe & Chiba (1978) - Dynamic programming algorithm optimization for spoken word recognition

---

**Need help?** See `__config__.py` for detailed parameter documentation.
