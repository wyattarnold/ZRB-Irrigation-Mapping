# Zambezi Irrigation Mapping

This repository maps irrigated and rainfed agriculture in the Zambezi River Basin (Kafue Flats, Zambia) using Google Earth Engine.

## Installation

You must have a Google Earth Engine account to run the analysis (signup here: https://earthengine.google.com/signup/)

Then, install environment from the repository root:

```bash
conda env create -f environment.yml
conda activate gee
earthengine authenticate
```

## Irrigation Mapping Application

Use the published Earth Engine app to view the mapping outputs:

- https://ee-warnold.projects.earthengine.app/view/zrb-irrigation-mapping

## Detailed Documentation

For full pipeline details, configuration notes, outputs, and data sources, see:

- [documentation/README.md](documentation/README.md)

## Code Overview

The codebase is organized as a 3-step pipeline:

1. **Training data collection** (`_01a` to `_01d`) for landcover and crop samples
2. **Clustering** (`_02b`, `_02c`) using DTW to separate crop behavior patterns
3. **Classification and consolidation** (`_03a` to `_03c`) using satellite embeddings and multi-year aggregation
4. **Transition analysis** (`_03d`) for year-to-year class-change matrices and Sankey diagrams

Core shared utilities live in:

- `src/gee_utils.py` (GEE setup and imagery utilities)
- `src/clustering_utils.py` (DTW and clustering helpers)
- `src/classification_utils.py` (classification, exports, and area statistics)

Main project settings are centralized in:

- `__config__.py`
