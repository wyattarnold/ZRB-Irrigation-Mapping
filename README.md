# Zambezi Irrigation Mapping

This repository maps irrigated and rainfed agriculture in the Zambezi River Basin (Kafue Flats, Zambia) using Google Earth Engine.

## Irrigation Mapping Application

Use the published Earth Engine app to view the mapping outputs:

- https://warnold.users.earthengine.app/view/zrb-cropping

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
