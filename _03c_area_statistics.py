# %% [markdown]
# # Step 3c: Area Statistics (Yearly + Consolidated)
#
# Calculates and visualizes area statistics from:
# 1) Yearly classification assets exported by `_03a_classification.py`
# 2) Consolidated assets exported by `_03b_consolidation.py`
#
# Run this script AFTER `_03b_consolidation.py`.

# %%
import sys
import json
from pathlib import Path

import ee
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
import __config__

from src.gee_utils import initialize_earth_engine, asset_exists, _retry_with_backoff
from src.classification_utils import (
    calculate_areas_from_assets,
    export_area_stats_to_asset,
    wait_for_exports,
)

# Configuration
AREA_CONFIG = __config__.AREA_STATISTICS
EMB_CONFIG = __config__.EMBEDDINGS_CLASSIFICATION
CONSOL_CONFIG = __config__.CONSOLIDATION

# Set up plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

# Output directory
OUTPUT_DIR = Path('output') / __config__.CURRENT_STUDY_AREA / AREA_CONFIG.get('output_subfolder', 'consolidation')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# %%
# Initialize Earth Engine
print("Initializing Earth Engine...")
initialize_earth_engine()

study_area_config = __config__.get_study_area()
study_area = study_area_config['geometry']
study_area_name = study_area_config['name']

print(f"Study area: {study_area_name}")
print(f"Output directory: {OUTPUT_DIR}")


# %%
# ===== Yearly class configuration =====
yearly_class_types = EMB_CONFIG.get('class_types', {})
yearly_cluster_labels = EMB_CONFIG.get('cluster_labels', {})
yearly_cluster_palette = EMB_CONFIG.get('cluster_palette', {})

crop_classes = list(EMB_CONFIG.get('crop_cluster_assignments', {}).keys())
landcover_classes = list(EMB_CONFIG.get('landcover_class_mapping', {}).values())
unique_classes = sorted(set(crop_classes + landcover_classes))

yearly_class_mapping = {
    'unique_classes': unique_classes,
    'to_idx': {c: i for i, c in enumerate(unique_classes)},
    'to_class': {i: c for i, c in enumerate(unique_classes)},
}

yearly_idx_labels = {
    yearly_class_mapping['to_idx'][c]: yearly_cluster_labels.get(c, f'Class {c}')
    for c in unique_classes if c in yearly_class_mapping['to_idx']
}
yearly_palette = [yearly_cluster_palette.get(c, '#CCCCCC') for c in unique_classes]

print("\nYearly class configuration:")
print(f"  Classes: {len(unique_classes)}")
print(f"  Original IDs: {unique_classes}")


# %%
# ===== Consolidated class configuration =====
aggregated_labels = CONSOL_CONFIG.get('aggregated_labels', {})
aggregated_palette = CONSOL_CONFIG.get('aggregated_palette', {})

consolidated_type_map = AREA_CONFIG.get('class_types', {
    'Irrigated': [1],
    'Rainfed': [0],
    'Urban': [2],
    'Water': [3],
    'Native Veg.': [4, 5, 6, 7],
    'Bare': [8],
})

consolidated_class_to_type = {}
for tname, cids in consolidated_type_map.items():
    for cid in cids:
        consolidated_class_to_type[cid] = tname


# %%
# ===== Yearly area calculation settings =====
yearly_years = AREA_CONFIG.get('years', EMB_CONFIG.get('classification_years', []))
yearly_asset_folder = EMB_CONFIG.get('asset_folder', f"{__config__.CURRENT_STUDY_AREA}/classification")
yearly_asset_prefix = 'classified'
area_scale = AREA_CONFIG.get('scale', 100)

print("\nYearly area calculation settings:")
print(f"  Years: {yearly_years}")
print(f"  Asset folder: projects/{__config__.PROJECT_ID}/assets/{yearly_asset_folder}/")
print(f"  Asset prefix: {yearly_asset_prefix}")
print(f"  Scale: {area_scale}m")


# %%
# Check yearly assets
print("\nChecking yearly classification assets...")
missing_years = []
for year in yearly_years:
    asset_id = f"projects/{__config__.PROJECT_ID}/assets/{yearly_asset_folder}/{yearly_asset_prefix}_{year}"
    if asset_exists(asset_id):
        print(f"  {year}: ✓ Found")
    else:
        print(f"  {year}: ✗ Missing")
        missing_years.append(year)

if missing_years:
    yearly_years = [y for y in yearly_years if y not in missing_years]
    print(f"\n⚠️ Missing yearly assets skipped: {missing_years}")


# %%
# Calculate yearly areas
yearly_areas = {}
if yearly_years:
    print("\nCalculating yearly area statistics...")
    yearly_areas = calculate_areas_from_assets(
        yearly_years,
        study_area,
        scale=area_scale,
        asset_folder=yearly_asset_folder,
        asset_prefix=yearly_asset_prefix,
    )
else:
    print("\n⚠️ No yearly assets available for area statistics.")


# %%
# ===== Consolidated area settings =====
consolidated_assets = AREA_CONFIG.get('consolidated_assets', {
    'early': {
        'asset_name': CONSOL_CONFIG.get('early_asset_name', 'consolidated_2024_weighted'),
        'label': '2024-weighted',
        'enabled': CONSOL_CONFIG.get('export_early_weighted', True),
    },
    'late': {
        'asset_name': CONSOL_CONFIG.get('late_asset_name', 'consolidated_2019_weighted'),
        'label': '2019-weighted',
        'enabled': CONSOL_CONFIG.get('export_late_weighted', True),
    },
    'combined': {
        'asset_name': CONSOL_CONFIG.get('combined_asset_name', 'consolidated_all_years_classified'),
        'label': 'All-years combined',
        'enabled': CONSOL_CONFIG.get('export_combined_classified', True),
    },
})

consolidated_asset_folder = AREA_CONFIG.get('asset_folder', CONSOL_CONFIG.get('asset_folder', yearly_asset_folder))

selected_consolidated = {
    k: v for k, v in consolidated_assets.items() if v.get('enabled', True)
}

print("\nConsolidated area settings:")
print(f"  Variants: {list(selected_consolidated.keys())}")
print(f"  Asset folder: projects/{__config__.PROJECT_ID}/assets/{consolidated_asset_folder}/")


def get_class_band_name(image):
    band_names = _retry_with_backoff(lambda: image.bandNames().getInfo())
    if 'classification' in band_names:
        return 'classification'
    if not band_names:
        raise ValueError("Image has no bands")
    return band_names[0]


def calculate_areas_from_image(image, region, scale, class_band='classification'):
    pixel_area = ee.Image.pixelArea().divide(1e6)
    class_img = image.select(class_band).rename('cluster')
    stacked = pixel_area.addBands(class_img)

    result = _retry_with_backoff(
        lambda: stacked.reduceRegion(
            reducer=ee.Reducer.sum().group(groupField=1, groupName='cluster'),
            geometry=region,
            scale=scale,
            maxPixels=1e10,
            bestEffort=True,
        ).getInfo()
    )

    area_by_class = {}
    for group in result.get('groups', []):
        cid = int(group['cluster'])
        area_by_class[cid] = group['sum']
    return area_by_class


# %%
# Calculate consolidated areas
consolidated_areas = {}
consolidated_images_for_export = {}
consolidated_meta = {}

print("\nChecking consolidated assets...")
for variant_key, info in selected_consolidated.items():
    asset_name = info['asset_name']
    asset_id = f"projects/{__config__.PROJECT_ID}/assets/{consolidated_asset_folder}/{asset_name}"

    if not asset_exists(asset_id):
        print(f"  {variant_key}: ✗ Missing")
        continue

    print(f"  {variant_key}: ✓ Found")
    img = ee.Image(asset_id)
    class_band = get_class_band_name(img)
    areas = calculate_areas_from_image(img, study_area, area_scale, class_band=class_band)

    consolidated_areas[variant_key] = areas
    consolidated_images_for_export[variant_key] = img.select(class_band)
    consolidated_meta[variant_key] = {
        'asset_id': asset_id,
        'asset_name': asset_name,
        'label': info.get('label', variant_key),
        'class_band': class_band,
        'total_area_km2': sum(areas.values()),
    }

if not consolidated_areas:
    print("\n⚠️ No consolidated assets available for consolidated area stats.")


# %%
# Optional exports as table assets
if AREA_CONFIG.get('export_assets', False):
    print("\nExporting area statistics tables to GEE assets...")
    all_export_tasks = []

    # Yearly table exports
    if yearly_years:
        yearly_classified_images = {
            year: ee.Image(f"projects/{__config__.PROJECT_ID}/assets/{yearly_asset_folder}/{yearly_asset_prefix}_{year}").select(0)
            for year in yearly_years
        }
        yearly_tasks = export_area_stats_to_asset(
            yearly_classified_images,
            study_area,
            scale=area_scale,
            asset_folder=yearly_asset_folder,
            asset_prefix='area_stats_yearly',
            overwrite=AREA_CONFIG.get('overwrite_assets', False),
        )
        all_export_tasks.extend(yearly_tasks)

    # Consolidated table exports
    if consolidated_images_for_export:
        consolidated_tasks = export_area_stats_to_asset(
            consolidated_images_for_export,
            study_area,
            scale=area_scale,
            asset_folder=consolidated_asset_folder,
            asset_prefix=AREA_CONFIG.get('asset_prefix_stats', 'area_stats_consolidated'),
            overwrite=AREA_CONFIG.get('overwrite_assets', False),
        )
        all_export_tasks.extend(consolidated_tasks)

    if AREA_CONFIG.get('wait_for_exports', True) and all_export_tasks:
        print(f"Waiting for {len(all_export_tasks)} export(s) to complete...")
        wait_for_exports(all_export_tasks, check_interval=30)


# %%
# Build yearly dataframe
yearly_records = []
for year, areas in yearly_areas.items():
    total = sum(areas.values()) if areas else 0
    for cid, area_km2 in areas.items():
        orig_class_id = yearly_class_mapping['to_class'].get(cid, cid)
        label = yearly_idx_labels.get(cid, f'Class {orig_class_id}')

        class_type_name = None
        for tname, class_ids in yearly_class_types.items():
            if orig_class_id in class_ids:
                class_type_name = tname
                break

        yearly_records.append({
            'year': int(year),
            'class_seq_id': int(cid),
            'class_orig_id': int(orig_class_id),
            'class_label': label,
            'class_type': class_type_name or 'Other',
            'area_km2': float(area_km2),
            'area_pct': (float(area_km2) / total * 100) if total > 0 else 0.0,
        })

df_yearly = pd.DataFrame(yearly_records)

# Build consolidated dataframe
consolidated_records = []
for variant, areas in consolidated_areas.items():
    total = sum(areas.values()) if areas else 0
    for cid, area_km2 in areas.items():
        consolidated_records.append({
            'variant_key': variant,
            'variant_label': consolidated_meta[variant]['label'],
            'class_id': int(cid),
            'class_label': aggregated_labels.get(cid, f'Class {cid}'),
            'class_type': consolidated_class_to_type.get(cid, 'Other'),
            'area_km2': float(area_km2),
            'area_pct': (float(area_km2) / total * 100) if total > 0 else 0.0,
        })

df_consolidated = pd.DataFrame(consolidated_records)


# %%
# Print summary
print("\n" + "="*70)
print("AREA STATISTICS SUMMARY")
print("="*70)

if not df_yearly.empty:
    print("\nYearly classified assets:")
    for year in sorted(df_yearly['year'].unique()):
        year_df = df_yearly[df_yearly['year'] == year]
        print(f"  {year}: {year_df['area_km2'].sum():,.2f} km²")

if not df_consolidated.empty:
    print("\nConsolidated assets:")
    for variant in df_consolidated['variant_key'].unique():
        vdf = df_consolidated[df_consolidated['variant_key'] == variant]
        print(f"  {consolidated_meta[variant]['label']}: {vdf['area_km2'].sum():,.2f} km²")

print("\n" + "="*70)


# %%
# ===== Improved plotting =====
# A) Yearly: stacked area by aggregated type over years
if not df_yearly.empty and len(df_yearly['year'].unique()) >= 1:
    yearly_type = (
        df_yearly
        .groupby(['year', 'class_type'], as_index=False)['area_km2']
        .sum()
        .pivot(index='year', columns='class_type', values='area_km2')
        .fillna(0)
    )

    type_order = [t for t in ['Irrigated', 'Rainfed', 'Native Veg.', 'Water', 'Urban', 'Bare'] if t in yearly_type.columns] + [c for c in yearly_type.columns if c not in ['Irrigated', 'Rainfed', 'Native Veg.', 'Water', 'Urban', 'Bare']]
    yearly_type = yearly_type[type_order]

    type_palette = {
        'Irrigated': '#67179C',
        'Rainfed': '#f4912e',
        'Native Veg.': '#2b8c3e',
        'Water': '#0055be',
        'Urban': '#000000',
        'Bare': '#a59b8f',
        'Other': '#777777',
    }
    type_colors = [type_palette.get(t, '#777777') for t in yearly_type.columns]

    fig, ax = plt.subplots(figsize=(12, 6))
    yearly_type.plot(kind='bar', stacked=True, ax=ax, color=type_colors, edgecolor='white', linewidth=0.5)
    ax.set_title('Yearly Classified Areas by Aggregated Type (km²)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel('Area (km²)')
    ax.legend(title='Type', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax.grid(axis='y', alpha=0.25, linestyle='--')
    plt.tight_layout()

    yearly_plot_path = OUTPUT_DIR / 'yearly_area_by_type.png'
    plt.savefig(yearly_plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to {yearly_plot_path}")

# B) Consolidated variants comparison
if not df_consolidated.empty:
    variant_order = [v for v in ['early', 'late', 'combined'] if v in df_consolidated['variant_key'].unique()] + [
        v for v in df_consolidated['variant_key'].unique() if v not in ['early', 'late', 'combined']
    ]
    variant_labels = [consolidated_meta[v]['label'] for v in variant_order]

    class_order = sorted(df_consolidated['class_id'].unique())
    class_labels = [aggregated_labels.get(c, f'Class {c}') for c in class_order]
    class_colors = [aggregated_palette.get(c, '#CCCCCC') for c in class_order]

    pivot_class = (
        df_consolidated
        .pivot_table(index='variant_label', columns='class_label', values='area_km2', aggfunc='sum', fill_value=0)
        .reindex(index=variant_labels)
    )
    pivot_class = pivot_class[[lbl for lbl in class_labels if lbl in pivot_class.columns]]

    pivot_pct = (
        df_consolidated
        .pivot_table(index='class_label', columns='variant_label', values='area_pct', aggfunc='sum', fill_value=0)
        .reindex(index=class_labels)
        .reindex(columns=variant_labels)
    )

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.1])

    ax1 = fig.add_subplot(gs[0, 0])
    pivot_class.plot(kind='bar', stacked=True, ax=ax1, color=class_colors[:len(pivot_class.columns)], edgecolor='white', linewidth=0.5)
    ax1.set_title('Consolidated Areas by Class (km²)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Variant')
    ax1.set_ylabel('Area (km²)')
    ax1.legend(title='Class', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    ax1.grid(axis='y', alpha=0.25, linestyle='--')

    ax2 = fig.add_subplot(gs[1, 0])
    sns.heatmap(
        pivot_pct,
        ax=ax2,
        annot=True,
        fmt='.1f',
        cmap='YlGnBu',
        linewidths=0.4,
        cbar_kws={'label': 'Share of total area (%)'}
    )
    ax2.set_title('Class Share by Consolidation Variant (%)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Variant')
    ax2.set_ylabel('Class')

    plt.tight_layout()
    consolidated_plot_path = OUTPUT_DIR / 'consolidated_area_comparison.png'
    plt.savefig(consolidated_plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to {consolidated_plot_path}")


# %%
# Save outputs
if not df_yearly.empty:
    yearly_csv = OUTPUT_DIR / 'class_areas_by_year.csv'
    df_yearly.sort_values(['year', 'class_seq_id']).to_csv(yearly_csv, index=False)
    print(f"✓ Yearly area statistics saved to {yearly_csv}")

if not df_consolidated.empty:
    consolidated_csv = OUTPUT_DIR / 'consolidated_class_areas.csv'
    df_consolidated.sort_values(['variant_key', 'class_id']).to_csv(consolidated_csv, index=False)
    print(f"✓ Consolidated area statistics saved to {consolidated_csv}")

summary = {
    'project_id': __config__.PROJECT_ID,
    'study_area': study_area_name,
    'date_processed': pd.Timestamp.now().isoformat(),
    'scale_meters': area_scale,
    'yearly': {
        'asset_folder': yearly_asset_folder,
        'asset_prefix': yearly_asset_prefix,
        'years': sorted([int(y) for y in yearly_areas.keys()]),
        'class_labels': {str(k): v for k, v in yearly_idx_labels.items()},
        'class_types': yearly_class_types,
        'statistics_km2': {
            str(y): {str(int(cid)): float(area) for cid, area in yearly_areas.get(y, {}).items()}
            for y in sorted(yearly_areas.keys())
        },
    },
    'consolidated': {
        'asset_folder': consolidated_asset_folder,
        'variants': {
            k: {
                'label': consolidated_meta[k]['label'],
                'asset_id': consolidated_meta[k]['asset_id'],
                'class_band': consolidated_meta[k]['class_band'],
                'total_area_km2': consolidated_meta[k]['total_area_km2'],
            }
            for k in consolidated_meta
        },
        'aggregated_labels': {str(k): v for k, v in aggregated_labels.items()},
        'class_types': consolidated_type_map,
        'statistics_km2': {
            k: {str(int(cid)): float(area) for cid, area in consolidated_areas.get(k, {}).items()}
            for k in consolidated_areas
        },
    },
}

summary_path = OUTPUT_DIR / 'area_statistics_summary.json'
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"✓ Summary saved to {summary_path}")


# %%
# Final summary
print("\n" + "="*70)
print("AREA STATISTICS COMPLETE")
print("="*70)
print(f"\nStudy Area: {study_area_name}")
print(f"Scale: {area_scale}m")
print(f"\nOutputs saved to: {OUTPUT_DIR}")
if not df_yearly.empty:
    print("  - class_areas_by_year.csv")
    print("  - yearly_area_by_type.png")
if not df_consolidated.empty:
    print("  - consolidated_class_areas.csv")
    print("  - consolidated_area_comparison.png")
print("  - area_statistics_summary.json")
print("\n" + "="*70)

# %%
