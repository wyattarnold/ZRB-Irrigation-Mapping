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
OUTPUT_DIR = Path('output') / __config__.CURRENT_STUDY_AREA / AREA_CONFIG.get('output_subfolder', 'embeddings_classification')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PLOTS_ONLY_MODE = AREA_CONFIG.get('regenerate_plots_only', False)
YEARLY_CSV_PATH = OUTPUT_DIR / 'class_areas_by_year.csv'
CONSOLIDATED_CSV_PATH = OUTPUT_DIR / 'consolidated_class_areas.csv'
REFERENCE_CSV_PATH = OUTPUT_DIR / 'reference_mask_areas.csv'


# %%
# Initialize Earth Engine
if not PLOTS_ONLY_MODE:
    print("Initializing Earth Engine...")
    initialize_earth_engine()

    study_area_config = __config__.get_study_area()
    study_area = study_area_config['geometry']
    study_area_name = study_area_config['name']
else:
    study_area = None
    study_area_name = __config__.STUDY_AREAS[__config__.CURRENT_STUDY_AREA]['name']
    print("Plots-only mode enabled: reusing existing CSV statistics (no EE computation).")

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
if not PLOTS_ONLY_MODE:
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
if not PLOTS_ONLY_MODE and yearly_years:
    print("\nCalculating yearly area statistics...")
    yearly_areas = calculate_areas_from_assets(
        yearly_years,
        study_area,
        scale=area_scale,
        asset_folder=yearly_asset_folder,
        asset_prefix=yearly_asset_prefix,
    )
elif not PLOTS_ONLY_MODE:
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


def calculate_mask_area_km2(mask_image, region, scale):
    """Calculate total area (km²) where a binary mask is 1."""
    masked_area_img = ee.Image.pixelArea().divide(1e6).updateMask(mask_image)
    result = _retry_with_backoff(
        lambda: masked_area_img.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=region,
            scale=scale,
            maxPixels=1e10,
            bestEffort=True,
        ).getInfo()
    )
    if not result:
        return 0.0
    return float(next(iter(result.values()))) if result else 0.0


# %%
# Calculate consolidated areas
consolidated_areas = {}
consolidated_images_for_export = {}
consolidated_meta = {}

if not PLOTS_ONLY_MODE:
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
# Calculate reference mask areas (Budkyo + HCCP strata)
reference_rows = []
reference_cfg = AREA_CONFIG.get('reference_masks', {})

if PLOTS_ONLY_MODE:
    if REFERENCE_CSV_PATH.exists():
        df_reference = pd.read_csv(REFERENCE_CSV_PATH)
        print(f"✓ Loaded reference mask statistics from {REFERENCE_CSV_PATH}")
    else:
        df_reference = pd.DataFrame()
        print(f"⚠️ Missing reference-mask CSV: {REFERENCE_CSV_PATH}")
else:
    print("\nCalculating reference mask areas...")

    budkyo_rainfed_asset = reference_cfg.get('budkyo_rainfed_asset')
    if budkyo_rainfed_asset:
        if asset_exists(budkyo_rainfed_asset):
            rainfed_mask = ee.Image(budkyo_rainfed_asset).select('b1').eq(1)
            rainfed_area = calculate_mask_area_km2(rainfed_mask, study_area, area_scale)
            reference_rows.append({
                'source': 'Budkyo Rainfed',
                'metric': 'b1==1',
                'area_km2': rainfed_area,
                'asset_id': budkyo_rainfed_asset,
            })
            print(f"  ✓ Budkyo Rainfed (b1==1): {rainfed_area:,.2f} km²")
        else:
            print(f"  ⚠️ Missing Budkyo Rainfed asset: {budkyo_rainfed_asset}")

    budkyo_irrigated_asset = reference_cfg.get('budkyo_irrigated_asset')
    if budkyo_irrigated_asset:
        if asset_exists(budkyo_irrigated_asset):
            irrigated_mask = ee.Image(budkyo_irrigated_asset).select('b1').eq(1)
            irrigated_area = calculate_mask_area_km2(irrigated_mask, study_area, area_scale)
            reference_rows.append({
                'source': 'Budkyo Irrigated',
                'metric': 'b1==1',
                'area_km2': irrigated_area,
                'asset_id': budkyo_irrigated_asset,
            })
            print(f"  ✓ Budkyo Irrigated (b1==1): {irrigated_area:,.2f} km²")
        else:
            print(f"  ⚠️ Missing Budkyo Irrigated asset: {budkyo_irrigated_asset}")

    hccp_asset = reference_cfg.get('hccp_asset')
    hccp_bins = reference_cfg.get('hccp_confidence_bins', [])
    if hccp_asset and hccp_bins:
        if asset_exists(hccp_asset):
            hccp_b1 = ee.Image(hccp_asset).select('b1')
            for bin_cfg in hccp_bins:
                min_conf = int(bin_cfg['min'])
                max_conf = int(bin_cfg['max'])
                label = str(bin_cfg.get('label', f"{min_conf}-{max_conf}"))
                hccp_mask = hccp_b1.gte(min_conf).And(hccp_b1.lte(max_conf))
                hccp_area = calculate_mask_area_km2(hccp_mask, study_area, area_scale)
                reference_rows.append({
                    'source': 'HCCP',
                    'metric': f'b1 in [{min_conf},{max_conf}]',
                    'stratum': label,
                    'area_km2': hccp_area,
                    'asset_id': hccp_asset,
                })
                print(f"  ✓ HCCP {label}% confidence (b1 {min_conf}-{max_conf}): {hccp_area:,.2f} km²")
        else:
            print(f"  ⚠️ Missing HCCP asset: {hccp_asset}")

    df_reference = pd.DataFrame(reference_rows)


# %%
# Optional exports as table assets
if not PLOTS_ONLY_MODE and AREA_CONFIG.get('export_assets', False):
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
if PLOTS_ONLY_MODE:
    if YEARLY_CSV_PATH.exists():
        df_yearly = pd.read_csv(YEARLY_CSV_PATH)
        print(f"✓ Loaded yearly statistics from {YEARLY_CSV_PATH}")
    else:
        df_yearly = pd.DataFrame()
        print(f"⚠️ Missing yearly CSV: {YEARLY_CSV_PATH}")
else:
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
if PLOTS_ONLY_MODE:
    if CONSOLIDATED_CSV_PATH.exists():
        df_consolidated = pd.read_csv(CONSOLIDATED_CSV_PATH)
        print(f"✓ Loaded consolidated statistics from {CONSOLIDATED_CSV_PATH}")
        for row in df_consolidated[['variant_key', 'variant_label']].drop_duplicates().itertuples(index=False):
            consolidated_meta[row.variant_key] = {
                'label': row.variant_label,
                'asset_id': None,
                'class_band': None,
                'total_area_km2': float(df_consolidated[df_consolidated['variant_key'] == row.variant_key]['area_km2'].sum()),
            }
    else:
        df_consolidated = pd.DataFrame()
        print(f"⚠️ Missing consolidated CSV: {CONSOLIDATED_CSV_PATH}")
else:
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

if not df_reference.empty:
    print("\nReference masks:")
    for row in df_reference.itertuples(index=False):
        descriptor = f"{row.source} {row.metric}"
        if hasattr(row, 'stratum') and pd.notna(getattr(row, 'stratum', None)):
            descriptor = f"{descriptor} ({row.stratum})"
        print(f"  {descriptor}: {float(row.area_km2):,.2f} km²")


# %%
# ===== Improved plotting =====
PLOT_DPI = AREA_CONFIG.get('plot_dpi', 300)
PLOT_FIGSIZE = tuple(AREA_CONFIG.get('plot_figsize', [13.33, 7.5]))
generated_plot_paths = []


def _save_and_close(fig, paths):
    if isinstance(paths, (str, Path)):
        paths = [paths]
    for path in paths:
        fig.savefig(path, dpi=PLOT_DPI, bbox_inches='tight')
        generated_plot_paths.append(Path(path))
        print(f"✓ Plot saved to {path}")
    plt.close(fig)


def _build_type_palette(type_map, class_palette):
    palette = {}
    for type_name, class_ids in type_map.items():
        chosen = None
        for cid in class_ids:
            if cid in class_palette:
                chosen = class_palette[cid]
                break
        palette[type_name] = chosen or '#777777'
    palette['Other'] = '#777777'
    return palette


preferred_type_order = ['Irrigated', 'Rainfed', 'Native Veg.', 'Water', 'Urban', 'Bare', 'Other']
LEGEND_RECT = [0, 0, 0.8, 1]
TITLE_FONT_SIZE = AREA_CONFIG.get('plot_title_fontsize', 9)
AXIS_LABEL_FONT_SIZE = AREA_CONFIG.get('plot_axis_label_fontsize', 8)
TICK_LABEL_FONT_SIZE = AREA_CONFIG.get('plot_tick_label_fontsize', 7)
LEGEND_FONT_SIZE = AREA_CONFIG.get('plot_legend_fontsize', 7)
LEGEND_TITLE_FONT_SIZE = AREA_CONFIG.get('plot_legend_title_fontsize', 8)

# A) Yearly plots (high-level + detailed)
if not df_yearly.empty and len(df_yearly['year'].unique()) >= 1:
    yearly_type = (
        df_yearly
        .groupby(['year', 'class_type'], as_index=False)['area_km2']
        .sum()
        .pivot(index='year', columns='class_type', values='area_km2')
        .fillna(0)
    )

    yearly_type_order = [t for t in preferred_type_order if t in yearly_type.columns] + [
        t for t in yearly_type.columns if t not in preferred_type_order
    ]
    yearly_type = yearly_type[yearly_type_order]

    yearly_type_palette = _build_type_palette(yearly_class_types, yearly_cluster_palette)
    yearly_type_colors = [yearly_type_palette.get(t, '#777777') for t in yearly_type.columns]

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    yearly_type.plot(
        kind='bar',
        stacked=True,
        ax=ax,
        color=yearly_type_colors,
        edgecolor='white',
        linewidth=0.4,
    )
    ax.set_title('Yearly Classified Area by Aggregated Type', fontsize=TITLE_FONT_SIZE, fontweight='bold')
    ax.set_xlabel('Year', fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel('Area (km²)', fontsize=AXIS_LABEL_FONT_SIZE)
    ax.tick_params(axis='x', labelrotation=90, labelsize=TICK_LABEL_FONT_SIZE)
    ax.tick_params(axis='y', labelsize=TICK_LABEL_FONT_SIZE)
    ax.legend(title='Class Type', loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_TITLE_FONT_SIZE)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    fig.tight_layout(rect=LEGEND_RECT)
    _save_and_close(fig, OUTPUT_DIR / 'yearly_area_by_type_km2.png')

    yearly_class = (
        df_yearly
        .groupby(['year', 'class_label'], as_index=False)['area_km2']
        .sum()
        .pivot(index='year', columns='class_label', values='area_km2')
        .fillna(0)
    )
    yearly_class_order = [
        yearly_cluster_labels.get(cid, f'Class {cid}')
        for cid in unique_classes
        if yearly_cluster_labels.get(cid, f'Class {cid}') in yearly_class.columns
    ]
    yearly_class = yearly_class[[c for c in yearly_class_order if c in yearly_class.columns]]
    yearly_class_colors = [
        yearly_cluster_palette.get(cid, '#CCCCCC')
        for cid in unique_classes
        if yearly_cluster_labels.get(cid, f'Class {cid}') in yearly_class.columns
    ]

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    yearly_class.plot(
        kind='bar',
        stacked=True,
        ax=ax,
        color=yearly_class_colors,
        edgecolor='white',
        linewidth=0.3,
    )
    ax.set_title('Yearly Classified Area by Detailed Class', fontsize=TITLE_FONT_SIZE, fontweight='bold')
    ax.set_xlabel('Year', fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel('Area (km²)', fontsize=AXIS_LABEL_FONT_SIZE)
    ax.tick_params(axis='x', labelrotation=90, labelsize=TICK_LABEL_FONT_SIZE)
    ax.tick_params(axis='y', labelsize=TICK_LABEL_FONT_SIZE)
    ax.legend(title='Detailed Class', loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_TITLE_FONT_SIZE)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    fig.tight_layout(rect=LEGEND_RECT)
    _save_and_close(fig, OUTPUT_DIR / 'yearly_area_by_class_km2.png')

    # Multirow panel line plot: major categories only (independent y-axis)
    yearly_line = (
        df_yearly
        .groupby(['year', 'class_type'], as_index=False)['area_km2']
        .sum()
    )
    line_class_order = [
        class_type for class_type in preferred_type_order
        if class_type in yearly_line['class_type'].unique() and class_type != 'Other'
    ]

    if line_class_order:
        from matplotlib.colors import to_rgba

        n_panels = len(line_class_order)
        panel_height = AREA_CONFIG.get('line_panel_height', 0.55)
        panel_figsize = (3.5, max(3.0, n_panels * panel_height))
        fig, axes = plt.subplots(n_panels, 1, figsize=panel_figsize, sharex=True)
        if n_panels == 1:
            axes = [axes]

        fig.patch.set_facecolor('white')

        for idx, class_label in enumerate(line_class_order):
            ax = axes[idx]
            ax.set_facecolor('white')
            class_df = yearly_line[yearly_line['class_type'] == class_label].sort_values('year')
            class_color = yearly_type_palette.get(class_label, '#777777')
            fill_color = to_rgba(class_color, alpha=0.10)

            x_vals = class_df['year'].astype(int).tolist()
            y_vals = class_df['area_km2'].astype(float).tolist()

            ax.plot(
                x_vals,
                y_vals,
                color=class_color,
                marker='o',
                linewidth=1.4,
                markersize=3,
                zorder=3,
            )
            ax.fill_between(x_vals, y_vals, alpha=0.10, color=class_color, zorder=1)

            y_min = min(y_vals) if y_vals else 0
            y_max = max(y_vals) if y_vals else 0
            y_span = max(y_max - y_min, 1.0)
            y_pad = max(y_span * 0.25, y_max * 0.01, 1.0)

            # Label only first and last points
            if len(x_vals) >= 2:
                first_val, last_val = y_vals[0], y_vals[-1]
                label_offset = y_pad * 0.12

                ax.text(
                    x_vals[0], first_val + label_offset,
                    f"{first_val:,.0f}",
                    ha='center', va='bottom', fontsize=5,
                    color=class_color, fontweight='semibold',
                )
                ax.text(
                    x_vals[-1], last_val + label_offset,
                    f"{last_val:,.0f}",
                    ha='center', va='bottom', fontsize=5,
                    color=class_color, fontweight='semibold',
                )

                # Net change annotation on the right margin
                net_change = last_val - first_val
                pct_change = (net_change / first_val * 100) if first_val != 0 else 0
                sign = '+' if net_change >= 0 else ''
                ax.annotate(
                    f"{sign}{pct_change:.1f}%",
                    xy=(1.01, 0.5), xycoords='axes fraction',
                    fontsize=5, color=class_color, fontweight='bold',
                    va='center', ha='left',
                )

            ax.set_title(class_label, loc='left', fontsize=7, fontweight='bold', pad=2)
            ax.set_ylabel('')
            ax.tick_params(axis='y', left=False, labelleft=False)

            # Minimal horizontal gridlines only
            ax.grid(axis='y', alpha=0.15, linestyle='-', linewidth=0.5)
            ax.grid(axis='x', visible=False)
            for spine in ['top', 'right', 'left']:
                ax.spines[spine].set_visible(False)
            ax.spines['bottom'].set_linewidth(0.4)
            ax.spines['bottom'].set_color('#cccccc')

            ax.set_ylim(y_min - y_pad, y_max + y_pad)

            # Hide x-tick labels on non-bottom panels
            if idx < n_panels - 1:
                ax.tick_params(axis='x', labelbottom=False, length=0)

        axes[-1].set_xlabel('Year', fontsize=AXIS_LABEL_FONT_SIZE)
        axes[-1].tick_params(axis='x', labelsize=TICK_LABEL_FONT_SIZE)
        axes[-1].spines['bottom'].set_color('#999999')
        fig.suptitle('Yearly Area Trend by Landcover Class (km²)', fontsize=TITLE_FONT_SIZE, fontweight='bold', y=0.998)
        fig.tight_layout(rect=[0, 0, 0.94, 0.99], h_pad=0.15)
        _save_and_close(fig, OUTPUT_DIR / 'yearly_area_by_landcover_class_panel_lines_km2.png')

    # A.1b) Consolidated all-years stacked bar (same height & class order as panel line plot)
    if not df_consolidated.empty and 'combined' in df_consolidated['variant_key'].unique():
        combined_by_type = (
            df_consolidated[df_consolidated['variant_key'] == 'combined']
            .groupby('class_type', as_index=False)['area_km2']
            .sum()
        )
        bar_types = [t for t in line_class_order if t in combined_by_type['class_type'].values]
        type_areas = combined_by_type.set_index('class_type').reindex(bar_types)['area_km2']
        total_area = type_areas.sum()

        # Reverse stacking order so first item in line_class_order ends up on TOP
        stack_order = list(reversed(bar_types))

        bar_figsize = (2.8, panel_figsize[1])
        fig, ax = plt.subplots(figsize=bar_figsize)
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')

        # Stack segments (bottom-to-top = reversed line_class_order)
        segment_midpoints = {}
        bottom = 0
        for class_type in stack_order:
            area = type_areas[class_type]
            color = yearly_type_palette.get(class_type, '#777777')
            ax.bar(
                0, area, bottom=bottom, width=0.5,
                color=color, edgecolor='white', linewidth=0.4,
            )
            segment_midpoints[class_type] = bottom + area / 2
            bottom += area

        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])

        # Build callout annotations to the right, spaced to avoid overlap
        callout_data = []
        for class_type in bar_types:
            area = type_areas[class_type]
            pct = area / total_area * 100
            mid_y = segment_midpoints[class_type]
            color = yearly_type_palette.get(class_type, '#777777')
            callout_data.append({
                'class_type': class_type,
                'area': area,
                'pct': pct,
                'mid_y': mid_y,
                'color': color,
            })

        # Sort callouts by midpoint position and space them out
        callout_data.sort(key=lambda d: d['mid_y'])
        y_range = bottom
        min_gap = y_range * 0.065  # minimum vertical gap between labels
        placed_y = []
        for item in callout_data:
            target_y = item['mid_y']
            # Push up if too close to previous label
            if placed_y and target_y < placed_y[-1] + min_gap:
                target_y = placed_y[-1] + min_gap
            placed_y.append(target_y)
            item['label_y'] = target_y

        # Draw callout lines + text
        x_bar_edge = 0.25  # right edge of bar (width=0.5, centered at 0)
        x_text = 0.42
        for item in callout_data:
            label = f"{item['class_type']}\n{item['area']:,.0f} km² ({item['pct']:.1f}%)"
            ax.annotate(
                label,
                xy=(x_bar_edge, item['mid_y']),
                xytext=(x_text, item['label_y']),
                fontsize=5, fontweight='semibold', color=item['color'],
                va='center', ha='left',
                arrowprops=dict(
                    arrowstyle='-',
                    color=item['color'],
                    lw=0.6,
                    shrinkA=0, shrinkB=2,
                ),
            )

        ax.set_ylabel('Area (km²)', fontsize=AXIS_LABEL_FONT_SIZE)
        ax.tick_params(axis='y', labelsize=TICK_LABEL_FONT_SIZE)
        ax.set_title('Consolidated Composition (km²)', fontsize=TITLE_FONT_SIZE, fontweight='bold')

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        ax.spines['left'].set_linewidth(0.4)
        ax.spines['left'].set_color('#cccccc')
        ax.spines['bottom'].set_linewidth(0.4)
        ax.spines['bottom'].set_color('#999999')
        ax.grid(axis='y', alpha=0.15, linestyle='-', linewidth=0.5)
        ax.grid(axis='x', visible=False)

        fig.tight_layout(rect=[0, 0, 0.99, 0.99])
        _save_and_close(fig, OUTPUT_DIR / 'consolidated_all_years_stacked_bar_km2.png')

# A.2) All-years consolidated class plot (from consolidated combined source)
if not df_consolidated.empty and 'combined' in df_consolidated['variant_key'].unique():
    combined_detail = (
        df_consolidated[df_consolidated['variant_key'] == 'combined']
        [['class_id', 'class_label', 'class_type', 'area_km2']]
        .copy()
    )
    type_rank_combined = {name: idx for idx, name in enumerate(preferred_type_order)}
    combined_detail['type_rank'] = combined_detail['class_type'].map(
        lambda value: type_rank_combined.get(value, 999)
    )
    combined_detail = combined_detail.sort_values(
        ['type_rank', 'class_id', 'class_label']
    ).reset_index(drop=True)

    consolidated_type_palette = _build_type_palette(consolidated_type_map, aggregated_palette)
    combined_colors = [
        consolidated_type_palette.get(type_name, '#777777')
        for type_name in combined_detail['class_type']
    ]

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    bar_container = ax.bar(
        combined_detail['class_label'],
        combined_detail['area_km2'],
        color=combined_colors,
        edgecolor='white',
        linewidth=0.4,
    )
    ax.set_title('All-Years Consolidated Class Area (Grouped by Landcover Class)', fontsize=TITLE_FONT_SIZE, fontweight='bold', pad=24)
    ax.set_xlabel('Consolidated class', fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel('Area (km²)', fontsize=AXIS_LABEL_FONT_SIZE)
    ax.tick_params(axis='x', rotation=70, labelsize=TICK_LABEL_FONT_SIZE)
    ax.tick_params(axis='y', labelsize=TICK_LABEL_FONT_SIZE)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    max_height = combined_detail['area_km2'].max() if not combined_detail.empty else 0
    label_offset = max(max_height * 0.008, 10)
    for rect, value in zip(bar_container, combined_detail['area_km2']):
        x_center = rect.get_x() + rect.get_width() / 2
        y_top = rect.get_height()
        ax.text(
            x_center,
            y_top + label_offset,
            f"{value:,.0f} km²",
            ha='center',
            va='bottom',
            fontsize=6,
            rotation=90,
        )

    ax.set_ylim(0, max(max_height * 1.28, max_height + label_offset * 6))

    legend_types_combined = [
        type_name for type_name in preferred_type_order
        if type_name in combined_detail['class_type'].unique()
    ]
    legend_handles_combined = [
        plt.Rectangle((0, 0), 1, 1, color=consolidated_type_palette.get(type_name, '#777777'))
        for type_name in legend_types_combined
    ]
    ax.legend(
        legend_handles_combined,
        legend_types_combined,
        title='Landcover Class',
        loc='upper left',
        bbox_to_anchor=(1.01, 1),
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize=LEGEND_TITLE_FONT_SIZE,
    )

    fig.tight_layout(rect=[0, 0, 0.8, 0.9])
    _save_and_close(fig, OUTPUT_DIR / 'consolidated_all_years_all_classes_by_main_category_km2.png')

# B) Consolidated variant plots (high-level + detailed)
if not df_consolidated.empty:
    variant_order = [v for v in ['early', 'late', 'combined'] if v in df_consolidated['variant_key'].unique()] + [
        v for v in df_consolidated['variant_key'].unique() if v not in ['early', 'late', 'combined']
    ]
    variant_labels = [consolidated_meta[v]['label'] for v in variant_order]

    class_order = sorted(df_consolidated['class_id'].unique())
    class_labels = [aggregated_labels.get(c, f'Class {c}') for c in class_order]
    class_colors = [aggregated_palette.get(c, '#CCCCCC') for c in class_order]

    consolidated_type = (
        df_consolidated
        .groupby(['variant_label', 'class_type'], as_index=False)['area_km2']
        .sum()
        .pivot(index='variant_label', columns='class_type', values='area_km2')
        .fillna(0)
        .reindex(index=variant_labels)
    )
    consolidated_type_order = [t for t in preferred_type_order if t in consolidated_type.columns] + [
        t for t in consolidated_type.columns if t not in preferred_type_order
    ]
    consolidated_type = consolidated_type[consolidated_type_order]

    consolidated_type_palette = _build_type_palette(consolidated_type_map, aggregated_palette)
    consolidated_type_colors = [consolidated_type_palette.get(t, '#777777') for t in consolidated_type.columns]

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    consolidated_type.plot(
        kind='bar',
        stacked=True,
        ax=ax,
        color=consolidated_type_colors,
        edgecolor='white',
        linewidth=0.4,
    )
    ax.set_title('Consolidated Area by Aggregated Type', fontsize=TITLE_FONT_SIZE, fontweight='bold')
    ax.set_xlabel('Consolidation variant', fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel('Area (km²)', fontsize=AXIS_LABEL_FONT_SIZE)
    ax.tick_params(axis='x', labelrotation=90, labelsize=TICK_LABEL_FONT_SIZE)
    ax.tick_params(axis='y', labelsize=TICK_LABEL_FONT_SIZE)
    ax.legend(title='Class Type', loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_TITLE_FONT_SIZE)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    fig.tight_layout(rect=LEGEND_RECT)
    _save_and_close(fig, OUTPUT_DIR / 'consolidated_area_by_type_km2.png')

    pivot_class = (
        df_consolidated
        .pivot_table(index='variant_label', columns='class_label', values='area_km2', aggfunc='sum', fill_value=0)
        .reindex(index=variant_labels)
    )
    pivot_class = pivot_class[[lbl for lbl in class_labels if lbl in pivot_class.columns]]

    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    pivot_class.plot(
        kind='bar',
        stacked=True,
        ax=ax,
        color=class_colors[:len(pivot_class.columns)],
        edgecolor='white',
        linewidth=0.3,
    )
    ax.set_title('Consolidated Area by Detailed Class', fontsize=TITLE_FONT_SIZE, fontweight='bold')
    ax.set_xlabel('Consolidation variant', fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel('Area (km²)', fontsize=AXIS_LABEL_FONT_SIZE)
    ax.tick_params(axis='x', labelrotation=90, labelsize=TICK_LABEL_FONT_SIZE)
    ax.tick_params(axis='y', labelsize=TICK_LABEL_FONT_SIZE)
    ax.legend(title='Detailed Class', loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_TITLE_FONT_SIZE)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    fig.tight_layout(rect=LEGEND_RECT)
    _save_and_close(fig, OUTPUT_DIR / 'consolidated_area_by_class_km2.png')


# %%
# Save outputs
if not PLOTS_ONLY_MODE:
    if not df_yearly.empty:
        yearly_csv = OUTPUT_DIR / 'class_areas_by_year.csv'
        df_yearly.sort_values(['year', 'class_seq_id']).to_csv(yearly_csv, index=False)
        print(f"✓ Yearly area statistics saved to {yearly_csv}")

    if not df_consolidated.empty:
        consolidated_csv = OUTPUT_DIR / 'consolidated_class_areas.csv'
        df_consolidated.sort_values(['variant_key', 'class_id']).to_csv(consolidated_csv, index=False)
        print(f"✓ Consolidated area statistics saved to {consolidated_csv}")

    if not df_reference.empty:
        reference_csv = OUTPUT_DIR / 'reference_mask_areas.csv'
        sort_cols = [col for col in ['source', 'stratum', 'metric'] if col in df_reference.columns]
        df_reference.sort_values(sort_cols).to_csv(reference_csv, index=False)
        print(f"✓ Reference mask area statistics saved to {reference_csv}")

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
        'reference_masks': {
            'assets': reference_cfg,
            'statistics_km2': df_reference.to_dict(orient='records') if not df_reference.empty else [],
        },
    }

    summary_path = OUTPUT_DIR / 'area_statistics_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary saved to {summary_path}")
else:
    summary_path = OUTPUT_DIR / 'area_statistics_summary.json'
    if summary_path.exists():
        print(f"✓ Existing summary preserved at {summary_path}")


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
if not df_consolidated.empty:
    print("  - consolidated_class_areas.csv")
if not df_reference.empty:
    print("  - reference_mask_areas.csv")
for plot_path in generated_plot_paths:
    print(f"  - {plot_path.name}")
print("  - area_statistics_summary.json")
print("\n" + "="*70)

# %%
