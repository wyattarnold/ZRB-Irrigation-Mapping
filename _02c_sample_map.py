# %% [markdown]
# # Step 2c: Self-Contained Sample Map (CSV-based)
#
# Builds a self-contained Leaflet HTML map from local CSVs only (no Earth Engine tokens).
#
# Includes:
# - Cleaned landcover samples (from _01c_clean_landcover_samples.py)
# - Clustered crop samples by year and cluster (from _02b_clustering.py + sampled points)
# - High-resolution public basemaps (Esri, Google Satellite, OSM)
# - Combined legend for landcover classes and cluster IDs
#
# Output:
# - output/{study_area}/training/samples_combined_map.html

# %%
import sys
from pathlib import Path

import pandas as pd
import folium
from branca.element import MacroElement, Template

sys.path.insert(0, str(Path(__file__).parent))

import __config__
from src.gee_utils import initialize_earth_engine


# %%
print("Initializing self-contained map build...")

initialize_earth_engine()

study_area_key = __config__.CURRENT_STUDY_AREA
study_cfg = __config__.get_study_area(study_area_key)
study_name = study_cfg['name']
center = study_cfg['center']
zoom = study_cfg['zoom']

print(f"Study area: {study_name}")

training_dir = Path('output') / study_area_key / 'training'
training_dir.mkdir(parents=True, exist_ok=True)

cleaned_landcover_csv = training_dir / 'landcover_samples_cleaned.csv'
cluster_results_csv = training_dir / 'clustering' / 'clustering_results.csv'
sample_points_csv = training_dir / 'sampled_points_data.csv'

for p in [cleaned_landcover_csv, cluster_results_csv, sample_points_csv]:
    if not p.exists():
        raise FileNotFoundError(f"Required input missing: {p}")


# %%
# Colors
LANDCOVER_COLORS = {
    'water': '#1F78B4',
    'trees': '#006400',
    'grass': '#FFFF99',
    'flooded_veg': '#33A02C',
    'shrubs': '#FF7F00',
    'urban': '#E31A1C',
    'bare': '#8C7A6B',
}

CLUSTER_COLORS = [
    '#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a',
    '#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f', '#cab2d6',
]

# Sample map controls from central config
PERF_DEFAULTS = {
    'prefer_canvas': True,
    'include_landcover_popups': False,
    'include_cluster_popups': True,
    'show_landcover_layers': False,
    'show_all_year_layers': True,
    'show_year_specific_layers': False,
    'build_all_year_layers': True,
    'build_year_specific_layers': False,
}
PERF = {
    **PERF_DEFAULTS,
    **getattr(__config__, 'SAMPLE_MAP', {}),
}


def _add_point_layer(
    feature_group: folium.FeatureGroup,
    layer_df: pd.DataFrame,
    color: str,
    radius: float,
    fill_opacity: float,
    popup_builder=None,
) -> None:
    """Add points to a feature group using individual CircleMarker rendering."""
    for row in layer_df.itertuples(index=False):
        popup_text = popup_builder(row) if popup_builder is not None else None
        folium.CircleMarker(
            location=[float(row.lat), float(row.lon)],
            radius=radius,
            color='#000000',
            fill=True,
            fill_color=color,
            fill_opacity=fill_opacity,
            weight=0.5,
            popup=popup_text,
        ).add_to(feature_group)


def _add_cluster_layers(
    map_obj: folium.Map,
    cluster_df: pd.DataFrame,
    cluster_color_map: dict[int, str],
    show_layers: bool,
    layer_name_fn,
    fill_opacity: float,
) -> None:
    """Build cluster feature groups from a cluster dataframe subset."""
    for cid in sorted(cluster_df['cluster'].unique()):
        cid_df = cluster_df[cluster_df['cluster'] == cid]
        color = cluster_color_map.get(int(cid), '#808080')
        layer_name = layer_name_fn(int(cid), cid_df)

        fg = folium.FeatureGroup(name=layer_name, show=show_layers)

        popup_builder = None
        if PERF['include_cluster_popups']:
            cid_popup_df = (
                cid_df.groupby(['lat', 'lon'], as_index=False)
                .agg(
                    years=('year', lambda s: sorted({int(v) for v in s if pd.notna(v)})),
                    sample_count=('sample_num', 'nunique'),
                )
                .copy()
            )

            popup_builder = lambda row: (
                f"Cluster: {int(cid)}<br>"
                f"Years active: {', '.join(str(y) for y in row.years)}<br>"
                f"N years: {len(row.years)}<br>"
                f"Samples: {int(row.sample_count)}<br>"
                f"Lon: {float(row.lon):.6f}<br>"
                f"Lat: {float(row.lat):.6f}"
            )
            layer_points_df = cid_popup_df
        else:
            layer_points_df = cid_df

        _add_point_layer(
            feature_group=fg,
            layer_df=layer_points_df,
            color=color,
            radius=3,
            fill_opacity=fill_opacity,
            popup_builder=popup_builder,
        )
        fg.add_to(map_obj)


# %%
print("Loading CSV data...")
df_landcover = pd.read_csv(cleaned_landcover_csv)
df_cluster = pd.read_csv(cluster_results_csv)
df_points = pd.read_csv(sample_points_csv)

required_point_cols = {'sample_num', 'lon', 'lat'}
if not required_point_cols.issubset(df_points.columns):
    raise ValueError(f"sampled_points_data.csv must contain columns: {required_point_cols}")

coords = df_points[['sample_num', 'lon', 'lat']].drop_duplicates(subset=['sample_num'])
df_cluster_map = df_cluster.merge(coords, on='sample_num', how='left')
df_cluster_map = df_cluster_map.dropna(subset=['lon', 'lat', 'year', 'cluster']).copy()

df_cluster_map['year'] = df_cluster_map['year'].astype(int)
df_cluster_map['cluster'] = df_cluster_map['cluster'].astype(int)

cluster_ids = [int(c) for c in sorted(df_cluster_map['cluster'].unique())]
cluster_years = [int(y) for y in sorted(df_cluster_map['year'].unique())]

if len(cluster_ids) == 0:
    raise ValueError("No clustered samples found after merging clustering_results.csv with coordinates.")

cluster_hex = {
    cid: CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
    for i, cid in enumerate(cluster_ids)
}

print(f"  Cleaned landcover points: {len(df_landcover)}")
print(f"  Clustered year-samples: {len(df_cluster_map)}")
print(f"  Cluster IDs: {cluster_ids}")
print(f"  Cluster years: {cluster_years}")


# %%
# Create Leaflet map
m = folium.Map(
    location=[center[0], center[1]],
    zoom_start=zoom,
    tiles=None,
    control_scale=True,
    prefer_canvas=PERF['prefer_canvas'],
)

# Basemaps (public, high-res where available)
folium.TileLayer(
    tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
    attr='Google',
    name='Google Satellite',
    overlay=False,
    control=True,
    show=True,
).add_to(m)

folium.TileLayer(
    tiles='OpenStreetMap',
    name='OpenStreetMap',
    overlay=False,
    control=True,
    show=False,
).add_to(m)


# %%
# Add landcover layers by class
print("Adding landcover layers...")
landcover_classes = sorted(df_landcover['class_name'].dropna().unique())

for class_name in landcover_classes:
    class_df = df_landcover[df_landcover['class_name'] == class_name]
    color = LANDCOVER_COLORS.get(class_name, '#808080')

    fg = folium.FeatureGroup(
        name=f"Landcover {class_name} ({len(class_df)})",
        show=PERF['show_landcover_layers'],
    )

    popup_builder = None
    if PERF['include_landcover_popups']:
        popup_builder = lambda row, class_name=class_name: (
            f"Landcover: {class_name}<br>"
            f"Year: {int(row.year) if pd.notna(row.year) else 'NA'}<br>"
            f"Lon: {float(row.lon):.6f}<br>"
            f"Lat: {float(row.lat):.6f}"
        )

    _add_point_layer(
        feature_group=fg,
        layer_df=class_df,
        color=color,
        radius=2,
        fill_opacity=0.85,
        popup_builder=popup_builder,
    )

    fg.add_to(m)


# %%
# Add clustered layers by year and cluster (year-specific)
if PERF['build_year_specific_layers']:
    print("Adding clustered sample layers (year-specific)...")

# Add all-years cluster toggle layers
print("Adding clustered sample layers (all-years toggles)...")
if PERF['build_all_year_layers']:
    _add_cluster_layers(
        map_obj=m,
        cluster_df=df_cluster_map,
        cluster_color_map=cluster_hex,
        show_layers=PERF['show_all_year_layers'],
        layer_name_fn=lambda cid, cid_df: f"Cluster ALL Years - C{cid} ({len(cid_df)})",
        fill_opacity=0.8,
    )

if PERF['build_year_specific_layers']:
    for year in cluster_years:
        year_df = df_cluster_map[df_cluster_map['year'] == year]
        _add_cluster_layers(
            map_obj=m,
            cluster_df=year_df,
            cluster_color_map=cluster_hex,
            show_layers=PERF['show_year_specific_layers'],
            layer_name_fn=lambda cid, cid_df, year=year: f"Clusters {year} - C{cid} ({len(cid_df)})",
            fill_opacity=0.9,
        )


# %%
# Combined legend (landcover + clusters)
landcover_items = [
    (name, LANDCOVER_COLORS.get(name, '#808080'))
    for name in landcover_classes
]
cluster_items = [
    (f"Cluster {cid}", cluster_hex[cid])
    for cid in cluster_ids
]

legend_html = "".join([
    f"<div style='display:flex;align-items:center;margin:2px 0;'>"
    f"<span style='display:inline-block;width:12px;height:12px;background:{color};border:1px solid #333;margin-right:6px;'></span>"
    f"<span>{label}</span></div>"
    for label, color in landcover_items
])

cluster_html = "".join([
    f"<div style='display:flex;align-items:center;margin:2px 0;'>"
    f"<span style='display:inline-block;width:12px;height:12px;background:{color};border:1px solid #333;margin-right:6px;'></span>"
    f"<span>{label}</span></div>"
    for label, color in cluster_items
])

template = Template(f"""
{{% macro html(this, kwargs) %}}
<div style="
    position: fixed;
    bottom: 30px;
    left: 30px;
    z-index: 9999;
    background-color: rgba(255, 255, 255, 0.95);
    border: 2px solid #444;
    border-radius: 6px;
    padding: 10px 12px;
    font-size: 12px;
    line-height: 1.2;
    max-height: 55vh;
    overflow-y: auto;
    min-width: 190px;
">
  <div style="font-weight:bold;margin-bottom:6px;">Landcover Classes</div>
  {legend_html}
  <hr style="margin:8px 0;">
  <div style="font-weight:bold;margin-bottom:6px;">Cluster IDs</div>
  {cluster_html}
</div>
{{% endmacro %}}
""")
macro = MacroElement()
macro._template = template
m.get_root().add_child(macro)


# %%
# Controls and save
folium.LayerControl(collapsed=False).add_to(m)

map_path = training_dir / 'samples_combined_map.html'
m.save(str(map_path))

print("\n" + "=" * 60)
print("SELF-CONTAINED MAP COMPLETE")
print("=" * 60)
print(f"✓ Saved map to: {map_path}")
print("\nIncluded:")
print("  - Public basemaps (Google Satellite / OSM)")
print("  - Cleaned landcover samples from CSV")
if PERF['build_all_year_layers']:
    print("  - Clustered samples by cluster (ALL years toggle)")
if PERF['build_year_specific_layers']:
    print("  - Clustered samples by year and cluster from CSV")
print("\nPerformance settings:")
print(f"  - prefer_canvas: {PERF['prefer_canvas']}")
print(f"  - include_landcover_popups: {PERF['include_landcover_popups']}")
print(f"  - include_cluster_popups: {PERF['include_cluster_popups']}")
print(f"  - build_all_year_layers: {PERF['build_all_year_layers']}")
print(f"  - build_year_specific_layers: {PERF['build_year_specific_layers']}")
print("  - Built-in legend for landcover classes and cluster IDs")
