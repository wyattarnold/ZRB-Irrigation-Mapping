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
import matplotlib.pyplot as plt
import numpy as np
from branca.element import MacroElement, Template

sys.path.insert(0, str(Path(__file__).parent))

import __config__


# %%
print("Initializing self-contained map build...")

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

cluster_colors = plt.cm.tab20(np.linspace(0, 1, max(len(cluster_ids), 1)))
cluster_hex = {
    cid: f"#{int(cluster_colors[i][0]*255):02x}{int(cluster_colors[i][1]*255):02x}{int(cluster_colors[i][2]*255):02x}"
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

    fg = folium.FeatureGroup(name=f"Landcover {class_name} ({len(class_df)})", show=True)

    for _, row in class_df.iterrows():
        popup_text = (
            f"Landcover: {class_name}<br>"
            f"Year: {int(row['year']) if pd.notna(row.get('year')) else 'NA'}<br>"
            f"Lon: {float(row['lon']):.6f}<br>"
            f"Lat: {float(row['lat']):.6f}"
        )
        folium.CircleMarker(
            location=[float(row['lat']), float(row['lon'])],
            radius=2,
            color='#000000',
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            weight=0.5,
            popup=popup_text,
        ).add_to(fg)

    fg.add_to(m)


# %%
# Add clustered layers by year and cluster (year-specific)
print("Adding clustered sample layers (year-specific)...")
latest_year = max(cluster_years)

# Add all-years cluster toggle layers
print("Adding clustered sample layers (all-years toggles)...")
for cid in cluster_ids:
    cluster_all_df = df_cluster_map[df_cluster_map['cluster'] == cid]
    color = cluster_hex.get(cid, '#808080')

    layer_name = f"Cluster ALL Years - C{cid} ({len(cluster_all_df)})"
    fg_all = folium.FeatureGroup(name=layer_name, show=True)

    for _, row in cluster_all_df.iterrows():
        popup_text = (
            f"Sample: {int(row['sample_num'])}<br>"
            f"Year: {int(row['year'])}<br>"
            f"Cluster: {int(row['cluster'])}<br>"
            f"Lon: {float(row['lon']):.6f}<br>"
            f"Lat: {float(row['lat']):.6f}"
        )
        folium.CircleMarker(
            location=[float(row['lat']), float(row['lon'])],
            radius=3,
            color='#000000',
            fill=True,
            fill_color=color,
            fill_opacity=0.8,
            weight=0.5,
            popup=popup_text,
        ).add_to(fg_all)

    fg_all.add_to(m)

for year in cluster_years:
    year_df = df_cluster_map[df_cluster_map['year'] == year]

    for cid in sorted(year_df['cluster'].unique()):
        cluster_df = year_df[year_df['cluster'] == cid]
        color = cluster_hex.get(cid, '#808080')

        layer_name = f"Clusters {year} - C{cid} ({len(cluster_df)})"
        fg = folium.FeatureGroup(name=layer_name, show=False)

        for _, row in cluster_df.iterrows():
            popup_text = (
                f"Sample: {int(row['sample_num'])}<br>"
                f"Year: {int(row['year'])}<br>"
                f"Cluster: {int(row['cluster'])}<br>"
                f"Lon: {float(row['lon']):.6f}<br>"
                f"Lat: {float(row['lat']):.6f}"
            )
            folium.CircleMarker(
                location=[float(row['lat']), float(row['lon'])],
                radius=3,
                color='#000000',
                fill=True,
                fill_color=color,
                fill_opacity=0.9,
                weight=0.5,
                popup=popup_text,
            ).add_to(fg)

        fg.add_to(m)


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
print("  - Clustered samples by cluster (ALL years toggle)")
print("  - Clustered samples by year and cluster from CSV")
print("  - Built-in legend for landcover classes and cluster IDs")
