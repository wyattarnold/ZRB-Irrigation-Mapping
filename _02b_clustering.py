"""
Hierarchical Clustering for Irrigation Timeseries

Clusters irrigation patterns using Dynamic Time Warping (DTW) distance on the
composite spectral index with hierarchical (agglomerative) clustering.

Run: python _02b_clustering.py
"""

import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform

warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')

# Project imports
import __config__
from src.clustering_utils import (
    compute_dtw_matrix,
    gap_fill_timeseries,
    merge_small_clusters,
    split_by_calendar_year
)
from src.gee_utils import initialize_earth_engine

# GEE import
import ee

# =============================================================================
# Configuration
# =============================================================================

CLUSTERING_CONFIG = __config__.CLUSTERING
TRAINING_DIR = Path('output') / __config__.CURRENT_STUDY_AREA / 'training'
OUTPUT_DIR = TRAINING_DIR / 'clustering'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TIMESERIES_DIR = TRAINING_DIR / 'timeseries'
S2_INDICES = CLUSTERING_CONFIG['s2_indices']
CLUSTERING_INDEX = CLUSTERING_CONFIG['clustering_index']
USE_DRY_SEASON_ONLY = CLUSTERING_CONFIG['use_dry_season_only']
MAX_PLOT_COLUMNS = CLUSTERING_CONFIG['max_plot_columns']
DTW_WINDOW = CLUSTERING_CONFIG.get('dtw_window', 15)
DTW_NORMALIZE = CLUSTERING_CONFIG.get('dtw_normalize', True)
USE_PRECOMPUTED_DTW = CLUSTERING_CONFIG.get('use_precomputed_dtw', True)
MAX_SAMPLES = CLUSTERING_CONFIG.get('max_samples', None)

# Hierarchical clustering parameters
N_CLUSTERS = CLUSTERING_CONFIG.get('n_clusters', 5)
LINKAGE_METHOD = CLUSTERING_CONFIG.get('linkage_method', 'average')
MIN_CLUSTER_SIZE = CLUSTERING_CONFIG.get('min_cluster_size', None)
MIN_OBS_PER_YEAR = CLUSTERING_CONFIG.get('min_obs_per_year', 20)

# Dry season DOY in calendar year (Jan 1 = DOY 0)
# May 1 = DOY 121, Sep 30 = DOY 273
DRY_SEASON_START_DOY = max(0, 121 - DTW_WINDOW)  # May 1
DRY_SEASON_END_DOY = min(364, 273 + DTW_WINDOW)  # Sep 30


# =============================================================================
# Main Script
# =============================================================================

print("="*80)
print("HIERARCHICAL CLUSTERING WITH DTW")
print("="*80)
season_info = "Dry season (May-Sep)" if USE_DRY_SEASON_ONLY else "Full year"
min_size_info = f" | min_size={MIN_CLUSTER_SIZE}" if MIN_CLUSTER_SIZE else ""
print(f"Config: {N_CLUSTERS} clusters | {LINKAGE_METHOD} linkage | {season_info}{min_size_info}")
print(f"Clustering index: {CLUSTERING_INDEX}")
print(f"DTW window: {DTW_WINDOW} days\n")

# %% Load and Process All Timeseries

s2_files = sorted(TIMESERIES_DIR.glob('sample_*_s2.csv'))
sample_nums = [int(f.stem.split('_')[1]) for f in s2_files]
print(f"Found {len(s2_files)} samples with S2 timeseries")

# Optionally limit number of samples
if MAX_SAMPLES is not None and MAX_SAMPLES < len(sample_nums):
    sample_nums = sample_nums[:MAX_SAMPLES]
    print(f"  Limited to first {MAX_SAMPLES} samples")

all_year_samples = {}
sample_metadata = []

for sample_num in sample_nums:
    try:
        # Load S2 data
        s2_path = TIMESERIES_DIR / f'sample_{sample_num:03d}_s2.csv'
        df_s2 = pd.read_csv(s2_path, parse_dates=['date'])
        
        # Normalize dates (remove time component for proper merging)
        df_s2['date'] = pd.to_datetime(df_s2['date']).dt.normalize()
        # Drop Feb 29 to keep consistent 365-day years
        df_s2 = df_s2[~((df_s2['date'].dt.month == 2) & (df_s2['date'].dt.day == 29))]
        
        df = df_s2.copy()
        
        # Split by calendar year (Jan 1 - Dec 31)
        yearly_data, year_ranges = split_by_calendar_year(df, min_obs_per_year=MIN_OBS_PER_YEAR)
        
        for year, year_df in yearly_data.items():
            available_indices = [idx for idx in S2_INDICES if idx in year_df.columns]
            start, end = year_ranges[year]
            
            # Gap-fill timeseries to daily values
            year_df_filled = gap_fill_timeseries(year_df, available_indices, start_date=start, end_date=end)
            year_df_filled['doy'] = year_df_filled['date'].dt.dayofyear - 1  # 0-indexed DOY
            
            key = (sample_num, year)
            all_year_samples[key] = year_df_filled
            
            sample_metadata.append({
                'sample_num': sample_num,
                'year': year,
                'sample_id': f'sample_{sample_num:03d}_y{year}',
                'n_days': len(year_df_filled),
            })
    
    except Exception as e:
        print(f"  Error loading sample {sample_num}: {e}")
        continue

metadata_df = pd.DataFrame(sample_metadata)
n_samples = len(all_year_samples)
print(f"Loaded {n_samples} year-samples ({metadata_df['sample_num'].nunique()} locations, "
      f"{len(metadata_df['year'].unique())} years)")

metadata_df.to_csv(OUTPUT_DIR / 'sample_metadata.csv', index=False)

# %% Calculate DTW Distance Matrix

print(f"\nCalculating DTW distance matrix for {CLUSTERING_INDEX}...")

sample_keys = list(all_year_samples.keys())
# Use 365 days for calendar year (ignore leap years for DTW alignment)
n_doy = 365

# Create dry season mask
if USE_DRY_SEASON_ONLY:
    dry_season_mask = np.zeros(n_doy, dtype=bool)
    dry_season_mask[DRY_SEASON_START_DOY:DRY_SEASON_END_DOY+1] = True
    n_doy_filtered = np.sum(dry_season_mask)
    print(f"  Using dry season only: {n_doy_filtered} days (DOY {DRY_SEASON_START_DOY}-{DRY_SEASON_END_DOY})")
else:
    dry_season_mask = None

# Check for precomputed DTW matrix
dtw_path = OUTPUT_DIR / f'dtw_distance_{CLUSTERING_INDEX.lower()}.csv'
distance_matrix = None

if USE_PRECOMPUTED_DTW and dtw_path.exists():
    print(f"  Loading precomputed DTW matrix...")
    try:
        matrix_df = pd.read_csv(dtw_path, index_col=0)
        if matrix_df.shape[0] == n_samples and matrix_df.shape[1] == n_samples:
            distance_matrix = matrix_df.values
            print(f"  ✓ Loaded: {matrix_df.shape}")
        else:
            print(f"  ✗ Shape mismatch: expected ({n_samples}, {n_samples}), got {matrix_df.shape}")
    except Exception as e:
        print(f"  Error loading: {e}")

if distance_matrix is None:
    print(f"  Computing DTW distance matrix...")
    t0 = time.time()
    
    # Align timeseries to common DOY grid
    aligned_data = np.full((n_samples, n_doy), np.nan)
    for i, key in enumerate(sample_keys):
        df = all_year_samples[key]
        doy_values = df['doy'].values.astype(int)
        valid_mask = (doy_values >= 0) & (doy_values < n_doy)
        if CLUSTERING_INDEX in df.columns:
            aligned_data[i, doy_values[valid_mask]] = df[CLUSTERING_INDEX].values[valid_mask]
    
    # Apply dry season mask
    if dry_season_mask is not None:
        aligned_data = aligned_data[:, dry_season_mask]
    
    # Compute DTW matrix
    distance_matrix = compute_dtw_matrix(aligned_data, window=DTW_WINDOW, normalize=DTW_NORMALIZE)
    
    # Handle inf/nan values
    finite_mask = np.isfinite(distance_matrix)
    if not np.all(finite_mask):
        max_finite = np.max(distance_matrix[finite_mask]) if np.any(finite_mask) else 1.0
        distance_matrix = np.where(finite_mask, distance_matrix, max_finite * 2)
    
    # Ensure symmetric
    distance_matrix = np.maximum(distance_matrix, 0)
    np.fill_diagonal(distance_matrix, 0)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    
    elapsed = time.time() - t0
    print(f"  ✓ Computed in {elapsed:.1f}s")
    
    # Save for future use
    matrix_df = pd.DataFrame(
        distance_matrix,
        index=[f"{k[0]:03d}_y{k[1]}" for k in sample_keys],
        columns=[f"{k[0]:03d}_y{k[1]}" for k in sample_keys]
    )
    matrix_df.to_csv(dtw_path)
    print(f"  Saved to {dtw_path}")

# Report statistics
valid_dists = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
print(f"  Distance stats: min={valid_dists.min():.3f}, median={np.median(valid_dists):.3f}, max={valid_dists.max():.3f}")

# %% Hierarchical Clustering

print(f"\nHierarchical clustering ({LINKAGE_METHOD} linkage, {N_CLUSTERS} clusters)...")
t_start = time.time()

# Convert to condensed distance matrix for scipy
t0 = time.time()
condensed_dist = squareform(distance_matrix, checks=False)
print(f"  squareform: {time.time()-t0:.1f}s")

# Compute linkage
t0 = time.time()
linkage_matrix = linkage(condensed_dist, method=LINKAGE_METHOD)
print(f"  linkage: {time.time()-t0:.1f}s")

# Cut dendrogram to get cluster labels
cluster_labels = fcluster(linkage_matrix, N_CLUSTERS, criterion='maxclust')
cluster_labels = cluster_labels - 1  # Make 0-indexed

# Enforce minimum cluster size by reassigning small clusters
if MIN_CLUSTER_SIZE is not None:
    print(f"  Enforcing min_cluster_size={MIN_CLUSTER_SIZE}...")
    t0 = time.time()
    cluster_labels = merge_small_clusters(
        cluster_labels, distance_matrix, MIN_CLUSTER_SIZE, verbose=True
    )
    print(f"  min_cluster_size enforcement: {time.time()-t0:.1f}s")

unique_clusters = np.unique(cluster_labels)
print(f"  Final clusters: {len(unique_clusters)}")
for c in unique_clusters:
    count = np.sum(cluster_labels == c)
    print(f"    Cluster {c}: {count} samples")

# Save results
metadata_df['cluster'] = cluster_labels
metadata_df.to_csv(OUTPUT_DIR / 'clustering_results.csv', index=False)

print(f"  Total clustering: {time.time()-t_start:.1f}s")

# Store actual number of clusters after min_size enforcement
n_final_clusters = len(unique_clusters)

# %% Visualize Dendrogram

print(f"\nGenerating visualizations...")
t0 = time.time()

fig, ax = plt.subplots(figsize=(16, 8))

dendrogram(
    linkage_matrix,
    ax=ax,
    labels=[f"{k[0]:03d}_y{k[1]}" for k in sample_keys],
    leaf_font_size=6,
    leaf_rotation=90,
    color_threshold=linkage_matrix[-(N_CLUSTERS-1), 2] if N_CLUSTERS > 1 else 0
)

ax.set_xlabel('Sample ID', fontsize=12)
ax.set_ylabel('DTW Distance', fontsize=12)
ax.set_title(f'Hierarchical Clustering Dendrogram ({LINKAGE_METHOD} linkage)\n'
             f'{n_samples} year-samples, {N_CLUSTERS} clusters, {CLUSTERING_INDEX}', fontsize=14)
ax.axhline(y=linkage_matrix[-(N_CLUSTERS-1), 2] if N_CLUSTERS > 1 else 0, 
           color='r', linestyle='--', label=f'Cut for {N_CLUSTERS} clusters')
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'dendrogram.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Dendrogram: {time.time()-t0:.1f}s")

# %% Distance Heatmap

t0 = time.time()

# Sort by cluster
sort_idx = np.argsort(cluster_labels)
sorted_distance = distance_matrix[sort_idx][:, sort_idx]
sorted_labels = [f"{sample_keys[i][0]:03d}_y{sample_keys[i][1]}" for i in sort_idx]
sorted_clusters = cluster_labels[sort_idx]

fig, ax = plt.subplots(figsize=(12, 10))

vmax = np.percentile(sorted_distance[sorted_distance > 0], 95)
im = ax.imshow(sorted_distance, cmap='RdYlGn_r', vmin=0, vmax=vmax, aspect='auto')

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('DTW Distance', fontsize=12)

# Draw cluster boundaries
cluster_changes = np.where(np.diff(sorted_clusters) != 0)[0] + 1
for boundary in cluster_changes:
    ax.axhline(boundary - 0.5, color='black', linewidth=2)
    ax.axvline(boundary - 0.5, color='black', linewidth=2)

ax.set_xlabel('Sample', fontsize=12)
ax.set_ylabel('Sample', fontsize=12)
ax.set_title(f'DTW Distance Matrix ({CLUSTERING_INDEX})\nSorted by Cluster', fontsize=14)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'distance_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Heatmap: {time.time()-t0:.1f}s")

# %% Cluster Characterization

print("\n" + "="*80)
print("CLUSTER CHARACTERIZATION")
print("="*80)

for cluster_id in sorted(unique_clusters):
    cluster_samples = metadata_df[metadata_df['cluster'] == cluster_id]
    
    print(f"\nCluster {cluster_id} (n={len(cluster_samples)}):")
    print(f"  Years: {sorted(cluster_samples['year'].unique())}")
    print(f"  Locations: {cluster_samples['sample_num'].nunique()} unique samples")
    

# %% Plot Cluster Patterns

t0 = time.time()

n_rows = int(np.ceil(n_final_clusters / MAX_PLOT_COLUMNS))
n_cols = min(n_final_clusters, MAX_PLOT_COLUMNS)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
if n_final_clusters == 1:
    axes = np.array([axes])
axes = axes.flatten()

colors = plt.cm.tab10(np.linspace(0, 1, max(n_final_clusters, 1)))

for cluster_id in sorted(unique_clusters):
    ax = axes[cluster_id]
    cluster_keys = [sample_keys[i] for i in range(n_samples) if cluster_labels[i] == cluster_id]
    
    cluster_data = []
    for key in cluster_keys:
        df = all_year_samples[key].copy()
        available_cols = ['doy'] + [c for c in S2_INDICES if c in df.columns]
        cluster_data.append(df[available_cols])
    
    combined = pd.concat(cluster_data, ignore_index=True)
    plot_indices = [c for c in S2_INDICES if c in combined.columns]
    doy_stats = combined.groupby('doy')[plot_indices].agg(['mean', 'std']).reset_index()
    
    for idx in plot_indices:
        mean_col = (idx, 'mean')
        std_col = (idx, 'std')
        if mean_col in doy_stats.columns:
            doy = doy_stats['doy'].values
            mean = doy_stats[mean_col].values
            std = doy_stats[std_col].values
            if not np.all(np.isnan(mean)):
                ax.plot(doy, mean, label=idx, linewidth=2, alpha=0.8)
                ax.fill_between(doy, mean - std, mean + std, alpha=0.2)
    
    ax.axvspan(DRY_SEASON_START_DOY, DRY_SEASON_END_DOY, alpha=0.1, color='orange', label='Dry Season')
    ax.set_title(f'Cluster {cluster_id} (n={len(cluster_keys)})', fontsize=12, fontweight='bold')
    ax.set_xlabel('Day of Year', fontsize=11)
    ax.set_ylabel('Index Value', fontsize=11)
    ax.set_xlim(0, 366)
    ax.set_ylim(-0.6, 1.0)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

for idx in range(n_final_clusters, len(axes)):
    axes[idx].axis('off')

plt.suptitle(f'Average Annual Patterns by Cluster (S2 Indices)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'cluster_patterns_s2.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Cluster patterns: {time.time()-t0:.1f}s")

# %% Cluster Comparison - Clustering Index

t0 = time.time()

fig, ax = plt.subplots(figsize=(12, 6))

for cluster_id in sorted(unique_clusters):
    cluster_keys = [sample_keys[i] for i in range(n_samples) if cluster_labels[i] == cluster_id]
    
    cluster_data = []
    for key in cluster_keys:
        df = all_year_samples[key].copy()
        if CLUSTERING_INDEX in df.columns:
            cluster_data.append(df[['doy', CLUSTERING_INDEX]])
    
    if cluster_data:
        combined = pd.concat(cluster_data, ignore_index=True)
        doy_stats = combined.groupby('doy')[CLUSTERING_INDEX].mean().reset_index()
        ax.plot(doy_stats['doy'], doy_stats[CLUSTERING_INDEX], 
                label=f'Cluster {cluster_id} (n={len(cluster_keys)})', 
                linewidth=2.5, alpha=0.9, color=colors[cluster_id])

ax.axvspan(DRY_SEASON_START_DOY, DRY_SEASON_END_DOY, alpha=0.12, color='orange', zorder=0)
ax.set_title(f'{CLUSTERING_INDEX} by Cluster', fontsize=14, fontweight='bold')
ax.set_xlabel('Day of Year', fontsize=12)
ax.set_ylabel(CLUSTERING_INDEX, fontsize=12)
ax.set_xlim(0, 366)
ax.legend(fontsize=10, loc='best')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'cluster_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  Cluster comparison: {time.time()-t0:.1f}s")

# %% Summary

print("\n" + "="*80)
print("COMPLETE")
print("="*80)
print(f"Results: {OUTPUT_DIR}/")
print(f"Clusters: {n_final_clusters} | Samples: {n_samples}")
print(f"Clustering index: {CLUSTERING_INDEX}")
print(f"Mean DTW distance: {np.mean(valid_dists):.3f}")
print()
print("Done!")

