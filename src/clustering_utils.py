"""Clustering Utility Functions

Shared functions for DTW-based timeseries clustering, gap-filling,
and Google Earth Engine (GEE) operations.
"""

import numpy as np
import pandas as pd

import __config__

# Try to import fast DTW implementation
try:
    from dtaidistance import dtw
    DTW_FAST = True
    # Check if C library with OpenMP is available
    DTW_C_AVAILABLE = dtw.try_import_c()
except ImportError:
    DTW_FAST = False
    DTW_C_AVAILABLE = False


# =============================================================================
# DTW Distance Functions
# =============================================================================

DEFAULT_MIN_OBS_PER_YEAR = __config__.CLUSTERING.get("min_obs_per_year", 20)

def compute_dtw_matrix(data, window=None, normalize=True, min_valid_points=30):
    """Compute pairwise DTW distance matrix for timeseries data.
    
    Parameters
    ----------
    data : np.ndarray
        2D array of shape (n_samples, n_timesteps)
    window : int, optional
        Sakoe-Chiba band width for DTW constraint (None = no constraint)
    normalize : bool, optional
        If True, normalize distance by sqrt(path_length) (default: True)
    min_valid_points : int, optional
        Minimum non-NaN points required per sample (default: 30)
    
    Returns
    -------
    np.ndarray
        Distance matrix (n_samples, n_samples) where lower = more similar
    """
    n = data.shape[0]
    n_timesteps = data.shape[1]
    dtw_matrix = np.full((n, n), np.inf, dtype=np.float32)
    
    # Prepare data: interpolate NaN values for DTW
    data_filled = np.zeros_like(data)
    valid_samples = np.zeros(n, dtype=bool)
    
    for i in range(n):
        series = data[i, :]
        valid_mask = ~np.isnan(series)
        
        if valid_mask.sum() < min_valid_points:
            continue
        
        valid_samples[i] = True
        
        # Linear interpolation for NaN values
        if valid_mask.all():
            data_filled[i, :] = series
        else:
            x = np.arange(n_timesteps)
            data_filled[i, :] = np.interp(x, x[valid_mask], series[valid_mask])
    
    # Use dtaidistance if available (much faster with C + OpenMP)
    use_fast = DTW_FAST
    if use_fast:
        valid_indices = np.where(valid_samples)[0]
        valid_data = data_filled[valid_indices, :].astype(np.float64)
        
        try:
            # Use the fastest available method
            if DTW_C_AVAILABLE:
                # C implementation with OpenMP parallelization
                dtw_distances = dtw.distance_matrix_fast(
                    valid_data, 
                    window=window,
                    use_pruning=True,
                    parallel=True,
                    compact=False  # Return full matrix
                )
            else:
                # Pure Python fallback (still faster than manual)
                dtw_distances = dtw.distance_matrix(
                    valid_data, 
                    window=window,
                    use_pruning=True,
                    parallel=True
                )
            
            # Map back to full matrix
            for i_local, i_global in enumerate(valid_indices):
                for j_local, j_global in enumerate(valid_indices):
                    dist = dtw_distances[i_local, j_local]
                    if normalize and dist != np.inf:
                        dist = dist / np.sqrt(n_timesteps)
                    dtw_matrix[i_global, j_global] = dist
                    
        except Exception:
            use_fast = False
    
    # Fallback: slow pairwise computation
    if not use_fast:
        for i in range(n):
            if not valid_samples[i]:
                continue
            for j in range(i, n):
                if not valid_samples[j]:
                    continue
                
                dist = _dtw_distance(data_filled[i, :], data_filled[j, :], window=window)
                if normalize:
                    dist = dist / np.sqrt(n_timesteps)
                
                dtw_matrix[i, j] = dist
                dtw_matrix[j, i] = dist
    
    np.fill_diagonal(dtw_matrix, 0.0)
    return dtw_matrix


def compute_dtw_matrices_parallel(all_year_samples, sample_keys, indices, n_doy, 
                                   dry_season_mask=None, window=30, normalize=True,
                                   n_jobs=-1):
    """Compute DTW distance matrices for multiple indices in parallel.
    
    Parameters
    ----------
    all_year_samples : dict
        Dictionary mapping (sample_num, year) -> DataFrame with timeseries
    sample_keys : list
        List of (sample_num, year) keys
    indices : list
        List of index names to compute DTW for (e.g., ['NDVI', 'NDWI', 'NDMI'])
    n_doy : int
        Number of days in the growing year
    dry_season_mask : np.ndarray, optional
        Boolean mask for dry season filtering
    window : int
        Sakoe-Chiba band width for DTW
    normalize : bool
        Whether to normalize distances
    n_jobs : int
        Number of parallel jobs (-1 = all cores)
    
    Returns
    -------
    dict
        Dictionary mapping index_name -> distance_matrix
    """
    from joblib import Parallel, delayed
    
    n_samples = len(sample_keys)
    
    def compute_single_index(index_name):
        """Compute DTW matrix for a single index."""
        # Align timeseries to common DOY grid
        aligned_data = np.full((n_samples, n_doy), np.nan)
        
        for i, key in enumerate(sample_keys):
            df = all_year_samples[key]
            doy_values = df['doy'].values.astype(int)
            valid_mask = (doy_values >= 0) & (doy_values < n_doy)
            
            if index_name in df.columns:
                aligned_data[i, doy_values[valid_mask]] = df[index_name].values[valid_mask]
        
        # Apply dry season mask if provided
        if dry_season_mask is not None:
            aligned_data = aligned_data[:, dry_season_mask]
        
        # Compute DTW matrix
        dist_matrix = compute_dtw_matrix(aligned_data, window=window, normalize=normalize)
        
        # Handle inf/nan values
        n_inf = np.sum(~np.isfinite(dist_matrix))
        if n_inf > 0:
            finite_mask = np.isfinite(dist_matrix)
            max_finite = np.max(dist_matrix[finite_mask]) if np.any(finite_mask) else 1.0
            dist_matrix = np.where(finite_mask, dist_matrix, max_finite * 2)
        
        # Ensure symmetric and valid
        dist_matrix = np.maximum(dist_matrix, 0)
        np.fill_diagonal(dist_matrix, 0)
        dist_matrix = (dist_matrix + dist_matrix.T) / 2
        
        return index_name, dist_matrix.astype(np.float64)
    
    # If C library with OpenMP is available, dtaidistance handles parallelization internally
    # So we run indices sequentially to avoid oversubscription
    if DTW_C_AVAILABLE:
        results = [compute_single_index(idx) for idx in indices]
    else:
        # Use joblib for parallel computation across indices
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(compute_single_index)(idx) for idx in indices
        )
    
    return dict(results)


def _dtw_distance(s1, s2, window=None):
    """Compute DTW distance between two series.
    
    Uses Sakoe-Chiba band constraint if window is specified.
    
    Parameters
    ----------
    s1, s2 : np.ndarray
        Input timeseries (1D arrays)
    window : int, optional
        Sakoe-Chiba band width (None = no constraint)
    
    Returns
    -------
    float
        DTW distance (Euclidean)
    """
    n, m = len(s1), len(s2)
    dtw_cost = np.full((n + 1, m + 1), np.inf)
    dtw_cost[0, 0] = 0.0
    
    for i in range(1, n + 1):
        if window is not None:
            j_start = max(1, i - window)
            j_end = min(m + 1, i + window + 1)
        else:
            j_start, j_end = 1, m + 1
            
        for j in range(j_start, j_end):
            cost = (s1[i-1] - s2[j-1]) ** 2
            dtw_cost[i, j] = cost + min(
                dtw_cost[i-1, j],
                dtw_cost[i, j-1],
                dtw_cost[i-1, j-1]
            )
    
    return np.sqrt(dtw_cost[n, m])


def _prepare_timeseries_df(df: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted copy with normalized datetime column."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.sort_values("date").reset_index(drop=True)


# =============================================================================
# Clustering Functions
# =============================================================================

def merge_small_clusters(labels, distance_matrix, min_size, verbose=True):
    """Merge clusters smaller than min_size into nearest larger cluster.
    
    Parameters
    ----------
    labels : np.ndarray
        Array of cluster labels
    distance_matrix : np.ndarray
        Pairwise distance matrix
    min_size : int
        Minimum cluster size
    verbose : bool
        Print merge information (default: True)
    
    Returns
    -------
    np.ndarray
        Updated labels with small clusters merged and renumbered from 0
    """
    labels = labels.copy()
    unique_labels = np.unique(labels)
    cluster_sizes = {c: np.sum(labels == c) for c in unique_labels}
    
    small_clusters = [c for c, size in cluster_sizes.items() if size < min_size]
    large_clusters = [c for c, size in cluster_sizes.items() if size >= min_size]
    
    if len(small_clusters) == 0:
        return labels
    
    if len(large_clusters) == 0:
        if verbose:
            print(f"  Warning: All clusters smaller than {min_size}, keeping largest")
        sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
        large_clusters = [c for c, _ in sorted_clusters[:max(2, len(sorted_clusters)//2)]]
        small_clusters = [c for c in unique_labels if c not in large_clusters]
    
    if verbose:
        print(f"  Merging {len(small_clusters)} small clusters (size < {min_size})...")
    
    # Merge each small cluster to nearest large cluster
    for small_c in small_clusters:
        small_indices = np.where(labels == small_c)[0]
        
        min_avg_dist = np.inf
        nearest_large = large_clusters[0]
        
        for large_c in large_clusters:
            large_indices = np.where(labels == large_c)[0]
            distances = distance_matrix[np.ix_(small_indices, large_indices)]
            avg_dist = np.mean(distances)
            
            if avg_dist < min_avg_dist:
                min_avg_dist = avg_dist
                nearest_large = large_c
        
        labels[labels == small_c] = nearest_large
        if verbose:
            print(f"    Cluster {small_c} ({len(small_indices)} samples) → Cluster {nearest_large}")
    
    # Renumber clusters consecutively from 0
    unique_remaining = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique_remaining)}
    return np.array([label_map[l] for l in labels])


def optimize_n_clusters(distance_matrix, feature_matrix, min_clusters, max_clusters, 
                        linkage_matrix, verbose=True):
    """Find optimal cluster count using multiple metrics with voting.
    
    Uses three metrics: Silhouette (precomputed distances), Calinski-Harabasz,
    and Davies-Bouldin (both use raw features). Returns the count that most
    metrics agree on, with silhouette as tie-breaker.
    
    Parameters
    ----------
    distance_matrix : np.ndarray
        Precomputed pairwise distance matrix
    feature_matrix : np.ndarray
        Raw feature matrix (n_samples, n_features) for CH and DB metrics
    min_clusters : int
        Minimum clusters to test
    max_clusters : int
        Maximum clusters to test
    linkage_matrix : np.ndarray
        Precomputed linkage matrix from scipy.cluster.hierarchy.linkage
    verbose : bool
        Print optimization progress (default: True)
    
    Returns
    -------
    optimal_n : int
        Optimal number of clusters
    results : dict
        Dictionary with scores and recommendations for each metric
    """
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
    from scipy.cluster.hierarchy import fcluster
    
    n_samples = distance_matrix.shape[0]
    cluster_range = list(range(min_clusters, min(max_clusters + 1, n_samples)))
    
    silhouette_scores = []
    calinski_scores = []
    davies_scores = []
    
    for n in cluster_range:
        labels = fcluster(linkage_matrix, n, criterion='maxclust') - 1
        
        sil = silhouette_score(distance_matrix, labels, metric='precomputed')
        ch = calinski_harabasz_score(feature_matrix, labels)
        db = davies_bouldin_score(feature_matrix, labels)
        
        silhouette_scores.append(sil)
        calinski_scores.append(ch)
        davies_scores.append(db)
        
        if verbose:
            print(f"    n={n}: silhouette={sil:.3f}  calinski={ch:.1f}  davies={db:.3f}")
    
    # Find best for each metric
    best_silhouette = cluster_range[np.argmax(silhouette_scores)]
    best_calinski = cluster_range[np.argmax(calinski_scores)]
    best_davies = cluster_range[np.argmin(davies_scores)]
    
    # Voting
    votes = {n: 0 for n in cluster_range}
    votes[best_silhouette] += 1
    votes[best_calinski] += 1
    votes[best_davies] += 1
    
    # Select winner (tie-break with silhouette)
    max_votes = max(votes.values())
    candidates = [n for n, v in votes.items() if v == max_votes]
    
    if len(candidates) == 1:
        optimal_n = candidates[0]
    else:
        candidate_sil = {n: silhouette_scores[cluster_range.index(n)] for n in candidates}
        optimal_n = max(candidate_sil, key=candidate_sil.get)
    
    if verbose:
        print(f"\n  Metric recommendations:")
        print(f"    Silhouette: {best_silhouette} clusters")
        print(f"    Calinski-Harabasz: {best_calinski} clusters")
        print(f"    Davies-Bouldin: {best_davies} clusters")
        print(f"  Selected {optimal_n} clusters ({votes[optimal_n]}/3 metrics agree)")
    
    results = {
        'cluster_range': cluster_range,
        'silhouette_scores': silhouette_scores,
        'calinski_scores': calinski_scores,
        'davies_scores': davies_scores,
        'best_silhouette': best_silhouette,
        'best_calinski': best_calinski,
        'best_davies': best_davies,
        'votes': votes,
        'optimal_n': optimal_n
    }
    
    return optimal_n, results


def plot_optimization_results(results, output_path):
    """Plot cluster optimization results from optimize_n_clusters.
    
    Parameters
    ----------
    results : dict
        Results dictionary from optimize_n_clusters
    output_path : str or Path
        Path to save the plot
    """
    import matplotlib.pyplot as plt
    
    cluster_range = results['cluster_range']
    optimal_n = results['optimal_n']
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Silhouette
    axes[0].plot(cluster_range, results['silhouette_scores'], 'bo-', linewidth=2, markersize=6)
    axes[0].axvline(results['best_silhouette'], color='red', linestyle='--', alpha=0.7)
    axes[0].axvline(optimal_n, color='green', linestyle='-', linewidth=2, alpha=0.5)
    axes[0].set_xlabel('Number of Clusters')
    axes[0].set_ylabel('Silhouette Score')
    axes[0].set_title(f"Silhouette (best: {results['best_silhouette']})")
    axes[0].grid(alpha=0.3)
    
    # Calinski-Harabasz
    axes[1].plot(cluster_range, results['calinski_scores'], 'go-', linewidth=2, markersize=6)
    axes[1].axvline(results['best_calinski'], color='red', linestyle='--', alpha=0.7)
    axes[1].axvline(optimal_n, color='green', linestyle='-', linewidth=2, alpha=0.5)
    axes[1].set_xlabel('Number of Clusters')
    axes[1].set_ylabel('Calinski-Harabasz Index')
    axes[1].set_title(f"Calinski-Harabasz (best: {results['best_calinski']})")
    axes[1].grid(alpha=0.3)
    
    # Davies-Bouldin
    axes[2].plot(cluster_range, results['davies_scores'], 'ro-', linewidth=2, markersize=6)
    axes[2].axvline(results['best_davies'], color='red', linestyle='--', alpha=0.7)
    axes[2].axvline(optimal_n, color='green', linestyle='-', linewidth=2, alpha=0.5)
    axes[2].set_xlabel('Number of Clusters')
    axes[2].set_ylabel('Davies-Bouldin Index')
    axes[2].set_title(f"Davies-Bouldin (best: {results['best_davies']})")
    axes[2].grid(alpha=0.3)
    
    fig.suptitle(f"Cluster Optimization - Selected: {optimal_n} ({results['votes'][optimal_n]}/3 agree)", 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# =============================================================================
# Timeseries Processing Functions
# =============================================================================

def gap_fill_timeseries(df, indices, start_date=None, end_date=None, method='linear'):
    """Gap-fill timeseries to continuous daily time step.
    
    Parameters
    ----------
    df : pd.DataFrame
        Timeseries with 'date' column
    indices : list of str
        Column names to gap-fill
    start_date : datetime-like, optional
        Start of date range (uses df min if None)
    end_date : datetime-like, optional
        End of date range (uses df max if None)
    method : str, optional
        Interpolation method (default: 'linear')
    
    Returns
    -------
    pd.DataFrame
        Gap-filled daily timeseries
    """
    df = _prepare_timeseries_df(df)
    
    start_date = pd.to_datetime(start_date or df['date'].min()).normalize()
    end_date = pd.to_datetime(end_date or df['date'].max()).normalize()
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    df_daily = pd.DataFrame({'date': date_range})
    df_daily = df_daily.merge(df[['date'] + indices], on='date', how='left')
    
    for idx in indices:
        if idx in df_daily.columns and df_daily[idx].notna().sum() >= 2:
            df_daily[idx] = df_daily[idx].interpolate(method=method, limit_direction='both').ffill().bfill()
    
    return df_daily


def split_by_growing_year(df, year_start_month=11, year_end_month=10, min_obs_per_year=30):
    """Split timeseries by growing year (e.g., Nov 2019 - Oct 2020).
    
    Parameters
    ----------
    df : pd.DataFrame
        Timeseries with 'date' column
    year_start_month : int
        Starting month of growing year (default: 11 = November)
    year_end_month : int
        Ending month of growing year (default: 10 = October)
    min_obs_per_year : int
        Minimum observations required per year (default: 30)
    
    Returns
    -------
    yearly_splits : dict
        {year: df_subset} where year is the ending year
    year_ranges : dict
        {year: (start_date, end_date)} full growing year date ranges
    """
    df = _prepare_timeseries_df(df)
    
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    # Assign growing year (year when season ends)
    df['growing_year'] = df['year']
    if year_start_month > year_end_month:
        df.loc[df['month'] >= year_start_month, 'growing_year'] = df['year'] + 1
    
    yearly_splits = {}
    year_ranges = {}
    
    for gy in sorted(df['growing_year'].unique()):
        year_df = df[df['growing_year'] == gy].copy()
        if len(year_df) >= min_obs_per_year:
            year_df_clean = year_df.drop(['year', 'month', 'growing_year'], axis=1)
            yearly_splits[gy] = year_df_clean
            
            if year_start_month > year_end_month:
                start = pd.Timestamp(year=gy-1, month=year_start_month, day=1)
                end = pd.Timestamp(year=gy, month=year_end_month, day=31)
            else:
                start = pd.Timestamp(year=gy, month=year_start_month, day=1)
                end = pd.Timestamp(year=gy, month=year_end_month, day=31)
            
            year_ranges[gy] = (start, end)
    
    return yearly_splits, year_ranges


def split_by_calendar_year(df, min_obs_per_year=DEFAULT_MIN_OBS_PER_YEAR):
    """Split timeseries by calendar year (Jan 1 - Dec 31).
    
    Parameters
    ----------
    df : pd.DataFrame
        Timeseries with 'date' column
    min_obs_per_year : int
        Minimum observations required per year (default: config or 20)
    
    Returns
    -------
    yearly_splits : dict
        {year: df_subset} where year is the calendar year
    year_ranges : dict
        {year: (start_date, end_date)} full year date ranges
    """
    df = _prepare_timeseries_df(df)
    
    df['year'] = df['date'].dt.year
    
    yearly_splits = {}
    year_ranges = {}
    
    for year in sorted(df['year'].unique()):
        year_df = df[df['year'] == year].copy()
        if len(year_df) >= min_obs_per_year:
            year_df_clean = year_df.drop(['year'], axis=1)
            yearly_splits[year] = year_df_clean
            
            start = pd.Timestamp(year=year, month=1, day=1)
            end = pd.Timestamp(year=year, month=12, day=31)
            year_ranges[year] = (start, end)
    
    return yearly_splits, year_ranges
