
"""
Google Earth Engine Utilities

Utilities for GEE authentication, data loading, and processing
for the Zambezi Irrigation Mapping project.
"""

import sys
import time
import random
import concurrent.futures
from pathlib import Path
import ee


# =============================================================================
# Rate Limit and Timeout Handling
# =============================================================================

class GEETimeoutError(Exception):
    """Raised when a GEE query exceeds the timeout."""
    pass


def _run_with_timeout(func, timeout_sec=15):
    """Run a function with a timeout using threads (works in any thread).
    
    Parameters
    ----------
    func : callable
        Function to execute
    timeout_sec : int
        Timeout in seconds
    
    Returns
    -------
    any
        Result of the function call
    
    Raises
    ------
    GEETimeoutError
        If the function doesn't complete within timeout_sec
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func)
        try:
            return future.result(timeout=timeout_sec)
        except concurrent.futures.TimeoutError:
            raise GEETimeoutError(f"Query timed out after {timeout_sec}s")


def _retry_with_backoff(func, max_retries=5, base_delay=3.0, max_delay=30.0, timeout_sec=None):
    """Execute a function with exponential backoff retry on transient errors.
    
    Handles both raised exceptions AND silent hangs via timeout.
    
    Parameters
    ----------
    func : callable
        Function to execute (should be a lambda or partial)
    max_retries : int
        Maximum number of retry attempts
    base_delay : float
        Initial delay in seconds
    max_delay : float
        Maximum delay cap in seconds
    timeout_sec : int | None
        Timeout per attempt in seconds (for silent hangs). If None, uses config.GEE_TIMEOUT_SEC
    
    Returns
    -------
    any
        Result of the function call
    
    Raises
    ------
    Exception
        Re-raises the last exception if all retries fail
    """
    last_exception = None
    
    if timeout_sec is None:
        timeout_sec = getattr(__config__, 'GEE_TIMEOUT_SEC', 15)

    for attempt in range(max_retries + 1):
        try:
            # Wrap with timeout to catch silent hangs
            return _run_with_timeout(func, timeout_sec)
        except (GEETimeoutError, Exception) as e:
            error_str = str(e).lower()
            # Check if it's a retryable error (rate limit, timeout, or our GEETimeoutError)
            is_rate_limit = 'too many requests' in error_str or 'rate' in error_str or '429' in error_str
            is_timeout = isinstance(e, GEETimeoutError) or 'timed out' in error_str or 'timeout' in error_str or 'deadline' in error_str
            
            if is_rate_limit or is_timeout:
                last_exception = e
                if attempt < max_retries:
                    # Exponential backoff with jitter
                    delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                    error_type = "Rate limited" if is_rate_limit else "Timeout"
                    print(f"  ⏳ {error_type}, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(delay)
                else:
                    raise  # Re-raise on final attempt
            else:
                raise  # Non-retryable errors propagate immediately
    
    raise last_exception

# Add project root to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
import __config__


# =============================================================================
# Asset Management Functions
# =============================================================================

def asset_exists(asset_id):
    """Check if a GEE asset exists.

    Parameters
    ----------
    asset_id : str
        Full asset path (e.g., 'projects/xxx/assets/folder/name')

    Returns
    -------
    bool
        True if asset exists, False otherwise
    """
    try:
        ee.data.getAsset(asset_id)
        return True
    except ee.EEException:
        return False


def delete_asset(asset_id):
    """Delete a GEE asset if it exists.

    Parameters
    ----------
    asset_id : str
        Full asset path

    Returns
    -------
    bool
        True if deleted, False if not found
    """
    try:
        ee.data.deleteAsset(asset_id)
        print(f"  🗑️  Deleted existing asset: {asset_id}")
        return True
    except ee.EEException:
        return False


def ensure_folder_exists(folder_path):
    """Create GEE folder asset hierarchy if it doesn't exist.

    Parameters
    ----------
    folder_path : str
        Full folder path (e.g., 'projects/xxx/assets/folder/subfolder')
    """
    parts = folder_path.split('/')
    # Build path incrementally, starting at index 4 to skip 'projects/xxx/assets'
    for i in range(4, len(parts) + 1):
        partial_path = '/'.join(parts[:i])
        if not asset_exists(partial_path):
            try:
                ee.data.createAsset({'type': 'Folder'}, partial_path)
                print(f"  📁 Created folder: {partial_path}")
            except ee.EEException as e:
                if 'already exists' not in str(e):
                    raise

DATASETS = getattr(__config__, "DATASETS", {})
S2_COLLECTION = DATASETS.get("sentinel2", "COPERNICUS/S2_SR_HARMONIZED")
CHIRPS_COLLECTION = DATASETS.get("chirps", "UCSB-CHG/CHIRPS/DAILY")


def initialize_earth_engine(project_id=None, force_auth=False, opt_url=None):
    """Initialize Google Earth Engine with authentication.

    Parameters
    ----------
    project_id : str, optional
        GEE project ID (uses config.PROJECT_ID if None)
    force_auth : bool, optional
        Force re-authentication (default: False)
    opt_url : str, optional
        Optional Earth Engine endpoint URL (uses config.GEE_OPT_URL if None)

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    project_id = project_id or __config__.PROJECT_ID
    opt_url = opt_url if opt_url is not None else getattr(__config__, 'GEE_OPT_URL', None)

    # Force authentication if requested
    if force_auth:
        try:
            ee.Authenticate(force=True, auth_mode='notebook')
            ee.Initialize(project=project_id, opt_url=opt_url)
            endpoint = f" ({opt_url})" if opt_url else ""
            print(f"✓ Earth Engine authenticated: {project_id}{endpoint}")
            return True
        except Exception as e:
            print(f"✗ Authentication failed: {e}")
            return _authenticate_alternative(project_id)

    # Try to initialize with existing credentials
    try:
        ee.Initialize(project=project_id, opt_url=opt_url)
        endpoint = f" ({opt_url})" if opt_url else ""
        print(f"✓ Earth Engine initialized: {project_id}{endpoint}")
        return True
    except Exception:
        # Try notebook authentication
        try:
            ee.Authenticate(auth_mode='notebook')
            ee.Initialize(project=project_id, opt_url=opt_url)
            endpoint = f" ({opt_url})" if opt_url else ""
            print(f"✓ Earth Engine authenticated: {project_id}{endpoint}")
            return True
        except Exception:
            return _authenticate_alternative(project_id, opt_url=opt_url)


def _authenticate_alternative(project_id, opt_url=None):
    """Alternative authentication using gcloud or manual process.

    Parameters
    ----------
    project_id : str
        GEE project ID

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    # Try gcloud authentication
    opt_url = opt_url if opt_url is not None else getattr(__config__, 'GEE_OPT_URL', None)
    try:
        ee.Authenticate(auth_mode='gcloud')
        ee.Initialize(project=project_id, opt_url=opt_url)
        print(f"✓ Authenticated using gcloud")
        return True
    except Exception:
        pass

    # Manual authentication instructions
    print("\n" + "="*60)
    print("MANUAL AUTHENTICATION REQUIRED")
    print("="*60)
    print("\nRun in terminal: earthengine authenticate")
    print("\nThen restart your session and try again.")
    print("="*60 + "\n")
    return False


# =============================================================================
# Sentinel-2 Processing Functions
# =============================================================================

def filter_by_months(collection, month_list):
    """Filter GEE ImageCollection by month indices across all years.
    
    Parameters
    ----------
    collection : ee.ImageCollection
        Image collection to filter
    month_list : list of int
        Month indices (1-12) to include
    
    Returns
    -------
    ee.ImageCollection
        Filtered collection
    """
    filters = [ee.Filter.calendarRange(month, month, 'month') for month in month_list]
    return collection.filter(ee.Filter.Or(*filters))


def mask_s2_clouds(img):
    """Mask clouds in Sentinel-2 SR images using QA60 band.
    
    Parameters
    ----------
    img : ee.Image
        Sentinel-2 SR image
    
    Returns
    -------
    ee.Image
        Cloud-masked image with preserved properties
    """
    qa = img.select('QA60')
    cloud_mask = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
    return img.updateMask(cloud_mask).copyProperties(img, ['system:time_start', 'system:index'])


def apply_cloud_score_plus_mask(img, threshold=0.60, cs_band='cs'):
    """Apply Cloud Score+ mask to Sentinel-2 image.
    
    Cloud Score+ provides per-pixel cloud probability scores.
    Higher scores = clearer pixels. Use after linkCollection().
    
    Parameters
    ----------
    img : ee.Image
        Sentinel-2 SR image linked with Cloud Score+ (has 'cs' band)
    threshold : float, optional
        Clear-sky threshold (0-1). Pixels >= threshold are kept. (default: 0.60)
    cs_band : str, optional
        Name of Cloud Score+ band (default: 'cs')
    
    Returns
    -------
    ee.Image
        Cloud-masked image with preserved properties
    
    Example
    -------
    >>> s2 = s2_raw.linkCollection(cs_plus, ['cs'])
    >>> s2_masked = s2.map(lambda img: apply_cloud_score_plus_mask(img, 0.60))
    """
    return img.updateMask(img.select(cs_band).gte(threshold))\
              .copyProperties(img, ['system:time_start', 'system:index'])


def scale_s2(img):
    """Scale Sentinel-2 reflectance values (divide by 10000).
    
    Parameters
    ----------
    img : ee.Image
        Sentinel-2 SR image
    
    Returns
    -------
    ee.Image
        Scaled image with preserved properties
    """
    return img.divide(10000).copyProperties(img, ['system:time_start', 'system:index'])


def add_s2_indices(img):
    """Add NDVI, NDWI, and NDMI indices to Sentinel-2 image.
    
    Parameters
    ----------
    img : ee.Image
        Sentinel-2 SR image with bands B2, B3, B4, B8, B11
    
    Returns
    -------
    ee.Image
        Image with added index bands and preserved properties
    """
    ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndwi = img.normalizedDifference(['B3', 'B8']).rename('NDWI')
    ndmi = img.normalizedDifference(['B8', 'B11']).rename('NDMI')
    return img.addBands([ndvi, ndwi, ndmi]).copyProperties(img, ['system:time_start', 'system:index'])


def load_hccp_crop_mask(region=None, threshold=60, verbose=True):
    """Load HCCP crop mask asset from GEE.
    
    The HCCP asset has b1 where higher values = higher crop probability.
    
    Parameters
    ----------
    region : ee.Geometry, optional
        Region to clip to
    threshold : int, optional
        Minimum b1 value for crop classification (default: 60)
    verbose : bool
        Print status messages
    
    Returns
    -------
    ee.Image
        Binary crop mask (1 = crop, 0 = non-crop)
    """
    hccp_path = getattr(__config__, 'HCCP_ASSET', None)
    if not hccp_path:
        raise ValueError("HCCP_ASSET not defined in config.py")
    
    # Load HCCP and create mask where b1 >= threshold is crop
    hccp = ee.Image(hccp_path)
    crop_mask = hccp.select('b1').gte(threshold)
    
    if region is not None:
        crop_mask = crop_mask.clip(region)
    
    if verbose:
        print(f"  ✓ Loaded HCCP crop mask (b1 >= {threshold}): {hccp_path}")
    
    return crop_mask


def load_budyko_rainfed_crop_mask(region=None, verbose=True):
    """Load Budyko Rainfed crop mask asset from GEE.
    
    The Budyko Rainfed asset has b1 where 1 = rainfed crops.
    
    Parameters
    ----------
    region : ee.Geometry, optional
        Region to clip to
    verbose : bool
        Print status messages
    
    Returns
    -------
    ee.Image
        Binary crop mask (1 = rainfed crop, 0 = non-crop)
    """
    budyko_path = getattr(__config__, 'BUDYKO_RAINFED_ASSET', None)
    if not budyko_path:
        raise ValueError("BUDYKO_RAINFED_ASSET not defined in config.py")
    
    # Load Budyko and create mask where b1 == 1 is rainfed crop
    budyko = ee.Image(budyko_path)
    crop_mask = budyko.select('b1').eq(1)
    
    if region is not None:
        crop_mask = crop_mask.clip(region)
    
    if verbose:
        print(f"  ✓ Loaded Budyko Rainfed crop mask (b1 = 1): {budyko_path}")
    
    return crop_mask


def load_training_crop_mask(mask_config, verbose=True):
    """Load crop mask for training data sampling based on config settings.
    
    Supports a single mask source:
    - 'annual_dw_landcover': Pixels classified as crop in ALL specified years
    
    Parameters
    ----------
    mask_config : dict
        Configuration dictionary from config.TRAINING_CROP_MASK containing:
        - 'source': str - mask type (see above)
        - For 'annual_dw_landcover':
            - 'asset_pattern': str - pattern with {year} placeholder
            - 'years': list - years to check
            - 'crop_class': int - Dynamic World crop class (typically 4)
    verbose : bool
        Print status messages
    
    Returns
    -------
    ee.Image
        Binary crop mask (selfMask applied - only crop pixels have values)
    """
    source = mask_config['source']
    
    if verbose:
        print(f"  Source: {source}")
    
    if source == 'annual_dw_landcover':
        # Load annual landcover assets and find pixels that are crops in ALL years
        asset_pattern = mask_config['asset_pattern']
        years = mask_config['years']
        crop_class = mask_config['crop_class']
        
        if verbose:
            print(f"  Years: {years}")
            print(f"  Crop class: {crop_class}")
            print(f"  Requirement: Crop in ALL {len(years)} years")
        
        # Load each year and create crop mask
        crop_masks_per_year = []
        for year in years:
            asset_id = asset_pattern.format(year=year)
            lc = ee.Image(asset_id)
            is_crop = lc.eq(crop_class)  # 1 where crop, 0 elsewhere
            crop_masks_per_year.append(is_crop)
            if verbose:
                print(f"    Loaded: {asset_id}")
        
        # Stack and sum - pixels that are crops in ALL years will have sum == len(years)
        crop_sum = ee.ImageCollection(crop_masks_per_year).sum()
        crop_mask = crop_sum.eq(len(years)).selfMask()
        
    else:
        raise ValueError(f"Unknown crop mask source: {source}. "
                        f"Use 'annual_dw_landcover'")
    
    return crop_mask


def load_sentinel2(region, start_date, end_date, cloud_threshold=20):
    """Load and preprocess Sentinel-2 SR Harmonized data.

    Parameters
    ----------
    region : ee.Geometry
        Study area
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    cloud_threshold : int, optional
        Maximum cloud cover percentage (default: 20)

    Returns
    -------
    ee.ImageCollection
        Cloud-masked and scaled Sentinel-2 collection
    """
    collection = (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_threshold))
        .map(mask_s2_clouds)
        .map(scale_s2)
    )

    return collection


def load_chirps(region, start_date, end_date):
    """Load CHIRPS daily precipitation data.

    Parameters
    ----------
    region : ee.Geometry
        Study area
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)

    Returns
    -------
    ee.ImageCollection
        CHIRPS precipitation collection
    """
    return (ee.ImageCollection(CHIRPS_COLLECTION)
           .filterBounds(region)
           .filterDate(start_date, end_date))


def add_month_property(image):
    """Add month property to GEE image.
    
    Parameters
    ----------
    image : ee.Image
        GEE image with system:time_start property
    
    Returns
    -------
    ee.Image
        Image with 'month' property added (1-12)
    """
    month = ee.Date(image.get('system:time_start')).get('month')
    return image.set('month', month)


def extract_timeseries_over_region(collection, geometry, band_name='precipitation', scale=5000, reducer='mean'):
    """Extract timeseries values over a region from image collection.
    
    Parameters
    ----------
    collection : ee.ImageCollection
        Image collection to extract from
    geometry : ee.Geometry
        Region to reduce over
    band_name : str, optional
        Band name to extract (default: 'precipitation')
    scale : int, optional
        Reduction scale in meters (default: 5000)
    reducer : str, optional
        Reduction method: 'mean', 'sum', 'median' (default: 'mean')
    
    Returns
    -------
    ee.FeatureCollection
        Features with date, timestamp, and extracted value properties
    """
    # Map reducer string to ee.Reducer
    reducer_map = {
        'mean': ee.Reducer.mean(),
        'sum': ee.Reducer.sum(),
        'median': ee.Reducer.median(),
        'min': ee.Reducer.min(),
        'max': ee.Reducer.max()
    }
    ee_reducer = reducer_map.get(reducer, ee.Reducer.mean())
    
    def extract_value(image):
        value = image.reduceRegion(
            reducer=ee_reducer,
            geometry=geometry,
            scale=scale,
            maxPixels=1e9
        ).get(band_name)
        
        return ee.Feature(None, {
            'date': image.date().format('YYYY-MM-dd'),
            'timestamp': image.date().millis(),
            band_name: value
        })
    
    return collection.map(extract_value)


def create_seasonal_true_color(region, year, season_months, cloud_threshold=20):
    """Create a true color composite for a specific season and year.
    
    Parameters
    ----------
    region : ee.Geometry
        Study area
    year : int
        Year for the composite
    season_months : list of int
        Month indices (1-12) defining the season
    cloud_threshold : int, optional
        Maximum cloud cover percentage (default: 20)
    
    Returns
    -------
    ee.Image
        Median true color composite (B4, B3, B2)
    """
    # Determine date range based on season months and year
    # If season crosses year boundary (e.g., Nov-Mar), handle appropriately
    if any(m <= 3 for m in season_months) and any(m >= 11 for m in season_months):
        # Season crosses year boundary
        start_date = f'{year-1}-01-01'
        end_date = f'{year}-12-31'
    else:
        start_date = f'{year}-01-01'
        end_date = f'{year}-12-31'
    
    # Load and preprocess Sentinel-2
    s2 = ee.ImageCollection(S2_COLLECTION)\
        .filterBounds(region)\
        .filterDate(start_date, end_date)\
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_threshold))\
        .map(mask_s2_clouds)\
        .map(scale_s2)
    
    # Filter to season months
    s2_season = filter_by_months(s2, season_months)
    
    # Create median composite for true color
    return s2_season.median().select(['B4', 'B3', 'B2'])


def load_worldcereal(region, product='temporarycrops', season=None):
    """Load ESA WorldCereal crop classification data.
    
    Parameters
    ----------
    region : ee.Geometry
        Study area for filtering
    product : str, optional
        WorldCereal product: 'temporarycrops', 'maize', 'wintercereals', 
        'springcereals', 'irrigation' (default: 'temporarycrops')
    season : str, optional
        Season filter: 'tc-annual', 'tc-wintercereals', 'tc-springcereals',
        'tc-maize-main', 'tc-maize-second' (default: None, uses all)
    
    Returns
    -------
    ee.Image
        Mosaic of WorldCereal classification (0=other, 100=detected)
    """
    dataset = ee.ImageCollection('ESA/WorldCereal/2021/MODELS/v100')\
        .filterBounds(region)\
        .filter(ee.Filter.eq('product', product))
    
    if season:
        dataset = dataset.filter(ee.Filter.eq('season', season))
    
    # Mosaic and mask "other" class (value 0)
    mosaic = dataset.mosaic()
    return mosaic.updateMask(mosaic.select('classification').neq(0))


def get_worldcereal_layers(region):
    """Get all ESA WorldCereal layers for a region.
    
    Parameters
    ----------
    region : ee.Geometry
        Study area for filtering
    
    Returns
    -------
    dict
        Dictionary with layer names mapping to ee.Image objects:
        - 'temporarycrops': Temporary crops classification
        - 'irrigation': Irrigation classification
        - 'maize': Maize classification
    """
    products = {
        'temporarycrops': {'color': 'ff0000', 'label': 'Temporary Crops'},
        'irrigation': {'color': '2d79eb', 'label': 'Active Irrigation'},
        'maize': {'color': 'ebc334', 'label': 'Maize'}
    }
    
    layers = {}
    for product, info in products.items():
        try:
            layers[product] = {
                'image': load_worldcereal(region, product=product),
                'color': info['color'],
                'label': info['label']
            }
        except Exception:
            pass  # Product may not be available for this region
    
    return layers


def load_asset(asset_id):
    """Load a Google Earth Engine asset (Image, ImageCollection, or FeatureCollection).
    
    Parameters
    ----------
    asset_id : str
        Full asset ID path (e.g., 'projects/username/assets/asset_name')
    
    Returns
    -------
    ee.Image, ee.ImageCollection, or ee.FeatureCollection
        The loaded asset
    """
    # Try to determine asset type and load accordingly
    try:
        # Try as FeatureCollection first (most common for boundaries/vectors)
        return ee.FeatureCollection(asset_id)
    except Exception:
        try:
            # Try as Image
            return ee.Image(asset_id)
        except Exception:
            # Try as ImageCollection
            return ee.ImageCollection(asset_id)



def extract_indices(collection, point, bands, scale=10, buffer_size=100, start_date=None, end_date=None):
    """Extract spectral index values from image collection at a point.
    
    Splits query into yearly chunks to avoid GEE timeouts on large date ranges.
    Uses aggregate_array() for efficient data transfer.
    
    Parameters
    ----------
    collection : ee.ImageCollection
        Collection with spectral indices (should NOT be pre-filtered by date)
    point : ee.Geometry.Point
        Location to extract values
    bands : list of str
        Band names to extract (e.g., ['NDVI', 'NDWI', 'NDMI'])
    scale : int, optional
        Pixel resolution in meters (default: 10)
    buffer_size : int, optional
        Buffer radius in meters (default: 100)
    start_date : str, optional
        Start date 'YYYY-MM-DD'. If None, uses config.START_DATE
    end_date : str, optional
        End date 'YYYY-MM-DD'. If None, uses config.END_DATE
    
    Returns
    -------
    list of tuples
        [(timestamp, val1, val2, ...), ...] for each image
    """
    import pandas as pd
    from datetime import datetime
    
    # Get date range from config if not specified
    if start_date is None:
        start_date = __config__.START_DATE
    if end_date is None:
        end_date = __config__.END_DATE
    
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    
    def extract_val(img):
        values = img.select(bands).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point.buffer(buffer_size),
            scale=scale,
            bestEffort=True,
            maxPixels=1e4
        )
        props = {'time': img.get('system:time_start')}
        for band in bands:
            props[band] = values.get(band)
        return ee.Feature(None, props)
    
    all_results = []
    
    # Process year by year to keep queries small
    for year in range(start_year, end_year + 1):
        year_start = f'{year}-01-01'
        year_end = f'{year}-12-31'
        
        # Filter collection for this year
        year_col = collection.filterDate(year_start, year_end)
        
        # Convert to FeatureCollection
        fc = ee.FeatureCollection(year_col.map(extract_val))
        
        # Build server-side dictionary
        all_properties = ['time'] + bands
        server_dict = ee.Dictionary({
            prop: fc.aggregate_array(prop) for prop in all_properties
        })
        
        try:
            # Single getInfo() call per year with retry
            arrays = _retry_with_backoff(lambda sd=server_dict: sd.getInfo())
            
            times = arrays.get('time', [])
            if not times:
                print(f"  Year {year}: no images")
                continue
            
            print(f"  Year {year}: {len(times)} images")
                
            # Build result tuples for this year - check array lengths match
            n_times = len(times)
            for i, t in enumerate(times):
                if t is not None:
                    # Safely get band values
                    vals = []
                    for band in bands:
                        band_arr = arrays.get(band, [])
                        if i < len(band_arr):
                            vals.append(band_arr[i])
                        else:
                            vals.append(None)
                    all_results.append((pd.Timestamp(t, unit='ms'), *vals))
                    
        except Exception as e:
            print(f"  ⚠️  Year {year} failed: {e}")
            continue
    
    return all_results


def extract_precip(collection, point, band_name='precipitation', scale=5566, buffer_size=1000):
    """Extract precipitation values from CHIRPS at a point.
    
    Parameters
    ----------
    collection : ee.ImageCollection
        CHIRPS precipitation collection
    point : ee.Geometry.Point
        Location to extract values
    band_name : str, optional
        Precipitation band name (default: 'precipitation')
    scale : int, optional
        Pixel resolution in meters (default: 5566 for CHIRPS)
    buffer_size : int, optional
        Buffer radius in meters (default: 1000)
    
    Returns
    -------
    list of tuples
        [(timestamp, precip_value), ...] for each day with non-null values
    """
    import pandas as pd
    
    def extract_val(img):
        precip_val = img.select(band_name).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point.buffer(buffer_size),
            scale=scale,
            bestEffort=True,
            maxPixels=1e4
        ).get(band_name)
        return img.set('precip_value', precip_val)
    
    with_vals = collection.map(extract_val)
    times = _retry_with_backoff(lambda: with_vals.aggregate_array('system:time_start').getInfo())
    precip_vals = _retry_with_backoff(lambda: with_vals.aggregate_array('precip_value').getInfo())
    
    return [(pd.Timestamp(t, unit='ms'), p) 
            for t, p in zip(times, precip_vals) if p is not None]


def extract_timeseries(collection, point, bands, scale=250, buffer_size=250):
    """Extract time series values for multiple bands from an image collection.
    
    Generic function for extracting WaPOR ET, interception, or other multi-band datasets.
    
    Parameters
    ----------
    collection : ee.ImageCollection
        Image collection to extract from
    point : ee.Geometry.Point
        Location to extract values
    bands : list of str or str
        Band name(s) to extract (e.g., ['L1_AETI_D', 'L1_I_D'] for WaPOR)
    scale : int, optional
        Pixel resolution in meters (default: 250 for WaPOR)
    buffer_size : int, optional
        Buffer radius in meters (default: 250)
    
    Returns
    -------
    list of tuples
        [(timestamp, val1, val2, ...), ...] for each image with non-null values
        Single band returns [(timestamp, value), ...]
    """
    import pandas as pd
    
    # Handle single band as string or list
    if isinstance(bands, str):
        bands = [bands]
    
    def extract_val(img):
        values = img.select(bands).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point.buffer(buffer_size),
            scale=scale,
            bestEffort=True,
            maxPixels=1e4
        )
        # Set properties for each band
        props = {f'{band}_value': values.get(band) for band in bands}
        return img.set(props)
    
    with_vals = collection.map(extract_val)
    times = _retry_with_backoff(lambda: with_vals.aggregate_array('system:time_start').getInfo())
    
    # Extract values for each band (with retry on each call)
    band_values = []
    for band in bands:
        vals = _retry_with_backoff(lambda b=band: with_vals.aggregate_array(f'{b}_value').getInfo())
        band_values.append(vals)
    
    # Filter out rows where all values are None
    results = []
    for t, *vals in zip(times, *band_values):
        if any(v is not None for v in vals):
            results.append((pd.Timestamp(t, unit='ms'), *vals))
    
    return results


# Visualization parameters
VIS_PARAMS = {
    'ndvi': {
        'min': 0,
        'max': 0.8,
        'palette': ['white', 'yellow', 'green', 'darkgreen']
    },
    'classification': {
        'min': 0,
        'max': 2,
        'palette': ['brown', 'blue', 'yellow']
    },
    'precipitation': {
        'min': 0,
        'max': 1000,
        'palette': ['white', 'lightblue', 'blue', 'darkblue']
    },
    'true_color': {
        'bands': ['B4', 'B3', 'B2'],
        'min': 0,
        'max': 0.3,
        'gamma': 1.4
    },
    'false_color': {
        'bands': ['B8', 'B4', 'B3'],
        'min': 0,
        'max': 0.5,
        'gamma': 1.4
    }
}
