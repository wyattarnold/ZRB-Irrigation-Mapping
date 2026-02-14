"""
Annual Wet Season Precipitation Analysis
Calculates total precipitation for each wet season (Nov-March)
Water year assignment: Nov 2021-March 2022 = Year 2022
"""

import ee
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import __config__
from src.gee_utils import (
    initialize_earth_engine, get_config, filter_by_months,
    add_month_property, extract_timeseries_over_region
)

# %% Initialize
initialize_earth_engine()

# %% Load configuration
study_area_config = __config__.get_study_area()
geometry = study_area_config['geometry']
study_area_name = __config__.CURRENT_STUDY_AREA

# Date range for analysis - 2017-2024
# Start from Nov 2016 to capture first full wet season (2017)
# End at March 2024 to capture last wet season (2024)
start_date = '2016-11-01'
end_date = '2024-03-31'

# Wet season months (Nov-March)
wet_months = __config__.WET_SEASON_MONTHS

print(f"Analyzing wet season precipitation for {study_area_name}")
print(f"Date range: {start_date} to {end_date}")
print(f"Wet season months: {wet_months}")

# %% Load CHIRPS precipitation data
chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
    .filterBounds(geometry) \
    .filterDate(start_date, end_date)

print(f"Total CHIRPS images: {chirps.size().getInfo()}")

# %% Add month property and filter to wet season
chirps_with_month = chirps.map(add_month_property)
wet_season_chirps = filter_by_months(chirps_with_month, wet_months)

print(f"Wet season images: {wet_season_chirps.size().getInfo()}")

# %% Extract precipitation time series
print("Extracting precipitation time series...")
precip_features = extract_timeseries_over_region(
    wet_season_chirps, 
    geometry, 
    band_name='precipitation',
    scale=5000,
    reducer='mean'
)
precip_data = precip_features.getInfo()

# %% Convert to pandas DataFrame
records = []
for feature in precip_data['features']:
    props = feature['properties']
    records.append({
        'date': props['date'],
        'precipitation': props['precipitation']
    })

df = pd.DataFrame(records)
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

print(f"Total daily records: {len(df)}")
print(f"\nData range: {df['date'].min()} to {df['date'].max()}")

# %% Assign water year (wet season year)
# Nov-Dec belong to the NEXT year's wet season
# Jan-Mar belong to the CURRENT year's wet season
def assign_water_year(row):
    if row['month'] >= 11:  # Nov, Dec
        return row['year'] + 1
    else:  # Jan, Feb, Mar
        return row['year']

df['water_year'] = df.apply(assign_water_year, axis=1)

# %% Calculate total precipitation per water year
annual_precip = df.groupby('water_year')['precipitation'].sum().reset_index()
annual_precip.columns = ['water_year', 'total_precip_mm']

# Calculate mean across all years
mean_annual_precip = annual_precip['total_precip_mm'].mean()

print(f"\n{'='*60}")
print("ANNUAL WET SEASON PRECIPITATION (mm)")
print(f"{'='*60}")
for _, row in annual_precip.iterrows():
    print(f"  {int(row['water_year'])}: {row['total_precip_mm']:.1f} mm")
print(f"{'='*60}")
print(f"  Mean: {mean_annual_precip:.1f} mm")
print(f"{'='*60}\n")

# %% Create line plot
fig, ax = plt.subplots(figsize=(12, 6))

# Plot line with markers
ax.plot(annual_precip['water_year'], 
        annual_precip['total_precip_mm'],
        color='steelblue',
        linewidth=2.5,
        marker='o',
        markersize=8,
        markerfacecolor='steelblue',
        markeredgecolor='white',
        markeredgewidth=1.5)

# Add mean line
ax.axhline(mean_annual_precip, 
           color='red', 
           linestyle='--', 
           linewidth=2,
           label=f'Mean: {mean_annual_precip:.1f} mm',
           alpha=0.7)

# Formatting
ax.set_xlabel('Water Year (Wet Season)', fontsize=12, fontweight='bold')
ax.set_ylabel('Total Precipitation (mm)', fontsize=12, fontweight='bold')
ax.set_title(f'Annual Wet Season Precipitation (2018-2024) - {study_area_name.replace("_", " ").title()}\n(Nov-March)', 
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=10)

# Add value labels at each point
for idx, row in annual_precip.iterrows():
    ax.text(row['water_year'], row['total_precip_mm'],
            f"{row['total_precip_mm']:.0f}",
            ha='center', va='bottom', fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor='gray', alpha=0.8))

# Set x-axis to show all years as integers
ax.set_xticks(annual_precip['water_year'])
ax.set_xticklabels([int(year) for year in annual_precip['water_year']])

plt.tight_layout()

# %% Save plot
output_dir = Path('output') / study_area_name / 'precip'
output_dir.mkdir(parents=True, exist_ok=True)

plot_path = output_dir / 'annual_wet_season_precipitation.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"✓ Plot saved: {plot_path}")

# %% Save data to CSV
csv_path = output_dir / 'annual_wet_season_precipitation.csv'
annual_precip.to_csv(csv_path, index=False)
print(f"✓ Data saved: {csv_path}")

plt.show()

print("\n✓ Analysis complete!")
