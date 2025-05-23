# IGRA (Integrated Global Radiosonde Archive) Toolkit Examples

This document provides comprehensive examples for using the IGRA toolkit to work with radiosonde data. The examples are organized by function and then by common use cases.

## Installation

### Requirements
- Python 3.7 or higher
- pip (Python package installer)

### Dependencies
The following packages are required:
- pandas
- numpy
- xarray
- netCDF4
- requests
- plotly
- matplotlib
- cartopy
- mplcursors

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/yourusername/igrat.git
cd igrat
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the package and its dependencies:
```bash
pip install -e .
```

Or install directly from PyPI:
```bash
pip install igrat
```

### Optional Dependencies
For additional functionality, you may want to install:
- jupyter (for interactive notebooks)
- seaborn (for enhanced plotting)
- scipy (for additional scientific computing functions)

Install optional dependencies with:
```bash
pip install jupyter seaborn scipy
```

## Table of Contents
1. [Basic Functions](#basic-functions)
   - [Downloading Data](#downloading-data)
   - [Reading Data](#reading-data)
   - [Opening Files](#opening-files)
2. [Data Processing](#data-processing)
   - [Filtering Data](#filtering-data)
   - [Interpolation](#interpolation)
3. [Visualization](#visualization)
   - [Station Maps](#station-maps)
   - [Vertical Profiles](#vertical-profiles)
4. [Common Use Cases](#common-use-cases)
   - [Climate Analysis](#climate-analysis)
   - [Weather Analysis](#weather-analysis)
   - [Research Applications](#research-applications)

## Basic Functions

### Downloading Data

#### Download a Single Station File
```python
# Download data for Albany, NY station
station = "USM00072520"
file_path = download_station_file(station)
print(f"Downloaded file: {file_path}")
```

#### Download Multiple Station Files
```python
# Download data for multiple stations
stations = ["USM00072520", "USM00072456", "USM00072469"]  # Albany, Buffalo, Syracuse
for station in stations:
    file_path = download_station_file(station)
    print(f"Downloaded {station}: {file_path}")
```

### Reading Data

#### Read Station Data as DataFrame
```python
# Read data for Albany, NY station as DataFrame
station = "USM00072520"
df = read_station_data(station, file_type='df')
print(df.head())

# Read with date filtering
start_date = datetime.datetime(2020, 1, 1)
end_date = datetime.datetime(2020, 1, 31)
df = read_station_data(station, 
                      start_date=start_date,
                      end_date=end_date,
                      file_type='df')
print(f"Data shape: {df.shape}")
```

#### Read Station Data as NetCDF
```python
# Read data for Albany, NY station as NetCDF
station = "USM00072520"
ds = read_station_data(station, file_type='netcdf')
print(ds.dims)

# Read with all variables
ds = read_station_data(station, main=False, file_type='netcdf')
print(ds.variables)
```

### Opening Files

#### Open NetCDF Files
```python
# Open a NetCDF file and print information
ds = open_nc("USM00072520-main.nc")
print(ds.dims)

# Open without printing information
ds = open_nc("USM00072520-main.nc", print_info=False)
```

#### Open DataFrame Files
```python
# Open a CSV file
df = open_df("USM00072520-main.csv")
print(df.head())

# Open an Excel file
df = open_df("USM00072520-main.xlsx")
print(df.describe())
```

#### Open Any File Type
```python
# Open any supported file type
data = open_data("USM00072520-main.nc")  # NetCDF
data = open_data("USM00072520-main.csv")  # CSV
data = open_data("USM00072520-main.xlsx")  # Excel
```

## Data Processing

### Filtering Data

#### Filter by Date Range
```python
# Filter DataFrame by date range
df = read_station_data("USM00072520", file_type='df')
filtered_df = filter_by_date_range(df, "2020-01-01", "2020-01-31", file_type='df')
print(f"Filtered data shape: {filtered_df.shape}")

# Filter NetCDF by date range
ds = read_station_data("USM00072520", file_type='netcdf')
filtered_ds = filter_by_date_range(ds, "2020-01-01", "2020-01-31", file_type='nc')
print(f"Filtered data shape: {filtered_ds.dims}")
```

#### Filter by Variables
```python
# Filter DataFrame to keep specific variables
df = read_station_data("USM00072520", file_type='df')
filtered_df = filter_variables(df, ['temperature', 'pressure'], file_type='df')
print(filtered_df.columns)

# Filter NetCDF to keep specific variables
ds = read_station_data("USM00072520", file_type='netcdf')
filtered_ds = filter_variables(ds, ['temperature', 'height'], file_type='nc')
print(list(filtered_ds.variables))
```

### Interpolation

#### Interpolate to Uniform Grid
```python
# Interpolate temperature to uniform height levels
df = read_station_data("USM00072520", file_type='df')
interpolated_df = interp_data(
    df,
    index_variable='height',
    variable='temperature',
    min_index=0,
    max_index=10000,
    step_size=100
)
print(interpolated_df['height'].unique())

# Interpolate wind speed to uniform pressure levels
ds = read_station_data("USM00072520", file_type='netcdf')
interpolated_ds = interp_data(
    ds,
    index_variable='pressure',
    variable='wind_speed',
    min_index=1000,
    max_index=100,
    step_size=10
)
print(interpolated_ds['pressure'].values)
```

#### Interpolate to Standard Pressure Levels
```python
# Interpolate temperature to standard pressure levels
df = read_station_data("USM00072520", file_type='df')
interpolated_df = interp_data_to_pressure_levels(
    df,
    variable='temperature'
)
print(interpolated_df['pressure'].unique())

# Interpolate multiple variables
ds = read_station_data("USM00072520", file_type='netcdf')
variables = ['temperature', 'wind_speed', 'relative_humidity']
for var in variables:
    interpolated_ds = interp_data_to_pressure_levels(ds, variable=var)
    print(f"{var} interpolated to standard pressure levels")
```

## Visualization

### Station Maps

#### Basic Station Map
```python
# Display all stations
plot_station_map()

# Display stations in North America
plot_station_map(
    lat_range=(30, 50),
    lon_range=(-130, -70)
)
```

#### Colored Station Maps
```python
# Color stations by elevation
plot_station_map(color_by='elevation')

# Color stations by last year of data
plot_station_map(color_by='last_year')

# Color stations by number of observations
plot_station_map(color_by='nobs')
```

### Vertical Profiles

#### Basic Profile Plot
```python
# Plot temperature profile
df = read_station_data("USM00072520", file_type='df')
plot_profile(
    df,
    x_variable='temperature',
    y_variable='height',
    date='2020-01-01',
    time='12:00:00'
)

# Plot wind speed profile
plot_profile(
    df,
    x_variable='wind_speed',
    y_variable='pressure',
    date='2020-01-01',
    time='12:00:00'
)
```

#### Customized Profile Plot
```python
# Plot with custom title and labels
plot_profile(
    df,
    x_variable='temperature',
    y_variable='height',
    date='2020-01-01',
    time='12:00:00',
    title='Temperature Profile - Albany, NY',
    xlabel='Temperature (°C)',
    ylabel='Height (m)',
    figsize=(12, 8)
)
```

## Common Use Cases

### Climate Analysis

#### Monthly Temperature Analysis
```python
# Get temperature data for a year
station = "USM00072520"
df = read_station_data(station, file_type='df')
df['month'] = pd.to_datetime(df['date']).dt.month

# Calculate monthly means at standard pressure levels
interpolated_df = interp_data_to_pressure_levels(df, variable='temperature')
monthly_means = interpolated_df.groupby('month')['temperature'].mean()
print(monthly_means)
```

#### Long-term Trend Analysis
```python
# Get 30 years of data
station = "USM00072520"
start_date = datetime.datetime(1990, 1, 1)
end_date = datetime.datetime(2020, 12, 31)
df = read_station_data(station, 
                      start_date=start_date,
                      end_date=end_date,
                      file_type='df')

# Interpolate to standard pressure levels
interpolated_df = interp_data_to_pressure_levels(df, variable='temperature')

# Calculate annual means
interpolated_df['year'] = pd.to_datetime(interpolated_df['date']).dt.year
annual_means = interpolated_df.groupby('year')['temperature'].mean()
print(annual_means)
```

### Weather Analysis

#### Sounding Analysis
```python
# Get a specific sounding
station = "USM00072520"
df = read_station_data(station, file_type='df')
profile = get_profile(df, '2020-01-01', '12:00:00')

# Plot multiple variables
variables = ['temperature', 'dewpoint_depression', 'wind_speed']
for var in variables:
    plot_profile(
        df,
        x_variable=var,
        y_variable='pressure',
        date='2020-01-01',
        time='12:00:00',
        title=f'{var.capitalize()} Profile'
    )
```

#### Storm Analysis
```python
# Get data before and during a storm
station = "USM00072520"
df = read_station_data(station, file_type='df')

# Plot profiles for different times
times = ['00:00:00', '06:00:00', '12:00:00', '18:00:00']
for time in times:
    plot_profile(
        df,
        x_variable='temperature',
        y_variable='pressure',
        date='2020-01-01',
        time=time,
        title=f'Temperature Profile at {time}'
    )
```

### Research Applications

#### Regional Climate Study
```python
# Get data for multiple stations in a region
stations = filter_stations(
    lat_range=(40, 45),
    lon_range=(-75, -70)
)

# Process each station
for station in stations:
    df = read_station_data(station, file_type='df')
    interpolated_df = interp_data_to_pressure_levels(df, variable='temperature')
    print(f"Processed {station}")
```

#### Vertical Structure Analysis
```python
# Get data for a station
station = "USM00072520"
df = read_station_data(station, file_type='df')

# Interpolate multiple variables to standard pressure levels
variables = ['temperature', 'relative_humidity', 'wind_speed']
for var in variables:
    interpolated_df = interp_data_to_pressure_levels(df, variable=var)
    print(f"Interpolated {var} to standard pressure levels")
```

#### Data Quality Analysis
```python
# Get data for a station
station = "USM00072520"
df = read_station_data(station, file_type='df')

# Check data availability
availability = get_availability(df)
print("Data availability by year:")
for year in sorted(availability.keys()):
    print(f"{year}: {len(availability[year])} months of data")
```

These examples demonstrate the versatility of the IGRA toolkit for various atmospheric science applications. The library can be used for both simple data access and complex analysis tasks. 