Basic Usage Examples
==================

This page provides examples of basic operations with the IGRA Toolkit.

Finding and Downloading Data
--------------------------

Find available stations and download their data:

.. code-block:: python

   import igrat
   
   # Get all stations
   stations = igrat.read_station_locations()
   print(f"Found {len(stations)} stations")
   
   # Find stations in North America
   na_stations = igrat.filter_stations(
       lat_range=(30, 50),
       lon_range=(-130, -70)
   )
   print(f"Found {len(na_stations)} stations in North America")
   
   # Download data for a specific station
   station = "USM00072520"  # Albany, NY
   df = igrat.read_station_data(station, file_type='df')
   print(f"Downloaded {len(df)} profiles")

Reading and Filtering Data
------------------------

Read and filter the downloaded data:

.. code-block:: python

   # Read the data
   df = igrat.open_data("USM00072520-main.csv")
   
   # Filter by date range
   filtered_df = igrat.filter_by_date_range(
       df,
       start_date="2020-01-01",
       end_date="2020-01-31"
   )
   
   # Filter variables
   variables = ['temperature', 'pressure', 'height']
   filtered_df = igrat.filter_variables(filtered_df, variables)

Basic Visualization
-----------------

Create simple visualizations:

.. code-block:: python

   # Plot station locations
   igrat.plot_station_map(
       color_by='elevation',
       lat_range=(30, 50),
       lon_range=(-130, -70)
   )
   
   # Plot a temperature profile
   igrat.plot_profile(
       df,
       x_variable='temperature',
       y_variable='pressure',
       date='2020-01-01',
       time='12:00:00',
       title='Temperature Profile'
   )

Data Processing
-------------

Process the data for analysis:

.. code-block:: python

   # Interpolate to standard pressure levels
   interpolated_df = igrat.interp_data_to_pressure_levels(
       df,
       variable='temperature'
   )
   
   # Get data availability
   availability = igrat.get_availability(df)
   print("Available dates:", list(availability.keys()))
   
   # Get a specific profile
   profile = igrat.get_profile(
       df,
       date='2020-01-01',
       time='12:00:00'
   ) 