Getting Started
==============

This guide will help you get started with the IGRA Toolkit.

Basic Usage
----------

First, import the library:

.. code-block:: python

   import igrat

Finding Stations
--------------

To find available stations:

.. code-block:: python

   # Get all stations
   stations = igrat.read_station_locations()
   
   # Filter stations by region
   north_america_stations = igrat.filter_stations(
       lat_range=(30, 50),
       lon_range=(-130, -70)
   )
   
   # Visualize station locations
   igrat.plot_station_map()

Downloading Data
--------------

To download and read station data:

.. code-block:: python

   # Download data for a single station
   station = "USM00072520"  # Albany, NY
   df = igrat.read_station_data(station, file_type='df')
   
   # Download data for multiple stations
   stations = ["USM00072520", "USM00072456"]
   for station in stations:
       igrat.download_station_file(station)

Reading Data
-----------

To read downloaded data:

.. code-block:: python

   # Read as DataFrame
   df = igrat.open_data("USM00072520-main.csv")
   
   # Read as NetCDF
   ds = igrat.open_data("USM00072520-main.nc")

Basic Visualization
-----------------

To create basic plots:

.. code-block:: python

   # Plot temperature profile
   igrat.plot_profile(
       df,
       x_variable='temperature',
       y_variable='pressure',
       date='2020-01-01',
       time='12:00:00'
   )

Next Steps
---------

* Learn more about :doc:`data_access`
* Explore :doc:`data_processing` capabilities
* Check out :doc:`visualization` options
* See :doc:`analysis` techniques 