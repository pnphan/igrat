API Reference
============

This page provides detailed information about all functions and classes in the IGRA Toolkit.

Data Access
----------

.. autofunction:: igrat.download_station_file
.. autofunction:: igrat.read_station_data
.. autofunction:: igrat.read_station_locations
.. autofunction:: igrat.open_data
.. autofunction:: igrat.open_nc
.. autofunction:: igrat.open_df

Data Processing
-------------

.. autofunction:: igrat.filter_by_date_range
.. autofunction:: igrat.filter_variables
.. autofunction:: igrat.filter_stations
.. autofunction:: igrat.interp_data
.. autofunction:: igrat.interp_data_to_pressure_levels
.. autofunction:: igrat.get_availability
.. autofunction:: igrat.get_profile

Visualization
-----------

.. autofunction:: igrat.plot_station_map
.. autofunction:: igrat.plot_profile

Data Types
---------

The toolkit works with two main data formats:

1. Pandas DataFrame
   * Contains columns for each variable
   * Includes metadata in column names and attributes
   * Easy to manipulate and analyze

2. NetCDF Dataset
   * Multi-dimensional arrays for each variable
   * Includes metadata in attributes
   * Efficient for large datasets

Common Parameters
---------------

* ``station_id``: Station identifier (e.g., "USM00072520")
* ``file_type``: Output format ('df' or 'netcdf')
* ``start_date``: Start date for filtering (YYYY-MM-DD)
* ``end_date``: End date for filtering (YYYY-MM-DD)
* ``variables``: List of variables to include
* ``lat_range``: Tuple of (min_lat, max_lat)
* ``lon_range``: Tuple of (min_lon, max_lon)

Return Values
-----------

Most functions return either:

* A pandas DataFrame
* An xarray Dataset
* None (if an error occurs)

Error Handling
------------

Functions handle errors by:

1. Printing informative error messages
2. Returning None for failed operations
3. Providing detailed error information in exceptions 