Welcome to IGRA Toolkit's documentation!
=====================================

The IGRA Toolkit is a Python library designed to simplify working with the Integrated Global Radiosonde Archive (IGRA) data. It provides tools for downloading, processing, analyzing, and visualizing radiosonde observations.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   user_guide/index
   api_reference
   examples/index
   contributing
   changelog

Installation
-----------

To install the IGRA Toolkit, use pip:

.. code-block:: bash

   pip install igrat

For development installation:

.. code-block:: bash

   git clone https://github.com/pnphan/igrat.git
   cd igrat
   pip install -e ".[dev]"

Quick Start
----------

Here's a minimal example to get you started:

.. code-block:: python

   import igrat

   # Download and read data for a station
   station = "USM00072520"  # Albany, NY
   df = igrat.read_station_data(station, file_type='df')

   # Plot a temperature profile
   igrat.plot_profile(
       df,
       x_variable='temperature',
       y_variable='pressure',
       date='2020-01-01',
       time='12:00:00'
   )

   # Get station locations
   stations = igrat.read_station_locations()
   igrat.plot_station_map()

Features
--------

* Download and access IGRA station data
* Process and analyze radiosonde observations
* Visualize atmospheric profiles and station locations
* Interpolate data to standard pressure levels
* Quality control and data validation

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 