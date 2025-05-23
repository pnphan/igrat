Installation
============

Requirements
-----------

* Python 3.7 or higher
* pip (Python package installer)

Dependencies
-----------

The following packages are required:

* pandas
* numpy
* xarray
* netCDF4
* requests
* plotly
* matplotlib
* cartopy
* mplcursors

Installation Methods
------------------

Using pip
~~~~~~~~

The simplest way to install IGRA Toolkit is using pip:

.. code-block:: bash

   pip install igrat

From Source
~~~~~~~~~~

To install from source:

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/pnphan/igrat.git
      cd igrat

2. Create a virtual environment (recommended):

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install the package and its dependencies:

   .. code-block:: bash

      pip install -e .

Optional Dependencies
-------------------

For additional functionality, you may want to install:

* jupyter (for interactive notebooks)
* seaborn (for enhanced plotting)
* scipy (for additional scientific computing functions)

Install optional dependencies with:

.. code-block:: bash

   pip install jupyter seaborn scipy

Development Installation
----------------------

For development, install with the development dependencies:

.. code-block:: bash

   pip install -e ".[dev]"

This will install additional packages needed for development, such as:

* pytest (for testing)
* black (for code formatting)
* flake8 (for linting)
* sphinx (for documentation)

Verifying Installation
--------------------

To verify the installation, try importing the package:

.. code-block:: python

   import igrat
   print(igrat.__version__)

You should see the version number printed without any errors.

Troubleshooting
--------------

Common installation issues and their solutions:

1. Missing Dependencies
   ~~~~~~~~~~~~~~~~~~~

   If you encounter missing dependency errors, try installing them manually:

   .. code-block:: bash

      pip install pandas numpy xarray netCDF4 requests plotly matplotlib cartopy mplcursors

2. Cartopy Installation Issues
   ~~~~~~~~~~~~~~~~~~~~~~~~~~

   Cartopy can be tricky to install. If you encounter issues:

   * On Ubuntu/Debian:
     .. code-block:: bash

        sudo apt-get install libgeos-dev libproj-dev
        pip install cartopy

   * On macOS:
     .. code-block:: bash

        brew install geos proj
        pip install cartopy

   * On Windows:
     Consider using Anaconda, which includes pre-built binaries for cartopy.

3. Virtual Environment Issues
   ~~~~~~~~~~~~~~~~~~~~~~~~~

   If you have issues with virtual environments:

   * Make sure you're using Python 3.7 or higher
   * Try creating a new virtual environment
   * Ensure you're activating the virtual environment before installing 