Contributing
===========

We welcome contributions to the IGRA Toolkit! This guide will help you get started.

Development Setup
---------------

1. Fork the repository
2. Clone your fork:

   .. code-block:: bash

      git clone https://github.com/your-username/igrat.git
      cd igrat

3. Create a virtual environment:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate

4. Install development dependencies:

   .. code-block:: bash

      pip install -e ".[dev]"

Code Style
---------

We follow these coding standards:

* Use Google-style docstrings
* Follow PEP 8 guidelines
* Use type hints
* Write unit tests for new features

Running Tests
-----------

Run the test suite:

.. code-block:: bash

   pytest

Check code style:

.. code-block:: bash

   black .
   flake8

Documentation
-----------

Build the documentation:

.. code-block:: bash

   cd docs
   make html

Pull Request Process
-----------------

1. Create a new branch for your feature
2. Write tests for your changes
3. Update documentation
4. Submit a pull request

Pull requests should:

* Have a clear description
* Include tests
* Update documentation
* Pass all CI checks

Code Review
---------

All submissions require review. We look for:

* Code quality
* Test coverage
* Documentation
* Performance impact

Getting Help
----------

* Open an issue for bugs
* Use discussions for questions
* Join our community chat 