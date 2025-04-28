.. _usage:

Usage
=====

.. _installation:

Installation
------------

To install ``findhrAPI``, first you have to:

- clone the repository
- create and activate a virtual environment (for example using conda) with python=3.11

.. code-block:: console

    $ git clone https://github.com/findhr/findhrAPI.git
    $ conda create -n findhrAPI python=3.11
    $ conda activate findhrAPI

After that, you can install the package using ``pip``:

- go to the src directory
- install the requirements from the file *requirements.txt*
- Install `wheel` to build the `findhr` package
- build the findhr package
- install the findhr package on the virtual environment

.. code-block:: console

    $ cd ./src
    $ pip install -r requirements.txt
    $ pip install wheel
    $ python setup.py sdist bdist_wheel
    $ pip install dist/findhr-2.0.0-py3-none-any.whl

If you plan to use the *fairness* subpackage, you also need to install `R <https://www.r-project.org/>`_.

Please note that the requirements have been obtained using `pipreqs <https://pypi.org/project/pipreqs/>`_.
We are updating the requirements file as we go along in the development process.


Documentation
-------------
You can navigate the documentation starting from `docs/build/html/index.html`.

Examples
--------
We provide a number of ``jupyter`` notebook examples on how to use ``findhrAPI`` in the documentation (:ref:`example_notebooks`) or in the `docs/source/example_notebooks` directory.

Future Work
-----------
- We plan to use ``numba`` for accelerating the preprocessing fairness methods.
- We plan to publish the package on ``pypi``.
