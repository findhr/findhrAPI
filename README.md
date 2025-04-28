# findhrAPI
Working repository for the design of findhr APIs



### Installation

To use findhrAPI, first you need to:
- clone the repository
- create and activate a virtual environment (for example using conda) with python=3.11

```bash
    git clone https://github.com/findhr/findhrAPI.git
    conda create -n findhrAPI python=3.11
    conda activate findhrAPI
```

After that, you can install the package using pip:
- go to the src directory
- install the requirements from the file *requirements.txt*
- Install `wheel` to build the `findhr` package
- build the findhr package
- install the findhr package on the virtual environment

```bash
    cd ./src
    pip install -r requirements.txt
    pip install wheel
    python setup.py sdist bdist_wheel
    pip install dist/findhr-2.0.0-py3-none-any.whl
```

The requirements have been obtained using [pipreqs](https://pypi.org/project/pipreqs/).
We are updating the requirements file as we go along in the development process.

### Documentation
You can navigate the documentation starting from [`docs/build/html/index.html`](./docs/build/html/index.html)

### Examples
You can find jupyter notebook examples of how to use the findhrAPI in the documentation or in the `docs/source/example_notebooks` directory

### Future Work
- We plan to use numba for accelerating fairness preprocessing computations 
- We plan to publish on pypi 