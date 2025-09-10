# findhrAPI
`findhrAPI` is an open-source library designed to support **Fairness and Intersectional Non-Discrimination in Human Recommendation**. It provides three main capabilities:  
- **Fair ranking interventions**: apply fairness-aware ranking algorithms to improve group and intersectional fairness across multiple protected attributes.  
- **Fairness monitoring**: measure fairness across input, output, and outcome levels, with privacy support based on multi-party computation.  
- **Explainability**: provide factual and counterfactual explanations to help understand ranking and recommendation decisions.


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
You can navigate the documentation starting from [`docs/build/index.html`](./docs/build/index.html)

### How findhrAPI works
The library is organized into modular components:  
- **Data preprocessing**: standardized input formats and feature pipelines.  
- **Fair ranking**: fairness-aware interventions at training or inference stages.  
- **Risk monitoring**: fairness monitoring protocols using secure two-party computation.  
- **Explainability**: factual and counterfactual explanation methods, including SHAP- and DiCE-based techniques.  
You can find jupyter notebook examples of how to use the findhrAPI in the documentation or in the `docs/source/example_notebooks` directory

### Contributors
This software was developed within the **FINDHR project** by contributors from multiple institutions:  
- **MPI-SP**: Asia Biega, Changyang He, Matthias Juentgen  
- **UNIPI**: Antonio Mastropietro, Salvatore Ruggieri  
- **UvA**: Clara Rus  
- **Adevinta**: Anna Via, Didac Fortuny, Guillem Escriba 
- **RAND**: David Graus, Volodymyr Medentsiy  

### Future Work
- We plan to use numba for accelerating fairness preprocessing computations 
- We plan to publish on pypi 