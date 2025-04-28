from setuptools import setup, find_packages
from codecs import open
from os import path

__author__ = "Asia Biega; Antonio Mastropietro; Salvatore Ruggieri; Clara Rus"
__license__ = "EUPL-1.2"
__email__ = "findhr-dev@llista.upf.edu"
VERSION = open('VERSION', 'r').read().strip()


here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "../README.md"), encoding="utf-8") as f:
    long_description = f.read()

with open(path.join(here, "requirements.txt"), encoding="utf-8") as f:
    requirements = f.read().splitlines()


setup(
    name="findhr",
    version=VERSION,
    license=__license__,
    description="",
    url="https://github.com/findhr/findhrAPI",
    author=__author__,
    author_email=__email__,
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 5 - Production/Stable",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        # Pick your license as you wish (should match "license" above)
        "License :: EUPL-1.2 License",
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
    ],
    keywords="fairness, bias, machine learning, data science, hiring, eXplainable AI",
    install_requires=[],#requirements,
    long_description=long_description,
    long_description_content_type="text/markdown",
    extras_require={"flag": []},
    packages=find_packages(
        exclude=[
            "*.test",
            "*.test.*",
            "test.*",
            "test",
            "package_name.test",
            "package_name.test.*",
        ]
    ),
)