![VAMPyR logo](https://github.com/MRChemSoft/VAMPyR/raw/master/docs/gfx/logo.png)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4117602.svg)](https://doi.org/10.5281/zenodo.4117602)
[![License](https://img.shields.io/badge/license-%20LGPLv3-blue.svg)](../master/LICENSE)
![Build and test VAMPyR](https://github.com/MRChemSoft/vampyr/workflows/Build%20and%20test%20VAMPyR/badge.svg)
[![codecov](https://codecov.io/gh/MRChemSoft/vampyr/branch/master/graph/badge.svg)](https://codecov.io/gh/MRChemSoft/vampyr)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/MRChemSoft/vampyr/master?urlpath=lab%2Ftree%2Fdocs%2Fnotebooks)

The Very Accurate Multiresolution Python Routines (VAMPyR) package is a high
level Python interface to the [MRCPP](https://github.com/MRChemSoft/mrcpp) code.

## Installation

### Prerequisites

To build VAMPyR from source, you need the following tools installed on your system:
- **C++ Compiler** (supporting C++17, e.g., GCC 9+, Clang 10+, or MSVC 2019+)
- **CMake** (version 3.17 or higher)
- **Ninja** (optional, but recommended for faster builds)

Python dependencies (`numpy`, `matplotlib`) and the **MRCPP** backend are handled automatically by the build system.

### Using uv (Recommended)

The fastest and most robust way to install VAMPyR from source is using [uv](https://docs.astral.sh/uv/):

```bash
# Standard install (fetches MRCPP automatically)
uv pip install .

# Editable install for development
uv pip install -e .
```

### Using pip

VAMPyR uses a modern PEP 517 build system (`scikit-build-core`), so it can be installed with any modern pip version:

```bash
python -m pip install .
```

### Linking to a local MRCPP build

If you are developing both MRCPP and VAMPyR, you can link against a local MRCPP installation using config settings:

```bash
uv pip install -e . --config-settings=cmake.args="-DCMAKE_PREFIX_PATH=/path/to/mrcpp/install"
```

### Using Conda

[![Anaconda-Server Badge](https://anaconda.org/conda-forge/vampyr/badges/version.svg)](https://anaconda.org/conda-forge/vampyr)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/vampyr/badges/latest_release_date.svg)](https://anaconda.org/conda-forge/vampyr)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/vampyr/badges/downloads.svg)](https://anaconda.org/conda-forge/vampyr)

To install VAMPyR in a Conda environment `myenv`:

    $ conda create -n myenv
    $ conda activate myenv
    $ conda install -c conda-forge vampyr               # latest version (OpenMP)
    $ conda install -c conda-forge vampyr=0.1.0rc0      # tagged version (OpenMP)

To list all available versions:

    $ conda search -c conda-forge vampyr

Note that the conda-forge package is _always_ built with OpenMP support enabled
in the MRCPP backend.

The VAMPyR module is now available whenever you have activated the `myenv` environment.

### Creating a Conda environment from a .yml file

You can also create a Conda environment from a .yml file that already specifies VAMPyR and
other useful packages such as numpy, and matplotlib. Here's how:

1. Write an `environment.yml` file, for example:

    ```yaml
    name: myenv
    channels:
      - conda-forge
    dependencies:
      - vampyr
      - numpy
      - matplotlib
      - jupyterlab
    ```

2. Create the environment from the `environment.yml` file:

    ```sh
    $ conda env create -f environment.yml
    ```
3. Activate the environment:

    ```sh
    $ conda activate myenv
    ```
The VAMPyR module, along with numpy and matplotlib, is now available whenever
you have activated the myenv environment.
