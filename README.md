# NESO calibration

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Tests](https://github.com/UCL/neso-calibration/actions/workflows/tests.yml/badge.svg)](https://github.com/UCL/neso-calibration/actions/workflows/tests.yml)
[![Linting](https://github.com/UCL/neso-calibration/actions/workflows/linting.yml/badge.svg)](https://github.com/UCL/neso-calibration/actions/workflows/linting.yml)
[![Documentation](https://github.com/UCL/neso-calibration/actions/workflows/docs.yml/badge.svg)](https://github-pages.ucl.ac.uk/neso-calibration/)
[![Licence][licence-badge]](./LICENCE.md)

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/UCL/neso-calibration/workflows/CI/badge.svg
[actions-link]:             https://github.com/UCL/neso-calibration/actions
[licence-badge]:            https://img.shields.io/badge/License-MIT-yellow.svg
<!-- prettier-ignore-end -->

The aim is for this repository to illustrate examples of calibrating
[Neptune Exploratory Software (NESO)](https://github.com/ExCALIBUR-NEPTUNE/NESO) models.

To facilitate this, this repository contains a Python package `nesopy` which provides a
simple interface for executing NESO solvers with specified parameter values and
extracting the generated outputs.

An example of using this interface to run and extract outputs from a NESO solver built and
installed in a Docker image is given in
[a tutorial notebook](examples/two_stream/docker_example.ipynb).

This project is developed in collaboration with the
[Centre for Advanced Research Computing](https://ucl.ac.uk/arc), University College London.

## About

### Project team

- Matt Graham ([matt-graham](https://github.com/matt-graham))
- Serge Guillas ([sguillas](https://github.com/sguillas))

### Research software engineering contact

Centre for Advanced Research Computing, University College London ([arc.collaborations@ucl.ac.uk](mailto:arc.collaborations.@ucl.ac.uk))

## Getting started

### Prerequisites

The `nesopy` package requires Python 3.10 or above. To use the NESO model wrapper
classes you will need to either locally build the required NESO solvers or build NESO
within a Docker image. If you wish to run the solver in parallel using MPI you will need
to build NESO against an appropriate MPI implementation.

### Installation

To install the latest development version of `nesopy` using `pip` run

```sh
pip install git+https://github.com/UCL/neso-calibration.git
```

Alternatively create a local clone of the repository with

```sh
git clone https://github.com/UCL/neso-calibration.git
```

and then install in editable mode by running

```sh
pip install -e .
```

from the root directory of your clone of the repository.

### Running tests

Tests can be run across all compatible Python versions in isolated environments using
[`tox`](https://tox.wiki/en/latest/) by running

```sh
tox
```

To run tests manually in a Python environment with `pytest` installed run

```sh
pytest tests
```

again from the root of the repository.

## Acknowledgements

This work was funded by a grant from the ExCALIBUR programme.
