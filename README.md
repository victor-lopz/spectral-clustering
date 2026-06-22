# Spectral clustering for Lagrangian particle flow analysis

> This repository explores a spectral clustering algorithm to discover coherent structures and patterns in dynamical systems.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/victor-lopz/spectral-clustering/actions/workflows/lint.yaml/badge.svg?branch=main)](https://github.com/victor-lopz/spectral-clustering/actions/workflows/lint.yaml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![GitHub last commit](https://img.shields.io/github/last-commit/SantanderAI/gen-fraud-graph)](https://github.com/SantanderAI/gen-fraud-graph/commits/main)

## Table of contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Repository structure](#repository-structure)

## Overview

The notebooks in this project show data generation, numerical integration of trajectories, construction of similarity matrices, spectral embedding, and clustering to identify coherent sets in flows. A coherent set or cluster is a group of alike trajectories.

## Requirements

Core (always installed):

- `Python` >= 3.14
- `NumPy` >= 1.24
- `SciPy` >= 1.18
- `Scikit-learn` >= 1.9
- `Matplotlib` >= 3.11

Optional:

- `pre-commit`: for running linters and formatters locally

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/victor-lopz/TFG
   cd your-repo-name
   ```

2. **Set up a virtual environment and install dependencies:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set up pre-commit hooks (optional):**

   ```bash
   pre-commit install
   ```

## Usage

- To reproduce the figures and experiments, open the notebooks in the `notebooks/` folder and run the cells in order using Jupyter.
- Notebooks:
  - [duffing_autonom.ipynb](notebooks/duffing_autonom.ipynb) — Autonomous Duffing flow experiments
  - [duffing_no_autonom.ipynb](notebooks/duffing_no_autonom.ipynb) — Non-autonomous experiments
  - [plot_trajectories.ipynb](notebooks/plot_trajectories.ipynb) — Plotting examples

## Repository structure

```text
├── .github/workflows/
│   └── lint.yaml       # CI workflow that lints and formats files
├── notebooks/          # Jupyter notebooks used for results and figures
│   ├── duffing_autonom.ipynb
│   ├── duffing_no_autonom.ipynb
│   └── plot_trajectories.ipynb
├── src/                # Library code used by the notebooks
│   ├── datatypes.py    # Dataclass definitions
│   ├── ode.py          # ODE system definitions
│   ├── plotting.py     # Plotting helper functions
│   ├── spectral.py     # Similarity, Laplacian, and spectral embedding routines
│   └── trajectories.py # Trajectory generation and processing
├── .pre-commit-config.yaml
├── README.md
└── requirements.txt
```
