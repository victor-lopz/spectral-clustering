# Spectral clustering for Lagrangian particle flow analysis

> Algorithm for discovering patterns in dynamical systems. Applications may include finding masses of air that move together in the atmosphere, detecting ocean currents, or identifying the boundary of the Antarctic ozone hole.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/victor-lopz/spectral-clustering/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/victor-lopz/spectral-clustering/actions/workflows/ci.yaml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![GitHub last commit](https://img.shields.io/github/last-commit/victor-lopz/spectral-clustering)](https://github.com/victor-lopz/spectral-clustering/commits/main)

## Table of contents

- [Overview](#overview)
- [Quickstart](#quickstart)
- [Notebooks](#notebooks)
- [Project structure](#project-structure)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## Overview

The code and notebooks show how to:

1. Generate multiple trajectories for dynamical systems (e.g., the Duffing oscillator).
2. Compute pairwise distances between trajectories.
3. Build the similarity matrix to see which pairs of trajectories are alike.
4. Build the graph Laplacian.
5. Compute spectral embeddings to reduce the dimensionality of the data.
6. Apply clustering with K-means to identify coherent sets.

A coherent set is a group of trajectories that evolve similarly through time.

## Quickstart

Prerequisites: `git` and `Python >= 3.14`. Recommended: [`uv`](https://docs.astral.sh/uv/). Install it with `pip install uv`.

1. **Clone the repository:**

    ```bash
    git clone https://github.com/victor-lopz/spectral-clustering.git
    cd spectral-clustering
    ```

2. **Set up a virtual environment:**

    ```bash
    uv venv                             # or: python -m venv .venv
    ```

    Activate the virtual environment

    ```bash
    source .venv/bin/activate           # macOS / Linux
    source .venv/Scripts/activate       # Windows (Bash)
    .venv\Scripts\Activate.ps1          # Windows (PowerShell)
    ```

3. **Install dependencies:**

    ```bash
    uv pip install -r requirements.txt   # or: pip install -r requirements.txt
    ```

4. **Run the notebooks** to reproduce experiments and figures.

## Notebooks

- [notebooks/autonomous_duffing.ipynb](notebooks/autonomous_duffing.ipynb) — autonomous Duffing experiments
- [notebooks/non_autonomous_duffing.ipynb](notebooks/non_autonomous_duffing.ipynb) — Time-dependent pertorbation added to the Duffing oscillator.
- [notebooks/plot_trajectories.ipynb](notebooks/plot_trajectories.ipynb) — plotting examples

## Project structure

```toml
spectral-clustering/
├── .github/workflows/                  # GitHub Actions workflows
│   └── ci.yaml                         # Lints and formats files
├── notebooks/                          # Jupyter notebooks
│   ├── autonomous_duffing.ipynb        # Autonomous Duffing experiments
│   ├── non_autonomous_duffing.ipynb    # Non-autonomous Duffing experiments
│   └── plot_trajectories.ipynb         # Plotting examples
├── output/                             # High-resolution figures
│   ├── autonomous/                     # Results for 90% sparsity
│   ├── autonom_local_maxs/             # Results for adaptive sparsity method
│   ├── non_autonomous/                 # Results for 90% sparsity
│   └── non_autonomous_local_maxs/      # Results for adaptive sparsity method
├── src/                                # Library code used by the notebooks
│   ├── __init__.py                     # Package initialization
│   ├── datatypes.py                    # Dataclass definitions
│   ├── ode.py                          # ODE system definitions
│   ├── plotting.py                     # Plotting helper functions
│   ├── spectral.py                     # Similarity and spectral embedding routines
│   └── trajectories.py                 # Path and distance computations
├── .editorconfig
├── .gitignore
├── .pre-commit-config.yaml             # Lints and formats files
├── .python-version
├── compose.yaml
├── Dockerfile
├── pyproject.toml
├── README.md
├── requirements.txt
└── uv.lock
```

## Development

- Install linters with `pre-commit`:

```bash
uv pip install pre-commit   # or: pip install pre-commit
pre-commit install
```

- Run linters and formatters:

```bash
pre-commit run --all-files
```

## Contributing

Contributions are welcome. Feel free to open issues or pull requests. If you plan to add features, please:

1. Open an issue to discuss the change.
2. Create a branch for your work.
3. Run `pre-commit run --all-files` before submitting a PR.

## License

This project is released under the Apache 2.0 License.

---
