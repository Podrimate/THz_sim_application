# THz_sim_application

THz time-domain simulation and fitting for arbitrary isotropic multilayer samples.

The repository now includes three Google Colab friendly user notebooks:

- simulation and study generation
- measured reference/sample fitting
- a visual guide that explains the main plots and outputs

## Notebooks

### THzSim User Workflow

Study and simulation notebook for arbitrary samples, measurement setups, and parameter sweeps.

- GitHub: [notebooks/THzSim_User_Workflow.ipynb](notebooks/THzSim_User_Workflow.ipynb)
- Colab: [Open THzSim_User_Workflow in Colab](https://colab.research.google.com/github/Podrimate/THz_sim_application/blob/main/notebooks/THzSim_User_Workflow.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Podrimate/THz_sim_application/blob/main/notebooks/THzSim_User_Workflow.ipynb)

This notebook is structured to:

1. define the measurement and reference setup
2. define an arbitrary sample and study sweep
3. write the setup into both `study_setup.json` and the legacy CSV file
4. rerun the study from the JSON file alone
5. generate heatmaps for `normalized_mse`, `relative_l2`, and `true - fit`

### THzFit User Workflow

Measured-data fitting notebook for uploaded reference/sample traces in transmission or reflection.

- GitHub: [notebooks/THzFit_User_Workflow.ipynb](notebooks/THzFit_User_Workflow.ipynb)
- Colab: [Open THzFit_User_Workflow in Colab](https://colab.research.google.com/github/Podrimate/THz_sim_application/blob/main/notebooks/THzFit_User_Workflow.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Podrimate/THz_sim_application/blob/main/notebooks/THzFit_User_Workflow.ipynb)

This notebook is structured to:

1. upload or select measured reference and sample CSV files
2. preview raw, aligned, and processed traces
3. define an arbitrary isotropic multilayer sample
4. export a reusable `fit_setup.json`
5. fit sample and optional measurement parameters
6. inspect fitted traces, residuals, and recovered parameters

### THz User Visual Guide

Visual explanation notebook for the preprocessing, fitting, and study plots.

- GitHub: [notebooks/THz_User_Visual_Guide.ipynb](notebooks/THz_User_Visual_Guide.ipynb)
- Colab: [Open THz_User_Visual_Guide in Colab](https://colab.research.google.com/github/Podrimate/THz_sim_application/blob/main/notebooks/THz_User_Visual_Guide.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Podrimate/THz_sim_application/blob/main/notebooks/THz_User_Visual_Guide.ipynb)

This notebook explains:

1. what the preprocessing plot means
2. what a good time-domain fit looks like
3. how to read the residual and optical response plots
4. how to interpret the study heatmaps

## Google Colab

All notebooks install the package automatically from this repository when they run inside Colab:

```python
!pip install --upgrade --force-reinstall --no-cache-dir https://github.com/Podrimate/THz_sim_application/archive/refs/heads/main.zip
```

The sharing flow is:

1. open one of the Colab links above
2. run the install cell
3. run the remaining notebook cells

If Colab has a stale runtime, use `Runtime -> Restart runtime` and then rerun the first cell.

## Review Files

The main review/share files are:

- `fit_setup.json` for measured fitting
- `study_setup.json` for synthetic studies

Those JSON files preserve the exact user choices needed to rerun and inspect a workflow.

## Local install

```bash
pip install -e .
```
