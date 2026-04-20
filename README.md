# THz_sim_application

THz time-domain simulation, fitting, and study workflows for arbitrary isotropic multilayer samples.

The repository is centered on two Google Colab friendly notebooks:

- `THzSim_User_Workflow.ipynb` for synthetic reference generation, arbitrary sample definition, parameter sweeps, study execution, and heatmap-based study inspection
- `THzFit_User_Workflow.ipynb` for measured reference/sample fitting with visible preprocessing, fit overlays, residuals, FFT comparisons, and saved setup snapshots

Both notebooks are now code-cell-driven. The user edits normal Python cells with inline explanations instead of a JSON-first interface.

## Notebooks

### THzSim User Workflow

- GitHub: [notebooks/THzSim_User_Workflow.ipynb](notebooks/THzSim_User_Workflow.ipynb)
- Colab: [Open THzSim_User_Workflow in Colab](https://colab.research.google.com/github/Podrimate/THz_sim_application/blob/main/notebooks/THzSim_User_Workflow.ipynb)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Podrimate/THz_sim_application/blob/main/notebooks/THzSim_User_Workflow.ipynb)

This notebook is organized into six main parts:

1. install and import
2. run setup plus uploaded or generated reference trace preview
3. sample stack definition plus time-domain and FFT preview against the reference
4. study setup, noise preview, runtime estimate, and saved setup snapshots
5. study execution with progress output
6. interactive-style study plotting by chosen x axis, y axis, and heatmap value

### THzFit User Workflow

- GitHub: [notebooks/THzFit_User_Workflow.ipynb](notebooks/THzFit_User_Workflow.ipynb)
- Colab: [Open THzFit_User_Workflow in Colab](https://colab.research.google.com/github/Podrimate/THz_sim_application/blob/main/notebooks/THzFit_User_Workflow.ipynb)  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Podrimate/THz_sim_application/blob/main/notebooks/THzFit_User_Workflow.ipynb)

This notebook walks through:

1. install and import
2. uploaded measured reference/sample selection and raw/prepared preview
3. measurement settings with visible angle and polarization controls
4. plain-Python sample-stack definition with built-in templates
5. fit controls and saved setup snapshots
6. fit execution plus overlay, residual, optical-response, and FFT comparison plots

## Google Colab

Inside Colab, the first notebook cell installs the package directly from `main`:

```python
!pip install --upgrade --force-reinstall --no-cache-dir --no-deps https://github.com/Podrimate/THz_sim_application/archive/refs/heads/main.zip
```

Recommended usage:

1. open one of the Colab links above
2. run the first install/import cell
3. if Colab is stale, use `Runtime -> Factory reset runtime`
4. run the remaining cells from top to bottom

## Saved Outputs

Each notebook run writes outputs under a timestamped folder using the run name you provide:

- synthetic studies write under `timestamp__run_name`
- measured fits write under `timestamp__run_name`

Saved artifacts include:

- machine-readable setup snapshots such as `fit_setup.json` and `study_setup.json`
- human-readable Python snapshots of the edited notebook settings
- trace CSV files
- fit summaries, study summaries, and manifests
- generated plots and heatmaps

## Local Install

```bash
pip install -e .
```
