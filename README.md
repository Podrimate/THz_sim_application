# THz_sim_application

THz time-domain simulation and fitting for arbitrary isotropic multilayer samples.

The repository now includes two Google Colab friendly user workflows:

- simulation and study generation
- measured reference/sample fitting

## Notebooks

### THzSim User Workflow

Study and simulation notebook for arbitrary samples, measurement setups, and parameter sweeps.

- GitHub: [notebooks/THzSim_User_Workflow.ipynb](notebooks/THzSim_User_Workflow.ipynb)
- Colab: [Open THzSim_User_Workflow in Colab](https://colab.research.google.com/github/Podrimate/THz_sim_application/blob/main/notebooks/THzSim_User_Workflow.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Podrimate/THz_sim_application/blob/main/notebooks/THzSim_User_Workflow.ipynb)

This notebook is structured to:

1. define the measurement and reference setup
2. define an arbitrary sample and study sweep
3. write the full setup into a single CSV file
4. rerun the study from that CSV alone
5. generate heatmaps for `normalized_mse`, `relative_l2`, and `true - fit`

### THzFit User Workflow

Measured-data fitting notebook for uploaded reference/sample traces in transmission or reflection.

- GitHub: [notebooks/THzFit_User_Workflow.ipynb](notebooks/THzFit_User_Workflow.ipynb)
- Colab: [Open THzFit_User_Workflow in Colab](https://colab.research.google.com/github/Podrimate/THz_sim_application/blob/main/notebooks/THzFit_User_Workflow.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Podrimate/THz_sim_application/blob/main/notebooks/THzFit_User_Workflow.ipynb)

This notebook is structured to:

1. upload or select measured reference and sample CSV files
2. preview raw traces and processed traces
3. define an arbitrary isotropic multilayer sample
4. fit sample and optional measurement parameters
5. inspect fitted traces, residuals, and recovered parameters

## Google Colab

Both notebooks install the package automatically from this repository when they run inside Colab:

```python
!pip install git+https://github.com/Podrimate/THz_sim_application.git
```

The sharing flow is:

1. open one of the Colab links above
2. run the install cell
3. run the remaining notebook cells

## Local install

```bash
pip install -e .
```
