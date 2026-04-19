# THz_sim_application

THz time-domain simulation and fitting for arbitrary isotropic multilayer samples.

## Google Colab

The easiest entry point is the workflow notebook:

- [THzSim_User_Workflow.ipynb](notebooks/THzSim_User_Workflow.ipynb)

In Google Colab, the notebook will automatically install the package from this repository with:

```python
!pip install git+https://github.com/Podrimate/THz_sim_application.git
```

So the intended sharing flow is:

1. Open the notebook from GitHub in Colab.
2. Run the install cell.
3. Run the remaining workflow cells.

The notebook is structured to:

1. define the measurement/reference setup
2. define an arbitrary sample and study sweep
3. write the full setup into a single CSV file
4. rerun the study from that CSV alone
5. generate pairwise heatmaps for `normalized_mse`, `relative_l2`, and `true - fit`

## Local install

```bash
pip install -e .
```
