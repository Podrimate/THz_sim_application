# THz_TDS_Fit_and_Study

THz time-domain simulation, fitting, and recovery-study workflows for arbitrary isotropic multilayer samples.

This repository is now organized around the parts you actually need day to day:

- one notebook for synthetic studies,
- one notebook for measured-data fitting,
- one package folder with the runtime code,
- one data folder with the test traces,
- one final lecture folder,
- one archive folder for older experiments and intermediate drafts.

## Start Here

If you want to run the project without digging through history, use these entry points:

- [notebooks/THzSim_User_Workflow.ipynb](notebooks/THzSim_User_Workflow.ipynb): build synthetic references, define stacks, run studies, and inspect heatmaps.
- [notebooks/THzFit_User_Workflow.ipynb](notebooks/THzFit_User_Workflow.ipynb): fit measured sample/reference pairs in the time domain.
- [docs/final_lecture/THzTDS_Lecture_Fit_Study_final.ipynb](docs/final_lecture/THzTDS_Lecture_Fit_Study_final.ipynb): final lecture notebook used for the lecture build and discussion.
- [docs/final_lecture/lecture_thz_tds_fit_study_notes_revised.tex](docs/final_lecture/lecture_thz_tds_fit_study_notes_revised.tex): final lecture-note source.
- [docs/final_lecture/lecture_thz_tds_fit_study_notes_revised.pdf](docs/final_lecture/lecture_thz_tds_fit_study_notes_revised.pdf): compiled lecture notes.

## Clean Workspace Layout

The visible workspace is meant to stay focused on the runnable project:

- `notebooks/`
  - active user notebooks only
  - `THzSim_User_Workflow.ipynb`
  - `THzFit_User_Workflow.ipynb`
- `thzsim2/`
  - the main package code for simulation, fitting, stacks, transfer functions, and notebook workflows
- `Test_data_for_fitter/`
  - local measured/reference data used by the workflows and tests
- `docs/final_lecture/`
  - final lecture notebook plus the final LaTeX/PDF notes
- `docs/lecture_build/`
  - a slim lecture-result snapshot kept only for the final lecture notes and example discussion
- `docs/lecture_assets_v2.py`
  - lecture build helper used to regenerate lecture assets
- `unimportant/`
  - archived notebook history, intermediate lecture drafts, temporary checks, and editor/context files

Older savecopies, deep-dive notebooks, reference note drafts, and intermediate rewrite sources have been moved under `unimportant/` so the top-level working surface stays small.

## Main Workflows

### 1. Synthetic study workflow

Use [notebooks/THzSim_User_Workflow.ipynb](notebooks/THzSim_User_Workflow.ipynb) when you want to:

- generate or upload a reference pulse,
- define an arbitrary stack,
- simulate transmission or reflection,
- run a recoverability study,
- inspect conductivity- or parameter-recovery heatmaps.

### 2. Measured fit workflow

Use [notebooks/THzFit_User_Workflow.ipynb](notebooks/THzFit_User_Workflow.ipynb) when you want to:

- load measured reference and sample traces,
- preprocess and crop them visibly,
- define the fit stack in Python,
- run the time-domain fit,
- inspect overlays, residuals, FFT comparisons, and fit summaries.

## Final Lecture Material

Everything intended as the final lecture package now lives in [docs/final_lecture](docs/final_lecture):

- final notebook,
- final revised lecture-note `.tex`,
- final compiled `.pdf`,
- generated lecture notebook/source files in `docs/final_lecture/generated/`.

The figures and study outputs referenced by the lecture notes stay in `docs/lecture_build/` because the lecture notebook and LaTeX both use that build tree directly. In this clean sibling repo, `docs/lecture_build/` is intentionally slimmed down to the measured examples and the selected study outputs that are actually used by the final lecture notes, so the branch stays practical to version and upload.

## Archived Material

The [unimportant](unimportant) folder is the deliberate holding area for:

- notebook savecopies,
- one-off deep dives,
- intermediate theory/results rewrite packages,
- reference-note drafts,
- temporary validation runs,
- editor and chat helper files.

That folder is meant to stay out of the way while still keeping the history available.

## Google Colab

The two main user notebooks are still Colab-friendly. Inside Colab, the first install cell can pull the package directly from `main`:

```python
!pip install --upgrade --force-reinstall --no-cache-dir --no-deps https://github.com/Podrimate/THz_sim_application/archive/refs/heads/main.zip
```

Recommended flow:

1. open one of the two user notebooks,
2. run the install/import cell,
3. reset the runtime if Colab is stale,
4. run the notebook top to bottom.

## Saved Outputs

Notebook runs write timestamped output folders under the configured run root. Typical saved artifacts include:

- `fit_setup.json` and `study_setup.json`,
- human-readable Python snapshots of the chosen settings,
- trace CSV files,
- fit and study summaries,
- generated plots and heatmaps.

## Local Install

```bash
pip install -e .
```
