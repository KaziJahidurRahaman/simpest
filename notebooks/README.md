# Notebooks

Demonstration notebooks for the `simpest` package, with the data they use.
The docs site renders them through the `docs/examples` link.

The notebooks only use the installed package: every model, input builder,
calibration and diagnostic comes from `import simpest`. To change what a
notebook does, change the package first and reinstall it (see
`docs/contributing.md`, "Package code and notebooks").

## Environment

Install the package, then the extra libraries the notebooks use:

```bash
pip install /path/to/simpest        # regular install; repeat after package changes
pip install jupyter torch torchcrop  # torchcrop is only used by the torchcrop notebooks
```

`simpest` itself does not depend on torch or torchcrop. The torchcrop
notebooks run torchcrop in their own cells and hand its output to simpest
through `simpest.models.fr_inputs`.

`1_Run Simpest.ipynb` and `2_Plot.ipynb` also need a SIMPLACE installation.

## Contents

| Notebook | What it shows |
|---|---|
| `1_Run Simpest.ipynb` | SIMPLACE crop growth, then FraNchEstYN disease simulation |
| `2_Plot.ipynb` | Plots of the outputs of `1_Run Simpest.ipynb` |
| `3_TorchCropIntegration.ipynb` | torchcrop crop growth feeding FraNchEstYN |
| `torchcrop-integration/indiana_potential_(winter_wheat)-calibration.ipynb` | Indiana winter wheat: potential yield vs observed yield, yield calibration, septoria calibration and diagnostics |
| `torchcrop-integration/indiana_potential_(winter_wheat).ipynb` | Earlier version of the notebook above, without the observed-yield sections |
| `Run Simpest - Copy.ipynb` | Backup copy of `1_Run Simpest.ipynb`, not rendered in the docs |
