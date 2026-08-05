# simpest

![simpest logo](imgs/simpest_logo.png)

[![PyPI](https://img.shields.io/pypi/v/simpest.svg)](https://pypi.python.org/pypi/simpest)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**simpest** is a Python package for coupled crop growth and plant disease/pest
simulation. It runs a [SIMPLACE](https://www.simplace.net)/Lintul5 crop growth
scenario, converts the daily crop trajectory into the inputs a disease and
fungicide simulation model expects, and runs that model — optionally
calibrating selected parameters against field reference observations — to
produce daily and seasonal outputs. The disease/pest simulation logic is
inspired by the [FraNchEstYN](https://github.com/GeoModelLab/FraNchEstYN)
model.

## The modelling pipeline

1. **Crop growth (SIMPLACE).** [`simpest.models.simplace`](models/simplace.md)
   configures and runs a SIMPLACE/Lintul5 scenario and reads back the daily
   trajectory (light interception, above-ground biomass, yield, and thermal
   time).
2. **Format conversion.** Helper functions in the same module reshape the
   SIMPLACE weather, management, and crop-model outputs into the tables the
   disease/fungicide simulation consumes.
3. **Disease, pest, and fungicide simulation.**
   [`simpest.models.franchestyn`](models/franchestyn.md) drives an hourly
   infection model and a daily SEIR (susceptible → latent → sporulating →
   dead) tissue-progression model, applies any scheduled fungicide
   treatments, and couples the epidemic back onto the crop through four
   damage mechanisms — light stealers, RUE reducers, assimilate sappers, and
   senescence accelerators.
4. **Calibration (optional).** A multi-start Nelder–Mead search
   ([`fr_optimizer`](models/fr_optimizer.md)) fits selected crop and/or
   disease parameters to reference observations by minimising RMSE.
5. **Outputs.** Daily simulation records and per-season summaries (AUDPC,
   yield loss, peak severity, weather aggregates) are returned as
   DataFrame-ready records and can be written to CSV.

## Package description

- Runs SIMPLACE scenarios and reads the resulting daily crop trajectory.
- Converts SIMPLACE outputs to the format expected by the disease/fungicide
  simulation.
- Runs the crop, disease, and fungicide simulation steps.
- Calibrates selected parameters using multi-start Nelder–Mead optimization.
- Exports daily simulation and seasonal summary outputs.

Core modules are available under [simpest/models](https://github.com/KaziJahidurRahaman/simpest/tree/main/simpest/models).

## Directory tree

```text
simpest/
|- docs/
|  |- examples/
|  |- index.md
|  |- installation.md
|  |- usage.md
|  |- simpest.md
|  |- common.md
|- simpest/
|  |- simpest.py
|  |- common.py
|  |- models/
|     |- simplace.py
|     |- franchestyn.py
|     |- fr_runner.py
|     |- fr_crop_model.py
|     |- fr_disease_model.py
|     |- fr_fungicide_model.py
|     |- fr_optimizer.py
|     |- fr_utilities.py
|     |- fr_data.py
|- tests/
|- mkdocs.yml
|- pyproject.toml
|- README.md
```

## Installation

```bash
pip install simpest
```

For development installs, use editable mode:

```bash
pip install -e .
```

See [Installation](installation.md) for the SIMPLACE/JVM prerequisites
needed for the crop growth stage.

## Quickstart

The disease/fungicide simulation stage takes plain pandas DataFrames, so it
can be exercised on its own once weather, management, crop-model, and
reference data are available (as CSVs or otherwise):

```python
import pandas as pd
from simpest.models.franchestyn import FranchestynConfig, run_franchestyn

weather_df = pd.read_csv("weather.csv")
management_df = pd.read_csv("management.csv")
crop_model_df = pd.read_csv("crop_model.csv")
ref_df = pd.read_csv("reference.csv")

config = FranchestynConfig(
    crop_type="wheat",
    disease_type="septoria",
    fungicide_type=None,
    site="indiana",
    variety="Generic",
    disease="thisDisease",
    is_calibration=False,
)

result = run_franchestyn(
    start_year=2018,
    end_year=2019,
    config=config,
    weather_df=weather_df,
    management_df=management_df,
    crop_model_df=crop_model_df,
    ref_df=ref_df,
)
print(result["outputs"]["summary"])
```

See the example notebooks at
[examples/1_Run Simpest.ipynb](examples/1_Run%20Simpest.ipynb) and
[examples/2_Plot.ipynb](examples/2_Plot.ipynb) for running SIMPLACE first and
converting its output into the DataFrames above.

## Quick Links

- [Installation](installation.md)
- [Usage](usage.md)
- [API Reference](simpest.md)
- [Examples](examples/1_Run%20Simpest.ipynb)
- [Contributing](contributing.md)
- [Changelog](changelog.md)

## Project

- License: MIT
- Source: <https://github.com/KaziJahidurRahaman/simpest>
- Documentation: <https://KaziJahidurRahaman.github.io/simpest>
