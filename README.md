# simpest

![simpest logo](docs/imgs/simpest_logo.png)

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

## What it does

- Runs SIMPLACE crop growth scenarios and reads the resulting daily trajectory.
- Converts SIMPLACE weather, management, and crop-model outputs into the
  format the disease/fungicide simulation expects.
- Simulates daily crop growth, epidemiological (SEIR) disease dynamics, and
  fungicide efficacy, coupled through four damage mechanisms (light
  interception loss, radiation-use-efficiency reduction, assimilate loss, and
  accelerated senescence).
- Calibrates selected crop and disease parameters with a multi-start
  Nelder–Mead search against reference observations.
- Exports daily simulation records and per-season summaries (AUDPC, yield
  loss, peak severity, weather aggregates).

Core modules live under [`simpest/models`](simpest/models).

## Installation

```bash
pip install simpest
```

For an editable, development install:

```bash
git clone https://github.com/KaziJahidurRahaman/simpest
cd simpest
pip install -e .
```

Running the crop growth stage additionally requires a
[SIMPLACE](https://www.simplace.net) installation and the `simplace` Python
package (a Java/JVM bridge via `jpype`); see
[Installation](docs/installation.md) for details.

## Quickstart

The disease/fungicide simulation stage takes plain pandas DataFrames, so it
can be exercised on its own once you have weather, management, crop-model,
and reference data available (as CSVs or otherwise):

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

For the full pipeline — running SIMPLACE first and converting its output into
the DataFrames above — see the example notebooks at
[`docs/examples/1_Run Simpest.ipynb`](docs/examples/1_Run%20Simpest.ipynb) and
[`docs/examples/2_Plot.ipynb`](docs/examples/2_Plot.ipynb).

## Directory layout

```text
simpest/
|- docs/                # MkDocs documentation site and example notebooks
|- simpest/
|  |- common.py
|  |- simpest.py
|  |- models/
|     |- simplace.py         # SIMPLACE crop growth integration
|     |- franchestyn.py      # disease/fungicide simulation entry points
|     |- fr_runner.py        # end-to-end daily simulation loop
|     |- fr_crop_model.py    # daily crop growth + disease damage mechanisms
|     |- fr_disease_model.py # SEIR disease model
|     |- fr_fungicide_model.py
|     |- fr_optimizer.py     # multi-start Nelder-Mead calibration
|     |- fr_data.py          # input/output/parameter data structures
|     |- fr_*_reader.py      # weather, parameter, and reference data readers
|- tests/
|- mkdocs.yml
|- pyproject.toml
|- README.md
```

## Documentation

Full documentation, including the API reference, is published at
<https://KaziJahidurRahaman.github.io/simpest>. Locally:

- [Installation](docs/installation.md)
- [Usage](docs/usage.md)
- [Examples](docs/examples/1_Run%20Simpest.ipynb)
- [API Reference](docs/simpest.md)
- [Contributing](docs/contributing.md)
- [Changelog](docs/changelog.md)

## Contributing

Contributions are welcome — see [Contributing](docs/contributing.md) for how
to set up a development environment and the pull request process.

## Project

- License: MIT
- Source: <https://github.com/KaziJahidurRahaman/simpest>
- Documentation: <https://KaziJahidurRahaman.github.io/simpest>
