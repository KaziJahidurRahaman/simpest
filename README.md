# simpest

<p align="center">
  <img src="imgs/simpest.png" alt="simpest logo" width="480"/>
</p>

<p align="center">
  <a href="https://pypi.python.org/pypi/simpest"><img alt="PyPI" src="https://img.shields.io/pypi/v/simpest.svg"></a>
  <a href="https://anaconda.org/conda-forge/simpest"><img alt="Conda" src="https://img.shields.io/conda/vn/conda-forge/simpest.svg"></a>
  <a href="https://opensource.org/licenses/MIT"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue">
</p>

---

## Introduction

**simpest** is an open-source Python package that couples the
[SIMPLACE](https://www.simplace.net) crop modelling platform with the
**FraNchEstYN** foliar disease model to simulate crop–disease interactions
under realistic agronomic and weather scenarios.

The package automates the full modelling pipeline: running SIMPLACE to obtain
crop growth outputs, converting those outputs into the FraNchEstYN input
format, executing the disease model, and optionally calibrating model
parameters against field observations using a Nelder-Mead optimiser.

---

## Description

| Component | Purpose |
|---|---|
| `simpest.models.simplace` | Initialise and run SIMPLACE; export weather and management data |
| `simpest.models.franchestyn` | Orchestrate FraNchEstYN runs; save results and season summaries |
| `simpest.models.fr_runner` | Low-level hourly/daily simulation loop |
| `simpest.models.fr_crop_model` | Daily crop growth step (light interception, biomass, yield) |
| `simpest.models.fr_disease_model` | Hourly + daily SEIR foliar disease progression |
| `simpest.models.fr_fungicide_model` | Fungicide degradation, tenacity, and efficacy |
| `simpest.models.fr_optimizer` | Multi-start Nelder-Mead parameter calibration |
| `simpest.models.fr_utilities` | Shared biophysical helper functions |
| `simpest.models.fr_data` | Dataclasses for inputs, outputs, and parameters |

-   **License:** MIT
-   **Documentation:** <https://KaziJahidurRahaman.github.io/simpest>

---

## Project Structure

```
simpest/
├── simpest/
│   ├── __init__.py
│   ├── common.py
│   ├── simpest.py              ← top-level pipeline entrypoints
│   └── models/
│       ├── franchestyn.py
│       ├── simplace.py
│       ├── fr_runner.py
│       ├── fr_crop_model.py
│       ├── fr_disease_model.py
│       ├── fr_fungicide_model.py
│       ├── fr_optimizer.py
│       ├── fr_utilities.py
│       ├── fr_data.py
│       ├── fr_param_reader.py
│       ├── fr_reference_reader.py
│       ├── fr_weather_reader.py
│       ├── fr_crop_parameters.json
│       ├── fr_disease_parameters.json
│       └── fr_fungicide_parameters.json
├── docs/
│   └── examples/
│       └── simpest_workflow_example.ipynb
├── tests/
├── imgs/
├── pyproject.toml
└── README.md
```

---

## Installation

**From PyPI (stable):**

```bash
pip install simpest
```

**From source (latest):**

```bash
git clone https://github.com/KaziJahidurRahaman/simpest.git
cd simpest
pip install -e .
```

**Dependencies** (`numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `simplace`) are installed automatically.

---

## Quickstart

```python
from simpest.models.simplace import (
    SimplaceConfig, init_simplace, run_simplace, get_project_row,
    export_crop_model_data, convert_weather, build_management,
    merge_simplace_and_franchestyn,
)
from simpest.models.franchestyn import (
    FranchestynConfig, run_franchestyn,
    build_season_summary, save_season_summary_csv,
)

# 1. Configure and run SIMPLACE
sp_cfg = SimplaceConfig(
    install_dir="path/to/simplace/workspace/",
    work_dir="path/to/simplace/workspace/simplace_run/simulation/",
    output_dir="path/to/output/",
    solution_path="MySolution/solution/MySol.sol.xml",
    project_path="MySolution/project/MyProj.proj.xml",
)
shell = init_simplace(sp_cfg)
run_simplace(shell, sp_cfg, project_lines=[1])

# 2. Export SIMPLACE outputs to FraNchEstYN format
from pathlib import Path
import simplace
dirs = simplace.getSimplaceDirectories(shell)
work_root = Path(dirs["_WORKDIR_"])
output_root = Path(dirs["_OUTPUTDIR_"])
project_row = get_project_row(work_root, selected_line=1)

weather_path = convert_weather(work_root, output_root, project_row["location"])
management_path = build_management(output_root, project_row)
crop_model_path = export_crop_model_data(output_root, project_row)

# 3. Configure and run FraNchEstYN
fr_cfg = FranchestynConfig(
    reference_path="path/to/reference.csv",
    crop_type="wheat",
    disease_type="septoria",
    site="my_site",
    variety="Generic",
    disease="thisDisease",
    is_calibration=True,
    calibration_variable="disease",
)
result = run_franchestyn(
    weather_path=str(weather_path),
    management_path=str(management_path),
    start_year=int(project_row["startdate"].split(".")[-1]),
    end_year=int(project_row["enddate"].split(".")[-1]),
    config=fr_cfg,
    cropmodel_path=str(crop_model_path),
)

# 4. Save outputs
import pandas as pd
simulation_df = pd.DataFrame(result["outputs"]["simulation"])
summary_df = build_season_summary(simulation_df, site=fr_cfg.site, variety=fr_cfg.variety)
save_season_summary_csv(summary_df, output_root)
print("RMSE:", result["outputs"]["summary"].get("rmse"))
```

See [`docs/examples/simpest_workflow_example.ipynb`](docs/examples/simpest_workflow_example.ipynb) for a full end-to-end worked example.

---

## Developer Guide

**Set up a development environment:**

```bash
git clone https://github.com/KaziJahidurRahaman/simpest.git
cd simpest
conda create -n simpest python=3.11
conda activate simpest
pip install -e ".[all]"
pip install -r requirements_dev.txt
```

**Run tests:**

```bash
pytest tests/
```

**Build documentation locally:**

```bash
pip install mkdocs mkdocs-material mkdocstrings[python] mkdocs-git-revision-date-localized-plugin
mkdocs serve
```

**Contributing:** see [CONTRIBUTING](docs/contributing.md) for guidelines.
