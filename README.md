# simpest

![simpest logo](/docs/imgs/simpest_logo.png)

[![PyPI](https://img.shields.io/pypi/v/simpest.svg)](https://pypi.python.org/pypi/simpest)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Introduction

**simpest** is a Python package for crop and disease simulation workflows.
It connects [SIMPLACE](https://www.simplace.net) outputs with the FraNchEstYN
model family so you can run end-to-end experiments from weather and management
data to simulation and summary results.

## Package Description

The package supports a practical modeling pipeline:

- Runs SIMPLACE scenarios
- Converts SIMPLACE outputs to FraNchEstYN-compatible inputs
- Runs crop, disease, and fungicide simulation steps
- Calibrates selected parameters using multi-start Nelder-Mead optimization
- Export daily simulation and seasonal summary outputs.

Core modules are available under [simpest/models](../simpest/models).

## Directory Tree

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

## Quickstart

```python
from simpest.models.simplace import SimplaceConfig, init_simplace, run_simplace
from simpest.models.franchestyn import FranchestynConfig, run_franchestyn

# 1) Configure and run SIMPLACE
sp_cfg = SimplaceConfig(
    install_dir="<SIMPLACE_INSTALL>",
    work_dir="<SIMPLACE_WORK>",
    output_dir="<OUTPUT_DIR>",
    solution_path="<SOLUTION_PATH>",
    project_path="<PROJECT_PATH>",
)
shell = init_simplace(sp_cfg)
run_simplace(shell, sp_cfg, project_lines=[1])

# 2) Configure and run FraNchEstYN
fr_cfg = FranchestynConfig(
    reference_path="<REFERENCE_CSV>",
    crop_type="wheat",
    disease_type="septoria",
    site="indiana",
    variety="Generic",
    disease="thisDisease",
)

result = run_franchestyn(
    weather_path="<WEATHER_FILE>",
    management_path="<MANAGEMENT_FILE>",
    start_year=2018,
    end_year=2019,
    config=fr_cfg,
)
print(result["outputs"]["summary"])
```

See the full workflow notebook at [docs/examples/simpest_workflow_example.ipynb](examples/simpest_workflow_example.ipynb).


## Quick Links

- [Installation](installation.md)
- [Usage](usage.md)
- [API Reference](simpest.md)
- [Examples](examples/intro.ipynb)
- [Contributing](contributing.md)
- [Changelog](changelog.md)

## Project

- License: MIT
- Source: <https://github.com/KaziJahidurRahaman/simpest>
- Documentation: <https://KaziJahidurRahaman.github.io/simpest>
