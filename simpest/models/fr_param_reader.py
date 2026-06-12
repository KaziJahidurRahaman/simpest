"""Readers for model parameter definitions.

Parameters can be supplied either as a flat CSV table or as modular JSON files
organised by crop, disease, and fungicide type. Each reader returns a dictionary
keyed by ``"Class_ParamName"`` (for example ``"crop_TbaseCrop"``) whose values
are :class:`~simpest.models.fr_data.Parameter` objects carrying the numeric
value together with the calibration bounds.

Parameter definition CSV columns:

| Column | Field             | Notes                          |
|--------|-------------------|--------------------------------|
| 0      | Name              | Parameter class                |
| 1      | Class             | Parameter name                 |
| 2      | Description       | Ignored                        |
| 3      | Unit              | Ignored                        |
| 4      | Min               | Lower calibration bound        |
| 5      | Max               | Upper calibration bound        |
| 6      | Value             | Default value                  |
| 7      | CalibrationSubset | Calibration inclusion tag      |

Calibrated-output CSV columns are ``Name``, ``Class``, and ``Value``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict

from .fr_data import Parameter


def read(file: str | Path, calibration_variable: str = "") -> Dict[str, Parameter]:
    """Read a parameter definition CSV and return a dict keyed 'Name_Class'.

    Args:
        file:                 Path to the parameter CSV file.
        calibration_variable: Name of the variable being calibrated (currently
                              reserved for future use; not applied as a filter here).

    Returns:
        Dictionary mapping 'ParamName_ClassName' → Parameter.
    """
    result: Dict[str, Parameter] = {}
    path = Path(file)

    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        next(reader, None)  # skip header

        for row in reader:
            if not row or all(cell.strip() == "" for cell in row):
                continue
            # Ensure we have at least 8 columns
            while len(row) < 8:
                row.append("")

            model_class = row[0].strip()   # col 0: Model  (e.g., "crop", "disease")
            param_name = row[1].strip()    # col 1: Parameter (e.g., "TbaseCrop")
            raw_value = row[6].strip()

            param = Parameter(param_class=model_class)

            # IsSplashBorne is the only boolean parameter (values are "0" or "1")
            is_bool_param = param_name.lower() == "issplashborne" and raw_value.lower() in (
                "true", "false", "1", "0"
            )

            if is_bool_param:
                param.value_bool = raw_value.lower() in ("true", "1")
                param.is_boolean = True
            else:
                try:
                    param.value = float(raw_value)
                    param.minimum = float(row[4].strip())
                    param.maximum = float(row[5].strip())
                except ValueError:
                    # Skip malformed rows
                    continue

            param.calibration = row[7].strip() if len(row) > 7 else ""

            key = f"{model_class}_{param_name}"
            result[key] = param

    return result


def calibrated_read(file: str | Path) -> Dict[str, float]:
    """Read a calibrated output CSV and return a dict keyed 'Name_Class'.

    Args:
        file: Path to the calibrated parameters CSV. If the file does not
              exist, an empty dict is returned.

    Returns:
        Dictionary mapping 'ParamName_ClassName' → calibrated float value.
    """
    result: Dict[str, float] = {}
    path = Path(file)

    if not path.exists():
        return result

    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        next(reader, None)  # skip header

        for row in reader:
            if not row or len(row) < 3:
                continue
            name = row[0].strip()
            cls = row[1].strip()
            try:
                value = float(row[2].strip())
            except ValueError:
                continue
            result[f"{name}_{cls}"] = value

    return result


def read_by_crop(file: str | Path, crop_type: str = "wheat") -> Dict[str, Parameter]:
    """Read parameters from JSON file organized by crop type.

    Args:
        file:      Path to the parameters_by_crop.json file.
        crop_type: Crop type key (e.g., "wheat", "rice"). Defaults to "wheat".

    Returns:
        Dictionary mapping 'ParamName_ClassName' → Parameter for the specified crop.
        Returns empty dict if crop_type not found.
    """
    result: Dict[str, Parameter] = {}
    path = Path(file)

    if not path.exists():
        raise FileNotFoundError(f"Parameter file not found: {path}")

    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    # Get the crop section
    if crop_type not in data:
        raise ValueError(f"Crop type '{crop_type}' not found in parameter file. Available: {list(data.keys())}")

    crop_params = data[crop_type]

    # Iterate through model classes (crop, disease, fungicide)
    for model_class, params in crop_params.items():
        for param_name, param_dict in params.items():
            param = Parameter(param_class=model_class)

            # IsSplashBorne is the only boolean parameter
            if param_name.lower() == "issplashborne":
                param.value_bool = bool(param_dict.get("value", 0))
                param.is_boolean = True
            else:
                param.value = float(param_dict.get("value", 0.0))
                param.minimum = float(param_dict.get("min", 0.0))
                param.maximum = float(param_dict.get("max", 1.0))

            param.calibration = "x" if param_dict.get("calibration", False) else ""

            key = f"{model_class}_{param_name}"
            result[key] = param

    return result


def read_crop_parameters(file: str | Path, crop_type: str = "wheat") -> Dict[str, Parameter]:
    """Read crop parameters from crop_parameters.json.

    Args:
        file:      Path to the crop_parameters.json file.
        crop_type: Crop type key (e.g., "wheat", "rice"). Defaults to "wheat".

    Returns:
        Dictionary mapping 'crop_ParamName' → Parameter for the specified crop type.
        Raises ValueError if crop_type not found.
    """
    result: Dict[str, Parameter] = {}
    path = Path(file)

    if not path.exists():
        raise FileNotFoundError(f"Crop parameter file not found: {path}")

    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    if crop_type not in data:
        raise ValueError(f"Crop type '{crop_type}' not found. Available: {list(data.keys())}")

    crop_params = data[crop_type]

    for param_name, param_dict in crop_params.items():
        param = Parameter(param_class="crop")
        param.value = float(param_dict.get("value", 0.0))
        param.minimum = float(param_dict.get("min", 0.0))
        param.maximum = float(param_dict.get("max", 1.0))
        param.calibration = "x" if param_dict.get("calibration", False) else ""

        key = f"crop_{param_name}"
        result[key] = param

    return result


def read_disease_parameters(file: str | Path, disease_type: str) -> Dict[str, Parameter]:
    """Read disease parameters from disease_parameters.json.

    Args:
        file:         Path to the disease_parameters.json file.
        disease_type: Disease type key (e.g., "septoria", "brown_rust", "black_rust",
                      "yellow_rust", "wheat_blast").

    Returns:
        Dictionary mapping 'disease_ParamName' → Parameter for the specified disease type.
        Raises ValueError if disease_type not found.
    """
    result: Dict[str, Parameter] = {}
    path = Path(file)

    if not path.exists():
        raise FileNotFoundError(f"Disease parameter file not found: {path}")

    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    if disease_type not in data:
        raise ValueError(f"Disease type '{disease_type}' not found. Available: {list(data.keys())}")

    disease_params = data[disease_type]

    for param_name, param_dict in disease_params.items():
        param = Parameter(param_class="disease")

        # IsSplashBorne is a boolean parameter
        if param_name.lower() == "issplashborne":
            param.value_bool = bool(param_dict.get("value", 0))
            param.is_boolean = True
        else:
            param.value = float(param_dict.get("value", 0.0))
            param.minimum = float(param_dict.get("min", 0.0))
            param.maximum = float(param_dict.get("max", 1.0))

        param.calibration = "x" if param_dict.get("calibration", False) else ""

        key = f"disease_{param_name}"
        result[key] = param

    return result


def read_fungicide_parameters(file: str | Path, fungicide_type: str = "protectant") -> Dict[str, Parameter]:
    """Read fungicide parameters from fungicide_parameters.json.

    Args:
        file:           Path to the fungicide_parameters.json file.
        fungicide_type: Fungicide type key (e.g., "protectant"). Defaults to "protectant".

    Returns:
        Dictionary mapping 'fungicide_ParamName' → Parameter for the specified type.
        Raises ValueError if fungicide_type not found.
    """
    result: Dict[str, Parameter] = {}
    path = Path(file)

    if not path.exists():
        raise FileNotFoundError(f"Fungicide parameter file not found: {path}")

    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    if fungicide_type not in data:
        raise ValueError(f"Fungicide type '{fungicide_type}' not found. Available: {list(data.keys())}")

    fungicide_params = data[fungicide_type]

    for param_name, param_dict in fungicide_params.items():
        param = Parameter(param_class="fungicide")
        param.value = float(param_dict.get("value", 0.0))
        param.minimum = float(param_dict.get("min", 0.0))
        param.maximum = float(param_dict.get("max", 1.0))
        param.calibration = "x" if param_dict.get("calibration", False) else ""

        key = f"fungicide_{param_name}"
        result[key] = param

    return result
