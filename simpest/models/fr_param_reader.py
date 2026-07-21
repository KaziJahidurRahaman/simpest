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


def _load_json(file: str | Path, kind: str) -> dict:
    """Load a JSON parameter file, raising a clear error if it is missing.

    Args:
        file: Path to the JSON parameter file.
        kind: Human-readable label used in the error message (e.g. "Crop").

    Returns:
        The parsed JSON object.
    """
    path = Path(file)
    if not path.exists():
        raise FileNotFoundError(f"{kind} parameter file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def build_params_from_specs(specs: dict, param_class: str) -> Dict[str, Parameter]:
    """Build a ``{class_ParamName: Parameter}`` dict from a raw spec dict.

    The ``specs`` mapping is the JSON structure for a single crop / disease /
    fungicide type, i.e. ``{ParamName: {"value": .., "min": .., "max": ..,
    "calibration": bool}}``. This is the single place that translates a spec
    dict into :class:`Parameter` objects, shared by the file readers and by
    callers that already hold an (optionally edited) spec dict.

    Args:
        specs:       Mapping of parameter name to its specification dict.
        param_class: Parameter class tag ("crop", "disease", or "fungicide").

    Returns:
        Dictionary mapping ``f"{param_class}_{ParamName}"`` → Parameter.
    """
    result: Dict[str, Parameter] = {}
    for param_name, spec in specs.items():
        param = Parameter(param_class=param_class)

        # IsSplashBorne is the only boolean parameter (disease only).
        if param_name.lower() == "issplashborne":
            param.value_bool = bool(spec.get("value", 0))
            param.is_boolean = True
        else:
            param.value = float(spec.get("value", 0.0))
            param.minimum = float(spec.get("min", 0.0))
            param.maximum = float(spec.get("max", 1.0))

        param.calibration = "x" if spec.get("calibration", False) else ""
        result[f"{param_class}_{param_name}"] = param

    return result


def load_crop_specs(file: str | Path, crop_type: str = "wheat") -> dict:
    """Load the raw parameter spec dict for a single crop type.

    Args:
        file:      Path to the crop parameters JSON file.
        crop_type: Crop type key (e.g., "wheat", "rice"). Defaults to "wheat".

    Returns:
        Mapping of ``ParamName`` → spec dict for the requested crop type.
        Raises ValueError if ``crop_type`` is not present.
    """
    data = _load_json(file, "Crop")
    if crop_type not in data:
        raise ValueError(f"Crop type '{crop_type}' not found. Available: {list(data.keys())}")
    return data[crop_type]


def load_disease_specs(file: str | Path, disease_type: str) -> dict:
    """Load the raw parameter spec dict for a single disease type.

    Args:
        file:         Path to the disease parameters JSON file.
        disease_type: Disease type key (e.g., "septoria", "brown_rust").

    Returns:
        Mapping of ``ParamName`` → spec dict for the requested disease type.
        Raises ValueError if ``disease_type`` is not present.
    """
    data = _load_json(file, "Disease")
    if disease_type not in data:
        raise ValueError(f"Disease type '{disease_type}' not found. Available: {list(data.keys())}")
    return data[disease_type]


def load_fungicide_specs(file: str | Path, fungicide_type: str = "protectant") -> dict:
    """Load the raw parameter spec dict for a single fungicide type.

    Args:
        file:           Path to the fungicide parameters JSON file.
        fungicide_type: Fungicide type key (e.g., "protectant").

    Returns:
        Mapping of ``ParamName`` → spec dict for the requested fungicide type.
        Raises ValueError if ``fungicide_type`` is not present.
    """
    data = _load_json(file, "Fungicide")
    if fungicide_type not in data:
        raise ValueError(f"Fungicide type '{fungicide_type}' not found. Available: {list(data.keys())}")
    return data[fungicide_type]


def read_crop_parameters(file: str | Path, crop_type: str = "wheat") -> Dict[str, Parameter]:
    """Read crop parameters from crop_parameters.json.

    Args:
        file:      Path to the crop_parameters.json file.
        crop_type: Crop type key (e.g., "wheat", "rice"). Defaults to "wheat".

    Returns:
        Dictionary mapping 'crop_ParamName' → Parameter for the specified crop type.
        Raises ValueError if crop_type not found.
    """
    return build_params_from_specs(load_crop_specs(file, crop_type), "crop")


def read_disease_parameters(file: str | Path) -> Dict[str, Dict[str, Parameter]]:
    """Read all disease parameters from disease_parameters.json.

    Args:
        file: Path to the disease_parameters.json file.

    Returns:
        Dictionary mapping ``disease_type`` to a parameter dictionary keyed as
        ``disease_ParamName``.
    """
    data = _load_json(file, "Disease")
    return {
        disease_type: build_params_from_specs(disease_specs, "disease")
        for disease_type, disease_specs in data.items()
    }


def read_fungicide_parameters(file: str | Path, fungicide_type: str = "protectant") -> Dict[str, Parameter]:
    """Read fungicide parameters from fungicide_parameters.json.

    Args:
        file:           Path to the fungicide_parameters.json file.
        fungicide_type: Fungicide type key (e.g., "protectant"). Defaults to "protectant".

    Returns:
        Dictionary mapping 'fungicide_ParamName' → Parameter for the specified type.
        Raises ValueError if fungicide_type not found.
    """
    return build_params_from_specs(load_fungicide_specs(file, fungicide_type), "fungicide")
