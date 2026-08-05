# Installation

## Stable release

To install simpest, run this command in your terminal:

```bash
pip install simpest
```

This is the preferred method to install simpest, as it will always install
the most recent stable release.

If you don't have [pip](https://pip.pypa.io) installed, this
[Python installation guide](http://docs.python-guide.org/en/latest/starting/installation/)
can guide you through the process.

## From sources

To install simpest from sources, run this command in your terminal:

```bash
pip install git+https://github.com/KaziJahidurRahaman/simpest
```

For an editable, development install, clone the repository first:

```bash
git clone https://github.com/KaziJahidurRahaman/simpest
cd simpest
pip install -e .
```

## Dependencies

The base install pulls in `numpy`, `pandas`, `simplace`, `seaborn`,
`matplotlib`, and `scipy` (see `requirements.txt`). The disease/fungicide
simulation stage (`simpest.models.franchestyn`) only needs these — it reads
plain CSV/DataFrame inputs and has no further external requirements.

The crop growth stage (`simpest.models.simplace`) drives a
[SIMPLACE](https://www.simplace.net) installation through the `simplace`
Python package, which in turn bridges to the SIMPLACE Java runtime via
[`jpype`](https://jpype.readthedocs.io). To run that stage you additionally
need:

- A local SIMPLACE installation (`install_dir` in
  [`SimplaceConfig`](models/simplace.md)).
- A Java Runtime Environment compatible with your SIMPLACE installation.

If you only need the disease/fungicide simulation and calibration (for
example, working from crop-model output already produced elsewhere), a
working SIMPLACE installation is not required.

## Development extras

To work on simpest itself (docs, linting, tests), install the development
requirements:

```bash
pip install -r requirements_dev.txt
```

This includes `pytest`, `flake8`, `black`, `mkdocs`, `mkdocstrings`, and the
other tools used by the CI workflows.
