# A Benchmark Dataset for Graph Regression

The repository is now organized around 3 workflows:
1. Learn and run examples: `src/Tutorial - Start Here!/`
2. Replicate paper results: `src/replicate paper results/`
3. Build a custom dataset: `src/build your own regression dataset/`

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18598713.svg)](https://doi.org/10.5281/zenodo.18598713)

## Repository map

```text
graph_regression_datasets/
├── src/
│   ├── Tutorial - Start Here!/
│   │   ├── Tutorial.ipynb
│   │   ├── Type1.py
│   │   ├── Type2.py
│   │   ├── h_models.py
│   │   ├── hg_models.py
│   │   └── early_stopping.py
│   ├── replicate paper results/
│   │   ├── type_2_experiments.py
│   │   └── dataset_statistics/
│   └── build your own regression dataset/
│       ├── FA-AST_java.py
│       └── edge_index.py
├── requirements.txt
└── Readme.md
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:
- `torch-geometric` must match your installed PyTorch version.
- If you use dataset-building scripts, install extra dependencies:
  - `pip install javalang anytree`

## Dataset download

Dataset files are hosted on Zenodo:
- https://zenodo.org/records/18598713

Expected dataset folders:

```text
graph_regression_datasets/
├── data/
│   ├── rdf.json
│   ├── dubbo.json
│   ├── H2.json
│   ├── hadoop.json
│   ├── systemds.json
│   └── ossbuilds.json
└── y_labels/
    ├── y_rdf.csv
    ├── y_dubbo.csv
    ├── y_H2.csv
    ├── y_hadoop.csv
    ├── y_systemds.csv
    └── y_ossbuilds.csv
```

## Workflow 1: Start here (tutorial)

Main entry point:
- `src/Tutorial - Start Here!/Tutorial.ipynb`

Launch from repo root:

```bash
PYTHONPATH="src/Tutorial - Start Here!" jupyter lab "src/Tutorial - Start Here!/Tutorial.ipynb"
```

If your dataset folders are in repo root (`data/`, `y_labels/`), use these paths in loaders:
- `x_folder="data"`
- `y_folder="y_labels"`


Example:

```python
from Type2 import Type2

dataset = Type2(root="tmp", x_folder="data", y_folder="y_labels", file_name="rdf")
train_split = dataset.load_split("train")
val_split = dataset.load_split("val")
test_split = dataset.load_split("test")
```

## Workflow 2: Replicate paper results (Type2)

Script:
- `src/replicate paper results/type_2_experiments.py`

Run from repo root, ensure to have all the libraries that are necessary:

```bash
PYTHONPATH="src/Tutorial - Start Here!" python3 "src/replicate paper results/type_2_experiments.py"
```

What it does:
- runs all configured Type2 datasets,
- runs two Type2 models (`HeteroGraphConv`, `HeteroTransformer`),
- evaluates multiple seeds,
- exports CSV tables (for example `Table3_TEST_MAE.csv`).

Static plots and stats used in the paper are in:
- `src/replicate paper results/dataset_statistics/`

## Workflow 3: Build your own regression dataset

Scripts:
- `src/build your own regression dataset/FA-AST_java.py`
- `src/build your own regression dataset/edge_index.py`

`FA-AST_java.py` is the parser/graph builder for Java sources. Before running it:
- set your source directory path in the script (`dirname` variable),
- install `javalang` and `anytree`.

## Common path note

Folder names under `src/` contain spaces. Quote paths in shell commands:

```bash
python3 "src/replicate paper results/type_2_experiments.py"
```

## Citation

Samoaa, P., Vukojevic, M., Haghir Chehreghani, M., & Longa, A. (2026). Broadening the Scope of Graph Regression: Introducing a Dataset with Multiple Representation Settings [Data set]. Zenodo. https://doi.org/10.5281/zenodo.18598713
