# A Benchmark Dataset for Graph Regression with Homogeneous and Multi‑Relational Variants

Code and dataset loaders for graph regression experiments on software-system graphs.

**Important:** start from `Tutorial.ipynb`. The notebook is the main step-by-step guide for data loading, training, and evaluation.

## Dataset download

Dataset files are hosted on Zenodo:
- https://zenodo.org/records/13741001

Expected folders used by the scripts:

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

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Note: `torch-geometric` must be installed with versions compatible with your PyTorch build. If needed, follow the official PyG install guide for your platform.

## What is in this repository

- `Type1.py`: `Type1` dataset class for homogeneous graph processing.
- `Type2.py`: `Type2` dataset class for heterogeneous graph processing.
- `h_models.py`: homogeneous GNN models.
- `hg_models.py`: heterogeneous GNN models.
- `Tutorial.ipynb`: main tutorial.
- `type_2_experiments.py`: script to run all Type2 experiments and generate result tables.

## Quick usage examples

### Load a Type2 dataset

```python
from Type2 import Type2

# Example dataset: rdf
# root is used by PyG for processed files
# x_folder and y_folder point to data downloaded from Zenodo

dataset = Type2(root="tmp", x_folder="data", y_folder="y_labels", file_name="rdf")
train_split = dataset.load_split("train")
val_split = dataset.load_split("val")
test_split = dataset.load_split("test")
```

### Load a Type1 dataset

```python
from Type1 import Type1

dataset = Type1(root="tmp", x_folder="data", y_folder="y_labels", file_name="rdf")
```

## Run Type2 example experiments

Use:

```bash
python3 type_2_experiments.py
```

The script:
- runs all configured Type2 datasets,
- runs a subset of Type2 models (currently `HeteroGraphConv` and `HeteroTransformer`),
- uses multiple seeds,
- supports resume via `all_results_type2.pt`,
- writes final tables to CSV files (for example `Table3_TEST_MAE.csv`).

This script is provided to show the testing workflow we used; it is not an exhaustive runner for every model presented in the paper.

## Citation

If you use this repository or dataset, please cite:
- Zenodo dataset record: https://zenodo.org/records/13741001

You can export BibTeX directly from the Zenodo page via the "Cite" button.