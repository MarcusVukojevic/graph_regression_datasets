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
│   │   ├── relsc_h.py
│   │   ├── relsc_m.py
│   │   ├── h_models.py
│   │   ├── hg_models.py
│   │   └── early_stopping.py
│   ├── replicate paper results/
│   │   ├── relsc_m_experiments.py
│   │   ├── relsc_m.py
│   │   └── dataset_statistics/
│   └── build your own regression dataset/
│       ├── FA-AST_java.py
│       └── mini_java/
│           ├── MiniTest.java
│           └── fa_ast_output.json
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
from relsc_m import RelSCM

dataset = RelSCM(root="./data", project_name="rdf") # or dataset = RelSC(project_name="rdf")

split_idx = relsc_m_dataset1.get_idx_split()
train_loader = DataLoader(relsc_m_dataset1[split_idx['train']], batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(relsc_m_dataset1[split_idx['val']],   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(relsc_m_dataset1[split_idx['test']],  batch_size=batch_size, shuffle=False)
```

## Workflow 2: Replicate paper results (Type2)

Script:
- `src/replicate paper results/relsc_m_experiments.py`

Run from repo root, ensure to have all the libraries that are necessary:

```bash
PYTHONPATH="src/Tutorial - Start Here!" python3 "src/replicate paper results/relsc_m_experiments.py"
```

What it does:
- runs all configured RelSCM datasets,
- runs two RelSCM models (`HeteroGraphConv`, `HeteroTransformer`),
- evaluates multiple seeds,
- exports CSV tables (for example `Table3_TEST_MAE.csv`).

Static plots and stats used in the paper are in:
- `src/replicate paper results/dataset_statistics/`

## Workflow 3: Build your own regression dataset

Script:
- `src/build your own regression dataset/FA-AST_java.py`

`FA-AST_java.py` parses Java files and builds FA-AST graph payloads.

Run on your own Java folder:

```bash
python3.11 "src/build your own regression dataset/FA-AST_java.py" \
  --input-dir "/path/to/java/files" \
  --output-json "fa_ast_output.json"
```

Included mini example (already in repo):
- Input file: `src/build your own regression dataset/mini_java/MiniTest.java`
- Generated graph: `src/build your own regression dataset/mini_java/fa_ast_output.json`

Regenerate the included sample graph:

```bash
python3.11 "src/build your own regression dataset/FA-AST_java.py" \
  --input-dir "src/build your own regression dataset/mini_java" \
  --output-json "src/build your own regression dataset/mini_java/fa_ast_output.json"
```

Quick verification test:

```bash
python3.11 - <<'PY'
import json

path = "src/build your own regression dataset/mini_java/fa_ast_output.json"
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("graphs:", len(data))
assert len(data) > 0, "No graphs generated"

sample_key, sample_value = next(iter(data.items()))
payload, ast_len = sample_value
x, edge_index, edge_attr = payload
print("sample:", sample_key)
print("ast_len:", ast_len, "nodes:", len(x), "edges:", len(edge_index[0]), "edge_attr:", len(edge_attr))
assert len(x) == ast_len, "Node count mismatch"
assert len(edge_index[0]) == len(edge_attr), "Edge/attribute mismatch"
print("FA-AST smoke test: OK")
PY
```

## Common path note

Folder names under `src/` contain spaces. Quote paths in shell commands:

```bash
python3 "src/replicate paper results/relsc_m_experiments.py"
```

## Citation

Samoaa, P., Vukojevic, M., Haghir Chehreghani, M., & Longa, A. (2026). Broadening the Scope of Graph Regression: Introducing a Dataset with Multiple Representation Settings [Data set]. Zenodo. https://doi.org/10.5281/zenodo.18598713
