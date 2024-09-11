# Graph Dataset for Relations between Node Types

This repository contains the code used to analyze a dataset of graph structures, focusing on relations between various node types. The dataset, including all necessary data files, labels, and PyTorch Geometric (PyG) objects, is available on Zenodo at the following link:

[Zenodo Dataset Link](https://zenodo.org/records/13741001)

## Dataset Overview

The dataset includes several JSON files representing graph data, CSV files containing labels, and PyTorch Geometric objects used for training and analyzing graph neural networks. Please refer to the Jupyter Notebook in the repo for a quick turorial on how to train a Heterogeneus Graph model using this datasets.

## Data Files

The dataset consists of three key components:

1. **Graph Data (JSON files)**: Contains the graph structure and node types for each dataset. Each JSON file represents one dataset with its corresponding node features and edges.
2. **Labels (CSV files)**: These files contain the labels associated with each graph for supervised learning tasks.
3. **PyG Objects**: Pre-processed PyTorch Geometric objects, ready for use in graph neural network models.

All these files are available in the Zenodo link provided above.

## How to Use

To download and use the dataset, follow these steps:

1. Visit the Zenodo dataset link: [https://zenodo.org/records/13741001](https://zenodo.org/records/13741001).
2. Download the entire dataset package or specific files you need.
3. Use the provided JSON files, CSV labels, and PyG objects in your graph-based machine learning models or data analysis.

### Code and Usage

You can use the code provided in this repository to load and analyze the dataset. For example, the following script shows how to load a dataset:

```python
from Type3 import Type3

# Load the dataset
dataset = Type3("path_to_root_folder","path_to_data_folder", "path_to_labels_folder", "your_dataset")