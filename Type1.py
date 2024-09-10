import os
import json
import pandas as pd
import numpy as np
import networkx as nx
import math
import torch
from torch_geometric.data import InMemoryDataset, HeteroData, Data
from collections import defaultdict
from sklearn.model_selection import train_test_split


class Type1(InMemoryDataset):
    # Initialize the Type1 dataset, inheriting from InMemoryDataset
    def __init__(self, root, x_folder="/", y_folder="/", file_name=None, split=["train", "test", "val"]):
        self.split = split  # A list containing the dataset splits (train, test, val)
        self.x_folder = x_folder  # Folder containing input feature data (X)
        self.y_folder = y_folder  # Folder containing labels (Y)
        self.file_name = file_name  # List of dataset filenames
        
        # Initialize the parent class with the provided root path
        super(Type1, self).__init__(root)

        self.data_list = []  # Holds all processed data across splits
        self.split_data = {}  # Dictionary to store data split information

        # Load preprocessed data for each split (if they exist)
        for split in self.split:
            file_path = os.path.join(self.processed_dir, f"type1_{self.file_name}_{split}_data.pt")
            if os.path.exists(file_path):
                split_dataset = torch.load(file_path)  # Load the processed dataset for each split
                self.split_data[split] = split_dataset  # Store the split dataset
                self.data_list.extend(split_dataset)  # Add to the full dataset list
            else:
                self.split_data[split] = []  # If the dataset file doesn't exist, initialize with empty list

    # Method to load data for a specific split
    def load_split(self, split_name):
        if split_name in self.split_data:
            return self.split_data[split_name]  # Return the data for the given split
        else:
            return []  # If the split doesn't exist, return an empty list

    # Check if processed files exist for all splits
    def _processed_files_exist(self):
        return all(os.path.exists(os.path.join(self.processed_dir, f"type1_{self.file_name}_{split}_data.pt")) for split in self.split)

    # Property to define the raw filenames (used by the parent class)
    @property
    def raw_file_names(self):
        return self.file_name

    # Property to define the processed filenames for each split (used by the parent class)
    @property
    def processed_file_names(self):
        return [f"type1_{self.file_name}_{split}_data.pt" for split in self.split]

    # Method to return the length of the dataset
    def len(self):
        return len(self.data_list)

    # Method to retrieve a specific data item by index
    def get(self, idx):
        return self.data_list[idx]
    
    # Helper function to retrieve both dataset and corresponding labels
    def get_set_and_labels(self, keys, dataset, labels):
        data_set = {key: dataset[key] for key in keys}  # Extract data based on keys
        label_set = {key: labels[key] for key in keys if key in labels}  # Extract labels based on keys
        return data_set, label_set

    # Function to aggregate multiple edges and their attributes
    def aggregate_multi_edges(self, edge_index, edge_attr):
        if edge_attr is not None:
            unique_edges, inverse_indices = torch.unique(edge_index, dim=1, return_inverse=True)
            num_edges = unique_edges.size(1)
            aggregated_edge_attr = torch.zeros((num_edges, edge_attr.size(1)), dtype=torch.float)

            # Aggregate edge attributes by logical OR operation
            for i in range(len(inverse_indices)):
                aggregated_edge_attr[inverse_indices[i]] = torch.logical_or(
                    aggregated_edge_attr[inverse_indices[i]].bool(), edge_attr[i].bool()
                ).float()

            return unique_edges, aggregated_edge_attr
        return edge_index, edge_attr  # Return original edge index and attributes if no attributes are provided

    # Core processing method to transform raw files into PyTorch geometric data objects
    def process(self):

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)  # Create the processed directory if it doesn't exist
        
        data_list = []  # List to hold processed data

        
        file_path_x = os.path.join(self.x_folder, f"{self.file_name}.json") if self.x_folder != "/" else f"{self.file_name}.json"
        with open(file_path_x, 'r') as f:
            dataset = json.load(f)  # Load the dataset from JSON file

        # Load and normalize the labels from the corresponding CSV file
        file_path_y = os.path.join(self.y_folder, f"y_{self.file_name}.csv") if self.y_folder != "/" else f"y_{self.file_name}.csv"
        labels_file = pd.read_csv(file_path_y)
        labels_file['Value'] = labels_file['Value'] + 1  # Add 1 to the 'Value' column to avoid log(0)
        labels_file['LogValue'] = labels_file['Value'].apply(math.log)  # Apply logarithmic transformation
        max_val = labels_file['LogValue'].max()  # Find the maximum log value for normalization
        labels_file['NormalizedValue'] = labels_file['LogValue'] / max_val  # Normalize the log values
        labels = labels_file.set_index('Key')['NormalizedValue'].to_dict()  # Convert the labels into a dictionary

        # Sort the dataset keys and match them with the corresponding labels
        sorted_keys = sorted(dataset.keys())
        sorted_dataset = {key: dataset[key] for key in sorted_keys}
        sorted_labels = {key: labels[key] for key in sorted_keys if key in labels}

        # Split the dataset into train, validation (dev), and test sets
        train_keys, temp_keys = train_test_split(sorted_keys, train_size=0.7, shuffle=False)
        dev_keys, test_keys = train_test_split(temp_keys, test_size=0.5, shuffle=False)

        # Mapping of keys to dataset splits
        split_map = {}
        if "train" in self.split:
            split_map["train"] = train_keys
        if "val" in self.split:
            split_map["val"] = dev_keys
        if "test" in self.split:
            split_map["test"] = test_keys

        # Create dictionaries for the data and labels in each split
        result = {}
        for split in self.split:
            result[f"{split}_set"], result[f"{split}_labels"] = self.get_set_and_labels(split_map[split], sorted_dataset, sorted_labels)

        # Maximum number of nodes in the graphs
        max_nodes = 73
        all_edge_features = []

        # Extract all edge features across the dataset
        for file_name, graph_data in dataset.items():
            edge_features = graph_data[0][2]  # Get edge features from the graph data
            all_edge_features.extend(edge_features)

        # Find the unique edge features in the dataset
        unique_edge_features = np.unique(all_edge_features)
        num_edge_features = len(unique_edge_features)  # Count the number of unique edge features

        # Process the data for each split (train, validation, test)
        for split in self.split:
            split_data_list = []
            for file_name, graph_data in result[f"{split}_set"].items():
                num_nodes = graph_data[1]  # Number of nodes in the graph
                node_features = graph_data[0][0]  # Get node features from the graph data
                unique_node_features = np.arange(0, max_nodes+1)  # Generate possible node feature indices

                # Initialize node attributes as a zero matrix of shape (number of nodes, max_nodes)
                node_attr = torch.zeros((len(node_features), max_nodes), dtype=torch.float)

                adj_list = np.array(graph_data[0][1])  # Adjacency list (edge list)
                edge_features = graph_data[0][2]  # Edge features
                edge_index = torch.tensor(adj_list, dtype=torch.long).contiguous()  # Convert adjacency list to tensor

                # One-hot encode the edge features
                edge_attr = torch.zeros((len(edge_features), num_edge_features), dtype=torch.int32)
                for i, feature in enumerate(edge_features):
                    feature_index = np.where(unique_edge_features == feature[0])[0][0]
                    edge_attr[i, feature_index] = 1.0  # One-hot encoding

                # One-hot encode the node features
                for i, feature in enumerate(node_features):
                    feature_index = np.where(unique_node_features == feature[0])[0][0]
                    node_attr[i, feature_index] = 1.0  # One-hot encoding for nodes

                # Aggregate multiple edges if they exist
                edge_index, edge_attr = self.aggregate_multi_edges(edge_index, edge_attr)

                # Initialize a matrix to hold aggregated edge information for each node
                node_edge_info = torch.zeros((num_nodes, num_edge_features), dtype=torch.float)
                for i in range(edge_index.size(1)):
                    src_node = edge_index[0, i]
                    node_edge_info[src_node] += edge_attr[i]

                # Concatenate node features and aggregated edge information
                node_attr = torch.cat([node_attr, node_edge_info], dim=1)

                # Get the normalized label for the graph
                label = labels[f"{file_name}"]

                # Create a Data object for the graph, containing node features, edge indices, and labels
                data = Data(x=node_attr, edge_index=edge_index, y=torch.tensor([label]).float())
                split_data_list.append(data)  # Add the graph data to the split data list

            # Save the processed data for each split (train, val, test)
            torch.save(split_data_list, os.path.join(self.processed_dir, f"type1_{self.file_name}_{split}_data.pt"))