import os
import json
import pandas as pd
import numpy as np
import math
import torch
from torch_geometric.data import InMemoryDataset, HeteroData
from collections import defaultdict
from sklearn.model_selection import train_test_split
from torch_geometric.data import Batch

class Type2(InMemoryDataset):
    # Initialize the Type2 dataset, inheriting from InMemoryDataset
    def __init__(self, root, x_folder="/", y_folder="/", file_name=None, split=["train", "test", "val"]):
        self.split = split  # A list containing dataset splits (train, test, val)
        self.x_folder = x_folder  # Folder for input features
        self.y_folder = y_folder  # Folder for labels
        self.file_name = file_name  # List of filenames for datasets

        # Dictionary that groups node features into semantic categories
        self.semantic_dict = {
            'declarations': [6, 11, 12, 13, 17, 55, 32, 33, 43, 27, 65, 72, 53, 42, 4, 68, 5, 0],
            'control_flow': [3, 16, 21, 23, 38, 40, 10, 50, 61, 63, 35, 69, 46, 18],
            'types_and_references': [1, 45, 15, 36, 49, 37, 9, 66, 28, 26, 29],
            'expressions_and_operations': [2, 7, 67, 39, 52, 54, 64, 70, 30, 58, 57],
            'code_structure': [8, 19, 25, 24, 48, 71, 14, 47, 62, 31, 22],
            'exceptions': [60, 41, 56, 34],
            'literals_and_constants': [59, 44, 51, 20]
        }
        
        # Call parent class initializer
        super(Type2, self).__init__(root)

        self.data_list = []  # Holds processed data across splits
        self.split_data = {}  # Dictionary to store data split information
        
        # Load preprocessed data for each split if it exists
        for split in self.split:
            file_path = os.path.join(self.processed_dir, f"type2_{self.file_name}_{split}_data.pt")
            if os.path.exists(file_path):
                split_dataset = torch.load(file_path, weights_only=False)  # Load the processed dataset for this split
                self.split_data[split] = split_dataset  # Store the split dataset
                self.data_list.extend(split_dataset)  # Add to the complete dataset list
            else:
                self.split_data[split] = []  # Initialize empty list if no dataset exists for the split

    # Load data for a specific split
    def load_split(self, split_name):
        if split_name in self.split_data:
            return self.split_data[split_name]  # Return data for the requested split
        else:
            return []  # Return empty list if split not found

    # Check if processed files for all splits exist
    def _processed_files_exist(self):
        return all(os.path.exists(os.path.join(self.processed_dir, f"type2_{self.file_name}_{split}_data.pt")) for split in self.split)
    
    # Custom HeteroData collate function
    def hetero_collate(self, data_list):
        return Batch.from_data_list(data_list)

    # Property to return raw filenames
    @property
    def raw_file_names(self):
        return self.file_name
    
    # Property to return processed filenames
    @property
    def processed_file_names(self):
        return [f"type2_{self.file_name}_{split}_data.pt" for split in self.split]
        
    # Return the number of data items in the dataset
    def len(self):
        return len(self.data_list)

    # Retrieve a specific data item by index
    def get(self, idx):
        return self.data_list[idx]

    # Function to retrieve the category of a node based on its feature using the semantic dictionary
    def get_category(self, numero, semantic_dict):
        for categoria, lista in semantic_dict.items():
            if numero in lista:
                return categoria
        return None

    # Function to find the index of a number within the list of a dictionary
    def trova_indice(self, dizionario, numero_cercato):
        for chiave, lista in dizionario.items():
            if numero_cercato in lista:
                indice = lista.index(numero_cercato)
                return indice
        return None

    # Adds reverse edges to the graph
    def aggiungi_archi_inversi(self, grafo):
        nuovi_archi = defaultdict(list)
        
        # Iterate over each edge type in the graph
        for chiave, archi in grafo.items():
            tipo_partenza, nodo_id, tipo_destinazione = chiave
            
            # Add the reverse edge if the source and destination types are the same
            if tipo_partenza == tipo_destinazione:
                for arco in archi:
                    nuovi_archi[chiave].append(arco)  # Add the original edge
                    arco_inverso = arco[::-1]  # Reverse the edge direction
                    nuovi_archi[chiave].append(arco_inverso)  # Add the reverse edge
            else:
                # Create a new key and add reverse edges when types differ
                nuova_chiave = (tipo_destinazione, nodo_id + "_rev", tipo_partenza)
                for arco in archi:
                    nuovi_archi[chiave].append(arco)
                    arco_inverso = arco[::-1]
                    nuovi_archi[nuova_chiave].append(arco_inverso)
        
        return nuovi_archi

    # Helper function to retrieve both dataset and corresponding labels
    def get_set_and_labels(self, keys, dataset, labels):
        data_set = {key: dataset[key] for key in keys}  # Extract data for the given keys
        label_set = {key: labels[key] for key in keys if key in labels}  # Extract labels for the given keys
        return data_set, label_set

    # Core processing method that transforms raw files into PyTorch geometric data objects
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

        # Sort dataset keys and match with corresponding labels
        sorted_keys = sorted(dataset.keys())
        sorted_dataset = {key: dataset[key] for key in sorted_keys}
        sorted_labels = {key: labels[key] for key in sorted_keys if key in labels}

        # Split the dataset into train, validation, and test sets
        train_keys, temp_keys = train_test_split(sorted_keys, train_size=0.7, shuffle=False)
        dev_keys, test_keys = train_test_split(temp_keys, test_size=0.5, shuffle=False)

        # Map the split keys into the corresponding data subsets
        split_map = {}
        if "train" in self.split:
            split_map["train"] = train_keys
        if "val" in self.split:
            split_map["val"] = dev_keys
        if "test" in self.split:
            split_map["test"] = test_keys

        # Create dictionaries to hold data and labels for each split
        result = {}
        for split in self.split:
            result[f"{split}_set"], result[f"{split}_labels"] = self.get_set_and_labels(split_map[split], sorted_dataset, sorted_labels)

        # Define maximum number of nodes and initialize feature arrays
        max_nodes = 73
        unique_node_features = np.arange(0, max_nodes + 1)
        all_edge_features = []

        # Gather all edge features from the dataset
        for file_name, graph_data in dataset.items():
            edge_features = graph_data[0][2]  # Extract edge features
            all_edge_features.extend(edge_features)
        
        # Determine unique edge features in the dataset
        unique_edge_features = np.unique(all_edge_features)
        num_edge_features = len(unique_edge_features)

        # Process the data for each split (train, validation, test)
        for split in self.split:
            split_data_list = []
            for file_name, graph_data in result[f"{split}_set"].items():
                num_nodes = graph_data[1]  # Number of nodes
                node_features = graph_data[0][0]  # Extract node features
                node_attr = torch.tensor(node_features, dtype=torch.int32)  # Convert node features to tensor

                adj_list = np.array(graph_data[0][1])  # Adjacency list (edge list)
                edge_features = graph_data[0][2]  # Extract edge features
                edge_index = torch.tensor(adj_list, dtype=torch.long).t().contiguous()  # Create edge index tensor
                or_edge_attr = torch.tensor(edge_features, dtype=torch.int32)

                # One-hot encode node features
                node_attr = torch.zeros((len(node_features), max_nodes), dtype=torch.float)
                for i, feature in enumerate(node_features):
                    feature_index = np.where(unique_node_features == feature[0])[0][0]
                    node_attr[i, feature_index] = 1.0

                # One-hot encode edge features
                edge_attr = torch.zeros((len(edge_features), num_edge_features), dtype=torch.int32)
                for i, feature in enumerate(edge_features):
                    feature_index = np.where(unique_edge_features == feature[0])[0][0]
                    edge_attr[i, feature_index] = 1.0

                node_types = [i[0] for i in node_features]  # Extract node types from features
                data = HeteroData()  # Initialize HeteroData object

                # Create a dictionary to group nodes by their types
                node_type_dict = {}
                for idx, node_type in enumerate(node_types):
                    key = self.get_category(int(node_type), self.semantic_dict)  # Get the category of each node
                    if key not in node_type_dict:
                        node_type_dict[key] = []
                    node_type_dict[key].append(idx)

                # Aggregate edge information into the node attributes
                node_edge_info = torch.zeros((num_nodes, num_edge_features), dtype=torch.float)
                for i in range(edge_index.size(1)):
                    src_node = edge_index[0, i]
                    node_edge_info[src_node] += edge_attr[i]
                
                # Concatenate node and edge attributes
                new_node_attr = torch.zeros((num_nodes, 84), dtype=torch.float)
                for i in range(num_nodes):
                    new_node_attr[i] = torch.cat((node_attr[i], node_edge_info[i]), dim=0)
                    
                # Add node data to the HeteroData object by category
                for category, indices in node_type_dict.items():
                    if 'x' in data[category]:
                        data[category].x = torch.cat([data[category].x, new_node_attr[indices]], dim=0)
                    else:
                        data[category].x = node_attr[indices]

                # Create a dictionary to hold edge indices based on node types
                edge_index_dict = defaultdict(list)
                for i, (src, dest) in enumerate(edge_index):
                    src_type = self.get_category(node_types[int(src.item())], self.semantic_dict)
                    dest_type = self.get_category(node_types[int(dest.item())], self.semantic_dict)
                    edge_type = or_edge_attr[i]

                    # Get local indices for the source and destination nodes
                    src_local = self.trova_indice(node_type_dict, src.item())
                    dest_local = self.trova_indice(node_type_dict, dest.item())
                    key = (str(src_type), str(edge_type.item()), str(dest_type))
                    edge_index_dict[key].append([src_local, dest_local])

                # Add reverse edges to the graph
                edge_index_dict = self.aggiungi_archi_inversi(edge_index_dict)

                # Add edge data to the HeteroData object by edge type
                for key in edge_index_dict:
                    data[key].edge_index = torch.tensor(edge_index_dict[key], dtype=torch.long).t()

                # Assign labels to the graph
                graph_label = torch.tensor([result[f"{split}_labels"][file_name]], dtype=torch.float)
                data.y = graph_label  # Assign graph label to data object
                split_data_list.append(data)  # Add the data to the split data list
                
            # Save processed data for each split
            torch.save(split_data_list, os.path.join(self.processed_dir, f"type2_{self.file_name}_{split}_data.pt"))