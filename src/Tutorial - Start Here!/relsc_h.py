import os
import json
import math
import urllib.request
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    extract_zip,
)

ZENODO_URLS = [
    'https://zenodo.org/records/18598713/files/data.zip?download=1',
    'https://zenodo.org/records/18598713/files/y_labels.zip?download=1'
]

def _find_file(directory, filename):
    target_lower = filename.lower()
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower() == target_lower:
                return os.path.join(root, f)
    raise FileNotFoundError(f"File {filename} (case-insensitive check) not found in {directory}")

def _download_with_tqdm(url, filepath):
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)
            
    # Extract filename from URL for the progress bar
    filename_desc = url.split('/')[-1].split('?')[0]
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=f"Downloading {filename_desc}") as t:
        urllib.request.urlretrieve(url, filename=filepath, reporthook=t.update_to)


class RelSCH(InMemoryDataset):
    r"""The homogeneous variant (RelSC-H) of the RelSC benchmark dataset."""
    
    urls = ZENODO_URLS

    def __init__(self, root: str = './data', project_name: str = 'rdf', transform=None, pre_transform=None, force_reload: bool = False):
        self.project_name = project_name
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.load(self.processed_paths[0], data_cls=Data)
        self.split_idx = torch.load(self.processed_paths[1], weights_only=False)

    @property
    def raw_dir(self) -> str:
        # Shared raw directory between the two dataset variants
        return os.path.join(self.root, 'RelSC', 'raw')

    @property
    def raw_file_names(self):
        return [f"{self.project_name}.json", f"y_{self.project_name}.csv"]

    @property
    def processed_file_names(self):
        return [f'data_homogeneous_{self.project_name}.pt', f'split_homogeneous_{self.project_name}.pt']

    def download(self):
        # Custom check to avoid re-downloading if files are in subfolders
        files_exist = True
        for f in self.raw_file_names:
            try:
                _find_file(self.raw_dir, f)
            except FileNotFoundError:
                files_exist = False
                break
                
        if files_exist:
            print(f"Files for {self.project_name} already exist in {self.raw_dir}. Skipping download.")
            return

        os.makedirs(self.raw_dir, exist_ok=True)
        for url in self.urls:
            filename = url.split('/')[-1].split('?')[0]
            filepath = os.path.join(self.raw_dir, filename)
            
            _download_with_tqdm(url, filepath)
            
            print(f"Extracting {filename}...")
            extract_zip(filepath, self.raw_dir)
            os.remove(filepath)

    def get_idx_split(self):
        return self.split_idx

    def _aggregate_multi_edges(self, edge_index, edge_attr):
        if edge_attr is not None:
            unique_edges, inverse_indices = torch.unique(edge_index, dim=1, return_inverse=True)
            num_edges = unique_edges.size(1)
            aggregated_edge_attr = torch.zeros((num_edges, edge_attr.size(1)), dtype=torch.float)

            for i in range(len(inverse_indices)):
                aggregated_edge_attr[inverse_indices[i]] = torch.logical_or(
                    aggregated_edge_attr[inverse_indices[i]].bool(), edge_attr[i].bool()
                ).float()
            return unique_edges, aggregated_edge_attr
        return edge_index, edge_attr

    def process(self):
        json_path = _find_file(self.raw_dir, self.raw_file_names[0])
        csv_path = _find_file(self.raw_dir, self.raw_file_names[1])

        with open(json_path, 'r') as f:
            dataset = json.load(f)

        labels_file = pd.read_csv(csv_path)
        labels_file['Value'] = labels_file['Value'] + 1
        labels_file['LogValue'] = labels_file['Value'].apply(math.log)
        max_val = labels_file['LogValue'].max()
        labels_file['NormalizedValue'] = labels_file['LogValue'] / max_val
        labels = labels_file.set_index('Key')['NormalizedValue'].to_dict()

        sorted_keys = sorted([k for k in dataset.keys() if k in labels])
        
        train_keys, temp_keys = train_test_split(sorted_keys, train_size=0.7, shuffle=False)
        dev_keys, test_keys = train_test_split(temp_keys, test_size=0.5, shuffle=False)

        all_edge_features = []
        for key in sorted_keys:
            all_edge_features.extend(dataset[key][0][2])
        
        unique_edge_features = np.unique(all_edge_features)
        num_edge_features = len(unique_edge_features)

        max_nodes = 73
        unique_node_features = np.arange(0, max_nodes + 1)

        data_list = []
        
        for key in sorted_keys:
            graph_data = dataset[key]
            num_nodes = graph_data[1]
            node_features = graph_data[0][0]
            edge_features = graph_data[0][2]
            
            node_attr = torch.zeros((len(node_features), max_nodes), dtype=torch.float)
            for i, feature in enumerate(node_features):
                feature_index = np.where(unique_node_features == feature[0])[0][0]
                node_attr[i, feature_index] = 1.0

            edge_attr = torch.zeros((len(edge_features), num_edge_features), dtype=torch.int32)
            for i, feature in enumerate(edge_features):
                feature_index = np.where(unique_edge_features == feature[0])[0][0]
                edge_attr[i, feature_index] = 1.0

            adj_list = np.array(graph_data[0][1])
            edge_index = torch.tensor(adj_list, dtype=torch.long).contiguous()

            edge_index, edge_attr = self._aggregate_multi_edges(edge_index, edge_attr)

            node_edge_info = torch.zeros((num_nodes, num_edge_features), dtype=torch.float)
            for i in range(edge_index.size(1)):
                src_node = edge_index[0, i]
                node_edge_info[src_node] += edge_attr[i]

            node_attr = torch.cat([node_attr, node_edge_info], dim=1)
            label = torch.tensor([labels[key]], dtype=torch.float)

            data = Data(x=node_attr, edge_index=edge_index, y=label)
            
            if self.pre_transform is not None:
                data = self.pre_transform(data)
                
            data_list.append(data)

        split_dict = {
            'train': torch.tensor([sorted_keys.index(k) for k in train_keys], dtype=torch.long),
            'val': torch.tensor([sorted_keys.index(k) for k in dev_keys], dtype=torch.long),
            'test': torch.tensor([sorted_keys.index(k) for k in test_keys], dtype=torch.long)
        }

        self.save(data_list, self.processed_paths[0])
        torch.save(split_dict, self.processed_paths[1])