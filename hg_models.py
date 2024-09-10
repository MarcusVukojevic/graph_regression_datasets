import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, GATConv ,SAGEConv,HeteroConv, Linear
from torch_geometric.nn.conv import HGTConv


class HeteroGNN_SAGE(torch.nn.Module):
    # Initialize the Heterogeneous GNN model using SAGEConv layers
    def __init__(self, dataset, hidden_channels, out_channels, num_layers, device):
        super().__init__()
        self.device = device  # Device (CPU or GPU) where the model will run
        self.metadata = extract_metadata(dataset)  # Extract metadata (node and edge types) from the dataset
        self.hidden_channels = hidden_channels  # Number of hidden channels for each layer
        self.convs = torch.nn.ModuleList()  # List to hold the convolutional layers (SAGEConv layers)
        self.dropout_prob = 0.2  # Dropout probability to apply after each layer

        # Create a Heterogeneous SAGEConv layer for each layer in the network
        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: SAGEConv((-1, -1), hidden_channels)  # Create a SAGEConv for each edge type in the dataset
                for edge_type in self.metadata[1]
            })
            self.convs.append(conv)  # Append the convolutional layer to the list
        
        # Multi-layer perceptron (MLP) for combining hidden features of all node types
        self.mlp = Linear(hidden_channels * len(self.metadata[0]), hidden_channels)
        # Linear layer for producing the final output
        self.lin = Linear(hidden_channels, out_channels)
    
    # Define the forward pass
    def forward(self, x_dict, edge_index_dict, batch_dict):
        # Convolutional layers for each SAGE layer
        # Convert node features to float type
        x_dict = {key: x.float() for key, x in x_dict.items()}
        
        # Pass through each SAGE convolutional layer
        for conv in self.convs:
            # Apply the heterogeneous convolution
            x_dict = conv(x_dict, edge_index_dict)
            # Apply Leaky ReLU non-linearity to each node type's features
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
            # Apply dropout to prevent overfitting
            x_dict = {key: F.dropout(x, p=self.dropout_prob, training=self.training) for key, x in x_dict.items()}

        # Initialize an empty list to hold the graph representations
        graph_representations = []
        
        # Iterate over each graph in the batch
        for b in range(batch_dict[next(iter(batch_dict))].max() + 1):  # Determine the number of graphs in the batch
            tmp = []  # Temporary list to hold node features for the current graph
            for node_type in self.metadata[0]:
                if node_type in x_dict:
                    # Select the nodes belonging to the current graph 'b' for this node type
                    node_features = x_dict[node_type][batch_dict[node_type] == b]
                    # Sum the node features (aggregation) for the current graph
                    tmp.append(torch.sum(node_features, dim=0).to(self.device)) 
                else:
                    # If the node type is not present in the graph, append a tensor of zeros
                    tmp.append(torch.zeros(self.hidden_channels).to(self.device))

            # Concatenate the node representations for the current graph
            graph_rep = torch.cat(tmp)
            graph_representations.append(graph_rep)  # Add the graph representation to the list

        # Stack the graph representations for all graphs in the batch
        x = torch.stack(graph_representations)

        # Pass through the MLP and then the final linear layer
        x = self.mlp(x)
        
        # Return the output from the final layer (predicted graph-level features)
        return self.lin(x)




class HeteroGNN_GAT(torch.nn.Module):
    # Initialize the Heterogeneous GNN model using GAT layers
    def __init__(self, dataset, hidden_channels, out_channels, num_layers, device):
        super().__init__()
        self.device = device  # Device (CPU or GPU) where the model will run
        self.metadata = extract_metadata(dataset)  # Extract metadata (node and edge types) from the dataset
        self.hidden_channels = hidden_channels  # Number of hidden channels for each GAT layer
        self.convs = torch.nn.ModuleList()  # List to hold the convolutional layers (GAT layers)
        self.dropout_prob = 0.2  # Dropout probability to apply after each layer

        # Create a Heterogeneous GATConv layer for each layer in the network
        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: GATConv((-1, -1), hidden_channels, add_self_loops=False)
                for edge_type in self.metadata[1]  # Create a GATConv for each edge type in the dataset
            })
            self.convs.append(conv)  # Append the convolutional layer to the list
        
        # Multi-layer perceptron (MLP) for combining hidden features of all node types
        self.mlp = Linear(hidden_channels * len(self.metadata[0]), hidden_channels)
        # Linear layer for producing the final output
        self.lin = Linear(hidden_channels, out_channels)
    
    # Define the forward pass
    def forward(self, x_dict, edge_index_dict, batch_dict):
        # Convolutional layers for each GAT layer
        # Convert node features to float type
        x_dict = {key: x.float() for key, x in x_dict.items()}
        
        # Pass through each GAT convolutional layer
        for conv in self.convs:
            # Apply the heterogeneous convolution
            x_dict = conv(x_dict, edge_index_dict)
            # Apply Leaky ReLU non-linearity to each node type's features
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
            # Apply dropout to prevent overfitting
            x_dict = {key: F.dropout(x, p=self.dropout_prob, training=self.training) for key, x in x_dict.items()}

        # Initialize an empty list to hold the graph representations
        graph_representations = []
        
        # Iterate over each graph in the batch
        for b in range(batch_dict[next(iter(batch_dict))].max() + 1):  # Determine the number of graphs in the batch
            tmp = []  # Temporary list to hold node features for the current graph
            for node_type in self.metadata[0]:
                if node_type in x_dict:
                    # Select the nodes belonging to the current graph 'b' for this node type
                    node_features = x_dict[node_type][batch_dict[node_type] == b]
                    # Sum the node features (aggregation) for the current graph
                    tmp.append(torch.sum(node_features, dim=0).to(self.device)) 
                else:
                    # If the node type is not present in the graph, append a tensor of zeros
                    tmp.append(torch.zeros(self.hidden_channels).to(self.device))

            # Concatenate the node representations for the current graph
            graph_rep = torch.cat(tmp)
            graph_representations.append(graph_rep)  # Add the graph representation to the list

        # Stack the graph representations for all graphs in the batch
        x = torch.stack(graph_representations)

        # Pass through the MLP and then the final linear layer
        x = self.mlp(x)
        
        # Return the output from the final layer (predicted graph-level features)
        return self.lin(x)


def extract_metadata(dataset):
        node_types = set()
        edge_types = set()
        # Itera su tutti i grafi del dataset
        for data in dataset:
            # Raccogli i tipi di nodi
            node_types.update(list(data.x_dict.keys()))
            # Raccogli i tipi di archi
            edge_types.update(list(data.edge_index_dict.keys()))
        # Converti in liste ordinate (per consistenza)
        node_types = list(sorted(node_types))
        edge_types = list(sorted(edge_types))
    
        return (node_types, edge_types)


class HeteroGNN_HGT(torch.nn.Module):
    # Initialize the Heterogeneous GNN model using HGTConv layers
    def __init__(self, dataset, hidden_channels, out_channels, num_layers, device):
        super().__init__()
        self.device = device  # Device (CPU or GPU) where the model will run
        self.metadata = extract_metadata(dataset)  # Extract metadata (node and edge types) from the dataset for HGTConv
        self.hidden_channels = hidden_channels  # Number of hidden channels for each layer
        self.convs = torch.nn.ModuleList()  # List to hold the HGT convolutional layers
        self.dropout_prob = 0.2  # Dropout probability to apply after each layer
        num_heads = 2  # Number of attention heads in the HGTConv layers

        # Create a dictionary of linear layers for each node type
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in extract_metadata(dataset)[0]:
            # Each node type has a linear transformation before entering the HGT layer
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        # Create the HGTConv layers
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            # HGTConv layer that handles heterogeneous graphs with attention across multiple node and edge types
            conv = HGTConv(hidden_channels, hidden_channels, extract_metadata(dataset), num_heads)
            self.convs.append(conv)

        # Final linear layer to produce the output
        self.lin = Linear(hidden_channels, out_channels)

        # Move the convolutional layers to the correct device (e.g., GPU if available)
        for conv in self.convs:
            conv.to(device)

        # Add an MLP layer followed by a final linear layer for output
        self.mlp = Linear(hidden_channels * len(self.metadata[0]), hidden_channels).to(device)
        self.lin = Linear(hidden_channels, out_channels).to(device)

    # Define the forward pass
    def forward(self, x_dict, edge_index_dict, batch_dict):
        # Apply a linear transformation and ReLU activation to each node type
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        # Pass through each HGT convolutional layer
        for conv in self.convs:
            # Apply the HGT convolution to the node embeddings and edge indices
            x_dict = conv(x_dict, edge_index_dict)

        # Initialize an empty list to hold the graph representations
        graph_representations = []

        # Iterate over each graph in the batch
        for b in range(batch_dict[next(iter(batch_dict))].max() + 1):  # Determine the number of graphs in the batch
            tmp = []  # Temporary list to hold node features for the current graph
            for node_type in self.metadata[0]:
                if node_type in x_dict:
                    # Select the nodes belonging to the current graph 'b' for this node type
                    node_features = x_dict[node_type][batch_dict[node_type] == b]
                    # Sum the node features (aggregation) for the current graph
                    tmp.append(torch.sum(node_features, dim=0)) 
                else:
                    # If the node type is not present in the graph, append a tensor of zeros
                    tmp.append(torch.zeros(self.hidden_channels, device=self.device))

            # Concatenate the node representations for the current graph
            graph_rep = torch.cat(tmp)
            graph_representations.append(graph_rep)  # Add the graph representation to the list

        # Stack the graph representations for all graphs in the batch
        x = torch.stack(graph_representations)

        # Pass through the MLP and then the final linear layer
        x = self.mlp(x)
        output = self.lin(x)  # Apply the final linear transformation to produce the output

        return output  # Return the predicted graph-level features




# Helper function to extract the metadata (node types and edge types) from the dataset
def extract_metadata(dataset):
    node_types = set()  # Set to hold unique node types
    edge_types = set()  # Set to hold unique edge types
    
    # Iterate through all graphs in the dataset
    for data in dataset:
        # Collect node types from the graph
        node_types.update(list(data.x_dict.keys()))
        # Collect edge types from the graph
        edge_types.update(list(data.edge_index_dict.keys()))
    
    # Convert sets to sorted lists for consistency
    node_types = list(sorted(node_types))
    edge_types = list(sorted(edge_types))
    
    return (node_types, edge_types)  # Return the node and edge types as a tuple
