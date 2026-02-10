import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, GATConv, GINConv, MLP,SAGEConv, ChebConv, dense_diff_pool, DenseSAGEConv, DenseGraphConv,HeteroConv, Linear,HGTConv
from torch_geometric.nn import global_mean_pool,global_max_pool,global_add_pool


class GraphConvModel(torch.nn.Module):
    def __init__(self,num_features,num_classes):
        super().__init__()
        
        self.drop1 = torch.nn.Dropout()
        self.conv1 = GraphConv(num_features, 30)
        self.bn1 = torch.nn.BatchNorm1d(30)
        
        self.drop2 = torch.nn.Dropout()
        self.conv2 = GraphConv(30, 20)
        self.bn2 = torch.nn.BatchNorm1d(20)
        
        self.lin1 = torch.nn.Linear(40,10)
        self.lin2 = torch.nn.Linear(10,1) 
        
        self.float()

    def forward(self,x,edge_index,batch):

        x = self.drop1(x)
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        
        x = self.drop2(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        
        x1 = global_max_pool(x,batch)
        x2 = global_mean_pool(x,batch)
        
        x = torch.cat((x1,x2), dim=1)
 
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        
        
        return torch.sigmoid(x)
    
    
class GINConvModel(torch.nn.Module):
    def __init__(self,num_features,num_classes):
        super().__init__()
        
        self.drop1 = torch.nn.Dropout()
        mlp1 = MLP([num_features, 50, 20])
        self.conv1 = GINConv(mlp1)
        self.bn1 = torch.nn.BatchNorm1d(20)
        
        self.drop2 = torch.nn.Dropout()
        mlp2 = MLP([20, 20, 20])
        self.conv2 = GINConv(mlp2)
        self.bn2 = torch.nn.BatchNorm1d(20)
        
        self.lin1 = torch.nn.Linear(40,10)
        self.lin2 = torch.nn.Linear(10,1) 
        
        self.float()

    def forward(self,x,edge_index,batch):

        x = self.drop1(x)
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        
        x = self.drop2(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        
        x1 = global_max_pool(x,batch)
        x2 = global_mean_pool(x,batch)
        
        x = torch.cat((x1,x2), dim=1)
 
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        
        
        return torch.sigmoid(x)
    
class SAGEConvModel(torch.nn.Module):
    def __init__(self,num_features,num_classes):
        super().__init__()
        
        self.drop1 = torch.nn.Dropout()
        self.conv1 = SAGEConv(num_features, 30)
        self.bn1 = torch.nn.BatchNorm1d(30)
        
        self.drop2 = torch.nn.Dropout()
        self.conv2 = SAGEConv(30, 30)
        self.bn2 = torch.nn.BatchNorm1d(30)
        
        self.lin1 = torch.nn.Linear(60,10)
        self.lin2 = torch.nn.Linear(10,1) 
        
        self.float()

    def forward(self,x,edge_index,batch):

        x = self.drop1(x)
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        
        x = self.drop2(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        
        x1 = global_max_pool(x,batch)
        x2 = global_mean_pool(x,batch)
        
        x = torch.cat((x1,x2), dim=1)
 
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return torch.sigmoid(x)

class ChebConvModel(torch.nn.Module):
    def __init__(self, num_features, num_classes, K=3):
        super().__init__()

        self.drop1 = torch.nn.Dropout()
        self.conv1 = ChebConv(num_features, 30, K)
        self.bn1 = torch.nn.BatchNorm1d(30)

        self.drop2 = torch.nn.Dropout()
        self.conv2 = ChebConv(30, 30, K)
        self.bn2 = torch.nn.BatchNorm1d(30)

        self.lin1 = torch.nn.Linear(60, 10)
        self.lin2 = torch.nn.Linear(10, 1) 

        self.float()

    def forward(self, x, edge_index, batch):
        x = self.drop1(x)
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)

        x = self.drop2(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)