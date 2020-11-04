import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GraphConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool as gadd
from torch_geometric.data import Dataset


################################################################################

class MyDataset(Dataset):
    def __init__(self,
                 data_list):
        self.data_list = data_list
        super(MyDataset, self).__init__("./_temp")

    def __getitem__(self, idx):
        return self.data_list[idx]

    def _download(self):
        pass

    def _process(self):
        pass

    def __len__(self):
        return len(self.data_list)


################################################################################

class FunnelGNN_old(torch.nn.Module):
    def __init__(self,
                 input_dim=0,
                 node_hidden_dim=128,
                 fc_hidden_dim=128,
                 dropout_rate=0.5,
                 out_dim=2):
        super(FunnelGNN, self).__init__()
        self.bn0 = torch.nn.BatchNorm1d(input_dim)
        self.conv1 = GraphConv(input_dim, node_hidden_dim)
        self.conv2 = GraphConv(node_hidden_dim, 2*node_hidden_dim)
        self.conv3 = GraphConv(2*node_hidden_dim, 3*node_hidden_dim)

        self.bn1 = torch.nn.BatchNorm1d(node_hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(2*node_hidden_dim)
        self.bn3 = torch.nn.BatchNorm1d(3*node_hidden_dim)

        self.lin1 = Linear(3*node_hidden_dim + 6*node_hidden_dim + 9*node_hidden_dim, fc_hidden_dim)
        self.lin2 = torch.nn.Linear(fc_hidden_dim, out_dim)
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index, batch, edge_attr=None):

        x = self.bn1(F.leaky_relu(self.conv1(x, edge_index)))
        x1 = torch.cat([gmp(x, batch), gap(x, batch), gadd(x, batch)], dim=1)

        x = self.bn2(F.leaky_relu(self.conv2(x, edge_index)))
        x2 = torch.cat([gmp(x, batch), gap(x, batch), gadd(x, batch)], dim=1)

        x = self.bn3(F.leaky_relu(self.conv3(x, edge_index)))
        x3 = torch.cat([gmp(x, batch), gap(x, batch), gadd(x, batch)], dim=1)

        x = torch.cat([x1, x2, x3], dim=1)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin1(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.log_softmax(self.lin2(x), dim=-1)

        return x


################################################################################

class FunnelGNN(torch.nn.Module):
    def __init__(self,
                 input_dim=0,
                 node_hidden_dim=128,
                 nr_hidden_layers=4,
                 incr_hidden_dim=False,
                 fc_hidden_dim=128,
                 dropout_rate=0.5,
                 out_dim=2):
        super(FunnelGNN, self).__init__()
        node_hidden_dim1 = node_hidden_dim
        node_hidden_dim2 = node_hidden_dim
        node_hidden_dim3 = node_hidden_dim
        node_hidden_dim4 = node_hidden_dim
        node_hidden_dim5 = node_hidden_dim
        if incr_hidden_dim:
            node_hidden_dim2 = 2*node_hidden_dim
            node_hidden_dim3 = 3*node_hidden_dim
            node_hidden_dim4 = 4*node_hidden_dim
            node_hidden_dim5 = 5*node_hidden_dim
        self.conv1 = GraphConv(input_dim, node_hidden_dim1)
        self.conv2 = GraphConv(node_hidden_dim1, node_hidden_dim2)
        self.conv3 = GraphConv(node_hidden_dim2, node_hidden_dim3)
        self.bn0 = torch.nn.BatchNorm1d(input_dim)
        self.bn1 = torch.nn.BatchNorm1d(node_hidden_dim1)
        self.bn2 = torch.nn.BatchNorm1d(node_hidden_dim2)
        self.bn3 = torch.nn.BatchNorm1d(node_hidden_dim3)
        lin1_in_dim = 2*node_hidden_dim1 + 2*node_hidden_dim2 + 2*node_hidden_dim3
        if nr_hidden_layers == 4:
            self.conv4 = GraphConv(node_hidden_dim3, node_hidden_dim4)
            self.bn4 = torch.nn.BatchNorm1d(node_hidden_dim4)
            lin1_in_dim += 2*node_hidden_dim4
        if nr_hidden_layers == 5:
            self.conv5 = GraphConv(node_hidden_dim4, node_hidden_dim5)
            self.bn5 = torch.nn.BatchNorm1d(node_hidden_dim5)
            lin1_in_dim += 2*node_hidden_dim5
        self.lin1 = Linear(lin1_in_dim, fc_hidden_dim)
        self.lin2 = torch.nn.Linear(fc_hidden_dim, out_dim)
        self.dropout_rate = dropout_rate
        self.nr_hidden_layers = nr_hidden_layers

    def forward(self, x, edge_index, batch, edge_attr=None):

        x = self.bn1(F.leaky_relu(self.conv1(x, edge_index)))
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.bn2(F.leaky_relu(self.conv2(x, edge_index)))
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self.bn3(F.leaky_relu(self.conv3(x, edge_index)))
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        if self.nr_hidden_layers == 4:
            x = self.bn4(F.leaky_relu(self.conv4(x, edge_index)))
            x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        if self.nr_hidden_layers == 5:
            x = self.bn4(F.leaky_relu(self.conv4(x, edge_index)))
            x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
            x = self.bn5(F.leaky_relu(self.conv5(x, edge_index)))
            x5 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        if self.nr_hidden_layers == 3:
            x = torch.cat([x1, x2, x3], dim=1)
        elif self.nr_hidden_layers == 4:
            x = torch.cat([x1, x2, x3, x4], dim=1)
        elif self.nr_hidden_layers == 5:
            x = torch.cat([x1, x2, x3, x4, x5], dim=1)

        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin1(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.log_softmax(self.lin2(x), dim=-1)

        return x


################################################################################

class FunnelGNN_EdgeAttr(torch.nn.Module):
    def __init__(self, input_dim=0, node_hidden_dim=128, fc_hidden_dim=128, out_dim=2):
        super(FunnelGNN_EdgeAttr, self).__init__()
        self.bn0 = torch.nn.BatchNorm1d(input_dim)
        self.conv1 = GraphConv(input_dim, node_hidden_dim)
        self.conv2 = GraphConv(node_hidden_dim, 2*node_hidden_dim)
        self.conv3 = GraphConv(2*node_hidden_dim, 3*node_hidden_dim)

        self.bn1 = torch.nn.BatchNorm1d(node_hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(2*node_hidden_dim)
        self.bn3 = torch.nn.BatchNorm1d(3*node_hidden_dim)

        self.lin1 = Linear(3*node_hidden_dim + 6*node_hidden_dim + 9*node_hidden_dim, fc_hidden_dim)
        self.lin2 = torch.nn.Linear(fc_hidden_dim, out_dim)

    def forward(self, x, edge_index, batch, edge_attr=None):
        edge_attr = torch.transpose(edge_attr, 0, 1)
        edge_attr = edge_attr[0]

        x = self.bn1(F.leaky_relu(self.conv1(x, edge_index, edge_attr)))
        x1 = torch.cat([gmp(x, batch),  gap(x, batch),   gadd(x, batch)], dim=1)
        x = self.bn2(F.leaky_relu(self.conv2(x, edge_index, edge_attr)))
        x2 = torch.cat([gmp(x, batch),  gap(x, batch),   gadd(x, batch)], dim=1)
        x = self.bn3(F.leaky_relu(self.conv3(x, edge_index, edge_attr)))
        x3 = torch.cat([gmp(x, batch),  gap(x, batch),   gadd(x, batch)], dim=1)

        x = torch.cat([x1, x2, x3], dim=1)
        x = (F.relu(self.lin1(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = (F.relu(self.lin2(x)))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x


################################################################################
