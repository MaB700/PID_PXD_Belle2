# %%
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
import torch
import torch.nn as nn
from torch_geometric.utils.convert import to_networkx
from torch_geometric.nn import (
    NNConv,
    global_mean_pool,
    graclus,
    max_pool,
    max_pool_x,
)
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from torch_geometric.data.data import Data
from torch.nn import Linear
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ARMAConv
from torch_geometric.nn import global_mean_pool

import networkx as nx


# %%

def CreateGraphDataset(path, n_samples, label):        
    data = pd.read_csv(path, header=None, delimiter= " ", nrows=n_samples).values.astype('int')
    def make_graph(index):     
        # Load num filled pixels & ADC values
        n = np.count_nonzero(data[index, 2:83] > 0)
        adc_indices = np.nonzero(data[index, 2:83])
        adc = data[index, 2:83][adc_indices]
        seed_index = np.argmax(adc) # TODO: can 2+ pixels have the adc of seed pixel ?
        y_pos, x_pos = np.divmod(adc_indices, 9)
        
        # Node features
        x = np.zeros((n, 3)) # num node features
        x[:, 0] = (adc.astype('float64'))/255.0
        x[:, 1] = (x_pos.astype('float64'))/8.0 # x_coord [0,1]
        x[:, 2] = (y_pos.astype('float64'))/8.0 # y_coord [0,1]
        
        # Edges
        if n > 1 :
            a = np.full((n - 1), seed_index)
            b = np.delete(np.arange(0, n), seed_index)
            # a = np.full((n), seed_index)
            # b = np.arange(0, n)
            edge_index = np.row_stack((np.concatenate((a,b)), np.concatenate((b,a))))   
            edge_index = torch.from_numpy(edge_index)       
        else :
            edge_index = torch.from_numpy(np.row_stack(([0], [0])))
        
        # Labels
        y = torch.tensor([np.double(label)])       
        
        return Data(x=torch.from_numpy(x), edge_index=edge_index, y=y)

    return [make_graph(i) for i in tqdm(range(n_samples))]

# %%
batch_size=1024
pion_graphs = CreateGraphDataset("E:\ML_data/vt/data/slow_pions_evtgen_big.txt", 50000, 1.0)
bg_graphs = CreateGraphDataset("E:\ML_data/vt/data/protons_big.txt", 50000, 0.0)
data_list = pion_graphs + bg_graphs
#data_list = data_list[data_list]
random.shuffle(data_list)
train_loader = DataLoader(data_list[:80000], batch_size=32, shuffle=True)
val_loader = DataLoader(data_list[80000:], batch_size=32)

# %%

class Net(torch.nn.Module):
    def __init__(self, n_node_feats, dim_out):
        super(Net, self).__init__()
        self.conv1 = ARMAConv(n_node_feats, 32)
        self.conv2 = ARMAConv(32, 32)
        self.conv3 = ARMAConv(32, 32)
        self.dense = torch.nn.Linear(32, dim_out)
        self.double()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        #x = F.dropout(x, training=self.training)
        out = global_mean_pool(x, data.batch)
        out = F.sigmoid(self.dense(out))
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(3, 1).to(device) # .float()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def train():
    model.train()
    all_loss = 0
    i = 0.0
    for data in train_loader:
        data = data.to(device)
        output = model(data)
        loss = F.binary_cross_entropy(output, data.y.unsqueeze(1), reduction="mean")
        all_loss += loss.item()
        i += 1.0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print('Accuracy: ', (torch.argmax(output, dim=1)==data.y.unsqueeze(1)).float().mean())
    return(all_loss/i)

def test():
    model.eval()
    correct = 0

    for data in val_loader:
        data = data.to(device)
        pred = model(data).max(1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / 5000.0


for epoch in range(1, 31):
    train_loss = train()
    # test_acc = test()
    print(f'Epoch: {epoch:02d}, loss: {train_loss:.5f}')



# %%
