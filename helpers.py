import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch_geometric.data.data import Data
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ARMAConv, global_mean_pool

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


class Net(torch.nn.Module):
    def __init__(self, n_node_feats, dim_out):
        super(Net, self).__init__()
        self.conv1 = ARMAConv(n_node_feats, 32)
        self.conv2 = ARMAConv(32, 32)
        self.conv3 = ARMAConv(32, 32)
        self.dense = Linear(32, dim_out)
        self.double()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))
        #x = F.dropout(x, training=self.training)
        out = global_mean_pool(x, data.batch)
        out = torch.sigmoid(self.dense(out))
        return out