import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch_geometric.data.data import Data
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ARMAConv, GENConv, GeneralConv, global_mean_pool
from torch_geometric.loader import DataLoader

from sklearn.metrics import roc_auc_score, confusion_matrix
import wandb

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
            edge_index = (torch.from_numpy(np.row_stack(([0], [0])))).long()
        
        # Labels
        y = torch.tensor([np.double(label)])       
        
        return Data(x=torch.from_numpy(x), edge_index=edge_index, y=y)

    return [make_graph(i) for i in tqdm(range(n_samples))]


class Net(torch.nn.Module):
    def __init__(self, n_node_feats, dim_out, hidden_nodes):
        super(Net, self).__init__()
        self.conv1 = GENConv(n_node_feats, hidden_nodes)
        self.conv2 = GENConv(hidden_nodes, hidden_nodes)
        self.conv3 = GENConv(hidden_nodes, hidden_nodes)
        self.dense = Linear(hidden_nodes, dim_out)
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


class LogWandb():
    def __init__(self, data, model, device, batch_size):
        self.model = model
        self.device = device
        self.loaders = [DataLoader(data, batch_size=batch_size), \
                        DataLoader([d for d in data if d.num_nodes == 1], batch_size=batch_size), \
                        DataLoader([d for d in data if d.num_nodes == 2], batch_size=batch_size), \
                        DataLoader([d for d in data if d.num_nodes > 2], batch_size=batch_size) ]
        self.log_all()

    def predict(self, loader):
        self.model.eval()
        tar = np.empty((0))
        prd = np.empty((0))
        for data in loader :
            data = data.to(self.device)
            pred = self.model(data).squeeze(1).cpu().detach().numpy()
            target = data.y.cpu().detach().numpy()
            tar = np.append(tar, target)
            prd = np.append(prd, np.array(pred))
        return tar, prd

    def log_all(self):
        
        def run(i):
            gt, pred = self.predict(self.loaders[i])
            auc = roc_auc_score(gt, pred)
            wandb.log({"test_auc{:d}".format(i): auc})

            tn, fp, fn, tp = confusion_matrix(y_true=[1 if a_ > 0.5 else 0 for a_ in gt], \
                                              y_pred=[1 if a_ > 0.5 else 0 for a_ in pred]).ravel()
            wandb.log({"test_prec{:d}".format(i): tp/(tp+fn)}) # efficiency
            wandb.log({"test_sens{:d}".format(i): tp/(tp+fp)}) # purity

            wandb.log({"cm{:d}".format(i): wandb.plot.confusion_matrix(   probs=None, 
                                                y_true=[1 if a_ > 0.5 else 0 for a_ in gt], 
                                                preds=[1 if a_ > 0.5 else 0 for a_ in pred], 
                                                class_names=["background", "signal"],
                                                title="CM{:d}".format(i))})

            wandb.log({"roc{:d}".format(i): wandb.plot.roc_curve( gt, 
                                        np.concatenate(((1-pred).reshape(-1,1),pred.reshape(-1,1)),axis=1), 
                                        classes_to_plot=[1],
                                        title="ROC{:d}".format(i))})


        for i in range(4):
            run(i)

