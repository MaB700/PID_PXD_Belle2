import numpy as np
import pandas as pd
import math
from tqdm import tqdm
import torch
from torch_geometric.data.data import Data
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ARMAConv, GENConv, GeneralConv, global_mean_pool
from torch_geometric.loader import DataLoader

from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
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
        # x_grad, y_grad = 0.0
        # for i in range(n):
        #     if i == seed_index :
        #         continue
        #     r_2 = (x_pos[i] - x_pos[seed_index])**2 + (y_pos[i] - y_pos[seed_index])**2
        #     adc_diff = - adc[i] + adc[seed_index] # -grad
        #     x_grad += (adc_diff*(x_pos[i]-x_pos[seed_index])/8.0)/r_2
        #     y_grad += (adc_diff*(y_pos[i]-y_pos[seed_index])/8.0)/r_2
        
        # Node features
        x = np.zeros((n, 3)) # num node features
        x[:, 0] = (adc.astype('float64'))/255.0 # adc [0, 1]
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
        
        # Edge features
        edge_features = np.zeros((edge_index.shape[1], 2))
        if n > 1 :
            for i in range(edge_index.shape[1]):
                x0 = x[edge_index[0, i].item(), 1] # i in range(0, edges)
                y0 = x[edge_index[0, i].item(), 2]
                x1 = x[edge_index[1, i].item(), 1]
                y1 = x[edge_index[1, i].item(), 2]
                edge_features[i, 0] = math.sqrt( ((x1-x0)**2 + (y1-y0)**2)/2.0 ) # node distance [0, 1]
                edge_features[i, 1] = x[edge_index[1, i].item(), 0] - x[edge_index[0, i].item(), 0] # adc diff [-1, 1]
            
        edge_features = torch.from_numpy(edge_features)
        
        
        # Labels
        y = torch.tensor([np.double(label)])       
        
        return Data(x=torch.from_numpy(x), edge_index=edge_index, edge_attr=edge_features, y=y)

    return [make_graph(i) for i in tqdm(range(n_samples))]


class Net(torch.nn.Module):
    def __init__(self, data, dim_out, hidden_nodes):
        super(Net, self).__init__()
        
        self.node_encoder = Linear(data.x.size(-1), hidden_nodes)
        self.edge_encoder = Linear(data.edge_attr.size(-1), hidden_nodes)
        
        self.conv1 = GENConv(hidden_nodes, hidden_nodes)
        self.conv2 = GENConv(hidden_nodes, hidden_nodes)
        self.conv3 = GENConv(hidden_nodes, hidden_nodes)
        self.dense = Linear(hidden_nodes, dim_out)
        self.double()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.relu(self.conv3(x, edge_index, edge_attr))
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

        def run(i, cut_value):
            gt, pred = self.predict(self.loaders[i])
            auc = roc_auc_score(gt, pred)
            wandb.log({"test_auc{:d}".format(i): auc})
            tn, fp, fn, tp = confusion_matrix(y_true=[1 if a_ > cut_value else 0 for a_ in gt], \
                                              y_pred=[1 if a_ > cut_value else 0 for a_ in pred]).ravel()
            wandb.log({"test_prec{:d}".format(i): tp/(tp+fn)}) # efficiency
            wandb.log({"test_sens{:d}".format(i): tp/(tp+fp)}) # purity

            wandb.log({"cm{:d}".format(i): wandb.plot.confusion_matrix(   probs=None,
                                                y_true=[1 if a_ > cut_value else 0 for a_ in gt],
                                                preds=[1 if a_ > cut_value else 0 for a_ in pred],
                                                class_names=["background", "signal"],
                                                title="CM{:d}".format(i))})

            wandb.log({"roc{:d}".format(i): wandb.plot.roc_curve( gt, 
                                        np.concatenate(((1-pred).reshape(-1,1),pred.reshape(-1,1)),axis=1), 
                                        classes_to_plot=[1],
                                        title="ROC{:d}".format(i))})

        def calcCut():
            gt, pred = self.predict(self.loaders[0])
            fpr, tpr, ths = roc_curve(gt, pred)
            index = np.argmax(tpr - fpr)
            cut = ths[index]
            print("best cut value: {:.3f}".format(cut))
            return cut
        
        cut = calcCut()
        wandb.log({"cut_value": cut})
        
        for i in range(4):
            run(i, cut)

