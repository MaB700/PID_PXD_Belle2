import numpy as np
import pandas as pd
import math
from tqdm import tqdm
import torch
from torch_geometric.data.data import Data
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geonn
from torch_geometric.loader import DataLoader

from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
import wandb

def CreateGraphDataset(path, n_samples, label, dist = 3):        
    """ Args:  dist : connect pixels in a (2*dist+1)x(2*dist+1) kernel
    
    """
    data = pd.read_csv(path, header=None, delimiter= " ", nrows=n_samples).values.astype('int')
    global_pos = pd.read_csv(path, header=None, delimiter= " ", nrows=n_samples).values[:, 83:86]
    
    def get_angle(a, b):
        nom = a[0]*b[0] + a[1]*b[1]
        denom = math.sqrt( (a[0]**2 + a[1]**2)*(b[0]**2 + b[1]**2) )
        if denom < 0.00001:
            return 1
        result = nom/denom
        return result # = cos(phi) 
    
    def make_graph(index):     
        # Load num filled pixels & ADC values
        n = np.count_nonzero(data[index, 2:83] > 0)
        adc_indices = np.nonzero(data[index, 2:83])
        adc = data[index, 2:83][adc_indices]
        seed_index = np.argmax(adc) # TODO: can 2+ pixels have the adc of seed pixel ?
        minvalue_index = np.argmin(adc)
        y_pos, x_pos = np.divmod(adc_indices, 9)
        
        x_grad = 0.0
        y_grad = 0.0
        for i in range(n):
            if i == seed_index :
                continue
            r_2 = (x_pos[0, i] - x_pos[0, seed_index])**2 + (y_pos[0, i] - y_pos[0, seed_index])**2
            adc_diff = - adc[i] + adc[seed_index] # -grad
            x_grad += (adc_diff*(x_pos[0, i]-x_pos[0, seed_index])/8.0)/r_2
            y_grad += (adc_diff*(y_pos[0, i]-y_pos[0, seed_index])/8.0)/r_2

        grad_norm = math.sqrt(x_grad**2 + y_grad**2) #TODO: normalize and use norm as global feature
        
        # Node features
        x = np.zeros((n, 3)) # num node features
        x[:, 0] = (adc.astype('float64'))/255.0 # adc [0, 1]
        x[:, 1] = (x_pos.astype('float64'))/8.0 # x_coord [0,1]
        x[:, 2] = (y_pos.astype('float64'))/8.0 # y_coord [0,1]
        
        # Edges
        start_index = np.empty((0), dtype=np.int_)
        end_index = np.empty((0), dtype=np.int_)
        if n > 1 :
            ###### Seed connected ######
            # a = np.full((n - 1), seed_index)
            # b = np.delete(np.arange(0, n), seed_index)
            # # a = np.full((n), seed_index)
            # # b = np.arange(0, n)
            # edge_index = np.row_stack((np.concatenate((a,b)), np.concatenate((b,a))))   
            # edge_index = torch.from_numpy(edge_index)     
            
            ###### Fully connected ######
            # for i in range(n):
            #     for j in range(n):
            #         if i == j :
            #             continue
            #         start_index = np.append(start_index, i)
            #         end_index = np.append(end_index, j)
            # edge_index = np.row_stack((start_index, end_index))
            # edge_index = torch.from_numpy(edge_index).long()

            ###### Distance connected ######
            for i in range(n):
                for j in range(n):
                    if i == j :
                        continue
                    if (x_pos[0, i]-x_pos[0, j]) <= dist and (y_pos[0, i]-y_pos[0, j]) <= dist :
                        start_index = np.append(start_index, i) #TODO: replace np.append since it's a copy each call and slow
                        end_index = np.append(end_index, j)
            edge_index = np.row_stack((start_index, end_index))
            edge_index = torch.from_numpy(edge_index).long()
        else :
            edge_index = (torch.from_numpy(np.row_stack(([0], [0])))).long()
        
        # Edge features
        edge_features = np.zeros((edge_index.shape[1], 3))
        if n > 1 :
            for i in range(edge_index.shape[1]):
                x0 = x[edge_index[0, i].item(), 1]
                y0 = x[edge_index[0, i].item(), 2]
                x1 = x[edge_index[1, i].item(), 1]
                y1 = x[edge_index[1, i].item(), 2]
                edge_features[i, 0] = math.sqrt( ((x1-x0)**2 + (y1-y0)**2)/2.0 ) # node distance [0, 1]
                edge_features[i, 1] = x[edge_index[1, i].item(), 0] - x[edge_index[0, i].item(), 0] # adc diff [-1, 1]
                edge_features[i, 2] = get_angle([x1-x0, y1-y0], [x_grad, y_grad])
            
        edge_features = torch.from_numpy(edge_features)
        
        
        # Labels
        y = torch.tensor([np.double(label)])

        # Graph features
        # x [-2, 2] y [-2, 2] z [-8 , 8] r [0, 2]
        r_radial = math.sqrt(global_pos[index, 0]**2 + global_pos[index, 1]**2 + global_pos[index, 2]**2)
        r_cyl = math.sqrt(global_pos[index, 0]**2 + global_pos[index, 1]**2)
        r_cyl = 0.0 if r_cyl < 2 else 1.0
        phi_cyl = np.arctan2(global_pos[index, 1], global_pos[index, 0])
        z_cyl = (global_pos[index, 2])/8.0
        tot_charge = 0
        cog_x = 0 # center of gravity x
        cog_y = 0
        for i in range(n):
            tot_charge += adc[i]
            cog_x += adc[i]*x[i, 1]
            cog_y += adc[i]*x[i, 2]
        cog_x = cog_x/tot_charge
        cog_y = cog_y/tot_charge
        avg_charge = tot_charge/(n*255.0)
        seed_charge = adc[seed_index]/255.0
        min_charge = adc[minvalue_index]/255.0
        u = torch.tensor([r_radial, r_cyl, math.sin(phi_cyl), z_cyl, cog_x, cog_y, avg_charge, seed_charge, min_charge]).unsqueeze(0) # [0 or 1], [-1, 1], [-1, 1]

        return Data(x=torch.from_numpy(x), edge_index=edge_index, edge_attr=edge_features, y=y, u=u)

    return [make_graph(i) for i in tqdm(range(len(data)))]


# GNN using global features only in the FC classification layer -> not used!
class Net(torch.nn.Module):
    def __init__(self, data, hidden_nodes, dense_nodes, edge_list, global_list):
        super(Net, self).__init__()
        self.edge_list = edge_list
        self.global_list = global_list

        self.node_encoder = nn.Linear(data.x.size(-1), hidden_nodes)
        self.edge_encoder = nn.Linear(len(edge_list), hidden_nodes)
        
        self.conv1 = geonn.GENConv(hidden_nodes, hidden_nodes)
        self.conv2 = geonn.GENConv(hidden_nodes, hidden_nodes)
        self.conv3 = geonn.GENConv(hidden_nodes, hidden_nodes)
        
        self.dense1 = nn.Linear(hidden_nodes + len(global_list), dense_nodes)
        self.do1 = nn.Dropout(0.2)
        self.dense2 = nn.Linear(dense_nodes, dense_nodes)
        self.do2 = nn.Dropout(0.2)
        self.denseOut = nn.Linear(dense_nodes, data.y.size(-1))
        self.double()

    def forward(self, data):
        x, edge_index, edge_attr, graph_attr = \
            data.x, data.edge_index, data.edge_attr[:, self.edge_list], data.graph_attr[:, self.global_list]
        
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = F.relu(self.conv3(x, edge_index, edge_attr))
        #x = F.dropout(x, training=self.training)
        # TODO: other ways the pool ?
        x = torch.cat((geonn.global_mean_pool(x, data.batch), graph_attr), dim=1)
        x = F.relu(self.dense1(x))
        # x = self.do1(x)
        x = F.relu(self.dense2(x))
        # x = self.do2(x)
        return torch.sigmoid(self.denseOut(x))


class LogWandb():
    def __init__(self, data, model, device, batch_size):
        self.model = model
        self.device = device
        self.loaders = [DataLoader(data, batch_size=batch_size), \
                        DataLoader([d for d in data if d.num_nodes == 1], batch_size=batch_size), \
                        DataLoader([d for d in data if d.num_nodes == 2], batch_size=batch_size), \
                        DataLoader([d for d in data if d.num_nodes > 2], batch_size=batch_size) ]
        test_loss = self.evaluate(self.loaders[0])
        wandb.log({"test_loss": test_loss})
        self.log_all()

    def evaluate(self, loader):
        self.model.eval()
        all_loss = 0
        i = 0.0
        for data in loader:
            data = data.to(self.device)
            output = self.model(data)
            loss = F.binary_cross_entropy(output, data.y.unsqueeze(1), reduction="mean")
            all_loss += loss.item()
            i += 1.0
    
        return all_loss/i

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
            wandb.log({"test_sens{:d}".format(i): tp/(tp+fn)})
            wandb.log({"test_prec{:d}".format(i): tp/(tp+fp)})

            wandb.log({"cm{:d}".format(i): wandb.plot.confusion_matrix(   probs=None,
                                                y_true=[1 if a_ > cut_value else 0 for a_ in gt],
                                                preds=[1 if a_ > cut_value else 0 for a_ in pred],
                                                class_names=["background", "signal"],
                                                title="CM{:d}".format(i))})
            len_roc = len(gt) if len(gt) <= 10000 else 10000
            wandb.log({"roc{:d}".format(i): wandb.plot.roc_curve( gt[:len_roc], 
                                        np.concatenate(((1-pred[:len_roc]).reshape(-1,1),pred[:len_roc].reshape(-1,1)),axis=1),
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

