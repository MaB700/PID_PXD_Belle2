import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve

import wandb

class CreateCNNDataset(Dataset):

    def __init__(self, path, n_samples, label):
        data = pd.read_csv(path, header=None, delimiter= " ", nrows=n_samples).values.astype('float64')
        x = torch.from_numpy(data[:, 2:83]/255.0)
        y = torch.full((n_samples, 1), label).double()

        self.x = torch.reshape(x, (n_samples, 1, 9, 9))
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample = {"x": self.x[idx], "y": self.y[idx]}
        return sample

    def __num_nodes__(self, idx):
        return torch.count_nonzero(self.x[idx] > 0.001)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        lin_nodes = 405
        dropout = 0.2
        self.conv1 = nn.Sequential( nn.Conv2d(1, 5, 3, padding='same'),
                                    nn.BatchNorm2d(5),
                                    # nn.ReplicationPad2d(1),
                                    nn.ReLU() )
        self.conv2 = nn.Sequential( nn.Conv2d(5, 5, 3, padding='same'),
                                    nn.BatchNorm2d(5),
                                    # nn.ReplicationPad2d(1),
                                    nn.ReLU() )
        self.conv3 = nn.Sequential( nn.Conv2d(5, 5, 3, padding='same'),
                                    nn.BatchNorm2d(5),
                                    # nn.ReplicationPad2d(1),
                                    nn.ReLU() )

        self.lin1 = nn.Sequential(  nn.Linear(5*9*9, lin_nodes),
                                    nn.BatchNorm1d(lin_nodes),
                                    nn.Dropout(dropout),
                                    nn.ReLU())
        self.lin2 = nn.Sequential(  nn.Linear(lin_nodes, lin_nodes),
                                    nn.BatchNorm1d(lin_nodes),
                                    nn.Dropout(dropout),
                                    nn.ReLU())
        self.lin3 = nn.Sequential(  nn.Linear(lin_nodes, lin_nodes),
                                    nn.BatchNorm1d(lin_nodes),
                                    nn.Dropout(dropout),
                                    nn.ReLU())
        self.lin4 = nn.Sequential(  nn.Linear(lin_nodes, lin_nodes),
                                    nn.BatchNorm1d(lin_nodes),
                                    nn.Dropout(dropout),
                                    nn.ReLU())
        self.lin5 = nn.Sequential(  nn.Linear(lin_nodes, lin_nodes),
                                    nn.BatchNorm1d(lin_nodes),
                                    nn.Dropout(dropout),
                                    nn.ReLU())

        self.denseOut = nn.Linear(lin_nodes, 1)
        self.double()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.lin3(x)
        x = self.lin4(x)
        x = self.lin5(x)
        return torch.sigmoid(self.denseOut(x))


class LogWandbCNN():
    def __init__(self, data, model, device, batch_size):
        self.model = model
        self.device = device
        self.loaders = [DataLoader(data, batch_size=batch_size), \
                        DataLoader([d for d in data if torch.count_nonzero(d['x'] > 0.001) == 1], batch_size=batch_size), \
                        DataLoader([d for d in data if torch.count_nonzero(d['x'] > 0.001) == 2], batch_size=batch_size), \
                        DataLoader([d for d in data if torch.count_nonzero(d['x'] > 0.001) > 2], batch_size=batch_size) ]
        test_loss = self.evaluate(self.loaders[0])
        wandb.log({"test_loss": test_loss})
        self.log_all()

    def evaluate(self, loader):
        self.model.eval()
        all_loss = 0
        i = 0.0
        for (idx, batch) in enumerate(loader):
            x = batch['x'].to(self.device)
            y = batch['y'].to(self.device)
            output = self.model(x)
            loss = F.binary_cross_entropy(output, y, reduction="mean")
            all_loss += loss.item()
            i += 1.0
    
        return all_loss/i
    
    def predict(self, loader):
        self.model.eval()
        tar = np.empty((0))
        prd = np.empty((0))
        for (idx, batch) in enumerate(loader):
            x = batch['x'].to(self.device)
            y = batch['y'].to(self.device)
            pred = self.model(x).squeeze(1).cpu().detach().numpy() # TODO: check dim
            target = y.cpu().detach().numpy() # use batch['y'] directly
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