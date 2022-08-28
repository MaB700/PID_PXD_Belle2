# %%
import numpy as np
import random
import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from helpers_gnn import CreateGraphDataset, LogWandb
from meta_layer import *

import wandb
wandb.init(project="PXD_SP", mode='disabled') #   
# %%
batch_size = 512
epochs = 200
es_patience = 10

reload_dataset = True
if os.path.exists('./dataset_fully_train.pt') and os.path.exists('./dataset_fully_test.pt') and not reload_dataset:
    data = torch.load('./dataset_fully_train.pt', map_location=torch.device('cpu'))
else:
    data = CreateGraphDataset("E:\ML_data/vt/data/slow_pions_evtgen_big.txt", 900000, 1.0) \
         + CreateGraphDataset("E:\ML_data/vt/data/electron_big.txt", 900000, 0.0)
    # data = CreateGraphDataset("sp_balanced_train.txt", 450000, 1.0) \
    #      + CreateGraphDataset("bg_balanced_train.txt", 450000, 0.0)    
    random.shuffle(data)
    # torch.save(data[:110000], './dataset_fully_train.pt')
    torch.save(data[1100000:], './dataset_fully_test.pt')
    data = data[:1100000]
# %%
np.random.seed(123)
idx_train, idx_val = np.split(np.random.permutation(len(data)), [int(0.727 * len(data))])
train_loader = DataLoader([data[index] for index in idx_train], batch_size=batch_size, shuffle=True)
val_loader = DataLoader([data[index] for index in idx_val], batch_size=batch_size)
# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNN(data=data[0], n_layers=3, hidden_nodes=64, residuals=True, normalize=True).to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[47, 60, 80], gamma=0.5, verbose=True)

def train_step():
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
    #scheduler.step()
    return all_loss/i

def evaluate(loader):
    model.eval()
    all_loss = 0
    i = 0.0
    for data in loader:
        data = data.to(device)
        output = model(data)
        loss = F.binary_cross_entropy(output, data.y.unsqueeze(1), reduction="mean")
        all_loss += loss.item()
        i += 1.0
    
    return all_loss/i

best_val_loss = np.inf
patience = es_patience

for epoch in range(1, epochs + 1):
    train_loss = train_step()
    val_loss = evaluate(val_loader)
    print(f'Epoch: {epoch:02d}, loss: {train_loss:.5f}, val_loss: {val_loss:.5f}')
    wandb.log({ "train_loss": train_loss, "val_loss": val_loss })

    if val_loss < best_val_loss :
        best_val_loss = val_loss
        patience = es_patience
        print("New best val_loss {:.4f}".format(val_loss))
        torch.save(model.state_dict(), './model_best.pt')
    else :
        patience -= 1
        if patience == 0:
            print("Early stopping (best val_loss: {})".format(best_val_loss))
            break
    
def predict(loader):
    model.eval()
    tar = np.empty((0))
    prd = np.empty((0))
    for data in loader :
        data = data.to(device)
        pred = model(data).squeeze(1).cpu().detach().numpy()
        target = data.y.cpu().detach().numpy()
        tar = np.append(tar, target)
        prd = np.append(prd, np.array(pred))
    return tar, prd

# %%
del data
data_test = torch.load('./dataset_fully_test.pt')
# data_test = CreateGraphDataset("sp_test.txt", 450000, 1.0) \
#           + CreateGraphDataset("bg_test.txt", 450000, 0.0)
model.load_state_dict(torch.load('./model_best.pt'))
model.eval()
LogWandb(data_test, model, device, 1024)