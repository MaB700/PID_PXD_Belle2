# %%
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from sklearn.metrics import roc_auc_score

import networkx as nx
from torch_geometric.utils.convert import to_networkx

from helpers import CreateGraphDataset, Net

# %%
batch_size = 1024
epochs = 50
es_patience = 5
nEventsEach = 100000

data = CreateGraphDataset("E:\ML_data/vt/data/slow_pions_evtgen_big.txt", nEventsEach, 1.0) \
     + CreateGraphDataset("E:\ML_data/vt/data/protons_big.txt", nEventsEach, 0.0)

np.random.seed(123)
idxs = np.random.permutation(len(data))
idx_train, idx_val, idx_test = np.split(idxs, [int(0.6 * len(data)), int(0.8 * len(data))])

train_loader = DataLoader([data[index] for index in idx_train], batch_size=batch_size, shuffle=True)
val_loader = DataLoader([data[index] for index in idx_val], batch_size=batch_size)
test_loader = DataLoader([data[index] for index in idx_test], batch_size=batch_size)
test_loader1 = DataLoader([data[index] for index in idx_test if data[index].num_nodes == 1], batch_size=batch_size)
test_loader2 = DataLoader([data[index] for index in idx_test if data[index].num_nodes == 2], batch_size=batch_size)
test_loader3 = DataLoader([data[index] for index in idx_test if data[index].num_nodes > 2], batch_size=batch_size)
# %%

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(3, 1).to(device) # .float()
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

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

    if val_loss < best_val_loss :
        best_val_loss = val_loss
        patience = es_patience
        print("New best val_loss {:.4f}".format(val_loss))
        torch.save(model.state_dict(), 'model_best.pt')
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

model.load_state_dict(torch.load('model_best.pt'))
model.eval()
loss_test = evaluate(test_loader)
print("Test_loss {:.4f}".format(loss_test))

test_gt, test_pred = predict(test_loader)
test_auc = roc_auc_score(test_gt, test_pred)
print("Test AUC: {:.4f}".format(test_auc))
# %%
