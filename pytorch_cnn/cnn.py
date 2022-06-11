# %%
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader, Subset, ConcatDataset
from helpers_cnn import *

wandb.init(project="PXD_SP", notes='CNN' ) #  , mode="disabled"

# %%
batch_size = 1024
epochs = 10
es_patience = 5
nEventsEach = 500000
data = ConcatDataset([  CreateCNNDataset("E:\ML_data/vt/data/slow_pions_evtgen_big.txt", nEventsEach, 1.0),
                        CreateCNNDataset("E:\ML_data/vt/data/electron_big.txt", nEventsEach, 0.0)])
np.random.seed(123)
idxs = np.random.permutation(data.__len__())
idx_train, idx_val, idx_test = np.split(idxs, [int(0.44 * data.__len__()), int(0.61 * data.__len__())])

train_loader = DataLoader(Subset(data, idx_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(Subset(data, idx_val), batch_size=batch_size)
# test_loader = DataLoader(Subset(data, idx_test), batch_size=batch_size)
# TODO: add wandb and split into cluster sizes
# something like [d for d in Subset(data, idx_test) if torch.count_nonzero(d['x'] > 0.001) == 1]
# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
# %%
def train_step():
    model.train()
    all_loss = 0
    i = 0.0
    for (idx, batch) in enumerate(train_loader):
        x = batch['x'].to(device)
        y = batch['y'].to(device)
        output = model(x)
        loss = F.binary_cross_entropy(output, y, reduction="mean")
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
    for (idx, batch) in enumerate(loader):
        x = batch['x'].to(device)
        y = batch['y'].to(device)
        output = model(x)
        loss = F.binary_cross_entropy(output, y, reduction="mean")
        all_loss += loss.item()
        i += 1.0
    
    return all_loss/i

best_val_loss = np.inf
patience = es_patience

for epoch in range(1, epochs + 1):
    train_loss = train_step()
    val_loss = evaluate(val_loader)
    print(f'Epoch: {epoch:02d}, loss: {train_loss:.5f}, val_loss: {val_loss:.5f}')
    #wandb.log({ "train_loss": train_loss, "val_loss": val_loss })

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

model.load_state_dict(torch.load('./model_best.pt'))
model.eval()
LogWandbCNN(Subset(data, idx_test), model, device, 1024)