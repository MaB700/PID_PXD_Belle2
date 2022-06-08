# %%
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.nn import Linear, Conv2d, Flatten, Dropout, BatchNorm1d, BatchNorm2d
from torch.utils.data import DataLoader
from helpers_cnn import *


# %%
data_pion = CreateCNNDataset("E:\ML_data/vt/data/slow_pions_evtgen_big.txt", 500000, 1.0)
data_e = CreateCNNDataset("E:\ML_data/vt/data/electron_big.txt", 500000, 0.0)
data_comb = torch.utils.data.ConcatDataset([data_pion, data_e])
train_loader = DataLoader(data_comb, batch_size=64, shuffle=True)
# %%
batch_size = 512
epochs = 50
es_patience = 50


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = Conv2d(1, 5, 3, padding='same')
        self.conv2 = Conv2d(5, 5, 3, padding='same')
        self.conv3 = Conv2d(5, 5, 3, padding='same')

        self.dense1 = Linear(5*9*9, 405)
        self.dense2 = Linear(405, 405)
        self.dense3 = Linear(405, 405)
        self.dense4 = Linear(405, 405)
        self.dense5 = Linear(405, 405)
        self.denseOut = Linear(405, 1)

        self.dropout = Dropout(0.5)
        self.double()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.dense1(x))
        x = self.dropout(x)
        x = F.relu(self.dense2(x))
        x = self.dropout(x)
        x = F.relu(self.dense3(x))
        x = self.dropout(x)
        x = F.relu(self.dense4(x))
        x = self.dropout(x)
        x = F.relu(self.dense5(x))
        x = self.dropout(x)
        return torch.sigmoid(self.denseOut(x))

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
# model = model.to(torch.float)
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# %%
def train_step():
    model.train()
    all_loss = 0
    i = 0.0
    for (x, y) in train_loader:
        x = x.to(device)
        y = y.to(device)
        output = model(x)
        loss = F.binary_cross_entropy(output, y.unsqueeze(1), reduction="mean")
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
        loss = F.binary_cross_entropy(output, data.y, reduction="mean")
        all_loss += loss.item()
        i += 1.0
    
    return all_loss/i

best_val_loss = np.inf
patience = es_patience

for epoch in range(1, epochs + 1):
    train_loss = train_step()
    # val_loss = evaluate(val_loader)
    val_loss = 1.0
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