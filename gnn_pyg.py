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
epochs = 0
es_patience = 10

reload_dataset = True
if os.path.exists('./dataset_fully_train.pt') and os.path.exists('./dataset_fully_test.pt') and not reload_dataset:
    data = torch.load('./dataset_fully_train.pt', map_location=torch.device('cpu'))
else:
    data = CreateGraphDataset("E:\ML_data/vt/data/slow_pions_evtgen_big.txt", 9000, 1.0) \
         + CreateGraphDataset("E:\ML_data/vt/data/electron_big.txt", 9000, 0.0)
    # data = CreateGraphDataset("sp_balanced_train.txt", 450000, 1.0) \
    #      + CreateGraphDataset("bg_balanced_train.txt", 450000, 0.0)    
    random.shuffle(data)
    # torch.save(data[:110000], './dataset_fully_train.pt')
    torch.save(data[11000:], './dataset_fully_test.pt')
    data = data[:11000]
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
# %%
test_loader = DataLoader(data_test, batch_size=128)
model.eval()
# Calculate the saliency map
def saliency_map(model, data):    
    data.requires_grad_(*['x', 'edge_attr', 'u'], True)
    output = model(data)
    # grads = torch.autograd.functional.jacobian(lambda x, e, u: model.forward(x, e, u, data.edge_index, data.batch), 
    #                                     (data.x, data.edge_attr, data.u))
    dx, de, du = torch.autograd.grad(output, 
                                inputs=[data.x, data.edge_attr, data.u],
                                grad_outputs=torch.ones_like(output),
                                retain_graph=True)

    return dx, de, du

def calc_grads(loader):
    du_sum = torch.tensor(0.0, dtype=torch.float64, device=device)
    du_sqr_sum = torch.tensor(0.0, dtype=torch.float64, device=device)
    i = 0
    du_x = np.array(())
    adc_x = np.array(())
    for data in loader :
        data = data.to(device)
        dx, de, du = saliency_map(model, data)
        # append all values of du[:, 7] into a numpy array
        # du_x = np.append(du_x, du[:, 7].cpu().detach().numpy().flatten())
        # adc_x = np.append(adc_x, data.u[:, 7].cpu().detach().numpy().flatten())
        du_mean = torch.mean(du, dim=0)
        du_sum = du_sum + du_mean
        du_sqr = torch.mean(torch.square(du), dim=0)
        du_sqr_sum = du_sqr_sum + du_sqr
        i += 1

    du_mean = du_sum.cpu().detach().numpy() / i
    du_sqr_mean = du_sqr_sum.cpu().detach().numpy() / i
    du_var= du_sqr_mean - (du_mean * du_mean)
    return du_mean, np.sqrt(du_var)

# # measure time to calculate the saliency map
# import time
# start = time.time()
grads_mean, grads_sigma = calc_grads(test_loader)
print(grads_mean)
print(grads_sigma)
# add dimension to grads_mean
grads_mean = np.expand_dims(grads_mean, axis=1)
grads_sigma = np.expand_dims(grads_sigma, axis=1)
# end = time.time()
# print(end - start)

# plot grads_mean in a 2D 1x9 grid with mathplotlib and the values shown in the color dimension
import matplotlib.pyplot as plt
fig = plt.figure()
axes = [fig.add_subplot(1, 2, i) for i in range(1, 3)]

max_grads_mean = np.amax(grads_mean)
im1 = axes[0].imshow(grads_mean, cmap='RdBu', vmin=-max_grads_mean, vmax=max_grads_mean, interpolation='none')
fig.colorbar(im1, ax=axes[0])
axes[0].set_xticks([])
axes[0].set_title('Mean of gradients (dO/du_i)')

max_grads_sigma = np.amax(grads_sigma)
im2 = axes[1].imshow(grads_sigma, cmap='RdBu', vmin=-max_grads_sigma, vmax=max_grads_sigma, interpolation='none')
fig.colorbar(im2, ax=axes[1])
axes[1].set_xticks([])
axes[1].set_title('Sigma of gradients (dO/du_i)')


# fig, (ax1, ax2) = plt.subplots(figsize=(10, 3), ncols=2)
# im1 = ax1.imshow(grads_mean, cmap='RdBu', vmin=-max_grads_mean, vmax=max_grads_mean, interpolation='none')
# fig.colorbar(im1, ax=ax1)
# ax1.xticks([])

# max_grads_sigma = np.amax(grads_sigma)
# im2 = ax2.imshow(grads_sigma, cmap='RdBu', vmin=-max_grads_sigma, vmax=max_grads_sigma, interpolation='none')
# fig.colorbar(im2, ax=ax2)
# ax2.xticks([])
#im2.clim(-max_grads_sigma*1.1, max_grads_sigma*1.1)

plt.show(block=True)







# # plot du against adc using matplotlib
# import matplotlib.pyplot as plt
# plt.scatter(adc, du)
# plt.xlabel('ADC')
# plt.ylabel('du')
# plt.show(block=True)

# %%
