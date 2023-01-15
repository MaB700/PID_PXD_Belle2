import numpy as np
import torch
from torch_geometric.loader import DataLoader

from helpers_gnn import CreateGraphDataset
from meta_layer import *

data_test = torch.load('./dataset_fully_test.pt')

# data_test = CreateGraphDataset("E:\ML_data/vt/data/slow_pions_evtgen_big.txt", 900, 1.0) \
#          + CreateGraphDataset("E:\ML_data/vt/data/electron_big.txt", 900, 0.0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNN(data=data_test[0], n_layers=3, hidden_nodes=64, residuals=True, normalize=True).to(device)
model.load_state_dict(torch.load('./model_best.pt'))
model.eval()

test_loader = DataLoader(data_test, batch_size=16)

def saliency_map(model, data):    
    data.requires_grad_(*['x', 'edge_attr', 'u'], True)
    output = model(data)
    dx, de, du = torch.autograd.grad(output, 
                                    inputs=[data.x, data.edge_attr, data.u],
                                    grad_outputs=torch.ones_like(output),
                                    retain_graph=True)

    return dx, de, du

def calc_grads(loader):
    # du_sum = torch.tensor(0.0, dtype=torch.float64, device=device)
    du_sum = [torch.tensor(0.0, dtype=torch.float64, device=device) for _ in range(4)]
    # du_sqr_sum = torch.tensor(0.0, dtype=torch.float64, device=device)
    du_sqr_sum = [torch.tensor(0.0, dtype=torch.float64, device=device) for _ in range(4)]
    num_samples = [0, 0, 0, 0]
    for data in loader :
        data = data.to(device)
        _, _, du = saliency_map(model, data)
        _, num_pixels = torch.unique(data.batch, return_counts=True)
        num_samples[0] += num_pixels.shape[0]
        num_samples[1] += num_pixels[num_pixels==1].shape[0]
        num_samples[2] += num_pixels[num_pixels==2].shape[0]
        num_samples[3] += num_pixels[num_pixels>=3].shape[0]

        du_mean_0 = torch.mean(du, dim=0)
        du_sqr_0 = torch.mean(torch.square(du), dim=0)
        du_sum[0] = du_sum[0] + du_mean_0
        du_sqr_sum[0] = du_sqr_sum[0] + du_sqr_0

        if num_pixels[num_pixels==1].shape[0] > 0:
            du_mean_1 = torch.mean(du[num_pixels==1], dim=0)
            du_sqr_1 = torch.mean(torch.square(du[num_pixels==1]), dim=0)
            du_sum[1] = du_sum[1] + du_mean_1
            du_sqr_sum[1] = du_sqr_sum[1] + du_sqr_1
        
        if num_pixels[num_pixels==2].shape[0] > 0:
            du_mean_2 = torch.mean(du[num_pixels==2], dim=0)
            du_sqr_2 = torch.mean(torch.square(du[num_pixels==2]), dim=0)
            du_sum[2] = du_sum[2] + du_mean_2
            du_sqr_sum[2] = du_sqr_sum[2] + du_sqr_2
        
        if num_pixels[num_pixels>=3].shape[0] > 0:
            du_mean_3p = torch.mean(du[num_pixels>=3], dim=0)
            du_sqr_3p = torch.mean(torch.square(du[num_pixels>=3]), dim=0)
            du_sum[3] = du_sum[3] + du_mean_3p
            du_sqr_sum[3] = du_sqr_sum[3] + du_sqr_3p
        
        # du_mean = [ torch.mean(du[num_pixels==1], dim=0),
        #             torch.mean(du[num_pixels==2], dim=0),
        #             torch.mean(du[num_pixels>=3], dim=0)]
        # print(du_mean[2])
        # du_sqr =  [ torch.mean(torch.square(du[num_pixels==1]), dim=0),
        #             torch.mean(torch.square(du[num_pixels==2]), dim=0),
        #             torch.mean(torch.square(du[num_pixels>=3]), dim=0)]
        
        # for i in range(3):
        #     du_sum[i] = du_sum[i] + du_mean[i]
        #     du_sqr_sum[i] = du_sqr_sum[i] + du_sqr[i]
        
        # du_sum = du_sum + du_mean
        # du_sqr_sum = du_sqr_sum + du_sqr

    du_meanx = [du_sum[i].cpu().detach().numpy() / num_samples[i] for i in range(4)]
    du_sqr_mean = [du_sqr_sum[i].cpu().detach().numpy() / num_samples[i] for i in range(4)]
    print(num_samples)
    # du_mean = du_sum.cpu().detach().numpy() / i1
    # du_sqr_mean = du_sqr_sum.cpu().detach().numpy() / i1
    # du_var = du_sqr_mean - (du_mean * du_mean)
    du_var = [du_sqr_mean[i] - (du_meanx[i] * du_meanx[i]) for i in range(4)]
    return np.array(du_meanx), np.sqrt(du_var)

grads_mean, grads_sigma = calc_grads(test_loader)
print(grads_mean.T)
print(grads_sigma.T)
# add dimension to grads_mean
# grads_mean = np.expand_dims(grads_mean, axis=1)
# grads_sigma = np.expand_dims(grads_sigma, axis=1)

# plot grads_mean in a 2D 1x9 grid with mathplotlib and the values shown in the color dimension
import matplotlib.pyplot as plt
fig = plt.figure()
axes = [fig.add_subplot(1, 2, i) for i in range(1, 3)]


max_grads_mean = np.amax(grads_mean)
im1 = axes[0].imshow(grads_mean.T, cmap='RdBu', vmin=-max_grads_mean, vmax=max_grads_mean, interpolation='none')
cb0 = fig.colorbar(im1, ax=axes[0])
cb0.set_label(r'$\overline{\frac{\partial O}{\partial u_{k}}} $', loc='top', rotation=0, labelpad=20, size=15)
axes[0].set_title('Mean of gradients')
axes[0].set_xlabel('Number of pixels in 9x9 window')
axes[0].set_ylabel(r'global input features $u_{k}$')
axes[0].set_xticks(axes[0].get_xticks())
axes[0].set_xticklabels(['', 'all', '1', '2', '3+', ''])
#axes[0].set_yticks(axes[0].get_yticks())
axes[0].set_yticklabels(['', 'r_radial', 'r_cyl', 'sin(phi_cyl)', 'z_cyl', 'cog_x', 'cog_y', 'avg_charge', 'seed_charge', 'min_charge', ''])


max_grads_sigma = np.amax(grads_sigma)
im2 = axes[1].imshow(grads_sigma.T, cmap='RdBu', vmin=-max_grads_sigma, vmax=max_grads_sigma, interpolation='none')
cb1 = fig.colorbar(im2, ax=axes[1])
cb1.set_label(r'$\sigma(\frac{\partial O}{\partial u_{k}}) $', loc='top', rotation=0, labelpad=30, size=15)
axes[1].set_title('Sigma of gradients')
axes[1].set_xlabel('Number of pixels in 9x9 window')
axes[1].set_ylabel(r'global input features $u_{k}$')
axes[1].set_xticks(axes[1].get_xticks())
axes[1].set_xticklabels(['', 'all', '1', '2', '3+', ''])
#axes[1].set_yticks(axes[1].get_yticks())
axes[1].set_yticklabels(['', 'r_radial', 'r_cyl', 'sin(phi_cyl)', 'z_cyl', 'cog_x', 'cog_y', 'avg_charge', 'seed_charge', 'min_charge', ''])

plt.savefig('saliency_map_u.pdf')
plt.show(block=True)
