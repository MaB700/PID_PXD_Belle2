import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, ModuleList, Sigmoid
from torch_geometric.nn import MessagePassing, MetaLayer, LayerNorm
from torch_scatter import scatter_mean, scatter_sum, scatter_max, scatter_min, scatter_add
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

class EdgeModel(torch.nn.Module):
    def __init__(self, node_in, node_out, edge_in, edge_out, global_in, global_out, hid_channels, residuals=True, norm=False):
        super().__init__()

        self.residuals = residuals
        self.norm = norm

        layers = [Linear(node_in*2 + edge_in + global_in, hid_channels),
                  ReLU(),
                  Linear(hid_channels, edge_out)]
        if self.norm:  layers.append(LayerNorm(edge_out))

        self.edge_mlp = Sequential(*layers)
        self.double()

    def forward(self, src, dest, edge_attr, u, batch):
        # src, dest: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: [E] with max entry B - 1.
        out = torch.cat([src, dest, edge_attr, u[batch]], dim=1)
        #out = torch.cat([src, dest, edge_attr, u[batch]], 1)
        out = self.edge_mlp(out)
        if self.residuals:
            out += edge_attr
        return out

class NodeModel(torch.nn.Module):
    def __init__(self, node_in, node_out, edge_in, edge_out, global_in, global_out, hid_channels, residuals=True, norm=False):
        super().__init__()

        self.residuals = residuals
        self.norm = norm

        layers = [Linear(node_in + 3*edge_out + global_in, hid_channels),
                  ReLU(),
                  Linear(hid_channels, node_out)]
        if self.norm:  layers.append(LayerNorm(node_out))

        self.node_mlp = Sequential(*layers)
        self.double()

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        row, col = edge_index
        out = edge_attr

        # Multipooling layer
        out1 = scatter_add(out, col, dim=0, dim_size=x.size(0))
        out2 = scatter_max(out, col, dim=0, dim_size=x.size(0))[0]
        out3 = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out1, out2, out3, u[batch]], dim=1)

        out = self.node_mlp(out)
        if self.residuals:
            out += x
        return out

class GlobalModel(torch.nn.Module):
    def __init__(self, node_in, node_out, edge_in, edge_out, global_in, global_out, hid_channels, residuals=True, norm=False):
        super().__init__()
        
        self.residuals = residuals
        self.norm = norm
        
        layers = [Linear(node_out + global_in, hid_channels),
                  ReLU(),
                  Linear(hid_channels, global_out)]
        if self.norm:  layers.append(LayerNorm(global_out))

        self.global_mlp = Sequential(*layers)
        self.double()

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.
        out = torch.cat([u, scatter_mean(x, batch, dim=0)], dim=1)
        out = self.global_mlp(out)
        if self.residuals:
            out += u
        return out

class GNN(torch.nn.Module):
    def __init__(self, data, n_layers, hidden_nodes, residuals=True, normalize=False):
        super().__init__()

        self.node_in = data.x.size(-1) # node features
        self.edge_in = data.edge_attr.size(-1) # edge features
        self.global_in = data.u.size(-1) # global features
        self.dim_out = data.y.size(-1) # target size
        self.n_layers = n_layers
        self.hidden_nodes = hidden_nodes
    
        node_out = self.hidden_nodes
        edge_out = self.hidden_nodes
        global_out = self.hidden_nodes
        lin_nodes = self.hidden_nodes
        layers = []

        # Encoder graph block
        encodeLayer = MetaLayer(node_model=NodeModel(self.node_in, node_out, None, edge_out, self.global_in, None, lin_nodes, residuals=False, norm=normalize),
                                edge_model=EdgeModel(self.node_in, None, self.edge_in, edge_out, self.global_in, None, lin_nodes, residuals=False, norm=normalize), \
                                global_model=GlobalModel(self.node_in, node_out, None, edge_out, self.global_in, global_out, lin_nodes, residuals=False, norm=normalize))

        layers.append(encodeLayer)

        # Change input node and edge feature sizes
        node_in = node_out
        edge_in = edge_out
        global_in = global_out

        # Hidden graph blocks
        for i in range(n_layers-1):

            l = MetaLayer(node_model=NodeModel(node_in, node_out, edge_in, edge_out, global_in, global_out, lin_nodes, residuals=residuals, norm=normalize),
                            edge_model=EdgeModel(node_in, node_out, edge_in, edge_out, global_in, global_out, lin_nodes, residuals=residuals, norm=normalize), \
                            global_model=GlobalModel(node_in, node_out, edge_in, edge_out, global_in, global_out, lin_nodes, residuals=residuals, norm=normalize))
            layers.append(l)

        self.layers = ModuleList(layers)

        # Final aggregation layer
        self.outlayer = Sequential(Linear(3*node_out + global_out, lin_nodes),
                              ReLU(),
                              Linear(lin_nodes, lin_nodes),
                              ReLU(),
                              # Linear(lin_nodes, lin_nodes),
                              # ReLU(),
                              Linear(lin_nodes, self.dim_out),
                              Sigmoid())
        self.double()

    def forward(self, data): #data

        h, edge_index, edge_attr, u = data.x, data.edge_index, data.edge_attr, data.u
        batch = data.batch
        # Message passing layers
        for layer in self.layers:
            h, edge_attr, u = layer(h, edge_index, edge_attr, u, batch)

        # Multipooling layer
        addpool = global_add_pool(h, batch)
        meanpool = global_mean_pool(h, batch)
        maxpool = global_max_pool(h, batch)

        out = torch.cat([addpool,meanpool,maxpool,u], dim=1)

        # Classification layer
        out = self.outlayer(out)

        return out