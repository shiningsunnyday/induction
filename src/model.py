import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch.utils.data import Dataset
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import from_networkx
from functools import reduce
import numpy as np
from src.config import *

class GraphDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]

def create_model(num_node_features, hidden_channels):
    class GCN(torch.nn.Module):
        def __init__(self, num_node_features, hidden_channels):
            super(GCN, self).__init__()
            self.conv1 = GCNConv(num_node_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.lin = torch.nn.Linear(hidden_channels, 1)

        def forward(self, x, edge_index, batch):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            x = global_mean_pool(x, batch) 
            return self.lin(x)

    return GCN(num_node_features, hidden_channels)

def label_to_one_hot(node_dict):
    index = (TERMS+NONTERMS).index(node_dict['label'])
    one_hot = torch.zeros((len(TERMS+NONTERMS,)))
    one_hot[index] = 1.
    return one_hot

def my_from_networkx(G):    
    keys = reduce(lambda x,y:x&y, (set(G.nodes[n]) for n in G))
    for n in G:
        for k in list(G.nodes[n]):
            if k not in keys:
                G.nodes[n].pop(k)
    for e in G.edges:
        for k in list(G.edges[e]):
            G.edges[e].pop(k)
    return from_networkx(G)


def convert_networkx_to_pyg_data(graphs):
    pyg_data_list = []
    for i, G in enumerate(graphs):
        for node in G.nodes():
            G.nodes[node]['x'] = [label_to_one_hot(G.nodes[node]),
                                  [G.nodes[node]['feat'] if 'feat' in G.nodes[node] else 0.0]
            ]            
            G.nodes[node]['x'] = torch.from_numpy(np.concatenate(G.nodes[node]['x'], dtype=np.float32))
        data = my_from_networkx(G)
        data.x = torch.stack(data.x)
        data.y = torch.tensor([G.graph[f'{i}:fom']], dtype=torch.float) 
        pyg_data_list.append(data)
    return pyg_data_list


def train(model, loader, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        for batch in loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = F.mse_loss(out, batch.y) 
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

def graph_regression(graphs):
    pyg_data_list = convert_networkx_to_pyg_data(graphs)
    pyg_dataset = GraphDataset(pyg_data_list)
    loader = DataLoader(pyg_dataset, batch_size=2)
    model = create_model(num_node_features=pyg_dataset[0].x.shape[-1], hidden_channels=16)
    optimizer = torch.optim.Adam(model.parameters())
    train(model, loader, optimizer, num_epochs=10000)

def transformer_regression(graphs):
    breakpoint()