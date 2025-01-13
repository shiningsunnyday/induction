import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
from sklearn.manifold import TSNE
import networkx as nx
import numpy as np
from scipy.spatial.distance import pdist
import scipy.stats as sps
from scipy.stats import pearsonr
# torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data as GraphData
import importlib
dagnn = importlib.import_module('dagnn.ogbg-code.model.dagnn')
utils_dag = importlib.import_module('dagnn.src.utils_dag')
DAGNN = dagnn.DAGNN
# if error, go to induction/dagnn/ogbg-code/model/dagnn.py, change a line to "from dagnn.src.constants import *""
import torch.nn.functional as F
torch.multiprocessing.set_sharing_strategy('file_system')
import pickle
import hashlib
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.config import *
from src.grammar.utils import get_next_version
from src.grammar.ednce import *
from src.examples.test_graphs import *
from src.draw.graph import draw_graph
import glob
import re

import sys
sys.path.append('dagnn/dvae/bayesian_optimization')
from sparse_gp import SparseGP

# Grammar
VOCAB_SIZE = 204  # Number of rules
# MAX_RULE_SIZE = 5 # Max size of rule's rhs graph
SEQ_LEN = 13  # Max length of sequences
NUM_SAMPLES = 3
# NN
LATENT_DIM = 256
ENCODER_LAYERS = 4
DECODER_LAYERS = 4
GNN = False
# Training
BATCH_SIZE = 256
EPOCHS = 230
CUDA = 'cuda:0'

# Generate a random vocabulary of small graphs using NetworkX
def generate_random_graphs(vocab_size):
    graphs = []
    for _ in range(vocab_size):
        num_nodes = np.random.randint(3, 6)  # Random number of nodes in each graph token
        G = nx.newman_watts_strogatz_graph(num_nodes, 3, p=0.5)
        for (u, v) in G.edges():
            G.edges[u, v]['weight'] = np.random.rand()  # Random edge weights
        graphs.append(G)
    return graphs

def hash_object(obj):
    """Create a deterministic hash for a Python object."""
    obj_bytes = pickle.dumps(obj)
    return hashlib.sha256(obj_bytes).hexdigest()

def top_sort(edge_index, graph_size):

    node_ids = np.arange(graph_size, dtype=int)

    node_order = np.zeros(graph_size, dtype=int)
    unevaluated_nodes = np.ones(graph_size, dtype=bool)

    parent_nodes = edge_index[0].numpy()
    child_nodes = edge_index[1].numpy()

    n = 0
    while unevaluated_nodes.any():
        # Find which parent nodes have not been evaluated
        unevaluated_mask = unevaluated_nodes[parent_nodes]
        
        # Find the child nodes of unevaluated parents
        unready_children = child_nodes[unevaluated_mask]

        # Mark nodes that have not yet been evaluated
        # and which are not in the list of children with unevaluated parent nodes
        nodes_to_evaluate = unevaluated_nodes & ~np.isin(node_ids, unready_children)

        node_order[nodes_to_evaluate] = n
        unevaluated_nodes[nodes_to_evaluate] = False

        n += 1

    return torch.from_numpy(node_order).long()


def assert_order(edge_index, o, ns):
    # already processed
    proc = []
    for i in range(max(o)+1):
        # nodes in position i in order
        l = o == i
        l = ns[l].tolist()
        for n in l:
            # predecessors
            ps = edge_index[0][edge_index[1] == n].tolist()
            for p in ps:
                assert p in proc
        proc += l    


# to be able to use pyg's batch split everything into 1-dim tensors
def add_order_info_01(graph):

    l0 = top_sort(graph.edge_index, graph.num_nodes)
    ei2 = torch.LongTensor([list(graph.edge_index[1]), list(graph.edge_index[0])])
    l1 = top_sort(ei2, graph.num_nodes)
    ns = torch.LongTensor([i for i in range(graph.num_nodes)])
    graph.__setattr__("_bi_layer_idx0", l0)
    graph.__setattr__("_bi_layer_index0", ns)
    graph.__setattr__("_bi_layer_idx1", l1)
    graph.__setattr__("_bi_layer_index1", ns)

    assert_order(graph.edge_index, l0, ns)
    assert_order(ei2, l1, ns)    

def to_one_hot(y, labels):
    one_hot_vector = torch.zeros((len(labels),))
    one_hot_vector[labels.index(y)] = 1.
    return one_hot_vector


# Convert NetworkX graphs to PyTorch Geometric Data objects
def convert_graph_to_data(graph):
    # Random node features
    graph = nx.relabel_nodes(graph, dict(zip(graph,range(len(graph)))))
    features = []
    term = True
    for i in range(len(graph)):           
        one_hot_vector = to_one_hot(graph.nodes[i]['label'], TERMS+NONTERMS)
        if 'feat' in graph.nodes[i]:
            feat_val = graph.nodes[i]['feat']
        else:
            feat_val = 0.
        if one_hot_vector.argmax().item() >= len(TERMS):
            term = False # nonterm node
        feat = torch.cat((one_hot_vector, torch.tensor([feat_val])))
        features.append(feat)
    x = torch.stack(features, dim=0)
    # x = torch.rand((graph.number_of_nodes(), EMBED_DIM))    
    edge_list = list(graph.edges)
    edge_index = torch.tensor(edge_list).t().contiguous()
    roots = np.setdiff1d(np.arange(len(graph)), edge_index[1])    
    dists = [nx.single_source_shortest_path_length(graph, root) for root in roots]
    dist = {d: min([dis[d] for dis in dists if d in dis]) for d in range(len(graph))}
    node_depth = [dist[i] for i in range(len(graph))]
    edge_attr = torch.stack([to_one_hot(graph.edges[edge]['label'], FINAL+NONFINAL) for edge in edge_list], dim=0)
    g = GraphData(x=x, edge_index=edge_index, node_depth=torch.tensor(node_depth), edge_attr=edge_attr)
    add_order_info_01(g) # dagnn
    g.len_longest_path = float(torch.max(g._bi_layer_idx0).item()) # dagnn
    return g, term

def add_ins(g, ins):
    if ins is None:
        return g
    for mu, p, q, i, d, d_ in ins:        
        n = len(g)
        g.add_node(n, label=mu)
        if d_ == 'out':
            g.add_edge(list(g)[i], n, label=q)        
        else:
            g.add_edge(n, list(g)[i], label=q)
    return g

# Prepare the graph vocabulary
graph_vocabulary = None # generate_random_graphs(VOCAB_SIZE)
graph_data_vocabulary = None
vocabulary_terminate = None


class TokenDataset(Dataset):
    def __init__(self, data):
        self.dataset = []
        self.data = data
        for idx in range(len(data)):
            seq = []
            graph_seq = {}
            for i in range(len(self.data[idx])):
                r, g, ins = self.data[idx][i]
                g = add_ins(g, ins)
                seq.append(r)
                graph, _ = convert_graph_to_data(g)
                graph_seq[i] = graph      
            self.dataset.append((torch.tensor(seq), graph_seq, idx))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.dataset[idx]


# Define GNN-based Token Embedding
class TokenGNN(nn.Module):
    def __init__(self, embed_dim):
        super(TokenGNN, self).__init__()
        self.conv1 = GCNConv(embed_dim, embed_dim)
        self.conv2 = GCNConv(embed_dim, embed_dim)
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, graph_data):
        x, edge_index = graph_data.x, graph_data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.pooling(x.t()).squeeze(-1)  # Aggregate node features to a single vector
        return x


# Define Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_layers=2):
        super(TransformerEncoder, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4),
            num_layers=num_layers
        )

    def forward(self, x, mask=None, src_key_padding_mask=None):
        return self.encoder(x, mask=mask, src_key_padding_mask=src_key_padding_mask)

# Define VAE
class TransformerVAE(nn.Module):
    def __init__(self, embed_dim, latent_dim, seq_len):
        super(TransformerVAE, self).__init__()        
        # self.token_gnn = TokenGNN(embed_dim)
        if GNN:
            self.gnn = DAGNN(None, None, 13, LATENT_DIM, None, w_edge_attr=False, bidirectional=False, num_class=LATENT_DIM)
        else:
            self.token_embedding = nn.Embedding(VOCAB_SIZE, embed_dim)
        self.transformer_encoder = TransformerEncoder(embed_dim, num_layers=ENCODER_LAYERS)
        self.fc_mu = nn.Linear(embed_dim, latent_dim)
        self.fc_logvar = nn.Linear(embed_dim, latent_dim)
        
        self.fc_decode = nn.Linear(latent_dim, latent_dim)
        self.transformer_decoder = TransformerEncoder(embed_dim + latent_dim, num_layers=DECODER_LAYERS)
        self.output_layer = nn.Linear(embed_dim + latent_dim, VOCAB_SIZE)
        self.start_token_embedding = nn.Parameter(torch.randn(1, 1, embed_dim))


    def visualize_tokens(self):
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(self.token_embedding.weight.detach().cpu())
        # Plot the 2D t-SNE
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
        ax.set_title("2D t-SNE of nn.Embedding")
        ax.set_xlabel("t-SNE Dim 1")
        ax.set_ylabel("t-SNE Dim 2")
        return fig


    def embed_tokens(self, token_ids, batch_g_list):
        if not GNN:
            return self.token_embedding(token_ids)
        if token_ids.dim() == 2:
            return torch.stack([self.embed_tokens(token_id_seq, g_list) for (token_id_seq, g_list) in zip(token_ids, batch_g_list)], dim=0)
        else:
            g_list = batch_g_list
        embedded_tokens = []
        g_list = [g_list[i] for i in range(len(g_list))]
        for token_id, graph_data in zip(token_ids, g_list):
            if graph_data is None:
                embedded_token = torch.zeros((LATENT_DIM,), device=CUDA)
            else:
                graph_data.to(CUDA)
                embedded_token = self.gnn(graph_data).flatten()
            embedded_tokens.append(embedded_token)
        return torch.stack(embedded_tokens, dim=0)
        # return torch.cat(embedded_tokens, dim=0)

    def encode(self, x, attention_mask):        
        encoded_seq = self.transformer_encoder(x.permute(1, 0, 2), src_key_padding_mask=~attention_mask).permute(1, 0, 2)
        # Apply mean pooling along the sequence length dimension to get a fixed-size representation
        pooled = torch.mean(encoded_seq, dim=1)  # (batch, embed_dim)
        mu, logvar = self.fc_mu(pooled), self.fc_logvar(pooled)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def autoregressive_training_step(self, x, z, max_seq_len_list):
        # Initialize with the learned start token embedding and z context
        batch_size = z.size(0)
        z_context = self.fc_decode(z)  # Shape (batch, latent_dim)
        
        # Initialize embeddings for each sequence using the start token
        token_embeddings = list(torch.unbind(self.start_token_embedding.expand(batch_size, -1, -1), dim=0))  # Shape (batch, 1, embed_dim)
        generated_logits = [[] for _ in range(batch_size)]  # Store logits for each sequence

        max_seq_len = max(max_seq_len_list)

        # Process each sequence individually, applying teacher forcing
        for t in range(max_seq_len):
            # Prepare a temporary batch of active sequences at the current timestep
            temp_batch_embeddings = []
            temp_z_context = []
            active_indices = []

            for idx in range(batch_size):
                # Check if the current sequence needs more tokens (based on its max_seq_len)
                if t < max_seq_len_list[idx]:
                    # Get accumulated embeddings up to this step
                    accumulated_embeddings = token_embeddings[idx]  # Concatenate embeddings so far
                    temp_batch_embeddings.append(accumulated_embeddings)
                    temp_z_context.append(z_context[idx].unsqueeze(0).expand(accumulated_embeddings.size(0), -1))
                    active_indices.append(idx)

            # If there are no active sequences, we’re done
            if not active_indices:
                break
            
            # Convert lists to tensors for batched transformer input
            temp_batch_embeddings = torch.stack(temp_batch_embeddings)
            temp_z_context = torch.stack(temp_z_context)

            # Concatenate accumulated embeddings with z_context
            transformer_input = torch.cat([temp_batch_embeddings, temp_z_context], dim=-1)

            # Apply causal mask for autoregressive decoding on the current batch
            seq_len = transformer_input.size(1)
            causal_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=1).to(z.device)

            output = self.transformer_decoder(
                transformer_input.permute(1, 0, 2),  # Shape (seq_len, batch, embed_dim + latent_dim)
                mask=causal_mask
            ).permute(1, 0, 2)
            

            # Predict logits for the next token
            logits = self.output_layer(output[:, -1, :])  # Only the last token's output
            for i, idx in enumerate(active_indices):
                generated_logits[idx].append(logits[i].unsqueeze(0))  # Collect logits for each sequence

            # Use ground-truth tokens from x as the next input (teacher forcing)
            for i, idx in enumerate(active_indices):
                if t < max_seq_len_list[idx]:  # Ensure there are more ground-truth tokens to process
                    # Get the ground-truth token from x and embed it using the GNN
                    # .unsqueeze(0)  # Ground truth for next token
                    next_token_embedding = x[idx, t].unsqueeze(0)
                    # next_token_embedding = self.gnn(graph_data_vocabulary[next_token_id.item()]).unsqueeze(0)                    
                    # next_token_embedding = batch_g_list[idx][t]
                    # Append the embedded token to this sequence’s token_embeddings list
                    token_embeddings[idx] = torch.cat((token_embeddings[idx], next_token_embedding), dim=0)

        # Concatenate logits across time steps for each sequence to compute the loss                        
        recon_logits_list = []
        for seq_logits in generated_logits:
            seq_logits_cat = torch.cat(seq_logits, dim=0)
            padding = torch.zeros(max_seq_len - len(seq_logits), VOCAB_SIZE, device=seq_logits_cat.device)
            recon_logits = torch.cat((seq_logits_cat, padding), dim=0)
            recon_logits_list.append(recon_logits)

        padded_logits = torch.stack(recon_logits_list, dim=0)
        mask = torch.zeros((padded_logits.shape[0], max_seq_len), dtype=torch.bool, device=padded_logits.device)
        for i, logits in enumerate(generated_logits):
            mask[i, :len(logits)] = 1  # Mark valid positions up to the length of each sequence
        return padded_logits.view(-1, VOCAB_SIZE), mask
    

    def _autoregressive_inference_active_indices(self, z_context, generated_sequences, token_embeddings, max_seq_len):
        batch_size = z_context.size(0)
        # Prepare a temporary batch of active sequences at the current timestep
        temp_batch_embeddings = []
        temp_z_context = []
        active_indices = []
        for idx in range(batch_size):
            # Check if the sequence needs more tokens
            if len(generated_sequences[idx]) < max_seq_len:
                if len(generated_sequences[idx]):
                    last_token = generated_sequences[idx][-1]
                    if vocabulary_terminate[last_token]:
                        continue
                accumulated_embeddings = token_embeddings[idx]  # Use accumulated embeddings up to this step
                temp_batch_embeddings.append(accumulated_embeddings)
                temp_z_context.append(z_context[idx].unsqueeze(0).expand(accumulated_embeddings.size(0), -1))
                active_indices.append(idx)        
        return temp_batch_embeddings, temp_z_context, active_indices


    def _autoregressive_inference_predict_logits(self, temp_batch_embeddings, temp_z_context):        
        # Convert lists to tensors for batched transformer input
        temp_batch_embeddings = torch.stack(temp_batch_embeddings)
        temp_z_context = torch.stack(temp_z_context)
        device = temp_z_context.device

        # Concatenate accumulated embeddings with z_context
        transformer_input = torch.cat([temp_batch_embeddings, temp_z_context], dim=-1)

        # Apply causal mask for autoregressive decoding
        seq_len = transformer_input.size(1)
        causal_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=1).to(device)

        output = self.transformer_decoder(
            transformer_input.permute(1, 0, 2),  # Shape (seq_len, batch, embed_dim + latent_dim)
            mask=causal_mask
        ).permute(1, 0, 2)

        # Predict logits for the next token and sample from the distribution
        logits = self.output_layer(output[:, -1, :])  # Only the last token's output
        return logits



    def autoregressive_inference(self, z, max_seq_len):
        # Initialize with the start token embedding and z context
        batch_size = z.size(0)
        z_context = self.fc_decode(z)  # Shape (batch, latent_dim)

        # Initialize embeddings for each sequence using the start token
        token_embeddings = list(torch.unbind(self.start_token_embedding.expand(batch_size, -1, -1), dim=0))  # Shape (batch, 1, embed_dim)
        generated_sequences = [[] for _ in range(batch_size)]  # Store generated tokens for each sequence

        for t in range(max_seq_len):
            temp_batch_embeddings, temp_z_context, active_indices = self._autoregressive_inference_active_indices(z_context, generated_sequences, token_embeddings, max_seq_len)
            # If no active sequences remain, end generation
            if not active_indices:
                break
            logits = self._autoregressive_inference_predict_logits(temp_batch_embeddings, temp_z_context)            
            next_tokens = torch.argmax(logits, dim=-1)  # Shape (batch,)

            # Update each active sequence with the newly generated token
            for i, idx in enumerate(active_indices):
                generated_sequences[idx].append(next_tokens[i].item())  # Store generated token
                if GNN:
                    next_token_embedding = self.gnn(graph_data_vocabulary[next_tokens[i].item()])
                else:
                    next_token_embedding = self.token_embedding(next_tokens[i])
                # Append the new embedding to this sequence’s token embeddings
                token_embeddings[idx] = torch.cat((token_embeddings[idx], next_token_embedding), dim=0)
        return generated_sequences


    def forward(self, x, attention_mask, seq_len_list, batch_g_list):
        embedded_tokens = self.embed_tokens(x, batch_g_list)  # Embeds each token (graph) using GNN
        mu, logvar = self.encode(embedded_tokens, attention_mask)
        z = self.reparameterize(mu, logvar)
        logits, logits_mask = self.autoregressive_training_step(embedded_tokens, z, seq_len_list)
        return logits, logits_mask, mu, logvar


# Loss function
def vae_loss(recon_logits, mask, x, mu, logvar):    
    x_flat = x.view(-1)
    recon_loss = F.cross_entropy(recon_logits, x_flat, reduction="none")
    recon_loss = recon_loss.view(x.size(0), -1)
    recon_loss = (recon_loss * mask).sum() / mask.sum()
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_divergence / x.size(0)


# Padding function
def collate_batch(batch):    
    lengths = [len(seq) for seq, _, _ in batch]
    max_len = max(lengths)
    padded_batch = torch.zeros(len(batch), max_len, dtype=torch.long)
    seq_len_list = torch.zeros(len(batch), dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    batch_g_list = []
    idxes = []
    for i, (seq, g_list, idx) in enumerate(batch):
        padded_batch[i, :len(seq)] = seq
        seq_len_list[i] = len(seq)  # Mask non-padded positions
        attention_mask[i, :len(seq)] = 1
        g_list.update({i: None for i in range(len(g_list), max_len)})
        batch_g_list.append(g_list)  
        idxes.append(idx)
    return padded_batch, attention_mask, seq_len_list, batch_g_list, idxes


# Sampling new sequences
def sample(model, num_samples=5, max_seq_len=10):
    model.eval()
    uniq_sequences = set()
    with torch.no_grad():
        while len(uniq_sequences) < num_samples:
            z = torch.randn(num_samples, LATENT_DIM)  # Sample from the prior
            z = z.to(CUDA)
            generated_sequences = model.autoregressive_inference(z, max_seq_len)  # Decode from latent space            
            uniq_sequences = uniq_sequences | set(map(tuple, generated_sequences))
    uniq_sequences = [list(l) for l in uniq_sequences]
    return uniq_sequences[:num_samples]


def decode_from_latent_space(z, model, max_seq_len=10):
    generated_sequences = model.autoregressive_inference(z, max_seq_len)
    return generated_sequences
    

def train(train_data, test_data):
    # Initialize model and optimizer
    model = TransformerVAE(LATENT_DIM, LATENT_DIM, SEQ_LEN)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Dummy dataset: Replace with actual sequence data
    # dataset = [torch.tensor(seq) for seq in data]
    train_dataset = TokenDataset(train_data)
    # train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch, num_workers=8, timeout=1000, prefetch_factor=1)
    test_dataset = TokenDataset(test_data)
    # test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch, num_workers=8, timeout=1000, prefetch_factor=1)    

    # Training loop
    model.train()
    model = model.to(CUDA)
    if GNN:
        for i, graph_data in enumerate(graph_data_vocabulary):
            graph_data_vocabulary[i] = graph_data.to(CUDA)
    ckpts = glob.glob('ckpts/api_ckt_ednce/*.pth')
    start_epoch = 0
    best_loss = float("inf")
    best_ckpt_path = None
    for ckpt in ckpts:
        epoch = int(re.findall('epoch=(\d+)',ckpt)[0])
        loss = float(re.findall('loss=([\d.]+).pth',ckpt)[0])
        if loss < best_loss:
            best_loss = loss
            start_epoch = epoch + 1
            best_ckpt_path = ckpt    
    if best_ckpt_path is not None:
        print(f"loaded {best_ckpt_path} loss {best_loss} start_epoch {start_epoch}")
        model.load_state_dict(torch.load(best_ckpt_path))
    
    train_latent = np.empty((len(train_data), LATENT_DIM))
    test_latent = np.empty((len(test_data), LATENT_DIM))
    for epoch in tqdm(range(start_epoch, EPOCHS+1)):
        model.train()
        train_loss = 0.
        g_batch = []
        for i in tqdm(range(len(train_dataset))):
            g_batch.append(train_dataset[i])
            if len(g_batch) == BATCH_SIZE or i == len(train_dataset)-1:
                batch = collate_batch(g_batch)
                x, attention_mask, seq_len_list, batch_g_list, batch_idxes = batch
                x, attention_mask = x.to(CUDA), attention_mask.to(CUDA)
                optimizer.zero_grad()
                recon_logits, mask, mu, logvar = model(x, attention_mask, seq_len_list, batch_g_list)            
                loss = vae_loss(recon_logits, mask, x, mu, logvar)
                loss.backward()            
                train_loss += loss.item()*len(batch_idxes)
                optimizer.step()
                train_latent[batch_idxes] = mu.detach().cpu().numpy()
                g_batch = []
        train_loss /= len(train_dataset)
        model.eval()
        val_loss = 0.
        g_batch = []
        for i in tqdm(range(len(test_dataset))):
            g_batch.append(test_dataset[i])
            if len(g_batch) == BATCH_SIZE or i == len(test_dataset)-1:
                batch = collate_batch(g_batch)
                x, attention_mask, seq_len_list, batch_g_list, batch_idxes = batch
                x, attention_mask = x.to(CUDA), attention_mask.to(CUDA)
                optimizer.zero_grad()
                recon_logits, mask, mu, logvar = model(x, attention_mask, seq_len_list, batch_g_list)            
                loss = vae_loss(recon_logits, mask, x, mu, logvar)            
                loss.backward()
                val_loss += loss.item()*len(batch_idxes)
                test_latent[batch_idxes] = mu.detach().cpu().numpy()
                optimizer.step()      
        val_loss /= len(test_dataset)
        if val_loss < best_loss:
            best_loss = val_loss
            ckpt_path = f'ckpts/api_ckt_ednce/epoch={epoch}_loss={best_loss}.pth'
            torch.save(model.state_dict(), ckpt_path)
            print(ckpt_path)
        print(f"Epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}")
        np.save(f'ckpts/api_ckt_ednce/train_latent_{epoch}.npy', train_latent)
        np.save(f'ckpts/api_ckt_ednce/test_latent_{epoch}.npy', test_latent)
        fig = model.visualize_tokens()
        fig.savefig(f'ckpts/{epoch}.png')        
        embedding = model.token_embedding.weight.detach().cpu().numpy()
        np.save(f'ckpts/api_ckt_ednce/embedding_{epoch}.npy', embedding)
    return model



def bo(args, model, y_train, y_test):
    ckpt_dir = 'ckpts/api_ckt_ednce/'    
    X_train = np.load(os.path.join(ckpt_dir, f"train_latent_{args.checkpoint}.npy"))    
    X_test = np.load(os.path.join(ckpt_dir, f"test_latent_{args.checkpoint}.npy"))    
    iteration = 0
    best_score = 1e15
    best_arc = None
    best_random_score = 1e15
    best_random_arc = None
    print("Average pairwise distance between train points = {}".format(np.mean(pdist(X_train))))
    print("Average pairwise distance between test points = {}".format(np.mean(pdist(X_test))))    
    while iteration < args.BO_rounds:
        print("Iteration", iteration)
        if args.predictor:
            pred = model.predictor(torch.FloatTensor(X_test).to(CUDA))
            pred = pred.detach().cpu().numpy()
            pred = (-pred - mean_y_train) / std_y_train
            uncert = np.zeros_like(pred)
        else:
            # We fit the GP
            M = 50
            # other BO hyperparameters
            lr = 0.0005  # the learning rate to train the SGP model
            max_iter = 500  # how many iterations to optimize the SGP each time
            sgp = SparseGP(X_train, 0 * X_train, y_train, M)
            sgp.train_via_ADAM(X_train, 0 * X_train, y_train, X_test, X_test * 0,  \
                y_test, minibatch_size = 2 * M, max_iterations = max_iter, learning_rate = lr)
            pred, uncert = sgp.predict(X_test, 0 * X_test)

        print("predictions: ", pred.reshape(-1))
        print("real values: ", y_test.reshape(-1))
        error = np.sqrt(np.mean((pred - y_test)**2))
        testll = np.mean(sps.norm.logpdf(pred - y_test, scale = np.sqrt(uncert)))
        print('Test RMSE: ', error)
        print('Test ll: ', testll)
        pearson = float(pearsonr(pred.flatten(), y_test.flatten())[0])
        print('Pearson r: ', pearson)
        with open('results/' + 'Test_RMSE_ll.txt', 'a') as test_file:
            test_file.write('Test RMSE: {:.4f}, ll: {:.4f}, Pearson r: {:.4f}\n'.format(error, testll, pearson))

        error_if_predict_mean = np.sqrt(np.mean((np.mean(y_train, 0) - y_test)**2))
        print('Test RMSE if predict mean: ', error_if_predict_mean)
        if args.predictor:
            pred = model.predictor(torch.FloatTensor(X_train).to(CUDA))
            pred = pred.detach().cpu().numpy()
            pred = (-pred - mean_y_train) / std_y_train
            uncert = np.zeros_like(pred)
        else:
            pred, uncert = sgp.predict(X_train, 0 * X_train)
        error = np.sqrt(np.mean((pred - y_train)**2))
        trainll = np.mean(sps.norm.logpdf(pred - y_train, scale = np.sqrt(uncert)))
        print('Train RMSE: ', error)
        print('Train ll: ', trainll)
        breakpoint()                   
        next_inputs = sgp.batched_greedy_ei(args.BO_batch_size, np.min(X_train, 0), np.max(X_train, 0), np.mean(X_train, 0), np.std(X_train, 0), sample=args.sample_dist)
        valid_arcs_final = decode_from_latent_space(torch.FloatTensor(next_inputs).to(CUDA), model)
        new_features = next_inputs
        print("Evaluating selected points")
        scores = []
        for i in range(len(valid_arcs_final)):
            arc = valid_arcs_final[ i ]
            if arc is not None:
                score = -eva.eval(arc)
                score = (score - mean_y_train) / std_y_train
            else:
                score = max(y_train)[ 0 ]
            if score < best_score:
                best_score = score
                best_arc = arc
            scores.append(score)
            # print(i, score)
        # print("Iteration {}'s selected arcs' scores:".format(iteration))
        # print(scores, np.mean(scores))
        save_object(scores, "{}scores{}.dat".format(save_dir, iteration))
        save_object(valid_arcs_final, "{}valid_arcs_final{}.dat".format(save_dir, iteration))

        if len(new_features) > 0:
            X_train = np.concatenate([ X_train, new_features ], 0)
            y_train = np.concatenate([ y_train, np.array(scores)[ :, None ] ], 0)
        #
        # print("Current iteration {}'s best score: {}".format(iteration, - best_score * std_y_train - mean_y_train))
        if best_arc is not None: # and iteration == 10:
            print("Best architecture: ", best_arc)
            with open(save_dir + 'best_arc_scores.txt', 'a') as score_file:
                score_file.write(best_arc + ', {:.4f}\n'.format(-best_score * std_y_train - mean_y_train))
            if data_type == 'ENAS':
                row = [int(x) for x in best_arc.split()]
                g_best, _ = decode_ENAS_to_igraph(flat_ENAS_to_nested(row, max_n-2))
            elif data_type == 'BN':
                row = adjstr_to_BN(best_arc)
                g_best, _ = decode_BN_to_igraph(row)
            plot_DAG(g_best, save_dir, 'best_arc_iter_{}'.format(iteration), data_type=data_type, pdf=True)
        #
        iteration += 1


def visualize_sequences(sampled_sequences, grammar, token2rule):
    print("===SAMPLED SEQUENCES===")
    for i, seq in enumerate(sampled_sequences):
        print('->'.join(map(str, seq)))
        # Visualize new sequence
        path = f'data/api_ckt_ednce/generate/{i}.png'
        img_path = f'data/api_ckt_ednce/generate/{i}_g.png'
        fig, axes = plt.subplots(len(seq), figsize=(5, 5*(len(seq))))
        for idx, j in enumerate(map(int, seq)):
            r = grammar.rules[j]
            draw_graph(r.subgraph, ax=axes[idx], scale=5)
        fig.savefig(path)
        print(os.path.abspath(path))
        g = grammar.derive(seq, token2rule)        
        draw_graph(g, path=img_path)    


def sample_sequences(model, grammar, token2rule, num_samples=5, max_seq_len=10):
    # Generate and print new sequences
    sampled_sequences = sample(model, num_samples, max_seq_len)
    visualize_sequences(sampled_sequences, grammar, token2rule)


def interactive_mask_logits(grammar, generated_sequences, logits, token2rule):
    batch_size = logits.shape[0]
    for i in range(batch_size):
        for j in range(logits.shape[1]):            
            g = grammar.derive(generated_sequences[i]+[j], token2rule)
            # sanity checks
            if not nx.is_connected(nx.Graph(g)):
                logits[i, j] = float("-inf")


def interactive_sample_sequences(model, grammar, token2rule, num_samples=5, max_seq_len=10):
    model.eval()
    uniq_sequences = set()
    with torch.no_grad():
        with tqdm(total=num_samples) as pbar:
            while len(uniq_sequences) < num_samples:
                z = torch.randn(num_samples, LATENT_DIM)  
                z = z.to(CUDA)            
                batch_size = z.size(0)
                z_context = model.fc_decode(z)              
                token_embeddings = list(torch.unbind(model.start_token_embedding.expand(batch_size, -1, -1), dim=0))  
                generated_sequences = [[] for _ in range(batch_size)]  
                for t in range(max_seq_len):
                    temp_batch_embeddings, temp_z_context, active_indices = model._autoregressive_inference_active_indices(z_context, generated_sequences, token_embeddings, max_seq_len)                
                    if not active_indices:
                        break
                    logits = model._autoregressive_inference_predict_logits(temp_batch_embeddings, temp_z_context)
                    interactive_mask_logits(grammar, generated_sequences, logits, token2rule)
                    next_tokens = torch.argmax(logits, dim=-1)
                    for i, idx in enumerate(active_indices):
                        generated_sequences[idx].append(next_tokens[i].item())  
                        next_token_embedding = model.gnn(graph_data_vocabulary[next_tokens[i].item()])
                        token_embeddings[idx] = torch.cat((token_embeddings[idx], next_token_embedding), dim=0)
                uniq_sequences = uniq_sequences | set(map(tuple, generated_sequences))                
                pbar.update(len(uniq_sequences)-pbar.n)
    uniq_sequences = [list(l) for l in uniq_sequences]
    sampled_sequences = uniq_sequences[:num_samples]
    print("===SAMPLED SEQUENCES===")
    for i, seq in enumerate(sampled_sequences):
        print('->'.join(map(str, seq)))
        # Visualize new sequence
        path = f'data/api_ckt_ednce/generate/{i}.png'
        img_path = f'data/api_ckt_ednce/generate/{i}_g.png'
        fig, axes = plt.subplots(len(seq), figsize=(5, 5*(len(seq))))
        for idx, j in enumerate(map(int, seq)):
            r = grammar.rules[j]
            draw_graph(r.subgraph, ax=axes[idx], scale=5)
        fig.savefig(path)
        print(os.path.abspath(path))
        g = grammar.derive(seq, token2rule)
        draw_graph(g, path=img_path)    
    

def load_y(g, num_graphs):
    y = []
    for pre in range(num_graphs):        
        y.append(g.graph[f'{pre}:fom'])
    return y


def derive(seq):        
    all_applied = []
    all_node_maps = []
    # find the initial rule
    start_rule = seq[0]
    cur = deepcopy(start_rule.subgraph)
    all_node_maps.append({n:n for n in cur})
    assert not check_input_xor_output(cur)
    for rule in seq[1:]:
        nt_nodes = list(filter(lambda x: cur.nodes[x]["label"] in NONTERMS, cur))
        if len(nt_nodes) == 0:
            break
        assert len(nt_nodes) == 1
        node = nt_nodes[0]
        cur, applied, node_map = rule(cur, node, return_applied=True)
        all_applied.append(applied)
        all_node_maps.append(node_map)                       
    return cur, all_applied, all_node_maps  


def process_single(g_orig, rules):
    rule_ids = [r[0] for r in rules]
    rules = [r[1] for r in rules]
    g, all_applied, all_node_maps = derive(rules)
    matcher = DiGraphMatcher(g, g_orig, node_match=node_match)
    iso = next(matcher.isomorphisms_iter())
    # use iso to embed feats and instructions        
    for i, r in enumerate(rules):
        sub = deepcopy(nx.DiGraph(r.subgraph))
        # node feats
        for n in sub:
            key = all_node_maps[i][n]
            if key in iso:
                sub.nodes[n]['feat'] = g_orig.nodes[iso[key]]['feat']
        rule_ids[i] = (rule_ids[i], sub, all_applied[i-1] if i else None)
    return rule_ids


def load_data(anno, grammar, orig, cache_dir, num_graphs):
    # for ckt only, duplicate and interleave anno
    print("begin load_data")
    anno_copy = deepcopy(anno)
    anno = {}
    for n in anno_copy:
        p = get_prefix(n)
        s = get_suffix(n)
        anno[f"{2*p}:{s}"] = anno_copy[n]
        anno[f"{2*p+1}:{s}"] = deepcopy(anno_copy[n])
    exist = os.path.exists(os.path.join(cache_dir, 'data_and_rule2token.pkl'))
    if exist:
        data, rule2token = pickle.load(open(os.path.join(cache_dir, 'data_and_rule2token.pkl'), 'rb'))
    else:
        find_anno = {}        
        for k in anno:
            if get_prefix(k) not in find_anno:
                find_anno[get_prefix(k)] = []
            find_anno[get_prefix(k)].append(k)
        rule2token = {}
        pargs = []
        for pre in tqdm(range(num_graphs), "gathering rule tokens"):
            seq = find_anno[pre]
            seq = seq[::-1] # derivation          
            rule_ids = [anno[s].attrs['rule'] for s in seq]
            # orig_nodes = [list(anno[s].attrs['nodes']) for s in seq]
            # orig_feats = [[orig.nodes[n]['feat'] if n in orig else 0.0 for n in nodes] for nodes in orig_nodes]    
            # g_orig = copy_graph(orig, orig.comps[pre])
            g_orig = nx.induced_subgraph(orig, orig.comps[pre]).copy()
            for i, r in enumerate(rule_ids):
                # from networkx.algorithms.isomorphism import DiGraphMatcher
                rule2token[r] = grammar.rules[r].subgraph
                # matcher = DiGraphMatcher(copy_graph(g, orig_nodes[i]), rule2token[r], node_match=node_match)
                # breakpoint()
                # assert any(all([iso[orig_nodes[i][j]] == list(rule2token[r])[j]]) for iso in matcher.isomorphisms_iter())                
            rules = [(r, grammar.rules[r]) for r in rule_ids]
            pargs.append((g_orig, rules))
        with mp.Pool(20) as p:
            data = p.starmap(process_single, tqdm(pargs, "processing data mp"))
        pickle.dump((data, rule2token), open(os.path.join(cache_dir, 'data_and_rule2token.pkl'), 'wb+'))
    relabel = dict(zip(list(sorted(rule2token)), range(len(rule2token))))    
    data = [[(relabel[s[0]],)+s[1:] for s in seq] for seq in data]
    globals()['MAX_SEQ_LEN'] = max([len(seq) for seq in data])
    globals()['graph_vocabulary'] = list(rule2token.values())
    terminate = {}    
    vocab = []
    for key, graph in rule2token.items():
        graph_data, term = convert_graph_to_data(graph)
        terminate[relabel[key]] = term
        vocab.append(graph_data)
    globals()['graph_data_vocabulary'] = vocab
    globals()['vocabulary_terminate'] = terminate
    globals()['VOCAB_SIZE'] = len(rule2token)    
    token2rule = dict(zip(relabel.values(), relabel.keys()))

    # split here
    indices = list(range(len(data)))
    # random.Random(0).shuffle(indices)
    train_indices, test_indices = indices[:int(len(data)*0.9)], indices[int(len(data)*0.9):]
    train_data = [data[i] for i in train_indices]
    test_data = [data[i] for i in test_indices]
    y = load_y(orig, num_graphs)    
    y = np.array(y)
    train_y = y[train_indices, None]
    mean_train_y = np.mean(train_y)
    std_train_y = np.std(train_y)    
    test_y = y[test_indices, None]
    train_y = (train_y-mean_train_y)/std_train_y
    test_y = (test_y-mean_train_y)/std_train_y
    return train_data, test_data, train_y, test_y, token2rule



def main(args):
    cache_dir = 'cache/api_ckt_ednce/'
    version = get_next_version(cache_dir)-1    
    print(f"loading version {version}")
    grammar, anno, g = pickle.load(open(os.path.join(cache_dir, f'{version}.pkl'),'rb'))    
    orig = load_ckt(args, load_all=True)
    train_data, test_data, train_y, test_y, token2rule = load_data(anno, grammar, orig, cache_dir, 10000)        
    breakpoint()
    model = train(train_data, test_data)
    # breakpoint()
    bo(args, model, train_y, test_y)
    interactive_sample_sequences(model, grammar, token2rule, max_seq_len=MAX_SEQ_LEN, num_samples=NUM_SAMPLES)    




if __name__ == "__main__":
    from src.grammar.common import get_parser
    parser = get_parser()
    args = parser.parse_args()
    main(args)
