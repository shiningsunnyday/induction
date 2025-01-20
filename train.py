import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset
from sklearn.manifold import TSNE
import networkx as nx
import numpy as np
from random import shuffle
from scipy.spatial.distance import pdist
import scipy.stats as sps
from scipy.stats import pearsonr
import hashlib
# import gpflow
# from gpflow.models import SVGP
# from gpflow.optimizers import NaturalGradient
# import tensorflow as tf
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
from src.grammar.common import *
from src.examples.test_graphs import *
from src.draw.graph import draw_graph
import glob
import re

import sys
sys.path.append('dagnn/dvae/bayesian_optimization')
sys.path.append('CktGNN')
from utils import is_valid_DAG, is_valid_Circuit

# Logging
logger = create_logger("train", "cache/api_ckt_ednce/train.log")

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
        self.perm = np.arange(len(data))
    
    def __len__(self):
        return len(self.data)
    

    def shuffle(self):
        self.perm = np.random.permutation(self.perm)


    def __getitem__(self, idx):
        return self.dataset[self.perm[idx]]


class GraphDataset(Dataset):
    def __init__(self, data):
        self.dataset = []
        self.data = data
        for idx in range(len(data)):
            seq, graph = self.data[idx]
            graph, _ = convert_graph_to_data(graph)
            self.dataset.append((torch.tensor(seq), graph, idx))
        self.perm = np.arange(len(data))
    
    def __len__(self):
        return len(self.data)

    def shuffle(self):
        self.perm = np.random.permutation(self.perm)        

    def __getitem__(self, idx):
        return self.dataset[self.perm[idx]]


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
    def __init__(self, encoder, encoder_layers, decoder_layers, embed_dim, latent_dim, seq_len):
        super(TransformerVAE, self).__init__()        
        # self.token_gnn = TokenGNN(embed_dim)
        self.encoder = encoder
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        if encoder == "TOKEN_GNN":
            self.gnn = DAGNN(None, None, 13, latent_dim, None, w_edge_attr=False, bidirectional=False, num_class=latent_dim)        
            self.transformer_encoder = TransformerEncoder(embed_dim, num_layers=encoder_layers)            
        elif encoder == "GNN":
            self.token_embedding = nn.Embedding(VOCAB_SIZE, embed_dim)
            self.gnn = DAGNN(None, None, 13, latent_dim, None, w_edge_attr=False, bidirectional=False, num_class=latent_dim)
        else:
            self.token_embedding = nn.Embedding(VOCAB_SIZE, embed_dim)
            self.transformer_encoder = TransformerEncoder(embed_dim, num_layers=encoder_layers)
        self.fc_mu = nn.Linear(embed_dim, latent_dim)
        self.fc_logvar = nn.Linear(embed_dim, latent_dim)
        
        self.fc_decode = nn.Linear(latent_dim, latent_dim)
        self.transformer_decoder = TransformerEncoder(embed_dim + latent_dim, num_layers=decoder_layers)
        self.output_layer = nn.Linear(embed_dim + latent_dim, VOCAB_SIZE)
        self.start_token_embedding = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.terminate_mask = torch.zeros((len(vocabulary_terminate),)).bool()
        for i in range(len(vocabulary_terminate)):
            self.terminate_mask[i] = vocabulary_terminate[i]
        self.init_mask = torch.zeros((len(vocabulary_init),)).bool()
        for i in range(len(vocabulary_init)):
            self.init_mask[i] = vocabulary_init[i]


    def visualize_tokens(self):
        weights = self.token_embedding.weight.detach().cpu()
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, weights.shape[0]-1))
        embeddings_2d = tsne.fit_transform(weights)
        # Plot the 2D t-SNE
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
        ax.set_title("2D t-SNE of nn.Embedding")
        ax.set_xlabel("t-SNE Dim 1")
        ax.set_ylabel("t-SNE Dim 2")
        plt.close(fig)
        return fig


    def embed_tokens(self, token_ids, batch_g_list):
        if token_ids.dim() == 2:
            return torch.stack([self.embed_tokens(token_id_seq, g_list) for (token_id_seq, g_list) in zip(token_ids, batch_g_list)], dim=0)
        else:
            if self.encoder != "TOKEN_GNN":
                return self.token_embedding(token_ids)
            g_list = batch_g_list
        embedded_tokens = []
        g_list = [g_list[i] for i in range(len(g_list))]
        for token_id, graph_data in zip(token_ids, g_list):
            if graph_data is None:
                embedded_token = torch.zeros((args.latent_dim,), device=args.cuda)
            else:
                graph_data.to(args.cuda)
                embedded_token = self.gnn(graph_data).flatten()
            embedded_tokens.append(embedded_token)
        return torch.stack(embedded_tokens, dim=0)
        # return torch.cat(embedded_tokens, dim=0)

    def encode(self, x, attention_mask=None):        
        if self.encoder == "GNN":            
            pooled = torch.stack([self.gnn(g.to(args.cuda)).flatten() for g in x], dim=0)
        else:
            assert attention_mask is not None
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
                    next_token_embedding = x[idx][t].unsqueeze(0)
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
        # Mask out logits for start and terminate rules
        if seq_len == self.seq_len:
            logits[:,~self.terminate_mask] = float("-inf")
        elif seq_len == 1:
            logits[:,~self.init_mask] = float("-inf")
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
                if ENCODER == "TOKEN_GNN":
                    next_token_embedding = self.gnn(graph_data_vocabulary[next_tokens[i].item()])
                else: # default to learnable embedding
                    next_token_embedding = self.token_embedding(next_tokens[i])
                # Append the new embedding to this sequence’s token embeddings
                token_embeddings[idx] = torch.cat((token_embeddings[idx], next_token_embedding), dim=0)
        return generated_sequences


    def forward(self, x, attention_mask, seq_len_list, batch_g_list):
        embedded_tokens = self.embed_tokens(x, batch_g_list)  # Embeds each token (graph) using GNN or learnable embedding        
        if self.encoder == "GNN":
            mu, logvar = self.encode(batch_g_list, attention_mask)
        else:
            mu, logvar = self.encode(embedded_tokens, attention_mask)
        z = self.reparameterize(mu, logvar)
        logits, logits_mask = self.autoregressive_training_step(embedded_tokens, z, seq_len_list)
        return logits, logits_mask, mu, logvar


# Loss function
def vae_loss(args, recon_logits, mask, x, mu, logvar):    
    x_flat = x.view(-1)
    recon_loss = F.cross_entropy(recon_logits, x_flat, reduction="none")
    recon_loss = recon_loss.view(x.size(0), -1)
    recon_loss = (recon_loss * mask).sum() / mask.sum()
    kl_divergence = -args.klcoeff * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_divergence / x.size(0)


# Padding function
def collate_batch(batch):    
    lengths = [len(seq) for seq, _, _ in batch]
    max_len = max(lengths)
    padded_batch = torch.zeros(len(batch), max_len, dtype=torch.long)    
    seq_len_list = torch.zeros(len(batch), dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    graphs = []
    idxes = []
    batch_g_list = []            
    for i, (seq, graph, idx) in enumerate(batch):
        padded_batch[i, :len(seq)] = seq
        seq_len_list[i] = len(seq)       
        idxes.append(idx)
        attention_mask[i, :len(seq)] = 1
        g_list = [None] * max_len # pad to max_len
        # g_list.update({i: None for i in range(len(g_list), max_len)})
        batch_g_list.append(g_list)  
    return padded_batch, attention_mask, seq_len_list, batch_g_list, idxes 
    

def collate_batch_gnn(batch):
    lengths = [len(seq) for seq, _, _ in batch]
    max_len = max(lengths)
    padded_batch = torch.zeros(len(batch), max_len, dtype=torch.long)    
    seq_len_list = torch.zeros(len(batch), dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    graphs = []
    idxes = []          
    for i, (seq, graph, idx) in enumerate(batch):
        padded_batch[i, :len(seq)] = seq
        seq_len_list[i] = len(seq)       
        idxes.append(idx)
        attention_mask[i, :len(seq)] = 1
        graphs.append(graph)         
    return padded_batch, attention_mask, seq_len_list, graphs, idxes   



# Sampling new sequences
def sample(model, num_samples=5, max_seq_len=10):
    model.eval()
    uniq_sequences = set()
    with torch.no_grad():
        while len(uniq_sequences) < num_samples:
            z = torch.randn(num_samples, args.latent_dim)  # Sample from the prior
            z = z.to(args.cuda)
            generated_sequences = model.autoregressive_inference(z, max_seq_len)  # Decode from latent space            
            uniq_sequences = uniq_sequences | set(map(tuple, generated_sequences))
    uniq_sequences = [list(l) for l in uniq_sequences]
    return uniq_sequences[:num_samples]


def decode_from_latent_space(z, model, max_seq_len=10):
    generated_sequences = model.autoregressive_inference(z, max_seq_len)
    return generated_sequences
    

def train(args, train_data, test_data):
    folder = args.datapkl if args.datapkl else args.folder
    # Initialize model and optimizer
    model = TransformerVAE(args.encoder, args.encoder_layers, args.decoder_layers, args.embed_dim, args.latent_dim, MAX_SEQ_LEN)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    if args.encoder == "GNN":
        train_dataset = GraphDataset(train_data)
        test_dataset = GraphDataset(test_data) 
    else:       
        # Dummy dataset: Replace with actual sequence data
        # dataset = [torch.tensor(seq) for seq in data]
        train_dataset = TokenDataset(train_data)
        # train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch, num_workers=8, timeout=1000, prefetch_factor=1)
        test_dataset = TokenDataset(test_data)
        # test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch, num_workers=8, timeout=1000, prefetch_factor=1)    

    # Training loop
    model.train()
    model = model.to(args.cuda)
    if args.encoder == "TOKEN_GNN":
        for i, graph_data in enumerate(graph_data_vocabulary):
            graph_data_vocabulary[i] = graph_data.to(args.cuda)
    ckpts = glob.glob(f'ckpts/api_ckt_ednce/{folder}/*.pth')
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
    logger.info(best_ckpt_path)    
    if best_ckpt_path is not None:
        logger.info(f"loaded {best_ckpt_path} loss {best_loss} start_epoch {start_epoch}")
        model.load_state_dict(torch.load(best_ckpt_path))

    patience = 25
    patience_counter = 0
    train_latent = np.empty((len(train_data), args.latent_dim))
    test_latent = np.empty((len(test_data), args.latent_dim))
    for epoch in tqdm(range(start_epoch, args.epochs+1)):
        model.train()
        train_loss = 0.
        rec_acc_sum = 0.
        train_dataset.shuffle()
        g_batch = []
        for i in tqdm(range(len(train_dataset))):
            g_batch.append(train_dataset[i])
            if len(g_batch) == args.batch_size or i == len(train_dataset)-1:
                if args.encoder == "GNN":
                    batch = collate_batch_gnn(g_batch)
                else:
                    batch = collate_batch(g_batch)
                x, attention_mask, seq_len_list, batch_g_list, batch_idxes = batch
                x, attention_mask = x.to(args.cuda), attention_mask.to(args.cuda)
                optimizer.zero_grad()
                recon_logits, mask, mu, logvar = model(x, attention_mask, seq_len_list, batch_g_list)                           
                loss = vae_loss(args, recon_logits, mask, x, mu, logvar)
                loss.backward()            
                train_loss += loss.item()*len(batch_idxes)
                rll = recon_logits.argmax(axis=-1).reshape(x.shape)
                rec_acc = (rll == x).all(axis=-1)
                rec_acc_sum += rec_acc.sum()
                optimizer.step()
                train_latent[batch_idxes] = mu.detach().cpu().numpy()
                g_batch = []
        train_loss /= len(train_dataset)
        train_rec_acc_mean = rec_acc_sum / len(train_dataset)
        model.eval()
        val_loss = 0.
        rec_acc_sum = 0.
        g_batch = []
        with torch.no_grad():
            for i in tqdm(range(len(test_dataset))):
                g_batch.append(test_dataset[i])
                if len(g_batch) == args.batch_size or i == len(test_dataset)-1:
                    if args.encoder == "GNN":
                        batch = collate_batch_gnn(g_batch)
                    else:
                        batch = collate_batch(g_batch)               
                    x, attention_mask, seq_len_list, batch_g_list, batch_idxes = batch
                    x, attention_mask = x.to(args.cuda), attention_mask.to(args.cuda)
                    recon_logits, mask, mu, logvar = model(x, attention_mask, seq_len_list, batch_g_list)
                    loss = vae_loss(args, recon_logits, mask, x, mu, logvar)            
                    val_loss += loss.item()*len(batch_idxes)
                    rll = recon_logits.argmax(axis=-1).reshape(x.shape)
                    rec_acc = (rll == x).all(axis=-1)
                    rec_acc_sum += rec_acc.sum()
                    test_latent[batch_idxes] = mu.detach().cpu().numpy()
                    g_batch = []   
        val_loss /= len(test_dataset)
        valid_rec_acc_mean = rec_acc_sum / len(test_dataset)
        if val_loss < best_loss:
            patience_counter = 0 # reset counter
            best_loss = val_loss
            ckpt_path = f'ckpts/api_ckt_ednce/{args.folder}/epoch={epoch}_loss={best_loss}.pth'
            torch.save(model.state_dict(), ckpt_path)
            logger.info(ckpt_path)
        else:
            patience_counter += 1
            logger.info(f"No improvement from best loss: {best_loss}, patience: {patience_counter}/{patience}")
        logger.info(f"Run Details:\n"
            f"  - Encoder: {args.encoder}"
            f"  - Embedding Dimension: {args.embed_dim}"
            f"  - Latent Dimension: {args.latent_dim}"
            f"  - Encoder Layers: {args.encoder_layers}"
            f"  - Decoder Layers: {args.decoder_layers}"
            f"  - Batch Size: {args.batch_size}"
            f"  - KL Divergence Coefficient: {args.klcoeff}")
        logger.info(f"Epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}, Train Rec: {train_rec_acc_mean}, Val Rec: {valid_rec_acc_mean}")
        np.save(f'ckpts/api_ckt_ednce/{args.folder}/train_latent_{epoch}.npy', train_latent)
        np.save(f'ckpts/api_ckt_ednce/{args.folder}/test_latent_{epoch}.npy', test_latent)
        fig = model.visualize_tokens()
        fig.savefig(f'ckpts/api_ckt_ednce/{args.folder}/{epoch}.png')        
        embedding = model.token_embedding.weight.detach().cpu().numpy()
        np.save(f'ckpts/api_ckt_ednce/{args.folder}/embedding_{epoch}.npy', embedding)
        if patience_counter > patience:
            logger.info(f"Early stopping triggered at epoch {epoch}. Best loss: {best_loss}")
            break
    return model


def train_sgp(sgp, input_means, training_targets, batch_size=1000, lr=1e-4, max_iter=500):
    train_dataset = tf.data.Dataset.from_tensor_slices((input_means, training_targets))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    # Define optimizer
    adam_optimizer = tf.optimizers.Adam(learning_rate=lr)
    # Training function
    @tf.function
    def train_step(model, optimizer, batch):
        with tf.GradientTape() as tape:
            data_input, data_output = batch
            # Compute variational ELBO loss
            loss = -model.elbo((data_input, data_output))
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    # Train for a given number of iterations
    for i in range(max_iter):
        loss_total = 0.
        for step, batch in enumerate(train_dataset):
            loss = train_step(sgp, adam_optimizer, batch)
            loss_total += loss.numpy()
        logger.info(f"Epoch {i}: Loss = {loss_total}")
        sgp.predict(X_test)


def visualize_sequences(sampled_sequences, grammar, token2rule):
    logger.info("===SAMPLED SEQUENCES===")
    for i, seq in enumerate(sampled_sequences):
        logger.info('->'.join(map(str, seq)))
        # Visualize new sequence
        path = f'data/api_ckt_ednce/generate/{i}.png'
        img_path = f'data/api_ckt_ednce/generate/{i}_g.png'
        fig, axes = plt.subplots(len(seq), figsize=(5, 5*(len(seq))))
        for idx, j in enumerate(map(int, seq)):
            r = grammar.rules[j]
            draw_graph(r.subgraph, ax=axes[idx], scale=5)
        fig.savefig(path)
        logger.info(os.path.abspath(path))
        g = grammar.derive(seq, token2rule)        
        draw_graph(g, path=img_path)    


def sample_sequences(model, grammar, token2rule, num_samples=5, max_seq_len=10):
    # Generate and logger.info new sequences
    sampled_sequences = sample(model, num_samples, max_seq_len)
    visualize_sequences(sampled_sequences, grammar, token2rule)


def check_connections(grammar, g, rule):
    I = rule.embedding
    nt = grammar.search_nts(g, NONTERMS)[0]
    conn = []
    for d in ["in", "out"]:
        for nei in neis(g, [nt], direction=[d]):
            mu = g.nodes[nei]['label']
            for i in range(len(rule.subgraph)):
                if (mu, "black", "black", i, d, "in") in I:
                    conn.append([nei, i])
                if (mu, "black", "black", i, d, "out") in I:
                    conn.append([i, nei])
    return conn   


def check_op_amp(grammar, conn, g, rule):
    nodes = grammar.search_nts(rule.subgraph, ["deepskyblue", "dodgerblue"])
    for n in nodes:
        i = list(rule.subgraph).index(n)
        preds = list(map(lambda a: a[0], filter(lambda a: a[1]==i, conn)))
        succs = list(map(lambda a: a[1], filter(lambda a: a[0]==i, conn)))
        new_succs = [list(rule.subgraph).index(j) for j in rule.subgraph.successors(n)]
        safe = False
        for a in preds:
            succs = g.successors(a)                    
            for s in filter(lambda x: g.nodes[x]['label'] in ['yellow', 'lawngreen'], succs):
                succ_inter = bool(set(g.successors(s)) & (set(succs)))
                # s connects to one of new_succ                        
                new_succ_inter = any([(s, j) in conn for j in new_succs])
                if succ_inter | new_succ_inter:
                    breakpoint()
                    safe = True
        if not safe:
            return False
    return True


def interactive_mask_logits(grammar, generated_graphs, generated_orders, logits, token2rule):
    batch_size = logits.shape[0]
    for i in tqdm(range(batch_size), desc="masking logits batch"):
        if generated_graphs[i] is None:
            continue
        g = generated_graphs[i]
        inmi = {}
        for n in g:
            if g.nodes[n]['label'] not in NONTERMS:
                inmi[n] = len(inmi) # inv node map index
        o = generated_orders[i]    
        for j in tqdm(range(logits.shape[1]), desc="masking logits single"):
            if logits[i, j] == float("-inf"):
                continue # init, terminating logic already in _autoregressive_inference_predict_logits
            # g = deepcopy(generated_graphs[i])
            if vocabulary_init[j]:
                continue
            rule = grammar.rules[token2rule[j]]
            o_j = order_init(rule.subgraph, ignore_nt=False)
            ### sanity checks
            ## stays connected
            # g_, applied, node_map = grammar.one_step_derive(g, token2rule[j], token2rule, return_applied=True)
            # inv_node_map_index = dict(zip(node_map.values(), map(lambda k: list(rule.subgraph).index(k), node_map.keys())))
            # stays_connected = nx.is_connected(nx.Graph(g_))            
            ## stays connected
            # new_edges = set(g_.edges)-set(g.edges)-set(product(inv_node_map_index, inv_node_map_index))
            conn = check_connections(grammar, g, rule)
            ## check acyclic
            # cycle can only form from ((a, x), (y, b)) in product(conn, conn) satisfying:
            # 1. a and b from g
            # 2. x and y from range(len(rule.subgraph))
            # 3. o_j[x,y]
            # 4. o[b,a]
            out_pairs = filter(lambda a: isinstance(a[1], int), conn)
            in_pairs = filter(lambda a: isinstance(a[0], int), conn)
            acyclic = True
            for (a, x), (y, b) in product(out_pairs, in_pairs):
                if o_j[x, y] and o[inmi[b], inmi[a]]:
                    acyclic = False
            # if len(conn) != len(new_edges):
            #     breakpoint()
            # for u, v in new_edges:
            #     if u in inv_node_map_index:
            #         u = inv_node_map_index[u]
            #     if v in inv_node_map_index:
            #         v = inv_node_map_index[v]
            #     if [u, v] not in conn:
            #         breakpoint()

            # stays_connected <=> len(conn)
            # assert bool(len(conn)) == stays_connected
            ## op-amp circuit constraints            
            
            # if nx.is_directed_acyclic_graph(g_):
            #     if vocabulary_terminate[j]:
            #         ig_ = nx_to_igraph(g_)
            #         assert is_valid_Circuit(ig_, subg=False)
            if not (acyclic and len(conn)):
                logits[i, j] = float("-inf")
            else:
                amp_valid = check_op_amp(grammar, conn, g, rule)
                if not amp_valid:
                    logits[i, j] = float("-inf")



def order_init(g, ignore_nt=True):
    is_term = lambda n: g.nodes[n]['label'] not in NONTERMS
    if ignore_nt:
        nodes = [n for n in g if is_term(n)]
        g_ = copy_graph(g, nodes)
    else:
        nodes = list(g)
        g_ = g
    paths = nx.all_pairs_shortest_path(g_)
    o = np.zeros((len(nodes), len(nodes)), dtype=int)
    for s, l in paths:
        for t in l:
            if ignore_nt and all([is_term(n) for n in l[t]]) or not ignore_nt:
                i = nodes.index(s)
                j = nodes.index(t)
                o[i, j] = 1
    return o


def get_inmi(g, ignore_nt=True):
    inmi = {}
    for n in g:
        if g.nodes[n]['label'] not in NONTERMS:
            inmi[n] = len(inmi) # inv node map index    
    return inmi


def update_order(g, o, grammar, j, token2rule):
    rule = grammar.rules[token2rule[j]]
    inmi = get_inmi(g, ignore_nt=True)
    rhs = rule.subgraph
    inmi_rhs = get_inmi(rhs, ignore_nt=True)
    n = len(o)
    m = len(inmi_rhs)
    r_adj = order_init(rhs)
    o_ = np.zeros((n+m, n+m))
    o_[:n, :n] = o
    o_[n:, n:] = r_adj
    conn = check_connections(grammar, g, rule)
    # get all (a, x) pairs
    out_pairs = filter(lambda a: isinstance(a[1], int), conn)
    for (a, x) in out_pairs:
        # get all y reachable from x
        if list(rhs)[x] not in inmi_rhs:
            continue
        for y in np.argwhere(r_adj[inmi_rhs[list(rhs)[x]]]).flatten():
            # get all b that reaches a
            for b in np.argwhere(o[:, inmi[a]]).flatten():
                o_[b, y+n] = 1
    in_pairs = filter(lambda a: isinstance(a[0], int), conn)
    for (y, b) in in_pairs:
        # get all x that reaches y
        if list(rhs)[y] not in inmi_rhs:
            continue
        for x in np.argwhere(r_adj[:, inmi_rhs[list(rhs)[y]]]).flatten():
            # get all a reachable from b
            for a in np.argwhere(o[inmi[b]]).flatten():
                o_[x+n, a] = 1
    # all paths going through y
    for y in inmi_rhs.values():
        for a in np.argwhere(o_[:n, y+n]).flatten():
            for b in np.argwhere(o_[y+n, :n]).flatten():
                o_[a, b] = 1
    # all paths going from x
    return o_


def interactive_sample_sequences(args, model, grammar, token2rule, num_samples=5, max_seq_len=10):
    num_samples = args.num_samples
    sample_batch_size = args.sample_batch_size
    model.eval()
    uniq_sequences = set()
    with torch.no_grad():
        with tqdm(total=num_samples) as pbar:
            while len(uniq_sequences) < num_samples:
                z = torch.randn(sample_batch_size, args.latent_dim)  
                z = z.to(args.cuda)            
                batch_size = z.size(0)
                z_context = model.fc_decode(z)              
                token_embeddings = list(torch.unbind(model.start_token_embedding.expand(batch_size, -1, -1), dim=0))  
                generated_sequences = [[] for _ in range(batch_size)]                  
                generated_graphs = [None for _ in range(batch_size)]
                generated_orders = [np.zeros((0, 0)) for _ in range(batch_size)]
                for t in range(max_seq_len):
                    temp_batch_embeddings, temp_z_context, active_indices = model._autoregressive_inference_active_indices(z_context, generated_sequences, token_embeddings, max_seq_len)                
                    if not active_indices:
                        break
                    logits = model._autoregressive_inference_predict_logits(temp_batch_embeddings, temp_z_context)
                    interactive_mask_logits(grammar, [generated_graphs[i] for i in active_indices], [generated_orders[i] for i in active_indices], logits, token2rule)
                    next_tokens = torch.argmax(logits, dim=-1)
                    for i, idx in enumerate(active_indices):
                        generated_sequences[idx].append(next_tokens[i].item())
                        if generated_graphs[idx] is None:                            
                            generated_graphs[idx] = grammar.derive([next_tokens[i].item()], token2rule)
                            # init acyclic order
                            generated_orders[idx] = order_init(generated_graphs[idx])
                        else:
                            order_copy = deepcopy(generated_orders[idx])
                            generated_orders[idx] = update_order(generated_graphs[idx], generated_orders[idx], grammar, next_tokens[i].item(), token2rule)
                            # update acyclic order
                            g_copy = deepcopy(generated_graphs[idx])
                            generated_graphs[idx] = grammar.one_step_derive(generated_graphs[idx], next_tokens[i].item(), token2rule)
                            # debug
                            g_ = copy_graph(generated_graphs[idx], [n for n in generated_graphs[idx] if generated_graphs[idx].nodes[n]['label'] not in NONTERMS])
                            for i_ in range(len(g_)):
                                for j_ in range(len(g_)):
                                    if bool(nx.has_path(g_, list(g_)[i_], list(g_)[j_])) != bool(generated_orders[idx][i_, j_]):
                                        breakpoint()
                                        update_order(g_copy, order_copy, grammar, next_tokens[i].item(), token2rule)
                        if not nx.is_directed_acyclic_graph(generated_graphs[idx]):
                            breakpoint()
                            update_order(g_copy, order_copy, grammar, next_tokens[i].item(), token2rule)
                        if model.encoder == "TOKEN_GNN":
                            next_token_embedding = model.gnn(graph_data_vocabulary[next_tokens[i].item()])
                        else: # default to learnable embedding
                            next_token_embedding = model.token_embedding(next_tokens[i]).unsqueeze(0)
                        token_embeddings[idx] = torch.cat((token_embeddings[idx], next_token_embedding), dim=0)       
                uniq_sequences = uniq_sequences | set(map(tuple, generated_sequences))                
                pbar.update(len(uniq_sequences)-pbar.n)
    uniq_sequences = [list(l) for l in uniq_sequences]
    sampled_sequences = uniq_sequences[:num_samples]
    logger.info("===SAMPLED SEQUENCES===")
    for i, seq in enumerate(sampled_sequences):
        logger.info('->'.join(map(str, seq)))
        # Visualize new sequence
        path = f'data/api_ckt_ednce/generate/{i}.png'
        img_path = f'data/api_ckt_ednce/generate/{i}_g.png'
        fig, axes = plt.subplots(len(seq), figsize=(5, 5*(len(seq))))
        for idx, j in enumerate(map(int, seq)):
            r = grammar.rules[j]
            draw_graph(r.subgraph, ax=axes[idx], scale=5)
        fig.savefig(path)
        logger.info(os.path.abspath(path))
        g = grammar.derive(seq, token2rule)
        draw_graph(g, path=img_path)    
    breakpoint()
    

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


def load_data(args, anno, grammar, orig, cache_dir, num_graphs):
    loaded = False
    if args.datapkl:
        save_path = os.path.join(cache_dir, args.datapkl, 'data.pkl')
        if os.path.exists(args.datapkl): # specified to load data from args.datapkl path
            logger.info(f"load data from {save_path}")
            data, rule2token = pickle.load(open(save_path, 'rb'))
            loaded = True
    if not loaded:        
        if args.datapkl:
            save_path = os.path.join(cache_dir, args.datapkl, 'data.pkl')
        else:
            save_path = os.path.join(cache_dir, args.folder, 'data.pkl')
        # for ckt only, duplicate and interleave anno
        logger.info(f"begin load_data")
        anno_copy = deepcopy(anno)
        anno = {}
        for n in anno_copy:
            p = get_prefix(n)
            s = get_suffix(n)
            anno[f"{2*p}:{s}"] = anno_copy[n]
            anno[f"{2*p+1}:{s}"] = deepcopy(anno_copy[n])
        exist = os.path.exists(save_path)
        if exist:
            logger.info(f"load data from {save_path}")
            data, rule2token = pickle.load(open(save_path, 'rb'))
        else:
            find_anno = {}        
            for k in anno:
                if get_prefix(k) not in find_anno:
                    find_anno[get_prefix(k)] = []
                find_anno[get_prefix(k)].append(k)
            rule2token = {}
            pargs = []
            data = []
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
                if args.encoder == "GNN":
                    data.append((rule_ids, g_orig))                
                else:                    
                    rules = [(r, grammar.rules[r]) for r in rule_ids]
                    pargs.append((g_orig, rules))
            if args.encoder != "GNN":
                with mp.Pool(20) as p:
                    data = p.starmap(process_single, tqdm(pargs, "processing data mp"))
            pickle.dump((data, rule2token), open(save_path, 'wb+'))
    relabel = dict(zip(list(sorted(rule2token)), range(len(rule2token))))    
    token2rule = dict(zip(relabel.values(), relabel.keys()))
    if args.encoder == "GNN":
        data = [([relabel[s] for s in seq], g) for seq, g in data]    
    else:
        data = [[(relabel[s[0]],)+s[1:] for s in seq] for seq in data]    
    globals()['graph_vocabulary'] = list(rule2token.values())
    terminate, init = {}, {}
    vocab = []
    for key, graph in rule2token.items():
        graph_data, term = convert_graph_to_data(graph)
        terminate[relabel[key]] = term
        init[relabel[key]] = grammar.rules[key].nt == 'black'
        vocab.append(graph_data)
    if args.encoder == "GNN":
        globals()['MAX_SEQ_LEN'] = max([len(seq) for seq, _ in data])
    else:
        globals()['MAX_SEQ_LEN'] = max([len(seq) for seq in data])
    globals()['graph_data_vocabulary'] = vocab
    globals()['vocabulary_terminate'] = terminate
    globals()['vocabulary_init'] = init
    globals()['VOCAB_SIZE'] = len(rule2token)
    # split here
    indices = list(range(num_graphs))
    # random.Random(0).shuffle(indices)
    train_indices, test_indices = indices[:int(num_graphs*0.9)], indices[int(num_graphs*0.9):]
    train_data = [data[i] for i in train_indices]
    test_data = [data[i] for i in test_indices]
    return train_data, test_data, token2rule


def hash_args(args, ignore_keys=['datapkl', 'checkpoint']):
    arg_dict = {k: v for k, v in args.__dict__.items() if k not in ignore_keys}
    return hashlib.md5(json.dumps(arg_dict, sort_keys=True).encode()).hexdigest()


def main(args):
    cache_dir = 'cache/api_ckt_ednce/'
    folder = hash_args(args)
    setattr(args, "folder", folder)
    os.makedirs(f'ckpts/api_ckt_ednce/{folder}', exist_ok=True)
    os.makedirs(f'cache/api_ckt_ednce/{folder}', exist_ok=True)
    #json.dumps(args.__dict__, folder)
    args_path = os.path.join(f'ckpts/api_ckt_ednce/{folder}', "args.txt")
    with open(args_path, "w") as f:
        for arg_name, arg_value in sorted(args.__dict__.items()):
            f.write(f"{arg_name}: {arg_value}\n")
    num_graphs = 10000
    version = get_next_version(cache_dir)-1    
    logger.info(f"loading version {version}")
    grammar, anno, g = pickle.load(open(os.path.join(cache_dir, f'{version}.pkl'),'rb'))    
    orig = load_ckt(args, load_all=True)
    train_data, test_data, token2rule = load_data(args, anno, grammar, orig, cache_dir, num_graphs)
    if args.datapkl:
        print(f'The folder being written to is: {args.datapkl}')
    else:
        print(f'The folder being written to is: {folder}')        
    # prepare y
    # TODO: remove this later
    # indices = list(range(num_graphs))
    # # random.Random(0).shuffle(indices)
    # train_indices, test_indices = indices[:int(num_graphs*0.9)], indices[int(num_graphs*0.9):]    
    # y = load_y(orig, num_graphs)    
    # y = np.array(y)
    # train_y = y[train_indices, None]
    # mean_train_y = np.mean(train_y)
    # std_train_y = np.std(train_y)    
    # test_y = y[test_indices, None]
    # train_y = (train_y-mean_train_y)/std_train_y
    # test_y = (test_y-mean_train_y)/std_train_y    
    model = train(args, train_data, test_data)
    # breakpoint()
    # bo(args, None, train_y, test_y)
    interactive_sample_sequences(args, model, grammar, token2rule, max_seq_len=MAX_SEQ_LEN)    




if __name__ == "__main__":
    from src.grammar.common import get_parser
    parser = get_parser()
    # data hparams
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--sample-batch-size", type=int, default=10)
    # nn hparams
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--encoder-layers", type=int, default=4)
    parser.add_argument("--decoder-layers", type=int, default=4)
    parser.add_argument("--encoder", choices=["TOKEN_GNN", "GNN", "TOKEN"], default="GNN")
    # training
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--cuda", default='cpu')
    parser.add_argument("--datapkl", help="path to folder")
    parser.add_argument("--klcoeff", type=float, default=0.5, help="coefficient to KL div term in VAE loss")
    # eval
    args = parser.parse_args()        
    breakpoint()
    main(args)
