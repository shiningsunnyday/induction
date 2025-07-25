import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
sys.path.append('dagnn/dvae/bayesian_optimization')
import tempfile
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
from sklearn.metrics import r2_score
import hashlib
import fcntl
from functools import partial
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
# import gpflow
# from gpflow.models import SVGP
# from gpflow.optimizers import NaturalGradient
# import tensorflow as tf
# torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data as GraphData
import torch.nn.functional as F
torch.multiprocessing.set_sharing_strategy('file_system')
import pickle
import hashlib
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import rcParams
# Set the font to Arial
rcParams['font.family'] = 'Arial'
from src.model import *
if DATASET == "enas":
    sys.path.append(os.path.join(os.path.dirname(__file__), 'dagnn/dvae/software/enas/src/cifar10'))   
else:
    sys.path.append(os.path.join(os.path.dirname(__file__), 'CktGNN'))
import glob
import re
sys.path.append(os.path.join(os.path.dirname(__file__), 'dagnn/dvae'))
from util import save_object, load_object, plot_DAG, flat_ENAS_to_nested, adjstr_to_BN, decode_igraph_to_ENAS, is_valid_ENAS, is_valid_BN, is_valid_DAG, decode_igraph_to_BN_adj
from evaluate_BN import Eval_BN
try:
    from sparse_gp import SparseGP
except:
    print("sparsegp import failed")
from utils import is_valid_DAG, is_valid_Circuit
from OCB.src.simulator.graph_to_fom import cktgraph_to_fom
from OCB.src.utils_src import plot_circuits
# Logging
logger = create_logger("train", f"data/api_{DATASET}_ednce/train.log")

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
    #breakpoint()
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
        if args.dataset == 'enas' and args.encoder == 'TOKEN':
            feat = one_hot_vector
        else:
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
        for idx in range(len(self.data)):
            seq, graph = self.data[idx]
            if isinstance(graph, list):
                assert 'orig' in globals()
                graph = nx.induced_subgraph(orig, graph)
            graph, _ = convert_graph_to_data(graph)
            self.dataset.append((torch.tensor(seq), graph, idx))
        self.perm = np.arange(len(self.data))
    
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


# Loss function
def vae_loss(args, recon_logits, mask, x, mu, logvar):        
    if args.repr == "ns":
        x_flat = x.view(-1, x.shape[-1])
        mask = mask.view(-1, x.shape[-1])
        recon_logits = recon_logits.view(-1, x.shape[-1])
        one_hot_logits = recon_logits[:, :graph_args.num_vertex_type]
        binary_logits = recon_logits[:, graph_args.num_vertex_type:]
        one_hot_y = x_flat[:, :graph_args.num_vertex_type]
        one_hot_y_indices = one_hot_y.argmax(dim=1).long()
        binary_y = x_flat[:, graph_args.num_vertex_type:]
        binary_y_flat = binary_y.reshape(-1,).float()
        recon_loss = F.binary_cross_entropy(binary_logits.reshape(-1,), binary_y_flat, reduction="sum")
        recon_loss += F.cross_entropy(one_hot_logits, one_hot_y_indices, reduction="sum")        
    else:
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


def collate_batch_gnn_ns(batch):
    lengths = [seq.shape for seq, _, _ in batch]
    max_len = torch.Size(tuple(map(max, zip(*lengths))))
    padded_batch = torch.zeros(len(batch), *max_len, dtype=torch.long)    
    seq_len_list = torch.zeros(len(batch), 2, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), *max_len), dtype=torch.bool)
    graphs = []
    idxes = []          
    for i, (seq, graph, idx) in enumerate(batch):
        padded_batch[i, :seq.shape[0], :seq.shape[1]] = seq
        seq_len_list[i] = torch.tensor(seq.shape)
        idxes.append(idx)
        attention_mask[i, :seq.shape[0], :seq.shape[1]] = 1
        graphs.append(graph)         
    return padded_batch, attention_mask, seq_len_list, graphs, idxes     


def process_ns(orig, arg_list):
    res = []
    for pre in arg_list:
        g_orig = nx.induced_subgraph(orig, orig.comps[pre])
        g_orig = g_orig.copy()
        node_str = stringify(g_orig)
        if args.order == "bfs":
            node_str = torch.tensor([node_str])
            adj, feat = G_to_adjfeat(node_str, graph_args.max_n, graph_args.num_vertex_type)
            node_str = adjfeat_to_G(*bfs(adj, feat))  # 1 * n_vertex * (n_types + n_vertex)
            # if g_orig.shape[1] < max_n:
            #     padding = torch.zeros(1, max_n-g.shape[1], g.shape[2]).to(get_device())
            #     padding[0, :, START_TYPE] = 1  # treat padding nodes as start_type
            #     g = torch.cat([g, padding], 1)  # 1 * max_n * (n_types + n_vertex)
            # if g.shape[2] < xs:
            #     padding = torch.zeros(1, g.shape[1], xs-g.shape[2]).to(get_device())
            #     g = torch.cat([g, padding], 2)  # pad zeros to indicate no connections to padding 
            #                                     # nodes, g: 1 * max_n * xs
            node_str = node_str[0]
        elif args.order == "random":
            node_str = torch.tensor([node_str])
            adj, feat = G_to_adjfeat(node_str, graph_args.max_n, graph_args.num_vertex_type)
            order = np.random.permutation(len(adj))
            adj, feat = adj[order, :][:, order], feat[order]
            node_str = adjfeat_to_G(adj, feat)  # 1 * n_vertex * (n_types + n_vertex)             
            node_str = node_str[0]
        res.append((node_str, g_orig))
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_path = tmp_file.name+'.pkl'
        pickle.dump(res, open(tmp_path, 'wb+'))
    print("done")
    return tmp_path


# Sampling new sequences
def sample(model, num_samples=5, max_seq_len=10):
    model.eval()
    uniq_sequences = set()
    with torch.no_grad():
        while len(uniq_sequences) < num_samples:
            z = torch.randn(num_samples, args.latent_dim)  # Sample from the prior
            z = z.to(args.cuda)
            generated_sequences = model.autoregressive_inference(z, token2rule, max_seq_len)  # Decode from latent space            
            uniq_sequences = uniq_sequences | set(map(tuple, generated_sequences))
    uniq_sequences = [list(l) for l in uniq_sequences]
    return uniq_sequences[:num_samples]


def standardize_enas(g, path=None):
    if path is None:
        path = TransformerVAE.path_init(g)[0] # could be done if valid        
    g = nx.relabel_nodes(g, dict(zip(path, list(range(len(g))))))
    g = copy_graph(nx.DiGraph(g), list(range(len(g))))
    for n in g:
        g.nodes[n]['type'] = list(LOOKUP).index(g.nodes[n]['type'])    
    return g

def standardize_ckt(g):
    subg_type = {
        "input": 0,
        "output": 1,
        "R": 2,
        "C": 3,
        "+gm+": 6,
        "-gm+": 7,
        "+gm-": 8,
        "-gm-": 9
    }   
    g = deepcopy(g)
    for n in g:
        g.nodes[n]['type'] = subg_type[g.nodes[n]['type']]
    return g    


def standardize_bn(g):
    g = deepcopy(g)
    label_lookup = {'input': 0, 'output': 1, 'A': 2, 'S': 3, 'T': 4, 'L': 5, 'B': 6, 'E': 7, 'X': 8, 'D': 9}    
    for n in g:
        g.nodes[n]['type'] = label_lookup[INVERSE_LOOKUP[g.nodes[n]['label']]]
    s = get_node_by_label(g, 0, attr='type')
    t = get_node_by_label(g, 1, attr='type')
    path = [s] + [n for n in g if n not in [s, t]] + [t]
    g = nx.relabel_nodes(g, dict(zip(path, range(len(g)))))
    return g


def decode_from_latent_space(z, grammar, model, token2rule, max_seq_len):
    generated_dags = [None for _ in range(z.shape[0])]
    generated_derivs = [None for _ in range(z.shape[0])]
    idxes = list(range(z.shape[0]))
    with tqdm(total=z.shape[0], desc="decoding") as pbar:
        while idxes:     
            generated_sequences, generated_orders, generated_paths = model.autoregressive_interactive_inference(z, grammar, token2rule, max_seq_len, decode='softmax')
            new_idxes = []
            mask = []
            for idx, deriv, order, paths in zip(idxes, generated_sequences, generated_orders, generated_paths):
                if DATASET == "enas":
                    assert len(paths) > 0
                    path = paths[0]
                g = grammar.derive(deriv, token2rule)
                for n in g:
                    g.nodes[n]['type'] = INVERSE_LOOKUP[g.nodes[n]['label']]
                if DATASET == 'ckt':
                    try: # not our fault, but due to the converter assuming 2 or 3-stage op-amps, we'll keep sampling until we satisfy that restriction
                        normalize_format(g)
                    except ValueError:
                        new_idxes.append(idx)
                        mask.append(True) 
                        continue
                elif DATASET == "enas":
                    g = standardize_enas(g, path)
                else:
                    g = standardize_bn(g)
                generated_dags[idx] = g
                generated_derivs[idx] = deriv
                mask.append(False)          
            idxes = new_idxes
            z = z[mask]
            pbar.update(len(idxes)-pbar.n)
    return generated_dags, generated_derivs
    

def decode_from_latent_space_ns(z, model, max_seq_len):
    G = [None for _ in range(z.shape[0])]
    valid_ns_final = [None for _ in range(z.shape[0])]
    idxes = list(range(z.shape[0]))
    total_decode_attempts = [0 for _ in range(z.shape[0])]
    with tqdm(total=z.shape[0], desc="decoding") as pbar:
        while idxes:
            generated_ns = model.ns_decode(z, max_seq_len=max_seq_len)
            # generated_dags, generated_sequences = decode_from_latent_space(z, grammar, model, token2rule, max_seq_len=max_seq_len)
            new_idxes = []
            mask = []
            for idx, seq in zip(idxes, generated_ns):
                adj = torch.stack(seq, dim=0).int()
                if args.order == "default":
                    g = construct_graph(adj)
                else:
                    g = construct_graph_full(adj)                
                if total_decode_attempts[idx] == 10: # failed 10 times
                    G[idx] = None
                    valid_ns_final[idx] = None
                    mask.append(False)
                    continue
                if DATASET == 'ckt':
                    try: # not our fault, but due to the converter assuming 2 or 3-stage op-amps, we'll keep sampling until we satisfy that restriction
                        normalize_format(g)
                    except ValueError:
                        mask.append(True)
                        new_idxes.append(idx)  
                        total_decode_attempts[idx] += 1
                        continue
                elif DATASET == "enas":
                    try:
                        g = standardize_enas(g)
                        assert len(g) == 8
                        assert is_valid_ENAS(nx_to_igraph(g))
                    except:
                        mask.append(True)
                        new_idxes.append(idx)
                        total_decode_attempts[idx] += 1
                        continue
                else:
                    try:
                        g = standardize_bn(g)
                        assert is_valid_BN(nx_to_igraph(g))        
                    except:
                        mask.append(True)
                        new_idxes.append(idx)
                        total_decode_attempts[idx] += 1
                        continue
                G[idx] = g
                valid_ns_final[idx] = seq
                mask.append(False)
            idxes = new_idxes
            z = z[mask]  
            pbar.update(len(idxes)-pbar.n)
    return G, valid_ns_final


def train(args, train_data, test_data):    
    ckpt_dir = f'{args.ckpt_dir}/ckpts/{Path(CACHE_DIR).stem}/{args.folder}'
    if len(os.listdir(f"{ckpt_dir}")) == 0: # add necc. ckpts
        breakpoint()
    # Initialize model
    if args.repr == "digged":
        model = TransformerVAE(args.encoder, args.repr, args.encoder_layers, args.decoder_layers, VOCAB_SIZE, vocabulary_init, vocabulary_terminate, args.embed_dim, args.latent_dim, MAX_SEQ_LEN, args.cuda)
    elif args.repr == "ns":
        model = TransformerVAE(args.encoder, args.repr, args.encoder_layers, args.decoder_layers, (VOCAB_SIZE, graph_args), None, None, args.embed_dim, args.latent_dim, MAX_SEQ_LEN, args.cuda)
    else:
        raise NotImplementedError
    model = model.to(args.cuda)

    # Load ckpts   
    ckpts = glob.glob(f'{ckpt_dir}/*.pth')
    logger.info(f'{ckpt_dir}/*.pth')
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
        model.load_state_dict(torch.load(best_ckpt_path, map_location=args.cuda))
    if start_epoch >= args.epochs:
        return model

    # Prepare data
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
    
    if args.encoder == "TOKEN_GNN":
        for i, graph_data in enumerate(graph_data_vocabulary):
            graph_data_vocabulary[i] = graph_data.to(args.cuda)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # Training loop
    model.train()    
    patience = 5
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
                    if args.repr == "digged":
                        batch = collate_batch_gnn(g_batch)
                    else:
                        batch = collate_batch_gnn_ns(g_batch)
                else:
                    batch = collate_batch(g_batch)
                x, attention_mask, seq_len_list, batch_g_list, batch_idxes = batch
                x, attention_mask = x.to(args.cuda), attention_mask.to(args.cuda)
                optimizer.zero_grad()
                recon_logits, mask, mu, logvar = model(x, attention_mask, seq_len_list, batch_g_list)                           
                loss = vae_loss(args, recon_logits, mask, x, mu, logvar)
                loss.backward()            
                train_loss += loss.item()*len(batch_idxes)                
                if args.repr == "digged":
                    rll = recon_logits.argmax(axis=-1).reshape(x.shape)
                    rec_acc = (rll == x).all(axis=-1)
                    rec_acc_sum += rec_acc.sum()
                else:
                    rll = recon_logits.detach().clone().reshape(x.shape)
                    type_probs = rll[:, :, :graph_args.num_vertex_type]
                    type_probs = F.softmax(type_probs)
                    type_probs = type_probs.reshape(-1, graph_args.num_vertex_type)
                    new_type = torch.multinomial(type_probs, 1)
                    rll[:, :, :graph_args.num_vertex_type] = model._one_hot(new_type.reshape(-1).tolist(), model.nvt).reshape(x.shape[:2]+(graph_args.num_vertex_type,))
                    rll[:, :, graph_args.num_vertex_type:] = rll[:, :, graph_args.num_vertex_type:] > 0.5
                    if args.order == "default":
                        gs = [construct_graph(adj) for adj in rll.unbind(0)]
                    else:
                        gs = [construct_graph_full(adj) for adj in rll.unbind(0)]
                    origs = [nx.induced_subgraph(orig, orig.comps[g[-1]]) for g in g_batch]
                    rec_acc_sum = sum([nx.is_isomorphic(g, o, node_match=node_match) for (g, o) in zip(gs, origs)])
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
                        if args.repr == "digged":
                            batch = collate_batch_gnn(g_batch)
                        else:
                            batch = collate_batch_gnn_ns(g_batch)
                    else:
                        batch = collate_batch(g_batch)               
                    x, attention_mask, seq_len_list, batch_g_list, batch_idxes = batch
                    x, attention_mask = x.to(args.cuda), attention_mask.to(args.cuda)
                    recon_logits, mask, mu, logvar = model(x, attention_mask, seq_len_list, batch_g_list)
                    loss = vae_loss(args, recon_logits, mask, x, mu, logvar)            
                    val_loss += loss.item()*len(batch_idxes)
                    if args.repr == "digged":
                        rll = recon_logits.argmax(axis=-1).reshape(x.shape)
                        rec_acc = (rll == x).all(axis=-1)
                        rec_acc_sum += rec_acc.sum()
                    else:
                        rll = recon_logits.reshape(x.shape)
                        rec_acc = ((rll > 0.5) == x).all(axis=[1,2])
                        rec_acc_sum += rec_acc.sum()
                    test_latent[batch_idxes] = mu.detach().cpu().numpy()
                    g_batch = []   
        val_loss /= len(test_dataset)
        valid_rec_acc_mean = rec_acc_sum / len(test_dataset)
        if val_loss < best_loss:
            patience_counter = 0 # reset counter
            best_loss = val_loss
            ckpt_path = f'{ckpt_dir}/epoch={epoch}_loss={best_loss}.pth'
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
        np.save(f'{ckpt_dir}/train_latent_{epoch}.npy', train_latent)
        np.save(f'{ckpt_dir}/test_latent_{epoch}.npy', test_latent)
        fig = model.visualize_tokens()
        fig.savefig(f'{ckpt_dir}/{epoch}.png')        
        embedding = model.token_embedding.weight.detach().cpu().numpy()
        np.save(f'{ckpt_dir}/embedding_{epoch}.npy', embedding)
        if patience_counter > patience:
            logger.info(f"Early stopping triggered at epoch {epoch}. Best loss: {best_loss}")
            break
    return model


def train_sgp(args, save_file, X_train, X_test, y_train, y_test):
    # We fit the GP
    M = 500
    # other BO hyperparameters
    lr = 0.005  # the learning rate to train the SGP model
    max_iter = args.max_iter  # how many iterations to optimize the SGP each time
    sgp = SparseGP(X_train, 0 * X_train, y_train, M)
    sgp.train_via_ADAM(X_train, 0 * X_train, y_train, X_test, X_test * 0, y_test, minibatch_size = 2 * M, max_iterations = max_iter, learning_rate = lr)
    pred, uncert = sgp.predict(X_test, 0 * X_test)    
    logger.info(f"predictions: {pred.reshape(-1)}")
    logger.info(f"real values: {y_test.reshape(-1)}")
    error = np.sqrt(np.mean((pred - y_test)**2))
    testll = np.mean(sps.norm.logpdf(pred - y_test, scale = np.sqrt(uncert)))
    logger.info(f'Test RMSE: {error}')
    logger.info(f'Test ll: {testll}')
    pearson = float(pearsonr(pred.flatten(), y_test.flatten())[0])
    logger.info(f'Pearson r: {pearson}')
    breakpoint()
    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test.flatten(), pred.flatten(), color="blue", alpha=0.6, label="Predictions")
    ax.plot(y_test.flatten(), y_test.flatten(), color="red", linestyle="--", label="Perfect Prediction (y=x)")
    # Add annotations
    ax.text(0.7, 0.10, f"Pearson R = {pearson:.2f}", transform=fig.gca().transAxes, fontsize=12)
    ax.text(0.7, 0.05, f"RMSE = {error:.2f}", transform=fig.gca().transAxes, fontsize=12)
    # Labels and legend
    ax.set_xlabel("Standardized True Values")
    ax.set_ylabel("Standardized Predicted Values")
    ax.set_title("Sparse GP Predictions on Test Set")
    ax.legend()
    ax.grid(alpha=0.3)
    path = Path(save_file)
    now = time.time()
    fig.savefig(os.path.join(path.parent, path.stem+f'_{now}.png'), bbox_inches='tight')
    print(os.path.abspath(os.path.join(path.parent, path.stem+f'_{now}.png')))
    with open(save_file, 'a+') as test_file:
        test_file.write('Test RMSE: {:.4f}, ll: {:.4f}, Pearson r: {:.4f}\n'.format(error, testll, pearson))
    error_if_predict_mean = np.sqrt(np.mean((np.mean(y_train, 0) - y_test)**2))
    logger.info(f'Test RMSE if predict mean: {error_if_predict_mean}')    
    return sgp



def bo(args, orig, grammar, model, token2rule, y_train, y_test, target_mean, target_std):
    ### IMPORTANT: y_train and y_test are 2D numpys, where the last column is the BO property
    folder = args.datapkl if args.datapkl else args.folder
    ckpt_dir = f'{args.ckpt_dir}/ckpts/{Path(CACHE_DIR).stem}/{args.folder}'
    save_dir = f'results/{Path(CACHE_DIR).stem}/{args.folder}/'
    os.makedirs(save_dir, exist_ok=True)
    X_train = np.load(os.path.join(ckpt_dir, f"train_latent_{args.checkpoint}.npy"))
    X_test = np.load(os.path.join(ckpt_dir, f"test_latent_{args.checkpoint}.npy"))
    if args.dataset == "bn":
        # remove duplicates, otherwise SGP ill-conditioned
        X_train, unique_idxs = np.unique(X_train, axis=0, return_index=True)
        y_train = y_train[unique_idxs]
        random_shuffle = np.random.permutation(range(len(X_train)))
        keep = 5000
        X_train = X_train[random_shuffle[:keep]]
        y_train = y_train[random_shuffle[:keep]]    
    else:
        keep = y_train.shape[0]
    X_train_mean = X_train.mean(axis=0)
    X_train_std = X_train.std(axis=0)
    X_train = (X_train-X_train_mean)/X_train_std
    X_test = (X_test-X_train_mean)/X_train_std    
    best_score = float("inf")
    best_arc = None
    novel_arcs = []
    iteration = 0
    logger.info("Average pairwise distance between train points = {}".format(np.mean(pdist(X_train))))
    logger.info("Average pairwise distance between test points = {}".format(np.mean(pdist(X_test))))    
    if DATASET == "ckt":
        evaluate_fn = partial(evaluate_ckt, args)
    elif DATASET == "enas":
        evaluate_fn = partial(evaluate_nn, default_val=min(y_train)*target_std+target_mean)
    elif DATASET == "bn":
        eva = Eval_BN(save_dir)
        evaluate_fn = lambda g: eva.eval(decode_igraph_to_BN_adj(nx_to_igraph(g)))
    else:
        raise NotImplementedError
    # for _ in range(10):
    #     for i in range(y_train.shape[1]): # evaluate latent space
    #         save_file = os.path.join(save_dir, f'Prop_{i}_Test_RMSE_ll.txt')
    #         sgp = train_sgp(args, save_file, X_train, X_test, y_train[:, i:i+1], y_test[:, i:i+1])
    # breakpoint()
    y_train, y_test = y_train[:, -1:], y_test[:, -1:]
    save_file = os.path.join(save_dir, f'Test_RMSE_ll.txt')
    while iteration < args.BO_rounds:
        logger.info(f"Iteration: {iteration}")
        sgp = train_sgp(args, save_file, X_train, X_test, y_train, y_test)
        pred, uncert = sgp.predict(X_train, 0 * X_train)
        error = np.sqrt(np.mean((pred - y_train)**2))
        trainll = np.mean(sps.norm.logpdf(pred - y_train, scale = np.sqrt(uncert)))
        logger.info(f'Train RMSE: {error}')
        logger.info(f'Train ll: {trainll}')        
        next_inputs = sgp.batched_greedy_ei(args.BO_batch_size, np.min(X_train, 0), np.max(X_train, 0), np.mean(X_train, 0), np.std(X_train, 0), sample=args.sample_dist, max_iter=args.max_ei_iter, factr=args.factr)
        valid_arcs_final, generated_sequences = decode_from_latent_space(torch.FloatTensor(next_inputs).to(args.cuda), grammar, model, token2rule, MAX_SEQ_LEN)
        new_features = next_inputs
        logger.info("Evaluating selected points")
        scores = []
        if args.dataset == "enas":
            # Prepare data            
            scores = send_enas_listener(valid_arcs_final)
            scores = [-score for score in scores]
            for i, score in enumerate(scores):
                if is_novel(valid_arcs_final[i], orig):
                    novel_arcs.append((valid_arcs_final[i], score, iteration*args.BO_batch_size+i))                
                if score < best_score:
                    best_score = score
                    best_deriv = '->'.join(map(str, generated_sequences[i]))
                    best_arc = valid_arcs_final[i]                    
        else:
            for i in range(len(valid_arcs_final)):
                score = -evaluate_fn(valid_arcs_final[i])
                if score == float("inf") or score != score:
                    score = y_train[:keep].max()*target_std+target_mean
                # if is_novel(valid_arcs_final[i], orig):
                #     novel_arcs.append((valid_arcs_final[i], score, iteration*args.BO_batch_size+i))
                if score < best_score:
                    best_score = score
                    best_deriv = '->'.join(map(str, generated_sequences[i]))
                    best_arc = valid_arcs_final[i]
                scores.append(score)
            # logger.info(i, score)
        # logger.info("Iteration {}'s selected arcs' scores:".format(iteration))
        # logger.info(scores, np.mean(scores))
        save_object(scores, "{}scores{}.dat".format(save_dir, iteration))
        save_object(valid_arcs_final, "{}valid_arcs_final{}.dat".format(save_dir, iteration))
        save_object(generated_sequences, "{}generated_sequences{}.dat".format(save_dir, iteration))
        # save_object(novel_arcs, "{}novel_arcs.dat".format(save_dir))
        if len(new_features) > 0:
            scores = np.array(scores)[:, None]
            X_train = np.concatenate([X_train, new_features], 0)
            std_scores = (scores-target_mean)/target_std
            y_train = np.concatenate([y_train, std_scores], 0)
        #
        # logger.info("Current iteration {}'s best score: {}".format(iteration, - best_score * std_y_train - mean_y_train))
        if best_arc is not None: # and iteration == 10:
            logger.info(f"Best deriv: {best_deriv}")
            with open(save_dir + 'best_deriv_scores.txt', 'a') as score_file:
                score_file.write(best_deriv + ',{:.4f}\n'.format(best_score))
            if args.dataset == 'enas':
                # row = [int(x) for x in best_arc.split()]
                # g_best, _ = decode_ENAS_to_igraph(flat_ENAS_to_nested(row, 8-2))
                g_best = nx_to_igraph(best_arc)
                plot_DAG(g_best, save_dir, 'best_arc_iter_{}'.format(iteration), data_type='ENAS', pdf=True)
            elif args.dataset == 'bn':
                # row = adjstr_to_BN(best_arc)
                # g_best, _ = decode_BN_to_igraph(row)
                g_best = nx_to_igraph(best_arc)
                plot_DAG(g_best, save_dir, 'best_arc_iter_{}'.format(iteration), data_type='BN', pdf=True)
            elif args.dataset == "ckt":
                g_best = nx_to_igraph(standardize_ckt(best_arc))
                plot_circuits((g_best,), save_dir, 'best_arc_iter_{}'.format(iteration), pdf=True)
        #
        iteration += 1


def bo_ns(args, model, y_train, y_test, target_mean, target_std):
    ### IMPORTANT: y_train and y_test are 2D numpys, where the last column is the BO property
    folder = args.datapkl if args.datapkl else args.folder
    ckpt_dir = f'{args.ckpt_dir}/ckpts/{Path(CACHE_DIR).stem}/{args.folder}'
    save_dir = f'results/{Path(CACHE_DIR).stem}/{args.folder}/'
    os.makedirs(save_dir, exist_ok=True)
    X_train = np.load(os.path.join(ckpt_dir, f"train_latent_{args.checkpoint}.npy"))
    X_test = np.load(os.path.join(ckpt_dir, f"test_latent_{args.checkpoint}.npy"))
    if args.dataset == "bn":
        # remove duplicates, otherwise SGP ill-conditioned
        X_train, unique_idxs = np.unique(X_train, axis=0, return_index=True)
        y_train = y_train[unique_idxs]
        random_shuffle = np.random.permutation(range(len(X_train)))
        keep = 5000
        X_train = X_train[random_shuffle[:keep]]
        y_train = y_train[random_shuffle[:keep]]    
    else:
        keep = y_train.shape[0]
    X_train_mean = X_train.mean(axis=0)
    X_train_std = X_train.std(axis=0)
    X_train = (X_train-X_train_mean)/X_train_std
    X_test = (X_test-X_train_mean)/X_train_std    
    best_score = float("inf")
    best_arc = None
    novel_arcs = []
    iteration = 0
    logger.info("Average pairwise distance between train points = {}".format(np.mean(pdist(X_train))))
    logger.info("Average pairwise distance between test points = {}".format(np.mean(pdist(X_test))))    
    if DATASET == "ckt":
        evaluate_fn = partial(evaluate_ckt, args)
    elif DATASET == "enas":
        evaluate_fn = partial(evaluate_nn, default_val=min(y_train)*target_std+target_mean)
    elif DATASET == "bn":
        eva = Eval_BN(save_dir)
        evaluate_fn = lambda g: eva.eval(decode_igraph_to_BN_adj(nx_to_igraph(g)))
    else:
        raise NotImplementedError
    # for _ in range(10):
    #     for i in range(y_train.shape[1]): # evaluate latent space
    #         save_file = os.path.join(save_dir, f'Prop_{i}_Test_RMSE_ll.txt')
    #         sgp = train_sgp(args, save_file, X_train, X_test, y_train[:, i:i+1], y_test[:, i:i+1])
    # breakpoint()
    y_train, y_test = y_train[:, -1:], y_test[:, -1:]
    save_file = os.path.join(save_dir, f'Test_RMSE_ll.txt')
    while iteration < args.BO_rounds:
        logger.info(f"Iteration: {iteration}")
        sgp = train_sgp(args, save_file, X_train, X_test, y_train, y_test)
        pred, uncert = sgp.predict(X_train, 0 * X_train)
        error = np.sqrt(np.mean((pred - y_train)**2))
        trainll = np.mean(sps.norm.logpdf(pred - y_train, scale = np.sqrt(uncert)))
        logger.info(f'Train RMSE: {error}')
        logger.info(f'Train ll: {trainll}')        
        next_inputs = sgp.batched_greedy_ei(args.BO_batch_size, np.min(X_train, 0), np.max(X_train, 0), np.mean(X_train, 0), np.std(X_train, 0), sample=args.sample_dist, max_iter=args.max_ei_iter, factr=args.factr)
        valid_arcs_final, valid_ns_final = decode_from_latent_space_ns(torch.FloatTensor(next_inputs).to(args.cuda), model, MAX_SEQ_LEN)
        new_features = next_inputs
        logger.info("Evaluating selected points")
        scores = []
        if args.dataset == "enas":
            # Prepare data
            mask = np.array([arc is not None for arc in valid_arcs_final])
            scores = np.zeros((len(valid_arcs_final,)))
            scores[mask] = send_enas_listener(list(filter(None, valid_arcs_final)))                        
            scores[~mask] = max(y_train)*target_std+target_mean
            scores = [-score for score in scores]
            for i, score in enumerate(scores):
                if valid_arcs_final[i] is None:
                    continue
                if is_novel(valid_arcs_final[i], orig):
                    novel_arcs.append((valid_arcs_final[i], score, iteration*args.BO_batch_size+i))                
                if score < best_score:
                    best_score = score
                    best_ns = valid_ns_final[i]
                    best_arc = valid_arcs_final[i]                    
        else:
            for i in range(len(valid_arcs_final)):
                if valid_arcs_final[i] is not None:
                    score = -evaluate_fn(valid_arcs_final[i])
                else:
                    score = float("inf")
                if score == float("inf") or score != score:
                    score = y_train[:keep].max()*target_std+target_mean
                if valid_arcs_final[i] is None:
                    continue                    
                # if is_novel(valid_arcs_final[i], orig):
                #     novel_arcs.append((valid_arcs_final[i], score, iteration*args.BO_batch_size+i))
                if score < best_score:
                    best_score = score
                    best_ns = valid_ns_final[i]
                    best_arc = valid_arcs_final[i]
                scores.append(score)
            # logger.info(i, score)
        # logger.info("Iteration {}'s selected arcs' scores:".format(iteration))
        # logger.info(scores, np.mean(scores))
        save_object(scores, "{}scores{}.dat".format(save_dir, iteration))
        save_object(valid_arcs_final, "{}valid_arcs_final{}.dat".format(save_dir, iteration))
        # save_object(novel_arcs, "{}novel_arcs.dat".format(save_dir))
        if len(new_features) > 0:
            scores = np.array(scores)[:, None]
            X_train = np.concatenate([X_train, new_features], 0)
            std_scores = (scores-target_mean)/target_std
            y_train = np.concatenate([y_train, std_scores], 0)
        #
        # logger.info("Current iteration {}'s best score: {}".format(iteration, - best_score * std_y_train - mean_y_train))
        if best_arc is not None: # and iteration == 10:
            # logger.info(f"Best ns: {best_ns}")
            # with open(save_dir + 'best_ns_scores.txt', 'a') as score_file:
            #     score_file.write(best_ns + ',{:.4f}\n'.format(best_score))
            if args.dataset == 'enas':
                # row = [int(x) for x in best_arc.split()]
                # g_best, _ = decode_ENAS_to_igraph(flat_ENAS_to_nested(row, 8-2))
                g_best = nx_to_igraph(best_arc)
                plot_DAG(g_best, save_dir, 'best_arc_iter_{}'.format(iteration), data_type='ENAS', pdf=True)
            elif args.dataset == 'bn':
                # row = adjstr_to_BN(best_arc)
                # g_best, _ = decode_BN_to_igraph(row)
                g_best = nx_to_igraph(best_arc)
                plot_DAG(g_best, save_dir, 'best_arc_iter_{}'.format(iteration), data_type='BN', pdf=True)
            elif args.dataset == "ckt":
                g_best = nx_to_igraph(standardize_ckt(best_arc))
                plot_circuits((g_best,), save_dir, 'best_arc_iter_{}'.format(iteration), pdf=True)
        #
        iteration += 1



def visualize_sequences(sampled_sequences, grammar, token2rule):
    logger.info("===SAMPLED SEQUENCES===")
    for i, seq in enumerate(sampled_sequences):
        logger.info('->'.join(map(str, seq)))
        # Visualize new sequence
        path = f'data/api_{DATASET}_ednce/generate/{i}.png'
        img_path = f'data/api_{DATASET}_ednce/generate/{i}_g.png'
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


def interactive_sample_sequences(args, model, grammar, token2rule, num_samples=5, max_seq_len=10, unique=False, visualize=False):
    num_samples = args.num_samples
    sample_batch_size = args.sample_batch_size
    model.eval()
    if unique:
        uniq_sequences = set()
    else:
        sequences = []
    with torch.no_grad():
        with tqdm(total=num_samples) as pbar:
            while len(uniq_sequences if unique else sequences) < num_samples:
                z = torch.randn(sample_batch_size, args.latent_dim)  
                z = z.to(args.cuda)            
                generated_sequences, _, _ = model.autoregressive_interactive_inference(z, grammar, token2rule, max_seq_len=max_seq_len, decode='softmax')
                # generated_dags, generated_sequences = decode_from_latent_space(z, grammar, model, token2rule, max_seq_len=max_seq_len)
                if unique:
                    for seq in generated_sequences:
                        if tuple(seq) not in uniq_sequences:
                            uniq_sequences.add(tuple(seq))
                            logger.info(f"generated {seq}")
                    pbar.update(len(uniq_sequences)-pbar.n)
                else:
                    for seq in generated_sequences:
                        logger.info(f"generated {seq}")
                    sequences += generated_sequences
                    pbar.update(len(sequences)-pbar.n)
    if unique:
        uniq_sequences = [list(l) for l in uniq_sequences]
        sampled_sequences = uniq_sequences[:num_samples]
    else:
        sampled_sequences = sequences[:num_samples]
    logger.info("===SAMPLED SEQUENCES===")
    gs = []
    for i, seq in enumerate(sampled_sequences):
        logger.info('->'.join(map(str, seq)))
        # Visualize new sequence
        if visualize:
            path = f'data/api_{args.dataset}_ednce/generate/{i}.png'
            img_path = f'data/api_{args.dataset}_ednce/generate/{i}_g.png'
            fig, axes = plt.subplots(len(seq), figsize=(5, 5*(len(seq))))
            for idx, j in enumerate(map(int, seq)):
                r = grammar.rules[j]
                draw_graph(r.subgraph, ax=axes[idx], scale=5)
            fig.savefig(path)
            logger.info(os.path.abspath(path))
            g = grammar.derive(seq, token2rule)
            draw_graph(g, path=img_path)    
        else:
            g = grammar.derive(seq, token2rule)
        gs.append(g)
    return gs


def ns_sample_sequences(args, model, max_seq_len=10, unique=False, visualize=False):
    num_samples = args.num_samples
    sample_batch_size = args.sample_batch_size
    model.eval()
    G = []
    with torch.no_grad():
        with tqdm(total=num_samples) as pbar:
            while len(G) < num_samples:
                z = torch.randn(sample_batch_size, args.latent_dim)  
                z = z.to(args.cuda)            
                generated_ns = model.ns_decode(z, max_seq_len=max_seq_len)
                # generated_dags, generated_sequences = decode_from_latent_space(z, grammar, model, token2rule, max_seq_len=max_seq_len)
                for seq in generated_ns:      
                    adj = torch.stack(seq, dim=0).int()
                    if args.order == "default":
                        g = construct_graph(adj)
                    else:
                        g = construct_graph_full(adj)
                    G.append(g)
                    pbar.update(len(G)-pbar.n)
    logger.info("===SAMPLED GRAPHS===")
    if visualize:
        for i, g in enumerate(G):
            img_path = f'data/api_{args.dataset}_ednce/generate/ns_{i}.png'
            draw_graph(g, path=img_path)    
    return G
    

def load_y(g, num_graphs, target):
    y = []
    for pre in range(num_graphs):
        label = []
        for t in target:
            score = g.graph[f'{pre}:{t}']
            label.append(-score) # loss
        y.append(label) 
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


def process_single(g_orig, rules, iso=True):
    # iso is for node feats, must match isomorphism first
    # node feats only present in circuit graphs
    rule_ids = [r[0] for r in rules]
    rules = [r[1] for r in rules]
    g, all_applied, all_node_maps = derive(rules)
    if iso:
        matcher = DiGraphMatcher(g, g_orig, node_match=node_match)
        iso = next(matcher.isomorphisms_iter())
    # use iso to embed feats and instructions        
    for i, r in enumerate(rules):
        sub = nx.DiGraph(r.subgraph)
        # node feats
        if iso:
            for n in sub:
                key = all_node_maps[i][n]
                if key in iso:
                    #print(g_orig.nodes[iso[key]]['feat'])
                    sub.nodes[n]['feat'] = g_orig.nodes[iso[key]]['feat']
        rule_ids[i] = (rule_ids[i], sub, all_applied[i-1] if i else None)
    return rule_ids

# def process_single_one_hot(g_orig, rules):
#     rule_ids = [r[0] for r in rules]
#     rules = [r[1] for r in rules]
#     g, all_applied, all_node_maps = derive(rules)
#     matcher = DiGraphMatcher(g, g_orig, node_match=node_match)
#     iso = next(matcher.isomorphisms_iter())
#     # use iso to embed feats and instructions        
#     for i, r in enumerate(rules):
#         sub = deepcopy(nx.DiGraph(r.subgraph))
#         # node feats
#         for n in sub:
#             key = all_node_maps[i][n]
#             if key in iso:
#                 node_label = g_orig.nodes[iso[key]]['label']
#                 one_hot_vec = to_one_hot(node_label, TERMS + NONTERMS)
#                 sub.nodes[n]['feat'] = one_hot_vec
#         rule_ids[i] = (rule_ids[i], sub, all_applied[i-1] if i else None)
#     return rule_ids


def stringify(g):
    assert sorted([get_suffix(n) for n in g]) == list(range(len(g)))
    arr = [[0 for _ in range(len(LOOKUP)+len(g)-1)] for _ in range(len(g)-1)]
    for n in g:
        i = get_suffix(n)
        if i == 0:
            continue
        j = list(LOOKUP).index(g.nodes[n]['type'])
        arr[i-1][j] = 1
    for a, b in g.edges:
        i1 = get_suffix(a)
        i2 = get_suffix(b)
        arr[i2-1][len(LOOKUP)+i1] = 1
    # use START_TYPE to pad
    for i in range(len(arr), graph_args.max_n-1):
        row = [0 for _ in range(len(LOOKUP)+len(g)-1)]
        row[graph_args.START_TYPE] = 1
        arr.append(row)
    arr = [row+[0 for _ in range(len(arr[0]), graph_args.num_vertex_type+graph_args.max_n-1)] for row in arr]
    # pad empty
    return arr
    
### the following functions are copied from D-VAE/models.py, should later import instead
def bfs(adj, feat):
    n = len(adj)
    queue = deque([random.randint(0, n-1)])
    visited = set()
    order = []
    while queue:
        cur = queue.popleft()
        if cur in visited:
            continue
        order.append(cur)
        visited.add(cur)
        successors = adj[cur].nonzero().flatten().tolist()
        predecessors = adj[:, cur].nonzero().flatten().tolist()
        neis = set(successors + predecessors)
        neis = neis - visited
        for x in neis:
            queue.append(x)
    return adj[order, :][:, order], feat[order]

def G_to_adjfeat(G, max_n, nvt):
    # convert SVAE's G tensor to adjacency matrix and node features
    assert(G.shape[0]==1)
    G = G[0]
    pad = torch.zeros(1, nvt).to(G.device)
    pad[:, 0] = 1
    input_features = torch.cat([pad, G[:, :nvt]], 0)  # add the start node
    pad2 = torch.zeros(max_n-1, 1).to(G.device)
    adj = torch.cat([pad2, G[:, nvt:].permute(1, 0)], 1)
    pad3 = torch.zeros(1, max_n).to(G.device)
    adj = torch.cat([adj, pad3], 0)
    return adj, input_features

def adjfeat_to_G(adj, feat):
    # the new G tensor contains starting node as well as connections of last node
    adj = adj.permute(1, 0)
    return torch.cat([feat, adj], 1).unsqueeze(0)

def construct_graph(adj):
    g = nx.DiGraph()
    g.add_node(0, type=graph_args.START_TYPE)
    for vj in range(1, graph_args.max_n):
        if vj == graph_args.max_n - 1:
            new_type = graph_args.END_TYPE
        else:
            new_type = torch.argmax(adj[vj-1], 0).item()
        g.add_node(vj, type=new_type)
        if new_type == graph_args.END_TYPE:  
            for v in range(vj):
                if g.out_degree(v) == 0:
                    g.add_edge(v, vj, label='black')
            break
        else:
            for ek in range(vj):
                ek_score = adj[vj-1][graph_args.num_vertex_type+ek].item()
                if ek_score > 0.5:
                    g.add_edge(ek, vj, label='black')
    for n in g:
        g.nodes[n]['type'] = list(LOOKUP)[g.nodes[n]['type']]
        g.nodes[n]['label'] = LOOKUP[g.nodes[n]['type']]
    return g

def construct_graph_full(adj):
    g = nx.DiGraph()
    for vj in range(graph_args.max_n):
        if vj == graph_args.max_n - 1:
            new_type = graph_args.END_TYPE
        else:
            new_type = torch.argmax(adj[vj], 0).item()
        g.add_node(vj, type=new_type)

    for vj in range(graph_args.max_n):
        for ek in range(graph_args.max_n):
            ek_score = adj[vj][graph_args.num_vertex_type+ek].item()
            if ek_score > 0.5:
                g.add_edge(ek, vj, label='black')
    
    for n in g:
        g.nodes[n]['type'] = list(LOOKUP)[g.nodes[n]['type']]
        g.nodes[n]['label'] = LOOKUP[g.nodes[n]['type']]
    return g
### end copying from D-VAE/models.py

def load_data(args, anno, grammar, orig, cache_dir, num_graphs, graph_args):
    globals()['graph_args'] = graph_args
    globals()['orig'] = orig
    loaded = False
    if args.datapkl:
        save_path = os.path.join(cache_dir, args.datapkl, 'data.pkl')
        if os.path.exists(args.datapkl): # specified to load data from args.datapkl path
            logger.info(f"load data from {save_path}")
            if args.repr == "digged":
                data, rule2token = pickle.load(open(save_path, 'rb'))
                loaded = True
            else:
                breakpoint()
    if not loaded:        
        if args.datapkl:
            save_path = os.path.join(cache_dir, args.datapkl, 'data.pkl')
        else:
            save_path = os.path.join(cache_dir, args.folder, 'data.pkl')
        if args.repr == "digged" and args.dataset == "ckt":
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
            if args.repr == "digged":
                data, rule2token = pickle.load(open(save_path, 'rb'))
            else:
                data = pickle.load(open(save_path, 'rb'))
        elif args.repr == "digged":
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
                g_orig = nx.induced_subgraph(orig, orig.comps[pre])
                if args.encoder == "GNN":
                    g_orig = g_orig.copy()
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
                    pargs.append((g_orig, rules, args.dataset == "ckt"))
            if args.encoder != "GNN":
                #data = [process_single(parg) for parg in tqdm(pargs, "processing data")]
                # if args.dataset == 'enas':
                #     #data = [process_single_one_hot(parg) for parg in tqdm(pargs, "processing data")]
                #     data = [process_single_one_hot(*parg) for parg in tqdm(pargs, "processing data sequentially")]
                #     # with mp.Pool(4) as p:
                #     #     data = p.starmap(process_single_one_hot, tqdm(pargs, "processing data mp"))
                # else:
                data = [process_single(*parg) for parg in tqdm(pargs, "processing data")]
                # with mp.Pool(20) as p:
                #     data = p.starmap(process_single, tqdm(pargs, "processing data mp"))
            pickle.dump((data, rule2token), open(save_path, 'wb+'))
        else:                    
            assert args.encoder == "GNN"            
            # pargs = []
            data = []
            for pre in tqdm(range(num_graphs), "gathering node strings"):
                g_orig = nx.induced_subgraph(orig, orig.comps[pre])
                # g_orig = g_orig.copy()
                node_str = stringify(g_orig)
                if args.order == "bfs":
                    node_str = torch.tensor([node_str])
                    adj, feat = G_to_adjfeat(node_str, graph_args.max_n, graph_args.num_vertex_type)
                    node_str = adjfeat_to_G(*bfs(adj, feat))  # 1 * n_vertex * (n_types + n_vertex)
                    if node_str.shape[1] < graph_args.max_n:
                        padding = torch.zeros(1, graph_args.max_n-node_str.shape[1], node_str.shape[2])
                        padding[0, :, graph_args.START_TYPE] = 1  # treat padding nodes as start_type
                        node_str = torch.cat([node_str, padding], 1)  # 1 * max_n * (n_types + n_vertex)
                    if node_str.shape[2] < graph_args.num_vertex_type+graph_args.max_n:
                        padding = torch.zeros(1, node_str.shape[1], graph_args.num_vertex_type+graph_args.max_n-node_str.shape[2])
                        node_str = torch.cat([node_str, padding], 2)  # pad zeros to indicate no connections to padding 
                                                        # nodes, g: 1 * max_n * xs
                    node_str = node_str[0]
                elif args.order == "random":
                    node_str = torch.tensor([node_str])
                    adj, feat = G_to_adjfeat(node_str, graph_args.max_n, graph_args.num_vertex_type)
                    order = np.random.permutation(len(adj))
                    adj, feat = adj[order, :][:, order], feat[order]
                    node_str = adjfeat_to_G(adj, feat)  # 1 * n_vertex * (n_types + n_vertex)             
                    node_str = node_str[0]
                data.append((node_str, orig.comps[pre]))
            #     parg = nx.induced_subgraph(orig, orig.comps[pre])
            #     pargs.append(parg)
            # with mp.Pool(100) as p:
            #     batch_size = 1000
            #     num_batches = (num_graphs + batch_size - 1) // batch_size
            #     futures = []
            #     # Prepare and submit each batch asynchronously.
            #     for i in tqdm(range(num_batches), desc="prepare args"):
            #         # Compute indices for this batch.
            #         start = batch_size * i
            #         end = min(batch_size * (i + 1), num_graphs)
            #         indices = range(start, end)
            #         # Sum over the comps for this batch and create the induced subgraph.
            #         batch_nodes = sum([orig.comps[j] for j in indices], [])
            #         batch_g = nx.induced_subgraph(orig, batch_nodes).copy()
            #         # If your worker needs the comps, attach them (or pass separately).
            #         batch_g.comps = orig.comps                    
            #         # Submit the batch to process_ns asynchronously.
            #         futures.append(p.apply_async(process_ns, args=(batch_g, indices)))                
            #     # Collect results as soon as they are ready.
            #     data = [None for _ in range(num_graphs)]
            #     for future in tqdm(futures, desc="gathering node string batches"):
            #         tmp_path = future.get()
            #         for (node_str, g) in pickle.load(open(tmp_path, 'rb')):
            #             data[get_prefix(list(g)[0])] = (node_str, g)
            #         os.remove(tmp_path)
            pickle.dump(data, open(save_path, 'wb+'))
    if args.repr == "digged":
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
            init[relabel[key]] = grammar.rules[key].nt == 'black'
            terminate[relabel[key]] = term
            vocab.append(graph_data)
        if args.encoder == "GNN":
            globals()['MAX_SEQ_LEN'] = max([len(seq) for seq, _ in data])
        else:
            globals()['MAX_SEQ_LEN'] = max([len(seq) for seq in data])
        globals()['graph_data_vocabulary'] = vocab
        globals()['vocabulary_terminate'] = terminate
        globals()['vocabulary_init'] = init
        globals()['VOCAB_SIZE'] = len(rule2token)
    else:
        globals()['MAX_SEQ_LEN'] = max([len(seq) for seq, _ in data])
        globals()['VOCAB_SIZE'] = max([len(seq[0]) for seq, _ in data])        
    # split here
    indices = list(range(num_graphs))
    # random.Random(0).shuffle(indices)
    train_indices, test_indices = indices[:int(num_graphs*0.9)], indices[int(num_graphs*0.9):]
    train_data = [data[i] for i in train_indices]
    test_data = [data[i] for i in test_indices]
    if args.repr == "digged":
        return train_data, test_data, token2rule
    else:
        return train_data, test_data


def hash_args(args, use_keys=['dataset', 'encoder', 'repr', 'order']):
    # if ablation
    arg_dict = {k: v for k, v in args.__dict__.items() if k in use_keys}
    if 'ablation' in os.environ['config']: # for ablation exps
        arg_dict['config'] = os.environ['config']    
    return hashlib.md5(json.dumps(arg_dict, sort_keys=True).encode()).hexdigest()

def is_novel(g, orig_graphs):
    if isinstance(orig_graphs, nx.DiGraph): # already a graph:
        matcher = DiGraphMatcher(orig_graphs, g, node_match)
        try:
            next(matcher.subgraph_isomorphisms_iter())
            return False
        except:
            print("novel")
            return True
    else:
        for o in orig_graphs:
            if len(o) != len(g):
                continue
            if len(o.edges) != len(g.edges):
                continue
            if nx.is_isomorphic(g, o, node_match=node_match):
                return False
        return True

def str_to_graph(arch):
    delim = ',' if ',' in arch else ' '
    row = list(map(int, arch.split(delim)))
    g_best, _ = decode_ENAS_to_igraph(flat_ENAS_to_nested(row, 8-2))
    g_best = g_best.to_networkx()
    for n in g_best: 
        g_best.nodes[n]['label']=list(LOOKUP.values())[g_best.nodes[n]['type']]    
    return g_best


def str_to_graph(arch):
    delim = ',' if ',' in arch else ' '
    row = list(map(int, arch.split(delim)))
    g_best, _ = decode_ENAS_to_igraph(flat_ENAS_to_nested(row, 8-2))
    g_best = g_best.to_networkx()
    for n in g_best: 
        g_best.nodes[n]['label']=list(LOOKUP.values())[g_best.nodes[n]['type']]    
    return g_best



def is_valid_circuit(g):    
    s = next(filter(lambda n: g.nodes[n]['type'] == 'input', g))
    t = next(filter(lambda n: g.nodes[n]['type'] == 'output', g))
    for path in nx.all_shortest_paths(g, s, t):
        for i in path:
            good = True
            if g.nodes[i]['type'] in ['-gm-','+gm-']:                
                good = False
                predecessors_ = g.predecessors(i)
                successors_ = g.successors(i)
                for v_p in predecessors_:
                    v_p_succ = g.successors(v_p)
                    for v_cand in v_p_succ:
                        inster_set = set(g.successors(v_cand)) & set(successors_)
                        # predecessor's sucessor's successors & successors
                            # each +gm- or -gm- "self" has a predecessor whose successor is R or C
                            # and that R or C shares a successor with the self
                        if g.nodes[v_cand]['type'] in ['R','C'] and len(inster_set):
                            good = True
            if not good:
                return False
    return good              


def evaluate(orig, graphs):
    orig_graphs = [nx.induced_subgraph(orig, nodes) for nodes in orig.comps.values()]
    valid, novel, unique = [], [], []    
    for i, g in tqdm(enumerate(graphs), desc="evaluating"):
        if DATASET == "ckt":
            is_valid_ckt = is_valid_circuit(g)
            valid.append(is_valid_ckt)
        elif DATASET == "enas":
            try:
                valid.append(is_valid_ENAS(nx_to_igraph(standardize_enas(g))))
            except: # not valid
                valid.append(False)
                continue
        elif DATASET == "bn":
            valid.append(is_valid_BN(nx_to_igraph(standardize_bn(g))))
        novel.append(is_novel(g, orig_graphs))
        unique.append(is_novel(g, graphs[:i]+graphs[i+1:]))
    return {"valid": np.mean(valid), "novel": np.mean(novel), "unique": np.mean(unique), "n": len(graphs)}


def evaluate_nn(args, g, default_val=0.0):    
    # if args.dataset == "enas":
    #     from evaluation import *
    #     eva = Eval_NN()
    g_ = nx_to_igraph(g)
    if is_valid_ENAS(g_):
        arc = decode_igraph_to_ENAS(g_)
        score = eva.eval(arc)
        score = (score - mean_y_train) / std_y_train
    else:
        score = 0.0


def send_enas_listener(valid_arcs_final):
    def lock(f):
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except IOError:
            return False
        return True    
    # Prepare data
    generated_samples = [decode_igraph_to_ENAS(nx_to_igraph(g)).replace(' ',',') for g in valid_arcs_final]
    # File communication to obtain retro-synthesis rate
    sender_filename = 'enas_in.txt'
    receiver_filename = 'enas_out.txt'
    receiver_filename_suffix = 'enas_out_copy.txt'
    open(sender_filename, 'w+').close() # clear
    open(receiver_filename, 'w+').close() # clear
    while(True):
        with open(sender_filename, 'r') as fr:
            editable = lock(fr)
            if editable:
                with open(sender_filename, 'w') as fw:
                    for sample in generated_samples:
                        fw.write('{}\n'.format(sample))                
                fcntl.flock(fr, fcntl.LOCK_UN)
                break
    num_samples = len(generated_samples)
    print("Waiting for enas evaluation...")
    latest = time.time()
    while(True):
        with open(receiver_filename, 'r') as fr:
            editable = lock(fr)
            if editable:
                syn_status = []
                lines = fr.readlines()
                if len(lines) == num_samples:
                    for idx, line in enumerate(lines):
                        splitted_line = line.strip().split()
                        syn_status.append((idx, splitted_line[2]))
                    break          
            fcntl.flock(fr, fcntl.LOCK_UN)
        time.sleep(1)
    generated_samples = [generated_samples[int(idx)] for idx, _ in syn_status]
    shutil.copyfile(receiver_filename, receiver_filename_suffix) # save res
    return [float(s[1]) for s in syn_status]


def evaluate_bn(args, g):
    breakpoint()


def normalize_format(g): 
    # get longest path from input->output
    s = next(filter(lambda n: g.nodes[n]['type'] == 'input', g))
    t = next(filter(lambda n: g.nodes[n]['type'] == 'output', g))
    num_stages = 0
    main_path = [None] * (len(g)+1)
    for path in nx.all_simple_paths(g, s, t):
        count = sum(['gm' in g.nodes[n]['type'] for n in path])
        if count > num_stages:
            main_path = [None] * (len(g)+1)
        if count >= num_stages:
            num_stages = count
            if len(path) < len(main_path):
                main_path = path
    if num_stages not in [2, 3]:
        raise ValueError(f"op-amp has {num_stages} stages")
    if len(set(sum([[n]+list(g.predecessors(n)) for n in main_path], []))) < len(g):
        # another unfortunate restriction of the converter
        raise ValueError("every node needs to be on main path or flow into one")
    if len(main_path) != num_stages+2:
        # another unfortunate restriction of the converter
        raise ValueError("only gm cells allowed on main path")
    # relabel main path from 0,2,...len(main_path)-1,1
    rename = {}
    for i, n in enumerate(main_path):
        if g.nodes[n]['type'] == 'input':
            rename[n] = 0
        elif g.nodes[n]['type'] == 'output':
            rename[n] = 1
        else:
            rename[n] = 1+i
    for n in g:
        if n not in rename:
            rename[n] = len(rename)
    g = nx.relabel_nodes(g, rename)    
    return g, num_stages



def convert_ckt(g, fname):
    g, stage = normalize_format(g)
    num_subg = len(g)
    num_nodes = len(g)
    to_write = ""
    to_write+=(f"{num_subg} {num_nodes} {stage}\n")
    SUB_FEAT = {}
    subg_type = {
        "input": 0,
        "output": 1,
        "R": 2,
        "C": 3,
        "+gm+": 6,
        "-gm+": 7,
        "+gm-": 8,
        "-gm-": 9
    }
    subg_list = [subg_type[g.nodes[i]['type']] for i in range(num_subg)]
    pos_subg_dict = [i for i in range(num_subg)]
    pre_subg_dict = [list(g.predecessors(i)) for i in range(num_subg)]
    pre_subg_dict[1] = pre_subg_dict[1]
    num_edge_dict = [len(pre_subg_dict[i]) for i in range(num_subg)]
    for i in range(num_subg):
        if i == 0:
            sub_inform = [0, i, 0, 0, 0, 1, 8, 0, 1]
            sub_feats = [-1,0, -1]
        elif i == 1:
            subg_t = subg_list[1]
            pos_ = pos_subg_dict[1]
            num_edge = num_edge_dict[1]
            predecessive_ind = pre_subg_dict[1]
            sub_inform = [subg_t, 1, pos_, num_edge] + predecessive_ind + [1, 9, 0, 1]
            sub_feats = [-1,0,-1]
            SUB_FEAT[1] = sub_feats
        else:
            subg_t = subg_list[i]
            num_edge = num_edge_dict[i]
            pos_ = pos_subg_dict[i]
            predecessive_ind = pre_subg_dict[i]
            if num_edge == 0 and len(predecessive_ind) == 0:
                predecessive_ind = [0]
            sub_types = [6]
            sub_feats = [-1]
            sub_types += [NODE_TYPE[i] for i in SUBG_NODE[subg_list[i]]]
            size = len(SUBG_NODE[subg_list[i]])
            sub_feats += [g.nodes[i]['feat']]
            sub_types += [7]
            sub_feats += [-1]
            assert(len(sub_types) == len(sub_feats))
            #print(sub_types)
            #print(sub_feats)
            nodes_in_subg = len(sub_types)
            flatten_adj = [0,1,0,1,0,1,0,1,0]
            sub_inform = [subg_t, i, pos_, num_edge] + predecessive_ind + [nodes_in_subg] + sub_types + sub_feats + flatten_adj
            #print(sub_inform)
        #SUB_INF[subg_list[i]] = sub_inform 
        SUB_FEAT[i] = sub_feats
        for val in sub_inform:
            to_write+=(str(val))
            to_write+=(' ')
        to_write+=('\r\n')
    all_predecessive_dict = {}
    all_type_dict = {}
    all_feat_dict = {}
    
    # ind_order = []
    # if stage == 3:
    #     main_path = [0,2,3,4,1]
    # elif stage == 2:
    #     main_path = [0,2,3,1]
    # else:
    #     raise MyException('Undefined number of stages')
    # for i in main_path:
    #     if i == 0:
    #         ind_order.append(i)
    #     else:
    #         for j in pre_subg_dict[i]:
    #             if j not in ind_order:
    #                 ind_order.append(j)
    #         ind_order.append(i)
    ind_order = next(nx.all_topological_sorts(g))
    ind_dict = {}
    node_count = 0
    #print(ind_order)
    for i in ind_order:
        #print(i)
        #print(pre_subg_dict[i])
        num_nodes_subg = len(SUBG_NODE[subg_list[i]])
        ind_dict[i] = [node_count, node_count + num_nodes_subg - 1]
        insubg_id = 0
        #print(ind_dict)
        #print(pre_subg_dict)
        for node_id in range(node_count, node_count + num_nodes_subg):
            all_type_dict[node_id] = NODE_TYPE[SUBG_NODE[subg_list[i]][insubg_id]]
            all_feat_dict[node_id] = SUB_FEAT[i][insubg_id + 1]
            if SUBG_CON[subg_list[i]] == 'series':
                pre_nodes = []
                if insubg_id == 0:
                    for j in pre_subg_dict[i]:
                        if SUBG_CON[subg_list[j]] == 'series':
                            pre_nodes.append(ind_dict[j][1])
                        elif SUBG_CON[subg_list[j]] == 'parral':
                            for h in range(ind_dict[j][0],ind_dict[j][1]+1):
                                pre_nodes.append(h)
                        else:
                            pre_nodes.append(ind_dict[j][0])
                else:
                    pre_nodes.append(node_id-1)
                all_predecessive_dict[node_id] = pre_nodes
            elif SUBG_CON[subg_list[i]] == 'parral':
                pre_nodes = []
                for j in pre_subg_dict[i]:
                    if SUBG_CON[subg_list[j]] == 'series':
                        pre_nodes.append(ind_dict[j][1])
                    elif SUBG_CON[subg_list[j]] == 'parral':
                        for h in range(ind_dict[j][0],ind_dict[j][1]+1):
                            pre_nodes.append(h)
                    else:
                        pre_nodes.append(ind_dict[j][0])
                all_predecessive_dict[node_id] = pre_nodes
            else:
                pre_nodes = []
                for j in pre_subg_dict[i]:
                    if SUBG_CON[subg_list[j]] == 'series':
                        pre_nodes.append(ind_dict[j][1])
                    elif SUBG_CON[subg_list[j]] == 'parral':
                        for h in range(ind_dict[j][0],ind_dict[j][1]+1):
                            pre_nodes.append(h)
                    else:
                        pre_nodes.append(ind_dict[j][0])
                all_predecessive_dict[node_id] = pre_nodes
            insubg_id += 1
        node_count += num_nodes_subg
    ### [node type, node feat, previous node]    
    for i in range(num_nodes):
        type_ =  all_type_dict[i]
        feat_ = all_feat_dict[i]
        predecessors_ = all_predecessive_dict[i]
        inform = [type_, feat_] + predecessors_
        for val in inform:
            to_write+=(str(val))
            to_write+=(' ')
        to_write+=('\r\n')
    with open(fname, 'w+') as f:
        f.write(to_write)


def evaluate_ckt(args, g):    
    folder = args.datapkl if args.datapkl else args.folder
    path = os.path.join(f'{args.cache_root}/cache/api_{args.dataset}_ednce/{folder}')
    # write converter:
    fname = os.path.join(path, f"{hash_object(g)}.txt")
    convert_ckt(g, fname)    
    fom = cktgraph_to_fom(fname)
    os.remove(fname)
    return fom


def main_sgp(args):
    """
    loads checkpoints and latent data for three encoders (GNN, TOKEN, TOKEN_GNN),
    assigns args.datapkl path for each encoder to load data.pkl,
    runs train_sgp three times per encoder, and prints mean and std for RMSE and R^2.
    """

    ckpt_paths = {
        "GNN": f"{args.cache_root}/cache/models/CKT/GNN/epoch-33_loss-3.8809502124786377.pth",
        "TOKEN": f"{args.cache_root}/cache/models/CKT/TOKEN/epoch-35_loss-3.8443312644958496.pth",
        "TOKEN_GNN": f"{args.cache_root}/cache/models/BN/TOKEN_GNN/epoch=0_loss=6.534396348571778.pth"
    }

    latent_paths = {
        "GNN": {
            "train_latent": f"{args.cache_root}/cache/models/CKT/GNN/train_latent_33.npy",
            "test_latent": f"{args.cache_root}/cache/models/CKT/GNN/test_latent_33.npy"
        },
        "TOKEN": {
            "train_latent": f"{args.cache_root}/cache/models/CKT/TOKEN/train_latent_35.npy",
            "test_latent": f"{args.cache_root}/cache/models/CKT/TOKEN/test_latent_35.npy"
        },
        "TOKEN_GNN": {
            "train_latent": f"{args.cache_root}/cache/models/CKT/TOKEN_GNN/train_latent_8.npy",
            "test_latent": f"{args.cache_root}/cache/models/CKT/TOKEN_GNN/test_latent_8.npy"
        }
    }

    datapkl_paths = {
        "GNN": f"{args.cache_root}/cache/models/CKT/GNN/data.pkl",         
        "TOKEN": f"{args.cache_root}/cache/models/CKT/TOKEN/data.pkl",     
        "TOKEN_GNN": f"{args.cache_root}/cache/models/CKT/TOKEN_GNN/data.pkl"  
    }

    results = {}
    encoder_name = args.encoder
    ckpt_path = ckpt_paths[encoder_name]
   
    # for encoder_name, ckpt_path in ckpt_paths.items():
    print(f"ENCODER: {encoder_name}")
    # use unique data.pkl folder for current encoder
    args.datapkl = datapkl_paths[encoder_name]
    print(args.datapkl)

    cache_dir = f'{args.cache_root}/cache/api_{args.dataset}_ednce/'
    folder = hash_args(args)
    print(f"folder: {folder}")
    setattr(args, "folder", folder)
    #os.makedirs(f'ckpts/api_{args.dataset}_ednce/{folder}', exist_ok=True)
    #os.makedirs(f'cache/api_{args.dataset}_ednce/{folder}', exist_ok=True)
    version = get_next_version(cache_dir) - 1
    logger.info(f"loading version {version}")
    grammar, anno, g = pickle.load(open(os.path.join(cache_dir, f'{version}.pkl'), 'rb'))
    if args.dataset == "ckt":
        num_graphs = 10000
        orig = load_ckt(args, load_all=True)
    elif args.dataset == "bn":
        num_graphs = 200000
        orig = load_bn(args)        
    elif args.dataset == "enas":
        num_graphs = 19020
        orig = load_enas(args)
    else:
        raise NotImplementedError
    train_data, test_data, token2rule = load_data(args, anno, grammar, orig, cache_dir, num_graphs)
    print(f"Processing encoder: {encoder_name}")

    #args.encoder = encoder_name
    model = TransformerVAE(args.encoder, args.encoder_layers, args.decoder_layers, VOCAB_SIZE, vocabulary_init, vocabulary_terminate, args.embed_dim, args.latent_dim, MAX_SEQ_LEN, args.cuda)    
    model = model.to(args.cuda)
    
    print("loading model")
    #checkpoint = torch.load(ckpt_path, map_location=args.cuda)
    model.load_state_dict(torch.load(ckpt_path, map_location=args.cuda))
    

    X_train = np.load(latent_paths[encoder_name]["train_latent"])
    X_test = np.load(latent_paths[encoder_name]["test_latent"])

    indices = list(range(num_graphs))
    train_indices = indices[:int(num_graphs * 0.9)]
    test_indices = indices[int(num_graphs * 0.9):]
    y = load_y(orig, num_graphs, target={"ckt": ["gain", "bw", "pm", "fom"],
                                            "bn": ["bic"],
                                            "enas": ["acc"]}[args.dataset])
    y = np.array(y)
    train_y = y[train_indices]
    mean_train_y = np.mean(train_y, axis=0)
    std_train_y = np.std(train_y, axis=0)
    test_y = y[test_indices]
    train_y = (train_y - mean_train_y) / std_train_y
    test_y = (test_y - mean_train_y) / std_train_y

    if args.dataset == "bn":
        X_train, unique_idxs = np.unique(X_train, axis=0, return_index=True)
        y_train = train_y[unique_idxs]
        random_shuffle = np.random.permutation(range(len(X_train)))
        keep = 5000
        X_train = X_train[random_shuffle[:keep]]
        y_train = y_train[random_shuffle[:keep]]
    else:
        y_train = train_y
        keep = y_train.shape[0]

    X_train_mean = X_train.mean(axis=0)
    X_train_std = X_train.std(axis=0)
    X_train = (X_train - X_train_mean) / X_train_std
    X_test = (X_test - X_train_mean) / X_train_std
    logger.info("Average pairwise distance (train) = {}".format(np.mean(pdist(X_train))))
    logger.info("Average pairwise distance (test) = {}".format(np.mean(pdist(X_test))))
    y_train, y_test = y_train[:, -1:], test_y[:, -1:]
    print(X_train.shape)


    # run train_sgp three times, track RMSE and R^2
    rmse_list = []
    r2_list = []

    for run_i in range(3):
        logger.info(f"Run {run_i + 1} for {encoder_name}")
        save_file = f"{encoder_name}_sgp_run{run_i + 1}.txt"
        sgp_model = train_sgp(args, save_file, X_train, X_test, y_train, y_test)
        print("train sgp completed")
        pred, _ = sgp_model.predict(X_test, 0 * X_test)
        rmse = np.sqrt(np.mean((pred - y_test)**2))
        r2 = r2_score(y_test, pred)
        rmse_list.append(rmse)
        r2_list.append(r2)

    rmse_mean, rmse_std = np.mean(rmse_list), np.std(rmse_list)
    r2_mean, r2_std = np.mean(r2_list), np.std(r2_list)
    results[encoder_name] = {
        "rmse_mean": rmse_mean,
        "rmse_std": rmse_std,
        "r2_mean": r2_mean,
        "r2_std": r2_std
    }

    for enc, vals in results.items():
        print(f"\nEncoder: {enc}")
        print(f"RMSE: {vals['rmse_mean']:.4f} ± {vals['rmse_std']:.4f}")
        print(f"R^2: {vals['r2_mean']:.4f} ± {vals['r2_std']:.4f}")
    logger.info("Done with main_sgp.")


def main(args):
    cache_dir = f'{args.cache_root}/{CACHE_DIR}'
    folder = hash_args(args)
    ckpt_dir = f'{args.ckpt_dir}/ckpts/{Path(CACHE_DIR).stem}/{folder}'
    print(f"folder: {folder}")
    setattr(args, "folder", folder)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(f'{args.cache_root}/{CACHE_DIR}{folder}', exist_ok=True)
    #json.dumps(args.__dict__, folder)
    args_path = os.path.join(ckpt_dir, "args.txt")
    with open(args_path, "w") as f:
        for arg_name, arg_value in sorted(args.__dict__.items()):
            f.write(f"{arg_name}: {arg_value}\n")
    if args.repr == "digged":
        version = get_next_version(cache_dir)-1    
        logger.info(f"loading version {version}")
        grammar, anno, g = pickle.load(open(os.path.join(cache_dir, f'{version}.pkl'),'rb'))
    if args.dataset == "ckt":
        num_graphs = 10000
        orig = load_ckt(args, load_all=True)        
        graph_args = argparse.Namespace()
        graph_args.num_vertex_type = len(LOOKUP)
        graph_args.max_n = max(map(len, orig.comps.values()))
        graph_args.START_TYPE = list(LOOKUP).index('input')
        graph_args.END_TYPE = list(LOOKUP).index('output')
    elif args.dataset == "bn":        
        num_graphs = 200000
        orig, graph_args = load_bn(args)
        num_graphs = len(orig.comps)
    elif args.dataset == "enas":   
        num_graphs = 19020
        orig, graph_args = load_enas(args)
    else:
        raise NotImplementedError            
    if args.repr == "digged":
        train_data, test_data, token2rule = load_data(args, anno, grammar, orig, cache_dir, num_graphs, graph_args)
    else:
        train_data, test_data = load_data(args, None, None, orig, cache_dir, num_graphs, graph_args)
    if args.datapkl:
        print(f'The folder being written to is: {args.datapkl}')
    else:
        print(f'The folder being written to is: {folder}')        
    model = train(args, train_data, test_data)
    # prepare y
    # TODO: remove this later
    indices = list(range(num_graphs))
    # random.Random(0).shuffle(indices)
    train_indices, test_indices = indices[:int(num_graphs*0.9)], indices[int(num_graphs*0.9):]
    y = load_y(orig, num_graphs, target={"ckt": ["gain", "bw", "pm", "fom"],"bn": ["bic"],"enas": ["acc"]}[args.dataset])
    y = np.array(y)
    train_y = y[train_indices]
    mean_train_y = np.mean(train_y, axis=0)
    std_train_y = np.std(train_y, axis=0)
    test_y = y[test_indices]
    train_y = (train_y-mean_train_y)/std_train_y
    test_y = (test_y-mean_train_y)/std_train_y
    if args.repr == "digged":
        bo(args, orig, grammar, model, token2rule, train_y, test_y, mean_train_y[-1], std_train_y[-1])
    else:
        bo_ns(args, model, train_y, test_y, mean_train_y[-1], std_train_y[-1])
    if args.repr == "digged":
        graphs = interactive_sample_sequences(args, model, grammar, token2rule,max_seq_len=MAX_SEQ_LEN, unique=False, visualize=False)
    else:
        graphs = ns_sample_sequences(args, model, max_seq_len=MAX_SEQ_LEN, unique=False, visualize=True)
    metrics = evaluate(orig, graphs)
    print(metrics)


if __name__ == "__main__":
    from src.grammar.common import get_parser
    parser = get_parser()
    # folder
    parser.add_argument("--ckpt_dir", default="/ssd/msun415/induction")
    # data hparams
    parser.add_argument("--dataset", choices=["ckt", "bn", "enas"], default="ckt")
    parser.add_argument("--sample-batch-size", type=int, default=10)
    # repr
    parser.add_argument("--repr", choices=["digged", "ns"], default="digged", help="digged or node string (ns)")
    parser.add_argument("--order", choices=["default", "bfs", "random"], default="default", help="for ns")
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
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--max-ei-iter", type=int, default=150)
    parser.add_argument("--factr", type=float, default=1e7)
    parser.add_argument('--BO-rounds', type=int, default=10, help="how many rounds of BO to perform")    
    # parser.add_argument('--bo',type=int, default=0, choices=[0, 1], help='whether to do BO')
    parser.add_argument('--predictor', action='store_true', default=False, help='if True, use the performance predictor instead of SGP')    
    parser.add_argument('--BO-batch-size', type=int, default=50, 
                        help="how many data points to select in each BO round")    
    parser.add_argument('--sample-dist', default='uniform', 
                        help='from which distrbiution to sample random points in the latent \
                        space as candidates to select; uniform or normal')       
    args = parser.parse_args()  
    breakpoint()
    main(args)
