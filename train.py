import os
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
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
import torch.nn.functional as F
torch.multiprocessing.set_sharing_strategy('file_system')
import pickle
import hashlib
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.model import *
import glob
import re

import sys
sys.path.append('dagnn/dvae/bayesian_optimization')
from sparse_gp import SparseGP
from utils import is_valid_DAG, is_valid_Circuit
from OCB.src.simulator.graph_to_fom import cktgraph_to_fom
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
            generated_sequences = model.autoregressive_inference(z, token2rule, max_seq_len)  # Decode from latent space            
            uniq_sequences = uniq_sequences | set(map(tuple, generated_sequences))
    uniq_sequences = [list(l) for l in uniq_sequences]
    return uniq_sequences[:num_samples]


def decode_from_latent_space(z, grammar, model, token2rule, max_seq_len):
    # generated_sequences = model.autoregressive_interactive_inference(z, max_seq_len)
    generated_dags = [None for _ in range(z.shape[0])]
    idxes = list(range(z.shape[0]))
    with tqdm(total=z.shape[0], desc="decoding") as pbar:
        while idxes:
            generated_sequences = model.autoregressive_interactive_inference(z, grammar, token2rule, max_seq_len, decode='softmax')     
            new_idxes = []
            mask = []
            for idx, deriv in zip(idxes, generated_sequences):
                g = grammar.derive(deriv, token2rule)
                for n in g:
                    g.nodes[n]['type'] = INVERSE_LOOKUP[g.nodes[n]['label']]
                try: # not our fault, but due to the converter assuming 2 or 3-stage op-amps, we'll keep sampling until we satisfy that restriction
                    normalize_format(g)
                    generated_dags[idx] = g
                    mask.append(False)
                except ValueError:
                    new_idxes.append(idx)
                    mask.append(True)        
            idxes = new_idxes
            z = z[mask]
            pbar.update(len(idxes)-pbar.n)
    return generated_dags
    

def train(args, train_data, test_data):
    print(args.folder)
    # Initialize model and optimizer
    model = TransformerVAE(args.encoder, args.encoder_layers, args.decoder_layers, VOCAB_SIZE, vocabulary_init, vocabulary_terminate, args.embed_dim, args.latent_dim, MAX_SEQ_LEN)
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
    ckpts = glob.glob(f'ckpts/api_ckt_ednce/{args.folder}/*.pth')
    logger.info(f'ckpts/api_ckt_ednce/{args.folder}/*.pth')
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

    patience = 10
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

def bo(args, grammar, model, token2rule, y_train, y_test):
    folder = args.datapkl if args.datapkl else args.folder
    ckpt_dir = f'ckpts/api_ckt_ednce/{folder}'    
    X_train = np.load(os.path.join(ckpt_dir, f"train_latent_{args.checkpoint}.npy"))    
    X_test = np.load(os.path.join(ckpt_dir, f"test_latent_{args.checkpoint}.npy"))    
    X_train_mean = X_train.mean(axis=0)
    X_train_std = X_train.std(axis=0)
    X_train = (X_train-X_train_mean)/X_train_std
    X_test = (X_test-X_train_mean)/X_train_std    
    iteration = 0
    best_score = 1e15
    best_arc = None
    best_random_score = 1e15
    best_random_arc = None
    logger.info("Average pairwise distance between train points = {}".format(np.mean(pdist(X_train))))
    logger.info("Average pairwise distance between test points = {}".format(np.mean(pdist(X_test))))    
    if DATASET == "ckt":
        evaluate_fn = evaluate_ckt
    elif DATASET == "enas":
        evaluate_fn = evaluate_nn
    elif DATASET == "bn":
        evaluate_fn = evaluate_bn
    else:
        raise NotImplementedError
    while iteration < args.BO_rounds:
        logger.info(f"Iteration: {iteration}")
        if args.predictor:
            pred = model.predictor(torch.FloatTensor(X_test).to(args.cuda))
            pred = pred.detach().cpu().numpy()
            pred = (-pred - mean_y_train) / std_y_train
            uncert = np.zeros_like(pred)
        else:
            # We fit the GP
            M = 500
            # other BO hyperparameters
            lr = 0.005  # the learning rate to train the SGP model
            max_iter = args.max_iter  # how many iterations to optimize the SGP each time
            sgp = SparseGP(X_train, 0 * X_train, y_train, M)
            sgp.train_via_ADAM(X_train, 0 * X_train, y_train, X_test, X_test * 0, y_test, minibatch_size = 2 * M, max_iterations = max_iter, learning_rate = lr)
            pred, uncert = sgp.predict(X_test, 0 * X_test)
            # input_means = X_train
            # input_vars = np.zeros_like(X_train)  # Variances initialized to 0
            # training_targets = y_train
            # n_inducing_points = M
            # # Define kernel
            # kernel = gpflow.kernels.RBF()
            # # Initialize inducing points (e.g., random subset of training data)
            # inducing_points = input_means[:n_inducing_points, :]
            # # Create sparse variational GP model
            # sgp = SVGP(kernel=kernel,
            #         likelihood=gpflow.likelihoods.Gaussian(),
            #         inducing_variable=inducing_points,
            #         num_latent_gps=1)
            # train_sgp(sgp, input_means, training_targets, batch_size=2*M)
            # pred, var = sgp.predict_y(X_test)

        logger.info(f"predictions: {pred.reshape(-1)}")
        logger.info(f"real values: {y_test.reshape(-1)}")
        error = np.sqrt(np.mean((pred - y_test)**2))
        testll = np.mean(sps.norm.logpdf(pred - y_test, scale = np.sqrt(uncert)))
        logger.info(f'Test RMSE: {error}')
        logger.info(f'Test ll: {testll}')
        pearson = float(pearsonr(pred.flatten(), y_test.flatten())[0])
        logger.info(f'Pearson r: {pearson}')
        with open('results/' + 'Test_RMSE_ll.txt', 'a') as test_file:
            test_file.write('Test RMSE: {:.4f}, ll: {:.4f}, Pearson r: {:.4f}\n'.format(error, testll, pearson))

        error_if_predict_mean = np.sqrt(np.mean((np.mean(y_train, 0) - y_test)**2))
        logger.info(f'Test RMSE if predict mean: {error_if_predict_mean}')
        if args.predictor:
            pred = model.predictor(torch.FloatTensor(X_train).to(args.cuda))
            pred = pred.detach().cpu().numpy()
            pred = (-pred - mean_y_train) / std_y_train
            uncert = np.zeros_like(pred)
        else:
            pred, uncert = sgp.predict(X_train, 0 * X_train)
        error = np.sqrt(np.mean((pred - y_train)**2))
        trainll = np.mean(sps.norm.logpdf(pred - y_train, scale = np.sqrt(uncert)))
        logger.info(f'Train RMSE: {error}')
        logger.info(f'Train ll: {trainll}')        
        next_inputs = sgp.batched_greedy_ei(args.BO_batch_size, np.min(X_train, 0), np.max(X_train, 0), np.mean(X_train, 0), np.std(X_train, 0), sample=args.sample_dist, max_iter=args.max_ei_iter)
        #breakpoint()
        valid_arcs_final = decode_from_latent_space(torch.FloatTensor(next_inputs).to(args.cuda), grammar, model, token2rule, MAX_SEQ_LEN)
        new_features = next_inputs
        logger.info("Evaluating selected points")
        scores = []        
        for i in range(len(valid_arcs_final)):
            score = evaluate_fn(args, valid_arcs_final[i])
            if score < best_score:
                best_score = score
                best_arc = arc
            scores.append(score)
            # logger.info(i, score)
        # logger.info("Iteration {}'s selected arcs' scores:".format(iteration))
        # logger.info(scores, np.mean(scores))
        save_object(scores, "{}scores{}.dat".format(save_dir, iteration))
        save_object(valid_arcs_final, "{}valid_arcs_final{}.dat".format(save_dir, iteration))

        if len(new_features) > 0:
            X_train = np.concatenate([ X_train, new_features ], 0)
            y_train = np.concatenate([ y_train, np.array(scores)[ :, None ] ], 0)
        #
        # logger.info("Current iteration {}'s best score: {}".format(iteration, - best_score * std_y_train - mean_y_train))
        if best_arc is not None: # and iteration == 10:
            logger.info(f"Best architecture: {best_arc}")
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
                generated_sequences = model.autoregressive_interactive_inference(z, grammar, token2rule, max_seq_len=max_seq_len)    
                if unique:
                    uniq_sequences = uniq_sequences | set(map(tuple, generated_sequences))                
                    pbar.update(len(uniq_sequences)-pbar.n)
                else:
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
        else:
            g = grammar.derive(seq, token2rule)
        gs.append(g)
    return gs
    

def load_y(g, num_graphs, target):
    y = []
    for pre in range(num_graphs):        
        y.append(g.graph[f'{pre}:{target}'])
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
        if args.dataset == "ckt":
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

def is_novel(g, orig_graphs):
    for o in orig_graphs:
        if nx.is_isomorphic(g, o, node_match=node_match):
            return False
    return True


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
                        if g.nodes[v_cand]['type'] in ['R','C'] and len(inster_set):
                            good = True
            if not good:
                return False
    return good              


def evaluate(orig_graphs, graphs):
    ckt_valid, dag_valid, novel = [], [], []
    for g in graphs:
        if not is_valid_circuit(g):
            print("not valid circuit")
            #breakpoint()
        is_valid_ckt = is_valid_circuit(g)
        ckt_valid.append(is_valid_ckt)
        dag_valid.append(is_valid_DAG(nx_to_igraph(g, subg=False)))
        novel.append(is_novel(g, orig_graphs))
    return {"valid_dag": np.mean(dag_valid), "valid_ckt": np.mean(ckt_valid), "novel": np.mean(novel), "n": len(graphs)}


def evaluate_nn(args, g):
    if arc is not None:
        score = -eva.eval(arc)
        score = (score - mean_y_train) / std_y_train
    else:
        score = max(y_train)[ 0 ]    


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
    with open(fname, 'w+') as f:
        f.write(f"{num_subg} {num_nodes} {stage}\n")
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
        pre_subg_dict[1] = [1] + pre_subg_dict[1]
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
                f.write(str(val))
                f.write(' ')
            f.write('\r\n')
        all_predecessive_dict = {}
        all_type_dict = {}
        all_feat_dict = {}
        
        ind_order = []
        if stage == 3:
            main_path = [0,2,3,4,1]
        elif stage == 2:
            main_path = [0,2,3,1]
        else:
            raise MyException('Undefined number of stages')
        for i in main_path:
            if i == 0:
                ind_order.append(i)
            else:
                for j in pre_subg_dict[i]:
                    if j not in ind_order:
                        ind_order.append(j)
                ind_order.append(i)
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
                f.write(str(val))
                f.write(' ')
            f.write('\r\n')            



def evaluate_ckt(args, g):    
    folder = args.datapkl if args.datapkl else args.folder
    path = os.path.join(f'cache/api_ckt_ednce/{folder}')
    # write converter:
    fname = os.path.join(path, f"{hash_object(g)}.txt")
    convert_ckt(g, fname)
    # for n in g:
        # if g.nodes[n]


def main(args):
    cache_dir = f'cache/api_{args.dataset}_ednce/'
    folder = hash_args(args)
    setattr(args, "folder", folder)
    os.makedirs(f'ckpts/api_{args.dataset}_ednce/{folder}', exist_ok=True)
    os.makedirs(f'cache/api_{args.dataset}_ednce/{folder}', exist_ok=True)
    #json.dumps(args.__dict__, folder)
    args_path = os.path.join(f'ckpts/api_{args.dataset}_ednce/{folder}', "args.txt")
    with open(args_path, "w") as f:
        for arg_name, arg_value in sorted(args.__dict__.items()):
            f.write(f"{arg_name}: {arg_value}\n")
    version = get_next_version(cache_dir)-1    
    logger.info(f"loading version {version}")
    grammar, anno, g = pickle.load(open(os.path.join(cache_dir, f'{version}.pkl'),'rb'))
    if args.dataset == "ckt":
        num_graphs = 10000
        orig = load_ckt(args, load_all=True)
    elif args.dataset == "bn":        
        num_graphs = 200000
        orig = load_bn(args)
    elif args.dataset == "enas":
        #breakpoint()
        num_graphs = None
        orig = load_bn(args)
    else:
        raise NotImplementeredError        
    train_data, test_data, token2rule = load_data(args, anno, grammar, orig, cache_dir, num_graphs)
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
    y = load_y(orig, num_graphs, target={"ckt":"fom", "bn":"bic", "enas":"acc"}[args.dataset])
    y = np.array(y)
    train_y = y[train_indices, None]
    mean_train_y = np.mean(train_y)
    std_train_y = np.std(train_y)    
    test_y = y[test_indices, None]
    train_y = (train_y-mean_train_y)/std_train_y
    test_y = (test_y-mean_train_y)/std_train_y        
    bo(args, grammar, model, token2rule, train_y, test_y)
    graphs = interactive_sample_sequences(args, model, grammar, token2rule, max_seq_len=MAX_SEQ_LEN, unique=False, visualize=False)    
    orig_graphs = [nx.induced_subgraph(orig, orig.comps[i]) for i in range(num_graphs)]
    metrics = evaluate(orig_graphs, graphs)
    print(metrics)




if __name__ == "__main__":
    from src.grammar.common import get_parser
    parser = get_parser()
    # data hparams
    parser.add_argument("--dataset", choices=["ckt", "bn", "enas"], default="ckt")
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
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--max-ei-iter", type=int, default=150)
    parser.add_argument('--BO-rounds', type=int, default=10, help="how many rounds of BO to perform")    
    # parser.add_argument('--bo',type=int, default=0, choices=[0, 1], help='whether to do BO')
    parser.add_argument('--predictor', action='store_true', default=False, help='if True, use the performance predictor instead of SGP')    
    parser.add_argument('--BO-batch-size', type=int, default=50, 
                        help="how many data points to select in each BO round")    
    parser.add_argument('--sample-dist', default='uniform', 
                        help='from which distrbiution to sample random points in the latent \
                        space as candidates to select; uniform or normal')       
    args = parser.parse_args()        
    #breakpoint()
    main(args)
