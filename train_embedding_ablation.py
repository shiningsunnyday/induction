import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import networkx as nx
import numpy as np
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data as GraphData
import torch.nn.functional as F
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
#from src.config import *
from src.grammar.utils import get_next_version
#from src.draw.graph import draw_graph
import glob
import re
import os

# Hyperparameters
VOCAB_SIZE = 204  # Number of rules
EMBED_DIM = 12
LATENT_DIM = 32
SEQ_LEN = 13  # Max length of sequences
BATCH_SIZE = 256
EPOCHS = 30

# Convert NetworkX graphs to PyTorch Geometric Data objects
def convert_graph_to_data(graph):
    # Random node features
    graph = nx.relabel_nodes(graph, dict(zip(graph,range(len(graph)))))
    one_hot_vector = torch.zeros((len(TERMS)+len(NONTERMS),))
    features = []
    term = True
    for i in range(len(graph)):
        index = (TERMS+NONTERMS).index(graph.nodes[i]['label'])
        one_hot_vector[index] = 1.
        if index >= len(TERMS):
            term = False # nonterm node
        features.append(one_hot_vector)
    x = torch.stack(features, dim=0)
    # x = torch.rand((graph.number_of_nodes(), EMBED_DIM))    
    edge_index = torch.tensor(list(graph.edges)).t().contiguous()
    res = GraphData(x=x, edge_index=edge_index)
    return res, term

# Prepare the graph vocabulary
graph_vocabulary = None # generate_random_graphs(VOCAB_SIZE)
graph_data_vocabulary = None
vocabulary_terminate = None

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
        self.token_embedding = nn.Embedding(VOCAB_SIZE, embed_dim)
        self.transformer_encoder = TransformerEncoder(embed_dim)
        self.fc_mu = nn.Linear(embed_dim, latent_dim)
        self.fc_logvar = nn.Linear(embed_dim, latent_dim)
        
        self.fc_decode = nn.Linear(latent_dim, latent_dim)
        self.transformer_decoder = TransformerEncoder(embed_dim + latent_dim)
        self.output_layer = nn.Linear(embed_dim + latent_dim, VOCAB_SIZE)
        self.start_token_embedding = nn.Parameter(torch.randn(1, 1, embed_dim))

    def embed_tokens(self, token_ids):
        return self.token_embedding(token_ids) # check shape of these

    def encode(self, x, attention_mask):
        embedded_tokens = self.embed_tokens(x)  # Embeds each token (graph) using GNN
        encoded_seq = self.transformer_encoder(embedded_tokens.permute(1, 0, 2), src_key_padding_mask=~attention_mask).permute(1, 0, 2)
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
                    next_token_id = x[idx, t].unsqueeze(0)  # Ground truth for next token
                    #next_token_embedding = self.gnn(graph_data_vocabulary[next_token_id.item()]).unsqueeze(0)
                    next_token_embedding = self.token_embedding(next_token_id)

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
                #next_token_embedding = self.gnn(graph_data_vocabulary[next_tokens[i].item()]).unsqueeze(0)
                next_token_embedding = self.token_embedding(next_tokens[i])
                # Append the new embedding to this sequence’s token embeddings
                token_embeddings[idx] = torch.cat((token_embeddings[idx], next_token_embedding), dim=0)
        return generated_sequences


    def forward(self, x, attention_mask, seq_len_list):
        mu, logvar = self.encode(x, attention_mask)
        z = self.reparameterize(mu, logvar)
        logits, logits_mask = self.autoregressive_training_step(x, z, seq_len_list)
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
    lengths = [len(seq) for seq in batch]
    max_len = max(lengths)
    padded_batch = torch.zeros(len(batch), max_len, dtype=torch.long)
    seq_len_list = torch.zeros(len(batch), dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    
    for i, seq in enumerate(batch):
        padded_batch[i, :len(seq)] = seq
        seq_len_list[i] = len(seq)  # Mask non-padded positions
        attention_mask[i, :len(seq)] = 1
    
    return padded_batch, attention_mask, seq_len_list


# Sampling new sequences
def sample(model, num_samples=5, max_seq_len=10):
    model.eval()
    uniq_sequences = set()
    with torch.no_grad():
        while len(uniq_sequences) < num_samples:
            z = torch.randn(num_samples, LATENT_DIM)  # Sample from the prior
            z = z.to('cuda')
            generated_sequences = model.autoregressive_inference(z, max_seq_len)  # Decode from latent space            
            uniq_sequences = uniq_sequences | set(map(tuple, generated_sequences))
    uniq_sequences = [list(l) for l in uniq_sequences]
    return uniq_sequences[:num_samples]


def train(data, device):
    # Initialize model and optimizer
    model = TransformerVAE(EMBED_DIM, LATENT_DIM, SEQ_LEN)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Dummy dataset: Replace with actual sequence data
    dataset = [torch.tensor(seq) for seq in data]
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

    # Training loop
    model.train()
    model = model.to(device)
    # for i, graph_data in enumerate(graph_data_vocabulary):
    #     graph_data_vocabulary[i] = graph_data.to(device)
    ckpts = glob.glob('ckpts/*.pth')
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
    
    for epoch in tqdm(range(start_epoch, EPOCHS)):
        for batch in dataloader:
            x, attention_mask, seq_len_list = batch            
            x, attention_mask = x.to(device), attention_mask.to(device)
            optimizer.zero_grad()
            recon_logits, mask, mu, logvar = model(x, attention_mask, seq_len_list)
            loss = vae_loss(recon_logits, mask, x, mu, logvar)
            loss.backward()
            optimizer.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            ckpt_path = f'ckpts/epoch={epoch}_loss={best_loss}.pth'
            #torch.save(model.state_dict(), ckpt_path)
            print(ckpt_path)            
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    return model


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


def interactive_mask_logits(grammar, generated_sequences, logits):
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
        while len(uniq_sequences) < num_samples:
            z = torch.randn(num_samples, LATENT_DIM)  
            z = z.to('cuda')            
            batch_size = z.size(0)
            z_context = model.fc_decode(z)              
            token_embeddings = list(torch.unbind(model.start_token_embedding.expand(batch_size, -1, -1), dim=0))  
            generated_sequences = [[] for _ in range(batch_size)]  
            for t in range(max_seq_len):
                temp_batch_embeddings, temp_z_context, active_indices = model._autoregressive_inference_active_indices(z_context, generated_sequences, token_embeddings, max_seq_len)                
                if not active_indices:
                    break
                logits = model._autoregressive_inference_predict_logits(temp_batch_embeddings, temp_z_context)
                interactive_mask_logits(grammar, generated_sequences, logits)
                next_tokens = torch.argmax(logits, dim=-1)
                for i, idx in enumerate(active_indices):
                    generated_sequences[idx].append(next_tokens[i].item())  
                    next_token_embedding = model.gnn(graph_data_vocabulary[next_tokens[i].item()]).unsqueeze(0)
                    token_embeddings[idx] = torch.cat((token_embeddings[idx], next_token_embedding), dim=0)
            uniq_sequences = uniq_sequences | set(map(tuple, generated_sequences))
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
    


def load_data(cache_dir, num_graphs):
    exist = os.path.exists(os.path.join(cache_dir, 'data_and_rule2token.pkl'))
    print(exist)
    if exist:
        data, rule2token = pickle.load(open(os.path.join(cache_dir, 'data_and_rule2token.pkl'), 'rb'))
    else:
        data = []
        rule2token = {}
        for pre in tqdm(range(num_graphs), "processing data"): # can change 4500 to debug
            prefix = f"{pre}:"
            seq = list(filter(lambda k: k[:len(prefix)]==f'{prefix}', anno))
            seq = seq[::-1] # derivation
            rule_ids = [anno[s].attrs['rule'] for s in seq]        
            for r in rule_ids:
                rule2token[r] = grammar.rules[r].subgraph
            data.append(rule_ids)
        pickle.dump((data, rule2token), open(os.path.join(cache_dir, 'data_and_rule2token.pkl'), 'wb+'))
    relabel = dict(zip(list(sorted(rule2token)), range(len(rule2token))))
    data = [[relabel[s] for s in seq] for seq in data]
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
    return data, token2rule

def generate_random_sequences(num_batches=3, batch_size=2, seq_len=8, vocab_size=60):
    data_list = torch.tensor([])
    for _ in range(num_batches):
        batch_data = torch.randint(low=1, high=vocab_size+1, size=(batch_size, seq_len))
        data_list = torch.cat((data_list, batch_data), 0)
    return data_list
    

if __name__ == "__main__":
    #cache_dir = 'cache/api_ckt_ednce/'
    #version = get_next_version(cache_dir)-1
    #print(version)
    #grammar, anno, g = pickle.load(open(os.path.join(cache_dir, f'{version}.pkl'),'rb'))
    #data, token2rule = load_data(cache_dir, 4500)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = generate_random_sequences()
    print(data)
    model = train(data, device)
    #interactive_sample_sequences(model, grammar, token2rule, max_seq_len=10)
