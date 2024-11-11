import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import networkx as nx
import numpy as np
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data as GraphData

# Hyperparameters
VOCAB_SIZE = 50  # Number of graph tokens
EMBED_DIM = 64
LATENT_DIM = 32
SEQ_LEN = 10  # Max length of sequences
BATCH_SIZE = 16
EPOCHS = 10

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

# Convert NetworkX graphs to PyTorch Geometric Data objects
def convert_graph_to_data(graph, embed_dim):
    # Random node features
    x = torch.rand((graph.number_of_nodes(), embed_dim))    
    edge_index = torch.tensor(list(graph.edges)).t().contiguous()
    try:
        res = GraphData(x=x, edge_index=edge_index)
    except:
        breakpoint()
    return res

# Prepare the graph vocabulary
graph_vocabulary = generate_random_graphs(VOCAB_SIZE)
graph_data_vocabulary = [convert_graph_to_data(g, EMBED_DIM) for g in graph_vocabulary]

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

    def forward(self, x, mask=None):
        return self.encoder(x, mask=mask)

# Define VAE
class TransformerVAE(nn.Module):
    def __init__(self, embed_dim, latent_dim, seq_len):
        super(TransformerVAE, self).__init__()
        self.gnn = TokenGNN(embed_dim)
        self.transformer_encoder = TransformerEncoder(embed_dim)
        self.fc_mu = nn.Linear(embed_dim * seq_len, latent_dim)
        self.fc_logvar = nn.Linear(embed_dim * seq_len, latent_dim)
        
        self.fc_decode = nn.Linear(latent_dim, embed_dim)
        self.transformer_decoder = TransformerEncoder(embed_dim)
        self.output_layer = nn.Linear(embed_dim, VOCAB_SIZE)

    def embed_tokens(self, token_ids):
        if token_ids.dim() == 2:
            return torch.stack([self.embed_tokens(token_id_seq) for token_id_seq in token_ids], dim=0)
        embedded_tokens = []
        for token_id in token_ids:
            graph_data = graph_data_vocabulary[token_id]
            embedded_token = self.gnn(graph_data)
            embedded_tokens.append(embedded_token)
        return torch.stack(embedded_tokens, dim=0)

    def encode(self, x):
        embedded_tokens = self.embed_tokens(x)  # Embeds each token (graph) using GNN
        encoded_seq = self.transformer_encoder(embedded_tokens.permute(1, 0, 2)).permute(1, 0, 2)
        flattened = encoded_seq.reshape(x.size(0), -1)
        mu, logvar = self.fc_mu(flattened), self.fc_logvar(flattened)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, max_seq_len=SEQ_LEN):
        batch_size = z.size(0)
        device = z.device

        # Initialize the first hidden state from latent vector z
        hidden_state = self.fc_decode(z).view(batch_size, 1, EMBED_DIM)  # [B, 1, EMBED_DIM]
        logits_list = []  # To collect logits at each step
        
        # Generate logits for the start token explicitly
        start_logits = self.output_layer(hidden_state[:, -1, :])  # [B, VOCAB_SIZE]
        logits_list.append(start_logits.unsqueeze(1))  # Add start token logits
        
        # Generate sequence autoregressively
        for t in range(1, max_seq_len):
            # Create a causal mask to prevent attending to future tokens
            causal_mask = torch.triu(torch.ones((t, t), device=device), diagonal=1).bool()            
            # Pass the current hidden state through the transformer decoder
            decoder_output = self.transformer_decoder(hidden_state.permute(1, 0, 2), mask=causal_mask).permute(1, 0, 2)
            
            # Get logits for the next token
            logits = self.output_layer(decoder_output[:, -1, :])  # [B, VOCAB_SIZE] for the last token position

            logits_list.append(logits.unsqueeze(1))
            
            # Sample or take the argmax to get the next token
            next_token = torch.argmax(logits, dim=-1)
            
            # Append the embedding of the next token to the hidden state for the next iteration
            next_token_embedded = torch.stack([self.gnn(graph_data_vocabulary[next_token[i]]).unsqueeze(0) for i in range(next_token.size(0))], dim=0)  # Embed next token with GNN
            hidden_state = torch.cat([hidden_state, next_token_embedded], dim=1)  # Append to the sequence
        
        # Stack logits along the sequence dimension to get shape [B, SEQ_LEN, VOCAB_SIZE]
        logits = torch.cat(logits_list, dim=1)
        return logits

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar

# Loss function
def vae_loss(recon_logits, x, mu, logvar):
    recon_loss = nn.CrossEntropyLoss()(recon_logits.view(-1, VOCAB_SIZE), x.view(-1))
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_divergence / x.size(0)


def main():
    # Initialize model and optimizer
    model = TransformerVAE(EMBED_DIM, LATENT_DIM, SEQ_LEN)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Dummy dataset: Replace with actual sequence data
    dataset = TensorDataset(torch.randint(0, VOCAB_SIZE, (100, SEQ_LEN)))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Training loop
    model.train()
    for epoch in range(EPOCHS):
        for batch in dataloader:
            x = batch[0]
            optimizer.zero_grad()
            recon_logits, mu, logvar = model(x)
            loss = vae_loss(recon_logits, x, mu, logvar)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    # Sampling new sequences
    def sample(model, num_samples=5):
        model.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, LATENT_DIM)  # Sample from the prior
            logits = model.decode(z)  # Decode from latent space
            sampled_tokens = torch.argmax(logits, dim=-1)  # Take the most likely token at each position
        return sampled_tokens

    # Generate and print new sequences
    sampled_sequences = sample(model)
    print("Sampled sequences:", sampled_sequences)


if __name__ == "__main__":
    breakpoint()
    main()
