import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import networkx as nx
import numpy as np
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data as GraphData
import torch.nn.functional as F

# Hyperparameters
VOCAB_SIZE = 50  # Number of graph tokens
EMBED_DIM = 64
LATENT_DIM = 32
SEQ_LEN = 10  # Max length of sequences
BATCH_SIZE = 16
EPOCHS = 1

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
    res = GraphData(x=x, edge_index=edge_index)
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

    def forward(self, x, mask=None, src_key_padding_mask=None):
        return self.encoder(x, mask=mask, src_key_padding_mask=src_key_padding_mask)

# Define VAE
class TransformerVAE(nn.Module):
    def __init__(self, embed_dim, latent_dim, seq_len):
        super(TransformerVAE, self).__init__()
        self.gnn = TokenGNN(embed_dim)
        self.transformer_encoder = TransformerEncoder(embed_dim)
        self.fc_mu = nn.Linear(embed_dim, latent_dim)
        self.fc_logvar = nn.Linear(embed_dim, latent_dim)
        
        self.fc_decode = nn.Linear(latent_dim, latent_dim)
        self.transformer_decoder = TransformerEncoder(embed_dim + latent_dim)
        self.output_layer = nn.Linear(embed_dim + latent_dim, VOCAB_SIZE)
        self.start_token_embedding = nn.Parameter(torch.randn(1, 1, embed_dim))

    def embed_tokens(self, token_ids):
        if token_ids.dim() == 2:
            return torch.stack([self.embed_tokens(token_id_seq) for token_id_seq in token_ids], dim=0)
        embedded_tokens = []
        for token_id in token_ids:
            graph_data = graph_data_vocabulary[token_id]
            embedded_token = self.gnn(graph_data)
            embedded_tokens.append(embedded_token)
        return torch.stack(embedded_tokens, dim=0)

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

    def decode(self, z, attention_mask):
        # Get the actual sequence length from attention_mask
        actual_seq_len = attention_mask.size(1) + 1
        # Use the learned start token embedding as the initial input
        start_token = self.start_token_embedding.expand(z.size(0), -1, -1)  # Shape (batch, 1, embed_dim)        
        # Project z to the embedding dimension and repeat across the sequence
        z_projected = self.fc_decode(z).unsqueeze(1).expand(-1, actual_seq_len - 1, -1)
        
        # Apply causal mask and padding mask for autoregressive decoding
        breakpoint()
        inputs = torch.cat([start_token, z_projected], dim=1)
        # Pad the attention mask with an additional True for the start token
        padded_attention_mask = torch.cat([torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=attention_mask.device), attention_mask], dim=1)        
        causal_mask = torch.triu(torch.ones((actual_seq_len, actual_seq_len), dtype=torch.bool), diagonal=1).to(z.device)

        # Transformer decoder with causal and padding masks
        decoded_seq = self.transformer_decoder(
            inputs.permute(1, 0, 2),  # (seq_len, batch, embed_dim)
            src_key_padding_mask=~padded_attention_mask,
            mask=causal_mask
        )
        
        decoded_seq = decoded_seq.permute(1, 0, 2)  # Convert back to (batch, seq_len, embed_dim)
        logits = self.output_layer(decoded_seq)
        return logits[:, 1:]
    
    # def autoregressive_training_step(self, x, z, max_seq_len=10):
    #     # Initialize with the start token embedding
    #     input_token = self.start_token_embedding.expand(x.size(0), -1, -1)  # Start token embedding
    #     z_context = self.fc_decode(z).unsqueeze(1)  # Shape (batch, 1, embed_dim)
    #     generated_logits = []

    #     for t in range(max_seq_len):
    #         # Concatenate input_token and z_context for the current step
    #         embedded_token = torch.cat([input_token, z_context], dim=-1)

    #         # Apply a causal mask that grows with each step t
    #         causal_mask = torch.triu(torch.ones((t + 1, t + 1), dtype=torch.bool), diagonal=1).to(z.device)

    #         output = self.transformer_decoder(
    #             embedded_token.permute(1, 0, 2),  # Shape (seq_len, batch, embed_dim)
    #             mask=causal_mask
    #         ).permute(1, 0, 2)

    #         # Predict the next token and accumulate logits
    #         logits = self.output_layer(output[:, -1, :])  # Only the last token's output is needed
    #         generated_logits.append(logits.unsqueeze(1))

    #         # Update input_token with the actual next token from x during training
    #         if t < max_seq_len - 1:
    #             next_token_id = x[:, t + 1].unsqueeze(-1)  # Use teacher forcing
    #             input_token = self.gnn(graph_data_vocabulary[next_token_id.squeeze(-1)]).unsqueeze(1)

    #     # Concatenate logits across time steps to match the full sequence length
    #     return torch.cat(generated_logits, dim=1)        
    
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
                    next_token_embedding = self.gnn(graph_data_vocabulary[next_token_id.item()]).unsqueeze(0)

                    # Append the embedded token to this sequence’s token_embeddings list
                    token_embeddings[idx] = torch.cat((token_embeddings[idx], next_token_embedding), dim=0)

        # Concatenate logits across time steps for each sequence to compute the loss                        
        recon_logits_list = []
        for seq_logits in generated_logits:
            seq_logits_cat = torch.cat(seq_logits, dim=0)
            padding = torch.zeros(max_seq_len - len(seq_logits), VOCAB_SIZE)
            recon_logits = torch.cat((seq_logits_cat, padding), dim=0)
            recon_logits_list.append(recon_logits)

        padded_logits = torch.stack(recon_logits_list, dim=0)
        mask = torch.zeros((padded_logits.shape[0], max_seq_len), dtype=torch.bool, device=padded_logits.device)
        for i, logits in enumerate(generated_logits):
            mask[i, :len(logits)] = 1  # Mark valid positions up to the length of each sequence
        return padded_logits.view(-1, VOCAB_SIZE), mask

    
    def autoregressive_decode(self, z, max_seq_len=10):
        # Initialize with the learned start token embedding
        input_token = self.start_token_embedding.expand(z.size(0), -1, -1)  # Start token embedding
        
        # Expand z as a context vector across all decoding steps
        z_context = self.fc_decode(z).unsqueeze(1)  # Shape (batch, 1, embed_dim)
        generated_sequence = []

        for t in range(max_seq_len):
            # Concatenate the start token embedding with z context at each step
            embedded_token = torch.cat([input_token, z_context], dim=-1)
            
            # Autoregressive decoding with causal mask
            causal_mask = torch.triu(torch.ones((t + 1, t + 1), dtype=torch.bool), diagonal=1).to(z.device)
            output = self.transformer_decoder(
                embedded_token.permute(1, 0, 2),  # Transformer expects (seq_len, batch, embed_dim)
                mask=causal_mask
            ).permute(1, 0, 2)
            
            # Predict the next token and append to sequence
            logits = self.output_layer(output[:, -1, :])  # Only the last token's output is needed
            next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
            generated_sequence.append(next_token)
            
            # Update input token for the next step
            input_token = self.gnn(graph_data_vocabulary[next_token.squeeze(-1)])

        # Concatenate the generated tokens into a full sequence
        return torch.cat(generated_sequence, dim=1)    
    

    def autoregressive_inference(self, z, max_seq_len):
        # Initialize with the start token embedding and z context
        batch_size = z.size(0)
        z_context = self.fc_decode(z)  # Shape (batch, latent_dim)

        # Initialize embeddings for each sequence using the start token
        token_embeddings = list(torch.unbind(self.start_token_embedding.expand(batch_size, -1, -1), dim=0))  # Shape (batch, 1, embed_dim)
        generated_sequences = [[] for _ in range(batch_size)]  # Store generated tokens for each sequence

        for t in range(max_seq_len):
            # Prepare a temporary batch of active sequences at the current timestep
            temp_batch_embeddings = []
            temp_z_context = []
            active_indices = []

            for idx in range(batch_size):
                # Check if the sequence needs more tokens
                if t == 0 or len(generated_sequences[idx]) < max_seq_len:
                    accumulated_embeddings = token_embeddings[idx]  # Use accumulated embeddings up to this step
                    temp_batch_embeddings.append(accumulated_embeddings)
                    temp_z_context.append(z_context[idx].unsqueeze(0).expand(accumulated_embeddings.size(0), -1))
                    active_indices.append(idx)

            # If no active sequences remain, end generation
            if not active_indices:
                break

            # Convert lists to tensors for batched transformer input
            temp_batch_embeddings = torch.stack(temp_batch_embeddings)
            temp_z_context = torch.stack(temp_z_context)

            # Concatenate accumulated embeddings with z_context
            transformer_input = torch.cat([temp_batch_embeddings, temp_z_context], dim=-1)

            # Apply causal mask for autoregressive decoding
            seq_len = transformer_input.size(1)
            causal_mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.bool), diagonal=1).to(z.device)

            output = self.transformer_decoder(
                transformer_input.permute(1, 0, 2),  # Shape (seq_len, batch, embed_dim + latent_dim)
                mask=causal_mask
            ).permute(1, 0, 2)

            # Predict logits for the next token and sample from the distribution
            logits = self.output_layer(output[:, -1, :])  # Only the last token's output
            next_tokens = torch.argmax(logits, dim=-1)  # Shape (batch,)

            # Update each active sequence with the newly generated token
            for i, idx in enumerate(active_indices):
                generated_sequences[idx].append(next_tokens[i].item())  # Store generated token
                next_token_embedding = self.gnn(graph_data_vocabulary[next_tokens[i].item()]).unsqueeze(0)

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


def main():
    # Initialize model and optimizer
    model = TransformerVAE(EMBED_DIM, LATENT_DIM, SEQ_LEN)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Dummy dataset: Replace with actual sequence data
    dataset = [torch.randint(0, VOCAB_SIZE, (np.random.randint(3, SEQ_LEN),)) for _ in range(100)]
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

    # Training loop
    model.train()
    for epoch in range(EPOCHS):
        for batch in dataloader:
            x, attention_mask, seq_len_list = batch
            optimizer.zero_grad()
            recon_logits, mask, mu, logvar = model(x, attention_mask, seq_len_list)
            loss = vae_loss(recon_logits, mask, x, mu, logvar)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    # Sampling new sequences
    def sample(model, num_samples=5):
        model.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, LATENT_DIM)  # Sample from the prior
            generated_sequences = model.autoregressive_inference(z, 10)  # Decode from latent space
        return generated_sequences

    # Generate and print new sequences
    sampled_sequences = sample(model)
    print("Sampled sequences:", sampled_sequences)


if __name__ == "__main__":
    breakpoint()
    main()
