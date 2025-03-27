import torch
import torch.nn as nn
from sklearn.manifold import TSNE
import networkx as nx
import numpy as np
import importlib
import matplotlib.pyplot as plt
from src.config import *
from src.grammar.utils import get_next_version
from src.grammar.ednce import *
from src.grammar.common import *
from src.examples.test_graphs import *
from torch.nn import functional as F

from src.draw.graph import draw_graph
dagnn = importlib.import_module('dagnn.ogbg-code.model.dagnn')
utils_dag = importlib.import_module('dagnn.src.utils_dag')
DAGNN = dagnn.DAGNN
# if error, go to induction/dagnn/ogbg-code/model/dagnn.py, change a line to "from dagnn.src.constants import *""


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
    def __init__(self, encoder, repr, encoder_layers, decoder_layers, output_dims, vocabulary_init, vocabulary_terminate, embed_dim, latent_dim, seq_len, cuda):
        super(TransformerVAE, self).__init__()        
        # self.token_gnn = TokenGNN(embed_dim)
        self.cuda = cuda
        self.encoder = encoder
        self.repr = repr
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        if repr == "digged":
            self.vocab_size = output_dims
        else:
            self.vocab_size, graph_args = output_dims
            self.nvt = graph_args.num_vertex_type
            self.max_n = graph_args.max_n
            
        if encoder == "TOKEN_GNN":
            self.gnn = DAGNN(None, None, len(TERMS+NONTERMS)+1, latent_dim, None, w_edge_attr=False, bidirectional=False, num_class=embed_dim)        
            self.transformer_encoder = TransformerEncoder(embed_dim, num_layers=encoder_layers)            
        elif encoder == "GNN":
            self.token_embedding = nn.Embedding(self.vocab_size, embed_dim)
            self.gnn = DAGNN(None, None, len(TERMS+NONTERMS)+1, latent_dim, None, w_edge_attr=False, bidirectional=False, num_class=embed_dim)
        else:
            self.token_embedding = nn.Embedding(self.vocab_size, embed_dim)
            self.transformer_encoder = TransformerEncoder(embed_dim, num_layers=encoder_layers)
        self.fc_mu = nn.Linear(embed_dim, latent_dim)
        self.fc_logvar = nn.Linear(embed_dim, latent_dim)        
        self.fc_decode = nn.Linear(latent_dim, latent_dim)
        self.transformer_decoder = TransformerEncoder(embed_dim + latent_dim, num_layers=decoder_layers)
        self.start_token_embedding = nn.Parameter(torch.randn(1, 1, embed_dim))
        if repr == "digged":
            self.output_layer = nn.Linear(embed_dim + latent_dim, self.vocab_size)
        else:
            self.fc_ns = nn.Linear(self.vocab_size, embed_dim)
            self.add_vertex = nn.Linear(embed_dim + latent_dim, self.nvt)
            if self.vocab_size == self.nvt+self.max_n:
                out_dim = self.max_n 
            else:
                out_dim = self.max_n-1
            self.add_edge = nn.Linear(embed_dim + latent_dim, out_dim)
        self.sigmoid = nn.Sigmoid()
        if repr == "digged":
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
                embedded_token = torch.zeros((self.latent_dim,), device=self.cuda)
            else:
                graph_data.to(self.cuda)                
                embedded_token = self.gnn(graph_data).flatten()
            embedded_tokens.append(embedded_token)
        return torch.stack(embedded_tokens, dim=0)
        # return torch.cat(embedded_tokens, dim=0)

    def encode(self, x, attention_mask=None):        
        if self.encoder == "GNN":
            pooled = torch.stack([self.gnn(g.to(self.cuda)).flatten() for g in x], dim=0)
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

        if self.repr == "ns":                    
            max_seq_len_list = max_seq_len_list[:, 0]

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
            if self.repr == "ns":
                n_types = self.add_vertex(output[:, -1, :])
                # n_types = F.softmax(n_types, 1)
                edge_logits = self.add_edge(output[:, -1, :])
                edge_logits = self.sigmoid(edge_logits)
                logits = torch.cat((n_types, edge_logits), dim=-1)
            else:
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
                    if self.repr == "ns":
                        next_token_embedding = self.fc_ns(next_token_embedding)
                    # Append the embedded token to this sequence’s token_embeddings list
                    token_embeddings[idx] = torch.cat((token_embeddings[idx], next_token_embedding), dim=0)

        # Concatenate logits across time steps for each sequence to compute the loss                        
        recon_logits_list = []
        for seq_logits in generated_logits:
            seq_logits_cat = torch.cat(seq_logits, dim=0)
            padding = torch.zeros(max_seq_len - len(seq_logits), self.vocab_size, device=seq_logits_cat.device)
            recon_logits = torch.cat((seq_logits_cat, padding), dim=0)
            recon_logits_list.append(recon_logits)
        padded_logits = torch.stack(recon_logits_list, dim=0)
        if self.repr == "ns":
            mask = torch.zeros_like(padded_logits, dtype=torch.bool, device=padded_logits.device)
        else:
            mask = torch.zeros((padded_logits.shape[0], max_seq_len), dtype=torch.bool, device=padded_logits.device)
        for i, logits in enumerate(generated_logits):
            mask[i, :len(logits)] = 1  # Mark valid positions up to the length of each sequence
        if self.repr == "ns":
            padded_logits = padded_logits.view(-1,)
            mask = mask.view(-1,)
        else:
            padded_logits = padded_logits.view(-1, self.vocab_size)

        return padded_logits, mask
    

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
                    if self.terminate_mask[last_token].item():
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
        if seq_len == 1:
            logits[:,~self.init_mask] = float("-inf")
            logits[:,self.terminate_mask] = float("-inf") # optionally mask out the terminating ones too
        else:
            logits[:,self.init_mask] = float("-inf")
        if seq_len == self.seq_len:
            logits[:,~self.terminate_mask] = float("-inf")
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
                if self.encoder == "TOKEN_GNN":
                    next_token_embedding = self.gnn(graph_data_vocabulary[next_tokens[i].item()])
                else: # default to learnable embedding
                    next_token_embedding = self.token_embedding(next_tokens[i])
                # Append the new embedding to this sequence’s token embeddings
                token_embeddings[idx] = torch.cat((token_embeddings[idx], next_token_embedding), dim=0)
        return generated_sequences

    @staticmethod
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

    @staticmethod
    def path_init(g):
        orders = []
        for order in nx.all_topological_sorts(g):
            bad = False
            for l in range(len(order)-1):
                if order[l+1] not in g[order[l]]:
                    bad = True 
                    break
            if not bad:
                orders.append(order)
        return orders if orders else None

    @staticmethod
    def get_inmi(g, ignore_nt=True):
        inmi = {}
        for n in g:
            if g.nodes[n]['label'] not in NONTERMS:
                inmi[n] = len(inmi) # inv node map index    
        return inmi

    @staticmethod
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

    @staticmethod
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
                        safe = True
            if not safe:
                return False
        return True

    @staticmethod
    def check_bn(grammar, conn, g, rule):
        rs = rule.subgraph
        nodes = grammar.search_nts(rs, TERMS)        
        cur_labels = set([g.nodes[n]['label'] for n in g])
        labels = set([rs.nodes[n]['label'] for n in nodes])
        # not in g
        for l in labels:
            if l in cur_labels:
                return False
        if len(nodes) == len(rs):
            if (set(TERMS)-set([g.nodes[n]['label'] for n in g])) != set(labels):
                return False
        return True      


    @staticmethod
    def check_enas(grammar, conn, g, rule):
        # imposed by dataset        
        nts = grammar.search_nts(rule.subgraph, NONTERMS)        
        if len(nts) == 0:
            return len(g) + len(rule.subgraph) == 9 # nt in g
        else:
            return True


    @staticmethod
    def update_path(g, o, grammar, j, token2rule):
        if o is None:
            return o
        rule = grammar.rules[token2rule[j]]
        # conn = TransformerVAE.check_connections(grammar, g, rule)
        # out_pairs = list(filter(lambda a: isinstance(a[1], int), conn))
        # in_pairs = list(filter(lambda a: isinstance(a[0], int), conn))
        # o_j, l = TransformerVAE.check_linear(rule, o, in_pairs, out_pairs)
        g = grammar.one_step_derive(g, token2rule[j], token2rule)
        return TransformerVAE.path_init(g)
        # return o[:l] + o_j + o[l:]
                


    @staticmethod
    def update_order(g, o, grammar, j, token2rule):
        rule = grammar.rules[token2rule[j]]
        conn = TransformerVAE.check_connections(grammar, g, rule)
        out_pairs = list(filter(lambda a: isinstance(a[1], int), conn))
        in_pairs = list(filter(lambda a: isinstance(a[0], int), conn))
        # updates node-node predecence info, ignoring nts
        inmi = TransformerVAE.get_inmi(g, ignore_nt=True)
        rhs = rule.subgraph
        inmi_rhs = TransformerVAE.get_inmi(rhs, ignore_nt=True)
        n = len(o)
        m = len(inmi_rhs)
        r_adj = TransformerVAE.order_init(rhs)
        o_ = np.zeros((n+m, n+m))
        o_[:n, :n] = o
        o_[n:, n:] = r_adj
        for (a, x) in out_pairs:
            # get all y reachable from x
            if list(rhs)[x] not in inmi_rhs:
                continue
            for y in np.argwhere(r_adj[inmi_rhs[list(rhs)[x]]]).flatten():
                # get all b that reaches a
                for b in np.argwhere(o[:, inmi[a]]).flatten():
                    o_[b, y+n] = 1            
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



    # @staticmethod
    # def check_linear(rule, orders, in_pairs, out_pairs):
        # rule itself needs to be linear
        # top_sorts = list(nx.all_topological_sorts(rule.subgraph))
        # for o in top_sorts: # fast, just one token
        #     # find an insertion point
        #     s, t = list(o)[0], list(o)[-1]
        #     for l in range(1, len(order)-1):
        #         a = order[l]
        #         b = order[l+1]
        #         if (a, s) in out_pairs and (t, b) in in_pairs:
        #             return o, l # index
        # for order in orders:
        #     for o in list(nx.all_topological_sorts(rule.subgraph)):
        #         s, t = list(o)[0], list(o)[-1]
        #         for l in range(1, len(order)-1):
        #             a = order[l]
        #             b = order[l+1]
        #             if (a, s) in out_pairs and (t, b) in in_pairs:
        #                 return True
        # return False
    
    @staticmethod
    def check_linear(grammar, j, g, token2rule):
        g = grammar.one_step_derive(g, token2rule[j], token2rule)
        top_sorts = list(nx.all_topological_sorts(g))
        for o in top_sorts:
            bad = False
            for l in range(len(o)-1):
                a = o[l]
                b = o[l+1]
                if b not in g[a]:
                    bad = True
            if not bad:
                return True
        return False
    
    def interactive_mask_logits(self, grammar, generated_graphs, generated_orders, generated_paths, logits, token2rule):
        def mask_init_ckt():
            # for simplicity for now, ensure the rule itself is reachable
            # if we can do lookahead on the first step, that'll be better
            s = get_node_by_label(rule.subgraph, LOOKUP['input'])
            t = get_node_by_label(rule.subgraph, LOOKUP['output'])
            x = list(rule.subgraph).index(s)
            y = list(rule.subgraph).index(t)            
            return min(o_j[x].sum(), o_j[:, y].sum()) < o_j.shape[0]
        def mask_init_enas():
            for order in nx.all_topological_sorts(rule.subgraph):
                bad = False
                for l in range(len(order)-1):
                    if order[l+1] not in rule.subgraph[order[l]]:
                        bad = True
                        break
                if not bad:
                    return False # good
            return True # bad
        def mask_init_bn():
            return False # always good
        def check_acyclic():
            acyclic = True
            for (a, x), (y, b) in product(out_pairs, in_pairs):
                if o_j[x, y] and o[inmi[b], inmi[a]]:
                    acyclic = False
            return acyclic
        def check_reachable(out_pairs, in_pairs):
            t = get_node_by_label(g, LOOKUP['output'])
            for s in g:
                if s not in inmi: # can get rewired some other way
                    continue
                reachable = False
                if o[inmi[s], inmi[t]]:
                    reachable = True
                    continue
                for (a, x), (y, b) in product(out_pairs, in_pairs):
                    if o_j[x, y] and o[inmi[s], inmi[a]] and o[inmi[b], inmi[t]]:
                        reachable = True
                if not reachable:
                    return False
            for x in range(o_j.shape[0]):
                reachable = False
                for (y, b) in in_pairs:
                    if o_j[x, y] and o[inmi[b], inmi[t]]:
                        reachable = True
                if not reachable:
                    return False                
            s = get_node_by_label(g, LOOKUP['input'])
            for t in g:
                if t not in inmi:
                    continue
                reachable = False
                if o[inmi[s], inmi[t]]:
                    reachable = True
                    continue
                for (a, x), (y, b) in product(out_pairs, in_pairs):
                    if o_j[x, y] and o[inmi[s], inmi[a]] and o[inmi[b], inmi[t]]:
                        reachable = True
                if not reachable:
                    return False                        
            for y in range(o_j.shape[0]):
                reachable = False
                for (a, x) in out_pairs:
                    if o_j[x, y] and o[inmi[s], inmi[a]]:
                        reachable = True
                if not reachable:
                    return False                
            return True   
        batch_size = logits.shape[0]
        for i in range(batch_size):
            g = generated_graphs[i]
            inmi = {}

            if g is not None:
                for n in g:
                    if g.nodes[n]['label'] not in NONTERMS:
                        inmi[n] = len(inmi)

            o = generated_orders[i]
            # for j in tqdm(range(logits.shape[1]), desc="masking logits single"):
            for j in torch.arange(logits.shape[1])[logits[i]!=float("-inf")].numpy(): # terminating logic already in _autoregressive_inference_predict_logits
                # g = deepcopy(generated_graphs[i])
                rule = grammar.rules[token2rule[j]]
                o_j = TransformerVAE.order_init(rule.subgraph, ignore_nt=False)
                if generated_graphs[i] is None:                    
                    if DATASET == "ckt":                        
                        cond = mask_init_ckt()
                    elif DATASET == "enas":
                        cond = mask_init_enas()
                    else:
                        cond = mask_init_bn()  
                    if cond:
                        logits[i, j] = float("-inf")                        
                    continue
                ### sanity checks
                ## stays connected
                # g_, applied, node_map = grammar.one_step_derive(g, token2rule[j], token2rule, return_applied=True)
                # inv_node_map_index = dict(zip(node_map.values(), map(lambda k: list(rule.subgraph).index(k), node_map.keys())))
                # stays_connected = nx.is_connected(nx.Graph(g_))            
                ## stays connected
                # new_edges = set(g_.edges)-set(g.edges)-set(product(inv_node_map_index, inv_node_map_index))
                conn = TransformerVAE.check_connections(grammar, g, rule)
                ## check acyclic
                # cycle can only form from ((a, x), (y, b)) in product(conn, conn) satisfying:
                # 1. a and b from g
                # 2. x and y from range(len(rule.subgraph))
                # 3. o_j[x,y]
                # 4. o[b,a]                              

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
                out_pairs = list(filter(lambda a: isinstance(a[1], int), conn))
                in_pairs = list(filter(lambda a: isinstance(a[0], int), conn))
                if DATASET == "ckt":
                    if not (len(conn) and check_reachable(out_pairs, in_pairs) and check_acyclic()):
                        logits[i, j] = float("-inf") 
                        continue                   
                    amp_valid = TransformerVAE.check_op_amp(grammar, conn, g, rule)
                    if not amp_valid:
                        logits[i, j] = float("-inf")
                elif DATASET == "enas":
                    if not (len(conn) and check_reachable(out_pairs, in_pairs) and check_acyclic()):
                        logits[i, j] = float("-inf")
                        continue
                    enas_valid = TransformerVAE.check_enas(grammar, conn, g, rule)
                    if not enas_valid:
                        logits[i, j] = float("-inf")
                        continue
                    linear = TransformerVAE.check_linear(grammar, j, g, token2rule)
                    if not linear:
                        logits[i, j] = float("-inf")
                elif DATASET == "bn":
                    if not (len(conn) and check_acyclic()):
                        logits[i, j] = float("-inf")
                        continue
                    bn_valid = TransformerVAE.check_bn(grammar, conn, g, rule)
                    if not bn_valid:
                        logits[i, j] = float("-inf")


    def autoregressive_interactive_inference(self, z, grammar, token2rule, max_seq_len=10, decode='greedy'):
        # decode is greedy or softmax
        batch_size = z.size(0)
        z_context = self.fc_decode(z)              
        token_embeddings = list(torch.unbind(self.start_token_embedding.expand(batch_size, -1, -1), dim=0))  
        generated_sequences = [[] for _ in range(batch_size)]                  
        generated_graphs = [None for _ in range(batch_size)]
        generated_orders = [np.zeros((0, 0)) for _ in range(batch_size)] # info to guarantee validity
        if DATASET == "enas":
            generated_paths = [[] for _ in range(batch_size)] # consecutive path
        else:
            generated_paths = [None for _ in range(batch_size)]

        for t in range(max_seq_len):
            temp_batch_embeddings, temp_z_context, active_indices = self._autoregressive_inference_active_indices(z_context, generated_sequences, token_embeddings, max_seq_len)                
            if not active_indices:
                break
            logits = self._autoregressive_inference_predict_logits(temp_batch_embeddings, temp_z_context)
            self.interactive_mask_logits(grammar, [generated_graphs[i] for i in active_indices], [generated_orders[i] for i in active_indices], [generated_paths[i] for i in active_indices], logits, token2rule)
            if decode == 'greedy':
                next_tokens = torch.argmax(logits, dim=-1) # greedy
            else:
                probs = torch.softmax(logits, dim=-1) # sample
                probs_copy = probs.clone()
                nan_idxes = (probs!=probs).any(axis=-1)
                # if nan_idxes.any(): # should not happen
                #     breakpoint()
                #     uniform = torch.ones_like(probs[nan_idxes, :])
                #     uniform[:,  ~self.terminate_mask] = 0.
                #     probs[nan_idxes, :] = uniform
                next_tokens = [None for _ in range(probs.shape[0])]
                idxes = deepcopy(active_indices)
                with tqdm(total=len(idxes), desc="one step lookaheads") as pbar:
                    while idxes: # one-step lookahead
                        nan_idxes = (probs!=probs).any(axis=-1)
                        if nan_idxes.any(): # give up and quickly terminate
                            print("give up on lookahead")
                            orig_probs = probs_copy.clone()
                            orig_probs[:,  ~self.terminate_mask] = 0.
                            probs[nan_idxes, :] = orig_probs[nan_idxes]
                        if t == 0:
                            cur_next_tokens = torch.multinomial(probs, 1).squeeze(-1)
                        else: # focus on the non (~self.terminate_mask&(~self.init_mask))
                            assert probs[:, self.init_mask].max() == 0.
                            cur_next_tokens = torch.multinomial(probs, 1).squeeze(-1)
                        mask = [True for _ in range(len(idxes))]
                        for j in range(len(idxes)-1,-1,-1):
                            # assert adding is ok
                            cur = cur_next_tokens[j]
   
                            if self.terminate_mask[cur]:
                                cond = True
                            else:
                                logits = torch.ones((1, logits.shape[1])) # dummy
                                logits[:, self.init_mask] = float("-inf")
                                if generated_graphs[idxes[j]] is None:
                                    lookahead_graph = grammar.derive([cur.item()], token2rule)
                                    lookahead_order = TransformerVAE.order_init(lookahead_graph)
                                    lookahead_path = TransformerVAE.path_init(lookahead_graph)
                                else:
                                    lookahead_graph = grammar.one_step_derive(generated_graphs[idxes[j]], cur.item(), token2rule)
                                    lookahead_order = TransformerVAE.update_order(generated_graphs[idxes[j]], generated_orders[idxes[j]], grammar, cur.item(), token2rule)                        
                                    lookahead_path = TransformerVAE.update_path(generated_graphs[idxes[j]], generated_paths[idxes[j]], grammar, cur.item(), token2rule)                                
                                self.interactive_mask_logits(grammar, [lookahead_graph], [lookahead_order], [lookahead_path], logits, token2rule)
                                cond = logits[:, self.terminate_mask&(~self.init_mask)].max() > float("-inf") # fail-safe
                            if cond:
                                next_tokens[active_indices.index(idxes[j])] = cur
                                mask[j] = False # done with this                            
                                idxes.pop(j)
                                pbar.update(1)
                            else:
                                probs[j, cur.item()] = 0.0
                                mask[j] = True
                        probs = probs[mask]
                assert all([token is not None for token in next_tokens])
            for i, idx in enumerate(active_indices):
                generated_sequences[idx].append(next_tokens[i].item())
                if generated_graphs[idx] is None:
                    generated_graphs[idx] = grammar.derive([next_tokens[i].item()], token2rule)
                    # init acyclic order
                    generated_orders[idx] = TransformerVAE.order_init(generated_graphs[idx]) # ignores nt
                    generated_paths[idx] = TransformerVAE.path_init(generated_graphs[idx]) # ignores nt
                else:
                    order_copy = deepcopy(generated_orders[idx])
                    generated_orders[idx] = TransformerVAE.update_order(generated_graphs[idx], generated_orders[idx], grammar, next_tokens[i].item(), token2rule)
                    generated_paths[idx] = TransformerVAE.update_path(generated_graphs[idx], generated_paths[idx], grammar, next_tokens[i].item(), token2rule)
                    # update acyclic order
                    g_copy = deepcopy(generated_graphs[idx])
                    generated_graphs[idx] = grammar.one_step_derive(generated_graphs[idx], next_tokens[i].item(), token2rule)
                    # debug
                    g_ = copy_graph(generated_graphs[idx], [n for n in generated_graphs[idx] if generated_graphs[idx].nodes[n]['label'] not in NONTERMS])
                    ## debug
                    # for i_ in range(len(g_)):
                    #     for j_ in range(len(g_)):
                    #         if bool(nx.has_path(g_, list(g_)[i_], list(g_)[j_])) != bool(generated_orders[idx][i_, j_]):
                    #             breakpoint()
                    #             TransformerVAE.update_order(g_copy, order_copy, grammar, next_tokens[i].item(), token2rule)
                    g = generated_graphs[idx]
                    s = get_node_by_label(g, LOOKUP['input'])
                    ## debug
                    # for t in range(len(g)):
                    #     if not nx.has_path(g, s, list(g)[t]):
                    #         breakpoint()
                    #         TransformerVAE.update_order(g_copy, order_copy, grammar, next_tokens[i].item(), token2rule)
                ## debug
                # if not nx.is_directed_acyclic_graph(generated_graphs[idx]):
                #     breakpoint()
                #     TransformerVAE.update_order(g_copy, order_copy, grammar, next_tokens[i].item(), token2rule)
                if self.encoder == "TOKEN_GNN":
                    next_token_embedding = self.gnn(graph_data_vocabulary[next_tokens[i].item()])
                else: # default to learnable embedding
                    next_token_embedding = self.token_embedding(next_tokens[i]).unsqueeze(0)
                token_embeddings[idx] = torch.cat((token_embeddings[idx], next_token_embedding), dim=0)
        return generated_sequences, generated_orders, generated_paths
    
    def _one_hot(self, idx, length):
        if type(idx) in [list, range]:
            if idx == []:
                return None
            idx = torch.LongTensor(idx).unsqueeze(0).t()
            x = torch.zeros((len(idx), length)).scatter_(1, idx, 1)
        else:
            idx = torch.LongTensor([idx]).unsqueeze(0)
            x = torch.zeros((1, length)).scatter_(1, idx, 1)
        return x    


    def ns_decode(self, z, max_seq_len=10):
        # decode is greedy or softmax
        batch_size = z.size(0)
        z_context = self.fc_decode(z)     
        # Initialize embeddings for each sequence using the start token
        token_embeddings = list(torch.unbind(self.start_token_embedding.expand(batch_size, -1, -1), dim=0))  # Shape (batch, 1, embed_dim)
        generated_logits = [[] for _ in range(batch_size)]  # Store logits for each sequence
        # Process each sequence individually, applying teacher forcing
        generated_sequences = [[] for _ in range(batch_size)]
        for t in range(max_seq_len):
            # Prepare a temporary batch of active sequences at the current timestep
            temp_batch_embeddings = []
            temp_z_context = []
            active_indices = []

            for idx in range(batch_size):
                # Check if the current sequence needs more tokens (based on its max_seq_len)
                if t < max_seq_len:
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
            if self.repr == "ns":
                breakpoint()
                n_types = self.add_vertex(output[:, -1, :])                
                edge_logits = self.add_edge(output[:, -1, :])
                edge_probs = self.sigmoid(edge_logits)
                logits = torch.cat((n_types, edge_probs), dim=-1)
            else:
                logits = self.output_layer(output[:, -1, :])  # Only the last token's output
            for i, idx in enumerate(active_indices):
                generated_logits[idx].append(logits[i].unsqueeze(0))  # Collect logits for each sequence

            # Use ground-truth tokens from x as the next input (teacher forcing)
            for i, idx in enumerate(active_indices):
                if t < max_seq_len:  # Ensure there are more ground-truth tokens to process
                    # Get the ground-truth token from x and embed it using the GNN
                    # .unsqueeze(0)  # Ground truth for next token                    
                    type_probs = logits[i, :self.nvt]
                    edge_probs = logits[i, self.nvt:]
                    breakpoint()
                    type_probs = F.softmax(type_probs)
                    new_type = torch.multinomial(type_probs, 1)
                    type_score = self._one_hot(new_type.reshape(-1).tolist(), self.nvt).to(new_type.device)
                    # ns_row = (logits[i] > 0.5).int()
                    edge_score = torch.rand((1, len(edge_probs)), device=edge_probs.device) < edge_probs
                    ns_row = torch.cat((type_score, edge_score), dim=-1)
                    generated_sequences[i].append(ns_row.view(-1,).cpu())
                    next_token_embedding = ns_row.float()
                    next_token_embedding = self.fc_ns(next_token_embedding)
                    # Append the embedded token to this sequence’s token_embeddings list
                    token_embeddings[idx] = torch.cat((token_embeddings[idx], next_token_embedding), dim=0)        
        return generated_sequences


    def forward(self, x, attention_mask, seq_len_list, batch_g_list):
        if self.repr == "digged":
            embedded_tokens = self.embed_tokens(x, batch_g_list)  # Embeds each token (graph) using GNN or learnable embedding   
        else:
            embedded_tokens = torch.tensor(x, dtype=torch.float32)
        if self.encoder == "GNN":
            mu, logvar = self.encode(batch_g_list, attention_mask)
        else:
            mu, logvar = self.encode(embedded_tokens, attention_mask)
        z = self.reparameterize(mu, logvar)
        logits, logits_mask = self.autoregressive_training_step(embedded_tokens, z, seq_len_list)
        return logits, logits_mask, mu, logvar
