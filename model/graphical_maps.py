import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import networkx as nx
from modules.gnn import GNNModel

class Decoder(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Decoder, self).__init__()
        self.logits_layer = nn.Linear(input_dim, num_classes)
        self.temperature_layer = nn.Linear(input_dim, 1)

    def forward(self, x):
        logits = self.logits_layer(x)
        temperature = torch.exp(self.temperature_layer(x))  # Positive temperature
        scaled_logits = logits / temperature
        probabilities = F.log_softmax(scaled_logits, dim=1)
        return probabilities

class ExampleModel(pl.LightningModule):
    def __init__(self, input_dims, hidden_dim, num_classes, sequence_length, num_heads, num_encoder_layers, dropout_rate):
        super().__init__()
        self.save_hyperparameters()

        self.gnn_model = GNNModel(*input_dims['gnn']) # We use GNN for convenience

        self.attention_pooling = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout_rate)
        self.decoder = Decoder(hidden_dim, num_classes)

    def forward(self, gnn_input):
        gnn_output, attn_weights = self.gnn_model(gnn_input, return_attention_weights=True) 

        attn_output, attn_weights = self.attention_pooling(gnn_output, gnn_output, gnn_output)
        probabilities = self.decoder(attn_output.mean(dim=1))
        return probabilities, attn_weights

    def training_step(self, batch, batch_idx):
        gnn_input, labels = batch
        probabilities, attn_weights = self.forward(gnn_input)
        loss = F.kl_div(probabilities, labels, reduction='batchmean')
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=1e-3)

# Visualization of attention weights
def visualize_attention_weights(attn_weights, node_labels):
    G = nx.from_numpy_matrix(attn_weights.detach().numpy(), create_using=nx.DiGraph)
    pos = nx.spring_layout(G)
    labels = {i: label for i, label in enumerate(node_labels)}
    nx.draw_networkx(G, pos, labels=labels, node_color=list(labels.keys()), cmap=plt.cm.viridis, node_size=700, font_color='white')
    plt.title('Attention Weights Visualization')
    plt.show()
