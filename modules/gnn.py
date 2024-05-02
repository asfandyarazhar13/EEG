import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class GNNModule(nn.Module):
    """Graph Neural Network Module using GATConv layers for EEG data analysis.

    This module uses multiple Graph Attention Network (GAT) layers to capture and enhance signal correlations and feature extraction.

    Attributes:
        gat_layers (nn.ModuleList): List of GATConv layers.
        activation_layers (nn.ModuleList): List of ReLU activation layers.
        global_pool (function): Global average pooling function.
    """
    def __init__(self, num_features: int, num_classes: int, num_layers: int = 3) -> None:
        """Initializes the GNNModule with specified number of features, classes, and layers."""
        super(GNNModule, self).__init__()

        self.gat_layers = nn.ModuleList()
        self.activation_layers = nn.ModuleList()

        # Initialize GAT layers and ReLU activations
        for i in range(num_layers):
            out_channels = num_features * 2 if i < num_layers - 1 else num_classes
            self.gat_layers.append(GATConv(num_features, out_channels))
            self.activation_layers.append(nn.ReLU())
            num_features = out_channels

        self.global_pool = global_mean_pool

    def forward(self, x, edge_index, batch_index) -> torch.Tensor:
        """Defines the forward pass of the GNNModule."""
        for gat_layer, activation in zip(self.gat_layers, self.activation_layers):
            x = gat_layer(x, edge_index)
            x = activation(x)
        
        # Apply global average pooling
        x = self.global_pool(x, batch_index)
        return x

class GNN(nn.Module):
    """Graph Neural Network for classifying EEG data.

    This network uses a GNN module followed by dense layers, dropout for regularization, and a softmax layer for classification.

    Attributes:
        gnn_module (GNNModule): The GNN module for feature extraction.
        dense1 (nn.Linear): First dense layer.
        dropout (nn.Dropout): Dropout layer for regularization.
        dense2 (nn.Linear): Second dense layer.
        softmax (nn.Softmax): Softmax layer for classification.
    """
    def __init__(self, num_features: int, num_classes: int, num_layers: int = 3) -> None:
        """Initializes the GNN with specified number of features, classes, and layers."""
        super(GNN, self).__init__()

        self.gnn_module = GNNModule(num_features, num_classes, num_layers)
        self.dense1 = nn.Linear(num_classes, num_classes * 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.dense2 = nn.Linear(num_classes * 2, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, data) -> torch.Tensor:
        """Defines the forward pass of the GNN."""
        x = self.gnn_module(data.x, data.edge_index, data.batch)

        x = self.dense1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.softmax(x)

        return x
