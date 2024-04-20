import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
import argparse

class GNN(torch.nn.Module):
    """
    Graph Neural Network (GNN) using Graph Attention Network (GAT) convolution layers.

    Attributes:
        conv1 (GATConv): First GAT convolution layer.
        convs (torch.nn.ModuleList): List of subsequent GAT convolution layers.
        lin1 (torch.nn.Linear): First linear transformation layer.
        lin2 (torch.nn.Linear): Second linear transformation layer for output.
    """
    def __init__(self, in_channels=10_000, num_conv_layers=3, hid_channels=32, num_classes=6):
        """
        Initializes the GNN model with configurable parameters.

        Args:
            in_channels (int): Number of input channels.
            num_conv_layers (int): Number of GAT convolution layers.
            hid_channels (int): Number of hidden channels for each GATConv layer.
            num_classes (int): Number of classes for classification.
        """
        super().__init__()
        self.conv1 = GATConv(in_channels, hid_channels)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_conv_layers - 1):
            self.convs.append(GATConv(hid_channels, hid_channels))
        self.lin1 = torch.nn.Linear(hid_channels, hid_channels)
        self.lin2 = torch.nn.Linear(hid_channels, num_classes)

    def reset_parameters(self):
        """Resets all the parameters of the GNN."""
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        """
        Forward pass of the GNN.

        Args:
            data: Input data containing features, edge index, and batch index.

        Returns:
            Output of the model after processing input data.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x

def get_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Configure GNN model parameters.")
    parser.add_argument('--in_channels', type=int, default=10_000, help='Number of input channels')
    parser.add_argument('--num_conv_layers', type=int, default=3, help='Number of GAT convolution layers')
    parser.add_argument('--hid_channels', type=int, default=32, help='Number of hidden channels')
    parser.add_argument('--num_classes', type=int, default=6, help='Number of classes for classification')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    model = GNN(
        in_channels=args.in_channels,
        num_conv_layers=args.num_conv_layers,
        hid_channels=args.hid_channels,
        num_classes=args.num_classes
    )
    print("Model initialized with the following parameters:", args)

if __name__ == "__main__":
    main()
