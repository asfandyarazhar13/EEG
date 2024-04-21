import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import argparse

class CNNModel(nn.Module):
    """
    A neural network model for processing 2D spectrograms using a residual CNN,
    followed by an attention mechanism, and a decoder with temperature scaling.

    Attributes:
        cnn: Sequential 2D CNN layers for feature extraction.
        transformer_encoder: Transformer encoder for the attention mechanism.
        decoder_logits: Linear layer to produce logits for classification.
        decoder_temperature: Linear layer to compute temperature for scaling logits.
    """

    def __init__(self, num_channels, num_classes, num_heads, num_encoder_layers, dropout_rate=0.1):
        """
        Initializes the Spectrogram CNN Model with the given parameters.

        Args:
            num_channels: Number of input channels in the spectrogram.
            num_classes: Number of output classes.
            num_heads: Number of heads for the multi-head attention mechanism.
            num_encoder_layers: Number of transformer encoder layers.
            dropout_rate: Dropout rate in the transformer encoder.
        """
        super(CNNModel, self).__init__()
        self.final_cnn_dim = 4 * 256  # Output dimension of CNN, should be a product of '4' and feature size '256'

        # 2D CNN for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, self.final_cnn_dim // 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.final_cnn_dim // 16, self.final_cnn_dim // 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.final_cnn_dim // 8, self.final_cnn_dim, kernel_size=3, padding=1)
        )

        # Global average pooling layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Transformer encoder layer
        encoder_layer = TransformerEncoderLayer(
            d_model=self.final_cnn_dim, nhead=num_heads, dropout=dropout_rate)
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=num_encoder_layers)

        # Decoder layers
        self.decoder_logits = nn.Linear(self.final_cnn_dim, num_classes)
        self.decoder_temperature = nn.Linear(self.final_cnn_dim, 1)

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x: Input tensor of shape (B, C, H, W), where B is the batch size, C is the number of channels,
               H is the height (frequency bins), and W is the width (time bins) of the spectrogram.

        Returns:
            The output probability distributions for each class after softmax with temperature scaling.
        """
        # Apply the CNN and add a residual connection
        cnn_out = self.cnn(x)
        residual_out = cnn_out + x  # Ensure x has the same number of channels

        # Apply global average pooling
        pooled = self.global_avg_pool(residual_out)
        flattened = pooled.view(pooled.size(0), -1)

        # Reshape to (B, 4, 256)
        reshaped = flattened.view(-1, 4, 256)

        # Apply transformer encoder
        attention_out = self.transformer_encoder(reshaped)

        # Apply mean pooling over the sequence length dimension
        pooled_attention_out = attention_out.mean(dim=1)

        # Decode into logits and temperature
        logits = self.decoder_logits(pooled_attention_out)
        temperature = torch.exp(self.decoder_temperature(pooled_attention_out))

        # Apply temperature scaling to logits and softmax
        scaled_logits = logits / temperature.unsqueeze(-1)
        probabilities = F.softmax(scaled_logits, dim=-1)

        return probabilities

def parse_arguments():
    """
    Parses command line arguments for the spectrogram model configuration.

    Returns:
        Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='2D Spectrogram Model Configuration')
    parser.add_argument('--num_channels', type=int, default=1, help='Number of input channels in the spectrogram')
    parser.add_argument('--num_classes', type=int, default=6, help='Number of output classes')
    parser.add_argument('--num_heads', type=int, default=2, help='Number of heads for the multi-head attention mechanism')
    parser.add_argument('--num_encoder_layers', type=int, default=2, help='Number of transformer encoder layers')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate in the transformer encoder')
    return parser.parse_args()

def main():
    """
    Main function to instantiate and showcase the CNN Model.
    """
    # Parse the command line arguments
    args = parse_arguments()

    # Create an instance of the Spectrogram CNN Model with the provided arguments
    model = CNNModel(
        num_channels=args.num_channels,
        num_classes=args.num_classes,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        dropout_rate=args.dropout_rate
    )

if __name__ == '__main__':
    main()
