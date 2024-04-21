import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import argparse

class BiLSTMModel(nn.Module):
    """
    EEG signal processing model using a 1D residual CNN, a bidirectional LSTM,
    and a transformer attention mechanism followed by a decoder with temperature scaling.

    Attributes:
        residual_cnn (nn.Sequential): A sequence of 1D convolutional layers for feature extraction.
        lstm (nn.LSTM): A bidirectional LSTM for temporal sequence processing.
        positional_encoding (nn.Parameter): Learned positional encodings for sequence order information.
        transformer_encoder (nn.TransformerEncoder): Transformer encoder for attention mechanism.
        fc (nn.Linear): Fully connected layer to process attention outputs.
        decoder_logits (nn.Linear): Linear layer to produce logits for classification.
        decoder_temperature (nn.Linear): Linear layer to compute temperature for scaling logits.
    """
    def __init__(self, input_channels, hidden_dim, num_classes, sequence_length, num_heads, num_encoder_layers, dropout_rate=0.1):
        """
        Initializes the EEGModel with the given architecture parameters.

        Args:
            input_channels (int): Number of input EEG channels.
            hidden_dim (int): Hidden dimension size for the internal layers.
            num_classes (int): Number of classes for the output layer.
            sequence_length (int): The length of the input sequences.
            num_heads (int): Number of attention heads in the transformer encoder.
            num_encoder_layers (int): Number of layers in the transformer encoder.
            dropout_rate (float): Dropout rate for regularization.
        """
        super(EEGModel, self).__init__()

        # 1D Residual CNN block
        self.residual_cnn = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        )

        # Bidirectional LSTM for temporal sequence processing
        self.lstm = nn.LSTM(hidden_dim, hidden_dim // 2, batch_first=True, bidirectional=True)

        # Positional encoding for attention layer
        self.positional_encoding = nn.Parameter(torch.zeros(1, sequence_length, hidden_dim))

        # Transformer Encoder layers for attention
        encoder_layers = TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout_rate)
        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layers, num_layers=num_encoder_layers)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, hidden_dim)

        # Decoder layers
        self.decoder_logits = nn.Linear(hidden_dim, num_classes)
        self.decoder_temperature = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        Defines the forward pass of the EEGModel.

        Args:
            x (torch.Tensor): The input tensor with shape (batch size, sequence length, input channels).

        Returns:
            torch.Tensor: The output probabilities after softmax with temperature scaling.
        """
        # Apply residual CNN and add skip connection
        cnn_out = self.residual_cnn(x.transpose(1, 2)) + x.transpose(1, 2)

        # LSTM layer expects input of shape (batch, seq, feature)
        lstm_out, _ = self.lstm(cnn_out)

        # Add positional encodings
        lstm_out += self.positional_encoding[:, :lstm_out.size(1)]

        # Apply transformer encoder (attention layers)
        attention_out = self.transformer_encoder(lstm_out)

        # Fully connected layer after attention
        fc_out = F.relu(self.fc(attention_out.mean(dim=1)))

        # Decode into logits and temperature
        logits = self.decoder_logits(fc_out)
        temperature = torch.exp(self.decoder_temperature(fc_out))  # ensuring temperature is positive

        # Apply temperature scaling to logits
        scaled_logits = logits / temperature

        # Softmax for probabilities
        probabilities = F.softmax(scaled_logits, dim=1)

        return probabilities

def parse_arguments():
    """
    Parses command line arguments for the EEG signal processing model.
    """
    parser = argparse.ArgumentParser(description='BiLSTM Model Configuration')
    parser.add_argument('--input_channels', type=int, default=18, help='Number of input EEG channels')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension size for internal layers')
    parser.add_argument('--num_classes', type=int, default=6, help='Number of classes for output layer')
    parser.add_argument('--sequence_length', type=int, default=100, help='Length of the input sequences')
    parser.add_argument('--num_heads', type=int, default=2, help='Number of attention heads in transformer encoder')
    parser.add_argument('--num_encoder_layers', type=int, default=2, help='Number of layers in transformer encoder')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate for regularization')
    return parser.parse_args()

def main():
    # Parse the command line arguments
    args = parse_arguments()

    # Instantiate the model with the provided arguments
    model = BiLSTMModel(
        input_channels=args.input_channels,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        sequence_length=args.sequence_length,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        dropout_rate=args.dropout_rate
    )

if __name__ == '__main__':
    main()
