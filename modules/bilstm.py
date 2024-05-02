import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import ResBlock1D, NodeAttention, BiLSTMBlock


class BiLSTMModule(nn.Module):
    """Module comprising of a projection layer, downsampling using residual blocks, and a BiLSTM block for embedding.

    Attributes:
        layers (list): A list specifying the number of features in each block.
        projection (nn.Conv1d): Initial convolution layer to project input into a higher dimensional space.
        downsample (nn.Sequential): Sequential model containing several residual blocks for downsampling.
        temporal_embed (BiLSTMBlock): BiLSTM block to capture temporal dependencies.
    """
    layers = [
        16,
        24,
        32,
        48,
        64,
        96,
        256
    ]

    def __init__(self, in_channels: int = 1, scale: int = 1) -> None:
        """Initializes the BiLSTMModule with optional scaling of feature dimensions."""
        super(BiLSTMModule, self).__init__()

        self.projection = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=self.layers[0] * scale, 
            kernel_size=5,
            padding=2
        )
        self.downsample = nn.Sequential(
            ResBlock1D(self.layers[0] * scale, self.layers[1] * scale, kernel_size=5, padding=2),
            ResBlock1D(self.layers[1] * scale, self.layers[2] * scale, kernel_size=5, padding=2),
            ResBlock1D(self.layers[2] * scale, self.layers[3] * scale, kernel_size=3, padding=1),
            ResBlock1D(self.layers[3] * scale, self.layers[4] * scale, kernel_size=3, padding=1),
            ResBlock1D(self.layers[4] * scale, self.layers[5] * scale, kernel_size=3, padding=1)
        )
        self.temporal_embed = BiLSTMBlock(
            input_size=self.layers[5] * scale, 
            embed_dim=self.layers[6] * scale
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the BiLSTMModule."""
        x = self.projection(x)
        x = self.downsample(x)
        x = self.temporal_embed(x)
        return x


class BiLSTMModel(nn.Module):
    """Complete BiLSTM Model integrating module, node attention, and output layers.

    Attributes:
        encoder (BiLSTMModule): Module for initial feature extraction and embedding.
        signal_dim (int): Dimension of the signal output from the encoder.
        embed_dim (int): Embedding dimension used for node attention and output.
        node_attention (NodeAttention): Attention mechanism to refine features.
        pool_factor (nn.Linear): Linear layer to aggregate features.
        decoder (nn.Linear): Linear layer for classification.
        temperature (nn.Sequential): Network to learn a temperature scaling for softmax.
    """
    def __init__(self, in_channels: int = 1, scale: int = 1) -> None:
        """Initializes the BiLSTMModel with optional scaling of feature dimensions."""
        super(BiLSTMModel, self).__init__()

        self.encoder = BiLSTMModule(in_channels, scale)
        self.signal_dim = self.encoder.layers[-1] * scale
        self.embed_dim = self.signal_dim

        self.node_attention = NodeAttention(
            input_size=self.signal_dim, embed_dim=self.signal_dim
        )
        self.pool_factor = nn.Linear(self.signal_dim, 1)

        self.decoder = nn.Linear(self.embed_dim, 6)
        self.temperature = nn.Sequential(
            nn.Linear(self.embed_dim, 1),
            nn.Softplus()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the BiLSTMModel."""
        x = self.forward_embedding(x)
        x = self.forward_head(x)

        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        """Processes features through attention and pooling to prepare for output.

        Args:
            x (torch.Tensor): Input tensor after initial embedding.
            pre_logits (bool): If True, return features before applying final classification layer.

        Returns:
            torch.Tensor: The processed tensor, either pre-logits or final predictions.
        """
        b = x.shape[0]

        # Processing relationships in each node of the montage
        x = self.node_attention(x)
        pf = F.softmax(self.pool_factor(x), dim=1)
        x = torch.sum(x * pf, dim=1).view(b, -1)

        if pre_logits:
            return x

        # Applying temperature scaling to logits
        t = self.temperature(x)
        logits = self.decoder(x)
        pred = F.log_softmax(logits / t, dim=-1)

        return pred

    def forward_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Embeds input using the encoder and reshapes for further processing.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The reshaped tensor after encoding.
        """
        b, c, l = x.shape
        x = x[:, :c-1]
        x = x.view(b, (c - 1), -1)
        x = x.reshape(b * (c - 1), 1, -1)

        # Downsample and embed
        x = self.encoder(x)
        x = x.view(b, c - 1, -1)
        return x
