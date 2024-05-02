import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import ConvBlock2D, ResBlock2D, NodeAttention

class ResNetModel(nn.Module):
    """
    A ResNet-like model for processing multi-dimensional data, enhanced with node attention for feature refinement.

    Attributes:
        layers (list): A list defining the number of output channels for each convolutional block.
        projection (ConvBlock2D): Initial convolutional block for projecting input into a higher-dimensional space.
        encoder (nn.Sequential): Sequence of residual blocks for deep feature extraction.
        pool (nn.AdaptiveAvgPool2d): Adaptive average pooling to reduce spatial dimensions to 1x1.
        node_attention (NodeAttention): Node attention mechanism to focus on specific features.
        pool_factor (nn.Linear): Linear layer to transform attention outputs for decision-making.
        decoder (nn.Linear): Decoder layer for classification.
        temperature (nn.Sequential): Network to compute a scaling factor for logits, aiding in numerical stability.
    """

    layers = [16, 32, 48, 64, 128, 256]

    def __init__(self, in_channels: int = 1):
        """Initializes the ResNetModel with specified input channels."""
        super(ResNetModel, self).__init__()

        embed_dim = self.layers[-1]  # Last layer's number of features

        self.projection = ConvBlock2D(in_channels, self.layers[0])
        self.encoder = nn.Sequential(
            ResBlock2D(self.layers[0], self.layers[1]),
            ResBlock2D(self.layers[1], self.layers[2]),
            ResBlock2D(self.layers[2], self.layers[3]),
            ResBlock2D(self.layers[3], self.layers[4]),
            ResBlock2D(self.layers[4], self.layers[5]),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.node_attention = NodeAttention(input_size=embed_dim, embed_dim=embed_dim, n_nodes=4)
        self.pool_factor = nn.Linear(embed_dim, 1)
        self.decoder = nn.Linear(embed_dim, 6)
        self.temperature = nn.Sequential(
            nn.Linear(embed_dim, 6),
            nn.Softplus()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the model."""
        x = self.forward_embedding(x)
        x = self.forward_head(x)

        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        """
        Processes the features through node attention and a decision layer, optionally returning pre-logit features.

        Args:
            x (torch.Tensor): Features after initial embedding.
            pre_logits (bool): If True, returns the features before applying final classification layers.

        Returns:
            torch.Tensor: The processed tensor, either pre-logits or final predictions.
        """
        b = x.shape[0]  # Batch size

        # Process features through node attention and weighted sum
        x = self.node_attention(x)
        pf = F.softmax(self.pool_factor(x), dim=1)
        x = torch.sum(x * pf, dim=1).view(b, -1)

        if pre_logits:
            return x

        # Apply temperature scaling and compute final predictions
        t = self.temperature(x)
        logits = self.decoder(x)
        pred = F.log_softmax(logits / t, dim=-1)

        return pred

    def forward_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embeds the input using a projection and a series of encoders followed by pooling.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Embedded tensor reshaped for subsequent processing.
        """
        b, c, h, w = x.shape  # Batch size, channels, height, width

        # Reshape and process through convolutional layers and pooling
        x = x.view(b * c, 1, h, w)
        x = self.projection(x)
        x = self.encoder(x)
        x = self.pool(x)
        x = x.view(b, c, -1)  # Reshape for node attention

        return x
