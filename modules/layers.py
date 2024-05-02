import torch
import torch.nn as nn
import torch.nn.functional as F

def ConvBlock1D(in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: str = "same", **kwargs) -> nn.Module:
    """Creates a 1D convolutional block with batch normalization and ReLU activation.

    Args:
        in_channels (int): Number of channels in the input.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolving kernel.
        stride (int): Stride of the convolution.
        padding (str): Padding added to both sides of the input.

    Returns:
        nn.Module: Sequential module consisting of Conv1d, BatchNorm1d, and ReLU.
    """
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False, **kwargs),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True)
    )

def ConvBlock2D(in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: str = "same", **kwargs) -> nn.Module:
    """Creates a 2D convolutional block with batch normalization and ReLU activation.

    Args:
        in_channels (int): Number of channels in the input.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolving kernel.
        stride (int): Stride of the convolution.
        padding (str): Padding added to both sides of the input.

    Returns:
        nn.Module: Sequential module consisting of Conv2d, BatchNorm2d, and ReLU.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, **kwargs),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class ResBlock1D(nn.Module):
    """A residual block for 1D inputs using convolutional layers, batch normalization, and ReLU activations.

    Attributes:
        relu (nn.ReLU): ReLU activation function.
        maxpool (nn.MaxPool1d): Max pooling layer.
        downsample (nn.Sequential): Downsample layer sequence.
        dropout (nn.Dropout1d): Dropout layer for regularization.
        conv1, conv2 (nn.Conv1d): Convolutional layers.
        bn1, bn2 (nn.BatchNorm1d): Batch normalization layers.
    """
    def __init__(self, in_channels: int, feature_maps: int, kernel_size: int = 3, padding: int = 1, bias: bool = False, **kwargs) -> None:
        super(ResBlock1D, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=2, ceil_mode=True)
        self.downsample = nn.Sequential(
            nn.Conv1d(in_channels, feature_maps, kernel_size=1, bias=False),
            nn.BatchNorm1d(feature_maps)
        )
        self.dropout = nn.Dropout1d(p=0.4)
        self.conv1 = nn.Conv1d(in_channels, feature_maps, kernel_size, padding=padding, bias=bias, **kwargs)
        self.bn1 = nn.BatchNorm1d(feature_maps)
        self.conv2 = nn.Conv1d(feature_maps, feature_maps, kernel_size, padding=padding, bias=bias, **kwargs)
        self.bn2 = nn.BatchNorm1d(feature_maps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + self.downsample(identity)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

class ResBlock2D(nn.Module):
    """A residual block for 2D inputs using convolutional layers, batch normalization, and ReLU activations.

    Attributes:
        relu (nn.ReLU): ReLU activation function.
        maxpool (nn.MaxPool2d): Max pooling layer.
        downsample (nn.Sequential): Downsample layer sequence.
        dropout (nn.Dropout2d): Dropout layer for regularization.
        conv1, conv2 (nn.Conv2d): Convolutional layers.
        bn1, bn2 (nn.BatchNorm2d): Batch normalization layers.
    """
    def __init__(self, in_channels: int, feature_maps: int, kernel_size: int = 3, padding: int = 1, bias: bool = False, **kwargs) -> None:
        super(ResBlock2D, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, ceil_mode=True)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, feature_maps, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_maps)
        )
        self.dropout = nn.Dropout2d(p=0.4)
        self.conv1 = nn.Conv2d(in_channels, feature_maps, kernel_size, padding=padding, bias=bias, **kwargs)
        self.bn1 = nn.BatchNorm2d(feature_maps)
        self.conv2 = nn.Conv2d(feature_maps, feature_maps, kernel_size, padding=padding, bias=bias, **kwargs)
        self.bn2 = nn.BatchNorm2d(feature_maps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + self.downsample(identity)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

class LayerNorm(nn.Module):
    """Applies layer normalization.

    Args:
        ndim (int): Number of dimensions for the normalization.
        bias (bool, optional): If True, add a learnable bias to the output. Defaults to True.
        eps (float, optional): A value added to the denominator for numerical stability. Defaults to 1e-5.

    Attributes:
        weight (nn.Parameter): The learnable weights of the module.
        bias (nn.Parameter): The learnable bias of the module.
        eps (float): A value added for numerical stability.
    """
    def __init__(self, ndim: int, bias: bool = True, eps: float = 1e-5) -> None:
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)

class LPE(nn.Module):
    """Module for learnable positional embeddings (LPE).

    Args:
        n_nodes (int, optional): Number of nodes in the embedding. Defaults to 16.
        embed_dim (int, optional): Dimensionality of embeddings. Defaults to 32.

    Attributes:
        embedding (nn.Embedding): Embedding layer.
        pos (nn.Parameter): Positions for the embeddings.
    """
    def __init__(self, n_nodes: int = 16, embed_dim: int = 32) -> None:
        super(LPE, self).__init__()
        self.embedding = nn.Embedding(n_nodes, embed_dim)
        self.pos = nn.Parameter(torch.arange(n_nodes).unsqueeze(0), requires_grad=False)

    def forward(self) -> torch.Tensor:
        return self.embedding(self.pos)

class AttentionBlock(nn.Module):
    """Attention block for transformer-based architectures.

    Args:
        input_size (int): Size of each input sample.
        embed_dim (int, optional): Dimensionality of the output space. Defaults to 128.
        num_heads (int, optional): Number of heads in the multiheadattention. Defaults to 2.
        n_nodes (int, optional): Number of positional nodes. Defaults to 19.

    Attributes:
        projection (nn.Linear): Projects the input into the embedding space.
        pos_embeddings (LearnablePosEmbeddings): Positional embeddings.
        attention (nn.MultiheadAttention): Multi-head attention mechanism.
        fc (nn.Sequential): Fully connected layer after attention.
        ln_0, ln_1 (LayerNorm): Layer normalization.
    """
    def __init__(self, input_size: int, embed_dim: int = 128, num_heads: int = 2, n_nodes: int = 19) -> None:
        super(AttentionBlock, self).__init__()
        self.projection = nn.Linear(input_size, embed_dim)
        self.pos_embeddings = LPE(n_nodes, embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.ln_0 = LayerNorm(embed_dim)
        self.ln_1 = LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        x = x + self.pos_embeddings()
        x = self.ln_0(x)
        x = x + self.attention(x, x, x, need_weights=False)[0]
        x = self.ln_1(x)
        x = x + self.fc(x)
        return x

class NodeAttention(nn.Module):
    """Multi-layer attention block for node-based architectures.

    Args:
        input_size (int): Size of each input sample.
        embed_dim (int): Dimensionality of the output space.
        num_layers (int, optional): Number of attention layers to apply. Defaults to 2.
        num_heads (int, optional): Number of heads in the multiheadattention. Defaults to 2.
        n_nodes (int, optional): Number of positional nodes. Defaults to 19.

    Attributes:
        attention_layers (nn.ModuleList): List of attention blocks.
    """
    def __init__(self, input_size: int, embed_dim: int, num_layers: int = 2, num_heads: int = 2, n_nodes: int = 19) -> None:
        super(NodeAttention, self).__init__()
        self.attention_layers = nn.ModuleList([
            AttentionBlock(input_size, embed_dim, num_heads, n_nodes) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(self.num_layers):
            x = self.attention_layers[i](x)
        return x

class BiLSTMBlock(nn.Module):
    """Bidirectional LSTM block for sequence modeling.

    Args:
        input_size (int): The number of expected features in the input `x`.
        embed_dim (int, optional): The number of features in the hidden state `h`. Defaults to 32.
        num_layers (int, optional): Number of recurrent layers. Defaults to 1.

    Attributes:
        lstm (nn.LSTM): LSTM layer.
    """
    def __init__(self, input_size: int, embed_dim: int = 32, num_layers: int = 1) -> None:
        super(BiLSTMBlock, this).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=embed_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # Reshape x to (batch, sequence, feature)
        output, (hn, cn) = self.lstm(x)
        output = output[:, :, :self.lstm.hidden_size] + output[:, :, self.lstm.hidden_size:]  # Sum the outputs of the bidirectional LSTM
        return output[:, -1, :]  # Return the last sequence element

class WaveBlock(nn.Module):
    """
    A building block for the WaveNet architecture utilizing dilated convolutions.

    Attributes:
        num_rates (int): Number of dilation levels.
        convs (nn.ModuleList): List of convolutional layers.
        filter_convs (nn.ModuleList): List of dilated convolutional layers for filter.
        gate_convs (nn.ModuleList): List of dilated convolutional layers for gate.
    """
    def __init__(self, in_channels: int, out_channels: int, dilation_rates: int, kernel_size: int = 3):
        """
        Initializes the Wave_Block module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            dilation_rates (int): Number of dilation levels.
            kernel_size (int): Size of the kernel for convolution operations.
        """
        super(WaveBlock, self).__init__()
        self.num_rates = dilation_rates
        self.convs = nn.ModuleList([nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True)])
        dilation_rates = [2 ** i for i in range(dilation_rates)]

        for dilation_rate in dilation_rates:
            self.filter_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                          padding=int((dilation_rate * (kernel_size - 1)) / 2), dilation=dilation_rate))
            self.gate_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                          padding=int((dilation_rate * (kernel_size - 1)) / 2), dilation=dilation_rate))
            self.convs.append(nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=True))

        self.initialize_weights()

    def initialize_weights(self):
        """Initializes weights of the convolution layers using Xavier uniform distribution."""
        for conv in self.convs:
            nn.init.xavier_uniform_(conv.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(conv.bias)

        for filter_conv, gate_conv in zip(self.filter_convs, self.gate_convs):
            nn.init.xavier_uniform_(filter_conv.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(filter_conv.bias)
            nn.init.xavier_uniform_(gate_conv.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(gate_conv.bias)

    def forward(self, x):
        """Forward pass of the Wave_Block."""
        x = self.convs[0](x)
        res = x
        for i in range(self.num_rates):
            tanh_out = torch.tanh(self.filter_convs[i](x))
            sigmoid_out = torch.sigmoid(self.gate_convs[i](x))
            x = tanh_out * sigmoid_out
            x = self.convs[i + 1](x)
            res += x
        return res
