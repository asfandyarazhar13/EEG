import torch
import torch.nn as nn

from layers import WaveBlock


class WaveNet(nn.Module):
    """
    WaveNet model for generating raw audio waveforms.

    Attributes:
        model (nn.Sequential): Sequential container of Wave_Block modules.
    """
    def __init__(self, input_channels: int = 1, kernel_size: int = 3):
        """
        Initializes the WaveNet model.

        Args:
            input_channels (int): Number of input channels.
            kernel_size (int): Size of the kernel for convolution operations.
        """
        super(WaveNet, self).__init__()
        self.model = nn.Sequential(
            WaveBlock(input_channels, 8, 12, kernel_size),
            WaveBlock(8, 16, 8, kernel_size),
            WaveBlock(16, 32, 4, kernel_size),
            WaveBlock(32, 64, 1, kernel_size)
        )

    def forward(self, x):
        """Forward pass of the WaveNet model."""
        x = x.permute(0, 2, 1)  # Change the dimension order for processing
        output = self.model(x)
        return output

class WaveNetModel(nn.Module):
    """
    Model leveraging WaveNet for audio processing.

    Attributes:
        model (WaveNet): Instance of the WaveNet model.
        global_avg_pooling (nn.AdaptiveAvgPool1d): Global average pooling layer.
        dropout (float): Dropout rate.
        head (nn.Sequential): Sequential container for the final classification layers.
    """
    def __init__(self):
        """Initializes the WaveNetModel."""
        super(CustomModel, self).__init__()
        self.model = WaveNet()
        self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.dropout = 0.0
        self.head = nn.Sequential(
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 6)
        )

    def forward(self, x):
        """Forward pass of the WaveNetModel."""
        # The following lines process different channels of the input and combine their results.
        x1 = self.model(x[:, :, 0:1])
        x1 = self.global_avg_pooling(x1)
        x1 = x1.squeeze()
        x2 = self.model(x[:, :, 1:2])
        x2 = self.global_avg_pooling(x2)
        x2 = x2.squeeze()
        z1 = torch.mean(torch.stack([x1, x2]), dim=0)

        x1 = self.model(x[:, :, 2:3])
        x1 = self.global_avg_pooling(x1)
        x1 = x1.squeeze()
        x2 = self.model(x[:, :, 3:4])
        x2 = self.global_avg_pooling(x2)
        x2 = x2.squeeze()
        z2 = torch.mean(torch.stack([x1, x2]), dim=0)

        x1 = self.model(x[:, :, 4:5])
        x1 = self.global_avg_pooling(x1)
        x1 = x1.squeeze()
        x2 = self.model(x[:, :, 5:6])
        x2 = self.global_avg_pooling(x2)
        x2 = x2.squeeze()
        z3 = torch.mean(torch.stack([x1, x2]), dim=0)

        x1 = self.model(x[:, :, 6:7])
        x1 = self.global_avg_pooling(x1)
        x1 = x1.squeeze()
        x2 = self.model(x[:, :, 7:8])
        x2 = self.global_avg_pooling(x2)
        x2 = x2.squeeze()
        z4 = torch.mean(torch.stack([x1, x2]), dim=0)

        y = torch.cat([z1, z2, z3, z4], dim=1)
        y = self.head(y)

        return y
