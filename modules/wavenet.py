import torch
import torch.nn as nn
import argparse

class Wave_Block(nn.Module):
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
        super(Wave_Block, self).__init__()
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
            Wave_Block(input_channels, 8, 12, kernel_size),
            Wave_Block(8, 16, 8, kernel_size),
            Wave_Block(16, 32, 4, kernel_size),
            Wave_Block(32, 64, 1, kernel_size)
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
        """Initializes the CustomModel."""
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
        """Forward pass of the CustomModel."""
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

def get_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument('--input_channels', type=int, default=1, help='Number of input channels')
    parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size for convolutions')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    model = CustomModel()
    print("Model initialized with input channels: {} and kernel size: {}".format(args.input_channels, args.kernel_size))

if __name__ == "__main__":
    main()
