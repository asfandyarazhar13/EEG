import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser
from torch.utils.data import DataLoader

from modules.bilstm import BiLSTMModel
from modules.cnn import ResNetModel
from modules.gnn import GNNModel
from modules.wavenet import WaveNetModel


class Decoder(nn.Module):
    """
    Decoder module with sample-dependent adaptive temperature scaling.

    The module takes an input feature tensor and applies two separate linear transformations
    to produce logits for classification and a temperature scaling factor. The temperature
    is used to scale the logits before applying softmax to obtain the final class probabilities.
    """

    def __init__(self, input_dim, num_classes):
        """
        Initializes the Decoder module.

        Args:
            input_dim (int): The number of input features.
            num_classes (int): The number of classes for classification.
        """
        super(Decoder, self).__init__()
        self.logits_layer = nn.Linear(input_dim, num_classes)
        self.temperature_layer = nn.Linear(input_dim, 1)

    def forward(self, x):
        """
        Forward pass for the Decoder module.

        Args:
            x (torch.Tensor): The input feature tensor.

        Returns:
            torch.Tensor: The output probabilities for each class.
        """
        logits = self.logits_layer(x)
        temperature = torch.exp(self.temperature_layer(x))  # Ensuring temperature is positive
        scaled_logits = logits / temperature
        probabilities = F.log_softmax(scaled_logits, dim=1)  # Use log_softmax for KLD
        return probabilities


class BigWarModel(pl.LightningModule):
    """
    PyTorch Lightning module for a multimodal architecture that processes inputs from
    different model backbones and combines their features using attention pooling
    followed by a decoder.

    Attributes are instantiated for BiLSTM, CNN, GNN, and WaveNet models, along with
    attention pooling and a final decoder module. The backbones of individual models
    are frozen before attention pooling.
    """

    def __init__(self, input_dims, hidden_dim, num_classes, sequence_length, num_heads, num_encoder_layers, dropout_rate):
        """
        Initializes the multimodal model with separate model backbones and attention pooling.

        Args:
            input_dims (dict): A dictionary with input dimension configurations for all models.
            hidden_dim (int): The hidden dimension size.
            num_classes (int): The number of output classes.
            sequence_length (int): The sequence length for BiLSTM model input.
            num_heads (int): The number of heads for the multi-head attention mechanism.
            num_encoder_layers (int): The number of layers in the transformer encoder.
            dropout_rate (float): Dropout rate for the attention mechanism.
        """
        super().__init__()
        self.save_hyperparameters()

        self.bilstm_model = BiLSTMModel(*input_dims['bilstm'], sequence_length, num_heads, num_encoder_layers, dropout_rate)
        self.resnet_model = ResNetModel(*input_dims['cnn'], num_heads, num_encoder_layers, dropout_rate)
        self.gnn_model = GNNModel(*input_dims['gnn'])
        self.wavenet_model = WaveNetModel(*input_dims['wavenet'])

        self.concat_dim = 30
        self.fc_concat = nn.Linear(hidden_dim * 4, self.concat_dim * hidden_dim)

        self.attention_pooling = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout_rate)

        self.decoder = Decoder(hidden_dim, num_classes)

        self.freeze_backbones()

    def freeze_backbones(self):
        """
        Freezes the parameters in the backbones of each model to prevent them
        from being updated during training.
        """
        for model in [self.bilstm_model, self.cnn_model, self.gnn_model, self.wavenet_model]:
            for param in model.parameters():
                param.requires_grad = False

    def forward(self, bilstm_input, cnn_input, gnn_input, wavenet_input):
        """
        Forward pass for the model, which processes inputs through each individual model backbone,
        concatenates their outputs, applies attention pooling, and finally passes the result
        through a decoder.

        Args:
            bilstm_input (torch.Tensor): Input tensor for the BiLSTM model.
            cnn_input (torch.Tensor): Input tensor for the CNN model.
            gnn_input (torch.Tensor): Input tensor for the GNN model.
            wavenet_input (torch.Tensor): Input tensor for the WaveNet model.

        Returns:
            torch.Tensor: The final output probabilities from the model.
        """
        # Process inputs through each model's backbone
        bilstm_output = self.bilstm_model(bilstm_input)
        cnn_output = self.resnet_model(cnn_input)
        gnn_output = self.gnn_model(gnn_input)
        wavenet_output = self.wavenet_model(wavenet_input)

        # Concatenate outputs from all models and match dimensions for attention pooling
        combined = torch.cat((bilstm_output, cnn_output, gnn_output, wavenet_output), dim=1)
        combined = self.fc_concat(combined)
        combined = combined.view(-1, self.concat_dim, 256)

        # Permute and apply attention pooling
        combined = combined.permute(1, 0, 2)
        attn_output, _ = self.attention_pooling(combined, combined, combined)
        attn_output = attn_output.permute(1, 0, 2)

        # Pass through decoder for classification
        probabilities = self.decoder(attn_output.mean(dim=1))
        return probabilities

    def training_step(self, batch, batch_idx):
        """
        Training step for the model where the Kullback-Leibler Divergence (KLD) loss is computed.

        Args:
            batch: The batch of data provided by the DataLoader.
            batch_idx (int): The index of the batch.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        bilstm_input, resnet_input, gnn_input, wavenet_input, labels = batch
        probabilities = self.forward(bilstm_input, cnn_input, gnn_input, wavenet_input)

        # Compute KLD loss
        loss = F.kl_div(probabilities, labels, reduction='batchmean')
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        """
        Configures the optimizers for the model.

        Returns:
            torch.optim.Optimizer: The Adam optimizer with a learning rate of 1e-3.
        """
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=1e-3)
        return optimizer

# Define a function to create the model using parsed arguments
def create_model_from_args(args):
    """
    Creates the multimodal model using configurations parsed from command-line arguments.

    Args:
        args: Command-line arguments.

    Returns:
        BigWarModel: The instantiated model.
    """
    input_dims = {
        'bilstm': [args.bilstm_input_channels, args.hidden_dim, args.num_classes],
        'resnet': [args.resnet_num_channels, args.num_classes, args.num_heads, args.num_encoder_layers, args.dropout_rate],
        'gnn': [args.gnn_in_channels, args.gnn_num_conv_layers, args.gnn_hid_channels, args.num_classes],
        'wavenet': [args.wavenet_input_channels, args.kernel_size]
    }

    model = BigWarModel(
        input_dims=input_dims,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        sequence_length=args.sequence_length,
        num_heads=args.num_heads,
        num_encoder_layers=args.num_encoder_layers,
        dropout_rate=args.dropout_rate
    )
    return model

def main():
    """
    Main function for the script which parses command-line arguments, creates the model,
    and starts the training process.
    """
    parser = ArgumentParser(description='Multimodal model training script.')

    # Add model configuration arguments
    parser = ArgumentParser()
    parser.add_argument('--bilstm_input_channels', type=int, default=18, help='Input channels for BiLSTM')
    parser.add_argument('--resnet_num_channels', type=int, default=1, help='Number of channels for ResNet')
    parser.add_argument('--gnn_in_channels', type=int, default=10000, help='Input channels for GNN')
    parser.add_argument('--wavenet_input_channels', type=int, default=1, help='Input channels for WaveNet')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension for the models')
    parser.add_argument('--num_classes', type=int, default=6, help='Number of classes')
    parser.add_argument('--sequence_length', type=int, default=100, help='Sequence length for BiLSTM')
    parser.add_argument('--num_heads', type=int, default=2, help='Number of attention heads')
    parser.add_argument('--num_encoder_layers', type=int, default=2, help='Number of encoder layers in transformer')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size for WaveNet')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--max_epochs', type=int, default=10, help='Maximum number of epochs')
    
    args = parser.parse_args()

    model = create_model_from_args(args)

    # Uncomment the following lines and replace with custom DataLoader and DataModule
    # train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    # data_module = CustomDataModule()  # Replace with custom actual DataModule

    # Instantiate the PyTorch Lightning trainer and train the model
    trainer = pl.Trainer(max_epochs=args.max_epochs)
    # trainer.fit(model, train_loader)  # Uncomment and replace with custom train_loader
    # trainer.fit(model, data_module)  # Uncomment and replace with custom DataModule

if __name__ == '__main__':
    main()
