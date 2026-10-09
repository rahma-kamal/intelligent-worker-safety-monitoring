import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from .base_har import BaseHARModel

class Chomp1d(nn.Module):
    """
    Crops the padding from the end of a 1D convolution to make it causal.
    """
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    """
    A single residual block in TCN, containing causal, dilated convolutions,
    weight normalization, ReLU, and dropout.
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        # First layer
        self.conv1 = weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Second layer
        self.conv2 = weight_norm(nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        # Combine into sequential block
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        # Downsample connection if input/output channels differ
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNHARModel(BaseHARModel):
    """
    Temporal Convolutional Network (TCN) for Human Activity Recognition.
    Processes sequences in parallel using 1D dilated causal convolutions.
    """
    def __init__(self, input_dim=92, num_classes=8, num_channels=[64, 128, 128], kernel_size=3, dropout=0.2):
        super().__init__()
        
        self.norm = nn.LayerNorm(input_dim)
        
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            # Padding keeps size consistent, then chomp trims extra elements at end
            padding_val = (kernel_size - 1) * dilation_size
            
            layers.append(
                TemporalBlock(
                    in_channels, out_channels, kernel_size,
                    stride=1, dilation=dilation_size,
                    padding=padding_val, dropout=dropout
                )
            )
            
        self.tcn = nn.Sequential(*layers)
        
        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(num_channels[-1], 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        Inputs:
            x (torch.Tensor): Tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            logits (torch.Tensor): Class logits of shape (batch_size, num_classes)
        """
        # Apply Layer Normalization to input features
        x = self.norm(x)
        
        # nn.Conv1D expects shape (batch_size, channels/features, seq_len)
        # Transpose input: [batch_size, seq_len, input_dim] -> [batch_size, input_dim, seq_len]
        x_t = x.transpose(1, 2)
        
        # Pass through residual temporal blocks
        tcn_out = self.tcn(x_t)  # shape: [batch_size, num_channels[-1], seq_len]
        
        # Extract features from the final temporal frame
        final_frame_feat = tcn_out[:, :, -1]  # shape: [batch_size, num_channels[-1]]
        
        logits = self.fc(final_frame_feat)
        return logits
