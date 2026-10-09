import torch
import torch.nn as nn
from .base_har import BaseHARModel

class PositionalEncoding(nn.Module):
    """
    Applies learnable positional embeddings to preserve temporal order in sequence inputs.
    """
    def __init__(self, d_model, max_len=100):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        seq_len = x.size(1)
        return x + self.pos_embedding[:, :seq_len, :]

class TransformerHARModel(BaseHARModel):
    """
    Spatial-Temporal Transformer Encoder for Human Activity Recognition.
    Applies multi-head self-attention across frames and keypoint channels.
    """
    def __init__(self, input_dim=92, num_classes=8, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1, max_len=100):
        super().__init__()
        
        self.norm = nn.LayerNorm(input_dim)
        
        # 1. Feature Projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        # 3. Transformer Encoder Blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Classification Head
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )
        
    def forward(self, x):
        """
        Inputs:
            x (torch.Tensor): Sequence tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            logits (torch.Tensor): Class logits of shape (batch_size, num_classes)
        """
        # Apply Layer Normalization to input features
        x = self.norm(x)
        
        # Project inputs to d_model: [batch_size, seq_len, d_model]
        x_proj = self.input_projection(x)
        
        # Add Positional Encodings
        x_pos = self.pos_encoder(x_proj)
        
        # Pass through Transformer encoder: [batch_size, seq_len, d_model]
        encoder_out = self.transformer_encoder(x_pos)
        
        # Global average pooling over frames
        pooled = torch.mean(encoder_out, dim=1)  # shape: [batch_size, d_model]
        
        # Classification
        logits = self.fc(pooled)
        return logits
