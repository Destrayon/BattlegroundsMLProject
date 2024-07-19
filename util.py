import torch.nn as nn
import torch
import numpy as np
from typing import Optional

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out
    
class ResNetStack(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResNetStack, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.resnet_block1 = ResNetBlock(out_channels, out_channels)
        self.resnet_block2 = ResNetBlock(out_channels, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.resnet_block1(x)
        x = self.resnet_block2(x)
        return x

class ResNetNetwork(nn.Module):
    def __init__(self, in_channels):
        super(ResNetNetwork, self).__init__()
        self.stack1 = ResNetStack(in_channels, 64)
        self.stack2 = ResNetStack(64, 128)
        self.stack3 = ResNetStack(128, 128)

    def forward(self, x):
        x = self.stack1(x)
        x = self.stack2(x)
        x = self.stack3(x)
        return x
    
class DenseProcessor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.dense1 = nn.Linear(input_size, 256)
        self.dense2 = nn.Linear(256, 4096)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.dense1(x))
        x = self.dense2(x)
        return x
    
class ResidualTransformerBlock(nn.Module):
    def __init__(self, d_model=4096, nhead=32, dim_feedforward=16384, dropout=0.1):
        super(ResidualTransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (b, t, d_model)
        # Unmasked self-attention
        attn_output, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_output)  # Residual connection
        x = self.norm1(x)
        
        # Feedforward network
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)  # Residual connection
        x = self.norm2(x)
        
        return x

class ResidualTransformerModel(nn.Module):
    def __init__(self, num_blocks=4, d_model=4096, nhead=32, dim_feedforward=16384, dropout=0.1):
        super(ResidualTransformerModel, self).__init__()
        self.blocks = nn.ModuleList([
            ResidualTransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        # x shape: (b, t, d_model)
        for block in self.blocks:
            x = block(x)
        return x

class ResidualTransformerModel(nn.Module):
    def __init__(self, num_blocks=4, d_model=4096, nhead=32, dim_feedforward=16384, dropout=0.1):
        super(ResidualTransformerModel, self).__init__()
        self.blocks = nn.ModuleList([
            ResidualTransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
    
class DenseResidualBlock(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=16384, output_dim=4096):
        super().__init__()
        self.dense1 = nn.Linear(input_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, output_dim)
        self.residual_proj = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        residual = self.residual_proj(x)
        out = self.activation(self.dense1(x))
        out = self.dense2(out)
        return out + residual

class FinalOutputLayer(nn.Module):
    def __init__(self, input_dim=4096):
        super().__init__()
        self.click_classifier = nn.Linear(input_dim, 1)
        self.mouse_position = nn.Linear(input_dim, 2)

    def forward(self, x):
        click = torch.sigmoid(self.click_classifier(x))
        mouse_pos = self.mouse_position(x)  # Linear activation (no activation function)
        return click, mouse_pos