# Copyright (C) Alibaba Group Holding Limited. 

import torch
import torch.nn as nn
import copy


__all__ = ['Transformer']

class TransformerEncoderLayer(nn.Module):

    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, pos):
        q = k = x + pos; v = x
        x = self.norm1(x + self.dropout1(self.self_attn(q, k, v)[0]))
        x = self.norm2(x + self.dropout3(self.fc2(self.dropout2(self.relu(self.fc1(x))))))
        return x


class TransformerDecoderLayer(nn.Module):
    
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout3 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, memory, q_pos, k_pos):
        # self-attention
        q = k = x + q_pos; v = x
        x = self.norm1(x + self.dropout1(self.self_attn(q, k, v)[0]))

        # encoder-decoder attention
        q = x + q_pos; k = memory + k_pos; v = memory
        x = self.norm2(x + self.dropout2(self.cross_attn(q, k, v)[0]))

        # feed-forward neural networks
        x = self.norm3(x + self.dropout4(self.fc2(self.dropout3(self.relu(self.fc1(x))))))

        return x


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(
            encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
    
    def forward(self, x, pos):
        for layer in self.layers:
            x = layer(x, pos)
        return x


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(
            decoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
    
    def forward(self, x, memory, q_pos, k_pos):
        for layer in self.layers:
            x = layer(x, memory, q_pos, k_pos)
        return x


class Transformer(nn.Module):

    def __init__(self,
                 hidden_dim=512,
                 num_heads=8,
                 num_encoder_layers=2,
                 num_decoder_layers=4,
                 dropout=0.1):
        super(Transformer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.encoder = TransformerEncoder(
            TransformerEncoderLayer(hidden_dim, num_heads, dropout),
            num_encoder_layers)
        self.decoder = TransformerDecoder(
            TransformerDecoderLayer(hidden_dim, num_heads, dropout),
            num_decoder_layers)
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, memory, q_pos, k_pos):
        memory = self.encoder(memory, k_pos)
        x = torch.zeros_like(q_pos)
        x = self.decoder(x, memory, q_pos, k_pos)
        return x
