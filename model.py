import torch
import torch.nn as nn
import math
import numpy as np

class Transformer4Rec(nn.Module):
    def __init__(self, num_users, num_items, dim_model=32, nhead=2, num_encoder_layers=2, layer_norm_eps=1e-05, dropout=0.2, padding_idx=-1, maxseqlen=20, device="cpu"):
        super(Transformer4Rec, self).__init__()
        self.dim_model = dim_model
        self.num_users = num_users
        self.num_items = num_items
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.layer_norm_eps = layer_norm_eps
        self.dropout = dropout
        self.padding_idx = padding_idx
        self.maxseqlen = maxseqlen
        self.device = torch.device(device)

        self.build_layers()
        self = self.to(self.device)
    

    def build_layers(self):
        self.positional_encoding = PositionalEncoding(
            dim_model=self.dim_model,
            dropout=self.dropout,
            max_len=self.maxseqlen
        )
        self.embedding = nn.Embedding(self.num_items, self.dim_model, padding_idx=self.padding_idx)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.dim_model, nhead=self.nhead, dropout=self.dropout
            ),
            num_layers=self.num_encoder_layers,
            norm=nn.LayerNorm(self.dim_model, eps=self.layer_norm_eps)
        )
        self.out = nn.Linear(self.dim_model, self.num_items)


    def getSourceEncoding(self, src, src_key_mask=None):
        sess = self.embedding(src) * math.sqrt(self.dim_model)
        sess = sess.permute(1, 0, 2)
        sess = self.positional_encoding(sess)
        encoder_output = self.transformer_encoder(sess, src_key_padding_mask=src_key_mask)
        return encoder_output

    def forward(self, curr_sess, curr_sess_lens, key_mask):
        encoding = self.getSourceEncoding(curr_sess, src_key_mask=key_mask) # [SEQ LEN, BATCH, DIM MODEL]
        out = self.out(encoding)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout=0.1, max_len=20):
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2) * (-math.log(10000.0) / dim_model))
        pe = torch.zeros(max_len, 1, dim_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


def load_model_checkpoint(model, checkpoint):
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model