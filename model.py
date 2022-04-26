import torch
import torch.nn as nn
import math
import numpy as np


class WIP(nn.Module):
    def __init__(self, hidden_size, num_users, num_items, dropout=0.2, padding_idx=-1, device="cpu"):
        super(WIP, self).__init__()
        self.hidden_size = hidden_size
        self.num_users = num_users
        self.num_items = num_items
        self.dropout = dropout
        self.padding_idx = padding_idx
        self.device = torch.device(device)

        self.build_layers()
        self = self.to(self.device)
    
    def build_layers(self):
        self.transformer = Transformer(self.num_items, self.hidden_size, 4, 3, 3, dropout=self.dropout, padding_idx=self.padding_idx)

    def forward(self, src, target, target_mask=None, src_pad_mask=None, target_pad_mask=None, memory=None, hist_sess=None, hist_sess_pad_mask=None, hist_sizes=None):
        return self.transformer(src, target, target_mask, src_pad_mask, target_pad_mask, memory, hist_sess, hist_sess_pad_mask, hist_sizes)


class Transformer(nn.Module):

    def __init__(self, num_tokens, dim_model, num_heads, num_encoder_layers, num_decoder_layers, dropout=0.1, padding_idx=-1):
        super(Transformer, self).__init__()
        
        self.dim_model = dim_model
        self.positional_encoding = PositionalEncoding(
            dim_model=dim_model,
            dropout=dropout,
            max_len=20
        )
        self.embedding = nn.Embedding(num_tokens, dim_model, padding_idx=padding_idx)
        # For faster inference, this change is made
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim_model, nhead=num_heads, dropout=dropout
            ),
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(dim_model, eps=1e-5)
        )
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=dim_model, nhead=num_heads, dropout=dropout
            ),
            num_layers=num_decoder_layers,
            norm=nn.LayerNorm(dim_model, eps=1e-5)
        )
        self.out = nn.Linear(dim_model, num_tokens)


    def getCurrentSessionEncoding(self, src, src_pad_mask):
        """
        Get encoder output on the current session
        Inputs :
            src : [BATCH, SEQ LEN]
            src_pad_mask : [BATCH, SEQ LEN]
        Returns :
            encoder_output : [SEQ LEN, BATCH SIZE, DIM MODEL]
        """
        src = self.embedding(src) * math.sqrt(self.dim_model)
        src = src.permute(1, 0, 2)
        src = self.positional_encoding(src)
        encoder_output = self.transformer_encoder(src, src_key_padding_mask=src_pad_mask)
        return encoder_output


    def getHistorySessionEncoding(self, hist_sess, hist_sess_pad_mask, hist_sizes=None):
        """
        Get encoder output on the history sessions
        Inputs :
            hist_sess : [*, SEQ LEN]
            hist_sess_pad_mask : [*, SEQ LEN]
            hist_sizes : [BATCH,]
            * : sum(hist_sizes)
        Returns :
            encoder_output : [SEQ LEN, *, DIM MODEL]
        """
        hist_sess = self.embedding(hist_sess) * math.sqrt(self.dim_model)
        hist_sess = hist_sess.permute(1, 0, 2)
        hist_sess = self.positional_encoding(hist_sess)
        encoder_output = self.transformer_encoder(hist_sess, src_key_padding_mask=hist_sess_pad_mask)
        return encoder_output

    
    def getHistSessionSimilarity(self, current_encoder_output, hist_encoder_output, hist_sizes):
        """
        Get encoder output on the history sessions
        Inputs :
            current_encoder_output : [SEQ LEN, BATCH SIZE, DIM MODEL]
            hist_encoder_output : [SEQ LEN, *, DIM MODEL]
            hist_sizes : [BATCH,]
            * : sum(hist_sizes)
        Returns :
            item_closeness : [*, SEQ LEN]
            similarities : [*, ]
        """ 
        current_encoder_output = current_encoder_output.permute(1, 0, 2)
        hist_encoder_output = hist_encoder_output.permute(1, 0, 2)
        current_encoder_output = torch.repeat_interleave(current_encoder_output, hist_sizes, dim=0)
        current_encoder_output = torch.repeat_interleave(current_encoder_output, hist_sizes, dim=0)
        item_closeness = torch.sum(current_encoder_output * hist_encoder_output, dim=-1) # [*, SEQ LEN]
        similarities, max_inds = item_closeness.max(dim=1) # [*]
        # TODO : Optimize
        current = 0
        for size in hist_sizes:
            similarities[current:current+size] = nn.functional.softmax(similarities[current:current+size], dim=0)
            current += size
        item_closeness = nn.functional.softmax(item_closeness, dim=1)
        return item_closeness, similarities


    def getHistSessionInfluence(self, current_encoder_output, hist_encoder_output, hist_sizes, item_closeness, similarities):
        """
        Get encoder output on the history sessions
        Inputs :
            current_encoder_output : [SEQ LEN, BATCH SIZE, DIM MODEL]
            hist_encoder_output : [SEQ LEN, *, DIM MODEL]
            hist_sizes : [BATCH,]
            item_closeness : [*, SEQ LEN]
            similarities : [*, ]
            * : sum(hist_sizes)
        Returns :
            current_encoder_output : [SEQ LEN, BATCH SIZE, DIM MODEL]
        """
        weighed_items = item_closeness.unsqueeze(2) * hist_encoder_output.permute(1, 0, 2)
        weighed_session = weighed_items.sum(dim=1) # [*, DIM MODEL]
        similarity_weighted_session = similarities.unsqueeze(1) * weighed_session # [*, DIM MODEL]
        cumsum = torch.cumsum(similarity_weighted_session, dim=0)
        inds = torch.cumsum(hist_sizes, dim=0) - 1
        result = torch.mul(cumsum[inds], (inds != -1).int().unsqueeze(1))
        result = torch.cat((torch.zeros_like(result[:1,:]), result), dim=0).diff(dim=0) # [BATCH, DIM MODEL]
        # TODO : Experiment other combinations
        # return current_encoder_output + result
        return result

    
    def getTargetDecoding(self, target, memory, tgt_mask=None, tgt_key_padding_mask=None):
        """
        Get encoder output on the current session
        Inputs :
            target : [BATCH, SEQ LEN]
            memory : [SEQ LEN, BATCH SIZE, DIM MODEL]
            tgt_pad_mask : [BATCH, SEQ LEN]
            tgt_mask : [SEQ LEN, SEQ LEN]
        Returns :
            decoder_output : [SEQ LEN, BATCH SIZE, DIM MODEL]
        """
        target = self.embedding(target) * math.sqrt(self.dim_model)
        target = target.permute(1, 0, 2)
        target = self.positional_encoding(target)
        decoder_output = self.transformer_decoder(target, memory, tgt_mask=target_mask, tgt_key_padding_mask=target_pad_mask)
        return decoder_output


    def forward(self, src, target, target_mask=None, src_pad_mask=None, target_pad_mask=None, memory=None, hist_sess=None, hist_sess_pad_mask=None, hist_sizes=None):
        if self.training:
            encoder_output = self.getCurrentSessionEncoding(src=src, src_pad_mask=src_pad_mask)
            hist_encoder_output = self.getHistorySessionEncoding(hist_sess=hist_sess, hist_sess_pad_mask=hist_sess_pad_mask, hist_sizes=hist_sizes)
            item_closeness, similarities = self.getHistSessionSimilarity(encoder_output, hist_encoder_output, hist_sizes)
            encoder_output = self.getHistSessionInfluence(encoder_output, hist_encoder_output, hist_sizes, item_closeness, similarities)

            decoder_output = self.getTargetDecoding(target, encoder_output, tgt_mask=target_mask, tgt_key_padding_mask=target_pad_mask)
            out = self.out(decoder_output)
            return out

        else:
            if memory == None:
                encoder_output = self.getCurrentSessionEncoding(src=src, src_pad_mask=src_pad_mask)
                hist_encoder_output = self.getHistorySessionEncoding(hist_sess=hist_sess, hist_sess_pad_mask=hist_sess_pad_mask, hist_sizes=hist_sizes)
                item_closeness, similarities = self.getHistSessionSimilarity(encoder_output, hist_encoder_output, hist_sizes)
                memory = self.getHistSessionInfluence(encoder_output, hist_encoder_output, hist_sizes, item_closeness, similarities)
            
            decoder_output = self.getTargetDecoding(target, encoder_output, tgt_mask=target_mask, tgt_key_padding_mask=target_pad_mask)
            out = self.out(decoder_output)
            return out, memory


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