""" Embeddings module from ONMT"""
import math
import warnings
import numpy as np

import torch
import torch.nn as nn


def future_mask(seq_length):
    future_mask = np.triu(np.ones((1, seq_length, seq_length)), k=0).astype("bool")
    return torch.from_numpy(future_mask)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for non-recurrent neural networks.
    Args:
       dropout (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, dropout, dim, max_len=5000):
        if dim % 2 != 0:
            raise ValueError(
                "Cannot use sin/cos positional encoding with "
                "odd dim (got dim={:d})".format(dim)
            )
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            (torch.arange(0, dim, 2, dtype=torch.float) * -(math.log(10000.0) / dim))
        )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)
        super(PositionalEncoding, self).__init__()
        self.register_buffer("pe", pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb):
        """Embed interactions.
        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(seq_len, batch_size, self.dim)``
        """

        # emb = emb * math.sqrt(self.dim)
        emb = emb + self.pe[: emb.size(0)]
        emb = self.dropout(emb)
        return emb

""" Multi-Head Attention module from ONMT"""


class MultiHeadedAttention(nn.Module):
    def __init__(self, head_count, model_dim, dropout=0.1):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim, head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None, layer_cache=None):
        """
        Compute the context vector and the attention vectors.
        Args:
           key (FloatTensor): set of `key_len`
               key vectors ``(batch, key_len, dim)``
           value (FloatTensor): set of `key_len`
               value vectors ``(batch, key_len, dim)``
           query (FloatTensor): set of `query_len`
               query vectors  ``(batch, query_len, dim)``
           mask: binary mask 1/0 indicating which keys have
               zero / non-zero attention ``(batch, query_len, key_len)``
        Returns:
           (FloatTensor, FloatTensor):
           * output context vectors ``(batch, query_len, dim)``
           * one of the attention vectors ``(batch, query_len, key_len)``
        """
        # CHECKS
        batch, k_len, d = key.size()
        batch_, k_len_, d_ = value.size()
        assert batch_ == batch
        assert k_len == k_len
        assert d == d_
        batch_, q_len, d_ = query.size()
        assert batch_ == batch
        assert d == d_

        # aeq(self.model_dim % 8, 0)
        if mask is not None:
            batch_, q_len_, k_len_ = mask.size()
            # assert batch_ == batch mask will be broadcasted
            assert k_len_ == k_len
            assert q_len_ == q_len
        # END CHECKS
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)

        def shape(x):
            """Projection."""
            return x.view(batch_size, -1, head_count, dim_per_head).transpose(1, 2)

        def unshape(x):
            """Compute context."""
            return (
                x.transpose(1, 2)
                .contiguous()
                .view(batch_size, -1, head_count * dim_per_head)
            )

        # 1) Project key, value, and query.
        key = self.linear_keys(key)
        value = self.linear_values(value)
        query = self.linear_query(query)
        key = shape(key)
        value = shape(value)
        query = shape(query)

        key_len = key.size(2)
        query_len = query.size(2)

        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)
        # batch x heads x query_len x key_len
        query_key = torch.matmul(query, key.transpose(2, 3))

        scores = query_key
        scores = scores.float()

        if mask is not None:
            mask = mask.unsqueeze(1)  # [B, 1, 1 (?), T_values]
            scores = scores.masked_fill(mask, -1e18)

        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores).to(query.dtype)
        drop_attn = self.dropout(attn)

        context_original = torch.matmul(drop_attn, value)

        context = unshape(context_original)
        output = self.final_linear(context)
        top_attn = attn.view(batch_size, head_count, query_len, key_len)[
            :, 0, :, :
        ].contiguous()

        return output, top_attn

    def update_dropout(self, dropout):
        self.dropout.p = dropout


class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.
    Args:
        d_model (int): the size of input for the first-layer of the FFN.
        d_ff (int): the hidden layer size of the second-layer
            of the FNN.
        dropout (float): dropout probability in :math:`[0, 1)`.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        """Layer definition.
        Args:
            x: ``(batch_size, input_len, model_dim)``
        Returns:
            (FloatTensor): Output ``(batch_size, input_len, model_dim)``.
        """

        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x

    def update_dropout(self, dropout):
        self.dropout_1.p = dropout
        self.dropout_2.p = dropout


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.
    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout, attention_dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(heads, d_model, dropout=attention_dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, interaction_embeds, item_embeds, mask):
        """
        Args:
            interaction_embeds (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, 1, src_len)`` pourquoi mask est de cette taille ??
        Returns:
            (FloatTensor):
            * outputs ``(batch_size, src_len, model_dim)``
        """
        context, _ = self.self_attn(
            interaction_embeds, interaction_embeds, item_embeds, mask=mask
        )
        out = self.dropout(context) + item_embeds
        return self.feed_forward(out)

    def update_dropout(self, dropout, attention_dropout):
        self.self_attn.update_dropout(attention_dropout)
        self.feed_forward.update_dropout(dropout)
        self.dropout.p = dropout


class SAKT(nn.Module):
    """Self-attentive knowledge tracing.
    
    Arguments:
            num_items (int): Number of items
            hid_size (int): Attention dot-product dimension
            heads (int): Number of parallel attention heads
            encode_pos (bool): If True, add positional encoding
            dropout (float): Dropout probability
    """

    def __init__(
        self, num_items, hid_size=512, heads=8, dropout=0.2, position_encoding=True
    ):
        super(SAKT, self).__init__()
        self.num_items = num_items
        self.interaction_embedding = nn.Embedding(
            2 * num_items, hid_size
        )  # maybe padding is needed
        self.item_embedding = nn.Embedding(
            num_items, hid_size
        )  # maybe padding is needed
        self.position_encoding = position_encoding
        if self.position_encoding:
            self.pe = PositionalEncoding(dropout, hid_size)
        self.encoder_layer = TransformerEncoderLayer(
            hid_size, heads, hid_size, dropout, dropout
        )
        self.layer_norm = nn.LayerNorm(hid_size, eps=1e-6)
        self.out = nn.Linear(hid_size, 1)

    def forward(self, interactions, items):
        # intercations and items must be batch first
        item_embeds = self.item_embedding(items)
        interaction_embeds = self.interaction_embedding(interactions)
        mask = future_mask(interactions.size(1))
        if interactions.is_cuda:
            mask = mask.cuda()
        if self.position_encoding:
            interaction_embeds = self.pe(interaction_embeds) # idea do a concatenate instead of just adding the position embeds
        
        out = self.encoder_layer(interaction_embeds, item_embeds, mask)
        return self.out(out).squeeze(2)
