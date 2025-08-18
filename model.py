import math
import torch
import torch.nn as nn

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model) # menthiond in paper
    

class PositionalEncoding(nn.Module):

    def __init__(self, d_model:int, seq_len: int, dropout: float) -> None :
        super().__inti__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Matrix (seq_len, d_model) PE for each 
        pe = torch.zeros(seq_len, d_model)

        # vector for position rep. position of words of shape(seq_len,1) due to unsqueeze
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # numerator
        
        # Calculated in log space for numerical stability
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # denominator

        # Apply sin to even and cos to odd position
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position + div_term)

        # For batch of sentences/ Adding new dimension
        pe = pe.unsqueeze(0) # (1,seq_len, d_model)

        # Save as buffer and not as parameter
        self.register_buffer('pe', pe)
    

    def forward(self, x):

        # This represents that PE only depends on position and not tokens. so its same for all batches
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad(False) # x -> (batch_size, seq_len, d_model)
        return self.dropout(x)



class LayerNormaliztion(nn.Module):

    def __init__(self, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # Added
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) +self.bias



class FeedForwardBlock(nn.Module):

    def __init__(self, d_model:int, d_ff:int, dropout:float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.liner_2 = nn.Linear(d_ff, d_model) # W2 and B2
    
    def forward(self, x):
        # Input -> (Batch, seq_len, d_model) -->linear_1--> (Batch, seq_len, d_ff) -->linear_2--> (Batch, seq_len, d_model)
        return self.liner_2(self.dropout(torch.relu(self.linear_1(x))))


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model:int, h:int, dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout)
        assert d_model % h == 0, "d_model is not divisible by h"

        # ??????? what is happening here, why linear layer?
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model) # Wq
        self.w_k = nn.Linear(d_model, d_model) # Wk
        self.w_v = nn.Linear(d_model, d_model) # Wv

        self.w_o = nn.Linear(d_model, d_model) # Wo

        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # (Batch, h, seq_len, d_k) ---> (Batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k) # transpose interchange the last two dims.
        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1) # (Batch, h, seq_len, seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        # (attention_scores @ value) --> SHape = (Batch, h, seq_len, d_k)
        return (attention_scores @ value), attention_scores


    def forward(self, q, k, v, mask):
        # mask for hiding attention for certain words

        query = self.w_q(q) # Q` = (Batch, seq_len, d_model) ---> (Batch, seq_len, d_model)
        key = self.w_q(k) #K`
        value = self.w_q(v) #V`

        # Seperating the heads 
        # (Batch, seq_len, d_model) ---> (Batch, seq_len, h, d_k) --transpose-> (Batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2) # Splitting the embedding and not the sentence
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2) 
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2) 

        # Calculating the attention
        x, self.attention_score = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (Batch, h, seq_len, d_k) --> (Batch, seq_len, h, d_k) --> (Batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        return self.w_o(x)


class ResidualConnection(nn.Module):

    def __init__(self, dropout:float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormaliztion()
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.feed_forward_block[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormaliztion

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)