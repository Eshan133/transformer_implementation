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


def MultiHeadAAttention