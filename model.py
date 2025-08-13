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
        pe = torch.zero(seq_len, d_model)

        # vector for position rep. position of words of shape(seq_len,1) due to unsqueeze
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        
        # Calculated in log space for numerical stability
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
