import torch
from torch import nn 
from torch.nn import functional as F
import math
from torch.nn import init

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        x = x / rms
        return x * self.scale

class FFN(nn.Module):
  def __init__(self, embedding_dim, hidden_dim, dropout = 0.1):
    super().__init__()
    self.linear_1 = nn.Linear(embedding_dim, hidden_dim)
    self.linear_2 = nn.Linear(hidden_dim, embedding_dim)
    self.act = nn.GELU()
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    return self.linear_2(self.dropout(self.act(self.linear_1(x))))

class AttentionHead(nn.Module):
    def __init__(self, embedding_dim, key_dim, context_length):
        super().__init__()
        
        self.w_q = nn.Parameter(torch.Tensor(embedding_dim, key_dim))
        self.w_k = nn.Parameter(torch.Tensor(embedding_dim, key_dim))
        self.w_v1 = nn.Parameter(torch.Tensor(embedding_dim, key_dim))
        self.w_v2 = nn.Parameter(torch.Tensor(key_dim, embedding_dim))

        init.xavier_uniform_(self.w_q)
        init.xavier_uniform_(self.w_k)
        init.xavier_uniform_(self.w_v1)
        init.xavier_uniform_(self.w_v2)

        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1).bool())
        self.k_dim = key_dim

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        
        keys = x @ self.w_k
        queries = x @ self.w_q
        
        attention = queries @ keys.transpose(-2, -1) * (self.k_dim ** -0.5)
        
        mask = self.mask[:seq_length, :seq_length] 
        attention = attention.masked_fill(mask.unsqueeze(0).expand(batch_size, -1, -1), -float('inf'))
        
        attention = F.softmax(attention, dim=-1)
        
        values = x @ self.w_v1 @ self.w_v2
        additions = attention @ values

        return additions


class MultiHeadAttention(nn.Module):
  def __init__(self, embedding_dim, num_heads, context_length):
    super().__init__()
    assert embedding_dim % num_heads == 0
    self.key_dim = int(embedding_dim / num_heads)
    self.heads = nn.ModuleList(AttentionHead(embedding_dim, self.key_dim, context_length) for _ in range(num_heads))

  def forward(self, x):
    for head in self.heads:
        x = x + head(x)
    return x
      
class TransformerBlock(nn.Module):
  def __init__(self, embedding_dim, num_heads, hidden_dim, context_length):
    super().__init__()
    self.ffn = FFN(embedding_dim, hidden_dim)
    self.attention = MultiHeadAttention(embedding_dim, num_heads, context_length)
    #self.norm1 = nn.LayerNorm((context_length, embedding_dim))
    #self.norm2 = nn.LayerNorm((context_length, embedding_dim))
    self.norm1 = RMSNorm((context_length, embedding_dim))
    self.norm2 = RMSNorm((context_length, embedding_dim))

  def forward(self, x):
    x = x + self.ffn(x)
    x = self.norm1.forward(x)
    x = x + self.attention(x)
    x = self.norm2.forward(x)
    return x

class PositionalEncoding(nn.Module):
    def __init__(self, context_length, embedding_dim):
        super(PositionalEncoding, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.context_length = context_length

        pe = torch.zeros(context_length, embedding_dim)
        position = torch.arange(0, context_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class HolgerGPT(nn.Module):
  def __init__(self, num_layers, embedding_dim, num_heads, hidden_dim, context_length, vocab_size):
    super().__init__()
    self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
    self.pos_embedding = PositionalEncoding(context_length, embedding_dim)
    self.transformer_blocks = nn.ModuleList(TransformerBlock(embedding_dim, num_heads, hidden_dim, context_length) for _ in range(num_layers))
    self.linear = nn.Linear(embedding_dim * context_length, vocab_size)
    self.embedding_dim = embedding_dim
    self.context_length = context_length

  def forward(self, x):
    x = self.word_embedding(x)
    x = self.pos_embedding(x)
    for block in self.transformer_blocks:
      x = block.forward(x)
    x = self.linear(x.reshape(-1, self.embedding_dim * self.context_length))
    return x

if __name__ == '__main__':
    model = HolgerGPT(2, 64, 4, 128, 128, 16)
    x = torch.ones((32, 128), dtype=torch.long)
    y = torch.ones((32), dtype=torch.long)
    y_pred = model.forward(x)
    loss = F.cross_entropy(y_pred, y)
    print(loss)