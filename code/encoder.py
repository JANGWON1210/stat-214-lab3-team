import torch
import torch.nn as nn
import torch.nn.functional as F

# --- BERT-STYLE MODEL ---
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        '''
        TODO: Implement multi-head self-attention
        Args:
            hidden_size: Hidden size of the model
            num_heads: Number of attention heads
        '''
        super().__init__()
        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Q, K, V projection
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, mask=None):
        '''
        TODO: Implement forward pass for multi-head self-attention
        Args:
            x: Input
            mask: Attention mask 
        '''
        batch_size, seq_len, _ = x.size()

        # Project input to query, key, and value
        Q = self.q_proj(x)  # (batch_size, seq_len, hidden_size)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Split into heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape: (batch_size, num_heads, seq_len, head_dim)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # Shape: (batch_size, num_heads, seq_len, seq_len)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        # Shape: (batch_size, num_heads, seq_len, head_dim)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        # Final projection
        output = self.out_proj(attn_output)  # (batch_size, seq_len, hidden_size)

        return output

class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        '''
        TODO: Implement feed-forward network
        Args:
            hidden_size: Hidden size of the model
            intermediate_size: Intermediate size of the model
        '''
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size):
        super().__init__()
        self.attn = MultiHeadSelfAttention(hidden_size, num_heads)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ffn = FeedForward(hidden_size, intermediate_size)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, x, mask=None):
        '''
        TODO: Implement forward pass for transformer block
        Args:
            x: Input
            mask: Attention mask
        '''
        # Self-attention layer with residual connection
        attn_output = self.attn(x, mask)  # (batch_size, seq_len, hidden_size)
        x = self.ln1(x + attn_output)

        # Feed-forward network with residual connection
        ffn_output = self.ffn(x)  # (batch_size, seq_len, hidden_size)
        x = self.ln2(x + ffn_output)

        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size=256, num_heads=4, num_layers=4,
                 intermediate_size=512, max_len=512):
        '''
        TODO: Implement encoder
        Args:
            vocab_size: Vocabulary size
            hidden_size: Hidden size of the model
            num_heads: Number of attention heads
            num_layers: Number of layers
            intermediate_size: Intermediate size of the model
            max_len: Maximum length of the input
        '''
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, hidden_size)
        self.pos_emb = nn.Embedding(max_len, hidden_size)
        self.type_emb = nn.Embedding(2, hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, intermediate_size)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.mlm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, token_type_ids, attention_mask, output_logits=True):
        batch_size, seq_len = input_ids.size()
        device = input_ids.device

        token_embeddings = self.token_emb(input_ids)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        position_embeddings = self.pos_emb(position_ids)
        type_embeddings = self.type_emb(token_type_ids)

        x = token_embeddings + position_embeddings + type_embeddings

        for layer in self.layers:
            x = layer(x, mask=attention_mask)

        x = self.norm(x)

        if output_logits:
            return self.mlm_head(x)  # MLM logits (batch_size, seq_len, vocab_size)
        else:
            return x  # Hidden states (batch_size, seq_len, hidden_size)

