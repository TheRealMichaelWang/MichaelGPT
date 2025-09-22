import torch
import torch.nn as nn
import torch.nn.functional as F
import dataclasses
import tiktoken
from typing import Optional

#MichaelGPT is based on GPT-2 and uses the same architecture

@dataclasses.dataclass
class Config:
    vocab_size: int = 50257
    context_window: int = 1024

    embedding_dim: int = 1024       

    attention_layers: int = 24
    heads_per_attention_layer: int = 16
    mlp_expansion_factor: int = 4

    dropout: float = 0.1

    pad_token_id: int = 50256      # EOF/End of Text marker
    init_std: float = 0.02

class CasualSelfAttention(nn.Module):
    def __init__(self, config: Config):
        super(CasualSelfAttention, self).__init__()
        self.config = config

        # the dimension of the query, key, and value vectors
        # calculated via division so embedding_to_qkv total weight params is 3 * embedding_dim^2
        self.head_dim = config.embedding_dim // config.heads_per_attention_layer

        # this is the layer that generates key, query, and value filters (of head_dim size) for every head in the attention layer
        # so that is 3 weight matricies for every heads_per_attention_layer head's of dim head dim.
        self.embedding_to_qkv = nn.Linear(config.embedding_dim, 3 * config.embedding_dim)

        # maps outputs of attention to shortcut for adjusting embedding
        self.projection = nn.Linear(config.embedding_dim, config.embedding_dim)

        # dropout layers
        self.projection_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        batch_size, sequence_length, embedding_dim = x.shape

        query_combined, key_combined, value_combined = self.embedding_to_qkv(x).chunk(3, dim=-1)

        # splits a embedding dim by embedding dim matrix into an embedding dim by head_dim matrix for every head in attention layer
        def split_heads(tensor):
            return tensor.view(batch_size, sequence_length, self.config.heads_per_attention_layer, self.head_dim).transpose(1, 2)

        queries = split_heads(query_combined)
        keys = split_heads(key_combined)
        values = split_heads(value_combined)

        #compute dot product attention pattern
        attention_pattern = F.scaled_dot_product_attention(queries, keys, values, attn_mask=None, dropout_p= self.config.dropout if self.training else 0, is_causal=True)

        #merge and combine output of attention heads
        adjustments = attention_pattern.transpose(1, 2).contiguous().view(batch_size, sequence_length, embedding_dim)
        adjustments = self.projection_dropout(self.projection(adjustments))

        return adjustments

class MultiLayerPerceptron(nn.Module):
    def __init__(self, config: Config):
        super(MultiLayerPerceptron, self).__init__()

        self.inner_dim = config.mlp_expansion_factor * config.embedding_dim

        self.front_layer = nn.Linear(config.embedding_dim, self.inner_dim)
        self.back_layer = nn.Linear(self.inner_dim, config.embedding_dim)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.front_layer(x)
        x = F.gelu(x)  # GPT-2 uses GELU
        x = self.dropout(self.back_layer(x))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, config: Config):
        super(DecoderBlock, self).__init__()

        self.layer_norm1 = nn.LayerNorm(config.embedding_dim)
        self.layer_norm2 = nn.LayerNorm(config.embedding_dim)

        self.attention = CasualSelfAttention(config)
        self.mlp = MultiLayerPerceptron(config)

    def forward(self, x):
        # both attention and mlp are normalized residual connections
        x = x + self.attention(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x

class GPT2Model(nn.Module):
    def __init__(self, config: Config):
        super(GPT2Model, self).__init__()

        self.config = config

        self.semantic_embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.positional_embedding = nn.Embedding(config.context_window, config.embedding_dim)

        self.decoder_blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.attention_layers)])

        # these run on final embedding vector of sequence
        self.layer_norm = nn.LayerNorm(config.embedding_dim)
        self.to_logits = nn.Linear(config.embedding_dim, config.vocab_size)
        self.to_logits.weight = self.semantic_embedding.weight

        # intialize weights
        self.apply(self.initialize_weights)

    def forward(self, test_input: torch.Tensor, expected_output: Optional[torch.Tensor] = None):
        batch_size, sequence_length = test_input.shape

        assert sequence_length <= self.config.context_window, "Sequence length must be within range of the model's context window"
        assert expected_output is None or expected_output.shape == (batch_size, sequence_length), "Expected output must have same shape as input"

        positions = torch.arange(0, sequence_length, dtype=torch.long, device=test_input.device) #extracts positions ass array (i.e [0, 1, 2, ..., sequence_length - 1])
        pos_embedding = self.positional_embedding(positions) #calculate embeddings

        semantic_embedding = self.semantic_embedding(test_input) #calculate semantic embeddings

        # combine to form main embedding
        x = semantic_embedding + pos_embedding

        # run decoder blocks
        for block in self.decoder_blocks:
            x = block(x)
        
        x = self.layer_norm(x)

        #map to logits
        logits = self.to_logits(x)

        #calculate loss
        loss = None
        if expected_output is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                expected_output.view(-1),
                ignore_index=self.config.pad_token_id
            )
        return logits, loss

    @torch.no_grad()
    def generate_tokens(self, prompt: torch.Tensor, new_tokens: int = 50, temperature: float = 1.0, top_k: Optional[int] = 50) -> torch.Tensor:
        for _ in range(new_tokens):
            prompt = prompt[:, -self.config.context_window:]

            logits, _ = self(prompt)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)

            prompt = torch.cat((prompt, next_token), dim=1)
        return prompt

    def generate_text(self, prompt: str, encoding: tiktoken.Encoding, device: str, new_tokens: int = 50, temperature: float = 1.0, top_k: Optional[int] = 50) -> str:
        prompt_tokens = encoding.encode(prompt)
        prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long).to(device)
        prompt_tokens = prompt_tokens.unsqueeze(0)
        generated_tokens = self.generate_tokens(prompt_tokens, new_tokens, temperature, top_k)
        generated_text = encoding.decode(generated_tokens[0].tolist())
        return generated_text

    def initialize_weights(self, module: nn.Module):
        match module:
            case nn.Linear() as lin:
                nn.init.normal_(lin.weight, mean=0.0, std=self.config.init_std)
                if lin.bias is not None:
                    nn.init.zeros_(lin.bias)
            case nn.Embedding() as emb:
                nn.init.normal_(emb.weight, mean=0.0, std=self.config.init_std)
            case nn.LayerNorm() as ln:
                nn.init.ones_(ln.weight)
                nn.init.zeros_(ln.bias)
            case _:
                pass

    def report_params(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params}")