import torch
import warnings
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from loguru import logger
from typing import Optional, Tuple

from .configuration_smt import SMTConfig

from transformers import ConvNextConfig, ConvNextModel, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

class PositionalEncoding2D(nn.Module):

    def __init__(self, dim, h_max, w_max):
        super(PositionalEncoding2D, self).__init__()
        self.h_max = h_max
        self.max_w = w_max
        self.dim = dim

        self.pe: torch.Tensor
        self.register_buffer("pe", torch.zeros((dim, h_max, w_max), requires_grad=False), persistent=False)

        div = torch.exp(-torch.arange(0., dim // 2, 2) / dim * torch.log(torch.tensor(1e+4))).unsqueeze(1)
        w_pos = torch.arange(0., w_max) * div
        h_pos = torch.arange(0., h_max) * div
        self.pe[:dim // 2:2] = torch.sin(h_pos).unsqueeze(2).repeat(1, 1, w_max)
        self.pe[1:dim // 2:2] = torch.cos(h_pos).unsqueeze(2).repeat(1, 1, w_max)
        self.pe[dim // 2::2] = torch.sin(w_pos).unsqueeze(1).repeat(1, h_max, 1)
        self.pe[dim // 2 + 1::2] = torch.cos(w_pos).unsqueeze(1).repeat(1, h_max, 1)

    def forward(self, x):
        """
        Add 2D positional encoding to x
        x: Tensor(B, C, H, W)
        returns:
        - Tensor(B, C, H, W)
        """
        return x + self.get_pe_by_size(x.size(-2), x.size(-1))

    def get_pe_by_size(self, h, w):
        return self.pe[:, :h, :w]


class PositionalEncoding1D(nn.Module):

    def __init__(self, dim, len_max):
        super(PositionalEncoding1D, self).__init__()
        self.len_max = len_max
        self.dim = dim
        self.pe: torch.Tensor
        self.register_buffer("pe", torch.zeros((len_max, dim), requires_grad=False), persistent=False)

        div = torch.exp(-torch.arange(0., dim, 2) / dim * torch.log(torch.tensor(1e+4)))
        l_pos = torch.arange(0., len_max).unsqueeze(1) * div
        self.pe[:, ::2] = torch.sin(l_pos)
        self.pe[:, 1::2] = torch.cos(l_pos)

    def forward(self, x, start = 0):
        """
        Add 1D positional encoding to x
        x: Tensor(B, L, C)
        start: index for x[:, 0, :]
        returns:
        - Tensor(B, L, C)
        """
        if isinstance(start, int):
            return x + self.pe[start:start+x.size(-2)]
        else:
            for i in range(x.size(0)):
                x[i] = x[i] + self.pe[start[i]:start[i]+x.size(-2)]
            return x

class MultiHeadAttention(nn.Module):
    """Multi-head attention matching the original checkpoint architecture.

    Uses sequence-first format (L, B, C) internally to match the trained weights.
    """

    def __init__(self, d_model:int, num_heads:int, dropout: float = 0.1,
                 bias:bool = True):
        super().__init__()

        assert(d_model % num_heads == 0), logger.error("The embeddings depth must be divisible by the number of heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale_factor = float(self.head_dim) ** -0.5

        # Named to match HuggingFace checkpoint: lq, lk, lv
        self.lq = nn.Linear(d_model, d_model, bias=bias)
        self.lk = nn.Linear(d_model, d_model, bias=bias)
        self.lv = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,
                query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                key_pad_mask: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None,
                return_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            query: (L_tgt, B, C) sequence-first format
            key: (L_src, B, C) sequence-first format
            value: (L_src, B, C) sequence-first format
            key_pad_mask: (B, L_src) padding mask
            attn_mask: (L_tgt, L_src) attention mask
            return_weights: whether to return attention weights
        """
        target_len, b, c = query.size()
        source_len = key.size(0)

        q = self.lq(query)
        k = self.lk(key)
        v = self.lv(value)

        # Reshape: (L, B, C) -> (L, B*H, D) -> (B*H, L, D)
        q = q.view(target_len, b * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(source_len, b * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(source_len, b * self.num_heads, self.head_dim).transpose(0, 1)

        # Attention: (B*H, L_tgt, D) @ (B*H, D, L_src) -> (B*H, L_tgt, L_src)
        # Note: Original model was trained WITHOUT scale factor
        attn_weights = torch.bmm(q, k.transpose(1, 2))

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if attn_mask.dtype == torch.bool:
                attn_weights.masked_fill_(attn_mask, float("-inf"))
            else:
                attn_weights += attn_mask

        if key_pad_mask is not None:
            attn_weights = attn_weights.view(b, self.num_heads, target_len, source_len)
            attn_weights = attn_weights.masked_fill(key_pad_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
            attn_weights = attn_weights.view(b * self.num_heads, target_len, source_len)

        attn_weights_softmax = self.softmax(attn_weights)
        attn_weights_dropout = self.dropout(attn_weights_softmax)

        # (B*H, L_tgt, L_src) @ (B*H, L_src, D) -> (B*H, L_tgt, D)
        attn_output = torch.bmm(attn_weights_dropout, v)

        # Reshape back: (B*H, L_tgt, D) -> (L_tgt, B*H, D) -> (L_tgt, B, C)
        attn_output = attn_output.transpose(0, 1).contiguous().view(target_len, b, c)
        attn_output = self.out_proj(attn_output)

        if return_weights:
            attn_weights_out = attn_weights_softmax.view(b, self.num_heads, target_len, source_len)
            return attn_output, attn_weights_out.sum(dim=1) / self.num_heads

        return attn_output, None

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dim_ff:int,
                 dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        # Named to match HuggingFace checkpoint
        self.input_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.cross_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)

        self.activation = nn.ReLU() if activation.lower() == "relu" else nn.GELU()

        # Named ffNet to match checkpoint
        self.ffNet = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model)
        )

        # Named norm1, norm2, norm3 to match checkpoint
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout_layers = nn.ModuleList([nn.Dropout(dropout) for _ in range(3)])

    def forward(self,
                x: torch.Tensor,
                encoder_output_key: torch.Tensor,
                encoder_output_value: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None,
                return_weights=False):
        """
        Args:
            x: (L, B, C) sequence-first decoder input
            encoder_output_key: (L_enc, B, C) encoder features for key
            encoder_output_value: (L_enc, B, C) encoder features for value
        """
        # Self-attention: query, key, value are all x
        attn_output, self_weights = self.input_attention(
            query=x, key=x, value=x,
            key_pad_mask=tgt_key_padding_mask, attn_mask=tgt_mask,
            return_weights=return_weights
        )

        x = x + self.dropout_layers[0](attn_output)
        x = self.norm1(x)

        # Cross-attention: query is x, key/value from encoder
        attn_output, cross_weights = self.cross_attention(
            query=x,
            key=encoder_output_key,
            value=encoder_output_value,
            key_pad_mask=memory_key_padding_mask,
            return_weights=return_weights
        )

        x = x + self.dropout_layers[1](attn_output)
        x = self.norm2(x)

        ffn_output = self.ffNet(x)
        x = x + self.dropout_layers[2](ffn_output)
        x = self.norm3(x)

        if return_weights:
            return x, [self_weights, cross_weights]

        return x, None

class DecoderStack(nn.Module):
    def __init__(self, num_dec_layers:int,
                 d_model:int, dim_ff:int, num_heads:int,
                 dropout:float):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  num_heads=num_heads,
                                                  dim_ff=dim_ff) for _ in range(num_dec_layers)])
    def forward(self,
                x:torch.Tensor, encoder_output_2D:torch.Tensor, encoder_output_raw:torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None,
                return_weights=False):

        output = x
        all_weights = {
            "self_attn": [],
            "cross_attn": []
        }

        for i, dec_layer in enumerate(self.layers):
            output, weights = dec_layer(x=output,
                                        encoder_output_key=encoder_output_2D,
                                        encoder_output_value=encoder_output_raw,
                                        tgt_mask=tgt_mask,
                                        tgt_key_padding_mask=tgt_key_padding_mask,
                                        memory_key_padding_mask=memory_key_padding_mask,
                                        return_weights=return_weights)
            if return_weights:
                all_weights["self_attn"].append(weights[0])
                all_weights["cross_attn"].append(weights[1])

        if return_weights:
            return output, all_weights

        return output, None

class Decoder(nn.Module):
    def __init__(self, num_dec_layers:int,
                 d_model:int, dim_ff:int, n_heads:int,
                 max_seq_length:int, out_categories:int, dropout:float = 0.1):

        super().__init__()

        self.decoder = DecoderStack(num_dec_layers=num_dec_layers,
                                    d_model=d_model, dim_ff=dim_ff, num_heads=n_heads,
                                    dropout=dropout)

        self.embedding = nn.Embedding(num_embeddings=out_categories, embedding_dim=d_model)

        self.position_encoding = PositionalEncoding1D(dim=d_model, len_max=max_seq_length)

        # Conv1d with kernel_size=1 to match checkpoint weight shape [out, in, 1]
        self.out_layer = nn.Conv1d(in_channels=d_model, out_channels=out_categories, kernel_size=1)

        # Original has ReLU before output layer
        self.end_relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, decoder_input:torch.Tensor,
                encoder_output_2D:torch.Tensor, encoder_output_raw:torch.Tensor,
                tgt_mask:Optional[torch.Tensor] = None,
                tgt_key_padding_mask:Optional[torch.Tensor] = None,
                memory_key_padding_mask:Optional[torch.Tensor] = None,
                return_weights = False):
        """
        Args:
            decoder_input: (B, L) token indices
            encoder_output_2D: (B, L_enc, C) encoder features with 2D pos encoding
            encoder_output_raw: (B, L_enc, C) raw encoder features
        """
        # Embed: (B, L) -> (B, L, C)
        decoder_input = self.embedding(decoder_input)
        decoder_input = self.position_encoding(decoder_input)

        # Convert to sequence-first for transformer: (B, L, C) -> (L, B, C)
        decoder_input = decoder_input.permute(1, 0, 2)
        encoder_output_2D = encoder_output_2D.permute(1, 0, 2)
        encoder_output_raw = encoder_output_raw.permute(1, 0, 2)

        output, weights = self.decoder(x=decoder_input, encoder_output_2D=encoder_output_2D,
                                       encoder_output_raw=encoder_output_raw,
                                       tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                                       memory_key_padding_mask=memory_key_padding_mask,
                                       return_weights=return_weights)

        # Apply ReLU and dropout (matching original: dpoutput = self.dropout(self.end_relu(output)))
        # Output is in (L, B, C) format
        output = self.dropout(self.end_relu(output))

        # Conv1d expects (B, C, L), output is (L, B, C)
        # Original: predictions = self.out_layer(dpoutput.permute(1,2,0).contiguous())
        predictions = self.out_layer(output.permute(1, 2, 0).contiguous())

        # Convert predictions to (B, L, C) for return
        predictions = predictions.permute(0, 2, 1)

        # Convert output to batch-first: (L, B, C) -> (B, L, C)
        output = output.permute(1, 0, 2)

        return output, predictions, weights

class SMTOutput(CausalLMOutputWithCrossAttentions):
    """Output wrapper for the SMT"""

class SMTModelForCausalLM(PreTrainedModel):
    config_class = SMTConfig

    def __init__(self, config:SMTConfig):
        super().__init__(config)
        conv_next_stages = 3
        next_config = ConvNextConfig(num_channels=config.in_channels,
                                     num_stages=conv_next_stages, hidden_sizes=[64,128,256], depths=[3,3,9])
        self.encoder = ConvNextModel(next_config)
        self.decoder = Decoder(num_dec_layers=config.num_dec_layers,
                               d_model=config.d_model, dim_ff=config.dim_ff, n_heads=config.num_attn_heads,
                               max_seq_length=config.maxlen, out_categories=config.out_categories)

        self.width_reduction = 2**(conv_next_stages+1)
        self.height_reduction = 2**(conv_next_stages+1)

        self.pos2D = PositionalEncoding2D(dim=config.d_model, h_max=config.maxh // self.height_reduction, w_max=config.maxw // self.width_reduction)

        self.padding_token = config.padding_token

        self.loss = nn.CrossEntropyLoss(ignore_index=config.padding_token)

        self.w2i = config.w2i
        self.i2w = config.i2w
        self.maxlen = int(config.maxlen)

    def forward_encoder(self, x):
        return self.encoder(pixel_values=x).last_hidden_state

    def forward_decoder(self, encoder_output, last_predictions, return_weights=False):
        b, _, _, _ = encoder_output.size()

        encoder_output_2D = self.pos2D(encoder_output)
        encoder_features = torch.flatten(encoder_output, start_dim=2, end_dim=3).permute(0, 2, 1)
        encoder_features_2D = torch.flatten(encoder_output_2D, start_dim=2, end_dim=3).permute(0, 2, 1)
        # Note: For inference with batch_size=1, padding mask is not needed
        # key_target_mask = self._generate_token_mask([lp.shape[0] for lp in last_predictions], last_predictions.size(), device=last_predictions.device)
        causal_mask = self._generate_causal_mask(last_predictions.size(1), last_predictions.device)

        output, predictions, weights = self.decoder(decoder_input=last_predictions,
                                                    encoder_output_2D=encoder_features_2D, encoder_output_raw=encoder_features,
                                                    tgt_mask=causal_mask, tgt_key_padding_mask=None,
                                                    memory_key_padding_mask=None,
                                                    return_weights=return_weights)

        return SMTOutput(
            logits=predictions,
            hidden_states=output,
            attentions=None if weights is None else weights["self_attn"],
            cross_attentions=None if weights is None else weights["cross_attn"]
        )

    def forward(self, encoder_input, decoder_input, labels=None):
        x = self.forward_encoder(encoder_input)
        output = self.forward_decoder(x, decoder_input)

        if labels is not None:
            output.loss = self.loss(output.logits.permute(0,2,1).contiguous(), labels)

        return output

    @torch.no_grad
    def predict(self, input, convert_to_str=False, return_weights=False):
        predicted_sequence = torch.from_numpy(np.asarray([self.w2i['<bos>']])).to(input.device).unsqueeze(0)
        encoder_output = self.forward_encoder(input)
        text_sequence = []
        for i in range(self.maxlen - predicted_sequence.shape[-1]):
            output = self.forward_decoder(encoder_output=encoder_output, last_predictions=predicted_sequence,
                                          return_weights=return_weights)
            predicted_token = torch.argmax(output.logits[:, -1, :], dim=-1).item()
            predicted_sequence = torch.cat([predicted_sequence, torch.argmax(output.logits[:, -1, :], dim=-1, keepdim=True)], dim=1)
            if convert_to_str:
                predicted_token = f"{predicted_token}"
            if self.i2w[predicted_token] == '<eos>':
                break
            text_sequence.append(self.i2w[predicted_token])

        return text_sequence, output


    def _generate_token_mask(self, token_len, total_size, device):
        batch_size, len_mask = total_size
        mask = torch.zeros((batch_size, len_mask), dtype=torch.bool, device=device)
        for i, len_ in enumerate(token_len):
            mask[i, :len_] = True

        return mask

    def _generate_causal_mask(self, token_len, device):
        causal_mask = torch.triu(
                torch.ones(token_len, token_len, dtype=torch.bool, device=device),
                diagonal=1
            )
        return causal_mask
