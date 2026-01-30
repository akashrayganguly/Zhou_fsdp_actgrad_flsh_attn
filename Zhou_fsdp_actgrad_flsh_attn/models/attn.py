"""
attn.py - Attention Mechanisms with FlashAttention Support

MODIFICATION: FullAttention now uses PyTorch 2.0+ Scaled Dot-Product Attention (SDPA)
which automatically selects FlashAttention2 on compatible hardware (A100, H100, etc.)

Benefits of FlashAttention:
- 2-4x faster than standard attention on A100 GPUs
- O(N) memory instead of O(N²) for attention matrix
- Fused dropout for efficiency
- No code changes needed elsewhere - same interface as before
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask


class FullAttention(nn.Module):
    """
    FlashAttention-powered Scaled Dot-Product Attention
    
    Uses PyTorch 2.0+ SDPA which automatically selects the best kernel:
    1. FlashAttention2 (fastest, requires SM80+ GPUs like A100/H100)
    2. Memory-Efficient Attention (xFormers-style)  
    3. Math fallback (standard attention)
    
    Interface is identical to the original FullAttention for drop-in replacement.
    
    Note: When output_attention=True, falls back to standard attention since
    FlashAttention doesn't compute explicit attention weights for memory efficiency.
    """
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout_p = attention_dropout
        
        # Dropout layer for fallback mode
        self.dropout = nn.Dropout(attention_dropout)
        
        # Check if SDPA is available (PyTorch 2.0+)
        self.sdpa_available = hasattr(F, 'scaled_dot_product_attention')
        
        if not self.sdpa_available:
            print("WARNING: PyTorch SDPA not available (requires PyTorch 2.0+). "
                  "Using standard attention implementation.")
        
    def forward(self, queries, keys, values, attn_mask):
        """
        Forward pass
        
        Args:
            queries: [B, L, H, E] - Query tensor
            keys: [B, S, H, E] - Key tensor
            values: [B, S, H, D] - Value tensor
            attn_mask: Optional attention mask
            
        Returns:
            output: [B, L, H, D] - Attention output (contiguous)
            attn_weights: Attention weights if output_attention=True, else None
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        
        scale = self.scale or 1./sqrt(E)
        
        # Use FlashAttention via SDPA when available and attention weights not needed
        if self.sdpa_available and not self.output_attention:
            return self._flash_attention(queries, keys, values, attn_mask, scale)
        else:
            # Fallback to standard attention (needed when output_attention=True or SDPA unavailable)
            return self._standard_attention(queries, keys, values, attn_mask, scale)
    
    def _flash_attention(self, queries, keys, values, attn_mask, scale):
        """
        FlashAttention implementation using PyTorch SDPA
        
        SDPA automatically selects the best available kernel:
        - flash_sdp: FlashAttention (SM80+, e.g., A100)
        - mem_efficient_sdp: Memory-efficient attention
        - math_sdp: Standard PyTorch attention
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        
        # SDPA expects shape: [B, H, L, E] - transpose from [B, L, H, E]
        q = queries.transpose(1, 2)  # [B, H, L, E]
        k = keys.transpose(1, 2)     # [B, H, S, E]
        v = values.transpose(1, 2)   # [B, H, S, D]
        
        # Prepare attention mask for SDPA
        # SDPA uses: True = KEEP, False = MASK (opposite of our mask convention)
        # Or uses additive mask where -inf means mask out
        sdpa_mask = None
        if self.mask_flag:
            if attn_mask is not None:
                # Convert boolean mask to additive mask for SDPA
                # Our mask: True = mask out (don't attend)
                # SDPA additive mask: -inf = mask out
                sdpa_mask = torch.zeros(B, H, L, S, device=queries.device, dtype=queries.dtype)
                sdpa_mask.masked_fill_(attn_mask.mask, float('-inf'))
            else:
                # Use built-in causal masking (more efficient)
                # is_causal=True only works when L == S
                pass
        
        # Determine if we can use is_causal flag (more efficient than explicit mask)
        use_causal = self.mask_flag and attn_mask is None and L == S
        
        # Set dropout for training
        dropout_p = self.dropout_p if self.training else 0.0
        
        # Call SDPA with FlashAttention
        # Note: scale in SDPA is applied as: scores = (Q @ K^T) * scale
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,       # Enable FlashAttention
            enable_math=True,        # Enable math fallback
            enable_mem_efficient=True  # Enable memory-efficient attention
        ):
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=sdpa_mask,
                dropout_p=dropout_p,
                is_causal=use_causal,
                scale=scale
            )
        
        # Transpose back: [B, H, L, D] -> [B, L, H, D]
        out = out.transpose(1, 2).contiguous()
        
        return (out, None)
    
    def _standard_attention(self, queries, keys, values, attn_mask, scale):
        """
        Standard attention implementation (fallback)
        
        Used when:
        - output_attention=True (need to return attention weights)
        - SDPA is not available (PyTorch < 2.0)
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        # Compute attention scores: [B, H, L, S]
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        
        # Apply masking
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        # Softmax and dropout
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        
        # Compute output: [B, L, H, D]
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module):
    """
    ProbSparse Attention for Informer
    
    Selects top-u queries based on sparsity measurement for efficiency.
    This is the key innovation of the Informer paper.
    
    Note: ProbAttention is kept as-is since it has custom sparse logic
    that doesn't map directly to FlashAttention.
    """
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        """
        Compute ProbSparse attention scores
        
        Args:
            Q: [B, H, L_Q, D]
            K: [B, H, L_K, D]
            sample_k: number of keys to sample
            n_top: number of top queries to select
        """
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # Calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # Find the Top_k query with sparsity measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # Use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert(L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)

        context_in[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(attn, V).type_as(context_in)
        
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V])/L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q
        
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # Add scale factor
        scale = self.scale or 1./sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        
        # Get the context
        context = self._get_initial_context(values, L_Q)
        
        # Update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)
        
        return context.transpose(2, 1).contiguous(), attn


class AttentionLayer(nn.Module):
    """
    Attention Layer wrapper
    
    Handles projection of queries, keys, values and wraps the attention mechanism.
    Works with both FullAttention (now FlashAttention-powered) and ProbAttention.
    """
    def __init__(self, attention, d_model, n_heads, 
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        
        if self.mix:
            out = out.transpose(2, 1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
