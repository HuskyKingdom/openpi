"""JAX/Flax implementation of Energy Model for OpenPI."""

import jax
import jax.numpy as jnp
import flax.nnx as nnx
from typing import Optional

from openpi.shared import array_typing as at


class PositionalEncoding(nnx.Module):
    """Positional encoding with sine and cosine functions."""
    
    def __init__(self, num_hiddens: int, rngs: nnx.Rngs, dropout: float = 0.2, max_len: int = 20000):
        super().__init__()
        self.dropout = nnx.Dropout(dropout, rngs=rngs)
        self.num_hiddens = num_hiddens
        self.max_len = max_len
    
    def _compute_pe(self, seq_len: int) -> jax.Array:
        """Compute positional encoding on-the-fly."""
        position = jnp.arange(seq_len, dtype=jnp.float32).reshape(-1, 1)
        div_term = jnp.arange(0, self.num_hiddens, 2, dtype=jnp.float32) / self.num_hiddens
        div_term = 1.0 / jnp.power(10000.0, div_term)
        
        # Create encoding matrix [1, seq_len, num_hiddens]
        pe = jnp.zeros((1, seq_len, self.num_hiddens))
        pe = pe.at[:, :, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, :, 1::2].set(jnp.cos(position * div_term))
        return pe
    
    def __call__(self, x: jax.Array, *, train: bool = True) -> jax.Array:
        """
        Args:
            x: Input tensor of shape [B, S, D]
            train: Whether in training mode (for dropout)
        Returns:
            Output tensor of shape [B, S, D]
        """
        seq_len = x.shape[1]
        pe = self._compute_pe(seq_len)
        x = x + pe
        return self.dropout(x, deterministic=not train)


class SeqPool(nnx.Module):
    """Sequence pooling layer."""
    
    def __init__(self, mode: str = "mean"):
        super().__init__()
        assert mode in ["cls", "mean"], f"mode must be 'cls' or 'mean', got {mode}"
        self.mode = mode
    
    def __call__(self, h: jax.Array) -> jax.Array:
        """
        Args:
            h: Input tensor of shape [B, S, D]
        Returns:
            Pooled tensor of shape [B, D]
        """
        if self.mode == "cls":
            return h[:, 0, :]
        else:
            return jnp.mean(h, axis=1)


class MLPResNetBlock(nnx.Module):
    """One MLP ResNet block with a residual connection."""
    
    def __init__(self, dim: int, rngs: nnx.Rngs):
        super().__init__()
        self.layer_norm = nnx.LayerNorm(dim, rngs=rngs)
        self.linear = nnx.Linear(dim, dim, rngs=rngs)
    
    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Args:
            x: Input tensor of shape [B, D] or [B, S, D]
        Returns:
            Output tensor of same shape as input
        """
        identity = x
        x = self.layer_norm(x)
        x = self.linear(x)
        x = nnx.silu(x)
        x = x + identity
        return x


class MLPResNet(nnx.Module):
    """MLP with residual connection blocks."""
    
    def __init__(
        self, 
        num_blocks: int, 
        input_dim: int, 
        hidden_dim: int, 
        output_dim: int,
        rngs: nnx.Rngs
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.layer_norm1 = nnx.LayerNorm(input_dim, rngs=rngs)
        self.fc1 = nnx.Linear(input_dim, hidden_dim, rngs=rngs)
        
        # Store blocks with string keys to avoid integer indexing in parameter paths
        for i in range(num_blocks):
            setattr(self, f"block_{i}", MLPResNetBlock(hidden_dim, rngs=rngs))
        
        self.layer_norm2 = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_dim, output_dim, rngs=rngs)
    
    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Args:
            x: Input tensor of shape [B, input_dim] or [B, S, input_dim]
        Returns:
            Output tensor of shape [B, output_dim] or [B, S, output_dim]
        """
        x = self.layer_norm1(x)
        x = self.fc1(x)
        x = nnx.silu(x)
        
        # Apply blocks sequentially
        for i in range(self.num_blocks):
            block = getattr(self, f"block_{i}")
            x = block(x)
        
        x = self.layer_norm2(x)
        x = self.fc2(x)
        return x


class EnergyModel(nnx.Module):
    """
    Energy Model: E_phi(s, a)
    
    Input: 
        - hN(s): [B, seq, D_h] - state/context representations
        - a: [B, chunk, D_a] - action sequence
    Output: 
        - energy: [B, 1] - scalar energy for each batch element
    """
    
    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        rngs: nnx.Rngs,
        hidden: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
    ):
        super().__init__()
        
        # Cross attention layer
        self.cross_attention = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=hidden,
            qkv_features=hidden,
            out_features=hidden,
            decode=False,
            rngs=rngs,
        )
        
        # Positional encoding
        self.pe_layer = PositionalEncoding(hidden, rngs=rngs, dropout=0.2)
        
        # MLP ResNets for state and action
        self.state_linear = MLPResNet(
            num_blocks=1, 
            input_dim=state_dim, 
            hidden_dim=hidden, 
            output_dim=hidden,
            rngs=rngs,
        )
        self.action_linear = MLPResNet(
            num_blocks=1, 
            input_dim=act_dim, 
            hidden_dim=hidden, 
            output_dim=hidden,
            rngs=rngs,
        )
        
        # Prediction head
        self.prediction_head = MLPResNet(
            num_blocks=2, 
            input_dim=hidden, 
            hidden_dim=hidden, 
            output_dim=1,
            rngs=rngs,
        )
        
        # Pooling layer
        self.pool = SeqPool(mode="mean")
        
        # Constants
        self.energy_scale = 2.0
        self.energy_offset = 0.1
    
    def __call__(
        self, 
        hN: jax.Array, 
        a: jax.Array, 
        pad_mask: Optional[jax.Array] = None,
        train: bool = True,
    ) -> jax.Array:
        """
        Forward pass of the energy model.
        
        Args:
            hN: State/context tensor [B, S, D_h]
            a: Action tensor [B, H, D_a]
            pad_mask: Optional padding mask [B, S], True for padding positions
            train: Whether in training mode
            
        Returns:
            energy: Energy values [B, 1]
        """
        # Map state and action to hidden dimension
        context_mapped = self.state_linear(hN)  # [B, S, hidden]
        action_mapped = self.action_linear(a)  # [B, H, hidden]
        action_mapped = self.pe_layer(action_mapped, train=train)  # Add positional encoding
        
        # Cross attention: actions attend to context
        # In JAX MultiHeadAttention: query, key, value order
        # mask: True means "do not attend", opposite of PyTorch
        # Note: Runtime checks on traced arrays are not possible in JIT.
        # Ensure pad_mask doesn't have all-True rows when calling this function.
        if pad_mask is not None:
            # MultiHeadAttention expects mask shape: [B, num_heads, query_len, key_len]
            # or a broadcastable shape like [B, 1, 1, key_len]
            # pad_mask is [B, S], expand to [B, 1, 1, S]
            attn_mask = pad_mask[:, None, None, :]  # [B, 1, 1, S]
        else:
            attn_mask = None
        
        # Cross attention: actions (query) attend to context (key, value)
        Z = self.cross_attention(action_mapped, context_mapped, mask=attn_mask)  # [B, H, hidden]
        
        # Predict energy for each action step
        energy_feature_step = self.prediction_head(Z)  # [B, H, 1]
        
        # Apply scaling and activation
        energy_feature_step = energy_feature_step * 0.5
        E = nnx.sigmoid(energy_feature_step) * self.energy_scale + self.energy_offset
        
        # Pool over action horizon
        energy_avg = self.pool(E)  # [B, 1]
        
        return energy_avg


@at.typecheck
def energy_inbatch_swap_infonce(
    energy_model: EnergyModel,
    h: at.Float[at.Array, "B S D"],
    a_pos: at.Float[at.Array, "B H Da"],
    pad_mask: Optional[at.Bool[at.Array, "B S"]] = None,
    tau: float = 0.5,
    train: bool = True,
) -> tuple[at.Float[at.Array, ""], at.Float[at.Array, ""], at.Float[at.Array, ""]]:
    """
    Compute in-batch swap InfoNCE loss for energy model.
    
    Creates all pairs of (context_i, action_j) and computes contrastive loss.
    Positive pairs are (i, i), negative pairs are (i, j) where i != j.
    
    Args:
        energy_model: The energy model
        h: Context/state representations [B, S, D]
        a_pos: Positive action samples [B, H, Da]
        pad_mask: Optional padding mask [B, S], True for padding
        tau: Temperature for InfoNCE
        train: Whether in training mode
        
    Returns:
        loss: Scalar loss
        E_pos_mean: Mean energy of positive pairs
        E_neg_mean: Mean energy of negative pairs
    """
    B, S, D = h.shape
    H, Da = a_pos.shape[1], a_pos.shape[2]
    
    # Create all pairs by repeating and reshaping
    # h_rep[i*B + j] = h[i] (context from sample i)
    # a_rep[i*B + j] = a_pos[j] (action from sample j)
    h_rep = jnp.repeat(h[:, None, :, :], B, axis=1).reshape(B * B, S, D)  # [B*B, S, D]
    a_rep = jnp.tile(a_pos[None, :, :, :], (B, 1, 1, 1)).reshape(B * B, H, Da)  # [B*B, H, Da]
    
    # Repeat padding mask if provided
    pm = None
    if pad_mask is not None:
        pm = jnp.repeat(pad_mask[:, None, :], B, axis=1).reshape(B * B, S)  # [B*B, S]
    
    # Compute energy for all pairs
    E_all = energy_model(h_rep, a_rep, pm, train=train)  # [B*B, 1]
    E_ij = E_all.squeeze(-1).reshape(B, B)  # [B, B] - squeeze first, then reshape
    
    # InfoNCE loss: lower energy = higher similarity
    logits = (-E_ij) / tau  # [B, B]
    labels = jnp.arange(B)  # Positive pairs are on the diagonal
    
    # Cross entropy loss
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    loss = -jnp.mean(log_probs[jnp.arange(B), labels])
    
    # Compute statistics
    E_pos_mean = jnp.mean(jnp.diag(E_ij))
    
    # Compute mean of off-diagonal elements (negative pairs)
    # Use jnp.where instead of boolean indexing for JIT compatibility
    mask = ~jnp.eye(B, dtype=bool)
    E_neg_sum = jnp.sum(jnp.where(mask, E_ij, 0.0))
    E_neg_count = jnp.sum(mask)
    # Avoid division by zero (though E_neg_count should always be B*(B-1) for B>1)
    E_neg_mean = E_neg_sum / jnp.maximum(E_neg_count, 1.0)
    
    return loss, E_pos_mean, E_neg_mean

