"""Test script to verify JAX EnergyModel conversion."""

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from src.openpi.models.energy_model_jax import EnergyModel, energy_inbatch_swap_infonce


def test_energy_model():
    """Test basic functionality of JAX EnergyModel."""
    print("Testing JAX EnergyModel...")
    
    # Configuration
    batch_size = 4
    seq_len = 100
    action_horizon = 10
    state_dim = 512
    act_dim = 7
    hidden = 256
    
    # Initialize model
    rng = jax.random.key(0)
    rngs = nnx.Rngs(rng)
    
    model = EnergyModel(
        state_dim=state_dim,
        act_dim=act_dim,
        hidden=hidden,
        num_heads=4,
        num_layers=2,
        rngs=rngs,
    )
    
    # Create dummy inputs
    h = jax.random.normal(jax.random.key(1), (batch_size, seq_len, state_dim))
    a = jax.random.normal(jax.random.key(2), (batch_size, action_horizon, act_dim))
    pad_mask = jnp.zeros((batch_size, seq_len), dtype=bool)
    # Mark last 10 positions as padding
    pad_mask = pad_mask.at[:, -10:].set(True)
    
    # Forward pass
    print(f"Input shapes: h={h.shape}, a={a.shape}, pad_mask={pad_mask.shape}")
    energy = model(h, a, pad_mask=pad_mask, train=True)
    print(f"Output energy shape: {energy.shape}")
    print(f"Energy values: {energy[:5, 0]}")
    
    # Test InfoNCE loss
    print("\nTesting InfoNCE loss...")
    loss, E_pos, E_neg = energy_inbatch_swap_infonce(
        model, h, a, pad_mask=pad_mask, tau=0.5, train=True
    )
    print(f"Loss: {loss:.4f}")
    print(f"E_pos_mean: {E_pos:.4f}")
    print(f"E_neg_mean: {E_neg:.4f}")
    
    # Test gradient computation
    print("\nTesting gradient computation...")
    def loss_fn(model, h, a, mask):
        loss, _, _ = energy_inbatch_swap_infonce(model, h, a, mask, train=True)
        return loss
    
    grad_fn = nnx.value_and_grad(loss_fn)
    loss_val, grads = grad_fn(model, h, a, pad_mask)
    print(f"Loss value: {loss_val:.4f}")
    print(f"Gradients computed successfully!")
    
    # Test JIT compilation
    print("\nTesting JIT compilation...")
    jitted_fn = nnx.jit(lambda m, h, a, mask: m(h, a, mask, train=True))
    energy_jit = jitted_fn(model, h, a, pad_mask)
    print(f"JIT compiled energy shape: {energy_jit.shape}")
    print(f"JIT energy matches: {jnp.allclose(energy, energy_jit)}")
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_energy_model()

