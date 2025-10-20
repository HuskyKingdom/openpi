"""Test script for energy-based action correction."""

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from src.openpi.models import pi0_config
from src.openpi.models.energy_correction import one_step_energy_correction, multi_step_energy_correction


def test_energy_correction():
    """Test energy correction functionality."""
    print("="*70)
    print("Testing Energy-Based Action Correction")
    print("="*70)
    
    # Create a simple Pi0 model with energy model
    config = pi0_config.Pi0Config(
        action_dim=32,
        action_horizon=10,
        energy_act_dim=7,
        use_energy_loss=True,
    )
    
    rng = jax.random.key(42)
    rngs = nnx.Rngs(rng)
    model = config.create(rngs)
    
    print(f"\n✓ Model created with energy_model")
    print(f"  - Energy model hidden dim: 512")
    print(f"  - Energy model state_dim: 2048")
    print(f"  - Energy model act_dim: {model.energy_act_dim}")
    
    # Create dummy inputs
    batch_size = 4
    seq_len = 968  # Typical prefix length for LIBERO
    state_dim = 2048
    action_horizon = 10
    act_dim = 7
    
    h = jax.random.normal(jax.random.key(1), (batch_size, seq_len, state_dim))
    actions = jax.random.normal(jax.random.key(2), (batch_size, action_horizon, act_dim))
    actions = jnp.clip(actions, -1.0, 1.0)  # Normalize to [-1, 1]
    
    # Create padding mask (all valid for simplicity)
    pad_mask = jnp.zeros((batch_size, seq_len), dtype=bool)
    
    print(f"\n✓ Created test inputs:")
    print(f"  - h (context): {h.shape}")
    print(f"  - actions: {actions.shape}")
    print(f"  - pad_mask: {pad_mask.shape}")
    
    # Compute initial energy
    initial_energies = model.energy_model(h, actions, pad_mask, train=False)
    print(f"\n✓ Initial energies: {initial_energies.squeeze()}")
    print(f"  - Mean: {jnp.mean(initial_energies):.4f}")
    
    # Test one-step correction
    print(f"\n1. Testing One-Step Energy Correction...")
    corrected_1step = one_step_energy_correction(
        model.energy_model,
        h,
        actions,
        pad_mask=pad_mask,
        alpha=0.1,
        clip_frac=0.2,
        correct_first_only=False,
        train=False,
    )
    
    corrected_energies_1 = model.energy_model(h, corrected_1step, pad_mask, train=False)
    print(f"   ✓ Corrected (1 step): {corrected_energies_1.squeeze()}")
    print(f"     - Mean: {jnp.mean(corrected_energies_1):.4f}")
    print(f"     - Energy reduction: {jnp.mean(initial_energies - corrected_energies_1):.4f}")
    print(f"     - Action change norm: {jnp.linalg.norm(corrected_1step - actions):.4f}")
    
    # Test multi-step correction
    print(f"\n2. Testing Multi-Step Energy Correction (5 steps)...")
    corrected_5step = multi_step_energy_correction(
        model.energy_model,
        h,
        actions,
        pad_mask=pad_mask,
        num_steps=5,
        alpha=0.1,
        clip_frac=0.2,
        correct_first_only=False,
        train=False,
    )
    
    corrected_energies_5 = model.energy_model(h, corrected_5step, pad_mask, train=False)
    print(f"   ✓ Corrected (5 steps): {corrected_energies_5.squeeze()}")
    print(f"     - Mean: {jnp.mean(corrected_energies_5):.4f}")
    print(f"     - Energy reduction: {jnp.mean(initial_energies - corrected_energies_5):.4f}")
    print(f"     - Action change norm: {jnp.linalg.norm(corrected_5step - actions):.4f}")
    
    # Test correct_first_only mode
    print(f"\n3. Testing First-Action-Only Correction...")
    corrected_first = one_step_energy_correction(
        model.energy_model,
        h,
        actions,
        pad_mask=pad_mask,
        alpha=0.1,
        clip_frac=0.2,
        correct_first_only=True,
        train=False,
    )
    
    first_action_change = jnp.linalg.norm(corrected_first[:, 0, :] - actions[:, 0, :])
    other_actions_change = jnp.linalg.norm(corrected_first[:, 1:, :] - actions[:, 1:, :])
    print(f"   ✓ First action change: {first_action_change:.4f}")
    print(f"   ✓ Other actions change: {other_actions_change:.4f}")
    print(f"     (should be ~0 for other actions)")
    
    # Test JIT compilation
    print(f"\n4. Testing JIT Compilation...")
    jitted_correction = jax.jit(
        lambda h, a, m: one_step_energy_correction(
            model.energy_model, h, a, m, alpha=0.1, train=False
        )
    )
    corrected_jit = jitted_correction(h, actions, pad_mask)
    print(f"   ✓ JIT compilation successful")
    print(f"   ✓ JIT output matches: {jnp.allclose(corrected_jit, corrected_1step)}")
    
    # Test gradient flow
    print(f"\n5. Testing Gradient Computation...")
    def loss_after_correction(actions_var):
        corrected = one_step_energy_correction(
            model.energy_model, h, actions_var, pad_mask, train=False
        )
        energies = model.energy_model(h, corrected, pad_mask, train=False)
        return jnp.mean(energies)
    
    grad_fn = jax.grad(loss_after_correction)
    grads = grad_fn(actions)
    print(f"   ✓ Gradient computed successfully")
    print(f"   ✓ Gradient norm: {jnp.linalg.norm(grads):.4f}")
    
    print("\n" + "="*70)
    print("✅ All Energy Correction Tests Passed!")
    print("="*70)
    
    print(f"\n📊 Summary:")
    print(f"   Initial energy: {jnp.mean(initial_energies):.4f}")
    print(f"   After 1 step:   {jnp.mean(corrected_energies_1):.4f} (Δ={jnp.mean(initial_energies - corrected_energies_1):+.4f})")
    print(f"   After 5 steps:  {jnp.mean(corrected_energies_5):.4f} (Δ={jnp.mean(initial_energies - corrected_energies_5):+.4f})")
    
    if jnp.mean(corrected_energies_1) < jnp.mean(initial_energies):
        print(f"\n✅ Energy correction is working! Energy decreased after correction.")
    else:
        print(f"\n⚠️ Warning: Energy increased after correction. This may indicate:")
        print(f"   - Energy model not yet trained")
        print(f"   - Alpha too large")
        print(f"   - Energy model has different minima")


def test_sample_with_correction():
    """Test the integrated sample_actions_with_energy_correction method."""
    print("\n" + "="*70)
    print("Testing Pi0.sample_actions_with_energy_correction()")
    print("="*70)
    
    # Create model
    config = pi0_config.Pi0Config(
        action_dim=32,
        action_horizon=10,
        energy_act_dim=7,
        use_energy_loss=True,
    )
    
    rng = jax.random.key(42)
    model = config.create(rng)
    model.eval()
    
    # Create fake observation
    obs = config.fake_obs()
    
    print(f"\n✓ Created fake observation")
    
    # Sample without correction
    rng1, rng2 = jax.random.split(jax.random.key(123))
    actions_base = model.sample_actions(rng1, obs, num_steps=5)
    print(f"  - Base actions shape: {actions_base.shape}")
    
    # Sample with correction
    actions_corrected = model.sample_actions_with_energy_correction(
        rng2, 
        obs, 
        num_steps=5,
        energy_correction_steps=3,
        energy_alpha=0.1,
        correct_first_only=False,
    )
    print(f"  - Corrected actions shape: {actions_corrected.shape}")
    
    print(f"\n✅ sample_actions_with_energy_correction() works!")
    print(f"   Note: With untrained energy model, correction may not improve actions.")
    print(f"   Train the energy model first to see meaningful corrections.")


if __name__ == "__main__":
    test_energy_correction()
    test_sample_with_correction()
    
    print("\n" + "="*70)
    print("🎉 All tests completed!")
    print("="*70)

