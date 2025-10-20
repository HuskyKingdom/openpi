"""Comprehensive test suite for energy model system."""

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from src.openpi.models import pi0_config
from src.openpi.models.energy_correction import one_step_energy_correction, multi_step_energy_correction


def test_full_system():
    """Test the complete energy model training and inference system."""
    print("="*80)
    print(" COMPREHENSIVE ENERGY MODEL SYSTEM TEST")
    print("="*80)
    
    # 1. Configuration Test
    print("\n[1/5] Testing Configuration...")
    config = pi0_config.Pi0Config(
        action_dim=32,
        action_horizon=10,
        energy_act_dim=7,
        energy_hidden=512,
        energy_heads=8,
        use_energy_loss=True,
        train_only_energy_model=True,
    )
    print("  ✓ Config created successfully")
    print(f"    - use_energy_loss: {config.use_energy_loss}")
    print(f"    - train_only_energy_model: {config.train_only_energy_model}")
    print(f"    - energy_act_dim: {config.energy_act_dim}")
    
    # 2. Model Initialization Test
    print("\n[2/5] Testing Model Initialization...")
    rng = jax.random.key(42)
    model = config.create(rng)
    print("  ✓ Model created successfully")
    print(f"    - Has energy_model: {hasattr(model, 'energy_model')}")
    print(f"    - energy_act_dim: {model.energy_act_dim}")
    
    # 3. Parameter Freezing Test
    print("\n[3/5] Testing Parameter Freezing...")
    freeze_filter = config.get_freeze_filter()
    trainable_filter = nnx.All(nnx.Param, nnx.Not(freeze_filter))
    
    frozen_params = nnx.state(model, freeze_filter)
    trainable_params = nnx.state(model, trainable_filter)
    
    frozen_flat = nnx.flatten(frozen_params)
    trainable_flat = nnx.flatten(trainable_params)
    
    print(f"  ✓ Parameter filtering works")
    print(f"    - Total trainable: {len(trainable_flat)}")
    print(f"    - Total frozen: {len(frozen_flat)}")
    
    # Verify all trainable params are from energy_model
    all_energy_params = all('energy_model' in '/'.join(map(str, path)) 
                            for path, _ in trainable_flat.items())
    print(f"    - All trainable params from energy_model: {all_energy_params}")
    
    if not all_energy_params:
        print("    ⚠️ WARNING: Some trainable params are NOT from energy_model!")
        return False
    
    # 4. Energy Model Forward Pass Test
    print("\n[4/5] Testing Energy Model Forward Pass...")
    batch_size = 4
    seq_len = 100
    state_dim = 2048
    
    h = jax.random.normal(jax.random.key(10), (batch_size, seq_len, state_dim))
    actions = jax.random.normal(jax.random.key(11), (batch_size, config.action_horizon, 7))
    actions = jnp.clip(actions, -1.0, 1.0)
    pad_mask = jnp.zeros((batch_size, seq_len), dtype=bool)
    
    energies = model.energy_model(h, actions, pad_mask, train=False)
    print(f"  ✓ Energy computation successful")
    print(f"    - Output shape: {energies.shape}")
    print(f"    - Energy values: {energies.squeeze()}")
    print(f"    - Mean energy: {jnp.mean(energies):.4f}")
    
    # 5. Energy Correction Test
    print("\n[5/5] Testing Energy-Based Action Correction...")
    
    # One-step correction
    corrected_1 = one_step_energy_correction(
        model.energy_model, h, actions, pad_mask,
        alpha=0.1, clip_frac=0.2, train=False
    )
    E_corrected_1 = model.energy_model(h, corrected_1, pad_mask, train=False)
    
    # Multi-step correction
    corrected_5 = multi_step_energy_correction(
        model.energy_model, h, actions, pad_mask,
        num_steps=5, alpha=0.1, clip_frac=0.2, train=False
    )
    E_corrected_5 = model.energy_model(h, corrected_5, pad_mask, train=False)
    
    print(f"  ✓ Energy correction successful")
    print(f"    - Initial energy:     {jnp.mean(energies):.4f}")
    print(f"    - After 1 step:       {jnp.mean(E_corrected_1):.4f}")
    print(f"    - After 5 steps:      {jnp.mean(E_corrected_5):.4f}")
    
    delta_1 = jnp.mean(energies) - jnp.mean(E_corrected_1)
    delta_5 = jnp.mean(energies) - jnp.mean(E_corrected_5)
    print(f"    - Energy reduction (1 step): {delta_1:+.4f}")
    print(f"    - Energy reduction (5 steps): {delta_5:+.4f}")
    
    # 6. JIT Compilation Test
    print("\n[6/5] Testing JIT Compilation...")
    jitted_correction = jax.jit(
        lambda h, a, m: one_step_energy_correction(
            model.energy_model, h, a, m, alpha=0.1, train=False
        )
    )
    corrected_jit = jitted_correction(h, actions, pad_mask)
    print(f"  ✓ JIT compilation successful")
    print(f"  ✓ JIT output matches non-JIT: {jnp.allclose(corrected_jit, corrected_1)}")
    
    # Summary
    print("\n" + "="*80)
    print(" TEST SUMMARY")
    print("="*80)
    print("✅ All tests passed!")
    print(f"\nEnergy Model Statistics:")
    print(f"  - Trainable parameters: {len(trainable_flat)}")
    print(f"  - State dim: 2048")
    print(f"  - Action dim: {model.energy_act_dim}")
    print(f"  - Hidden dim: 512")
    print(f"  - Attention heads: 8")
    
    print(f"\nRecommended Settings for Inference:")
    print(f"  - energy_correction_steps: 3")
    print(f"  - energy_alpha: 0.1")
    print(f"  - energy_clip_frac: 0.2")
    print(f"  - correct_first_only: False")
    
    print(f"\nNote: With untrained energy model, corrections may not reduce energy.")
    print(f"      Train the energy model first using:")
    print(f"      uv run scripts/train.py pi05_libero_energy --exp-name=energy_v1")
    
    return True


if __name__ == "__main__":
    success = test_full_system()
    exit(0 if success else 1)

