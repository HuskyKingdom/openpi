"""Test script to verify that only energy_model parameters are trainable."""

import jax
import flax.nnx as nnx

from src.openpi.models import pi0_config
from src.openpi.shared import nnx_utils


def test_energy_model_freeze():
    """Test that train_only_energy_model correctly freezes all params except energy_model."""
    print("Testing parameter freezing for energy_model-only training...\n")
    
    # Create config with train_only_energy_model=True
    config = pi0_config.Pi0Config(
        train_only_energy_model=True,
        use_energy_loss=True,
        action_dim=32,
        action_horizon=10,
    )
    
    # Initialize model
    rng = jax.random.key(0)
    rngs = nnx.Rngs(rng)
    model = config.create(rngs)
    
    # Get all parameters
    all_params = nnx.state(model, nnx.Param)
    
    # Get freeze filter and trainable filter
    freeze_filter = config.get_freeze_filter()
    trainable_filter = nnx.All(nnx.Param, nnx.Not(freeze_filter))
    
    # Get frozen and trainable parameters
    frozen_params = nnx.state(model, freeze_filter)
    trainable_params = nnx.state(model, trainable_filter)
    
    # Flatten to count
    all_params_flat = nnx.flatten(all_params)
    frozen_params_flat = nnx.flatten(frozen_params)
    trainable_params_flat = nnx.flatten(trainable_params)
    
    print(f"Total parameters: {len(all_params_flat)}")
    print(f"Frozen parameters: {len(frozen_params_flat)}")
    print(f"Trainable parameters: {len(trainable_params_flat)}")
    print()
    
    # Print trainable parameter paths
    print("Trainable parameter paths (should all contain 'energy_model'):")
    for path, _ in trainable_params_flat.items():
        path_str = '/'.join(map(str, path))
        print(f"  {path_str}")
        # Verify all trainable params have 'energy_model' in path
        assert 'energy_model' in path_str, f"Trainable param without 'energy_model': {path_str}"
    
    print()
    
    # Print a few frozen parameter paths
    print("Sample frozen parameter paths (first 10):")
    for i, (path, _) in enumerate(frozen_params_flat.items()):
        if i >= 10:
            break
        path_str = '/'.join(map(str, path))
        print(f"  {path_str}")
        # Verify frozen params do NOT have 'energy_model' in path
        assert 'energy_model' not in path_str, f"Frozen param with 'energy_model': {path_str}"
    
    print()
    
    # Verify counts
    assert len(trainable_params_flat) > 0, "No trainable parameters found!"
    assert len(frozen_params_flat) > 0, "No frozen parameters found!"
    assert len(trainable_params_flat) + len(frozen_params_flat) == len(all_params_flat), \
        "Trainable + Frozen != Total"
    
    print("✅ All checks passed!")
    print(f"   - {len(trainable_params_flat)} energy_model params are trainable")
    print(f"   - {len(frozen_params_flat)} other params are frozen")


def test_normal_training():
    """Test that normal training (without train_only_energy_model) trains all params."""
    print("\n" + "="*70)
    print("Testing normal training mode (all params trainable)...\n")
    
    # Create config with train_only_energy_model=False
    config = pi0_config.Pi0Config(
        train_only_energy_model=False,
        use_energy_loss=False,
        action_dim=32,
        action_horizon=10,
    )
    
    # Initialize model
    rng = jax.random.key(0)
    rngs = nnx.Rngs(rng)
    model = config.create(rngs)
    
    # Get all parameters
    all_params = nnx.state(model, nnx.Param)
    
    # Get freeze filter and trainable filter
    freeze_filter = config.get_freeze_filter()
    trainable_filter = nnx.All(nnx.Param, nnx.Not(freeze_filter))
    
    # Get frozen and trainable parameters
    frozen_params = nnx.state(model, freeze_filter)
    trainable_params = nnx.state(model, trainable_filter)
    
    # Flatten to count
    all_params_flat = nnx.flatten(all_params)
    frozen_params_flat = nnx.flatten(frozen_params)
    trainable_params_flat = nnx.flatten(trainable_params)
    
    print(f"Total parameters: {len(all_params_flat)}")
    print(f"Frozen parameters: {len(frozen_params_flat)}")
    print(f"Trainable parameters: {len(trainable_params_flat)}")
    print()
    
    # In normal mode (no LoRA), freeze_filter should be nnx.Nothing
    # So all params should be trainable
    assert len(frozen_params_flat) == 0, f"Expected 0 frozen params, got {len(frozen_params_flat)}"
    assert len(trainable_params_flat) == len(all_params_flat), \
        f"Not all params are trainable: {len(trainable_params_flat)} vs {len(all_params_flat)}"
    
    print("✅ All checks passed!")
    print(f"   - All {len(trainable_params_flat)} params are trainable")


if __name__ == "__main__":
    test_energy_model_freeze()
    test_normal_training()
    print("\n" + "="*70)
    print("🎉 All tests completed successfully!")

