"""Diagnostic script to check energy model training setup."""

import jax
import flax.nnx as nnx

from src.openpi.training import config as _config
from src.openpi.models import pi0_config


def check_config():
    """Check if the config is correctly set up."""
    print("="*70)
    print("Checking pi05_libero_energy configuration...")
    print("="*70)
    
    config = _config.get_config("pi05_libero_energy")
    
    print(f"\n1. Model Config:")
    print(f"   - pi05: {config.model.pi05}")
    print(f"   - action_horizon: {config.model.action_horizon}")
    print(f"   - energy_act_dim: {config.model.energy_act_dim}")
    print(f"   - use_energy_loss: {config.model.use_energy_loss}")
    print(f"   - train_only_energy_model: {config.model.train_only_energy_model}")
    
    print(f"\n2. Training Config:")
    print(f"   - batch_size: {config.batch_size}")
    print(f"   - ema_decay: {config.ema_decay}")
    print(f"   - num_train_steps: {config.num_train_steps}")
    
    print(f"\n3. Optimizer Config:")
    print(f"   - type: {type(config.optimizer).__name__}")
    print(f"   - lr_schedule: {type(config.lr_schedule).__name__}")
    
    print(f"\n4. Freeze Filter:")
    freeze_filter = config.freeze_filter
    trainable_filter = config.trainable_filter
    print(f"   - freeze_filter type: {type(freeze_filter)}")
    print(f"   - trainable_filter type: {type(trainable_filter)}")
    
    # Test parameter filtering
    print(f"\n5. Testing Parameter Filtering:")
    rng = jax.random.key(42)
    rngs = nnx.Rngs(rng)
    model = config.model.create(rngs)
    
    all_params = nnx.state(model, nnx.Param)
    frozen_params = nnx.state(model, freeze_filter)
    trainable_params = nnx.state(model, trainable_filter)
    
    all_flat = nnx.flatten(all_params)
    frozen_flat = nnx.flatten(frozen_params)
    trainable_flat = nnx.flatten(trainable_params)
    
    print(f"   - Total parameters: {len(all_flat)}")
    print(f"   - Frozen parameters: {len(frozen_flat)}")
    print(f"   - Trainable parameters: {len(trainable_flat)}")
    
    print(f"\n6. Trainable Parameter Paths (should all contain 'energy_model'):")
    for i, (path, _) in enumerate(trainable_flat.items()):
        if i < 10:  # Show first 10
            path_str = '/'.join(map(str, path))
            print(f"   [{i+1}] {path_str}")
            if 'energy_model' not in path_str:
                print(f"       ⚠️ WARNING: This parameter does NOT contain 'energy_model'!")
        elif i == 10:
            print(f"   ... and {len(trainable_flat) - 10} more")
            break
    
    print(f"\n7. Sample Frozen Parameter Paths (should NOT contain 'energy_model'):")
    for i, (path, _) in enumerate(frozen_flat.items()):
        if i < 5:  # Show first 5
            path_str = '/'.join(map(str, path))
            print(f"   [{i+1}] {path_str}")
            if 'energy_model' in path_str:
                print(f"       ⚠️ WARNING: This parameter contains 'energy_model' but is frozen!")
        elif i == 5:
            print(f"   ... and {len(frozen_flat) - 5} more")
            break
    
    print("\n" + "="*70)
    print("✅ Configuration check complete!")
    print("="*70)
    
    # Check if config looks correct
    issues = []
    if not config.model.use_energy_loss:
        issues.append("❌ use_energy_loss is False - energy loss won't be used in training!")
    if not config.model.train_only_energy_model:
        issues.append("⚠️ train_only_energy_model is False - all parameters will be trained")
    if config.ema_decay is not None:
        issues.append("❌ ema_decay is not None - will cause RNG key error!")
    if len(trainable_flat) == 0:
        issues.append("❌ No trainable parameters found!")
    if any('energy_model' not in '/'.join(map(str, path)) for path, _ in trainable_flat.items()):
        issues.append("⚠️ Some trainable parameters don't belong to energy_model")
    
    if issues:
        print("\n⚠️ ISSUES FOUND:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print("\n✅ No issues found - configuration looks good!")
    
    return model, config


if __name__ == "__main__":
    model, config = check_config()

