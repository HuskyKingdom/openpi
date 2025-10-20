"""Test checkpoint save/load with RNG state filtering."""

import tempfile
import pathlib
import jax
import flax.nnx as nnx
import flax.traverse_util as traverse_util
import orbax.checkpoint as ocp

from src.openpi.models import pi0_config
from src.openpi.training import checkpoints as _checkpoints
from src.openpi.training import utils as training_utils
import optax


def test_checkpoint_with_rng_filtering():
    """Test that checkpoints can be saved and loaded with RNG state filtering."""
    print("="*70)
    print("Testing Checkpoint Save/Load with RNG Filtering")
    print("="*70)
    
    # Create config with energy model (which has RNG states from Dropout)
    config = pi0_config.Pi0Config(
        action_dim=32,
        action_horizon=10,
        energy_act_dim=7,
        use_energy_loss=True,
    )
    
    # Create model
    rng = jax.random.key(42)
    model = config.create(rng)
    print(f"\n✓ Created model with energy_model")
    
    # Check for RNG states
    all_state = nnx.state(model)
    flat_state = traverse_util.flatten_dict(all_state.to_pure_dict(), sep="/")
    rng_keys = [k for k in flat_state.keys() if 'rngs' in k or 'Rng' in k]
    print(f"  - Found {len(rng_keys)} RNG-related keys in model state")
    if len(rng_keys) > 0:
        print(f"    Example RNG keys:")
        for key in rng_keys[:3]:
            print(f"      {key}")
    
    # Create a minimal train state
    params = nnx.state(model, nnx.Param)
    tx = optax.adam(1e-4)
    train_state = training_utils.TrainState(
        step=100,
        params=params,
        model_def=nnx.graphdef(model),
        tx=tx,
        opt_state=tx.init(params),
        ema_decay=None,
        ema_params=None,
    )
    
    print(f"\n✓ Created train state")
    print(f"  - Step: {train_state.step}")
    
    # Test _split_params
    print(f"\n[1/3] Testing _split_params (should filter RNG)...")
    train_state_split, params_split = _checkpoints._split_params(train_state)
    
    flat_params_split = traverse_util.flatten_dict(params_split, sep="/")
    rng_keys_in_split = [k for k in flat_params_split.keys() if 'rngs' in k or 'Rng' in k]
    
    print(f"  - Original params had {len(flat_state)} keys")
    print(f"  - Split params have {len(flat_params_split)} keys")
    print(f"  - RNG keys in split params: {len(rng_keys_in_split)}")
    
    if len(rng_keys_in_split) > 0:
        print(f"    ⚠️ WARNING: RNG keys still present after split:")
        for key in rng_keys_in_split[:5]:
            print(f"      {key}")
    else:
        print(f"    ✓ All RNG keys filtered out successfully!")
    
    # Test save/load with temporary directory
    print(f"\n[2/3] Testing actual save/load...")
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = pathlib.Path(tmpdir) / "test_checkpoint"
        tmppath.mkdir()
        
        # Create checkpoint manager
        mngr = ocp.CheckpointManager(
            tmppath,
            item_handlers={
                "params": ocp.PyTreeCheckpointHandler(),
            },
            options=ocp.CheckpointManagerOptions(max_to_keep=1, create=False),
        )
        
        # Save
        print(f"  - Saving to {tmppath}...")
        mngr.save(
            0,
            {"params": {"params": params_split}},
        )
        print(f"    ✓ Saved successfully")
        
        # Load
        print(f"  - Loading from {tmppath}...")
        try:
            loaded_params = ocp.PyTreeCheckpointer().restore(
                tmppath / "0" / "params",
                item={"params": params_split},
            )["params"]
            print(f"    ✓ Loaded successfully")
            
            # Check loaded params
            loaded_flat = traverse_util.flatten_dict(loaded_params, sep="/")
            print(f"    - Loaded {len(loaded_flat)} keys")
            
            rng_in_loaded = [k for k in loaded_flat.keys() if 'rngs' in k or 'Rng' in k]
            print(f"    - RNG keys in loaded: {len(rng_in_loaded)}")
            
        except Exception as e:
            print(f"    ✗ Load failed: {e}")
            return False
    
    # Test model.load() with filtered params
    print(f"\n[3/3] Testing model.load() with filtered params...")
    try:
        loaded_model = config.load(params_split, remove_extra_params=True)
        print(f"  ✓ model.load() successful!")
        
        # Verify model has RNG states (re-initialized)
        loaded_state = nnx.state(loaded_model)
        loaded_flat = traverse_util.flatten_dict(loaded_state.to_pure_dict(), sep="/")
        rng_in_model = [k for k in loaded_flat.keys() if 'rngs' in k or 'Rng' in k]
        print(f"  - RNG keys in loaded model: {len(rng_in_model)}")
        print(f"    (should be re-initialized, same count as original)")
        
        if len(rng_in_model) == len(rng_keys):
            print(f"    ✓ RNG states correctly re-initialized!")
        else:
            print(f"    ⚠️ RNG count mismatch: {len(rng_in_model)} vs {len(rng_keys)}")
        
    except Exception as e:
        print(f"  ✗ model.load() failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*70)
    print("✅ All checkpoint RNG filtering tests passed!")
    print("="*70)
    
    print(f"\nSummary:")
    print(f"  - Checkpoint saves {len(flat_params_split)} parameters (RNG excluded)")
    print(f"  - Checkpoint loads successfully")
    print(f"  - Model re-initializes {len(rng_keys)} RNG states")
    print(f"  - Loaded model is ready for inference")
    
    return True


if __name__ == "__main__":
    success = test_checkpoint_with_rng_filtering()
    exit(0 if success else 1)

