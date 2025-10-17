"""Quick test to verify EnergyModel serialization fixes."""

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import flax.traverse_util

from src.openpi.models.energy_model_jax import EnergyModel


def test_energy_model_serialization():
    """Test that EnergyModel can be properly initialized and serialized."""
    print("Testing EnergyModel serialization fix...")
    
    # Initialize
    rng = jax.random.key(0)
    rngs = nnx.Rngs(rng)
    model = EnergyModel(
        state_dim=256,
        act_dim=7,
        rngs=rngs,
        hidden=128,
        num_heads=4,
        num_layers=2,
    )
    
    # Test forward pass
    h = jax.random.normal(jax.random.key(1), (4, 50, 256))
    a = jax.random.normal(jax.random.key(2), (4, 10, 7))
    out = model(h, a, train=True)
    print(f"✓ Forward pass works: h={h.shape}, a={a.shape} -> out={out.shape}")
    
    # Test state extraction (this was failing with list indexing)
    try:
        state = nnx.state(model, nnx.Param)
        print(f"✓ State extraction works")
    except (ValueError, TypeError) as e:
        print(f"✗ State extraction failed: {e}")
        return False
    
    # Test parameter flattening (this was failing with "expected str instance, int found")
    try:
        state_dict = state.to_pure_dict()
        flat_params = flax.traverse_util.flatten_dict(state_dict, sep="/")
        print(f"✓ Parameter flattening works: {len(flat_params)} parameters")
        
        # Check that all keys are strings
        for key in flat_params.keys():
            if not isinstance(key, str):
                print(f"✗ Found non-string key: {key}")
                return False
        print(f"✓ All parameter keys are strings")
        
    except TypeError as e:
        print(f"✗ Parameter flattening failed: {e}")
        return False
    
    # Test graphdef/state split
    try:
        graphdef, state = nnx.split(model)
        print(f"✓ GraphDef/State split works")
        
        # Test merge
        model_restored = nnx.merge(graphdef, state)
        print(f"✓ Merge works")
        
        # Test that restored module works
        out2 = model_restored(h, a, train=True)
        print(f"✓ Restored module forward pass works")
        
    except Exception as e:
        print(f"✗ GraphDef operations failed: {e}")
        return False
    
    print("\n✅ All EnergyModel serialization tests passed!")
    return True


if __name__ == "__main__":
    success = test_energy_model_serialization()
    exit(0 if success else 1)

