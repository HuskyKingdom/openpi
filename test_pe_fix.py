"""Quick test to verify PositionalEncoding fix."""

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from src.openpi.models.energy_model_jax import PositionalEncoding


def test_pe_module():
    """Test that PositionalEncoding can be properly initialized and serialized."""
    print("Testing PositionalEncoding fix...")
    
    # Initialize
    rng = jax.random.key(0)
    rngs = nnx.Rngs(rng)
    pe_layer = PositionalEncoding(num_hiddens=128, dropout=0.1)
    
    # Test forward pass
    x = jax.random.normal(jax.random.key(1), (4, 10, 128))
    out = pe_layer(x, train=True)
    print(f"✓ Forward pass works: input {x.shape} -> output {out.shape}")
    
    # Test state extraction (this was failing before)
    try:
        state = nnx.state(pe_layer)
        print(f"✓ State extraction works: {len(nnx.flatten(state))} parameters")
    except ValueError as e:
        print(f"✗ State extraction failed: {e}")
        return False
    
    # Test graphdef extraction
    try:
        graphdef, state = nnx.split(pe_layer)
        print(f"✓ GraphDef/State split works")
        
        # Test merge
        pe_restored = nnx.merge(graphdef, state)
        print(f"✓ Merge works")
        
        # Test that restored module works
        out2 = pe_restored(x, train=True)
        print(f"✓ Restored module forward pass works")
        
    except Exception as e:
        print(f"✗ GraphDef operations failed: {e}")
        return False
    
    print("\n✅ All PositionalEncoding tests passed!")
    return True


if __name__ == "__main__":
    success = test_pe_module()
    exit(0 if success else 1)

