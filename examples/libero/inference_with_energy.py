"""
Example script for running LIBERO inference with energy-based action correction.

This script demonstrates how to use a trained energy model to refine policy predictions
at test time using gradient descent on the energy landscape.

Usage:
    # Without energy correction (baseline)
    python examples/libero/inference_with_energy.py --host localhost --port 8000
    
    # With energy correction (3 gradient steps)
    python examples/libero/inference_with_energy.py --host localhost --port 8000 \
        --use-energy-correction --energy-steps 3 --energy-alpha 0.1
"""

import dataclasses
import logging

import jax
import numpy as np
import tyro

from openpi.models import model as _model
from openpi.models.energy_correction import multi_step_energy_correction


@dataclasses.dataclass
class InferenceArgs:
    # Server parameters
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Energy correction parameters
    use_energy_correction: bool = False  # Enable energy-based correction
    energy_steps: int = 3  # Number of gradient descent steps
    energy_alpha: float = 0.1  # Step size for energy correction
    energy_clip_frac: float = 0.2  # Maximum correction as fraction of action norm
    correct_first_only: bool = False  # Only correct the first action in chunk
    
    # Inference parameters
    flow_steps: int = 10  # Number of flow matching steps
    
    # Logging
    verbose: bool = True  # Print energy statistics


class EnergyCorrectingPolicy:
    """Wrapper for policy that applies energy correction at inference time."""
    
    def __init__(self, model, args: InferenceArgs):
        """
        Args:
            model: The Pi0 model with trained energy_model
            args: Inference arguments including energy correction config
        """
        self.model = model
        self.args = args
        self.model.eval()
        
        # Statistics tracking
        self.total_inferences = 0
        self.total_energy_reduction = 0.0
    
    def infer(self, observation_dict: dict) -> dict:
        """
        Run inference with optional energy correction.
        
        Args:
            observation_dict: Dictionary with keys matching the policy's expected format
                (e.g., 'observation/image', 'observation/state', 'prompt')
        
        Returns:
            result_dict: Dictionary with 'actions' key containing corrected actions
        """
        # First, use the policy's transformation pipeline to convert to model format
        # (This would normally go through the policy's transform pipeline)
        # For this example, we assume observation_dict is already in the right format
        
        # Generate random key for this inference
        rng = jax.random.fold_in(jax.random.key(0), self.total_inferences)
        self.total_inferences += 1
        
        # Sample base actions (without correction)
        if self.args.use_energy_correction and hasattr(self.model, 'sample_actions_with_energy_correction'):
            # Use the built-in energy correction method
            actions = self.model.sample_actions_with_energy_correction(
                rng,
                observation_dict,  # Assuming this is already a proper Observation object
                num_steps=self.args.flow_steps,
                energy_correction_steps=self.args.energy_steps,
                energy_alpha=self.args.energy_alpha,
                energy_clip_frac=self.args.energy_clip_frac,
                correct_first_only=self.args.correct_first_only,
            )
            
            if self.args.verbose:
                # Compute energy statistics
                # Note: We'd need access to prefix_out for this
                # This is simplified for the example
                logging.info(f"Applied energy correction with {self.args.energy_steps} steps")
        else:
            # Standard sampling without correction
            actions = self.model.sample_actions(
                rng,
                observation_dict,
                num_steps=self.args.flow_steps,
            )
            
            if self.args.verbose and self.args.use_energy_correction:
                logging.warning("Energy correction requested but not available for this model")
        
        return {"actions": np.array(actions)}
    
    def print_statistics(self):
        """Print accumulated statistics."""
        if self.total_inferences > 0:
            avg_reduction = self.total_energy_reduction / self.total_inferences
            print(f"\nEnergy Correction Statistics:")
            print(f"  Total inferences: {self.total_inferences}")
            print(f"  Avg energy reduction: {avg_reduction:.4f}")


def test_correction_standalone():
    """Standalone test of energy correction without policy server."""
    print("Testing energy correction in standalone mode...\n")
    
    # Create model
    config = pi0_config.Pi0Config(
        action_dim=32,
        action_horizon=10,
        energy_act_dim=7,
    )
    
    rng = jax.random.key(42)
    model = config.create(rng)
    model.eval()
    
    # Create dummy inputs
    batch_size = 1
    seq_len = 968
    state_dim = 2048
    action_horizon = 10
    act_dim = 7
    
    h = jax.random.normal(jax.random.key(1), (batch_size, seq_len, state_dim))
    actions = jax.random.normal(jax.random.key(2), (batch_size, action_horizon, act_dim))
    actions = jnp.clip(actions, -1.0, 1.0)
    pad_mask = jnp.zeros((batch_size, seq_len), dtype=bool)
    
    print("Initial setup:")
    print(f"  Context shape: {h.shape}")
    print(f"  Actions shape: {actions.shape}")
    
    # Compute initial energy
    E_init = model.energy_model(h, actions, pad_mask, train=False)
    print(f"\nInitial energy: {E_init[0, 0]:.4f}")
    
    # Apply correction with different settings
    configs = [
        {"name": "1 step, α=0.05", "steps": 1, "alpha": 0.05},
        {"name": "1 step, α=0.10", "steps": 1, "alpha": 0.10},
        {"name": "3 steps, α=0.10", "steps": 3, "alpha": 0.10},
        {"name": "5 steps, α=0.10", "steps": 5, "alpha": 0.10},
        {"name": "5 steps, α=0.20", "steps": 5, "alpha": 0.20},
    ]
    
    print("\nTesting different correction settings:")
    for cfg in configs:
        corrected = multi_step_energy_correction(
            model.energy_model,
            h,
            actions,
            pad_mask=pad_mask,
            num_steps=cfg["steps"],
            alpha=cfg["alpha"],
            clip_frac=0.2,
            train=False,
        )
        E_corrected = model.energy_model(h, corrected, pad_mask, train=False)
        delta_E = E_init[0, 0] - E_corrected[0, 0]
        action_change = jnp.linalg.norm(corrected - actions)
        
        print(f"  {cfg['name']:20s}: E={E_corrected[0, 0]:.4f}, ΔE={delta_E:+.4f}, Δa={action_change:.4f}")
    
    print("\n✅ Standalone test complete!")
    print("Note: With untrained energy model, corrections may not reduce energy.")
    print("Train the energy model first to see meaningful energy reductions.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Run standalone test
    test_correction_standalone()
    
    print("\n" + "="*70)
    print("To use energy correction with the policy server:")
    print("="*70)
    print("""
1. Train the energy model:
   uv run scripts/train.py pi05_libero_energy --exp-name=energy_v1

2. Start policy server with the trained checkpoint:
   uv run scripts/serve_policy.py policy:checkpoint \\
       --policy.config=pi05_libero_energy \\
       --policy.dir=./checkpoints/pi05_libero_energy/energy_v1

3. Modify the client code to use energy correction:
   In examples/libero/main.py, when calling client.infer(), the server
   will use the trained energy model automatically if it's present.
   
4. For manual control over correction parameters, use the Pi0 model directly:
   
   # Load model with trained energy model
   model = config.model.load(params)
   
   # Sample with correction
   actions = model.sample_actions_with_energy_correction(
       rng, obs,
       energy_correction_steps=3,  # Number of correction iterations
       energy_alpha=0.1,            # Step size
       energy_clip_frac=0.2,        # Maximum change
       correct_first_only=False,    # Correct all actions
   )
""")

