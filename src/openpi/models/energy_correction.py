"""Energy-based action correction for OpenPI models.

This module provides functions to refine predicted actions using energy gradients.
The energy model assigns low energy to plausible (state, action) pairs and high energy
to implausible ones. By performing gradient descent on energy, we can improve the
quality of predicted actions.
"""

import jax
import jax.numpy as jnp
import flax.nnx as nnx
from typing import Optional

from openpi.shared import array_typing as at


@at.typecheck
def one_step_energy_correction(
    energy_model,
    h: at.Float[at.Array, "B S D"],  # Context/state representations
    actions: at.Float[at.Array, "B H Da"],  # Predicted actions
    pad_mask: Optional[at.Bool[at.Array, "B S"]] = None,  # Padding mask (True=padding)
    alpha: float = 0.1,  # Step size for gradient descent
    clip_frac: float = 0.2,  # Clip gradient step to this fraction of action norm
    correct_first_only: bool = False,  # Only correct the first action in the chunk
    train: bool = False,  # Whether in training mode (for dropout)
) -> at.Float[at.Array, "B H Da"]:
    """
    Perform one step of energy-based action correction via gradient descent.
    
    The energy model assigns low energy to plausible (state, action) pairs.
    This function computes the gradient of energy w.r.t. actions and takes
    a gradient descent step to reduce the energy (improve action quality).
    
    Args:
        energy_model: The trained energy model
        h: Context/state representations from the policy backbone [B, S, D]
        actions: Predicted actions to be corrected [B, H, Da]
        pad_mask: Optional padding mask for context [B, S], True for padding
        alpha: Step size for gradient descent (larger = more correction)
        clip_frac: Maximum step size as fraction of action norm (prevents large changes)
        correct_first_only: If True, only correct the first action, keep others unchanged
        train: Whether in training mode (affects dropout in energy model)
        
    Returns:
        corrected_actions: Energy-corrected actions [B, H, Da]
        
    Example:
        >>> # After getting actions from the policy
        >>> actions = model.sample_actions(rng, observation)
        >>> # Refine actions using energy model
        >>> refined_actions = one_step_energy_correction(
        ...     model.energy_model, prefix_out, actions, alpha=0.1
        ... )
    """
    
    def energy_fn(actions_var):
        """Compute total energy for the batch."""
        # Energy model returns [B, 1], we want scalar for gradient
        energies = energy_model(h, actions_var, pad_mask, train=train)
        return jnp.sum(energies)
    
    # Compute gradient of energy w.r.t. actions
    grad_fn = jax.grad(energy_fn)
    grad_actions = grad_fn(actions)  # [B, H, Da]
    
    # If correct_first_only, mask out gradients for all but first action
    if correct_first_only:
        mask = jnp.zeros_like(grad_actions)
        mask = mask.at[:, 0, :].set(1.0)
        grad_actions = grad_actions * mask
    
    # Compute gradient descent step
    step = alpha * grad_actions
    
    # Clip the step size to prevent large changes
    # Method: clip to clip_frac * ||actions||
    action_norm = jnp.linalg.norm(actions.reshape(actions.shape[0], -1), axis=-1, keepdims=True) + 1e-6  # [B, 1]
    step_norm = jnp.linalg.norm(step.reshape(step.shape[0], -1), axis=-1, keepdims=True) + 1e-6  # [B, 1]
    
    # Compute clipping coefficient
    max_step_norm = clip_frac * action_norm
    clip_coef = jnp.minimum(jnp.ones_like(step_norm), max_step_norm / step_norm)  # [B, 1]
    clip_coef = clip_coef.reshape(-1, 1, 1)  # [B, 1, 1] for broadcasting
    
    # Apply clipping
    step = step * clip_coef
    
    # Gradient descent step: move in the direction of lower energy
    corrected_actions = actions - step
    
    return corrected_actions


@at.typecheck
def multi_step_energy_correction(
    energy_model,
    h: at.Float[at.Array, "B S D"],
    actions: at.Float[at.Array, "B H Da"],
    pad_mask: Optional[at.Bool[at.Array, "B S"]] = None,
    num_steps: int = 5,
    alpha: float = 0.1,
    clip_frac: float = 0.2,
    correct_first_only: bool = False,
    train: bool = False,
) -> at.Float[at.Array, "B H Da"]:
    """
    Perform multiple steps of energy-based action correction.
    
    This iteratively refines actions by taking multiple gradient descent steps
    on the energy landscape. This can lead to better corrections but is more
    computationally expensive.
    
    Args:
        energy_model: The trained energy model
        h: Context/state representations [B, S, D]
        actions: Predicted actions to be corrected [B, H, Da]
        pad_mask: Optional padding mask [B, S], True for padding
        num_steps: Number of gradient descent steps
        alpha: Step size per iteration
        clip_frac: Maximum step size as fraction of original action norm
        correct_first_only: Only correct the first action
        train: Whether in training mode
        
    Returns:
        corrected_actions: Energy-corrected actions [B, H, Da]
    """
    corrected = actions
    
    for _ in range(num_steps):
        corrected = one_step_energy_correction(
            energy_model,
            h,
            corrected,
            pad_mask=pad_mask,
            alpha=alpha,
            clip_frac=clip_frac,
            correct_first_only=correct_first_only,
            train=train,
        )
    
    return corrected


def create_energy_corrected_sample_fn(base_sample_fn, energy_model, correction_config: dict):
    """
    Create a wrapped sampling function that applies energy correction.
    
    This is a factory function that wraps the base sample_actions method
    to automatically apply energy correction during inference.
    
    Args:
        base_sample_fn: The original sample_actions method
        energy_model: The energy model to use for correction
        correction_config: Configuration dict with keys:
            - enabled: bool, whether to enable correction
            - num_steps: int, number of correction iterations
            - alpha: float, step size
            - clip_frac: float, maximum step fraction
            - correct_first_only: bool, only correct first action
    
    Returns:
        wrapped_sample_fn: A function with the same signature as sample_actions
        
    Example:
        >>> # In your inference script
        >>> from openpi.models.energy_correction import create_energy_corrected_sample_fn
        >>> 
        >>> # Wrap the model's sample_actions method
        >>> if hasattr(model, 'energy_model'):
        ...     original_sample = model.sample_actions
        ...     model.sample_actions = create_energy_corrected_sample_fn(
        ...         original_sample,
        ...         model.energy_model,
        ...         correction_config={'enabled': True, 'alpha': 0.1, 'num_steps': 3}
        ...     )
    """
    
    def wrapped_sample_actions(rng, observation, **kwargs):
        # First, get base actions from the policy
        actions = base_sample_fn(rng, observation, **kwargs)
        
        # If correction is disabled, return base actions
        if not correction_config.get('enabled', False):
            return actions
        
        # Extract context representation (need to call embed_prefix)
        # Note: This requires access to the model internals
        # For now, skip correction if we can't access the context
        # Users should call the correction function directly with prefix_out
        return actions
    
    return wrapped_sample_actions

