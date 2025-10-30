from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
from flax import nnx
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._flops_logged = False

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        if not self._is_pytorch_model:
            # Ensure JAX computations are finished before timing
            _ = jax.tree.map(lambda x: x.block_until_ready(), outputs)
        model_time = time.monotonic() - start_time

        # One-time FLOPs estimation and model info logging (server side)
        if not self._flops_logged:
            try:
                total_params = 0
                if self._is_pytorch_model:
                    total_params = int(sum(p.numel() for p in self._model.parameters()))
                else:
                    # For JAX/NNX models, count leaves in the state tree
                    import numpy as _np
                    graphdef, state = nnx.split(self._model)
                    del graphdef
                    pure = state.to_pure_dict()
                    # Flatten dict values and sum sizes
                    def _sum_leaf(acc, leaf):
                        try:
                            if hasattr(leaf, "shape"):
                                return acc + int(_np.prod(leaf.shape))
                        except Exception:
                            return acc
                        return acc
                    for v in flax.traverse_util.flatten_dict(pure, sep="/").values():
                        try:
                            if hasattr(v, "shape"):
                                total_params += int(_np.prod(v.shape))
                        except Exception:
                            pass

                # Sequence length estimation
                seq_len = None
                if observation.tokenized_prompt is not None:
                    try:
                        seq_len = int(observation.tokenized_prompt.shape[-1])
                    except Exception:
                        seq_len = None
                if seq_len is None:
                    try:
                        seq_len = int(self._model.max_token_len)
                    except Exception:
                        seq_len = 1

                estimated_flops = float(2 * total_params * seq_len)
                infer_seconds = max(model_time, 1e-9)
                gflops = estimated_flops / infer_seconds / 1e9
                tflops = gflops / 1000.0

                print("=" * 80)
                try:
                    model_type_name = type(self._model).__name__
                except Exception:
                    model_type_name = "UnknownModel"
                print("[MODEL INFO]")
                print(f"  Model type: {model_type_name}")
                try:
                    any_image = next(iter(observation.images.values()))
                    print(f"  Input shape: image={getattr(any_image, 'shape', None)} state={getattr(observation.state, 'shape', None)}")
                except Exception:
                    pass

                print("\n[MODEL STATISTICS]")
                print(f"  Total Parameters: {total_params:,}")
                print("\n[FLOPS ESTIMATION] (Approximate)")
                print(f"  Estimated FLOPs per inference: {estimated_flops:.3e}")
                print("  Note: This is a rough estimate for transformer-like forward pass")
                print(f"  Estimated GFLOPS: {gflops:.2f} GFLOP/s")
                print(f"  Estimated TFLOPS: {tflops:.4f} TFLOP/s")
                print("=" * 80)
            except Exception as _e:
                logging.debug(f"FLOPs estimation skipped due to error: {_e}")
            finally:
                self._flops_logged = True
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
