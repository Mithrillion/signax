from __future__ import annotations

import logging
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp
from jax import flatten_util
from jaxtyping import Array, Float

from signax import utils
from signax.tensor_ops import log, mult, mult_fused_restricted_exp, restricted_exp
from signax.signatures import signature_combine, multi_signature_combine, signature

from einops import rearrange

logger = logging.getLogger(__name__)


@jax.jit
def scale_signature(signature: list[jax.Array], factor: float) -> list[jax.Array]:
    """
    Scales a signature in graded tensor format by a factor,
    where each level is scaled by the factor to the power of the level.

    Args:
       signature: list of arrays, each array is a level of the signature
       factor: float, the factor to scale the signature by
    """
    return [factor ** (i + 1) * level for i, level in enumerate(signature)]


@jax.jit
def ema_scaled_concat(
    sig_a: list[jax.Array], sig_b: list[jax.Array], len_b: int, factor: float
) -> list[jax.Array]:
    """
    Concatenates two signatures in graded tensor format, where the first signature is scaled by a factor raised to the power of the length of the second signature.

    Args:
      sig_a: list of arrays, each array is a level of the first signature
      sig_b: list of arrays, each array is a level of the second signature
      len_b: int, the length of the second signature
      factor: float, the factor to scale the first signature by
    """
    scaled_a = scale_signature(sig_a, factor**len_b)
    return jax.vmap(signature_combine)(scaled_a, sig_b)


# @jax.jit
# def _scaled_signature(
#     signature: list[jax.Array], factor: list[float]
# ) -> list[jax.Array]:
#     return [factor[i] ** (i + 1) * level for i, level in enumerate(signature)]


# @jax.jit
# def _ema_scaled_concat(
#     sig_a: list[jax.Array], sig_b: list[jax.Array], len_b: list[int], factor: float
# ) -> list[jax.Array]:
#     scaled_a = _scaled_signature(sig_a, factor**len_b)
#     return jax.vmap(signature_combine)(scaled_a, sig_b)


@jax.jit
def _scale_concat_op(
    sig_a_n_len: tuple[list[jax.Array], int],
    sig_b_n_len: tuple[list[jax.Array], int],
    factor: float,
) -> tuple[list[jax.Array], int]:
    sig_a, len_a = sig_a_n_len
    sig_b, len_b = sig_b_n_len
    if len(len_b) == 0:
        return sig_a, len_a + len_b
    sig_concat = ema_scaled_concat(sig_a, sig_b, jnp.max(len_b), factor)
    return sig_concat, len_a + len_b


@partial(jax.jit, static_argnums=(1, 2))
def _moving_window(a: jax.Array, size: int, axis: int):
    starts = jnp.arange(a.shape[axis] - size + 1)
    return jax.vmap(
        lambda start: jax.lax.dynamic_slice_in_dim(a, start, size, axis), out_axes=1
    )(starts)


def ema_rolling_signature(
    path: jax.Array, depth: int, factor: float, scan: bool = True
) -> list[jax.Array]:
    """
    Compute rolling decaying weight signature of a path.

    Args:
     path: array of shape (batch, length, channel), the path to compute the signature of
     depth: int, the depth of the signature
     factor: float, the weight decay factor per time step
     scan: bool, whether to use associative scan or not
    """
    # duplicate first element along length dimension to the left
    path = jnp.concatenate([path[:, :1], path], axis=1)
    # unfold path to sliding window of size 2 along t dimension
    path_elem = _moving_window(path, 2, 1)
    # compute the signature
    batch_sig_fn = jax.vmap(lambda x: signature(x, depth, flatten=False))
    sig_elem = jax.vmap(batch_sig_fn)(path_elem)

    if scan:
        sig_d = jnp.ones(sig_elem[0].shape[:2])

        batch_reduce_fn = lambda x, y: jax.lax.associative_scan(
            lambda u, t: _scale_concat_op(u, t, factor), (x, y)
        )

        rolling_sig, lengths = jax.vmap(batch_reduce_fn)(sig_elem, sig_d)
    else:
        trace = [[e[:, 0, ...] for e in sig_elem]]
        for i in range(1, sig_elem[0].shape[1]):
            trace += [
                jax.vmap(lambda x, y: ema_scaled_concat(x, y, 1, factor))(
                    trace[-1], [e[:, i, ...] for e in sig_elem]
                )
            ]
        t_dim = len(trace)
        rolling_sig = [jnp.stack([trace[t][d] for t in range(t_dim)], 1) for d in range(depth)]
    return rolling_sig
