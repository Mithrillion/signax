from __future__ import annotations

import logging
from functools import partial
from typing import Callable

from einops import rearrange
import jax
import jax.numpy as jnp
from jax import flatten_util
from jaxtyping import Array, Float
from scipy.signal.windows import tukey

from signax import utils
from signax.tensor_ops import log, mult, mult_fused_restricted_exp, restricted_exp
from signax.signatures import signature_combine, multi_signature_combine, signature

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


@jax.jit
def ema_scaled_concat_right(
    sig_a: list[jax.Array], sig_b: list[jax.Array], len_a: int, factor: float
) -> list[jax.Array]:
    scaled_b = scale_signature(sig_b, factor**len_a)
    return jax.vmap(signature_combine)(sig_a, scaled_b)


@jax.jit
def _scaled_signature_list(
    signature: list[jax.Array], factor: list[float]
) -> list[jax.Array]:
    return [factor[i] ** (i + 1) * level for i, level in enumerate(signature)]


@jax.jit
def _ema_scaled_concat_list(
    sig_a: list[jax.Array], sig_b: list[jax.Array], len_b: list[int], factor: float
) -> list[jax.Array]:
    scaled_a = _scaled_signature_list(sig_a, factor**len_b)
    return jax.vmap(signature_combine)(scaled_a, sig_b)


@jax.jit
def _ema_scaled_concat_right_list(
    sig_a: list[jax.Array], sig_b: list[jax.Array], len_a: list[int], factor: float
) -> list[jax.Array]:
    scaled_b = _scaled_signature_list(sig_b, factor**len_a)
    return jax.vmap(signature_combine)(sig_a, scaled_b)


# TODO: confusion between batched signatures [(b, d), (b, d, d), ...]
# and non-batched [(d,), (d, d), ...] may cause silent errors here


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
    sig_concat = _ema_scaled_concat_list(sig_a, sig_b, len_b, factor)
    return sig_concat, len_a + len_b


@jax.jit
def _scale_concat_op_right(
    sig_b_n_len: tuple[list[jax.Array], int],
    sig_a_n_len: tuple[list[jax.Array], int],
    factor: float,
) -> tuple[list[jax.Array], int]:
    sig_a, len_a = sig_a_n_len
    sig_b, len_b = sig_b_n_len
    if len(len_a) == 0:
        return sig_b, len_a + len_b
    sig_concat = _ema_scaled_concat_right_list(sig_a, sig_b, len_a, factor)
    return sig_concat, len_a + len_b


@partial(jax.jit, static_argnums=(1, 2))
def _moving_window(a: jax.Array, size: int, axis: int):
    starts = jnp.arange(a.shape[axis] - size + 1)
    return jax.vmap(
        lambda start: jax.lax.dynamic_slice_in_dim(a, start, size, axis), out_axes=1
    )(starts)


def ema_rolling_signature(
    path: jax.Array,
    depth: int,
    factor: float,
    inverse: bool = False,
    padding: bool = True,
    batch_size: int = None,
) -> list[jax.Array]:
    """
    Compute rolling decaying weight signature of a path.

    Args:
     path: array of shape (batch, length, channel), the path to compute the signature of
     depth: int, the depth of the signature
     factor: float, the weight decay factor per time step
     inverse: bool, whether to compute the inversely weighted signature or not
     padding: bool, whether to pad the signature with edge values or not
     batch_size: int, the processing size to break path instance dimension into
    """
    # duplicate first element along length dimension to the left
    if inverse:
        if padding:
            path = jnp.concatenate([path, path[:, -1:]], axis=1, dtype=path.dtype)
        scan_concat_op_fn = _scale_concat_op_right
    else:
        if padding:
            path = jnp.concatenate([path[:, :1], path], axis=1, dtype=path.dtype)
        scan_concat_op_fn = _scale_concat_op
    # unfold path to sliding window of size 2 along t dimension
    path_elem = _moving_window(path, 2, 1)
    # compute the signature
    batch_sig_fn = jax.vmap(lambda x: signature(x, depth, flatten=False))
    if batch_size is None:
        sig_elem = jax.vmap(batch_sig_fn)(path_elem)
        sig_d = jnp.ones(sig_elem[0].shape[:2], dtype=jnp.int32)
    else:
        path_batches = [
            path_elem[path_elem_batch_idx : path_elem_batch_idx + batch_size]
            for path_elem_batch_idx in range(0, path_elem.shape[0], batch_size)
        ]
        sig_list = []
        for path_batch in path_batches:
            sig_elem_batch = jax.vmap(batch_sig_fn)(path_batch)
            sig_list.append(sig_elem_batch)
        sig_d = jnp.ones((path_elem.shape[0], sig_list[0][0].shape[1]), dtype=jnp.int32)

    # shared associative scan (reducion) step
    batch_reduce_fn = lambda x, y: jax.lax.associative_scan(
        lambda u, t: scan_concat_op_fn(u, t, factor), (x, y), reverse=inverse
    )

    if batch_size is None:
        rolling_sig, lengths = jax.vmap(batch_reduce_fn)(sig_elem, sig_d)
    else:
        # break (sig_elem, sig_d) into list of batches [(sig_elem_batch, sig_d_batch) ...]
        sig_batches = [
            (
                path_batch,
                sig_d[sig_elem_batch_idx : sig_elem_batch_idx + batch_size],
            )
            for path_batch, sig_elem_batch_idx in zip(
                sig_list,
                range(
                    0,
                    path_elem.shape[0],
                    batch_size,
                ),
            )
        ]  # depends on JAX out of bound index behaviour
        rolling_sig_list = []
        for sig_elem_batch, sig_d_batch in sig_batches:
            rolling_sig_batch, lengths_batch = jax.vmap(batch_reduce_fn)(
                sig_elem_batch, sig_d_batch
            )
            rolling_sig_list.append(rolling_sig_batch)
        rolling_sig = [
            jnp.concatenate(terms, axis=0) for terms in zip(*rolling_sig_list)
        ]  # concatenate batches
    return rolling_sig


@jax.jit
def flatten_signature_stream(signature_stream: jax.Array) -> jax.Array:
    return jnp.concatenate(
        [x.reshape(*x.shape[:2], -1) for x in signature_stream],
        axis=-1,
    )


def ema_rolling_signature_transform(
    path: jax.Array, depth: int, factor: float, stride: int = 1, batch_size: int = None
) -> jax.Array:
    """Compute the rolling signature of a path using the EMA transform.
    Args:
       path: jax.Array, the path to compute the rolling signature of.
       depth: int, the depth of the signature.
       factor: float, the factor to use for the EMA transform.
       stride: int, the stride to use for the rolling signature.
       batch_size: int, the batch size to use for the rolling signature. If None, no batching is used.
    """
    if stride == 1:
        forward_rolling_sigs = ema_rolling_signature(
            path, depth, factor, batch_size=batch_size
        )
        # path_ = jnp.concatenate([path[:, 1:], path[:, -1:]], axis=1, dtype=path.dtype)
        backward_rolling_sigs = ema_rolling_signature(
            path, depth, factor, True, batch_size=batch_size
        )
    else:
        if batch_size is not None:
            raise NotImplementedError(
                "Strided EMA rolling signatures are not implemented for batched paths."
                "Please set batch_size=None."
            )
        forward_rolling_sigs = ema_rolling_signature_strided(
            path, depth, factor, stride
        )
        backward_rolling_sigs = ema_rolling_signature_strided(
            path, depth, factor, stride, True
        )
    timewise_fn = jax.vmap(signature_combine)
    transformed = jax.vmap(timewise_fn)(forward_rolling_sigs, backward_rolling_sigs)
    return transformed


def ema_rolling_signature_strided(
    path: jax.Array, depth: int, factor: float, stride: int, inverse: bool = False
) -> jax.Array:
    """
    Compute rolling decaying weight signature of a path, evaluated at strided points.

    Args:
       path: jax.Array, the path to compute the rolling signature of.
       depth: int, the depth of the signature.
       factor: float, the factor to use for the EMA transform.
       stride: int, the stride to use for the rolling signature.
       reverse: bool, whether to compute the inversely weighted signature or not
    """
    path_len = path.shape[1]
    # find the smallest multiple of stride that is greater than or equal to path_len
    padded_len = ((path_len + stride - 1) // stride) * stride
    # pad left with first value and right with last value
    pad_left = (padded_len - path_len) // 2
    pad_right = padded_len - path_len - pad_left
    # configure the scan function
    if inverse:
        path = jnp.pad(
            path, ((0, 0), (pad_left, pad_right + stride), (0, 0)), mode="edge"
        )
        scan_concat_op_fn = _scale_concat_op_right
        select_idx = 0
        # break into non-overlapping windows
        path_elem = rearrange(path, "b (w t) c -> b w t c", t=stride)
        # append the last entry of the previous window to each current window
        # pad the first window with the first entry of the first window
        prev_path_tail = jnp.concatenate(
            [path_elem[:, :1, :1, :], path_elem[:, :-1, -1:, :]],
            axis=1,
            dtype=path.dtype,
        )
        path_elem = jnp.concatenate(
            [
                prev_path_tail,
                path_elem,
            ],
            axis=2,
            dtype=path.dtype,
        )
    else:
        path = jnp.pad(
            path, ((0, 0), (pad_left + stride, pad_right), (0, 0)), mode="edge"
        )
        scan_concat_op_fn = _scale_concat_op
        select_idx = -1
        # break into non-overlapping windows
        path_elem = rearrange(path, "b (w t) c -> b w t c", t=stride)
        # append the first entry of the next window to each current window
        # pad the last window with the last entry of the last window
        next_path_head = jnp.concatenate(
            [path_elem[:, 1:, :1, :], path_elem[:, -1:, -1:, :]],
            axis=1,
            dtype=path.dtype,
        )
        path_elem = jnp.concatenate(
            [
                path_elem,
                next_path_head,
            ],
            axis=2,
            dtype=path.dtype,
        )

    # compute ema signatures per path segment
    batch_ema_sig_fn = lambda x: [
        x[:, select_idx, ...]
        for x in ema_rolling_signature(x, depth, factor, inverse, padding=False)
    ]
    sig_elem = jax.vmap(batch_ema_sig_fn, 1, 1)(path_elem)
    segment_len = path_elem.shape[2]
    sig_d = jnp.ones(sig_elem[0].shape[:2]) * segment_len

    batch_reduce_fn = lambda x, y: jax.lax.associative_scan(
        lambda u, t: scan_concat_op_fn(u, t, factor), (x, y), reverse=inverse
    )

    rolling_sig, lengths = jax.vmap(batch_reduce_fn)(sig_elem, sig_d)
    return rolling_sig


def windowed_sliding_signature(
    path: jnp.ndarray,
    depth: int,
    window_len: int,
    alpha: float = 0.5,
    batch_size: int = None,
) -> jnp.ndarray:
    """
    Sliding window signature with a window function.

    Args:
       path: jax.Array, the path to compute the rolling signature of.
       depth: int, the depth of the signature.
       window_len: lenth of the window. The window function is a Tukey window.
       alpha: float, the alpha parameter of the Tukey window. Default is 0.5.
       batch_size: int, the batch size to use for the computation.
    """
    # Ensure the window length is odd
    window_len = window_len + 1 if window_len % 2 == 0 else window_len
    window = jnp.array(tukey(window_len, alpha=alpha))
    # Pad the path with the window length
    padding_len = window_len // 2
    padded_path = jnp.pad(
        path, ((0, 0), (padding_len, padding_len), (0, 0)), mode="edge"
    )
    # unfold path to sliding window of size window_len along t dimension
    path_elem = _moving_window(padded_path, window_len, 1)
    # apply window function
    path_elem = path_elem * window[None, :, None]
    # compute the signature
    batch_sig_fn = jax.vmap(lambda x: signature(x, depth, flatten=False))
    if batch_size is None:
        sigs = jax.vmap(batch_sig_fn)(path_elem)
    else:
        path_batches = [
            path_elem[path_elem_batch_idx : path_elem_batch_idx + batch_size]
            for path_elem_batch_idx in range(0, path_elem.shape[0], batch_size)
        ]
        sig_list = []
        for path_batch in path_batches:
            sig_elem_batch = jax.vmap(batch_sig_fn)(path_batch)
            sig_list.append(sig_elem_batch)
        sigs = [
            jnp.concatenate(terms, axis=0) for terms in zip(*sig_list)
        ]  # concatenate batches
    return sigs
