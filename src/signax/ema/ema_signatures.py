from __future__ import annotations

import logging
from functools import partial
from typing import Callable, Union

from einops import rearrange, repeat
import jax
import jax.numpy as jnp
from jax import flatten_util
from jaxtyping import Array, Float
from scipy.signal.windows import tukey

from signax import utils
from signax.tensor_ops import log, mult, mult_fused_restricted_exp, restricted_exp
from signax.signatures import signature_combine, multi_signature_combine, signature

logger = logging.getLogger(__name__)


@partial(jax.jit, static_argnames=("dim",))
def scale_path(path: jax.Array, factor: float, dim: int = -2) -> jax.Array:
    """
    Scale a path by a factor along a given dimension.
    Args:
        path: array of shape (..., T, D)
        factor: float, the factor to scale the path by
        dim: int, the dimension to scale along
    Returns:
        array of shape (..., T, D)
    """
    if dim != -2:
        # swap dim and -2 axis
        path = jnp.swapaxes(path, dim, -2)
    # compute diff along time axis
    time_diff = jnp.diff(path, 1, axis=-2, prepend=path[..., [0], :])
    # scale the diff by [factor ** n, ..., factor ** 3, factor ** 2, factor ** 1, 1]
    time_diff = (
        time_diff * factor ** (jnp.arange(time_diff.shape[-2], 0, -1) - 1)[..., None]
    )
    # cumsum to get the scaled path
    path = jnp.cumsum(time_diff, axis=-2)
    # swap back if necessary
    if dim != -2:
        path = jnp.swapaxes(path, dim, -2)
    return path


@partial(
    jax.jit, static_argnames=("depth", "inverse", "stream", "flatten", "num_chunks")
)
def ema_signature(
    path: jax.Array,
    depth: int,
    factor: float,
    inverse: bool = False,
    stream: bool = False,
    flatten: bool = True,
    num_chunks: int = 1,
) -> jax.Array:
    """
    Compute the signature of a path scaled by a factor along a given dimension.
    Args:
        path: array of shape (..., T, D)
        depth: int, the depth of the signature
        factor: float, the factor to scale the path by
        inverse: bool, if True, compute the inversely weighted signature
        kwargs: additional arguments to pass to the signature function
    Returns:
        array of shape (..., C)
    """
    if inverse:
        path = scale_path(path[..., ::-1, :], factor)[..., ::-1, :]
        return signature(
            path, depth=depth, stream=stream, flatten=flatten, num_chunks=num_chunks
        )
    else:
        path = scale_path(path, factor)
        return signature(
            path, depth=depth, stream=stream, flatten=flatten, num_chunks=num_chunks
        )


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
    if jnp.ndim(sig_a[0]) == 1:
        return signature_combine(scaled_a, sig_b)
    else:
        return jax.vmap(signature_combine)(scaled_a, sig_b)


@jax.jit
def ema_scaled_concat_right(
    sig_a: list[jax.Array], sig_b: list[jax.Array], len_a: int, factor: float
) -> list[jax.Array]:
    scaled_b = scale_signature(sig_b, factor**len_a)
    if jnp.ndim(sig_a[0]) == 1:
        return signature_combine(sig_a, scaled_b)
    else:
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
    if jnp.ndim(sig_a[0]) == 1:
        return signature_combine(scaled_a, sig_b)
    else:
        return jax.vmap(signature_combine)(scaled_a, sig_b)


@jax.jit
def _ema_scaled_concat_right_list(
    sig_a: list[jax.Array], sig_b: list[jax.Array], len_a: list[int], factor: float
) -> list[jax.Array]:
    scaled_b = _scaled_signature_list(sig_b, factor**len_a)
    if jnp.ndim(sig_a[0]) == 1:
        return signature_combine(sig_a, scaled_b)
    else:
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
    if jnp.size(len_b) == 0:
        return sig_a, len_a + len_b
    sig_concat = _ema_scaled_concat_list(
        sig_a, sig_b, unsqueeze_zero_dim(len_b), factor
    )
    return sig_concat, len_a + len_b


@jax.jit
def _scale_concat_op_right(
    sig_b_n_len: tuple[list[jax.Array], int],
    sig_a_n_len: tuple[list[jax.Array], int],
    factor: float,
) -> tuple[list[jax.Array], int]:
    sig_a, len_a = sig_a_n_len
    sig_b, len_b = sig_b_n_len
    if jnp.size(len_a) == 0:
        return sig_b, len_a + len_b
    sig_concat = _ema_scaled_concat_right_list(
        sig_a, sig_b, unsqueeze_zero_dim(len_a), factor
    )
    return sig_concat, len_a + len_b


@partial(jax.jit, static_argnums=(1, 2))
def _moving_window(a: jax.Array, size: int, axis: int):
    starts = jnp.arange(a.shape[axis] - size + 1)
    return jax.vmap(
        lambda start: jax.lax.dynamic_slice_in_dim(a, start, size, axis), out_axes=1
    )(starts)


@partial(jax.jit, static_argnames=("depth", "inverse", "padding", "batch_size"))
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
def flatten_signature(signature: list[jax.Array]) -> jax.Array:
    return jnp.concatenate(
        [x.reshape(*x.shape[:1], -1) for x in signature],
        axis=-1,
    )


@jax.jit
def flatten_signature_stream(signature_stream: list[jax.Array]) -> jax.Array:
    return jnp.concatenate(
        [x.reshape(*x.shape[:2], -1) for x in signature_stream],
        axis=-1,
    )


@partial(jax.jit, static_argnames=("depth", "stride", "batch_size"))
def ema_rolling_signature_transform(
    path: jax.Array,
    depth: int,
    factor: float,
    stride: int = 1,
    batch_size: int = None,
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
        forward_rolling_sigs = ema_rolling_signature_strided(
            path, depth, factor, stride, batch_size=batch_size
        )
        backward_rolling_sigs = ema_rolling_signature_strided(
            path, depth, factor, stride, True, batch_size=batch_size
        )
    timewise_fn = jax.vmap(signature_combine)
    transformed = jax.vmap(timewise_fn)(forward_rolling_sigs, backward_rolling_sigs)
    return transformed


@partial(jax.jit, static_argnames=("depth", "stride", "inverse", "batch_size"))
def ema_rolling_signature_strided(
    path: jax.Array,
    depth: int,
    factor: float,
    stride: int,
    inverse: bool = False,
    batch_size: int = None,
) -> jax.Array:
    """
    Compute rolling decaying weight signature of a path, evaluated at strided points.

    Args:
       path: jax.Array, the path to compute the rolling signature of.
       depth: int, the depth of the signature.
       factor: float, the factor to use for the EMA transform.
       stride: int, the stride to use for the rolling signature.
       reverse: bool, whether to compute the inversely weighted signature or not
       batch_size: int, the batch size to use for the computation.
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
    # strided_ema_sig_fn = lambda x: [
    #     x[:, select_idx, ...]
    #     for x in ema_rolling_signature(x, depth, factor, inverse, padding=False)
    # ]
    strided_ema_sig_fn = lambda x: ema_signature(
        x, depth, factor, inverse, flatten=False
    )
    if batch_size is None:
        sig_elem = jax.vmap(strided_ema_sig_fn, 1, 1)(path_elem)
        segment_len = path_elem.shape[2]
        sig_d = jnp.ones(sig_elem[0].shape[:2], dtype=jnp.int32) * segment_len
    else:
        path_batches = [
            path_elem[path_elem_batch_idx : path_elem_batch_idx + batch_size]
            for path_elem_batch_idx in range(0, path_elem.shape[0], batch_size)
        ]
        sig_list = []
        for path_batch in path_batches:
            sig_elem_batch = jax.vmap(strided_ema_sig_fn, 1, 1)(path_batch)
            sig_list.append(sig_elem_batch)
        sig_elem = [jnp.concatenate(level, axis=0) for level in zip(*sig_list)]
        segment_len = path_elem.shape[2]
        sig_d = jnp.ones(sig_elem[0].shape[:2], dtype=jnp.int32) * segment_len

    strided_reduce_fn = lambda x, y: jax.lax.associative_scan(
        lambda u, t: scan_concat_op_fn(u, t, factor), (x, y), reverse=inverse
    )

    rolling_sig, lengths = jax.vmap(strided_reduce_fn)(sig_elem, sig_d)
    return rolling_sig


@partial(
    jax.jit,
    static_argnames=("depth", "window_len", "alpha", "batch_size", "num_chunks"),
)
def windowed_sliding_signature(
    path: jnp.ndarray,
    depth: int,
    window_len: int,
    alpha: float = 0.5,
    batch_size: int = None,
    num_chunks: int = 1,
) -> jnp.ndarray:
    """
    Sliding window signature with a window function.

    Args:
       path: jax.Array, the path to compute the rolling signature of.
       depth: int, the depth of the signature.
       window_len: lenth of the window. The window function is a Tukey window.
       alpha: float, the alpha parameter of the Tukey window. Default is 0.5.
       batch_size: int, the batch size to use for the computation.
       num_chunks: int, the number of chunks to split the path into. Default is 1.
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
    batch_sig_fn = jax.vmap(
        lambda x: signature(x, depth, flatten=False, num_chunks=num_chunks)
    )
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


def tree_transpose(list_of_trees):
    """
    Converts a list of trees of identical structure into a single tree of lists.
    """
    return jax.tree.map(lambda *xs: list(xs), *list_of_trees)


def _append_last_endpoint(carry, x):
    return x[:, [-1], :], jnp.concatenate(carry, x, axis=-2)


def indexed_ema_signature(
    path: jax.Array,
    depth: int,
    factor: float,
    indices: jax.Array,
    inverse: bool = False,
    padding: bool = True,
    # batch_size: int = None,
) -> jax.Array:
    """
    Compute the signature of a path scaled by a factor at specific indices along the time dimension.

    Args:
        path: The path to compute the signature of.
        depth: The depth of the signature.
        factor: The factor to scale the path by.
        indices: The indices at which to scale the path.
        inverse: Whether to compute the inverse signature.
        padding: Whether to pad the path to the nearest power of 2.
        # batch_size: The batch size to use for computing the signature.
    """
    if inverse:
        if padding:
            path = jnp.concatenate([path, path[:, -1:]], axis=1, dtype=path.dtype)
        # scan_concat_op_fn = _scale_concat_op_right
        scan_concat_op_fn = _debug_right_scan_fn
    else:
        if padding:
            path = jnp.concatenate([path[:, :1], path], axis=1, dtype=path.dtype)
        # scan_concat_op_fn = _scale_concat_op
        scan_concat_op_fn = _debug_scan_fn
    split_path = jnp.split(path, indices, axis=1)
    split_lengths = jnp.array(
        jax.tree_map(lambda x: x.shape[1], split_path), dtype=jnp.int32
    )
    split_path_ends = [x[..., [-1], :] for x in split_path[:-1]]
    split_path = [split_path[0]] + [
        jnp.concatenate([e, p], axis=1) for e, p in zip(split_path_ends, split_path[1:])
    ]

    split_ema_sigs = jax.tree_map(
        lambda x: ema_signature(x, depth, factor, inverse=inverse, flatten=False),
        split_path,
    )
    split_ema_sigs = tree_transpose(split_ema_sigs)
    split_ema_sigs = [jnp.stack(x, 1) for x in split_ema_sigs]

    # batch_reduce_fn = lambda x, y: jax.lax.associative_scan(
    #     lambda u, t: scan_concat_op_fn(u, t, factor), (x, y), reverse=inverse
    # )
    # rolling_sig, lengths = jax.vmap(batch_reduce_fn)(
    #     split_ema_sigs, repeat(split_lengths, "t -> b t", b=path.shape[0])
    # )

    # TODO: figure out why associative scan does not match the manual results

    # DEBUG
    # def batch_reduce_fn(sigs_lens):
    #     sigs, b_lens = sigs_lens[:-1], sigs_lens[-1]
    #     res = [s[[0], ...] for s in sigs]
    #     out = [res]
    #     for i, b_len in enumerate(b_lens[1:]):
    #         res = ema_scaled_concat(res, [s[[i + 1], ...] for s in sigs], b_len, factor)
    #         out.append(res)
    #     out = tree_transpose(out)
    #     out = [jnp.stack(x, 0) for x in out]
    #     return out

    # def batch_reduce_fn(sigs_lens):
    #     out = jax.lax.associative_scan(
    #         lambda u, t: scan_concat_op_fn((u[:-1], u[-1]), (t[:-1], t[-1]), factor),
    #         sigs_lens,
    #         reverse=inverse,
    #     )
    #     return out

    def batch_reduce_fn(sigs_lens):
        init = (
            *(s[0, ...] * 0 for s in sigs_lens[:-1]),
            jnp.array([], dtype=jnp.int32),
        )
        _, out = jax.lax.scan(
            lambda u, t: scan_concat_op_fn((u[:-1], u[-1]), (t[:-1], t[-1]), factor),
            init,
            sigs_lens,
            reverse=inverse,
        )
        return out

    rolling_sig = jax.vmap(batch_reduce_fn, in_axes=0)(
        (*split_ema_sigs, repeat(split_lengths, "t -> b t", b=path.shape[0]))
    )
    rolling_sig = rolling_sig[:-1]

    return rolling_sig


@jax.jit
def unsqueeze_zero_dim(x: jax.Array):
    if jnp.ndim(x) == 0:
        return jnp.array([x])
    return x


@jax.jit
def _debug_scan_fn(
    sig_a_n_len: tuple[list[jax.Array], int],
    sig_b_n_len: tuple[list[jax.Array], int],
    factor: float,
) -> tuple[list[jax.Array], int]:
    sig_a, len_a = sig_a_n_len
    sig_b, len_b = sig_b_n_len
    if jnp.ndim(len_a) == 0:
        return (*sig_b, len_b), (*sig_b, len_b)
    sig_concat = _ema_scaled_concat_list(
        sig_a, sig_b, unsqueeze_zero_dim(len_b), factor
    )
    return (*sig_concat, len_a + len_b), (*sig_concat, len_a + len_b)


@jax.jit
def _debug_right_scan_fn(
    sig_a_n_len: tuple[list[jax.Array], int],
    sig_b_n_len: tuple[list[jax.Array], int],
    factor: float,
) -> tuple[list[jax.Array], int]:
    sig_a, len_a = sig_a_n_len
    sig_b, len_b = sig_b_n_len
    if jnp.ndim(len_b) == 0:
        return (*sig_a, len_a), (*sig_a, len_a)
    sig_concat = _ema_scaled_concat_right_list(
        sig_a, sig_b, unsqueeze_zero_dim(len_a), factor
    )
    return (*sig_concat, len_a + len_b), (*sig_concat, len_a + len_b)


def _indexed_ema_signature_debug(
    path: jax.Array,
    depth: int,
    factor: float,
    indices: jax.Array,
    inverse: bool = False,
    padding: bool = True,
) -> jax.Array:
    if inverse:
        if padding:
            path = jnp.concatenate([path, path[:, -1:]], axis=1, dtype=path.dtype)
    else:
        if padding:
            path = jnp.concatenate([path[:, :1], path], axis=1, dtype=path.dtype)
    split_path = jnp.split(path, indices, axis=1)
    split_path_ends = [x[..., [-1], :] for x in split_path[:-1]]
    split_path = [split_path[0]] + [
        jnp.concatenate([e, p], axis=1) for e, p in zip(split_path_ends, split_path[1:])
    ]
    split_ema_sigs = jax.tree_map(
        lambda x: ema_signature(x, depth, factor, inverse=inverse, flatten=False),
        split_path,
    )
    split_ema_sigs = tree_transpose(split_ema_sigs)
    split_ema_sigs = [jnp.stack(x, 1) for x in split_ema_sigs]

    return split_ema_sigs
