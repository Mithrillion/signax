from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from signax import signature_combine
from signax.tensor_ops import log


# exp(x) = 1 + x + x/2(x + x/3(x + x/4(x + ...)))
@jax.jit
def tensor_exp(logsig: list[jax.Array]):
    result = jax.tree.map(jnp.zeros_like, logsig)
    depth = len(logsig)
    for level in range(depth, 0, -1):
        combined = signature_combine(logsig, result)
        result = jax.tree.map(
            lambda x, y, z, l: (x - y - z) / (level + 1) + l,
            combined,
            logsig,
            result,
            logsig,
        )
    return result


def invert_sig(sig: list[jax.Array]):
    # sig^-1=exp(-log(sig))
    return tensor_exp(jax.tree.map(jnp.negative, log(sig)))
