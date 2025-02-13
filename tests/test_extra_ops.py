from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from numpy.random import default_rng

from signax import signature, signature_to_logsignature, signature_combine
from signax.tensor_ops import log
from signax.ema.extra_ops import tensor_exp, invert_sig

rng = default_rng()


jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)


def test_tensor_exp():
    n_paths = 10
    path_len = 492
    channels = 5
    depth = 4

    path = rng.standard_normal((n_paths, path_len, channels))
    S = signature(path, depth, flatten=False)

    L = jax.vmap(log)(S)
    S_ = jax.vmap(tensor_exp)(L)

    assert np.allclose(S[0], S_[0])
    assert np.allclose(S[1], S_[1])
    assert np.allclose(S[2], S_[2])
    assert np.allclose(S[3], S_[3])


def test_invert_sig():
    n_paths = 10
    path_len = 492
    channels = 5
    depth = 4

    path = rng.standard_normal((n_paths, path_len, channels))
    S = signature(path, depth, flatten=False)
    iS = jax.vmap(invert_sig)(S)
    combined = jax.vmap(signature_combine)(S, iS)
    assert np.allclose(combined[0], jnp.zeros_like(S[0]))
    assert np.allclose(combined[1], jnp.zeros_like(S[1]))
    assert np.allclose(combined[2], jnp.zeros_like(S[2]))
    assert np.allclose(combined[3], jnp.zeros_like(S[3]))


test_invert_sig()
