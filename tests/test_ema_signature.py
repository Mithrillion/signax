from __future__ import annotations

import iisignature
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.random import default_rng
from numpy.lib.stride_tricks import sliding_window_view

from signax import logsignature, multi_signature_combine, signature, signature_combine
from signax.utils import compress, lyndon_words, unravel_signature
from signax.ema.ema_signatures import (
    scale_signature,
    ema_scaled_concat,
    ema_rolling_signature,
    _moving_window,
    ema_scaled_concat_right,
    ema_rolling_signature_transform,
    flatten_signature_stream,
)

rng = default_rng()

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

dummy = jnp.array(
    (
        [
            [
                [0.14024666, 1.14105411, -0.07038419, -0.16438929],
                [1.07464205, -0.88456035, -0.9034352, -1.57423315],
                [-1.50131919, -1.21303644, -1.17165841, 0.07051902],
                [-0.16596071, -1.35624276, 0.00736479, -1.57084809],
                [0.18260821, -0.27170315, -1.68847025, -0.49455531],
                [0.88457708, -1.90597544, 0.49220008, 0.24276723],
                [1.07776549, -0.83897105, -0.04295127, 0.30294634],
                [-0.05484876, -0.60599237, -0.11774495, 0.19043512],
                [0.62987134, 2.0291481, 1.54385575, -0.14819874],
                [0.03866277, 0.79375768, -0.02801477, 1.96228418],
                [0.08417827, -0.54904664, -0.15876974, -0.11788892],
                [-0.13267415, 1.27247522, 2.22899079, 1.28995949],
                [-1.19521727, 0.84741743, 0.45845532, -1.14985409],
                [0.09561451, -0.79266755, 0.79705505, -0.47623769],
                [0.59765993, 1.32588841, 1.10450921, -0.69181138],
                [0.56983277, 0.6137129, 0.69056099, 0.53824605],
                [-0.21567643, 0.05123328, 0.1340739, -0.71210021],
                [0.18650358, 0.12834083, 1.60464822, 1.37702413],
                [0.99578332, 1.70309675, 0.83267074, -0.55969057],
                [0.94477475, 1.46564023, -0.75341982, 0.4913519],
            ],
            [
                [-0.04499377, -0.4354317, 0.30081155, -0.40475747],
                [-0.03246431, -0.87998261, 1.68988994, -0.31507837],
                [1.42233775, 1.10545134, 1.91083935, 0.60057188],
                [0.80463161, -0.45308879, 1.3716943, 0.63138086],
                [-0.04475449, 1.2414561, -0.19587846, 0.22085203],
                [-0.18088055, 0.67016762, 0.4812716, -0.90649296],
                [-0.46941034, -1.65393717, 0.01015725, 1.5470266],
                [-1.03223149, -1.0520105, 0.49539543, 0.26768668],
                [-0.96999, 0.07000881, -0.91905888, -0.43331753],
                [-1.09934149, -2.16160142, -0.27672335, 0.51014415],
                [-0.17503229, 1.08492855, -0.63133996, -0.16663142],
                [0.5285148, 0.18384565, -0.61033177, 0.65476258],
                [-1.72785607, 0.22898451, -0.1936712, 0.53896009],
                [0.93292252, 0.83777802, -0.83609556, 1.85252035],
                [1.36450014, 0.50917111, 1.88881399, 1.10017925],
                [0.73495324, 1.92090169, 0.1323662, -0.61661194],
                [1.01104359, 0.90900591, 0.67802983, -2.24169837],
                [-1.46711102, 1.48078304, 1.69026133, 0.00843408],
                [-0.61306296, -3.60585698, -0.47245996, 0.53771631],
                [-0.14677228, -0.16692028, -0.79410499, -1.09716341],
            ],
        ]
    )
)


def test_scale_signature():
    # test scale_signature
    # S = [jnp.array([1] * 3), jnp.array([[1] * 3] * 3), jnp.array([[[1] * 3] * 3] * 3)]
    n_signatures = 2
    dim = 5
    S = [
        rng.standard_normal((n_signatures, dim)),
        rng.standard_normal((n_signatures, dim, dim)),
        rng.standard_normal((n_signatures, dim, dim, dim)),
    ]
    scale = 0.8
    S_scaled = scale_signature(S, scale)
    assert S_scaled[0].shape == S[0].shape
    assert S_scaled[1].shape == S[1].shape
    assert S_scaled[2].shape == S[2].shape
    assert jnp.allclose(S_scaled[0], S[0] * scale)
    assert jnp.allclose(S_scaled[1], S[1] * scale**2)
    assert jnp.allclose(S_scaled[2], S[2] * scale**3)


def test_ema_scaled_concat():
    # assumes test of signature passes
    n_signatures = 2
    dim = 5
    S1 = [
        rng.standard_normal((n_signatures, dim)),
        rng.standard_normal((n_signatures, dim, dim)),
        rng.standard_normal((n_signatures, dim, dim, dim)),
    ]
    S2 = [
        rng.standard_normal((n_signatures, dim)),
        rng.standard_normal((n_signatures, dim, dim)),
        rng.standard_normal((n_signatures, dim, dim, dim)),
    ]
    S3 = [
        rng.standard_normal((n_signatures, dim)),
        rng.standard_normal((n_signatures, dim, dim)),
        rng.standard_normal((n_signatures, dim, dim, dim)),
    ]
    scale = 0.9
    S1_scaled = scale_signature(S1, scale)
    S12 = jax.vmap(signature_combine)(S1_scaled, S2)
    S12_scaled = scale_signature(S12, scale)
    S123 = jax.vmap(signature_combine)(S12_scaled, S3)
    S2_scaled = scale_signature(S2, scale)
    S23 = jax.vmap(signature_combine)(S2_scaled, S3)
    S123_alt = ema_scaled_concat(S1, S23, 2, scale)

    assert S123[0].shape == S123_alt[0].shape
    assert S123[1].shape == S123_alt[1].shape
    assert S123[2].shape == S123_alt[2].shape

    assert np.allclose(S123[0], S123_alt[0])
    assert np.allclose(S123[1], S123_alt[1])
    assert np.allclose(S123[2], S123_alt[2])


def test_ema_rolling_signature():
    n_paths = 2
    path_len = 20
    channels = 4
    depth = 3
    factor = 0.9

    path = rng.standard_normal((n_paths, path_len, channels))

    # path = dummy.copy()

    rolling_sig = ema_rolling_signature(path, depth, factor)

    rolling_sig = jnp.concatenate(
        [x.reshape(*x.shape[:2], -1) for x in rolling_sig],
        axis=-1,
    )

    path_ = jnp.concatenate([path[:, :1], path], axis=1)
    sig = signature(path_[:, :2, :], depth, flatten=False)

    trace = [sig]
    for i in range(1, path_len):
        # sig = scale_signature(sig, factor)
        new_sig = signature(path_[:, i : i + 2, :], depth, flatten=False)
        # sig = jax.vmap(signature_combine)(sig, new_sig)
        sig = ema_scaled_concat(sig, new_sig, 1, factor)
        trace += [sig]

    rolling_sig_alt = jnp.stack(
        [jnp.concatenate([x.reshape((n_paths, -1)) for x in t], axis=-1) for t in trace]
    ).transpose(1, 0, 2)

    assert jnp.allclose(rolling_sig, rolling_sig_alt, atol=1e-4)


def test_sliding_window():
    path = dummy.copy()
    path_ = jnp.concatenate([path[:, :1], path], axis=1)
    jax_mw = _moving_window(path_, 2, 1)
    np_mw = sliding_window_view(path_, 2, 1).transpose(0, 1, 3, 2)
    assert jnp.allclose(jax_mw, np_mw)


def test_inverse_rolling_signature():
    n_paths = 2
    path_len = 20
    channels = 4
    depth = 3
    factor = 0.9

    # path = rng.standard_normal((n_paths, path_len, channels))

    path = dummy.copy()

    rolling_sig = ema_rolling_signature(path, depth, factor, inverse=True)

    rolling_sig = jnp.concatenate(
        [x.reshape(*x.shape[:2], -1) for x in rolling_sig],
        axis=-1,
    )

    path_ = jnp.concatenate([path, path[:, -1:]], axis=1)
    sig = signature(path_[:, -2:, :], depth, flatten=False)

    trace = [sig]
    for i in range(1, path_len):
        # sig = scale_signature(sig, factor)
        new_sig = signature(path_[:, -i - 2 : -i, :], depth, flatten=False)
        # sig = jax.vmap(signature_combine)(sig, new_sig)
        sig = ema_scaled_concat_right(new_sig, sig, 1, factor)
        trace += [sig]

    trace = trace[::-1]

    rolling_sig_alt = jnp.stack(
        [jnp.concatenate([x.reshape((n_paths, -1)) for x in t], axis=-1) for t in trace]
    ).transpose(1, 0, 2)

    assert jnp.allclose(rolling_sig, rolling_sig_alt, atol=1e-4)


def test_flatten_signature_stream():
    n_paths = 2
    path_len = 20
    channels = 4
    depth = 3

    sig_stream = [
        jnp.ones((n_paths, path_len, channels**i)) for i in range(1, depth + 1)
    ]
    flat_sig = flatten_signature_stream(sig_stream)
    flat_sig_alt = jnp.ones(
        (n_paths, path_len, sum([channels**i for i in range(1, depth + 1)]))
    )
    assert jnp.allclose(flat_sig, flat_sig_alt, atol=1e-4)


def test_ema_sig_transform():
    depth = 3
    factor = 0.9

    path = dummy.copy()
    sig = ema_rolling_signature_transform(path, depth, factor)
    # sig_flat = flatten_signature_stream(sig)
    assert True


# test_ema_rolling_signature()
# test_inverse_rolling_signature()
# test_ema_sig_transform()
# test_flatten_signature_stream()