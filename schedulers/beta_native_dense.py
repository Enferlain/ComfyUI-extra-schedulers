"""
Beta scheduling over a dense interpolation of the model's own sigma table.
"""

import numpy as np
from scipy import stats

from .common import build_dense_native_sigmas, finalize_sigma_schedule


def get_beta_schedule_native_dense(
    model_sampling,
    steps,
    alpha=0.6,
    beta=0.6,
    density_factor=32,
    minimum_sigma_points=1000,
):
    """
    Build a dense base curve directly from model_sampling.sigmas, then beta-remap it.

    This keeps the model's native sigma table as the source of truth while avoiding the
    coarse snapping of the original V2C schedule.
    """
    base_sigmas = build_dense_native_sigmas(
        model_sampling,
        steps,
        density_factor=density_factor,
        minimum_sigma_points=minimum_sigma_points,
    )
    total_timesteps = len(base_sigmas) - 1

    quantiles = (np.arange(steps) + 0.5) / steps
    quantiles = np.clip(quantiles, 1e-6, 1.0 - 1e-6)

    beta_values = stats.beta.ppf(quantiles, alpha, beta)
    mapped_indices = beta_values * total_timesteps

    idx_floor = np.floor(mapped_indices).astype(int)
    idx_ceil = np.clip(idx_floor + 1, 0, total_timesteps)
    weights = mapped_indices - idx_floor

    result_sigmas = (
        base_sigmas[idx_floor] * (1.0 - weights) + base_sigmas[idx_ceil] * weights
    )

    return finalize_sigma_schedule(result_sigmas, model_sampling)
