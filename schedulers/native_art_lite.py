"""
ART-lite scheduler over the model's native sigma table.

This is a heuristic trajectory-aware grid: it allocates more steps where the
native log-sigma trajectory changes rapidly or bends more sharply.
"""

from .common import (
    build_dense_native_sigmas,
    compute_log_sigma_curvature_density,
    finalize_sigma_schedule,
    sample_from_cumulative_density,
)


def get_native_art_lite_schedule(
    model_sampling,
    steps,
    density_factor=32,
    minimum_sigma_points=1000,
    slope_weight=1.0,
    curvature_weight=2.0,
    response_exponent=1.0,
):
    base_sigmas = build_dense_native_sigmas(
        model_sampling,
        steps,
        density_factor=density_factor,
        minimum_sigma_points=minimum_sigma_points,
    )
    density = compute_log_sigma_curvature_density(
        base_sigmas,
        slope_weight=slope_weight,
        curvature_weight=curvature_weight,
        exponent=response_exponent,
    )
    result_sigmas = sample_from_cumulative_density(base_sigmas, density, steps)
    return finalize_sigma_schedule(result_sigmas, model_sampling)
