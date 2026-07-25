"""
CRS-lite scheduler over the model's native sigma table.

This is a heuristic approximation inspired by Constant Rate Scheduling:
we redistribute steps over a dense native sigma curve to make progress in
log-sigma space more uniform.
"""

from .common import (
    build_dense_native_sigmas,
    compute_log_sigma_density,
    finalize_sigma_schedule,
    sample_from_cumulative_density,
)


def get_native_crs_lite_schedule(
    model_sampling,
    steps,
    density_factor=32,
    minimum_sigma_points=1000,
    rate_exponent=1.0,
):
    base_sigmas = build_dense_native_sigmas(
        model_sampling,
        steps,
        density_factor=density_factor,
        minimum_sigma_points=minimum_sigma_points,
    )
    density = compute_log_sigma_density(base_sigmas, exponent=rate_exponent)
    result_sigmas = sample_from_cumulative_density(base_sigmas, density, steps)
    return finalize_sigma_schedule(result_sigmas, model_sampling)
