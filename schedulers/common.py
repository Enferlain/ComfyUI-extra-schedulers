import numpy as np
import torch


def get_sigmas_like_kdiffusion(model_sampling, n):
    """
    Replicate k-diffusion's get_sigmas(n) interpolation against the model sigma table.
    """
    sigmas = model_sampling.sigmas.detach().cpu().numpy()
    t_max = len(sigmas) - 1

    t = np.linspace(t_max, 0, n)
    low_idx = np.floor(t).astype(int)
    high_idx = np.ceil(t).astype(int)

    low_idx = np.clip(low_idx, 0, t_max)
    high_idx = np.clip(high_idx, 0, t_max)
    w = t - np.floor(t)

    return (1.0 - w) * sigmas[low_idx] + w * sigmas[high_idx]


def build_dense_native_sigmas(model_sampling, steps, density_factor=32, minimum_sigma_points=1000):
    density_factor = max(2, int(density_factor))
    minimum_sigma_points = max(int(steps) + 1, int(minimum_sigma_points))
    dense_points = max(minimum_sigma_points, int(steps) * density_factor)
    return get_sigmas_like_kdiffusion(model_sampling, dense_points + 1)


def sample_from_cumulative_density(base_sigmas, density, steps):
    """
    Sample `steps` sigma locations from a monotone base curve using a custom density.
    """
    base_sigmas = np.asarray(base_sigmas, dtype=np.float64)
    density = np.asarray(density, dtype=np.float64)
    steps = max(1, int(steps))

    if len(base_sigmas) < 2:
        return np.repeat(base_sigmas[:1], steps)

    density = np.clip(density, 0.0, None)
    if not np.any(density > 0):
        density = np.ones_like(density)

    cumulative = np.cumsum(density)
    cumulative -= cumulative[0]
    total = cumulative[-1]

    if total <= 0:
        cumulative = np.linspace(0.0, 1.0, len(base_sigmas))
    else:
        cumulative /= total

    targets = (np.arange(steps, dtype=np.float64) + 0.5) / steps
    positions = np.interp(targets, cumulative, np.arange(len(base_sigmas), dtype=np.float64))

    idx_floor = np.floor(positions).astype(int)
    idx_ceil = np.clip(idx_floor + 1, 0, len(base_sigmas) - 1)
    weights = positions - idx_floor

    return base_sigmas[idx_floor] * (1.0 - weights) + base_sigmas[idx_ceil] * weights


def compute_log_sigma_density(base_sigmas, exponent=1.0, floor=1.0e-12):
    sigmas = np.asarray(base_sigmas, dtype=np.float64)
    log_sigmas = np.log(np.maximum(sigmas, floor))
    delta = np.abs(np.diff(log_sigmas))
    density = np.concatenate([delta, delta[-1:]])
    exponent = max(0.01, float(exponent))
    return np.power(np.maximum(density, floor), exponent)


def compute_log_sigma_curvature_density(
    base_sigmas,
    slope_weight=1.0,
    curvature_weight=1.0,
    exponent=1.0,
    floor=1.0e-12,
):
    sigmas = np.asarray(base_sigmas, dtype=np.float64)
    log_sigmas = np.log(np.maximum(sigmas, floor))
    slope = np.abs(np.diff(log_sigmas))
    if slope.size == 0:
        return np.ones_like(sigmas)

    curvature = np.abs(np.diff(slope, prepend=slope[:1]))
    slope_full = np.concatenate([slope, slope[-1:]])
    curvature_full = np.concatenate([curvature, curvature[-1:]])

    density = (float(slope_weight) * slope_full) + (float(curvature_weight) * curvature_full)
    exponent = max(0.01, float(exponent))
    return np.power(np.maximum(density, floor), exponent)


def finalize_sigma_schedule(sigmas, model_sampling):
    sigma_min = float(model_sampling.sigma_min)
    sigma_max = float(model_sampling.sigma_max)

    result = np.asarray(sigmas, dtype=np.float32)
    if sigma_min < sigma_max:
        result = np.clip(result, sigma_min, sigma_max)

    result = np.minimum.accumulate(result)
    result = np.append(result, 0.0).astype(np.float32)
    return torch.from_numpy(result)
