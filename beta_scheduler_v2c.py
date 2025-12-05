"""
Beta Scheduler V2C for ComfyUI
Based on "Beta Sampling is All You Need" [arXiv:2407.12173] (Lee et. al, 2024)
Improved version with stratified sampling and sigma bounds
"""

import torch
import numpy as np
from scipy import stats
import comfy.samplers

def get_beta_schedule_v2c(model_sampling, steps, alpha=0.6, beta=0.6):
    """
    Improved Beta scheduler based on "Beta Sampling is All You Need" [arXiv:2407.12173]
    Uses stratified sampling and honors sigma_min/sigma_max bounds
    """
    # Get base sigma schedule from inner model
    # In ComfyUI, "normal" scheduler matches the get_sigmas(n) behavior (linear t interpolation)
    base_sigmas = comfy.samplers.calculate_sigmas(model_sampling, "normal", steps + 1).cpu().numpy()
    total_timesteps = len(base_sigmas) - 1
    
    # Use stratified quantiles to avoid duplicates (like align_your_steps approach)
    quantiles = (np.arange(steps) + 0.5) / steps  # Mid-point sampling
    quantiles = np.clip(quantiles, 1e-6, 1 - 1e-6)  # Avoid endpoints
    
    # Beta inverse CDF transform
    beta_values = stats.beta.ppf(quantiles, alpha, beta)
    beta_indices = beta_values * (total_timesteps - 1)
    
    # Linear interpolation instead of rounding (preserves exact step count)
    result_sigmas = []
    for idx in beta_indices:
        i0 = int(np.floor(idx))
        i1 = min(i0 + 1, total_timesteps)
        weight = idx - i0
        
        sigma = float(base_sigmas[i0] * (1 - weight) + base_sigmas[i1] * weight)
        result_sigmas.append(sigma)
    
    # Honor sigma bounds (like other schedulers in your collection)
    sigma_min = float(model_sampling.sigma_min)
    sigma_max = float(model_sampling.sigma_max)
    if sigma_min < sigma_max:
        result_sigmas = [np.clip(s, sigma_min, sigma_max) for s in result_sigmas]
    
    # Ensure monotonic decrease (common pattern in your schedulers)
    for i in range(1, len(result_sigmas)):
        result_sigmas[i] = min(result_sigmas[i], result_sigmas[i-1])
    
    # Final sigma
    result_sigmas.append(0.0)
    
    return torch.FloatTensor(result_sigmas)


def get_beta_schedule_v2c_raw(model_sampling, steps, alpha=0.6, beta=0.6):
    """
    Improved Beta scheduler based on "Beta Sampling is All You Need" [arXiv:2407.12173]
    Uses stratified sampling and honors sigma_min/sigma_max bounds.
    Uses the model's raw sigmas as the base schedule.
    """
    # Get base sigma schedule from inner model (raw sigmas)
    base_sigmas = model_sampling.sigmas.cpu().numpy()
    total_timesteps = len(base_sigmas) - 1
    
    # Use stratified quantiles to avoid duplicates (like align_your_steps approach)
    quantiles = (np.arange(steps) + 0.5) / steps  # Mid-point sampling
    quantiles = np.clip(quantiles, 1e-6, 1 - 1e-6)  # Avoid endpoints
    
    # Beta inverse CDF transform
    beta_values = stats.beta.ppf(quantiles, alpha, beta)
    beta_indices = beta_values * (total_timesteps - 1)
    
    # Linear interpolation instead of rounding (preserves exact step count)
    result_sigmas = []
    for idx in beta_indices:
        i0 = int(np.floor(idx))
        i1 = min(i0 + 1, total_timesteps)
        weight = idx - i0
        
        sigma = float(base_sigmas[i0] * (1 - weight) + base_sigmas[i1] * weight)
        result_sigmas.append(sigma)
    
    # Honor sigma bounds (like other schedulers in your collection)
    sigma_min = float(model_sampling.sigma_min)
    sigma_max = float(model_sampling.sigma_max)
    if sigma_min < sigma_max:
        result_sigmas = [np.clip(s, sigma_min, sigma_max) for s in result_sigmas]
    
    # Ensure monotonic decrease (common pattern in your schedulers)
    for i in range(1, len(result_sigmas)):
        result_sigmas[i] = min(result_sigmas[i], result_sigmas[i-1])
    
    # Final sigma
    result_sigmas.append(0.0)
    
    return torch.FloatTensor(result_sigmas)


class BetaSchedulerV2C:
    """
    Custom ComfyUI node for Beta Scheduler V2C
    Provides adjustable alpha and beta parameters
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {
                    "default": 20, 
                    "min": 1, 
                    "max": 10000
                }),
                "alpha": ("FLOAT", {
                    "default": 0.6, 
                    "min": 0.01, 
                    "max": 10.0, 
                    "step": 0.01,
                    "display": "slider"
                }),
                "beta": ("FLOAT", {
                    "default": 0.6, 
                    "min": 0.01, 
                    "max": 10.0, 
                    "step": 0.01,
                    "display": "slider"
                }),
                "denoise": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01
                }),
            }
        }
    
    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    CATEGORY = "sampling/custom_sampling/schedulers"
    FUNCTION = "get_sigmas"
    
    def get_sigmas(self, model, steps, alpha, beta, denoise):
        """
        Generate sigma schedule using beta distribution
        
        Args:
            model: ComfyUI model object
            steps: Number of sampling steps
            alpha: Alpha parameter for beta distribution
            beta: Beta parameter for beta distribution
            denoise: Denoise strength (1.0 = full denoise)
        
        Returns:
            tuple: (sigmas tensor,)
        """
        total_steps = steps
        if denoise < 1.0:
            total_steps = int(steps / denoise)
        
        model_sampling = model.get_model_object("model_sampling")
        
        sigmas = get_beta_schedule_v2c(
            model_sampling, 
            total_steps, 
            alpha, 
            beta
        )
        
        if denoise < 1.0:
            sigmas = sigmas[-(steps + 1):]
        
        return (sigmas,)


class BetaSchedulerV2CRaw:
    """
    Custom ComfyUI node for Beta Scheduler V2C (Raw Sigmas)
    Provides adjustable alpha and beta parameters, using model's raw sigmas as base.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {
                    "default": 20, 
                    "min": 1, 
                    "max": 10000
                }),
                "alpha": ("FLOAT", {
                    "default": 0.6, 
                    "min": 0.01, 
                    "max": 10.0, 
                    "step": 0.01,
                    "display": "slider"
                }),
                "beta": ("FLOAT", {
                    "default": 0.6, 
                    "min": 0.01, 
                    "max": 10.0, 
                    "step": 0.01,
                    "display": "slider"
                }),
                "denoise": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 1.0, 
                    "step": 0.01
                }),
            }
        }
    
    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    CATEGORY = "sampling/custom_sampling/schedulers"
    FUNCTION = "get_sigmas"
    
    def get_sigmas(self, model, steps, alpha, beta, denoise):
        """
        Generate sigma schedule using beta distribution on raw sigmas
        """
        total_steps = steps
        if denoise < 1.0:
            total_steps = int(steps / denoise)
        
        model_sampling = model.get_model_object("model_sampling")
        
        sigmas = get_beta_schedule_v2c_raw(
            model_sampling, 
            total_steps, 
            alpha, 
            beta
        )
        
        if denoise < 1.0:
            sigmas = sigmas[-(steps + 1):]
        
        return (sigmas,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "BetaSchedulerV2C": BetaSchedulerV2C,
    "BetaSchedulerV2CRaw": BetaSchedulerV2CRaw,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BetaSchedulerV2C": "Beta Scheduler V2C",
    "BetaSchedulerV2CRaw": "Beta Scheduler V2C (Raw)",
}
