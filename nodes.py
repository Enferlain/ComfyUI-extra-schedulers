"""
ComfyUI node definitions for the custom_schedulers package.
"""

import torch

from .beta_scheduler_v2c import get_beta_schedule_v2c, get_beta_schedule_v3
from .schedulers.beta_native_dense import get_beta_schedule_native_dense
from .schedulers.native_art_lite import get_native_art_lite_schedule
from .schedulers.native_crs_lite import get_native_crs_lite_schedule


def _resolve_total_steps(steps, denoise):
    if denoise <= 0.0:
        return 0
    total_steps = steps
    if denoise < 1.0:
        total_steps = int(steps / denoise)
    return total_steps


def _trim_for_denoise(sigmas, steps, denoise):
    if denoise < 1.0:
        sigmas = sigmas[-(steps + 1) :]
    return sigmas


class BetaSchedulerV2C:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "alpha": (
                    "FLOAT",
                    {"default": 0.6, "min": 0.01, "max": 10.0, "step": 0.01},
                ),
                "beta": (
                    "FLOAT",
                    {"default": 0.6, "min": 0.01, "max": 10.0, "step": 0.01},
                ),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    CATEGORY = "sampling/custom_sampling/schedulers"
    FUNCTION = "get_sigmas"

    def get_sigmas(self, model, steps, alpha, beta, denoise):
        total_steps = _resolve_total_steps(steps, denoise)
        if total_steps == 0:
            return (torch.FloatTensor([]),)
        model_sampling = model.get_model_object("model_sampling")
        sigmas = get_beta_schedule_v2c(model_sampling, total_steps, alpha, beta)
        return (_trim_for_denoise(sigmas, steps, denoise),)


class BetaSchedulerV3:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "alpha": (
                    "FLOAT",
                    {"default": 0.6, "min": 0.01, "max": 10.0, "step": 0.01},
                ),
                "beta": (
                    "FLOAT",
                    {"default": 0.6, "min": 0.01, "max": 10.0, "step": 0.01},
                ),
                "base_scheduler": (
                    ["normal", "karras", "exponential", "simple", "ddim_uniform"],
                ),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    CATEGORY = "sampling/custom_sampling/schedulers"
    FUNCTION = "get_sigmas"

    def get_sigmas(self, model, steps, alpha, beta, base_scheduler, denoise):
        total_steps = _resolve_total_steps(steps, denoise)
        if total_steps == 0:
            return (torch.FloatTensor([]),)
        model_sampling = model.get_model_object("model_sampling")
        sigmas = get_beta_schedule_v3(
            model_sampling, total_steps, alpha, beta, base_scheduler
        )
        return (_trim_for_denoise(sigmas, steps, denoise),)


class BetaSchedulerNativeDense:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {"default": 28, "min": 1, "max": 10000}),
                "alpha": (
                    "FLOAT",
                    {"default": 0.6, "min": 0.01, "max": 10.0, "step": 0.01},
                ),
                "beta": (
                    "FLOAT",
                    {"default": 0.6, "min": 0.01, "max": 10.0, "step": 0.01},
                ),
                "density_factor": (
                    "INT",
                    {"default": 32, "min": 2, "max": 512},
                ),
                "minimum_sigma_points": (
                    "INT",
                    {"default": 1000, "min": 64, "max": 200000},
                ),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    CATEGORY = "sampling/custom_sampling/schedulers"
    FUNCTION = "get_sigmas"

    def get_sigmas(
        self,
        model,
        steps,
        alpha,
        beta,
        density_factor,
        minimum_sigma_points,
        denoise,
    ):
        total_steps = _resolve_total_steps(steps, denoise)
        if total_steps == 0:
            return (torch.FloatTensor([]),)
        model_sampling = model.get_model_object("model_sampling")
        sigmas = get_beta_schedule_native_dense(
            model_sampling,
            total_steps,
            alpha=alpha,
            beta=beta,
            density_factor=density_factor,
            minimum_sigma_points=minimum_sigma_points,
        )
        return (_trim_for_denoise(sigmas, steps, denoise),)


class NativeCRSLiteScheduler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {"default": 28, "min": 1, "max": 10000}),
                "density_factor": ("INT", {"default": 32, "min": 2, "max": 512}),
                "minimum_sigma_points": (
                    "INT",
                    {"default": 1000, "min": 64, "max": 200000},
                ),
                "rate_exponent": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 8.0, "step": 0.05},
                ),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    CATEGORY = "sampling/custom_sampling/schedulers"
    FUNCTION = "get_sigmas"

    def get_sigmas(
        self,
        model,
        steps,
        density_factor,
        minimum_sigma_points,
        rate_exponent,
        denoise,
    ):
        total_steps = _resolve_total_steps(steps, denoise)
        if total_steps == 0:
            return (torch.FloatTensor([]),)
        model_sampling = model.get_model_object("model_sampling")
        sigmas = get_native_crs_lite_schedule(
            model_sampling,
            total_steps,
            density_factor=density_factor,
            minimum_sigma_points=minimum_sigma_points,
            rate_exponent=rate_exponent,
        )
        return (_trim_for_denoise(sigmas, steps, denoise),)


class NativeARTLiteScheduler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {"default": 28, "min": 1, "max": 10000}),
                "density_factor": ("INT", {"default": 32, "min": 2, "max": 512}),
                "minimum_sigma_points": (
                    "INT",
                    {"default": 1000, "min": 64, "max": 200000},
                ),
                "slope_weight": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.05},
                ),
                "curvature_weight": (
                    "FLOAT",
                    {"default": 2.0, "min": 0.0, "max": 16.0, "step": 0.05},
                ),
                "response_exponent": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 8.0, "step": 0.05},
                ),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("sigmas",)
    CATEGORY = "sampling/custom_sampling/schedulers"
    FUNCTION = "get_sigmas"

    def get_sigmas(
        self,
        model,
        steps,
        density_factor,
        minimum_sigma_points,
        slope_weight,
        curvature_weight,
        response_exponent,
        denoise,
    ):
        total_steps = _resolve_total_steps(steps, denoise)
        if total_steps == 0:
            return (torch.FloatTensor([]),)
        model_sampling = model.get_model_object("model_sampling")
        sigmas = get_native_art_lite_schedule(
            model_sampling,
            total_steps,
            density_factor=density_factor,
            minimum_sigma_points=minimum_sigma_points,
            slope_weight=slope_weight,
            curvature_weight=curvature_weight,
            response_exponent=response_exponent,
        )
        return (_trim_for_denoise(sigmas, steps, denoise),)


NODE_CLASS_MAPPINGS = {
    "BetaSchedulerV2C": BetaSchedulerV2C,
    "BetaSchedulerV3": BetaSchedulerV3,
    "BetaSchedulerNativeDense": BetaSchedulerNativeDense,
    "NativeCRSLiteScheduler": NativeCRSLiteScheduler,
    "NativeARTLiteScheduler": NativeARTLiteScheduler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BetaSchedulerV2C": "Beta Scheduler V2C",
    "BetaSchedulerV3": "Beta Scheduler V3",
    "BetaSchedulerNativeDense": "Beta Scheduler Native Dense",
    "NativeCRSLiteScheduler": "Native CRS Lite Scheduler",
    "NativeARTLiteScheduler": "Native ART Lite Scheduler",
}
