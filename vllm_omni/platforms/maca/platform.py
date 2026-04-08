# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.logger import init_logger
from vllm.utils.torch_utils import cuda_device_count_stateless
from vllm_metax.platform import MacaPlatform as MacaPlatformBase

from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
from vllm_omni.diffusion.envs import PACKAGES_CHECKER
from vllm_omni.platforms.interface import OmniPlatform, OmniPlatformEnum

logger = init_logger(__name__)


class MacaOmniPlatform(OmniPlatform, MacaPlatformBase):
    """MetaX MACA implementation of OmniPlatform.

    Inherits MACA-specific behavior from vLLM-metax and adds Omni worker / diffusion hooks.
    """

    _omni_enum = OmniPlatformEnum.MACA

    @classmethod
    def get_omni_ar_worker_cls(cls) -> str:
        return "vllm_omni.platforms.maca.worker.maca_ar_worker.MacaARWorker"

    @classmethod
    def get_omni_generation_worker_cls(cls) -> str:
        return "vllm_omni.platforms.maca.worker.maca_generation_worker.MacaGenerationWorker"

    @classmethod
    def get_default_stage_config_path(cls) -> str:
        return "vllm_omni/model_executor/stage_configs"

    @classmethod
    def get_diffusion_model_impl_qualname(cls, op_name: str) -> str:
        if op_name == "hunyuan_fused_moe":
            return "vllm_omni.diffusion.models.hunyuan_image_3.hunyuan_fused_moe.HunyuanFusedMoEDefault"
        return super().get_diffusion_model_impl_qualname(op_name)

    @classmethod
    def get_diffusion_attn_backend_cls(
        cls,
        selected_backend: str | None,
        head_size: int,
    ) -> str:
        """Diffusion attention backend selection (CUDA-compatible path, like CudaOmniPlatform)."""

        compute_capability = cls.get_device_capability()
        compute_supported = False
        if compute_capability is not None:
            major, minor = compute_capability
            capability = major * 10 + minor
            compute_supported = 80 <= capability < 100

        packages_info = PACKAGES_CHECKER.get_packages_info()
        packages_available = packages_info.get("has_flash_attn", False)
        flash_attn_supported = compute_supported and packages_available

        if selected_backend is not None:
            backend_upper = selected_backend.upper()
            if backend_upper == "FLASH_ATTN" and not flash_attn_supported:
                if not compute_supported:
                    logger.warning(
                        "Flash Attention expects compute capability >= 8.0 and < 10.0. "
                        "Falling back to TORCH_SDPA backend."
                    )
                elif not packages_available:
                    logger.warning("Flash Attention packages not available. Falling back to TORCH_SDPA backend.")
                logger.info("Defaulting to diffusion attention backend SDPA")
                return DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()
            backend = DiffusionAttentionBackendEnum[backend_upper]
            logger.info("Using diffusion attention backend '%s'", backend_upper)
            return backend.get_path()

        if flash_attn_supported:
            logger.info("Defaulting to diffusion attention backend FLASH_ATTN")
            return DiffusionAttentionBackendEnum.FLASH_ATTN.get_path()

        logger.info("Defaulting to diffusion attention backend SDPA")
        return DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()

    @classmethod
    def supports_torch_inductor(cls) -> bool:
        # vLLM-metax currently disables generic Triton kernel paths; keep inductor off by default.
        return False

    @classmethod
    def get_torch_device(cls, local_rank: int | None = None) -> torch.device:
        if local_rank is None:
            return torch.device("cuda")
        return torch.device("cuda", local_rank)

    @classmethod
    def get_device_count(cls) -> int:
        return cuda_device_count_stateless()

    @classmethod
    def get_device_version(cls) -> str | None:
        return torch.version.cuda

    @classmethod
    def synchronize(cls) -> None:
        torch.cuda.synchronize()

    @classmethod
    def get_free_memory(cls, device: torch.device | None = None) -> int:
        free, _ = torch.cuda.mem_get_info(device)
        return free
