# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""MACA (MetaX) generation worker for vLLM-Omni.

Non-AR stages use the same CUDA-compatible device path as ``GPUGenerationWorker``.
"""

from vllm_omni.worker.gpu_generation_worker import GPUGenerationWorker


class MacaGenerationWorker(GPUGenerationWorker):
    """Non-autoregressive generation stages on MetaX MACA via vLLM-metax."""
