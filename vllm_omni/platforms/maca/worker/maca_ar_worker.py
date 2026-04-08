# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""MACA (MetaX) AR worker for vLLM-Omni.

MetaX uses the CUDA-compatible runtime (``device_type="cuda"`` in vLLM-metax); this
worker reuses the standard Omni GPU AR worker implementation.
"""

from vllm_omni.worker.gpu_ar_worker import GPUARWorker


class MacaARWorker(GPUARWorker):
    """Autoregressive omni stages on MetaX MACA via vLLM-metax."""
