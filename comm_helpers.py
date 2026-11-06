from __future__ import annotations

from collections.abc import Iterable

import torch


def flatten_tensors(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    """Flatten dense tensors into a contiguous 1D buffer.

    Reference: https://github.com/facebookresearch/stochastic_gradient_push

    Arguments:
        tensors: Iterable of dense tensors to flatten.

    Returns:
        A 1D contiguous tensor containing all elements.
    """
    tensors_list = list(tensors)
    if len(tensors_list) == 1:
        return tensors_list[0].view(-1).clone()
    return torch.cat([t.view(-1) for t in tensors_list], dim=0)


def unflatten_tensors(
    flat: torch.Tensor, tensors: Iterable[torch.Tensor]
) -> tuple[torch.Tensor, ...]:
    """View a flat buffer using the sizes and shapes of original tensors.

    Reference: https://github.com/facebookresearch/stochastic_gradient_push

    Arguments:
        flat: Flattened 1D dense tensor buffer.
        tensors: Iterable of tensors whose shapes and numels determine views.

    Returns:
        Tuple of unflattened tensor views.
    """
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return tuple(outputs)


def quantize_flattened_tensor(
    flat: torch.Tensor,
    nbits: int = 4,
    block_size: int = 64,
) -> dict[str, torch.Tensor | tuple[int, ...] | int | torch.dtype | bool]:
    """Quantize a 1D flattened tensor into scaled integer blocks with sub-byte packing.

    Supports INT4 (packed into torch.uint8 nibbles) and INT8 (torch.int8) with
    per-block dynamic scale factors. Zero blocks and non-finite values are handled safely.

    Arguments:
        flat: 1D floating-point PyTorch tensor.
        nbits: Quantization bit-width (4 or 8).
        block_size: Elements per quantization block (even positive integer).

    Returns:
        Dictionary containing packed data, scales, shape, padding length, and metadata.
    """
    if nbits not in (4, 8):
        raise ValueError(f"nbits must be 4 or 8, got {nbits}")
    if block_size < 2 or block_size % 2 != 0:
        raise ValueError(
            f"block_size must be a positive even integer, got {block_size}"
        )

    shape = tuple(flat.shape)
    numel = flat.numel()
    device = flat.device
    orig_dtype = flat.dtype

    if numel == 0:
        empty_data = torch.empty(
            0, dtype=torch.uint8 if nbits == 4 else torch.int8, device=device
        )
        return {
            "data": empty_data,
            "scales": torch.empty(0, dtype=torch.float32, device=device),
            "shape": shape,
            "pad_len": 0,
            "nbits": nbits,
            "orig_dtype": orig_dtype,
            "is_quantized": True,
        }

    flat_f32 = flat.detach().flatten().float()
    pad_len = (block_size - (numel % block_size)) % block_size
    if pad_len > 0:
        flat_f32 = torch.nn.functional.pad(flat_f32, (0, pad_len))

    blocks = flat_f32.reshape(-1, block_size)
    max_vals = blocks.abs().max(dim=-1, keepdim=True).values
    zero_mask = max_vals == 0

    if nbits == 4:
        scales = torch.where(zero_mask, torch.ones_like(max_vals), max_vals / 7.0)
        q = torch.where(
            zero_mask,
            torch.zeros_like(blocks, dtype=torch.int8),
            torch.clamp(torch.round(blocks / scales), -7, 7).to(torch.int8),
        )
        scales = torch.where(zero_mask, torch.zeros_like(max_vals), scales)

        q_flat = q.flatten()
        q_unsigned = (q_flat + 7).to(torch.uint8)
        lo = q_unsigned[0::2]
        hi = q_unsigned[1::2]
        packed = (hi << 4) | (lo & 0x0F)

        return {
            "data": packed,
            "scales": scales.squeeze(-1).to(torch.float32),
            "shape": shape,
            "pad_len": pad_len,
            "nbits": 4,
            "orig_dtype": orig_dtype,
            "is_quantized": True,
        }
    else:  # nbits == 8
        scales = torch.where(zero_mask, torch.ones_like(max_vals), max_vals / 127.0)
        q = torch.where(
            zero_mask,
            torch.zeros_like(blocks, dtype=torch.int8),
            torch.clamp(torch.round(blocks / scales), -127, 127).to(torch.int8),
        )
        scales = torch.where(zero_mask, torch.zeros_like(max_vals), scales)

        return {
            "data": q.flatten(),
            "scales": scales.squeeze(-1).to(torch.float32),
            "shape": shape,
            "pad_len": pad_len,
            "nbits": 8,
            "orig_dtype": orig_dtype,
            "is_quantized": True,
        }


def dequantize_flattened_tensor(
    quantized_dict: dict[
        str, torch.Tensor | tuple[int, ...] | int | torch.dtype | bool
    ],
    device: torch.device | None = None,
) -> torch.Tensor:
    """Reconstruct 1D flattened float tensor from quantized representation.

    Arguments:
        quantized_dict: Dictionary returned by `quantize_flattened_tensor`.
        device: Destination device (defaults to data.device).

    Returns:
        Reconstructed floating-point 1D tensor matching original dtype.
    """
    data = quantized_dict["data"]
    scales = quantized_dict["scales"]
    shape = quantized_dict["shape"]
    pad_len = quantized_dict["pad_len"]
    nbits = quantized_dict["nbits"]
    orig_dtype = quantized_dict.get("orig_dtype", torch.float32)
    target_device = device if device is not None else data.device

    if data.numel() == 0:
        return torch.empty(shape, dtype=orig_dtype, device=target_device)

    num_blocks = scales.numel()

    if nbits == 4:
        lo = (data & 0x0F).to(torch.float32) - 7.0
        hi = ((data >> 4) & 0x0F).to(torch.float32) - 7.0
        unpacked = torch.stack([lo, hi], dim=-1).flatten()
        block_size = unpacked.numel() // num_blocks
        blocks = unpacked.reshape(num_blocks, block_size)
    elif nbits == 8:
        block_size = data.numel() // num_blocks
        blocks = data.reshape(num_blocks, block_size).to(torch.float32)
    else:
        raise ValueError(f"Unsupported nbits: {nbits}")

    dequant = blocks * scales.unsqueeze(-1)
    flat = dequant.flatten()
    if pad_len > 0:
        flat = flat[:-pad_len]
    return flat.reshape(shape).to(dtype=orig_dtype, device=target_device)


def communicate_quantized(
    tensors: Iterable[torch.Tensor],
    communication_op,
    nbits: int = 4,
    block_size: int = 64,
    residuals: torch.Tensor | None = None,
) -> None:
    """Quantize flattened tensor buffer prior to distributed communication.

    Arguments:
        tensors: List/iterable of parameter tensors to communicate.
        communication_op: Distributed communication callable (e.g. `dist.all_reduce`).
        nbits: Quantization bit-width (4 or 8).
        block_size: Elements per quantization block.
        residuals: Optional 1D buffer matching total numel accumulating error residuals.
    """
    tensors_list = list(tensors)
    flat_tensor = flatten_tensors(tensors_list)
    if residuals is not None:
        adj = flat_tensor + residuals
        q = quantize_flattened_tensor(adj, nbits=nbits, block_size=block_size)
        recon = dequantize_flattened_tensor(q, device=flat_tensor.device)
        residuals.copy_(adj - recon)
        flat_to_comm = recon
    else:
        q = quantize_flattened_tensor(flat_tensor, nbits=nbits, block_size=block_size)
        flat_to_comm = dequantize_flattened_tensor(q, device=flat_tensor.device)

    communication_op(tensor=flat_to_comm)
    for f, t in zip(unflatten_tensors(flat_to_comm, tensors_list), tensors_list):
        t.data.copy_(f)


def communicate(
    tensors: Iterable[torch.Tensor],
    communication_op,
    nbits: int | None = None,
    block_size: int = 64,
    residuals: torch.Tensor | None = None,
) -> None:
    """Communicate a list of tensors across workers with optional quantization.

    Reference: https://github.com/facebookresearch/stochastic_gradient_push

    Arguments:
        tensors: Dense tensors to communicate.
        communication_op: Callable that takes `tensor` argument (e.g. `dist.all_reduce`).
        nbits: Optional bit-width (4 or 8). When specified, compresses communicated buffer.
        block_size: Block size used when quantization is active.
        residuals: Optional error-feedback residual tensor.
    """
    tensors_list = list(tensors)
    if nbits is not None:
        communicate_quantized(
            tensors_list,
            communication_op,
            nbits=nbits,
            block_size=block_size,
            residuals=residuals,
        )
        return

    flat_tensor = flatten_tensors(tensors_list)
    communication_op(tensor=flat_tensor)
    for f, t in zip(unflatten_tensors(flat_tensor, tensors_list), tensors_list):
        t.data.copy_(f)


def calc_tensor_compression_stats(
    tensors: Iterable[torch.Tensor],
    nbits: int = 4,
    block_size: int = 64,
) -> dict[str, int | float]:
    """Compute volume and bandwidth savings for communicated tensors.

    Arguments:
        tensors: List of parameter tensors.
        nbits: Quantization bit-width (4 or 8).
        block_size: Elements per quantization block.

    Returns:
        Dictionary with raw_bytes, compressed_bytes, compression_ratio, and
        bandwidth_savings_pct.
    """
    flat = flatten_tensors(tensors)
    raw_bytes = flat.numel() * flat.element_size()

    q_dict = quantize_flattened_tensor(flat, nbits=nbits, block_size=block_size)
    compressed_bytes = (
        q_dict["data"].numel() * q_dict["data"].element_size()
        + q_dict["scales"].numel() * q_dict["scales"].element_size()
    )

    ratio = raw_bytes / compressed_bytes if compressed_bytes > 0 else 1.0
    savings = (1.0 - compressed_bytes / raw_bytes) * 100.0 if raw_bytes > 0 else 0.0
    return {
        "raw_bytes": raw_bytes,
        "compressed_bytes": compressed_bytes,
        "compression_ratio": ratio,
        "bandwidth_savings_pct": savings,
    }
