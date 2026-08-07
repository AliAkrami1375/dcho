"""Shared tensor helpers for the dcho acoustic model."""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F


def init_weights(m, mean: float = 0.0, std: float = 0.01):
    """Initialise convolution weights in place. Applied via `Module.apply`."""
    if m.__class__.__name__.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    """Padding that keeps the temporal length unchanged for odd kernels."""
    return int((kernel_size * dilation - dilation) / 2)


def convert_pad_shape(pad_shape: list[list[int]]) -> list[int]:
    """Reorder a per-dimension pad spec into the flat form F.pad expects."""
    return [item for sublist in pad_shape[::-1] for item in sublist]


def sequence_mask(length: torch.Tensor, max_length: int | None = None) -> torch.Tensor:
    """Boolean mask [B, T] that is True for positions inside each sequence."""
    if max_length is None:
        max_length = int(length.max())
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def subsequent_mask(length: int) -> torch.Tensor:
    return torch.tril(torch.ones(length, length)).unsqueeze(0).unsqueeze(0)


@torch.jit.script
def fused_add_tanh_sigmoid_multiply(
    input_a: torch.Tensor, input_b: torch.Tensor, n_channels: torch.Tensor
) -> torch.Tensor:
    """Gated activation unit used throughout the WaveNet stacks."""
    n_channels_int = n_channels[0]
    acts = input_a + input_b
    t_act = torch.tanh(acts[:, :n_channels_int, :])
    s_act = torch.sigmoid(acts[:, n_channels_int:, :])
    return t_act * s_act


def slice_segments(x: torch.Tensor, ids_str: torch.Tensor, segment_size: int) -> torch.Tensor:
    """Gather a fixed-length window from each item of a batch."""
    ret = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        idx_str = int(ids_str[i])
        ret[i] = x[i, :, idx_str : idx_str + segment_size]
    return ret


def rand_slice_segments(
    x: torch.Tensor, x_lengths: torch.Tensor | None = None, segment_size: int = 4
):
    """Pick a random window per item; returns the slice and the start indices.

    Training the decoder on a short random window instead of the full
    utterance is what keeps VITS-style training affordable: the decoder is
    by far the most expensive part and its cost is proportional to the
    number of output samples.
    """
    b, _, t = x.size()
    if x_lengths is None:
        x_lengths = torch.full((b,), t, dtype=torch.long, device=x.device)
    ids_str_max = torch.clamp(x_lengths - segment_size + 1, min=1)
    ids_str = (torch.rand([b], device=x.device) * ids_str_max).to(dtype=torch.long)
    return slice_segments(x, ids_str, segment_size), ids_str


def generate_path(duration: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Expand per-phoneme durations into a hard alignment matrix.

    duration: [B, 1, T_text]      integer frame counts per input token
    mask:     [B, 1, T_mel, T_text]
    returns:  [B, 1, T_mel, T_text] one-hot over the text axis
    """
    b, _, t_y, t_x = mask.shape
    cum_duration = torch.cumsum(duration, -1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    # Differencing along the text axis turns cumulative spans into
    # exclusive spans, i.e. each frame is claimed by exactly one token.
    path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    path = path.unsqueeze(1).transpose(2, 3) * mask
    return path


def clip_grad_value_(parameters, clip_value, norm_type: float = 2.0) -> float:
    """Clip gradients in place and return the pre-clip total norm.

    The returned norm is logged during training: a sudden spike is the
    earliest visible sign that adversarial training is destabilising.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    if clip_value is not None:
        clip_value = float(clip_value)

    total_norm = 0.0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
        if clip_value is not None:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)
    return total_norm ** (1.0 / norm_type)


def rand_gumbel(shape) -> torch.Tensor:
    uniform_samples = torch.rand(shape) * 0.99998 + 0.00001
    return -torch.log(-torch.log(uniform_samples))


def intersperse(lst: list, item):
    """Insert `item` between every element and at both ends.

    Used to interleave a blank symbol into the phoneme sequence, which
    gives the monotonic alignment search somewhere to park the silence
    between phones instead of stretching a real phone across it.
    """
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result
