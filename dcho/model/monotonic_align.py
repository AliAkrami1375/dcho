"""Monotonic alignment search.

Given a log-likelihood matrix between phonemes and spectrogram frames, find
the highest-scoring alignment subject to three constraints: every frame is
assigned to exactly one phoneme, the phoneme index never decreases, and no
phoneme is skipped. That is the alignment the model trains against, and it
is discovered rather than supplied - there are no forced-alignment labels
anywhere in this project.

The reference VITS implementation ships a Cython kernel. This one is pure
torch: the dynamic program is vectorised across the batch and the phoneme
axis, leaving a Python loop only over frames. That keeps the build free of
a compile step and runs on whatever device the tensors already live on,
at a cost of a few milliseconds per step.
"""

from __future__ import annotations

import torch

NEG_INF = -1e9


@torch.no_grad()
def maximum_path(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Highest-likelihood monotonic alignment.

    value: [B, T_text, T_spec]  per-pair log-likelihood
    mask:  [B, T_text, T_spec]  1 inside the valid rectangle of each item
    returns: [B, T_text, T_spec] hard one-hot alignment, same dtype as value
    """
    b, t_text, t_spec = value.shape
    device = value.device
    orig_dtype = value.dtype

    v = value.float()
    m = mask.bool()
    # Anything outside an item's valid rectangle must never win a max.
    v = torch.where(m, v, torch.full_like(v, NEG_INF))

    text_lengths = m[:, :, 0].sum(dim=1).long()
    spec_lengths = m[:, 0, :].sum(dim=1).long()

    # dp[b, t] = best score of an alignment covering frames 0..s that ends
    # with frame s assigned to phoneme t. Only phoneme 0 can hold frame 0.
    dp = torch.full((b, t_text), NEG_INF, device=device, dtype=torch.float32)
    dp[:, 0] = v[:, 0, 0]

    # came_from_prev[b, t, s] records whether the optimal way to reach
    # (t, s) advanced the phoneme index at this frame.
    came_from_prev = torch.zeros((b, t_text, t_spec), device=device, dtype=torch.bool)

    pad = torch.full((b, 1), NEG_INF, device=device, dtype=torch.float32)
    for s in range(1, t_spec):
        # dp_advance[b, t] is dp[b, t-1] from the previous frame: the score
        # of moving on to phoneme t at this frame.
        dp_advance = torch.cat([pad, dp[:, :-1]], dim=1)
        advance = dp_advance > dp
        came_from_prev[:, :, s] = advance
        dp = torch.where(advance, dp_advance, dp) + v[:, :, s]

    # Backtrack from the last valid frame of each item, which must be
    # aligned to that item's last phoneme.
    path = torch.zeros((b, t_text, t_spec), device=device, dtype=torch.float32)
    batch_idx = torch.arange(b, device=device)
    idx = (text_lengths - 1).clamp(min=0)

    for s in range(t_spec - 1, -1, -1):
        active = s < spec_lengths
        path[batch_idx, idx, s] = active.float()
        step_back = came_from_prev[batch_idx, idx, s] & active
        idx = torch.where(step_back, idx - 1, idx).clamp(min=0)

    return path.to(orig_dtype)
