# src/data/tokenize.py
import torch

@torch.no_grad()
def make_tsf_tokens(
    x,                      # [B, T, S, F, M] float
    avail_mask,             # [S, F] or [S, F, M] (1 if exists at that station/feature[/measurement])
    miss_mask=None,         # [B, T, S, F, M] bool (True if missing now)
    append_missing_flags=True,
):
    """
    Returns:
      tokens:            [B, L=T*S*F, D],  D = M (+M if append_missing_flags)
      t_idx, s_idx, f_idx: [L] long (embedding indices for time/station/feature)
      key_padding_mask:  [B, L] bool (True => ignore this token entirely)
      meas_axis_mask:    [B, L, M] bool (True if that measurement channel is missing) for fine-grained IG/ablations
    """
    B, T, S, F, M = x.shape
    device = x.device

    # normalize avail_mask shape
    if avail_mask.dim() == 2:
        avail_mask = avail_mask.unsqueeze(-1).expand(S, F, M)   # -> [S,F,M]
    avail_mask = (avail_mask > 0).to(torch.bool).to(device)     # [S,F,M]

    # runtime missing; also mark permanently unavailable as missing
    if miss_mask is None:
        miss_mask = torch.zeros_like(x, dtype=torch.bool)
    miss_mask = miss_mask | (~avail_mask.unsqueeze(0).unsqueeze(0))  # [B,T,S,F,M]

    # zero out permanently unavailable channels to avoid leakage
    x_masked = x * avail_mask.unsqueeze(0).unsqueeze(0).float()      # [B,T,S,F,M]

    # reshape to tokens: (T,S,FEAT) becomes sequence; M stays as token channels
    vals  = x_masked.view(B, T*S*F, M)                               # [B,L,M]
    flags = miss_mask.view(B, T*S*F, M)                               # [B,L,M] bool

    if append_missing_flags:
        tokens = torch.cat([vals, flags.float()], dim=-1)            # [B,L, 2M]
        D = 2 * M
    else:
        tokens = vals                                                # [B,L,M]
        D = M

    # build embedding indices for (t,s,f)
    t_grid = torch.arange(T, device=device).view(T,1,1).expand(T,S,F)
    s_grid = torch.arange(S, device=device).view(1,S,1).expand(T,S,F)
    f_grid = torch.arange(F, device=device).view(1,1,F).expand(T,S,F)
    t_idx = t_grid.reshape(-1).long()                                 # [L]
    s_idx = s_grid.reshape(-1).long()                                 # [L]
    f_idx = f_grid.reshape(-1).long()                                 # [L]

    # token-level validity: usable iff ANY measurement channel present
    has_any = (~flags).any(dim=-1)                                    # [B,L]
    key_padding_mask = ~has_any                                       # [B,L] True => drop token

    # keep per-measurement-channel mask (for IG/ablations later)
    meas_axis_mask = flags                                            # [B,L,M]

    return tokens, t_idx, s_idx, f_idx, key_padding_mask, meas_axis_mask
