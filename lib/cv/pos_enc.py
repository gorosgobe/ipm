import torch
import numpy as np


class PositionalEncodings(object):
    """
    Positional encodings from
    "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"
    Available at https://arxiv.org/pdf/2003.08934.pdf
    """
    @staticmethod
    def get_positional_encodings(L, batched_coord_maps):
        b, c, h, w = batched_coord_maps.size()
        assert c == 2
        enc_range = torch.arange(L).to(batched_coord_maps.device)
        exp = (2 ** enc_range) * np.pi
        # Here we have (B, C=2, H, W) and (L,)
        # Get (B, C, 1, H, W) from (B, C, H, W) and (B, C, L, H, W) from (L)
        expanded_maps = batched_coord_maps.unsqueeze(2)
        expanded_exp = exp.unsqueeze(-1).unsqueeze(-1).unsqueeze(0).unsqueeze(0).repeat(b, c, 1, h, w)
        # Multiply to get (B, C, L, H, W) and view as (B, C * L, H, W)
        res_channel = (expanded_maps * expanded_exp).view(b, c * L, h, w)
        sin_channels = torch.sin(res_channel)
        cos_channels = torch.cos(res_channel)
        out = torch.cat((sin_channels, cos_channels), dim=1)
        assert out.size() == (b, 2 * c * L, h, w)
        # finally
        return out