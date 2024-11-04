"""
From Diffusion Transformers 
"""

import torch

import torch.nn     as nn

from generation.modules.attention import AttentionBlock, ContextualAttentionBlock

class Patchify(nn.Module):
    def __init__(self, 
                 input_shape,
                 patch_size_factor,
                 token_channels):
        super().__init__()

        input_channel_amount    = input_shape[0]

        self.patch_amount = input_shape[1] // patch_size_factor

        self.patchify = nn.Conv2d(input_channel_amount, 
                                  token_channels, 
                                  kernel_size   = patch_size_factor, 
                                  stride        = patch_size_factor)
        
    def forward(self, x):
        x = self.patchify(x)
        # Might not need
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        return x

class _DiTBlock(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, context, time):
        
        return x
    

class DiT(nn.Module):
    def __init__(self):
        super().__init__()
        amount_dit_blocks = 8

        self.patchify = Patchify(1,2,3)

        self.dit_blocks = nn.ModuleList()
        for _ in range(amount_dit_blocks):
            self.dit_blocks.append(_DiTBlock())

        self.layer_norm = nn.LayerNorm()

    def forward(self, x, context, time):
        x = self.patchify(x)

        for dit_block in self.dit_blocks:
            x = dit_block(x, context, time)

        return x



