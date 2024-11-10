"""
Modified version from Diffusion Transformers 
https://github.com/facebookresearch/DiT/blob/main/models.py
"""

import torch
import util

import numpy        as np
import torch.nn     as nn

from debug                          import Printer
from generation.modules.attention   import SelfAttention

class DiTFactory():
    def create_dit(configuration, double_output=False):
        input_shape             = (configuration["input_num_channels"], 
                                   configuration["input_resolution_x"],
                                   configuration["input_resolution_y"])
        
        output_channel_amount   = (input_shape[0] 
                                   + double_output * input_shape[0])


        architecture            = configuration["DiT_architecture"]
        amount_dit_blocks       = architecture["amount_DiT_blocks"]
        patch_size              = architecture["patch_size"]
        token_channel_amount    = configuration["time_embedding_size"] * 4
        amount_heads            = architecture["amount_heads"]
    
        return DiT(input_shape,
                   output_channel_amount,
                   amount_dit_blocks,
                   patch_size,
                   token_channel_amount,
                   amount_heads)


class DiT(nn.Module):
    """
    Scalable Diffusion Models with Transformers
    """

    def __init__(self, 
                 input_shape,
                 output_channel_amount,
                 amount_dit_blocks,
                 patch_size, 
                 token_channel_amount,
                 amount_heads):
        super().__init__()

        self.data_channel_amount    = input_shape[0]
        self.data_size              = input_shape[1]
        self.output_channel_amount  = output_channel_amount

        # Patchify ============================================================
        self.patch_size             = patch_size
        self.patch_amount           = self.data_size // patch_size
        self.patchify               = nn.Conv2d(self.data_channel_amount, 
                                                token_channel_amount, 
                                                kernel_size   = patch_size, 
                                                stride        = patch_size)
        
        self.token_channel_amount   = token_channel_amount
        
        self.register_buffer("positional_encoding", self.__get_cos_embedding(
            token_channel_amount, 
            self.patch_amount))
        
        # N x DiT-Blocks ======================================================
        self.dit_blocks = nn.ModuleList()
        for _ in range(amount_dit_blocks):
            self.dit_blocks.append(_DiTBlock(token_channel_amount,
                                             amount_heads))

        # Layer Norm, Linear and Reshape ======================================
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(token_channel_amount, 2 * token_channel_amount))
        
        self.layer_norm = nn.LayerNorm(token_channel_amount, 
                                       elementwise_affine=False)
        self.linear     = nn.Linear(token_channel_amount, 
                                    np.square(self.patch_size) 
                                    * output_channel_amount)
        
        self.__initialize_weights()
        
    def forward(self, x, context, time):
        x = self.patchify(x)
        x = x.flatten(2).transpose(1, 2) 
        
        x       += self.positional_encoding
        context += time

        for dit_block in self.dit_blocks:
            x = dit_block(x, context)

        gamma, beta = self.mlp(context).unsqueeze(1).chunk(2, dim=2)
        x = self.layer_norm(x)
        x = x * (1 + gamma) + beta
        x = self.linear(x)

        x = self.__unpatchify(x)

        return x

    def __unpatchify(self, x):
        x = x.reshape(shape=(x.shape[0], 
                             self.patch_amount, 
                             self.patch_amount, 
                             self.patch_size, 
                             self.patch_size, 
                             self.output_channel_amount))
        x = x.permute(0, 5, 1, 3, 2, 4)
        x = x.reshape(shape=(x.shape[0], 
                             self.output_channel_amount, 
                             self.data_size,  
                             self.data_size))
        return x

    def __initialize_weights(self):
        """Shorter version of the original DiT initialization"""
        self.apply(self.__basic_init)

        patchify_weights = self.patchify.weight.data
        nn.init.xavier_uniform_(
            patchify_weights.view([patchify_weights.shape[0], -1]))
        nn.init.constant_(self.patchify.bias, 0)

        for dit_block in self.dit_blocks:
            nn.init.constant_(dit_block.mlp[-1].weight, 0)
            nn.init.constant_(dit_block.mlp[-1].bias, 0)

        # Zero out final layer
        nn.init.constant_(self.mlp[-1].weight, 0)
        nn.init.constant_(self.mlp[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def __basic_init(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
                    
    def __get_cos_embedding(self, embedding_size, image_size):
        grid = torch.arange(start   = 0, 
                            end     = image_size, 
                            dtype   = torch.float32,
                            device  = util.get_device())
        
        grid = torch.meshgrid(grid, grid, indexing="xy")

        y = self.__get_cos_embedding_1D(embedding_size // 2, grid[0].flatten())
        x = self.__get_cos_embedding_1D(embedding_size // 2, grid[1].flatten())

        embedding = torch.cat([y, x], dim=1)
        embedding = embedding.unsqueeze(0)
        return embedding

    def __get_cos_embedding_1D(self, embedding_size, pos):
        embedding_size = embedding_size // 2

        freqs = torch.arange(start = 0, 
                             end   = embedding_size, 
                             dtype = torch.float32,
                             device=util.get_device()) 
        freqs /= embedding_size
        freqs = torch.pow(10000, -freqs)
        
        pos         = torch.reshape(pos, (-1,))
        embedding   = torch.outer(pos, freqs)
        embedding   = torch.cat([torch.cos(embedding), 
                                 torch.sin(embedding)], 
                                 dim=-1)
    
        return embedding


class _DiTBlock(nn.Module):
    """
    See figure 3 of DiT paper
    """
    def __init__(self, 
                 token_channels, 
                 num_heads, 
                 feedforward_channel_factor = 4.0):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(token_channels, 6 * token_channels))
        
        self.layernorm_1    = nn.LayerNorm(token_channels,
                                           elementwise_affine=False)
        self.attention      = SelfAttention(token_channels, num_heads)

        feedforward_channels = int(token_channels * 
                                   feedforward_channel_factor)
        
        self.layernorm_2    = nn.LayerNorm(token_channels,
                                           elementwise_affine=False)
        self.feedforward = nn.Sequential(
            nn.Linear(token_channels, feedforward_channels),
            nn.GELU(approximate="tanh"),
            nn.Linear(feedforward_channels, token_channels))

    def __modulate(self, x, gamma, beta):
        return x * (1 + gamma) + beta

    def forward(self, x, context):
        # adaLN ===============================================================
        # gamma_1, beta_1 alpha_1 --------------------------------------------- 
        (gamma_attention, 
         beta_attention, 
         gate_attention,
        # gamma_2, beta_2 alpha_2 --------------------------------------------- 
         gamma_feedforward, 
         beta_feedforward, 
         gate_feedforward) = self.mlp(context).unsqueeze(1).chunk(6, dim=2)
        
        residual_x = x
        x = self.layernorm_1(x)
        x = self.__modulate(x, gamma_attention, beta_attention)
        x = self.attention(x)
        x *= gate_attention
        x += residual_x
        
        residual_x = x
        x = self.layernorm_2(x)
        x = self.__modulate(x, gamma_feedforward, beta_feedforward)
        x = self.feedforward(x)
        x *= gate_feedforward
        x += residual_x

        return x    