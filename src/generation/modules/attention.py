import constants
import math
import torch
import torch.nn             as nn
import torch.nn.functional  as f

from generation.modules.util_modules    import Normalize, NonLinearity
    
class _Attention(nn.Module):
    def __init__(self, 
                 query_dimension, 
                 num_heads              = 1, 
                 cross_attention        = False,
                 context_dimension      = -1,
                 in_projection_bias     = True, 
                 out_projection_bias    = True):
        super().__init__()

        self.out_projection = nn.Linear(query_dimension,    
                                        query_dimension, 
                                        bias=out_projection_bias)
        
        self.attention = nn.MultiheadAttention(embed_dim= query_dimension, 
                                               num_heads= num_heads, 
                                               bias     = in_projection_bias,
                                               batch_first=True)

    def compute_attention(self, query, context):
        query   = query
        key     = context
        value   = context

        output  = self.attention(query, key, value)
        output  = self.out_projection(query)

        return output

    
class CrossAttention(_Attention):
    def __init__(self, 
                 query_dimension, 
                 context_dimension, 
                 num_heads, 
                 in_projection_bias=True, 
                 out_projection_bias=True):
        
        super().__init__(query_dimension,
                         context_dimension=context_dimension,
                         cross_attention=True, 
                         num_heads=num_heads,
                         in_projection_bias=in_projection_bias,
                         out_projection_bias=out_projection_bias)

    def forward(self, x, y):
        return self.compute_attention(x, y)
    

class SelfAttention(_Attention):
    def __init__(self, 
                 query_dimension, 
                 num_heads=1, 
                 in_projection_bias=True, 
                 out_projection_bias=True):
        
        super().__init__(query_dimension,
                         num_heads=num_heads,
                         in_projection_bias=in_projection_bias,
                         out_projection_bias=out_projection_bias)

    def forward(self, x):
        return self.compute_attention(x, x)


class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
 
        self.norm       = Normalize(channels, 32)
        self.attention  = SelfAttention(channels, 1)

    def forward(self, x):
        residual_x = x

        x = self.norm(x)

        batch_size, num_channels, height, width = x.shape
        x = x.view(batch_size, num_channels, height * width)
        
        x = x.transpose(-1, -2)
        x = self.attention(x)
        x = x.transpose(-1, -2)

        x = x.view(batch_size, num_channels, height, width)

        return x + residual_x
    
class ContextualAttentionBlock(nn.Module):
    def __init__(self, num_heads, n_embed, d_context=768):
        super().__init__()
        channels = num_heads * n_embed
 
        self.group_norm     = Normalize(channels, 32)
        self.conv_input     = nn.Conv2d(channels, 
                                        channels, 
                                        kernel_size=1, 
                                        padding=0)

        self.layernorm_1    = nn.LayerNorm(channels)
        self.attention_1    = SelfAttention(channels, num_heads)
        
        self.layernorm_2    = nn.LayerNorm(channels)
        self.attention_2    = CrossAttention(channels, 
                                             d_context, 
                                             num_heads, 
                                             in_projection_bias=False)

        self.layernorm_3    = nn.LayerNorm(channels)
        self.linear_1       = nn.Linear(channels, 4 * channels * 2)
        self.linear_2       = nn.Linear(4 * channels, channels)

        self.conv_output    = nn.Conv2d(channels, 
                                        channels, 
                                        kernel_size=1, 
                                        padding=0)

    def forward(self, x, context):
        residual_x_long = x

        x = self.group_norm(x)
        x = self.conv_input(x)

        batch_size, num_channels, height, width = x.shape()
        x = x.view(batch_size, num_channels, height * width)
        
        x = x.transpose(-1, -2)
        residual_x_short = x  

        # self attention block
        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x += residual_x_short
        residual_x_short = x

        # cross attention block
        x = self.layernorm_2(x)
        x = self.attention_2(x, context)
        x += residual_x_short
        residual_x_short = x

        # feed forward
        x = self.layernorm_3(x)
        x, gate = self.linear_1(x).chunk(2, dim=1)
        x *= f.gelu(gate)
        x = self.linear_2(x)
        x += residual_x_short

        x = x.transpose(-1, -2)
        x = x.view(batch_size, num_channels, height, width)
        
        return self.conv_output(x) + residual_x_long