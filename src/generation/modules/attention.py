import constants
import math
import torch
import torch.nn             as nn
import torch.nn.functional  as f

from generation.modules.util_modules    import Normalize, NonLinearity
    
class _Attention(nn.Module):
    def __init__(self, 
                 query_dimension, 
                 num_heads=1, 
                 cross_attention = False,
                 context_dimension=-1,
                 in_projetion_bias=True, 
                 out_projetion_bias=True):
        super().__init__()

        self.num_heads      = num_heads
        self.head_dimension = query_dimension // num_heads

        if not cross_attention:
            context_dimension = query_dimension

        self.query_matrix   = nn.Linear(query_dimension,    
                                        query_dimension, 
                                        bias=in_projetion_bias)
        self.key_matrix     = nn.Linear(context_dimension,  
                                        query_dimension, 
                                        bias=in_projetion_bias)
        self.value_matrix   = nn.Linear(context_dimension,  
                                        query_dimension, 
                                        bias=in_projetion_bias)
        self.out_projection = nn.Linear(query_dimension,    
                                        query_dimension, 
                                        bias=out_projetion_bias)

    def compute_attention(self, query, context):
        batch_size, num_channels, dimension = query.shape
        
        # possibly -1 for num_channels
        interim_shape = (batch_size, 
                           num_channels, 
                           self.num_heads, 
                           self.head_dimension)

        query   = self.query_matrix(query).view(interim_shape).transpose(1, 2)
        key     = self.key_matrix(context).view(interim_shape).transpose(1, 2)
        value   = self.value_matrix(context).view(interim_shape).transpose(1, 2)

        weight  = query @ key.transpose(-1, -2) 
        weight  = weight / math.sqrt(self.head_dimension)
        weight  = f.softmax(weight, dim=-1)

        output = (weight @ value)
        output = output.transpose(1,2).contiguous().view(query.shape)
        output = self.out_projection(query)

        return output

    
class CrossAttention(_Attention):
    def __init__(self, 
                 query_dimension, 
                 context_dimension, 
                 num_heads, 
                 in_projetion_bias=True, 
                 out_projetion_bias=True):
        
        super().__init__(query_dimension,
                         context_dimension=context_dimension,
                         cross_attention=True, 
                         num_heads=num_heads,
                         in_projetion_bias=in_projetion_bias,
                         out_projetion_bias=out_projetion_bias)

    def forward(self, x, y):
        return self.compute_attention(x, y)
    

class SelfAttention(_Attention):
    def __init__(self, 
                 query_dimension, 
                 num_heads=1, 
                 in_projetion_bias=True, 
                 out_projetion_bias=True):
        
        super().__init__(query_dimension,
                         num_heads=num_heads,
                         in_projetion_bias=in_projetion_bias,
                         out_projetion_bias=out_projetion_bias)

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
    def __init__(self, n_heads, n_embed, d_context=768):
        super().__init__()
        channels = n_heads * n_embed
 
        self.group_norm     = Normalize(channels, 32)
        self.conv_input     = nn.Conv2d(channels, 
                                        channels, 
                                        kernel_size=1, 
                                        padding=0)

        self.layernorm_1    = nn.LayerNorm(channels)
        self.attention_1    = SelfAttention(channels, n_heads)
        
        self.layernorm_2    = nn.LayerNorm(channels)
        self.attention_2    = CrossAttention(channels, 
                                             d_context, 
                                             n_heads, 
                                             in_proj_bias=False)

        self.layernorm_3    = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

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
        x = x + residual_x_short
        residual_x_short = x

        # cross attention block
        x = self.layernorm_2(x)
        x = self.attention_2(x, context)
        x = x + residual_x_short
        residual_x_short = x

        # feed forward
        x = self.layernorm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=1)
        x = x * f.gelu(gate)
        x= self.linear_geglu_2(x)
        x = x + residual_x_short

        x = x.transpose(-1, -2)
        x = x.view(batch_size, num_channels, height, width)
        return self.conv_output(x) + residual_x_long