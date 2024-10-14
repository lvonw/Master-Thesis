import torch
import torch.nn             as nn
import torch.nn.functional  as f

from generation.modules.util_modules    import (ResNetBlock, 
                                                Downsample, 
                                                Upsample, 
                                                Normalize)
from generation.modules.attention       import ContextualAttentionBlock, AttentionBlock



class SwitchSequential(nn.Sequential):
    def forward(self, x, control_signal, time):
        for layer in self:
            if isinstance(layer, ContextualAttentionBlock):
                x = layer(x, control_signal)
            elif isinstance(layer, ResNetBlock):
                x = layer(x, time)
            else:
                x = layer(x)

        return x
    
class UNETFactory():
    def __init__(self):
        pass

    def create_unet(configuration):
        input_shape         = (configuration["input_num_channels"], 
                               configuration["input_resolution_x"],
                               configuration["input_resolution_y"])
        
        use_control_signal  = configuration["use_control_signal"]
        

        architecture = configuration["unet_architecture"]
        
        resNet_per_level_encoder    = architecture["ResNet_blocks_per_level"]
        resNet_per_level_decoder    = resNet_per_level_encoder + 1

        channel_multipliers_encoder = architecture["channel_multipliers"]
        channel_multipliers_decoder = channel_multipliers_encoder[::-1]

        starting_channels           = architecture["starting_channels"]

        attention_levels            = architecture["attention_levels"]

        num_heads                   = architecture["num_heads"]
        embedding_channels          = architecture["embedding_channels"]

        return UNET(input_shape,
                    starting_channels,
                    channel_multipliers_encoder,
                    channel_multipliers_decoder,
                    resNet_per_level_encoder,
                    resNet_per_level_decoder,
                    attention_levels,
                    num_heads,
                    embedding_channels,
                    use_control_signal)


class UNET(nn.Module):
    def __init__(self, 
                 input_shape,
                 starting_channels,
                 channel_multipliers_encoder,
                 channel_multipliers_decoder,
                 resNet_per_level_encoder,
                 resNet_per_level_decoder,
                 attention_levels,
                 num_heads = 8,
                 embedding_channels = 40,
                 use_control_signal = False):
        super().__init__()

        self.use_control_signal = use_control_signal
        self.input_shape        = input_shape
        input_channel_amount    = input_shape[0]
        
        skip_channels = [] 

        # Encoder
        self.encoder = nn.ModuleList()
        # TODO Do this for the autoencoder too
        self.input_conv = nn.Conv2d(input_channel_amount, 
                                    starting_channels, 
                                    kernel_size=3,
                                    stride=1, 
                                    padding=1)
        skip_channels.append(starting_channels)
        
        previous_channel_amount = starting_channels

        for level, multiplier in enumerate(channel_multipliers_encoder):
            current_channel_amount  = starting_channels * multiplier
            current_embed           = embedding_channels * multiplier
            
            for layer in range(resNet_per_level_encoder):
                current_block           = []

                current_block.append(
                    ResNetBlock(previous_channel_amount, 
                                current_channel_amount)
                )

                if level in attention_levels:
                    current_block.append(
                        self.__get_attention_block(current_channel_amount,
                                                   current_embed,
                                                   num_heads)
                    )
                
                self.encoder.append(SwitchSequential(*current_block))
                skip_channels.append(current_channel_amount)
                previous_channel_amount = current_channel_amount

            if level + 1 < len(channel_multipliers_encoder):
                self.encoder.append(SwitchSequential(
                    Downsample(current_channel_amount, asymmetric_padding=False)
                ))
                skip_channels.append(current_channel_amount)
        
        # Bottleneck
        self.bottleneck = SwitchSequential(
            ResNetBlock(current_channel_amount, current_channel_amount),
            self.__get_attention_block(current_channel_amount,
                                       current_embed,
                                       num_heads),
            ResNetBlock(current_channel_amount, current_channel_amount)
        )

        # Decoder 
        self.decoder = nn.ModuleList()

        for level_idx, multiplier in enumerate(channel_multipliers_decoder):
            # Reverse levels, because we are ascending
            level = len(channel_multipliers_decoder) - (level_idx + 1)

            current_channel_amount  = starting_channels * multiplier
            current_embed           = embedding_channels * multiplier

            for layer in range(resNet_per_level_decoder):
                current_block           = []
                current_skip_channel    = skip_channels.pop()
                
                # ResNet
                current_block.append(
                    ResNetBlock(current_skip_channel + previous_channel_amount, 
                                current_channel_amount)
                )
                # Attention
                if level in attention_levels:
                    current_block.append(
                        self.__get_attention_block(current_channel_amount,
                                                   current_embed,
                                                   num_heads) 
                    )
                # Upsampling
                if (layer + 1 == resNet_per_level_decoder and level):
                    current_block.append(
                        Upsample(current_channel_amount, True)
                    )

                self.decoder.append(SwitchSequential(*current_block))
                previous_channel_amount = current_channel_amount
        
        # Output
        self.norm           = Normalize(current_channel_amount, 32)                           
        self.output_conv    = nn.Conv2d(current_channel_amount, 
                                        input_channel_amount, 
                                        kernel_size=3, 
                                        padding=1)

    def forward(self, x, context, time):
        skip_connections = []

        # Adjusting Input
        x = self.input_conv(x)
        skip_connections.append(x)

        # Encoding
        for layers in self.encoder:
            x = layers(x, context, time)
            skip_connections.append(x)

        # Bottleneck
        x = self.bottleneck(x, context, time)

        # Decoding
        for layers in self.decoder:
            x = torch.cat((x, skip_connections.pop()), dim=1) 
            x = layers(x, context, time)
        
        # Adjusting Output
        x = self.norm(x)
        x = f.silu(x)
        x = self.output_conv(x)

        return x
    
    def __get_attention_block(self, 
                              channels, 
                              embedding_channels, 
                              amount_heads):
        if self.use_control_signal:
            return ContextualAttentionBlock(amount_heads, embedding_channels)
        else:
            return AttentionBlock(channels)