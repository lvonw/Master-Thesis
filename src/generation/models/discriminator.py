import torch
import torch.functional as f
import torch.nn         as nn

class Discriminator(nn.Module):
    """ Inspired by taming NLayerDiscriminator 
    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/discriminator/model.py
    """


    # TODO Config
    def __init__(self, 
                 amount_input_channels = 1, 
                 amount_layers = 3,
                 starting_channels = 64):
        super().__init__()

        # They do some weird weight init, might have to do that too

        self.input_conv = nn.Conv2d(amount_input_channels,
                                    starting_channels,
                                    kernel_size = 3,
                                    stride      = 2,
                                    padding     = 1,
                                    bias        = False) 
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)


        self.discriminator  = nn.ModuleList()
        current_channel_amount = starting_channels

        for layer in range(amount_layers):
            previous_channel_amount = current_channel_amount
            current_channel_amount  *= 2

            # Downsample (might actually make sense to use util_module down)
            self.discriminator.append(nn.Conv2d(previous_channel_amount,
                                                current_channel_amount,
                                                kernel_size = 3,
                                                stride      = 2,
                                                padding     = 1,
                                                bias        = False))
            # Normalize
            self.discriminator.append(nn.BatchNorm2d(current_channel_amount))
        
            # Activation
            self.discriminator.append(nn.LeakyReLU(0.2, inplace=True))

        self.output_conv = nn.Conv2d(current_channel_amount,
                                     1,
                                     kernel_size    = 3,
                                     stride         = 1,
                                     padding        = 1)            

    def forward(self, x):
        x = self.input_conv(x)
        x = self.leaky_relu(x)

        for module in self.discriminator:
            x = module(x)
        
        x = self.output_conv(x)

        return x
    