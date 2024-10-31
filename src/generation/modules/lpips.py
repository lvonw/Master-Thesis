"""Adjusts version of LPIPS from Taming Transformers and Original LPIPS repo. 
Because for some reason conda cant download LPIPS, pretrained weights are
taken from the original LPIPS Repository"""

import torch
import util

import torch.nn             as nn

from debug                  import Printer, LogLevel
from torchvision            import models

class LPIPS(nn.Module):
    def __init__(self, net_type="vgg"):
        super().__init__()
        self.model_family   = "lpips"
        self.name           = net_type

        if net_type == "vgg" or net_type == "vgg16":
            self.channels   = [64, 128, 256, 512, 512]
            self.net        = VGG16()
        else: 
            Printer().print_log(f"{net_type} not valid for LPIPS",
                                LogLevel.ERROR)
            return 

        self.scaling_layer = ScalingLayer()
        
        self.lin0 = NetLinLayer(self.channels[0])
        self.lin1 = NetLinLayer(self.channels[1])
        self.lin2 = NetLinLayer(self.channels[2])
        self.lin3 = NetLinLayer(self.channels[3])
        self.lin4 = NetLinLayer(self.channels[4])
        
        Printer().print_log(f"Loading LPIPS with: {net_type}")
        if not util.load_model(self, strict=False):
            Printer().print_log(f"Could not load net {net_type}",
                                LogLevel.ERROR)
            return

        self.linear_layers = [
            self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        self.amount_layers = len(self.linear_layers)
        
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, target):
        # LPIPS assumes we have exactly 3 channels, so we need to adjust the
        # shape for black and white images
        if x.shape[1] == 1:
            x       = x.repeat(1, 3, 1, 1).contiguous()
            target  = target.repeat(1, 3, 1, 1).contiguous() 

        scaled_x        = self.scaling_layer(x)
        scaled_target   = self.scaling_layer(target) 
                                
        activations_x       = self.net(scaled_x)
        activations_target  = self.net(scaled_target)

        res = 0
        for idx in range(self.amount_layers):
            activation_x        = normalize_tensor(activations_x[idx])
            activation_target   = normalize_tensor(activations_target[idx])
            l2_norm             = (activation_x - activation_target) ** 2
            
            res += spatial_average(self.linear_layers[idx](l2_norm))

        return res


class ScalingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "shift", torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer(
            "scale", torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, x):
        return (x - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """
    def __init__(self, input_channels, output_channels=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Dropout(),
            nn.Conv2d(input_channels, 
                      output_channels, 
                      kernel_size   = 1, 
                      stride        = 1, 
                      padding       = 0, 
                      bias          = False))
    
    def forward(self, x):
        return self.model(x)
    

class VGG16(nn.Module):
    def __init__(self):
        super().__init__()
        vgg_pretrained_features = models.vgg16(
            weights=models.VGG16_Weights.IMAGENET1K_V1).features
        
        self.slice_1 = nn.Sequential()
        self.slice_2 = nn.Sequential()
        self.slice_3 = nn.Sequential()
        self.slice_4 = nn.Sequential()
        self.slice_5 = nn.Sequential()
        
        for x in range(4):
            self.slice_1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice_2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice_3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice_4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice_5.add_module(str(x), vgg_pretrained_features[x])
        
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x           = self.slice_1(x)
        relu_1_2    = x
        x           = self.slice_2(x)
        relu_2_2    = x
        x           = self.slice_3(x)
        relu_3_3    = x
        x           = self.slice_4(x)
        relu_4_3    = x
        x           = self.slice_5(x)
        relu_5_3    = x

        return (relu_1_2, relu_2_2, relu_3_3, relu_4_3, relu_5_3)

def normalize_tensor(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True)) + eps
    return x / norm_factor

def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keepdim = keepdim)