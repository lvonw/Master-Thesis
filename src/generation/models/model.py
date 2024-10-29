import torch.nn as nn

class Model(nn.Module):
    def forward(self, training=False, *args, **kwargs):
        if training:
            return self.training_step 


    def training_step(self, *args, **kwargs):
        pass