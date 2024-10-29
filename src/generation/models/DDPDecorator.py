import torch

from torch.autograd         import profiler
from torch.nn.parallel      import DistributedDataParallel

class DDPTrainingDecorator(DistributedDataParallel):
    """
    Custom Implementation of DDP so that we use Trainingstep instead of forward

    if this doesnt work we have to change the forward methods to call
    training step instead
    """
    def forward(self, *args, **kwargs):
        with profiler.record_function("DistributedDataParallel.forward"):
            args, kwargs = self._pre_forward(*args, **kwargs)
            output = (
                self.module.training_step(*args, **kwargs)
                #self.module.forward(*inputs, **kwargs)
                if self._delay_all_reduce_all_params
                else self._run_ddp_forward(*args, **kwargs)
            )
            return self._post_forward(output)
        
    def training_step(self, *args, **kwargs):
        return self(*args, **kwargs)
    
    def on_training_step_completed(self, *args, **kwargs):
        return self.module.on_training_step_completed(*args, **kwargs)