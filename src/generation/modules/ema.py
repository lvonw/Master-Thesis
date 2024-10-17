import copy
import torch.nn     as nn

class EMA(nn.Module):
    def __init__(self, model, configuration):
        super().__init__()
        self.ema_model = copy.deepcopy(model).eval().requires_grad_(False)
        
        self.ema_weight             = configuration["weight"]
        self.one_minus_ema_weight   = 1 - self.ema_weight 
        self.warmup_steps           = configuration["warm_up_steps"]
        self._current_step           = 0

    def ema_step(self, model):
        self._current_step += 1
        if self._current_step < self.warmup_steps:
            return 
        elif self._current_step == self.warmup_steps:
            self.ema_model.load_state_dict(model.state_dict())
            return 

        for model_params, ema_params in zip(model.parameters(), 
                                            self.ema_model.parameters()):

            old_weights     = ema_params.data * self.ema_weight
            new_weights     = model_params.data * self.one_minus_ema_weight
            ema_params.data = old_weights + new_weights


    def apply_to_model(self, model):
        model.load_state_dict(self.ema_model.state_dict())


    

    