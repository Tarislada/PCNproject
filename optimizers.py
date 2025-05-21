from PRECO.optim import Adam as BaseAdam
import torch
import numpy as np
from PRECO.utils import *

class KPAdam(BaseAdam):
    """
    Adam optimizer with support for Kolen-Pollack backward weight updates.
    Extends the PRECO Adam optimizer to handle error weights separately.
    """

    def __init__(self, params, learning_rate, batch_scale=False, grad_clip=None, 
                 beta1=0.9, beta2=0.999, epsilon=1e-7, weight_decay=0, AdamW=False, kp_decay=0.01,backward_lr_scale=1.0):
        super().__init__(params, learning_rate, batch_scale, grad_clip, beta1, beta2, epsilon, weight_decay, AdamW)
        self.kp_decay = kp_decay
        self.backward_lr_scale = backward_lr_scale  # Optional scale factor for backward weight learning rate
        
        # Initialize error weight momentum/velocity if needed
        if "e_w" in params:
            self.m_e_w = [torch.zeros_like(w, device=w.device) for w in params["e_w"]]
            self.v_e_w = [torch.zeros_like(w, device=w.device) for w in params["e_w"]]
    
    def step(self, params, grads, batch_size):
        # Call the parent class step first
        super().step(params, grads, batch_size)
        
        # Additionally handle error weights with Kolen-Pollack rule
        if "e_w" in params and "e_w" in grads:
            self.t += 1
            lr_t = self.learning_rate * self.backward_lr_scale * np.sqrt(1. - self.beta2 ** self.t) / (1. - self.beta1 ** self.t)
            
            for i in range(len(params["e_w"])):
                if grads["e_w"][i] is not None:
                    # Apply Kolen-Pollack decay
                    grads["e_w"][i] = grads["e_w"][i] - self.kp_decay * params["e_w"][i]
                    
                    # Update using Adam
                    self._update_single_param(params["e_w"], grads["e_w"], 
                                             self.m_e_w, self.v_e_w, 
                                             i, batch_size, lr_t, self.AdamW)