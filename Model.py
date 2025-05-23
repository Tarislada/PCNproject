import torch
import numpy as np
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(project_root, 'Tightening-the-Biological-Constraints-on-Gradient-Based-Predictive-Coding'))
sys.path.append(os.path.join(project_root, 'PRECO'))

from Structure import *
from PRECO import optim
from PRECO import PCN
from PRECO.PCN import PCnet
from PRECO.utils import *


DEVICE = torch.device('cuda:0')

class PCnet_KP(PCN.PCnet):
    """
    PCnet with separate trainable weights for forward and backward paths.
    
    Args:
        lr_x (float): Learning rate for nodes.
        T_train (int): Number of training iterations.
        structure (PCN_AMB_KP): Structure of the model.
        incremental (bool): Whether to use incremental EM.
        min_delta (float): Minimum change in energy for early stopping.
        early_stop (bool): Whether to use early stopping.
        kp_decay (float): Decay rate for Kolen-Pollack weight updates.
    """
    
    def __init__(self, lr_x, T_train, structure, incremental=False,
                 min_delta=0, early_stop=False, kp_decay=0.01, T_test=None):
        # Ensure structure is a PCN_seperable_AMB instance
        assert isinstance(structure, PCN_seperable_AMB), "Structure must be an instance of PCN_seperable_AMB"
        super().__init__(lr_x, T_train, structure, incremental, min_delta, early_stop)
        self.kp_decay = kp_decay
        self.T_test = T_test if T_test is not None else T_train
    
    def _reset_params(self):
        super()._reset_params()
        # Initialize error weigths
        self.e_w = []
        for l in range(len(self.w)):
            if self.w[l].numel() > 0:
                if self.structure.use_true_gradient:
                    self.e_w.append(self.w[l].T.clone())
                else:
                    # Initialize as random weights same shape as transposed
                    shape = (self.w[l].shape[1],self.w[l].shape[0])
                    self.e_w.append(torch.randn(shape, device=DEVICE))
                    self.weight_init(self.e_w[-1])
            else:
                self.e_w.append(torch.empty(0,0,device=DEVICE))
                
    def _reset_grad(self):
        super()._reset_grad()
        # Reset gradients for error weights
        self.de_w = [None for _ in range(len(self.w))]
    
    def update_xs(self,train=True):
        """
        Update node activities using true gradients for separate weights.
        """
        if self.early_stop:
            early_stopper = optim.EarlyStopper(patience=0, min_delta=self.min_delta)
        
        T = self.T_train if train else self.T_test
        
        for t in range(T):
            # Error computation
            for l in self.structure.error_layers:
                self.e[l] = self.x[l] - self.strcuture.pred(l, self.x, self.w, self.b)
                
            # Update hidden layers
            for l in self.structure.hidden_layers:
                dEdx = self.structure.grad_x(l, self.x, self.e, self.w, self.b, self.e_w, train)
                self.x[l] -= self.lr_x*dEdx
            
            if self.incremental:
                self.update_w()
                if not self.structure.use_true_gradient and self.structure.train_error_weights:
                    self.update_e_w()
                self.optimizer.step(self.params, self.grads, batch_size=1)
            
            if self.early_stop and early_stopper.early_stop(self.get_energy()):
                break
    
    def train_updates(self, batch_no=None):
        """
        Override to properly pass e_w parameter to structure.grad_x
        due to separate error weights e_w not working for PRECO PCN
        """
        # First compute errors (same as parent class)
        for l in self.error_layers:
            self.e[l] = self.x[l] - self.structure.pred(l, self.x, self.w, self.b)
        
        if self.early_stop:
            early_stopper = optim.EarlyStopper(patience=0, min_delta=self.min_delta)
    
        for t in range(self.T_train): 
            # Update hidden nodes with custom grad_x that uses e_w
            for l in self.hidden_layers: 
                dEdx = self.structure.grad_x(l=l, train=True, x=self.x, e=self.e, w=self.w, b=self.b, e_w=self.e_w)
                self.x[l] -= self.lr_x*dEdx
    
            # Recompute errors
            for l in self.error_layers:
                self.e[l] = self.x[l] - self.structure.pred(l, self.x, self.w, self.b)
    
            # Rest is same as parent
            if self.incremental and self.dw.count(None) <= 1:
                self.optimizer.step(self.params, self.grads, batch_size=self.x.shape[0])
    
            if self.early_stop:
                if early_stopper.early_stop(self.get_energy()):
                    print(f"\nEarly stopping inference at t={t}.")          
                    break
    
    def update_w(self):
        """
        Update forward weights
        """
        super().update_w()
        
    def update_e_w(self):
        """
        Update error weights using either Kolen-Pollack or direct gradient
        """
        for l in self.structure.weight_layers:
            if self.w[l].numel() > 0: 
                self.de_w[l] = self.structure.grad_e_w(l, self.x, self.e, self.w, self.e_w)
                
                # Apply Kolen-Pollack rule with decay
                if self.kp_decay > 0:
                    self.de_w[l] = self.de_w[l] - self.kp_decay * self.e_w[l]
                else:
                    self.de_w[l] = self.de_w[l] - self.structure.grad_e_w(l, self.x, self.e, self.w, self.b, self.e_w)
                
    @property
    def params(self):
        """
        Returns the parameters of the model.
        """
        params = super().params
        if not self.structure.use_true_gradient and self.structure.train_error_weights:
            params["e_w"] = self.e_w
        return params
    
    @property
    def grads(self):
        """
        Returns the gradients of the model.
        """
        grads = super().grads
        if not self.structure.use_true_gradient and self.structure.train_error_weights:
            grads["e_w"] = self.de_w
        return grads
    