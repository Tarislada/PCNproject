import torch
import numpy as np
import sys
import os
# Add the directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(project_root, 'Tightening-the-Biological-Constraints-on-Gradient-Based-Predictive-Coding'))
sys.path.append(os.path.join(project_root, 'PRECO'))
from PRECO.structure import PCNStructure
import PRECO

# Trying AMB structure first
class PCN_seperable_AMB(PCNStructure):
    """
    PCN structure with Kolen-Pollack-like separate weights for forward and backward paths.
    Addresses the weight transport problem with trainable error weights.
    Using AMB convention (mu = wf(x)+b).
    
    Args:
        layers (list): List of layer sizes.
        f (callable): Activation function.
        use_bias (bool): Whether to use bias in the layers.
        upward (bool): Whether the network is upward or downward.
        use_true_grad (bool): Whether to use true gradients (weight transposed) or seperate error weights.
        train_error_weights (bool): Whether to train error weights.
        fL (callable, optional): Optional activation function for the last layer.
    """
    def __init__(self, layers, f, use_bias, upward, use_true_gradient=False, 
                 train_error_weights=True, fL=None):
        super().__init__(layers, f, use_bias, upward, fL)
        self.train_error_weights = train_error_weights
        self.use_true_gradient = use_true_gradient
        
    def pred(self, l, x, w, b):
        """
        Compute prediction for layer l using AMB convention (mu = wf(x)+b).
        """
        k = l - 1 if self.upward else l + 1
        bias = b[k] if self.use_bias else 0
        out = torch.matmul(self.fl(x[k], l), w[k])
        return out + bias
    
    def grad_x(self, l, x, e, w, b, e_w, train):
        """
        Compute gradient of energy with respect to x
        Using either true gradients (weight transposed) or separate error weights.
        """
        k = l +1 if self.upward else l - 1
        
        if l != self.L:
            if self.use_true_gradient:
                # Use true gradients (weight transposed)
                grad = e[l] - self.dfldx(x[l],k) * (torch.matmul(e_w[k], w[k].T))
            else:
                # Use separate error weights
                grad = e[l] - self.dfldx(x[l],k) * (torch.matmul(e_w[k], e_w[l]))
        else:
            if train:
                grad = 0
            else:
                if self.upward:
                    grad = e[l]
                else:
                    if self.use_true_gradient:
                        grad = -self.dfldx(x[l],k) * (torch.matmul(e_w[k], w[k].T))
                    else:
                        grad = -self.dfldx(x[l],k) * (torch.matmul(e_w[k], e_w[l]))
        return grad
    
    def grad_w(self, l, x, e, w, b):
        """
        Compute gradient of energy with respect to forward weights.
        """
        k = l + 1 if self.upward else l - 1
        return -torch.matmul(self.fl(x[l].T,k),e[k])
    
    def grad_e_w(self, l, x, e, w, b, e_w):
        """
        Compute gradient of energy with respect to error weights.
        """
        k = l + 1 if self.upward else l - 1
        return -torch.matmul(e[k].T, self.fl(x[l], k))
    
    def grad_b(self, l, x, e, w, b):
        """
        Compute gradient of energy with respect to bias.
        """
        k = l + 1 if self.upward else l - 1
        return -e[k]
1