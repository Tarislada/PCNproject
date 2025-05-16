from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
# from torch.nn.functional import one_hot
from Model import *
from PRECO import *
from PRECO import optim
from PRECO.utils import onehot, sigmoid
import torch
from torch.utils.data import DataLoader

structure = PCN_seperable_AMB(
    layers=[784, 300, 300, 10],  # Same as in original PC.py
    f=sigmoid,             # Or any activation function
    use_bias=False,              # Match original implementation
    upward=True,                 # For discriminative PCN
    use_true_gradient=False,     # Use separate weights
    train_error_weights=True     # Train the error weights
)

model = PCnet_KP(
    lr_x=0.1,                   # Inference rate
    T_train=20,                  # Inference iterations
    structure=structure,
    incremental=False,
    kp_decay=0.01                # Decay for KP update rule
)

optimizer = optim.Adam(
    model.params,
    learning_rate=0.001,
    weight_decay=0.0
)
model.set_optimizer(optimizer)


train_loader = DataLoader(
    dataset=MNIST(root='data', train=True, download=True, transform=None),
    batch_size=64,
    shuffle=True
)

epochs = 10
print("Model initalized.")

for epoch in range(epochs):
    for batch_idx, (images, y) in enumerate(train_loader):
        # Transform targets to one-hot
        target = onehot(y, N=10)
        
        # Train the model
        model.train_supervised(images, target)