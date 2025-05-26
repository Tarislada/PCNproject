from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader
# from torch.nn.functional import one_hot
from Model import *
from randompretrain import *
from optimizers import *
from PRECO import *
from PRECO import optim
from PRECO.utils import onehot, sigmoid
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from randompretrain import pretrain_model, get_default_pretrain_config

structure = PCN_seperable_AMB(
    layers=[784, 300, 300, 10],  # Same as in original PC.py
    # layers = [3072, 1000, 1000, 1000, 10],  # For CIFAR10
    f=sigmoid,             # Or any activation function
    use_bias=False,              # Match original implementation
    upward=True,                 # For discriminative PCN
    use_true_gradient=False,     # Use separate weights
    train_error_weights=True     # Train the error weights
)

model = PCnet_KP(
    lr_x=0.1,                   # Inference rate
    T_train=5,                  # Inference iterations
    structure=structure,
    incremental=False,
    kp_decay=0.01                # Decay for KP update rule
)
# optimizer that uses Kolen-Pollack update rule
# optimizer = KPAdam(
#     model.params,
#     learning_rate=0.00005,
#     grad_clip=1,
#     batch_scale=False,
#     weight_decay=0.0
# )
# Optimizer that uses Adam update rule
optimizer = BaseAdam(
    model.params,
    learning_rate=0.0001,
    grad_clip=1,
    batch_scale=False,
    weight_decay=0.0
)

model.set_optimizer(optimizer)

MNIST_train = MNIST(root='data', train=True, download=True, transform=ToTensor())
MNIST_test = MNIST(root='data', train=False, download=True, transform=ToTensor())
CIFAR10_train = CIFAR10(root='data', train=True, download=True, transform=ToTensor())
CIFAR10_test = CIFAR10(root='data', train=False, download=True, transform=ToTensor())


MNIST_train_loader = DataLoader(
    dataset=MNIST_train,
    batch_size=64,
    shuffle=True
)

MNIST_test_loader = DataLoader(
    dataset=MNIST_test,
    batch_size=64,
    shuffle=False
)

CIFAR10_train_loader = DataLoader(
    dataset=CIFAR10_train,
    batch_size=128,
    shuffle=True
)
CIFAR10_test_loader = DataLoader(
    dataset=CIFAR10_test,
    batch_size=128,
    shuffle=False
)
train_loader = MNIST_train_loader
test_loader = MNIST_test_loader
# train_loader = CIFAR10_train_loader
# test_loader = CIFAR10_test_loader


# Datashape = [batch_size, channels, height, width]
# MNIST shape = [64, 1, 28, 28]
# CIFAR10 shape = [64, 3, 32, 32]
# print("Data loaded.")
# labels = not one-hot encoded

test_config = {
        'epochs': 30,
        'batch_size': 64,
        'input_size': 784,
        'output_size': 10,
        'num_batches': 50,
        'input_distribution': 'normal',
        'distribution_params': {
            'mean': 0.0,
            'std': 1.0
        }
    }

# Pretrain the model
# pretrain_config = get_default_pretrain_config()
pretrain_model(model, test_config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 50
print("Model initalized.")
with torch.no_grad():
    for epoch in range(epochs):
        for batch_idx, (x, y) in enumerate(train_loader):
            # Transform targets to one-hot
            # target = onehot(y, N=10)
            x = x.to(device)
            y = y.to(device)
            # Train the model
            model.train_supervised(x, y)

            # Print loss
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {model.get_energy():.4f}")

        print(f"Epoch [{epoch+1}/{epochs}] completed.")
        if epoch % 5 == 0:
            # Print the gradients
            for l in range(len(model.w)):
                if model.dw[l] is not None and model.de_w[l] is not None:
                    print(f"Layer {l} - fw_grad_norm: {model.dw[l].norm().item():.4f}, "
                        f"bw_grad_norm: {model.de_w[l].norm().item():.4f}")
            fw_norm = sum(w.norm().item() for w in model.w if w.numel() > 0)
            bw_norm = sum(w.norm().item() for w in model.e_w if w.numel() > 0)
            print(f"Forward weight norm: {fw_norm:.4f}, Backward weight norm: {bw_norm:.4f}")

                
        # Test the model

        for batch_idx, (x, y) in enumerate(test_loader):
            x = x.to(device)
            y = y.to(device)
            output = model.test_supervised(x)
            
            # Print loss
            loss = torch.nn.MSELoss()(output, onehot(y, N=10)).item()
            accuracy = torch.mean((torch.argmax(output, axis=1) == y).float()).item()
        
            if batch_idx % 100 == 0:
                print(f"Test Step [{batch_idx}/{len(test_loader)}], Loss: {loss:.4f}, Acc: {accuracy:.4f}")
        print(f"Test completed for epoch [{epoch+1}/{epochs}].")

        # Save the model
        # torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
        # print(f"Model saved for epoch [{epoch+1}/{epochs}].")

