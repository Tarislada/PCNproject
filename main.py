import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from config import *
from Model import PCnet_KP
from Structure import PCN_seperable_AMB
from optimizers import KPAdam
from trainer import Trainer
from randompretrain import pretrain_model, get_default_pretrain_config
from visualizer import *
from PRECO.utils import sigmoid
from PRECO.optim import Adam as BaseAdam


def create_model(config):
    """Create and configure the model"""
    structure = PCN_seperable_AMB(
        layers=config.LAYERS,
        f=sigmoid,
        use_bias=config.USE_BIAS,
        upward=config.UPWARD,
        use_true_gradient=config.USE_TRUE_GRADIENT,
        train_error_weights=config.TRAIN_ERROR_WEIGHTS
    )
    
    model = PCnet_KP(
        lr_x=config.LR_X,
        T_train=config.T_TRAIN,
        structure=structure,
        incremental=False,
        kp_decay=config.KP_DECAY
    )
    
    # # Optimizer that uses Kolen-Pollack method
    # optimizer = KPAdam(
    #     model.params,
    #     learning_rate=config.LEARNING_RATE,
    #     grad_clip=1,
    #     batch_scale=False,
    #     weight_decay=0.0
    # )
    # Optimizer that does not use Kolen-Pollack method
    optimizer = BaseAdam(
        model.params,
        learning_rate=config.LEARNING_RATE,
        grad_clip=1,
        batch_scale=False,
        weight_decay=0.0
    )
    
    model.set_optimizer(optimizer)
    
    return model

def create_MNISTdata_loaders(config):
    """Create train and test data loaders"""
    train_dataset = MNIST(root='data', train=True, download=True, transform=ToTensor())
    test_dataset = MNIST(root='data', train=False, download=True, transform=ToTensor())
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader

def create_CIFAR10data_loaders(config):
    """Create CIFAR-10 train and test data loaders"""
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import ToTensor, Normalize, Compose
    
    transform = Compose([
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = CIFAR10(root='data', train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root='data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader

def main():
    """Main training pipeline"""
    # config = Config()
    config = MNISTmodelConfig()  # or CIFAR10modelConfig() for CIFAR-10
    
    # Create model and data loaders
    model = create_model(config)
    train_loader, test_loader = create_MNISTdata_loaders(config)
    
    weight_tracker = WeightAlignmentTracker(model)
    weight_tracker.capture_snapshot('before_pretraining')
    
    trainer = Trainer(model, config, weight_tracker=weight_tracker)
    
    # Pretraining (BEFORE main training)
    print("Starting pretraining...")
    pretrain_config = get_default_pretrain_config()
    pretrain_config['epochs'] = config.PRETRAIN_EPOCHS
    trainer.pretrain_model(model, pretrain_config)
    
    weight_tracker.capture_snapshot('after_pretraining')

    # Main training
    print("Starting main training...")
    
    with torch.no_grad():
        for epoch in range(config.EPOCHS):
            # Training
            trainer.train_epoch(train_loader, epoch)
            print(f"Epoch [{epoch+1}/{config.EPOCHS}] completed.")
            
            # Print statistics
            trainer.print_weight_stats(epoch)
            
            # Testing
            avg_loss, avg_acc = trainer.test_epoch(test_loader)
            print(f"Test completed - Avg Loss: {avg_loss:.4f}, Avg Acc: {avg_acc:.4f}")

    print("Generating weight alignment visualizations...")
    weight_tracker.plot_comparison_histograms()
    weight_tracker.plot_alignment_progression()
    
    # print("Analyzing model representations...")
    # mnist_repr, mnist_labels, cifar_repr, cifar_labels = analyze_representations(
    #     model, MNIST_test_loader=, CIFAR10_test_loader=, device=config.DEVICE)

if __name__ == "__main__":
    main()