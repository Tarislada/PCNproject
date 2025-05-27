import torch
class Config:
    """
    Configuration class for the PCN project.
    This class contains all the necessary parameters for model architecture,
    training, and pretraining.
    Attributes:
        LAYERS (list): List of layer sizes.
        USE_BIAS (bool): Whether to use bias in layers.
        UPWARD (bool): Direction of propagation.
        USE_TRUE_GRADIENT (bool): Whether to use true gradients.
        TRAIN_ERROR_WEIGHTS (bool): Whether to train error weights.
        
        LR_X (float): Learning rate for node activities.
        T_TRAIN (int): Number of training iterations.
        EPOCHS (int): Number of epochs for training.
        BATCH_SIZE (int): Size of each training batch.
        LEARNING_RATE (float): Learning rate for optimizer.
        KP_DECAY (float): Decay rate for Kolen-Pollack weight updates.
        
        PRETRAIN_EPOCHS (int): Number of epochs for pretraining.
        PRETRAIN_BATCH_SIZE (int): Batch size for pretraining.
        PRETRAIN_NUM_BATCHES (int): Number of batches for pretraining.
        
        DEVICE (str): Device to run the model on ('cuda' or 'cpu').
    """
    # Model parameters
    LAYERS = [784, 300, 300, 10]
    USE_BIAS = False
    UPWARD = True
    USE_TRUE_GRADIENT = False
    TRAIN_ERROR_WEIGHTS = True
    
    # Training parameters
    LR_X = 0.1
    T_TRAIN = 5
    EPOCHS = 50
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    KP_DECAY = 0.01
    
    # Pretraining parameters
    PRETRAIN_EPOCHS = 3
    PRETRAIN_BATCH_SIZE = 64
    PRETRAIN_NUM_BATCHES = 50
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class CIFAR10modelConfig(Config):
    """
    Configuration class for CIFAR-10 model.
    Inherits from Config and overrides specific parameters for CIFAR-10.
    """
    LAYERS = [3072, 512, 512, 10]  # CIFAR-10 input size is 32x32x3 = 3072
    USE_BIAS = True
    UPWARD = True
    USE_TRUE_GRADIENT = False
    TRAIN_ERROR_WEIGHTS = True
    
    LR_X = 0.01
    T_TRAIN = 10
    EPOCHS = 100
    BATCH_SIZE = 128
    LEARNING_RATE = 0.001
    KP_DECAY = 0.01
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class MNISTmodelConfig(Config):
    """
    Configuration class for MNIST model.
    Inherits from Config and overrides specific parameters for MNIST.
    """
    LAYERS = [784, 300, 300, 10]  # MNIST input size is 28x28 = 784
    USE_BIAS = True
    UPWARD = True
    USE_TRUE_GRADIENT = False
    TRAIN_ERROR_WEIGHTS = True
    
    LR_X = 0.1
    T_TRAIN = 5
    EPOCHS = 50
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    KP_DECAY = 0.01
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"