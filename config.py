import torch
class Config:
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
    