import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def generate_random_data(batch_size, input_size, output_size, num_batches=100, 
                        input_distribution='uniform', distribution_params=None):
    """
    Generate random data for pretraining with various distributions.
    
    Args:
        batch_size (int): Size of each batch
        input_size (int): Input dimension
        output_size (int): Output dimension  
        num_batches (int): Number of batches to generate
        input_distribution (str): Type of distribution ('uniform', 'normal', 'bernoulli', 'exponential')
        distribution_params (dict): Parameters for the distribution
        
    Returns:
        DataLoader: DataLoader with random data
    """
    total_samples = num_batches * batch_size
    
    # Set default parameters if not provided
    if distribution_params is None:
        distribution_params = {}
    
    # Generate random inputs based on distribution type
    if input_distribution == 'uniform':
        low = distribution_params.get('low', 0.0)
        high = distribution_params.get('high', 1.0)
        random_inputs = torch.rand(total_samples, input_size) * (high - low) + low
        
    elif input_distribution == 'normal':
        mean = distribution_params.get('mean', 0.0)
        std = distribution_params.get('std', 1.0)
        random_inputs = torch.randn(total_samples, input_size) * std + mean
        
    elif input_distribution == 'bernoulli':
        prob = distribution_params.get('prob', 0.5)
        random_inputs = torch.bernoulli(torch.full((total_samples, input_size), prob))
        
    elif input_distribution == 'exponential':
        rate = distribution_params.get('rate', 1.0)
        random_inputs = torch.exponential(torch.full((total_samples, input_size), 1.0/rate))
        
    elif input_distribution == 'beta':
        alpha = distribution_params.get('alpha', 1.0)
        beta = distribution_params.get('beta', 1.0)
        random_inputs = torch.distributions.Beta(alpha, beta).sample((total_samples, input_size))
        
    elif input_distribution == 'mnist_like':
        # Generate data similar to MNIST statistics
        mean = distribution_params.get('mean', 0.1307)
        std = distribution_params.get('std', 0.3081)
        random_inputs = torch.randn(total_samples, input_size) * std + mean
        random_inputs = torch.clamp(random_inputs, 0, 1)  # Clamp to valid range
        
    else:
        raise ValueError(f"Unsupported distribution: {input_distribution}")
    
    # Generate random labels
    label_distribution = distribution_params.get('label_distribution', 'uniform')
    if label_distribution == 'uniform':
        random_labels = torch.randint(0, output_size, (total_samples,))
    elif label_distribution == 'weighted':
        # Allow weighted label distribution
        weights = distribution_params.get('label_weights', None)
        if weights is None:
            weights = torch.ones(output_size)
        random_labels = torch.multinomial(weights, total_samples, replacement=True)
    else:
        random_labels = torch.randint(0, output_size, (total_samples,))
    
    dataset = TensorDataset(random_inputs, random_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# def pretrain_model(model, config, tracker=None):
#     """
#     Pretrain model with random data.
    
#     Args:
#         model: PCnet_KP model instance
#         config: Dictionary with pretraining parameters
#     """
#     print(f"Starting pretraining for {config['epochs']} epochs...")
#     print(f"Using {config['input_distribution']} distribution for inputs")
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     with torch.no_grad():
#         for epoch in range(config['epochs']):
#             pretrain_loader = generate_random_data(
#                 config['batch_size'], 
#                 config['input_size'], 
#                 config['output_size'], 
#                 config['num_batches'],
#                 config['input_distribution'],
#                 config.get('distribution_params', {})
#             )
            
#             epoch_energy = 0
#             batch_count = 0
            
#             for batch_idx, (x, y) in enumerate(pretrain_loader):
#                 x = x.to(device)
#                 y = y.to(device)
                
#                 # Reshape input to match expected format if needed
#                 if len(x.shape) == 2 and config['input_size'] == 784:
#                     x = x.view(x.shape[0], 1, 28, 28)
                
#                 # Train the model
#                 model.train_supervised(x, y)
                
#                 current_energy = model.get_energy()
#                 epoch_energy += current_energy
#                 batch_count += 1
                
#                 if batch_idx % 20 == 0:
#                     print(f"Pretrain Epoch [{epoch+1}/{config['epochs']}], "
#                           f"Batch [{batch_idx}/{len(pretrain_loader)}], "
#                           f"Energy: {current_energy:.4f}")
#             if tracker:
#                 tracker.record_alignment_epoch(epoch)
            
            
#             avg_energy = epoch_energy / batch_count
#             print(f"Pretrain Epoch [{epoch+1}/{config['epochs']}] completed. Average Energy: {avg_energy:.4f}")
            
#             # Monitor weight norms during pretraining
#             if epoch % 2 == 0:
#                 fw_norm = sum(w.norm().item() for w in model.w if w.numel() > 0)
#                 bw_norm = sum(w.norm().item() for w in model.e_w if w.numel() > 0)
#                 print(f"Pretrain - Forward weight norm: {fw_norm:.4f}, Backward weight norm: {bw_norm:.4f}")
    
#     print("Pretraining completed!")

def get_default_pretrain_config():
    """Return default pretraining configuration."""
    return {
        'epochs': 20,
        'batch_size': 64,
        'input_size': 784,
        'output_size': 10,
        'num_batches': 50,
        'input_distribution': 'uniform',
        'distribution_params': {
            'low': 0.0,
            'high': 1.0
        }
    }

def get_pretrain_config_normal():
    """Return pretraining configuration with normal distribution."""
    return {
        'epochs': 20,
        'batch_size': 64,
        'input_size': 784,
        'output_size': 10,
        'num_batches': 50,
        'input_distribution': 'normal',
        'distribution_params': {
            'mean': 0.0,
            'std': 0.5
        }
    }

def get_pretrain_config_mnist_like():
    """Return pretraining configuration mimicking MNIST statistics."""
    return {
        'epochs': 20,
        'batch_size': 64,
        'input_size': 784,
        'output_size': 10,
        'num_batches': 50,
        'input_distribution': 'mnist_like',
        'distribution_params': {
            'mean': 0.1307,
            'std': 0.3081
        }
    }