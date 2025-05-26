import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns

class ModelVisualizer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    def extract_representations(self, dataloader, max_batches=10):
        """
        Extract hidden layer representations from the model.
        
        Args:
            dataloader: DataLoader for the dataset
            max_batches: Maximum number of batches to process
            
        Returns:
            dict: Dictionary containing representations for each layer
        """
        representations = {l: [] for l in self.model.hidden_layers}
        labels = []
        
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break
                    
                x, y = x.to(self.device), y.to(self.device)
                
                # Forward pass through the model
                self.model.reset_nodes()
                self.model.clamp_input(x)
                self.model.forward(self.model.error_layers)
                
                # Extract hidden layer activations
                for l in self.model.hidden_layers:
                    representations[l].append(self.model.x[l].cpu().numpy())
                
                labels.append(y.cpu().numpy())
        
        # Concatenate all batches
        for l in representations:
            representations[l] = np.concatenate(representations[l], axis=0)
        labels = np.concatenate(labels)
        
        return representations, labels
    
    def plot_pca_analysis(self, representations, labels, dataset_name):
        """Plot PCA analysis of representations"""
        n_layers = len(representations)
        fig, axes = plt.subplots(1, n_layers, figsize=(5*n_layers, 4))
        
        if n_layers == 1:
            axes = [axes]
        
        for idx, (layer_idx, layer_repr) in enumerate(representations.items()):
            # Apply PCA
            pca = PCA(n_components=2)
            repr_2d = pca.fit_transform(layer_repr)
            
            # Plot
            scatter = axes[idx].scatter(repr_2d[:, 0], repr_2d[:, 1], 
                                     c=labels, cmap='tab10', alpha=0.7, s=20)
            axes[idx].set_title(f'{dataset_name} - Layer {layer_idx}\n'
                              f'Explained Variance: {pca.explained_variance_ratio_.sum():.3f}')
            axes[idx].set_xlabel('First Principal Component')
            axes[idx].set_ylabel('Second Principal Component')
            
            # Add colorbar
            plt.colorbar(scatter, ax=axes[idx])
        
        plt.tight_layout()
        plt.savefig(f'{dataset_name}_pca_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_tsne_analysis(self, representations, labels, dataset_name):
        """Plot t-SNE analysis of representations"""
        n_layers = len(representations)
        fig, axes = plt.subplots(1, n_layers, figsize=(5*n_layers, 4))
        
        if n_layers == 1:
            axes = [axes]
        
        for idx, (layer_idx, layer_repr) in enumerate(representations.items()):
            # Apply t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            repr_2d = tsne.fit_transform(layer_repr)
            
            # Plot
            scatter = axes[idx].scatter(repr_2d[:, 0], repr_2d[:, 1], 
                                     c=labels, cmap='tab10', alpha=0.7, s=20)
            axes[idx].set_title(f'{dataset_name} - Layer {layer_idx} (t-SNE)')
            axes[idx].set_xlabel('t-SNE 1')
            axes[idx].set_ylabel('t-SNE 2')
            
            # Add colorbar
            plt.colorbar(scatter, ax=axes[idx])
        
        plt.tight_layout()
        plt.savefig(f'{dataset_name}_tsne_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_activation_distributions(self, representations, dataset_name):
        """Plot distribution of activations for each layer"""
        n_layers = len(representations)
        fig, axes = plt.subplots(n_layers, 1, figsize=(10, 3*n_layers))
        
        if n_layers == 1:
            axes = [axes]
        
        for idx, (layer_idx, layer_repr) in enumerate(representations.items()):
            # Plot histogram of activations
            axes[idx].hist(layer_repr.flatten(), bins=50, alpha=0.7, density=True)
            axes[idx].set_title(f'{dataset_name} - Layer {layer_idx} Activation Distribution')
            axes[idx].set_xlabel('Activation Value')
            axes[idx].set_ylabel('Density')
            
            # Add statistics
            mean_act = np.mean(layer_repr)
            std_act = np.std(layer_repr)
            axes[idx].axvline(mean_act, color='red', linestyle='--', 
                            label=f'Mean: {mean_act:.3f}')
            axes[idx].axvline(mean_act + std_act, color='orange', linestyle='--', 
                            label=f'±1 STD: {std_act:.3f}')
            axes[idx].axvline(mean_act - std_act, color='orange', linestyle='--')
            axes[idx].legend()
        
        plt.tight_layout()
        plt.savefig(f'{dataset_name}_activation_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def compare_datasets(self, mnist_repr, mnist_labels, cifar_repr, cifar_labels):
        """Compare representations between MNIST and CIFAR10"""
        fig, axes = plt.subplots(2, len(mnist_repr), figsize=(5*len(mnist_repr), 8))
        
        datasets = [('MNIST', mnist_repr, mnist_labels), ('CIFAR10', cifar_repr, cifar_labels)]
        
        for row, (name, representations, labels) in enumerate(datasets):
            for col, (layer_idx, layer_repr) in enumerate(representations.items()):
                # Apply PCA
                pca = PCA(n_components=2)
                repr_2d = pca.fit_transform(layer_repr)
                
                # Plot
                scatter = axes[row, col].scatter(repr_2d[:, 0], repr_2d[:, 1], 
                                               c=labels, cmap='tab10', alpha=0.7, s=20)
                axes[row, col].set_title(f'{name} - Layer {layer_idx}')
                axes[row, col].set_xlabel('PC1')
                axes[row, col].set_ylabel('PC2')
        
        plt.tight_layout()
        plt.savefig('dataset_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

def analyze_representations(model, mnist_loader, cifar_loader, device):
    """
    Main function to analyze and visualize model representations
    """
    visualizer = ModelVisualizer(model, device)
    
    print("Extracting MNIST representations...")
    mnist_repr, mnist_labels = visualizer.extract_representations(mnist_loader, max_batches=10)
    
    print("Extracting CIFAR10 representations...")
    cifar_repr, cifar_labels = visualizer.extract_representations(cifar_loader, max_batches=10)
    
    print("Generating visualizations...")
    
    # Individual dataset analysis
    visualizer.plot_pca_analysis(mnist_repr, mnist_labels, "MNIST")
    visualizer.plot_pca_analysis(cifar_repr, cifar_labels, "CIFAR10")
    
    visualizer.plot_tsne_analysis(mnist_repr, mnist_labels, "MNIST")
    visualizer.plot_tsne_analysis(cifar_repr, cifar_labels, "CIFAR10")
    
    visualizer.plot_activation_distributions(mnist_repr, "MNIST")
    visualizer.plot_activation_distributions(cifar_repr, "CIFAR10")
    
    # Comparison between datasets
    visualizer.compare_datasets(mnist_repr, mnist_labels, cifar_repr, cifar_labels)
    
    print("Visualization complete! Check the saved PNG files.")
    
    return mnist_repr, mnist_labels, cifar_repr, cifar_labels