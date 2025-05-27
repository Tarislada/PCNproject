import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import math

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

class WeightAlignmentTracker:
    def __init__(self, model):
        self.model = model
        self.before_pretraining = None
        self.after_pretraining = None
        self.alignment_history = []
        self.epoch_history = []
    
    def compute_cosine_similarities(self):
        """Compute cosine similarities between corresponding rows of W^T and B"""
        similarities = {}
        angles = {}
        
        for l in range(len(self.model.w)):
            if self.model.w[l].numel() > 0 and self.model.e_w[l].numel() > 0:
                W_T = self.model.w[l].T.detach().cpu()
                B = self.model.e_w[l].detach().cpu()
                
                cos_sims = []
                row_angles = []
                
                for i in range(W_T.shape[0]):
                    w_row = W_T[i] / (torch.norm(W_T[i]) + 1e-8)
                    b_row = B[i] / (torch.norm(B[i]) + 1e-8)
                    
                    cos_sim = torch.sum(w_row * b_row).item()
                    cos_sims.append(cos_sim)
                    
                    angle = math.acos(min(max(cos_sim, -1.0), 1.0)) * 180 / math.pi
                    row_angles.append(angle)
                
                similarities[f'layer_{l}'] = np.array(cos_sims)
                angles[f'layer_{l}'] = np.array(row_angles)
        
        return similarities, angles
    
    def capture_snapshot(self, label):
        """Capture current weight alignment state"""
        similarities, angles = self.compute_cosine_similarities()
        setattr(self, f'{label}_sim', similarities)
        setattr(self, f'{label}_angles', angles)
        print(f"Captured weight alignment: {label}")
    
    def record_alignment_epoch(self, epoch):
        """Record alignment for a specific epoch"""
        _, angles = self.compute_cosine_similarities()
        
        # Calculate mean angle across all layers
        all_angles = []
        for layer_angles in angles.values():
            all_angles.extend(layer_angles)
        
        if all_angles:
            mean_angle = np.mean(all_angles)
            self.alignment_history.append(mean_angle)
            self.epoch_history.append(epoch)
    
    def print_alignment_stats(self, epoch):
        """Print current alignment statistics"""
        _, angles = self.compute_cosine_similarities()
        
        print(f"\n--- Weight Alignment Stats (Epoch {epoch}) ---")
        for layer_name, layer_angles in angles.items():
            mean_angle = np.mean(layer_angles)
            std_angle = np.std(layer_angles)
            print(f"{layer_name}: Mean={mean_angle:.1f}°, Std={std_angle:.1f}°")
    
    def plot_comparison_histograms(self, save_path="weight_alignment_comparison.png"):
        """Plot before vs after comparison"""
        if not hasattr(self, 'before_pretraining_angles') or not hasattr(self, 'after_pretraining_angles'):
            print("Error: Need both before and after pretraining data!")
            return
            
        n_layers = len(self.before_pretraining_angles)
        fig, axes = plt.subplots(1, n_layers, figsize=(n_layers*5, 4))
        
        if n_layers == 1:
            axes = [axes]
        
        for idx, layer in enumerate(self.before_pretraining_angles.keys()):
            before_angles = self.before_pretraining_angles[layer]
            after_angles = self.after_pretraining_angles[layer]
            
            axes[idx].hist(before_angles, bins=20, alpha=0.6, color='skyblue', 
                          label=f'Before (μ={np.mean(before_angles):.1f}°)', density=True)
            axes[idx].hist(after_angles, bins=20, alpha=0.6, color='orange',
                          label=f'After (μ={np.mean(after_angles):.1f}°)', density=True)
            
            axes[idx].axvline(np.mean(before_angles), color='blue', linestyle='--', alpha=0.8)
            axes[idx].axvline(np.mean(after_angles), color='red', linestyle='--', alpha=0.8)
            axes[idx].axvline(90, color='black', linestyle='-', alpha=0.3, label='Orthogonal')
            
            axes[idx].set_title(f'{layer.replace("_", " ").title()}')
            axes[idx].set_xlabel('Angle (degrees)')
            axes[idx].set_ylabel('Density')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_xlim(0, 180)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_alignment_progression(self, save_path="alignment_progression.png"):
        """Plot alignment progression over time"""
        if not self.alignment_history:
            print("No alignment history to plot")
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(self.epoch_history, self.alignment_history, 'o-', linewidth=2, markersize=4)
        plt.axhline(y=90, color='black', linestyle='--', alpha=0.7, label="Orthogonal (90°)")
        
        plt.title('Weight Alignment During Training', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Mean Angle W^T vs B (degrees)', fontsize=14)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.ylim(0, 180)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()