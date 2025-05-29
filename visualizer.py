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
        # For snapshots
        self.before_pretraining_sim = None
        self.before_pretraining_angles = None
        self.after_pretraining_sim = None
        self.after_pretraining_angles = None
        
        # For tracking progression
        self.epoch_history = []
        self.alignment_history = []
        self.angle_means = []  # Mean angle per epoch
        self.angle_stds = []   # Std dev of angles per epoch
        
        # For weight distributions
        self.fw_distributions = []  # Forward weight distributions per epoch
        self.bw_distributions = []  # Backward weight distributions per epoch
        
        # Weight norms over time
        self.fw_norm_history = []
        self.bw_norm_history = []
        
        # Per-layer statistics
        self.layer_stats = {}  # Detailed stats per layer over time
    
    def compute_cosine_similarities(self):
        """Compute cosine similarities between corresponding rows of W^T and B"""
        similarities = {}
        angles = {}
        
        # TODO: Check if W^T is correct, or W 
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
    
    def record_alignment_epoch_simple(self, epoch):
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
    
    def record_alignment_epoch(self, epoch):
        """Record detailed weight and alignment data for this epoch"""
        # Get angle data
        _, angles = self.compute_cosine_similarities()
        
        # Calculate overall stats across all layers
        all_angles = []
        for layer_angles in angles.values():
            all_angles.extend(layer_angles)
        
        if all_angles:
            mean_angle = np.mean(all_angles)
            std_angle = np.std(all_angles)
            self.angle_means.append(mean_angle)
            self.angle_stds.append(std_angle)
            self.alignment_history.append(mean_angle)
            self.epoch_history.append(epoch)
        
        # Record per-layer stats in layer_stats
        for layer_name, layer_angles in angles.items():
            if layer_name not in self.layer_stats:
                self.layer_stats[layer_name] = {'angles': [], 'fw_dist': [], 'bw_dist': []}
            
            self.layer_stats[layer_name]['angles'].append({
                'epoch': epoch,
                'mean': np.mean(layer_angles),
                'std': np.std(layer_angles)
            })
        
        # Record weight distribution data
        fw_dist = []
        bw_dist = []
        
        for l in range(len(self.model.w)):
            if self.model.w[l].numel() > 0 and self.model.e_w[l].numel() > 0:
                fw_values = self.model.w[l].detach().cpu().numpy().flatten()
                bw_values = self.model.e_w[l].detach().cpu().numpy().flatten()
                
                fw_dist.extend(fw_values)
                bw_dist.extend(bw_values)
                
                # Store per-layer weight distributions
                layer_key = f'layer_{l}'
                self.layer_stats[layer_key]['fw_dist'].append({
                    'epoch': epoch,
                    'values': fw_values
                })
                self.layer_stats[layer_key]['bw_dist'].append({
                    'epoch': epoch,
                    'values': bw_values
                })
        
        self.fw_distributions.append(fw_dist)
        self.bw_distributions.append(bw_dist)
        
        # Record weight norms
        fw_norm = sum(w.norm().item() for w in self.model.w if w.numel() > 0)
        bw_norm = sum(w.norm().item() for w in self.model.e_w if w.numel() > 0)
        self.fw_norm_history.append(fw_norm)
        self.bw_norm_history.append(bw_norm)
            
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
            
            axes[idx].hist(before_angles, bins=10, alpha=0.6, color='skyblue', 
                          label=f'Before (μ={np.mean(before_angles):.1f}°)', density=True)
            axes[idx].hist(after_angles, bins=10, alpha=0.6, color='orange',
                          label=f'After (μ={np.mean(after_angles):.1f}°)', density=True)
            
            axes[idx].axvline(np.mean(before_angles), color='blue', linestyle='--', alpha=0.8)
            axes[idx].axvline(np.mean(after_angles), color='red', linestyle='--', alpha=0.8)
            axes[idx].axvline(90, color='black', linestyle='-', alpha=0.3, label='Orthogonal')
            
            axes[idx].set_title(f'{layer.replace("_", " ").title()}')
            axes[idx].set_xlabel('Angle (degrees)')
            axes[idx].set_ylabel('Density')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_xlim(0, 150)
        
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
    
    def plot_alignment_with_std(self, save_path="alignment_with_std.png"):
        """Plot alignment progression with mean and std shaded area"""
        if not self.alignment_history:
            print("No alignment history to plot")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Plot mean line
        plt.plot(self.epoch_history, self.angle_means, 'b-', linewidth=2, label='Mean Angle')
        
        # Plot std area
        lower_bound = [mean - std for mean, std in zip(self.angle_means, self.angle_stds)]
        upper_bound = [mean + std for mean, std in zip(self.angle_means, self.angle_stds)]
        plt.fill_between(self.epoch_history, lower_bound, upper_bound, alpha=0.3, color='blue')
        
        plt.axhline(y=90, color='black', linestyle='--', alpha=0.7, label="Orthogonal (90°)")
        
        plt.title('Weight Alignment During Training (Mean ± Std)', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Mean Angle W^T vs B (degrees)', fontsize=14)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.ylim(0, 180)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_weight_distribution_heatmap(self, save_path="weight_heatmap.png"):
        """Plot heatmap of weight distribution changes over epochs"""
        if not self.fw_distributions or not self.bw_distributions:
            print("No weight distribution data available")
            return
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Process forward weights
        fw_data = []
        for dist in self.fw_distributions:
            hist, edges = np.histogram(dist, bins=100, range=(-1, 1), density=True)
            fw_data.append(hist)
        fw_data = np.array(fw_data).T  # Transpose for correct orientation
        
        # Process backward weights
        bw_data = []
        for dist in self.bw_distributions:
            hist, edges = np.histogram(dist, bins=100, range=(-1, 1), density=True)
            bw_data.append(hist)
        bw_data = np.array(bw_data).T  # Transpose for correct orientation
        
        # Create heatmaps
        im0 = axes[0].imshow(fw_data, aspect='auto', cmap='viridis', 
                          extent=[min(self.epoch_history), max(self.epoch_history)+1, -1, 1])
        im1 = axes[1].imshow(bw_data, aspect='auto', cmap='viridis',
                          extent=[min(self.epoch_history), max(self.epoch_history)+1, -1, 1])
        
        fig.colorbar(im0, ax=axes[0], label='Density')
        fig.colorbar(im1, ax=axes[1], label='Density')
        
        axes[0].set_title('Forward Weight Distribution Over Epochs', fontsize=14)
        axes[1].set_title('Backward Weight Distribution Over Epochs', fontsize=14)
        
        axes[0].set_ylabel('Weight Value', fontsize=12)
        axes[1].set_ylabel('Weight Value', fontsize=12)
        axes[1].set_xlabel('Epoch', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def generate_all_visualizations(self, output_dir="visualizations"):
        """Generate all visualizations in one call"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate plots
        self.plot_comparison_histograms(os.path.join(output_dir, "angle_histograms.png"))
        self.plot_alignment_progression(os.path.join(output_dir, "angle_progression.png"))
        self.plot_alignment_with_std(os.path.join(output_dir, "angle_with_std.png"))
        self.plot_weight_distribution_heatmap(os.path.join(output_dir, "weight_heatmap.png"))
        
        print(f"All visualizations saved to {output_dir}/")
        
class ConvergenceAnalyzer:
    def __init__(self):
        self.training_runs = {}  # Dictionary to store different training runs
    
    def add_run(self, name, epochs, metrics, times=None):
        """
        Add data from a training run
        
        Args:
            name (str): Name of the model/training scheme
            epochs (list): List of epoch numbers
            metrics (dict): Dictionary of metrics (loss, accuracy, etc)
            times (list, optional): Time per epoch
        """
        self.training_runs[name] = {
            'epochs': epochs,
            'metrics': metrics,
            'times': times if times else [1.0] * len(epochs)
        }
    
    def plot_metric_comparison(self, metric='loss', save_path="convergence_comparison.png"):
        """Plot comparison of a specific metric across different runs"""
        if not self.training_runs:
            print("No training runs to compare")
            return
        
        plt.figure(figsize=(12, 6))
        
        for name, data in self.training_runs.items():
            if metric in data['metrics']:
                plt.plot(data['epochs'], data['metrics'][metric], label=name, linewidth=2)
        
        plt.title(f'Convergence Comparison: {metric.capitalize()}', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel(metric.capitalize(), fontsize=14)
        plt.grid(alpha=0.3)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_convergence_time(self, metric='loss', threshold=0.1, save_path="convergence_time.png"):
        """Plot time to reach convergence threshold for different runs"""
        if not self.training_runs:
            print("No training runs to compare")
            return
        
        # Find epochs to reach threshold
        epochs_to_converge = {}
        for name, data in self.training_runs.items():
            if metric in data['metrics']:
                # Find first epoch where metric is below threshold
                for i, value in enumerate(data['metrics'][metric]):
                    if value <= threshold:
                        epochs_to_converge[name] = i
                        break
                else:
                    # If threshold never reached
                    epochs_to_converge[name] = len(data['epochs']) - 1
        
        # Calculate time to convergence
        times_to_converge = {}
        for name, epoch in epochs_to_converge.items():
            # Sum up time for all epochs until convergence
            total_time = sum(self.training_runs[name]['times'][:epoch+1])
            times_to_converge[name] = total_time
        
        # Plot
        plt.figure(figsize=(10, 6))
        names = list(times_to_converge.keys())
        times = list(times_to_converge.values())
        
        bars = plt.bar(names, times)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.1f}s', ha='center', va='bottom')
        
        plt.title(f'Time to Convergence (Threshold: {threshold} {metric})', fontsize=16)
        plt.xlabel('Training Scheme', fontsize=14)
        plt.ylabel('Time (seconds)', fontsize=14)
        plt.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
class WeightTrajectoryAnalyzer:
    def __init__(self, model):
        """
        Initialize weight trajectory analyzer
        
        Args:
            model: Neural network model to analyze
        """
        self.model = model
        self.weight_history = {}  # Stores weights for each epoch
        
    def record_weights(self, epoch):
        """Record weights for current epoch"""
        # Flatten and concatenate forward weights
        fw_weights = [w.detach().cpu().numpy().flatten() for w in self.model.w 
                     if w.numel() > 0]
        fw_flat = np.concatenate(fw_weights) if fw_weights else np.array([])
        
        # Flatten and concatenate backward weights
        bw_weights = [w.detach().cpu().numpy().flatten() for w in self.model.e_w 
                     if w.numel() > 0]
        bw_flat = np.concatenate(bw_weights) if bw_weights else np.array([])
        
        self.weight_history[epoch] = {
            'forward': fw_flat,
            'backward': bw_flat
        }
    
    def compute_trajectory(self, weight_type='forward', method='pca'):
        """
        Compute trajectory in latent space
        
        Args:
            weight_type (str): 'forward' or 'backward'
            method (str): 'pca' or 'tsne'
        
        Returns:
            dict: Mapping from epoch to coordinates in latent space
        """
        if not self.weight_history:
            print("No weight history available")
            return {}
        
        # Extract weights in order of epochs
        epochs = sorted(self.weight_history.keys())
        weights = []
        valid_epochs = []
        
        for e in epochs:
            weight = self.weight_history[e][weight_type]
            if len(weight) > 0:
                weights.append(weight)
                valid_epochs.append(e)
        
        if not weights:
            print(f"No {weight_type} weights available")
            return {}
        
        # Convert to numpy array
        weights = np.array(weights)
        
        # Apply dimensionality reduction
        if method == 'pca':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
        else:  # tsne
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
        
        # Project to 2D
        projected = reducer.fit_transform(weights)
        
        # Map epochs to coordinates
        trajectory = {e: projected[i] for i, e in enumerate(valid_epochs)}
        
        return trajectory
    
    def plot_trajectory(self, save_path="weight_trajectory.png"):
        """Plot weight trajectory in 2D latent space"""
        fw_trajectory = self.compute_trajectory('forward', 'pca')
        bw_trajectory = self.compute_trajectory('backward', 'pca')
        
        if not fw_trajectory and not bw_trajectory:
            print("No trajectory data available")
            return
        
        fig, axes = plt.subplots(1, 2 if fw_trajectory and bw_trajectory else 1, 
                                figsize=(12 if fw_trajectory and bw_trajectory else 6, 6))
        
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        
        # Plot forward weights trajectory
        if fw_trajectory:
            epochs = sorted(fw_trajectory.keys())
            x_coords = [fw_trajectory[e][0] for e in epochs]
            y_coords = [fw_trajectory[e][1] for e in epochs]
            
            # Plot points with color gradient
            scatter = axes[0].scatter(x_coords, y_coords, c=epochs, 
                                     cmap='viridis', s=80, alpha=0.8)
            
            # Connect points with arrows
            for i in range(len(epochs) - 1):
                axes[0].arrow(x_coords[i], y_coords[i], 
                           x_coords[i+1] - x_coords[i], 
                           y_coords[i+1] - y_coords[i],
                           head_width=0.1, head_length=0.2, 
                           fc='gray', ec='gray', alpha=0.5)
            
            # Label points with epoch numbers
            for i, e in enumerate(epochs):
                axes[0].annotate(str(e), (x_coords[i], y_coords[i]), 
                              fontsize=9, ha='right', va='bottom')
            
            axes[0].set_title('Forward Weight Trajectory (PCA)', fontsize=14)
            axes[0].set_xlabel('PC1', fontsize=12)
            axes[0].set_ylabel('PC2', fontsize=12)
            axes[0].grid(alpha=0.3)
            
            fig.colorbar(scatter, ax=axes[0], label='Epoch')
        
        # Plot backward weights trajectory
        if bw_trajectory:
            ax_idx = 1 if fw_trajectory else 0
            epochs = sorted(bw_trajectory.keys())
            x_coords = [bw_trajectory[e][0] for e in epochs]
            y_coords = [bw_trajectory[e][1] for e in epochs]
            
            # Plot points with color gradient
            scatter = axes[ax_idx].scatter(x_coords, y_coords, c=epochs, 
                                         cmap='viridis', s=80, alpha=0.8)
            
            # Connect points with arrows
            for i in range(len(epochs) - 1):
                axes[ax_idx].arrow(x_coords[i], y_coords[i], 
                               x_coords[i+1] - x_coords[i], 
                               y_coords[i+1] - y_coords[i],
                               head_width=0.1, head_length=0.2, 
                               fc='gray', ec='gray', alpha=0.5)
            
            # Label points with epoch numbers
            for i, e in enumerate(epochs):
                axes[ax_idx].annotate(str(e), (x_coords[i], y_coords[i]), 
                                  fontsize=9, ha='right', va='bottom')
            
            axes[ax_idx].set_title('Backward Weight Trajectory (PCA)', fontsize=14)
            axes[ax_idx].set_xlabel('PC1', fontsize=12)
            axes[ax_idx].set_ylabel('PC2', fontsize=12)
            axes[ax_idx].grid(alpha=0.3)
            
            fig.colorbar(scatter, ax=axes[ax_idx], label='Epoch')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()