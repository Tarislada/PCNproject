import torch
from torch.utils.data import DataLoader
from PRECO.utils import onehot
from  randompretrain import generate_random_data

class Trainer:
    def __init__(self, model, config, weight_tracker=None):
        self.model = model
        self.config = config
        self.device = torch.device(config.DEVICE)
        self.weight_tracker = weight_tracker
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(self.device), y.to(self.device)
            self.model.train_supervised(x, y)
            
            if batch_idx % 100 == 0:
                energy = self.model.get_energy()
                print(f"Epoch [{epoch+1}], Step [{batch_idx}/{len(train_loader)}], Loss: {energy:.4f}")
        if self.weight_tracker:
            self.weight_tracker.record_alignment_epoch(epoch)
    
    def test_epoch(self, test_loader):
        """Test for one epoch"""
        total_loss, total_acc, num_batches = 0, 0, 0
        
        for batch_idx, (x, y) in enumerate(test_loader):
            x, y = x.to(self.device), y.to(self.device)
            output = self.model.test_supervised(x)
            
            loss = torch.nn.MSELoss()(output, onehot(y, N=10)).item()
            accuracy = torch.mean((torch.argmax(output, axis=1) == y).float()).item()
            
            total_loss += loss
            total_acc += accuracy
            num_batches += 1
            
            if batch_idx % 100 == 0:
                print(f"Test Step [{batch_idx}/{len(test_loader)}], Loss: {loss:.4f}, Acc: {accuracy:.4f}")
        
        return total_loss / num_batches, total_acc / num_batches
    def pretrain_epoch(self, pretrain_config, epoch):
        """
        Run one epoch of pretraining with random data
        
        Args:
            pretrain_config: Configuration for pretraining
            epoch: Current pretraining epoch
        """
        # Generate random data for this epoch
        pretrain_loader = generate_random_data(
            pretrain_config['batch_size'], 
            pretrain_config['input_size'], 
            pretrain_config['output_size'], 
            pretrain_config['num_batches'],
            pretrain_config['input_distribution'],
            pretrain_config.get('distribution_params', {})
        )
        
        epoch_energy = 0
        batch_count = 0
        
        for batch_idx, (x, y) in enumerate(pretrain_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            
            # Reshape input to match expected format if needed
            if len(x.shape) == 2 and pretrain_config['input_size'] == 784:
                x = x.view(x.shape[0], 1, 28, 28)
            
            # Train the model
            self.model.train_supervised(x, y)
            
            current_energy = self.model.get_energy()
            epoch_energy += current_energy
            batch_count += 1
            
            if batch_idx % 20 == 0:
                print(f"Pretrain Epoch [{epoch+1}/{pretrain_config['epochs']}], "
                      f"Batch [{batch_idx}/{len(pretrain_loader)}], "
                      f"Energy: {current_energy:.4f}")
        
        # Record alignment after each pretraining epoch
        if self.weight_tracker:
            self.weight_tracker.record_alignment_epoch(epoch)
        
        avg_energy = epoch_energy / batch_count if batch_count > 0 else 0
        return avg_energy
    
    def pretrain_model(self, pretrain_config):
        """
        Run full pretraining with weight alignment tracking
        
        Args:
            pretrain_config: Dictionary with pretraining parameters
        """
        print(f"Starting pretraining for {pretrain_config['epochs']} epochs...")
        print(f"Using {pretrain_config['input_distribution']} distribution for inputs")
        
        with torch.no_grad():
            for epoch in range(pretrain_config['epochs']):
                avg_energy = self.pretrain_epoch(pretrain_config, epoch)
                print(f"Pretrain Epoch [{epoch+1}/{pretrain_config['epochs']}] completed. Average Energy: {avg_energy:.4f}")
                
                # Monitor weight norms during pretraining
                if epoch % 2 == 0:
                    fw_norm = sum(w.norm().item() for w in self.model.w if w.numel() > 0)
                    bw_norm = sum(w.norm().item() for w in self.model.e_w if w.numel() > 0)
                    print(f"Pretrain - Forward weight norm: {fw_norm:.4f}, Backward weight norm: {bw_norm:.4f}")
                    
                    # Print alignment stats if tracker available
                    if self.weight_tracker:
                        self.weight_tracker.print_alignment_stats(epoch)
        
        print("Pretraining completed!")

    def print_weight_stats(self, epoch):
        """Print weight statistics"""
        if epoch % 5 == 0:
            for l in range(len(self.model.w)):
                if self.model.dw[l] is not None and self.model.de_w[l] is not None:
                    print(f"Layer {l} - fw_grad_norm: {self.model.dw[l].norm().item():.4f}, "
                          f"bw_grad_norm: {self.model.de_w[l].norm().item():.4f}")
            
            fw_norm = sum(w.norm().item() for w in self.model.w if w.numel() > 0)
            bw_norm = sum(w.norm().item() for w in self.model.e_w if w.numel() > 0)
            print(f"Forward weight norm: {fw_norm:.4f}, Backward weight norm: {bw_norm:.4f}")
            # Weight alignment stats if tracker available
            if self.weight_tracker:
                self.weight_tracker.print_alignment_stats(epoch)