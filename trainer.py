import torch
from torch.utils.data import DataLoader
from PRECO.utils import onehot

class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = torch.device(config.DEVICE)
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(self.device), y.to(self.device)
            self.model.train_supervised(x, y)
            
            if batch_idx % 100 == 0:
                energy = self.model.get_energy()
                print(f"Epoch [{epoch+1}], Step [{batch_idx}/{len(train_loader)}], Loss: {energy:.4f}")
    
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