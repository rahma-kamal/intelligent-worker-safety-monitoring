import os
import argparse
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from data.generator import SyntheticMotionGenerator
from data.dataset import get_dataloaders
from models import BiLSTMHARModel, TCNHARModel, TransformerHARModel

def parse_args():
    parser = argparse.ArgumentParser(description="Train Workplace HAR Models Comparative Framework")
    parser.add_argument("--model", type=str, default="all", choices=["bilstm", "tcn", "transformer", "all"],
                        help="Select sequence model to train or 'all' to train consecutively")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override training batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    return parser.parse_args()

class Trainer:
    """
    Orchestrates the training process, validation scoring, learning rate scheduling,
    and checkpoint saving for comparative evaluation.
    """
    def __init__(self, model_name, config, device, epochs_override=None, batch_override=None, lr_override=None):
        self.model_name = model_name
        self.config = config
        self.device = device
        
        # Hyperparameters
        self.epochs = epochs_override or 80  # Default to 80 epochs for optimized learning
        self.batch_size = batch_override or config['training']['batch_size']
        self.lr = lr_override or 0.0005     # Lower learning rate (5e-4) for highly stable convergence
        self.weight_decay = config['training']['weight_decay']
        
        self.classes = config['dataset']['classes']
        self.num_classes = len(self.classes)
        self.num_features = 92  # 46 coordinates/angles + 46 first-order velocities
        
        # Save pathways
        self.checkpoint_dir = config['training']['save_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(self.checkpoint_dir, f"har_{self.model_name}_best.pt")
        
        # Load Data
        kps_dir = config['dataset']['extracted_keypoints_dir']
        print(f"[*] Initializing DataLoaders from '{kps_dir}'...")
        self.train_loader, self.val_loader = get_dataloaders(kps_dir, batch_size=self.batch_size)
        
        # Compute training class distribution weights for weighted loss (combating dataset imbalance)
        y_train = self.train_loader.dataset.y
        class_counts = np.bincount(y_train, minlength=self.num_classes)
        class_weights = 1.0 / (class_counts + 1e-5)
        # Normalize weights so that average weight is 1.0
        class_weights = class_weights / np.sum(class_weights) * self.num_classes
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        print(f"[i] Calculated Class Imbalance Weights: {class_weights}")
        
        # Instantiate Model
        self.model = self._build_model().to(device)
        total_p, trainable_p = self.model.get_num_parameters()
        print(f"[i] Model '{model_name.upper()}' loaded: {total_p:,} Total Params | {trainable_p:,} Trainable Params.")
        
        # Loss, Optimizer & Scheduler
        self.criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        
    def _build_model(self):
        if self.model_name == "bilstm":
            opts = self.config['models']['bilstm']
            return BiLSTMHARModel(
                input_dim=self.num_features,
                num_classes=self.num_classes,
                hidden_size=opts['hidden_size'],
                num_layers=opts['num_layers'],
                dropout=opts['dropout']
            )
        elif self.model_name == "tcn":
            opts = self.config['models']['tcn']
            return TCNHARModel(
                input_dim=self.num_features,
                num_classes=self.num_classes,
                num_channels=opts['num_channels'],
                kernel_size=opts['kernel_size'],
                dropout=opts['dropout']
            )
        elif self.model_name == "transformer":
            opts = self.config['models']['transformer']
            return TransformerHARModel(
                input_dim=self.num_features,
                num_classes=self.num_classes,
                d_model=opts['d_model'],
                nhead=opts['nhead'],
                num_layers=opts['num_layers'],
                dim_feedforward=opts['dim_feedforward'],
                dropout=opts['dropout']
            )
        else:
            raise ValueError(f"Unknown model architecture: {self.model_name}")

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in self.train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(batch_x)
            loss = self.criterion(logits, batch_y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(logits, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = (correct / total) * 100
        return epoch_loss, epoch_acc

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in self.val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                logits = self.model(batch_x)
                loss = self.criterion(logits, batch_y)
                
                running_loss += loss.item() * batch_x.size(0)
                _, predicted = torch.max(logits, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
        val_loss = running_loss / total
        val_acc = (correct / total) * 100
        return val_loss, val_acc

    def fit(self):
        print(f"\n[*] Starting training loop for '{self.model_name.upper()}' on {self.device} for {self.epochs} epochs...")
        best_val_acc = 0.0
        
        for epoch in range(1, self.epochs + 1):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            self.scheduler.step()
            
            # Print stats
            if epoch % 5 == 0 or epoch == 1 or epoch == self.epochs:
                print(f"Epoch {epoch:03d}/{self.epochs:03d} | "
                      f"Train Loss: {train_loss:.4f} - Acc: {train_acc:.2f}% | "
                      f"Val Loss: {val_loss:.4f} - Acc: {val_acc:.2f}% | "
                      f"LR: {self.scheduler.get_last_lr()[0]:.6f}")
                
            # Checkpoint saving
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.model.save_checkpoint(self.checkpoint_path)
                
        print(f"[SUCCESS] Model '{self.model_name.upper()}' training completed. Best Val Acc: {best_val_acc:.2f}%")
        print(f"    Checkpoint saved to '{self.checkpoint_path}'")
        return best_val_acc

def main():
    args = parse_args()
    
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Dataset availability safeguard
    kps_dir = config['dataset']['extracted_keypoints_dir']
    x_train_path = os.path.join(kps_dir, "X_train.npy")
    
    if not os.path.exists(x_train_path):
        print("[!] Extracted keypoint dataset not found. Initializing automated synthetic motion generation...")
        generator = SyntheticMotionGenerator(config_path="config.yaml")
        generator.build_and_save_dataset()
    else:
        print("[OK] Preprocessed feature coordinates dataset detected.")
        
    # 2. Select model training strategy
    models_to_train = ["bilstm", "tcn", "transformer"] if args.model == "all" else [args.model]
    
    results = {}
    for m in models_to_train:
        trainer = Trainer(
            model_name=m,
            config=config,
            device=device,
            epochs_override=args.epochs,
            batch_override=args.batch_size,
            lr_override=args.lr
        )
        best_acc = trainer.fit()
        results[m] = best_acc
        
    print("\n" + "="*50)
    print(" TRAINING SUMMARY RESULTS ".center(50, "="))
    print("="*50)
    for model_name, best_acc in results.items():
        print(f"[*] {model_name.upper():12s} : Best Validation Accuracy = {best_acc:.2f}%")
    print("="*50)

if __name__ == "__main__":
    main()
