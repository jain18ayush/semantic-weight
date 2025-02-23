from library.dataset import MonoToPolySemanticsDataset
from library.model import NNModel
from library.analysis import WeightDynamicsAnalyzer

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from typing import Dict, List
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score
from typing import Dict, List
import os

# Set random seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    phase: int,
    epochs: int = 50,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    analyzer: WeightDynamicsAnalyzer = WeightDynamicsAnalyzer()
) -> List[Dict]:
    """
    Basic training loop with validation.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        phase: Training phase (1 or 2)
        epochs: Number of training epochs
        device: Device to run training on
    
    Returns:
        List of dictionaries containing training statistics per epoch
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    stats = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            # Convert to class indices for consistent format
            preds = outputs.argmax(dim=1).cpu().numpy()
            target_indices = torch.argmax(targets, dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_targets.extend(target_indices)
        
        # Validation
        val_loss, val_preds, val_targets, test_activations = test_model(model, val_loader, criterion, device)
        
        # Record statistics
        epoch_stats = {
            'phase': phase,
            'epoch': epoch + 1,
            'train_loss': train_loss / len(train_loader),
            'train_acc': accuracy_score(train_targets, train_preds),
            'val_loss': val_loss / len(val_loader),
            'val_acc': accuracy_score(val_targets, val_preds)
        }
        stats.append(epoch_stats)
        
        with torch.no_grad():
            metrics = analyzer.analyze_epoch(
              model=model,
              epoch=epoch,
              activations=test_activations,
              phase='phase1'  # or 'phase2'
            )

        if (epoch + 1) % 10 == 0:
            print(f'Phase {phase}, Epoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {epoch_stats["train_loss"]:.4f}, Train Acc: {epoch_stats["train_acc"]:.4f}')
            print(f'Val Loss: {epoch_stats["val_loss"]:.4f}, Val Acc: {epoch_stats["val_acc"]:.4f}')
    
    return stats

def test_model(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: str
) -> tuple:
    """
    Test the model and return loss and predictions.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, hidden = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            # Convert to class indices for consistent format
            preds = outputs.argmax(dim=1).cpu().numpy()
            target_indices = torch.argmax(targets, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(target_indices)
    
    return total_loss, all_preds, all_targets, hidden

# Load the dataset

print("Starting experiment")

train_split = 0.8
batch_size = 32

dataset = MonoToPolySemanticsDataset(num_samples=1000, phase=1)

# Split dataset into train and validation
train_size = int(train_split * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

fig = dataset.visualize_samples(5)

analyzer = WeightDynamicsAnalyzer()
model = NNModel(input_dim=3*32*32, hidden_dim=20, output_dim=2)
phase1_stats = train_model(model, train_loader, val_loader, phase=1, analyzer=analyzer)

stats_df = pd.DataFrame(phase1_stats)
stats_df.to_csv('phase1_stats.csv', index=False)

# Switch to phase 2
dataset.switch_to_phase(2)

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

fig = dataset.visualize_samples(5)

analyzer.current_phase = 2
#keep training the same model 
phase2_stats = train_model(model, train_loader, val_loader, phase=2, analyzer=analyzer)
stats_df = pd.DataFrame(phase2_stats)
stats_df.to_csv('phase2_stats.csv', index=False)

analyzer.generate_visualizations()