from library.model import NNModel
from library.analysis import CurriculumWeightAnalyzer
from library.dataset import CurriculumDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
from typing import Dict, List

def train_curriculum(
    model: nn.Module,
    dataset: CurriculumDataset,
    num_epochs: int = 100,
    batch_size: int = 32,
    train_split: float = 0.8,
    device: str = 'mps' if torch.backends.mps.is_available() else 'cpu',
    analyzer: CurriculumWeightAnalyzer = None
) -> List[Dict]:
    """
    Train with gradually increasing complexity.
    
    Args:
        model: Neural network model
        dataset: Curriculum dataset
        num_epochs: Total number of epochs
        batch_size: Batch size
        train_split: Fraction of data for training
        device: Device to run on
        analyzer: Weight dynamics analyzer
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Calculate split sizes
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    stats = []
    
    for epoch in range(num_epochs):
        # Update complexity based on epoch
        current_complexity = epoch / (num_epochs - 1)  # Linear scaling from 0 to 1
        dataset.set_complexity(current_complexity)
        
        # Create new train/val split
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
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
            preds = outputs.argmax(dim=1).cpu().numpy()
            target_indices = torch.argmax(targets, dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_targets.extend(target_indices)
            
        # Validation
        val_loss, val_preds, val_targets, test_activations = validate_model(
            model, val_loader, criterion, device
        )
        
        # Record statistics
        epoch_stats = {
            'epoch': epoch,
            'complexity': current_complexity,
            'train_loss': train_loss / len(train_loader),
            'train_acc': accuracy_score(train_targets, train_preds),
            'val_loss': val_loss / len(val_loader),
            'val_acc': accuracy_score(val_targets, val_preds)
        }
        stats.append(epoch_stats)
        
        # Analyze weights if analyzer provided
        if analyzer is not None:
            metrics = analyzer.analyze_epoch(
                model=model,
                epoch=epoch,
                activations=test_activations,
                complexity=current_complexity
            )
            
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Complexity: {current_complexity:.2f}')
            print(f'Train Loss: {epoch_stats["train_loss"]:.4f}, Train Acc: {epoch_stats["train_acc"]:.4f}')
            print(f'Val Loss: {epoch_stats["val_loss"]:.4f}, Val Acc: {epoch_stats["val_acc"]:.4f}')
            
    return stats

def validate_model(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str
) -> tuple:
    """Validate the model and return performance metrics"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_activations = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, hidden = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            target_indices = torch.argmax(targets, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(target_indices)
            all_activations.append(hidden.cpu())
            
    # Concatenate all activations
    all_activations = torch.cat(all_activations, dim=0)
    
    return total_loss, all_preds, all_targets, all_activations

# Main execution
if __name__ == "__main__":
    # Initialize dataset
    dataset = CurriculumDataset(num_samples=1000, save_dir='data/curriculum_dataset')
    
    # Initialize model and analyzer
    model = NNModel(input_dim=3*32*32, hidden_dim=70, output_dim=2)
    analyzer = CurriculumWeightAnalyzer()
    
    # Train with curriculum
    stats = train_curriculum(model, dataset, analyzer=analyzer)
    
    # Save statistics
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv('curriculum_training_stats.csv', index=False)
    
    # Generate visualizations
    analyzer.generate_visualizations()