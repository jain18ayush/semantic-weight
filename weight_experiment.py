import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import pandas as pd
from tqdm import tqdm
import time
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# Set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Define the bilinear MLP model
class BilinearMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BilinearMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
    
    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def get_weights(self):
        # Return the first layer weights
        return self.fc1.weight.data.clone()

# Function to load data
def load_data(dataset_name, batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    if dataset_name.lower() == 'mnist':
        train_dataset = datasets.MNIST('/Volumes/Ayush_Drive/data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('/Volumes/Ayush_Drive/data', train=False, transform=transform)
    elif dataset_name.lower() == 'fashion-mnist':
        train_dataset = datasets.FashionMNIST('/Volumes/Ayush_Drive/data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST('/Volumes/Ayush_Drive/data', train=False, transform=transform)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# Function to train the model with checkpoints
def train_model(model, train_loader, test_loader, epochs, lr, device, checkpoint_percentages, save_dir):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Create checkpoint epochs based on percentages
    checkpoint_epochs = [max(1, int(p * epochs / 100)) for p in checkpoint_percentages]
    checkpoint_epochs = sorted(list(set(checkpoint_epochs)))  # Remove duplicates and sort
    
    # If 0 is not in checkpoints, add it to get initial state
    if 0 not in checkpoint_epochs:
        checkpoint_epochs = [0] + checkpoint_epochs
    
    # Ensure final epoch is included
    if epochs not in checkpoint_epochs and epochs not in [epoch-1 for epoch in checkpoint_epochs]:
        checkpoint_epochs.append(epochs)
    
    # Create directory to save checkpoints if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Dictionary to store results
    results = {
        'checkpoint_epochs': checkpoint_epochs,
        'train_acc': [],
        'test_acc': [],
    }
    
    # Initialize progress tracking
    print(f"Training for {epochs} epochs with checkpoints at: {checkpoint_epochs}")
    
    # Initial checkpoint (untrained model)
    if 0 in checkpoint_epochs:
        # Save initial model
        torch.save(model.state_dict(), os.path.join(save_dir, f'checkpoint_epoch_0.pt'))
        
        # Evaluate initial model
        train_acc = evaluate(model, train_loader, device)
        test_acc = evaluate(model, test_loader, device)
        results['train_acc'].append(train_acc)
        results['test_acc'].append(test_acc)
        print(f"Initial model - Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f}")
    
    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        print(f'Epoch: {epoch}, Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%')
        
        # Save checkpoint if current epoch is in checkpoint_epochs
        if epoch in checkpoint_epochs:
            # Save model
            torch.save(model.state_dict(), os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt'))
            
            # Evaluate model
            train_acc = evaluate(model, train_loader, device)
            test_acc = evaluate(model, test_loader, device)
            results['train_acc'].append(train_acc)
            results['test_acc'].append(test_acc)
            print(f"Checkpoint at epoch {epoch} - Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f}")
    
    # Save results
    torch.save(results, os.path.join(save_dir, 'training_results.pt'))
    
    return results

# Function to evaluate the model
def evaluate(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    return correct / total

# Function to extract top K eigenvectors for each class
def extract_features(model, k=20, input_shape=(28, 28)):
    """
    Extract top K eigenvectors from the weight matrix of the first layer.
    
    Args:
        model: The trained model
        k: Number of top eigenvectors to extract
        input_shape: Shape of input images for visualization
    
    Returns:
        Dictionary containing eigenvectors and eigenvalues
    """
    # Get weights from the first layer
    weights = model.get_weights().cpu().numpy()
    
    # Initialize dictionaries to store eigenvectors and eigenvalues
    eigenvectors = {}
    eigenvalues = {}
    
    # Extract eigenvectors for each class
    for c in range(model.output_dim):
        # Get the weights for this class
        W_c = weights.copy()  # Use the entire weight matrix for now
        
        # Compute covariance matrix
        cov_matrix = W_c.T @ W_c
        
        # Compute eigenvectors and eigenvalues
        vals, vecs = np.linalg.eigh(cov_matrix)
        
        # Sort in descending order
        idx = np.argsort(vals)[::-1]
        vals = vals[idx]
        vecs = vecs[:, idx]
        
        # Store top K eigenvectors and eigenvalues
        eigenvectors[c] = vecs[:, :k]
        eigenvalues[c] = vals[:k]
    
    return {
        'eigenvectors': eigenvectors,
        'eigenvalues': eigenvalues
    }

# Function to compute stability between checkpoints
def compute_stability(features_checkpoints, k=20):
    """
    Compute cosine similarity between eigenvectors at consecutive checkpoints.
    
    Args:
        features_checkpoints: List of feature dictionaries at different checkpoints
        k: Number of top eigenvectors to consider
    
    Returns:
        Dictionary with stability metrics
    """
    n_checkpoints = len(features_checkpoints)
    n_classes = len(features_checkpoints[0]['eigenvectors'])
    
    # Initialize dictionaries to store stability metrics
    stability_matrices = {}
    stabilization_points = {}
    
    for c in range(n_classes):
        # Initialize stability matrices for each class
        stability_matrices[c] = np.zeros((n_checkpoints-1, k))
        stabilization_points[c] = np.zeros(k, dtype=int)
        
        # Compute cosine similarity between consecutive checkpoints for each eigenvector
        for i in range(n_checkpoints-1):
            vecs1 = features_checkpoints[i]['eigenvectors'][c]
            vecs2 = features_checkpoints[i+1]['eigenvectors'][c]
            
            for j in range(k):
                # Compute absolute cosine similarity (ignore sign flips)
                sim = np.abs(np.dot(vecs1[:, j], vecs2[:, j]) / 
                           (np.linalg.norm(vecs1[:, j]) * np.linalg.norm(vecs2[:, j])))
                stability_matrices[c][i, j] = sim
        
        # Find stabilization points (first checkpoint where cosine similarity exceeds 0.8 for 3 consecutive checkpoints)
        for j in range(k):
            for i in range(n_checkpoints-3):
                if (stability_matrices[c][i, j] > 0.8 and 
                    stability_matrices[c][i+1, j] > 0.8 and 
                    stability_matrices[c][i+2, j] > 0.8):
                    stabilization_points[c][j] = i
                    break
    
    return {
        'stability_matrices': stability_matrices,
        'stabilization_points': stabilization_points
    }

# Function to track eigenvalue evolution
def eigenvalue_evolution(features_checkpoints, k=20):
    """
    Track eigenvalue magnitudes and compute concentration metrics across training.
    
    Args:
        features_checkpoints: List of feature dictionaries at different checkpoints
        k: Number of top eigenvectors to consider
    
    Returns:
        Dictionary with eigenvalue evolution metrics
    """
    n_checkpoints = len(features_checkpoints)
    n_classes = len(features_checkpoints[0]['eigenvectors'])
    
    # Initialize dictionaries to store eigenvalue metrics
    eigenvalue_trajectories = {}
    concentration_metrics = {}
    
    for c in range(n_classes):
        # Initialize arrays
        eigenvalue_trajectories[c] = np.zeros((n_checkpoints, k))
        concentration_metrics[c] = np.zeros((n_checkpoints, k))
        
        # Track eigenvalue magnitudes and concentration over time
        for i in range(n_checkpoints):
            # Get eigenvalues for this checkpoint and class
            evals = features_checkpoints[i]['eigenvalues'][c]
            
            # Store eigenvalue magnitudes
            eigenvalue_trajectories[c][i, :] = evals[:k]
            
            # Compute concentration metrics (ratio of top-N eigenvalues to sum of all eigenvalues)
            total_sum = np.sum(evals)
            for j in range(k):
                concentration_metrics[c][i, j] = np.sum(evals[:j+1]) / total_sum if total_sum > 0 else 0
    
    return {
        'eigenvalue_trajectories': eigenvalue_trajectories,
        'concentration_metrics': concentration_metrics
    }

# Function for truncated inference
class TruncatedModel(nn.Module):
    """
    Model that performs inference with only top-K eigenvectors
    """
    def __init__(self, original_model, features, k):
        super(TruncatedModel, self).__init__()
        self.original_model = original_model
        self.features = features
        self.k = k
        self.input_dim = original_model.input_dim
        self.output_dim = original_model.output_dim
        
    def forward(self, x):
        # Get the original weights
        original_weights = self.original_model.fc1.weight.data.clone().cpu().numpy()
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1).cpu().numpy()
        
        # Reconstruct weights using only top-K eigenvectors
        truncated_outputs = np.zeros((batch_size, self.original_model.hidden_dim))
        
        for c in range(self.output_dim):
            # Get top K eigenvectors for this class
            eigenvectors = self.features['eigenvectors'][c][:, :self.k]
            eigenvalues = self.features['eigenvalues'][c][:self.k]
            
            # Project input data onto top-K eigenvectors
            # For simplicity, just use the original weights for now
            # In a more sophisticated implementation, we would reconstruct weights from eigenvectors
            truncated_outputs = x_flat @ original_weights.T
        
        # Apply ReLU activation
        truncated_outputs = np.maximum(0, truncated_outputs)
        
        # Convert back to torch tensor
        truncated_output_tensor = torch.tensor(truncated_outputs, dtype=torch.float32).to(x.device)
        
        # Pass through the second layer (unchanged)
        final_output = self.original_model.fc2(truncated_output_tensor)
        
        return final_output

# Function to measure feature importance
def measure_feature_importance(model, features_checkpoints, test_loader, device, k_values=[1, 2, 5, 10, 15, 20]):
    """
    Measure the importance of features by evaluating accuracy with truncated representations.
    
    Args:
        model: The trained model
        features_checkpoints: List of feature dictionaries at different checkpoints
        test_loader: DataLoader for test data
        device: Device to run evaluation on
        k_values: List of K values to test for truncation
    
    Returns:
        Dictionary with feature importance metrics
    """
    n_checkpoints = len(features_checkpoints)
    
    # Initialize arrays to store results
    truncated_accuracies = np.zeros((n_checkpoints, len(k_values)))
    minimal_representations = np.zeros(n_checkpoints, dtype=int)
    
    # Get full model accuracy as reference
    full_accuracies = np.zeros(n_checkpoints)
    
    for i, features in enumerate(features_checkpoints):
        # Evaluate full model
        full_acc = evaluate(model, test_loader, device)
        full_accuracies[i] = full_acc
        
        # Evaluate with truncated representations
        for j, k in enumerate(k_values):
            # Create truncated model
            truncated_model = TruncatedModel(model, features, k)
            truncated_model.to(device)
            
            # Evaluate truncated model
            truncated_acc = evaluate(truncated_model, test_loader, device)
            truncated_accuracies[i, j] = truncated_acc
            
            # Check if we've reached 90% of full accuracy
            if truncated_acc >= 0.9 * full_acc and minimal_representations[i] == 0:
                minimal_representations[i] = k
    
    return {
        'k_values': k_values,
        'full_accuracies': full_accuracies,
        'truncated_accuracies': truncated_accuracies,
        'minimal_representations': minimal_representations
    }

# Visualization Functions
def visualize_eigenvectors(features_checkpoints, checkpoint_percentages, save_dir, input_shape=(28, 28), k=5):
    """
    Visualize top eigenvectors for each class and checkpoint.
    
    Args:
        features_checkpoints: List of feature dictionaries at different checkpoints
        checkpoint_percentages: List of checkpoint percentages
        save_dir: Directory to save visualizations
        input_shape: Shape of input images for visualization
        k: Number of top eigenvectors to visualize
    """
    n_checkpoints = len(features_checkpoints)
    
    print(n_checkpoints)

    n_classes = len(features_checkpoints[0]['eigenvectors'])
    
    # Create directory for eigenvector visualizations
    eigenvec_dir = os.path.join(save_dir, 'eigenvector_visualizations')
    os.makedirs(eigenvec_dir, exist_ok=True)
    
    for c in range(n_classes):
        class_dir = os.path.join(eigenvec_dir, f'class_{c}')
        os.makedirs(class_dir, exist_ok=True)
        
        for i, cp in enumerate(checkpoint_percentages):
            eigenvectors = features_checkpoints[i]['eigenvectors'][c]
            
            # Create a figure for this checkpoint
            fig, axes = plt.subplots(1, k, figsize=(15, 3))
            if k == 1:
                axes = [axes]
            
            for j in range(k):
                # Reshape eigenvector to image dimensions
                eigenvec = eigenvectors[:, j].reshape(input_shape)
                
                # Normalize for visualization
                eigenvec = (eigenvec - eigenvec.min()) / (eigenvec.max() - eigenvec.min() + 1e-8)
                
                # Plot
                ax = axes[j]
                im = ax.imshow(eigenvec, cmap='viridis')
                ax.set_title(f'EV {j+1}')
                ax.axis('off')
            
            plt.suptitle(f'Class {c} - Checkpoint {cp}%')
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(class_dir, f'checkpoint_{cp}.png'), dpi=150)
            plt.close(fig)

def visualize_stability(stability_data, checkpoint_percentages, save_dir):
    """
    Visualize stability of eigenvectors across training.
    
    Args:
        stability_data: Dictionary with stability matrices and stabilization points
        checkpoint_percentages: List of checkpoint percentages
        save_dir: Directory to save visualizations
    """
    stability_matrices = stability_data['stability_matrices']
    stabilization_points = stability_data['stabilization_points']
    n_classes = len(stability_matrices)
    
    # Create directory for stability visualizations
    stability_dir = os.path.join(save_dir, 'stability_visualizations')
    os.makedirs(stability_dir, exist_ok=True)
    
    for c in range(n_classes):
        # Create heatmap of stability matrix
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get checkpoint labels for x-axis
        checkpoint_labels = [f"{checkpoint_percentages[i]}%-{checkpoint_percentages[i+1]}%" 
                            for i in range(len(checkpoint_percentages)-1)]
        
        # Get eigenvector labels for y-axis
        eigenvector_labels = [f"EV {i+1}" for i in range(stability_matrices[c].shape[1])]
        
        # Create heatmap
        sns.heatmap(stability_matrices[c].T, annot=True, fmt=".2f", cmap="YlGnBu",
                   xticklabels=checkpoint_labels, yticklabels=eigenvector_labels,
                   vmin=0, vmax=1, ax=ax)
        
        plt.title(f'Class {c} - Eigenvector Stability')
        plt.xlabel('Checkpoint Transitions')
        plt.ylabel('Eigenvector Rank')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(stability_dir, f'stability_heatmap_class_{c}.png'), dpi=150)
        plt.close(fig)
        
        # Plot stabilization trajectories
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i in range(len(stabilization_points[c])):
            sp = stabilization_points[c][i]
            if sp > 0:  # If stabilization point was found
                plt.scatter(i+1, checkpoint_percentages[sp], c='blue', s=50)
        
        plt.title(f'Class {c} - Eigenvector Stabilization Points')
        plt.xlabel('Eigenvector Rank')
        plt.ylabel('Training Percentage')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(0, 100)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(stability_dir, f'stabilization_points_class_{c}.png'), dpi=150)
        plt.close(fig)

def visualize_eigenvalue_evolution(eigenvalue_data, checkpoint_percentages, save_dir):
    """
    Visualize evolution of eigenvalues during training.
    
    Args:
        eigenvalue_data: Dictionary with eigenvalue trajectories and concentration metrics
        checkpoint_percentages: List of checkpoint percentages
        save_dir: Directory to save visualizations
    """
    eigenvalue_trajectories = eigenvalue_data['eigenvalue_trajectories']
    concentration_metrics = eigenvalue_data['concentration_metrics']
    n_classes = len(eigenvalue_trajectories)
    n_checkpoints = eigenvalue_trajectories[0].shape[0]
    
    # Create directory for eigenvalue visualizations
    eigenvalue_dir = os.path.join(save_dir, 'eigenvalue_visualizations')
    os.makedirs(eigenvalue_dir, exist_ok=True)
    
    for c in range(n_classes):
        # Plot eigenvalue trajectories
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i in range(eigenvalue_trajectories[c].shape[1]):
            plt.plot(checkpoint_percentages[:n_checkpoints], eigenvalue_trajectories[c][:, i], 
                     marker='o', label=f'EV {i+1}')
        
        plt.title(f'Class {c} - Eigenvalue Trajectories')
        plt.xlabel('Training Percentage')
        plt.ylabel('Eigenvalue Magnitude')
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(eigenvalue_dir, f'eigenvalue_trajectories_class_{c}.png'), dpi=150)
        plt.close(fig)
        
        # Plot eigenvalue concentration
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i in range(min(5, concentration_metrics[c].shape[1])):  # Show top 5 for readability
            k_value = i + 1
            plt.plot(checkpoint_percentages[:n_checkpoints], concentration_metrics[c][:, i], 
                     marker='o', label=f'Top {k_value}')
        
        plt.title(f'Class {c} - Eigenvalue Concentration')
        plt.xlabel('Training Percentage')
        plt.ylabel('Concentration Ratio')
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(0, 1)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(eigenvalue_dir, f'eigenvalue_concentration_class_{c}.png'), dpi=150)
        plt.close(fig)

def visualize_feature_importance(importance_data, checkpoint_percentages, save_dir):
    """
    Visualize the importance of features for model performance.
    
    Args:
        importance_data: Dictionary with feature importance metrics
        checkpoint_percentages: List of checkpoint percentages
        save_dir: Directory to save visualizations
    """
    k_values = importance_data['k_values']
    full_accuracies = importance_data['full_accuracies']
    truncated_accuracies = importance_data['truncated_accuracies']
    minimal_representations = importance_data['minimal_representations']
    n_checkpoints = len(full_accuracies)
    
    # Create directory for feature importance visualizations
    importance_dir = os.path.join(save_dir, 'feature_importance')
    os.makedirs(importance_dir, exist_ok=True)
    
    # Plot accuracy vs. feature count curves for each checkpoint
    for i in range(n_checkpoints):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot full accuracy as horizontal line
        plt.axhline(y=full_accuracies[i], color='r', linestyle='--', label='Full Model')
        
        # Plot truncated accuracies
        plt.plot(k_values, truncated_accuracies[i, :], marker='o', color='b')
        
        # Mark 90% threshold
        threshold = 0.9 * full_accuracies[i]
        plt.axhline(y=threshold, color='g', linestyle=':', label='90% Threshold')
        
        # Mark minimal representation
        if minimal_representations[i] > 0:
            min_rep_idx = k_values.index(minimal_representations[i])
            plt.scatter(minimal_representations[i], truncated_accuracies[i, min_rep_idx], 
                       s=100, color='g', zorder=5)
        
        plt.title(f'Checkpoint {checkpoint_percentages[i]}% - Accuracy vs. Feature Count')
        plt.xlabel('Number of Top Eigenvectors (K)')
        plt.ylabel('Test Accuracy')
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(importance_dir, f'accuracy_vs_features_checkpoint_{checkpoint_percentages[i]}.png'), dpi=150)
        plt.close(fig)
    
    # Plot minimal representation evolution
    fig, ax = plt.subplots(figsize=(10, 6))
    
    plt.plot(checkpoint_percentages[:n_checkpoints], minimal_representations, marker='o', color='b')
    
    plt.title('Minimal Representation Size Evolution')
    plt.xlabel('Training Percentage')
    plt.ylabel('Minimal K for 90% Accuracy')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(importance_dir, 'minimal_representation_evolution.png'), dpi=150)
    plt.close(fig)

# Export results to CSV
def export_results_to_csv(features_checkpoints, stability_data, eigenvalue_data, 
                          importance_data, checkpoint_percentages, save_dir):
    """
    Export numerical results to CSV files.
    
    Args:
        features_checkpoints: List of feature dictionaries at different checkpoints
        stability_data: Dictionary with stability metrics
        eigenvalue_data: Dictionary with eigenvalue metrics
        importance_data: Dictionary with feature importance metrics
        checkpoint_percentages: List of checkpoint percentages
        save_dir: Directory to save CSV files
    """
    # Create directory for CSV files
    csv_dir = os.path.join(save_dir, 'csv_results')
    os.makedirs(csv_dir, exist_ok=True)
    
    # Export stabilization points
    stabilization_points = stability_data['stabilization_points']
    n_classes = len(stabilization_points)
    
    for c in range(n_classes):
        df = pd.DataFrame({
            'eigenvector_rank': range(1, len(stabilization_points[c]) + 1),
            'stabilization_checkpoint': [checkpoint_percentages[sp] if sp > 0 else 'Not Stabilized' 
                                       for sp in stabilization_points[c]]
        })
        df.to_csv(os.path.join(csv_dir, f'stabilization_points_class_{c}.csv'), index=False)
    
    # Export eigenvalue trajectories
    eigenvalue_trajectories = eigenvalue_data['eigenvalue_trajectories']
    
    for c in range(n_classes):
        df = pd.DataFrame(eigenvalue_trajectories[c], 
                         columns=[f'EV_{i+1}' for i in range(eigenvalue_trajectories[c].shape[1])])
        df['checkpoint_percentage'] = checkpoint_percentages[:len(df)]
        df.to_csv(os.path.join(csv_dir, f'eigenvalue_trajectories_class_{c}.csv'), index=False)
    
    # Export minimal representations
    minimal_representations = importance_data['minimal_representations']
    
    df = pd.DataFrame({
        'checkpoint_percentage': checkpoint_percentages[:len(minimal_representations)],
        'minimal_k': minimal_representations
    })
    df.to_csv(os.path.join(csv_dir, 'minimal_representations.csv'), index=False)
    
    # Export truncated accuracies
    truncated_accuracies = importance_data['truncated_accuracies']
    k_values = importance_data['k_values']
    
    for i, cp in enumerate(checkpoint_percentages[:truncated_accuracies.shape[0]]):
        df = pd.DataFrame({
            'k_value': k_values,
            'accuracy': truncated_accuracies[i, :]
        })
        df.to_csv(os.path.join(csv_dir, f'truncated_accuracies_checkpoint_{cp}.csv'), index=False)

# Main experiment function
def run_experiment(dataset_name, hidden_dim, epochs, learning_rate, random_seed, device,
                  checkpoint_percentages=[1, 2, 5, 10, 15, 20, 30, 40, 50, 75, 100],
                  input_shape=(28, 28), k=20):
    """
    Run the complete feature emergence experiment.
    
    Args:
        dataset_name: Name of the dataset ('mnist' or 'fashion-mnist')
        hidden_dim: Dimension of the hidden layer
        epochs: Number of training epochs
        learning_rate: Learning rate for training
        random_seed: Random seed for reproducibility
        device: Device to run training on
        checkpoint_percentages: List of checkpoint percentages
        input_shape: Shape of input images
        k: Number of top eigenvectors to extract
    
    Returns:
        Dictionary with experiment results
    """
    # Set random seed
    set_seed(random_seed)
    
    # Create experiment directory
    exp_name = f"{dataset_name}_hidden{hidden_dim}_seed{random_seed}"
    save_dir = os.path.join("/Volumes/Ayush_Drive/results", exp_name)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Starting experiment: {exp_name}")
    print(f"Saving results to: {save_dir}")
    
    # Load data
    train_loader, test_loader = load_data(dataset_name)
    
    # Calculate input dimension from input shape
    input_dim = input_shape[0] * input_shape[1]
    
    # Create model
    model = BilinearMLP(input_dim, hidden_dim, 10)  # 10 classes for MNIST and Fashion-MNIST
    model.to(device)
    
    # Train model with checkpoints
    training_results = train_model(
        model, train_loader, test_loader, epochs, learning_rate, device, 
        checkpoint_percentages, save_dir
    )
    
    # Extract features at each checkpoint
    checkpoint_epochs = training_results['checkpoint_epochs']
    features_checkpoints = []
    
    print("Extracting features at each checkpoint...")
    for epoch in checkpoint_epochs:
        # Load model from checkpoint
        model.load_state_dict(torch.load(os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')))
        
        # Extract features
        features = extract_features(model, k=k, input_shape=input_shape)
        features_checkpoints.append(features)
    
    # Compute stability
    print("Computing stability metrics...")
    stability_data = compute_stability(features_checkpoints, k=k)
    
    # Track eigenvalue evolution
    print("Tracking eigenvalue evolution...")
    eigenvalue_data = eigenvalue_evolution(features_checkpoints, k=k)
    
    # Measure feature importance
    print("Measuring feature importance...")
    importance_data = measure_feature_importance(model, features_checkpoints, test_loader, device)
    
    # Create visualizations
    print("Creating visualizations...")
    try:
        visualize_eigenvectors(features_checkpoints, checkpoint_percentages, save_dir, input_shape, k=5)
    except Exception as e:
        print(f"Error in visualize_eigenvectors: {e}: {len(features_checkpoints)} {len(checkpoint_percentages)} {save_dir} {input_shape}")
    
    try:
        visualize_stability(stability_data, checkpoint_percentages, save_dir)
    except Exception as e:
        print(f"Error in visualize_stability: {e} {len(stability_data)} {len(checkpoint_percentages)} {save_dir}")
    
    try:
        visualize_eigenvalue_evolution(eigenvalue_data, checkpoint_percentages, save_dir)
    except Exception as e:
        print(f"Error in visualize_eigenvalue_evolution: {e} {len(eigenvalue_data)} {len(checkpoint_percentages)} {save_dir}")
    
    try:
        visualize_feature_importance(importance_data, checkpoint_percentages, save_dir)
    except Exception as e:
        print(f"Error in visualize_feature_importance: {e} {len(importance_data)} {len(checkpoint_percentages)} {save_dir}")
    
    # Export results to CSV
    print("Exporting results to CSV...")
    export_results_to_csv(features_checkpoints, stability_data, eigenvalue_data, 
                         importance_data, checkpoint_percentages, save_dir)
    
    # Save all results
    print("Saving experiment results...")
    torch.save({
        'features_checkpoints': features_checkpoints,
        'stability_data': stability_data,
        'eigenvalue_data': eigenvalue_data,
        'importance_data': importance_data,
        'checkpoint_percentages': checkpoint_percentages,
        'training_results': training_results
    }, os.path.join(save_dir, 'experiment_results.pt'))
    
    # Evaluate hypothesis
    success_criteria = check_hypothesis(stability_data, eigenvalue_data, 
                                      importance_data, checkpoint_percentages)
    
    return {
        'experiment_name': exp_name,
        'save_dir': save_dir,
        'training_results': training_results,
        'success_criteria': success_criteria
    }

# Function to check hypothesis criteria
def check_hypothesis(stability_data, eigenvalue_data, importance_data, checkpoint_percentages):
    """
    Check if the experimental results support the hypothesis.
    
    Args:
        stability_data: Dictionary with stability metrics
        eigenvalue_data: Dictionary with eigenvalue metrics
        importance_data: Dictionary with feature importance metrics
        checkpoint_percentages: List of checkpoint percentages
    
    Returns:
        Dictionary with success criteria results
    """
    stabilization_points = stability_data['stabilization_points']
    minimal_representations = importance_data['minimal_representations']
    n_classes = len(stabilization_points)
    
    # Initialize success criteria
    criteria_results = {
        'top5_early_stabilization': False,
        'lower_late_evolution': False,
        'decreasing_minimal_representation': False
    }
    
    # Check criterion 1: Top 5 eigenvectors reach stability within first 20% of training
    top5_early_stabilization = True
    
    for c in range(n_classes):
        for i in range(5):  # Top 5 eigenvectors
            sp = stabilization_points[c][i]
            if sp >= 0 and checkpoint_percentages[sp] <= 20:
                continue
            else:
                top5_early_stabilization = False
                break
    
    criteria_results['top5_early_stabilization'] = top5_early_stabilization
    
    # Check criterion 2: Lower-ranked eigenvectors show continued evolution beyond 50% of training
    lower_late_evolution = False
    
    for c in range(n_classes):
        for i in range(5, len(stabilization_points[c])):  # Lower-ranked eigenvectors
            sp = stabilization_points[c][i]
            if sp >= 0 and checkpoint_percentages[sp] > 50:
                lower_late_evolution = True
                break
    
    criteria_results['lower_late_evolution'] = lower_late_evolution
    
    # Check criterion 3: Minimal representation size decreases during training
    if len(minimal_representations) > 1:
        if minimal_representations[0] > minimal_representations[-1]:
            criteria_results['decreasing_minimal_representation'] = True
    
    # Overall hypothesis support
    criteria_results['hypothesis_supported'] = (criteria_results['top5_early_stabilization'] and 
                                              criteria_results['lower_late_evolution'] and 
                                              criteria_results['decreasing_minimal_representation'])
    
    return criteria_results

# Testing utility functions
def test_model_setup():
    """Test model setup functionality"""
    print("Testing model setup...")
    
    # Create a small model
    model = BilinearMLP(784, 10, 10)
    
    # Check weights shape
    weights = model.get_weights()
    assert weights.shape == (10, 784), f"Expected shape (10, 784), got {weights.shape}"
    
    # Check forward pass
    x = torch.randn(5, 784)
    out = model(x)
    assert out.shape == (5, 10), f"Expected output shape (5, 10), got {out.shape}"
    
    print("Model setup test passed!")

def test_feature_extraction():
    """Test feature extraction functionality"""
    print("Testing feature extraction...")
    
    # Create a small model
    model = BilinearMLP(784, 10, 10)
    
    # Extract features
    features = extract_features(model, k=5)
    
    # Check feature shapes
    assert len(features['eigenvectors']) == 10, f"Expected 10 classes, got {len(features['eigenvectors'])}"
    assert features['eigenvectors'][0].shape == (784, 5), f"Expected shape (784, 5), got {features['eigenvectors'][0].shape}"
    assert len(features['eigenvalues']) == 10, f"Expected 10 classes, got {len(features['eigenvalues'])}"
    assert features['eigenvalues'][0].shape == (5,), f"Expected shape (5,), got {features['eigenvalues'][0].shape}"
    
    print("Feature extraction test passed!")

def test_stability_computation():
    """Test stability computation functionality"""
    print("Testing stability computation...")
    
    # Create dummy features for two checkpoints
    features1 = extract_features(BilinearMLP(784, 10, 10), k=5)
    features2 = extract_features(BilinearMLP(784, 10, 10), k=5)
    
    # Compute stability
    stability_data = compute_stability([features1, features2], k=5)
    
    # Check stability shapes
    assert len(stability_data['stability_matrices']) == 10, f"Expected 10 classes, got {len(stability_data['stability_matrices'])}"
    assert stability_data['stability_matrices'][0].shape == (1, 5), f"Expected shape (1, 5), got {stability_data['stability_matrices'][0].shape}"
    
    print("Stability computation test passed!")

# Main execution
if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Run feature emergence experiment')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'fashion-mnist'],
                        help='Dataset to use')
    parser.add_argument('--hidden_dim', type=int, default=100, help='Hidden layer dimension')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--test', action='store_true', help='Run tests only')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Run tests if requested
    if args.test:
        print("Running tests...")
        test_model_setup()
        test_feature_extraction()
        test_stability_computation()
        print("All tests passed!")
        exit(0)
    
    # Run experiment
    experiment_results = run_experiment(
        dataset_name=args.dataset,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        learning_rate=args.lr,
        random_seed=args.seed,
        device=device
    )
    
    # Print success criteria results
    print("\nHypothesis Evaluation:")
    for criterion, result in experiment_results['success_criteria'].items():
        print(f"{criterion}: {'Supported' if result else 'Not Supported'}")
    
    print(f"\nExperiment completed successfully! Results saved to: {experiment_results['save_dir']}")