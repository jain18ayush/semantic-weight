import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import math
from collections import defaultdict

# Minimal Transformer with exposed attention weights
class MinimalTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, d_ff: int = 128):
        super().__init__()
        self.d_model = d_model
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Single-head attention
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Feed-forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        
        # Output projection
        self.output = nn.Linear(d_model, vocab_size)
        
        # Store attention weights for analysis
        self.last_attention_weights = None

    def attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                 mask: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        self.last_attention_weights = attn.detach()
        
        return torch.matmul(attn, v), attn

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        # Embedding
        x = self.embedding(x)
        
        # Self-attention
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        x_attn, attention_weights = self.attention(Q, K, V)
        x_attn = self.W_o(x_attn)
        
        # Feed-forward
        out = self.ff(x_attn)
        
        # Output projection
        logits = self.output(out)
        
        if return_attention:
            return logits, attention_weights
        return logits
    
    def get_attention_weights(self) -> torch.Tensor:
        return self.last_attention_weights

# Dataset Generator
class SemanticDataset(Dataset):
    def __init__(self, num_samples: int = 1000, seq_length: int = 10, 
                 vocab_size: int = 1000, semantic_complexity: float = 0.5):
        """
        Generate synthetic data with controllable semantic complexity.
        
        Args:
            semantic_complexity: 0.0 to 1.0, controls how many semantic relationships
                               are present in each sequence
        """
        self.num_samples = num_samples
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.semantic_complexity = semantic_complexity
        
        # Generate data
        self.data, self.targets = self._generate_data()
        
    def _generate_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences with embedded semantic relationships.
        As semantic_complexity increases, more types of relationships are included.
        """
        data = []
        targets = []
        
        for _ in range(self.num_samples):
            # Base sequence
            seq = torch.randint(0, self.vocab_size, (self.seq_length,))
            
            # Add semantic relationships based on complexity
            if self.semantic_complexity > 0.3:
                # Add subject-verb agreement pattern
                subject_pos = torch.randint(0, self.seq_length-1, (1,))
                verb_pos = subject_pos + 1
                if seq[subject_pos] % 2 == 0:  # singular
                    seq[verb_pos] = torch.randint(0, self.vocab_size//4, (1,))
                else:  # plural
                    seq[verb_pos] = torch.randint(self.vocab_size//4, self.vocab_size//2, (1,))
            
            if self.semantic_complexity > 0.6:
                # Add temporal relationships
                time_pos = torch.randint(0, self.seq_length-2, (1,))
                seq[time_pos+2] = (seq[time_pos] + seq[time_pos+1]) % self.vocab_size
            
            if self.semantic_complexity > 0.8:
                # Add more complex dependencies
                for i in range(self.seq_length-3):
                    if torch.rand(1) > 0.7:
                        seq[i+3] = (seq[i] * seq[i+1] + seq[i+2]) % self.vocab_size
            
            # Target is next token prediction
            data.append(seq[:-1])
            targets.append(seq[1:])
        
        return torch.stack(data), torch.stack(targets)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.targets[idx]

# Semantic Transition Analyzer
class SemanticTransitionAnalyzer:
    def __init__(self, model: MinimalTransformer):
        self.model = model
        self.history = []
        
    def analyze_step(self, batch_idx: int, complexity: float) -> Dict:
        """Analyze semantic responsibilities after each batch"""
        attn_weights = self.model.get_attention_weights()
        if attn_weights is None:
            return {}
        
        # Calculate metrics
        entropy = self._compute_attention_entropy(attn_weights)
        gini = self._compute_gini_coefficient(attn_weights)
        
        metrics = {
            'batch': batch_idx,
            'complexity': complexity,
            'attention_entropy': entropy,
            'gini_coefficient': gini
        }
        
        self.history.append(metrics)
        return metrics
    
    def _compute_attention_entropy(self, attention_weights: torch.Tensor) -> float:
        """Compute entropy of attention distribution"""
        # Add small epsilon to avoid log(0)
        entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-10), dim=-1)
        return entropy.mean().item()
    
    def _compute_gini_coefficient(self, weights: torch.Tensor) -> float:
        """Compute Gini coefficient to measure weight specialization"""
        sorted_weights = torch.sort(weights.abs().flatten())[0]
        n = sorted_weights.size(0)
        index = torch.arange(1, n + 1, device=sorted_weights.device)
        return ((torch.sum((2 * index - n - 1) * sorted_weights)) / 
                (n * torch.sum(sorted_weights))).item()
    
    def plot_transitions(self) -> None:
        """Plot the trajectory of semantic transitions"""
        metrics = list(zip(*[(d['batch'], 
                            d['complexity'],
                            d['attention_entropy'],
                            d['gini_coefficient']) 
                           for d in self.history]))
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        
        # Plot complexity over time
        axes[0].plot(metrics[0], metrics[1])
        axes[0].set_title('Semantic Complexity Over Time')
        axes[0].set_xlabel('Batch')
        axes[0].set_ylabel('Complexity')
        
        # Plot entropy over time
        axes[1].plot(metrics[0], metrics[2])
        axes[1].set_title('Attention Entropy Over Time')
        axes[1].set_xlabel('Batch')
        axes[1].set_ylabel('Entropy')
        
        # Plot Gini coefficient over time
        axes[2].plot(metrics[0], metrics[3])
        axes[2].set_title('Weight Specialization Over Time')
        axes[2].set_xlabel('Batch')
        axes[2].set_ylabel('Gini Coefficient')
        
        plt.tight_layout()
        plt.savefig('semantic_transitions.png')
        plt.show()

# Training Loop
def train_model(
    model: MinimalTransformer,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    analyzer: SemanticTransitionAnalyzer,
    complexity: float
) -> float:
    """Single epoch training loop"""
    model.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        
        # Reshape for cross entropy
        output = output.view(-1, output.size(-1))
        target = target.view(-1)
        
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Analyze semantic transitions
        if batch_idx % 10 == 0:
            analyzer.analyze_step(batch_idx, complexity)
    
    return total_loss / len(train_loader)

# Main Experiment
def run_experiment():
    # Parameters
    vocab_size = 1000
    d_model = 64
    batch_size = 32
    num_epochs = 50
    
    # Initialize model and optimizer
    model = MinimalTransformer(vocab_size=vocab_size, d_model=d_model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    analyzer = SemanticTransitionAnalyzer(model)
    
    # Training with gradually increasing complexity
    complexities = np.linspace(0.2, 1.0, num_epochs)
    train_losses = []
    
    for epoch, complexity in enumerate(complexities):
        # Generate data with current complexity
        dataset = SemanticDataset(
            num_samples=1000,
            seq_length=10,
            vocab_size=vocab_size,
            semantic_complexity=complexity
        )
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Train epoch
        avg_loss = train_model(model, train_loader, optimizer, analyzer, complexity)
        train_losses.append(avg_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Complexity: {complexity:.2f}, Loss: {avg_loss:.4f}')
    
    # Plot results
    analyzer.plot_transitions()
    
    # Plot training loss
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

if __name__ == "__main__":
    run_experiment()