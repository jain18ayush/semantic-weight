import numpy as np
import torch
from scipy.stats import entropy
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

class WeightDynamicsAnalyzer:
    """
    Analyzes and tracks neural network weight dynamics during mono-to-poly semantic transitions.
    
    Key metrics tracked:
    1. Weight Distribution (Entropy)
    2. Weight Magnitude (L2 norm)
    3. Angular Distance between neurons
    4. Concept Response Selectivity (CRS)
    """
    
    def __init__(self, save_dir: str = 'results'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage for metrics
        self.metrics = {
            'phase1': {
                'weight_entropy': [],
                'weight_magnitude': [],
                'angular_distances': [],
                'semanticity_scores': [],
                'category_activations': [],
                'epoch': []
            },
            'phase2': {
                'weight_entropy': [],
                'weight_magnitude': [],
                'angular_distances': [],
                'semanticity_scores': [],
                'category_activations': [],
                'epoch': []
            }
        }
        
        # Keep track of current phase
        self.current_phase = 'phase1'
        
    def analyze_epoch(self, 
                     model: torch.nn.Module,
                     epoch: int,
                     activations: torch.Tensor,
                     phase: Optional[str] = None) -> Dict:
        """
        Analyze network weights and activations after each epoch.
        
        Args:
            model: The neural network model
            epoch: Current epoch number
            activations: Hidden layer activations for the test set
            phase: Current training phase ('phase1' or 'phase2')
        
        Returns:
            Dictionary containing computed metrics for the current epoch
        """
        if phase:
            self.current_phase = phase
            
        # Get model weights
        fc1_weights, _ = model.get_weights()
        
        # Compute metrics
        entropy_scores = self._compute_weight_entropy(fc1_weights)
        magnitudes = self._compute_weight_magnitude(fc1_weights)
        angular_dists = self._compute_angular_distances(fc1_weights)
        semanticity_scores = self._compute_selectivity(activations)
        """
            Semanticity score should be per neuron so the shape should be (n_neurons,)
        """

        # Store metrics
        current_metrics = {
            'weight_entropy': entropy_scores,
            'weight_magnitude': magnitudes,
            'angular_distances': angular_dists,
            'semanticity_scores': semanticity_scores,
            'epoch': epoch
        }
        
        # Update storage
        for key, value in current_metrics.items():
            self.metrics[self.current_phase][key].append(value)
            
        return current_metrics
    
    def _compute_weight_entropy(self, weights: np.ndarray) -> np.ndarray:
        """
        Compute entropy of weight distributions for each neuron.
        Higher entropy indicates more distributed weights.
        """
        # Normalize weights to get probability-like distributions
        weights_norm = np.abs(weights) / np.sum(np.abs(weights), axis=1, keepdims=True)
        return np.array([entropy(w) for w in weights_norm])
    
    def _compute_weight_magnitude(self, weights: np.ndarray) -> np.ndarray:
        """Calculate L2 norm of weight vectors for each neuron."""
        return np.linalg.norm(weights, axis=1)
    
    def _compute_angular_distances(self, weights: np.ndarray) -> float:
        """
        Compute mean angular distance between all pairs of weight vectors.
        """
        # Normalize weights
        weights_norm = weights / np.linalg.norm(weights, axis=1, keepdims=True)
        
        # Compute cosine similarities
        cos_sims = np.dot(weights_norm, weights_norm.T)
        
        # Convert to angles in degrees
        angles = np.arccos(np.clip(cos_sims, -1.0, 1.0)) * 180.0 / np.pi
        
        # Return mean of upper triangle (excluding diagonal)
        return np.mean(angles[np.triu_indices_from(angles, k=1)])
    
    def _calculate_semanticity(self, neuron_acts):
        """
        Calculate neuron semanticity using entropy.
        Lower entropy = more monosemantic (concentrated activations)
        Higher entropy = more polysemantic (distributed activations)
        
        Args:
            neuron_acts: Dict[neuron_idx: Dict[category: List[activations]]]
        Returns:
            Dict[neuron_idx: float]: entropy scores (0-1, higher means more monosemantic)
        """
        scores = {}
        
        for neuron_idx, categories in neuron_acts.items():
            # Get mean activation per category using PyTorch's mean
            avgs = np.array([acts.mean().item() for acts in categories.values()])
            
            # Handle the case where all activations are zero
            if np.all(avgs == 0):
                # Option 1: Assign equal probability to all categories
                # This indicates maximum entropy (most polysemantic)
                scores[neuron_idx] = 0.0
                continue
                
            # Handle case where some activations are negative
            avgs = avgs - np.min(avgs) if np.min(avgs) < 0 else avgs
            
            # Convert to probability distribution (normalize)
            sum_avgs = np.sum(avgs)
            if sum_avgs > 0:
                probs = avgs / sum_avgs
            else:
                # This shouldn't happen after our zero check, but just in case
                probs = np.ones_like(avgs) / len(avgs)
            
            # Calculate entropy and normalize it to 0-1 range
            max_entropy = np.log2(len(categories))
            entropy = -np.sum(probs * np.log2(probs + 1e-10))  # add small epsilon to avoid log(0)
            
            # Convert to semanticity score (1 - normalized entropy)
            scores[neuron_idx] = 1 - (entropy / max_entropy)
        
        return scores
    
    def _compute_selectivity(self, activations: np.ndarray) -> Tuple[Dict[int, float], Dict[int, Dict[str, float]]]:
        """
        Calculate selectivity for each neuron based on category responses.
        A neuron is monosemantic if its average activation for one category 
        is significantly higher than other categories.
        
        Args:
            activations: Array of shape (n_samples, n_neurons) containing neuron activations
            
        Returns:
            Tuple containing:
                - Dict mapping neuron index to selectivity score
                - Dict mapping neuron index to dict of category averages
        """
        neuron_acts = {}
        
        # Organize activations by neuron and category
        for neuron_idx in range(activations.shape[1]):
            if self.current_phase == 'phase1':
                categories = {
                    'vertical': activations[::2, neuron_idx],
                    'horizontal': activations[1::2, neuron_idx]
                }
            else:  # phase2
                categories = {
                    'vertical_red': activations[0::4, neuron_idx],
                    'vertical_blue': activations[1::4, neuron_idx],
                    'horizontal_red': activations[2::4, neuron_idx],
                    'horizontal_blue': activations[3::4, neuron_idx]
                }
            neuron_acts[neuron_idx] = categories

        # Calculate selectivity using the provided function
        neuron_selectivity = self._calculate_semanticity(neuron_acts)
        
        return [neuron_selectivity[idx] for idx in sorted(neuron_selectivity.keys())]
    
    def generate_visualizations(self):
        """
        Generate and save visualizations of tracked metrics.
        """
        self._plot_metric_evolution('weight_entropy', 'Weight Distribution Entropy')
        self._plot_metric_evolution('weight_magnitude', 'Weight Vector Magnitude')
        self._plot_angular_evolution('angular_distances', 'Mean Angular Distance')
        self._plot_metric_evolution('semanticity_scores', 'Semanticity')
        self._plot_phase_comparison()
        
    def _plot_metric_evolution(self, metric_name: str, title: str):
        """Plot evolution of a metric over training epochs."""
        if metric_name == 'category_activations':
            self._plot_category_activations()
            return
            
        plt.figure(figsize=(10, 6))
        
        # Plot Phase 1
        phase1_data = np.array(self.metrics['phase1'][metric_name])
        epochs = self.metrics['phase1']['epoch']
        
        # Handle dictionary data type for selectivity scores
        if isinstance(phase1_data[0], dict):
            phase1_data = np.array([[d[i] for d in phase1_data] for i in range(len(phase1_data[0]))])
        
        plt.plot(epochs, np.mean(phase1_data, axis=1), label='Phase 1', color='blue')
        
        # Plot Phase 2 if exists
        if len(self.metrics['phase2'][metric_name]) > 0:
            phase2_data = np.array(self.metrics['phase2'][metric_name])
            epochs = self.metrics['phase2']['epoch']
            
            if isinstance(phase2_data[0], dict):
                phase2_data = np.array([[d[i] for d in phase2_data] for i in range(len(phase2_data[0]))])
            
            plt.plot(epochs, np.mean(phase2_data, axis=1), label='Phase 2', color='red')
        
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel(metric_name.replace('_', ' ').title())
        plt.legend()
        plt.savefig(self.save_dir / f'{metric_name}_evolution.png')
        plt.close()

    def _plot_angular_evolution(self, metric_name: str, title: str):
        """
        Plot evolution of angular distances over training epochs.
        Unlike other metrics, angular distances don't have a neuron dimension.
        
        Args:
            metric_name (str): Name of the metric to plot
            title (str): Title for the plot
        """
        plt.figure(figsize=(10, 6))
        
        # Plot Phase 1
        phase1_data = np.array(self.metrics['phase1'][metric_name])
        epochs = self.metrics['phase1']['epoch']
        
        # No need for axis manipulation since angular distances are already scalar
        plt.plot(epochs, phase1_data, label='Phase 1', color='blue')
        
        # Plot Phase 2 if exists
        if len(self.metrics['phase2'][metric_name]) > 0:
            phase2_data = np.array(self.metrics['phase2'][metric_name])
            epochs = self.metrics['phase2']['epoch']
            plt.plot(epochs, phase2_data, label='Phase 2', color='red')
        
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Mean Angular Distance (degrees)')
        
        # Add horizontal line at 90 degrees for reference
        plt.axhline(y=90, color='gray', linestyle='--', alpha=0.5, 
                label='Orthogonal (90°)')
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add shaded confidence region if you have min/max or std data
        if 'angular_distances_std' in self.metrics['phase1']:
            std_data = np.array(self.metrics['phase1']['angular_distances_std'])
            mean_data = phase1_data
            plt.fill_between(epochs, 
                            mean_data - std_data,
                            mean_data + std_data,
                            color='blue', alpha=0.2)
            
            if len(self.metrics['phase2'][metric_name]) > 0:
                std_data = np.array(self.metrics['phase2']['angular_distances_std'])
                mean_data = phase2_data
                plt.fill_between(epochs,
                            mean_data - std_data,
                            mean_data + std_data,
                            color='red', alpha=0.2)
        
        plt.savefig(self.save_dir / f'{metric_name}_evolution.png')
        plt.close()

    def _plot_category_activations(self):
        """Create detailed plots of category-specific activations for each phase."""
        for phase in ['phase1', 'phase2']:
            if not self.metrics[phase]['category_activations']:
                continue
                
            # Get the last epoch's category activations
            latest_activations = self.metrics[phase]['category_activations'][-1]
            
            # Create a subplot for each neuron
            n_neurons = len(latest_activations)
            fig, axes = plt.subplots(1, n_neurons, figsize=(4*n_neurons, 4))
            if n_neurons == 1:
                axes = [axes]
            
            for neuron_idx, ax in enumerate(axes):
                neuron_data = latest_activations[neuron_idx]
                categories = list(neuron_data.keys())
                values = list(neuron_data.values())
                
                ax.bar(range(len(categories)), values)
                ax.set_xticks(range(len(categories)))
                ax.set_xticklabels(categories, rotation=45)
                ax.set_title(f'Neuron {neuron_idx}')
                
            plt.suptitle(f'{phase} Category Activations')
            plt.tight_layout()
            plt.savefig(self.save_dir / f'{phase}_category_activations.png')
            plt.close()
        
    def _plot_phase_comparison(self):
        """Generate violin plots comparing distributions between phases."""
        metrics_to_plot = ['weight_entropy', 'weight_magnitude', 'semanticity_scores']
        
        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(15, 5))
        
        for i, metric in enumerate(metrics_to_plot):
            data = {
                'Phase 1': np.array(self.metrics['phase1'][metric]).flatten(),
                'Phase 2': np.array(self.metrics['phase2'][metric]).flatten()
            }
            
            df = pd.DataFrame({
                'Phase': np.repeat(['Phase 1', 'Phase 2'], 
                                 [len(data['Phase 1']), len(data['Phase 2'])]),
                'Value': np.concatenate([data['Phase 1'], data['Phase 2']])
            })
            
            sns.violinplot(data=df, x='Phase', y='Value', ax=axes[i])
            axes[i].set_title(metric.replace('_', ' ').title())
            
        plt.tight_layout()
        plt.savefig(self.save_dir / 'phase_comparison.png')
        plt.close()
        
    def save_metrics(self):
        """Save all metrics to CSV files."""
        for phase in ['phase1', 'phase2']:
            df = pd.DataFrame(self.metrics[phase])
            df.to_csv(self.save_dir / f'{phase}_metrics.csv', index=False)
            
    def load_metrics(self, path: str):
        """Load previously saved metrics."""
        for phase in ['phase1', 'phase2']:
            df = pd.read_csv(Path(path) / f'{phase}_metrics.csv')
            self.metrics[phase] = df.to_dict('list')