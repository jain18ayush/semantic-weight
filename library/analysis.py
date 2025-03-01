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

import numpy as np
import torch
from scipy.stats import entropy
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional

class CurriculumWeightAnalyzer:
    """
    Analyzes neural network weight dynamics during curriculum learning.
    Tracks changes across complexity levels instead of discrete phases.
    """
    
    def __init__(self, save_dir: str = 'results'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage for metrics
        self.metrics = {
            'complexity': [],
            'epoch': [],
            'weight_entropy': [],
            'weight_magnitude': [],
            'angular_distances': [],
            'semanticity_scores': [],
            'feature_importance': []
        }
        
    def analyze_epoch(self, 
                     model: torch.nn.Module,
                     epoch: int,
                     activations: torch.Tensor,
                     complexity: float) -> Dict:
        """
        Analyze network weights and activations for current complexity level.
        
        Args:
            model: Neural network model
            epoch: Current epoch number
            activations: Hidden layer activations
            complexity: Current curriculum complexity (0.0 to 1.0)
        """
        # Get model weights
        fc1_weights, _ = model.get_weights()
        
        # Compute metrics
        entropy_scores = self._compute_weight_entropy(fc1_weights)
        magnitudes = self._compute_weight_magnitude(fc1_weights)
        angular_dists = self._compute_angular_distances(fc1_weights)
        semanticity = self._compute_selectivity(activations)
        feature_importance = self._compute_feature_importance(fc1_weights, complexity)
        
        # Store metrics
        self.metrics['complexity'].append(complexity)
        self.metrics['epoch'].append(epoch)
        self.metrics['weight_entropy'].append(entropy_scores)
        self.metrics['weight_magnitude'].append(magnitudes)
        self.metrics['angular_distances'].append(angular_dists)
        self.metrics['semanticity_scores'].append(semanticity)
        self.metrics['feature_importance'].append(feature_importance)
        
        return {
            'weight_entropy': entropy_scores,
            'weight_magnitude': magnitudes,
            'angular_distances': angular_dists,
            'semanticity_scores': semanticity,
            'feature_importance': feature_importance
        }
    
    def _compute_weight_entropy(self, weights: np.ndarray) -> np.ndarray:
        """Compute entropy of weight distributions for each neuron."""
        weights_norm = np.abs(weights) / np.sum(np.abs(weights), axis=1, keepdims=True)
        return np.array([entropy(w) for w in weights_norm])
    
    def _compute_weight_magnitude(self, weights: np.ndarray) -> np.ndarray:
        """Calculate L2 norm of weight vectors for each neuron."""
        return np.linalg.norm(weights, axis=1)
    
    def _compute_angular_distances(self, weights: np.ndarray) -> float:
        """Compute mean angular distance between all pairs of weight vectors."""
        weights_norm = weights / np.linalg.norm(weights, axis=1, keepdims=True)
        cos_sims = np.dot(weights_norm, weights_norm.T)
        angles = np.arccos(np.clip(cos_sims, -1.0, 1.0)) * 180.0 / np.pi
        return np.mean(angles[np.triu_indices_from(angles, k=1)])
    
    def _compute_selectivity(self, activations: np.ndarray) -> List[float]:
        """
        Calculate selectivity for each neuron based on activation patterns.
        Now considers continuous complexity levels.
        """
        neuron_selectivity = []
        for neuron_idx in range(activations.shape[1]):
            neuron_acts = activations[:, neuron_idx]
            # Calculate entropy of activation distribution
            act_hist, _ = np.histogram(neuron_acts, bins=20, density=True)
            act_hist = act_hist / np.sum(act_hist)
            selectivity = 1 - entropy(act_hist) / np.log2(len(act_hist))
            neuron_selectivity.append(selectivity)
        return neuron_selectivity

    def _compute_feature_importance(self, weights: np.ndarray, complexity: float) -> Dict[str, float]:
        """
        Estimate importance of different features based on weight patterns.
        Separates weights into regions corresponding to different features.
        """
        # Assuming input image is 32x32x3, weights shape is (n_neurons, 3072)
        n_neurons = weights.shape[0]
        
        # Separate weights for different channels (RGB)
        weights_r = weights[:, :1024].reshape(n_neurons, 32, 32)
        weights_g = weights[:, 1024:2048].reshape(n_neurons, 32, 32)
        weights_b = weights[:, 2048:].reshape(n_neurons, 32, 32)
        
        # Compute importance scores
        color_importance = np.mean(np.abs(weights_r - weights_b))  # Red vs Blue
        texture_importance = np.mean(np.abs(np.diff(weights_r, axis=1))) # Horizontal gradients
        orientation_importance = np.mean(np.abs(weights_r + weights_g + weights_b))
        
        return {
            'color': float(color_importance),
            'texture': float(texture_importance),
            'orientation': float(orientation_importance)
        }
    
    def generate_visualizations(self):
        """Generate visualizations of metric evolution with complexity."""
        self._plot_metrics_vs_complexity()
        self._plot_feature_importance()
        self._plot_correlation_matrix()
        self._plot_neuron_trajectories()
    
    def _plot_metrics_vs_complexity(self):
        """Plot how different metrics evolve with complexity."""
        metrics_to_plot = ['weight_entropy', 'weight_magnitude', 'angular_distances', 'semanticity_scores']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_to_plot):
            data = np.array(self.metrics[metric])
            complexity = np.array(self.metrics['complexity'])
            
            if data.ndim > 1:  # If we have per-neuron metrics
                mean_data = np.mean(data, axis=1)
                std_data = np.std(data, axis=1)
                axes[idx].fill_between(complexity, mean_data - std_data, 
                                     mean_data + std_data, alpha=0.3)
                axes[idx].plot(complexity, mean_data)
            else:
                axes[idx].plot(complexity, data)
                
            axes[idx].set_title(metric.replace('_', ' ').title())
            axes[idx].set_xlabel('Complexity')
            axes[idx].grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(self.save_dir / 'metrics_vs_complexity.png')
        plt.close()
    
    def _plot_feature_importance(self):
        """Plot evolution of feature importance scores."""
        complexities = self.metrics['complexity']
        importance_data = pd.DataFrame(self.metrics['feature_importance'])
        
        plt.figure(figsize=(10, 6))
        for feature in ['color', 'texture', 'orientation']:
            plt.plot(complexities, importance_data[feature], label=feature)
            
        plt.xlabel('Complexity')
        plt.ylabel('Feature Importance')
        plt.title('Feature Importance Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.save_dir / 'feature_importance.png')
        plt.close()
    
    def _plot_correlation_matrix(self):
        """Plot correlation matrix between different metrics."""
        metrics_to_correlate = ['weight_entropy', 'weight_magnitude', 
                              'angular_distances', 'semanticity_scores']
        
        # Prepare data for correlation
        data_dict = {}
        for metric in metrics_to_correlate:
            metric_data = np.array(self.metrics[metric])
            if metric_data.ndim > 1:
                data_dict[metric] = np.mean(metric_data, axis=1)
            else:
                data_dict[metric] = metric_data
                
        data_df = pd.DataFrame(data_dict)
        
        # Plot correlation matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(data_df.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Metric Correlations')
        plt.savefig(self.save_dir / 'metric_correlations.png')
        plt.close()
    
    def _plot_neuron_trajectories(self):
        """
        Plot the trajectory of each neuron's complexity over time.
        Visualizes how neuron complexity evolves during training.
        """
        plt.figure(figsize=(10, 6))
        
        # Get complexity scores over time
        entropies = np.array(self.metrics['weight_entropy'])
        magnitudes = np.array(self.metrics['weight_magnitude'])
        
        # Calculate complexity scores for each neuron at each timepoint
        n_timepoints, n_neurons = entropies.shape
        complexity_scores = np.zeros((n_timepoints, n_neurons))
            
            # Calculate complexity scores for each neuron at each timepoint
        n_timepoints, n_neurons = entropies.shape
        complexity_scores = np.zeros((n_timepoints, n_neurons))
        
        for t in range(n_timepoints):
            for n in range(n_neurons):
                complexity_scores[t, n] = self._calculate_complexity_score(
                    entropies[t, n], 
                    magnitudes[t, n]
                )
            
        # Calculate complexity for each timepoint and neuron
        for t in range(n_timepoints):
            for n in range(n_neurons):
                complexity_scores[t, n] = self._calculate_complexity_score(
                    entropies[t, n], 
                    magnitudes[t, n]
                )
        
        # Get epochs
        epochs = np.array(self.metrics['epoch'])
        
        # Create colormap for neuron trajectories
        colors = plt.cm.viridis(np.linspace(0, 1, n_neurons))
        
        # Plot each neuron's trajectory
        for neuron_idx in range(n_neurons):
            neuron_complexity = complexity_scores[:, neuron_idx]
            
            # Plot with arrows to show direction of change
            plt.quiver(epochs[:-1], 
                    neuron_complexity[:-1],
                    epochs[1:] - epochs[:-1],
                    neuron_complexity[1:] - neuron_complexity[:-1],
                    color=colors[neuron_idx],
                    angles='xy', 
                    scale_units='xy', 
                    scale=1,
                    width=0.003,
                    alpha=0.6)
            
            # Plot final position
            plt.scatter(epochs[-1], 
                    neuron_complexity[-1], 
                    c=[colors[neuron_idx]], 
                    marker='o',
                    s=100,
                    label=f'Neuron {neuron_idx}')
            
        plt.xlabel('Training Epoch')
        plt.ylabel('Complexity Score')
        plt.title('Neuron Complexity Trajectories')
        plt.ylim(0, 1)  # Complexity score is normalized between 0 and 1
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                norm=plt.Normalize(vmin=0, vmax=n_neurons-1))
        sm.set_array([])
        plt.colorbar(sm, label='Neuron Index')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.save_dir / 'neuron_trajectories.png')
        plt.close()



    def _calculate_complexity_score(self, entropy: float, magnitude: float) -> float:
        """
        Calculate a neuron's complexity score based on its weight entropy and magnitude.
        
        Args:
            entropy: Weight distribution entropy
            magnitude: Weight vector magnitude
            
        Returns:
            float: Complexity score between 0 and 1
        """
        # Normalize both metrics to 0-1 range using recorded min/max values
        max_entropy = max(self.metrics['weight_entropy'])
        max_magnitude = max(self.metrics['weight_magnitude'])
        
        norm_entropy = entropy / max_entropy
        norm_magnitude = magnitude / max_magnitude
        
        # Combine metrics with equal weighting
        return (norm_entropy + norm_magnitude) / 2    
    
    def save_metrics(self):
        """Save all metrics to CSV files."""
        df = pd.DataFrame({
            'epoch': self.metrics['epoch'],
            'complexity': self.metrics['complexity']
        })
        
        # Add metrics that are scalar per epoch
        for metric in ['angular_distances']:
            df[metric] = self.metrics[metric]
            
        # Add metrics that are per-neuron
        for metric in ['weight_entropy', 'weight_magnitude', 'semanticity_scores']:
            metric_data = np.array(self.metrics[metric])
            for neuron_idx in range(metric_data.shape[1]):
                df[f'{metric}_neuron_{neuron_idx}'] = metric_data[:, neuron_idx]
                
        # Add feature importance scores
        importance_df = pd.DataFrame(self.metrics['feature_importance'])
        df = pd.concat([df, importance_df], axis=1)
        
        df.to_csv(self.save_dir / 'curriculum_metrics.csv', index=False)
        
    def load_metrics(self, path: str):
        """Load previously saved metrics."""
        df = pd.read_csv(Path(path) / 'curriculum_metrics.csv')
        
        # Reconstruct metrics dictionary
        self.metrics['epoch'] = df['epoch'].tolist()
        self.metrics['complexity'] = df['complexity'].tolist()
        self.metrics['angular_distances'] = df['angular_distances'].tolist()
        
        # Reconstruct per-neuron metrics
        n_neurons = len([col for col in df.columns if 'weight_entropy_neuron_' in col])
        
        for metric in ['weight_entropy', 'weight_magnitude', 'semanticity_scores']:
            metric_data = []
            for i in range(len(df)):
                neuron_values = [df[f'{metric}_neuron_{j}'][i] for j in range(n_neurons)]
                metric_data.append(neuron_values)
            self.metrics[metric] = metric_data
            
        # Reconstruct feature importance
        self.metrics['feature_importance'] = df[['color', 'texture', 'orientation']].to_dict('records')