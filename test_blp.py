import unittest
import torch
import numpy as np
import os
import shutil
from torch.utils.data import TensorDataset, DataLoader

# Import from the main script
from bilinear_mlp_feature_emergence import (
    BilinearMLP, 
    set_seed, 
    extract_features, 
    compute_stability,
    eigenvalue_evolution,
    measure_feature_importance,
    TruncatedModel
)

class TestBilinearMLPFeatureEmergence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set consistent random seed
        set_seed(42)
        
        # Create a temporary directory for test outputs
        cls.test_dir = 'test_output'
        os.makedirs(cls.test_dir, exist_ok=True)
        
        # Create small test data
        cls.input_dim = 16
        cls.hidden_dim = 10
        cls.output_dim = 5
        cls.batch_size = 4
        
        # Create a small model for testing
        cls.model = BilinearMLP(cls.input_dim, cls.hidden_dim, cls.output_dim)
        
        # Create dummy data
        x = torch.randn(20, cls.input_dim)
        y = torch.randint(0, cls.output_dim, (20,))
        dataset = TensorDataset(x, y)
        cls.data_loader = DataLoader(dataset, batch_size=cls.batch_size)
    
    @classmethod
    def tearDownClass(cls):
        # Remove test directory
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
    
    def test_model_initialization(self):
        """Test model initialization and structure"""
        model = BilinearMLP(self.input_dim, self.hidden_dim, self.output_dim)
        
        # Check dimensions
        self.assertEqual(model.input_dim, self.input_dim)
        self.assertEqual(model.hidden_dim, self.hidden_dim)
        self.assertEqual(model.output_dim, self.output_dim)
        
        # Check weights shape
        weights = model.get_weights()
        self.assertEqual(weights.shape, (self.hidden_dim, self.input_dim))
    
    def test_model_forward(self):
        """Test model forward pass"""
        # Input batch
        x = torch.randn(self.batch_size, self.input_dim)
        
        # Forward pass
        out = self.model(x)
        
        # Check output shape
        self.assertEqual(out.shape, (self.batch_size, self.output_dim))
    
    def test_feature_extraction(self):
        """Test eigenvector extraction from weights"""
        # Define number of top features to extract
        k = 5
        
        # Extract features
        features = extract_features(self.model, k=k)
        
        # Check returned dictionary
        self.assertIn('eigenvectors', features)
        self.assertIn('eigenvalues', features)
        
        # Check structure
        self.assertEqual(len(features['eigenvectors']), self.output_dim)
        self.assertEqual(len(features['eigenvalues']), self.output_dim)
        
        # Check dimensions
        for c in range(self.output_dim):
            self.assertEqual(features['eigenvectors'][c].shape, (self.input_dim, k))
            self.assertEqual(features['eigenvalues'][c].shape, (k,))
            
            # Check eigenvalues are sorted in descending order
            eigenvalues = features['eigenvalues'][c]
            self.assertTrue(np.all(eigenvalues[:-1] >= eigenvalues[1:]))
    
    def test_stability_computation(self):
        """Test stability computation between checkpoints"""
        # Create features for two checkpoints
        model1 = BilinearMLP(self.input_dim, self.hidden_dim, self.output_dim)
        model2 = BilinearMLP(self.input_dim, self.hidden_dim, self.output_dim)
        
        k = 5
        features1 = extract_features(model1, k=k)
        features2 = extract_features(model2, k=k)
        
        # Compute stability
        stability_data = compute_stability([features1, features2], k=k)
        
        # Check structure
        self.assertIn('stability_matrices', stability_data)
        self.assertIn('stabilization_points', stability_data)
        
        # Check dimensions
        self.assertEqual(len(stability_data['stability_matrices']), self.output_dim)
        self.assertEqual(len(stability_data['stabilization_points']), self.output_dim)
        
        for c in range(self.output_dim):
            self.assertEqual(stability_data['stability_matrices'][c].shape, (1, k))
            self.assertEqual(stability_data['stabilization_points'][c].shape, (k,))
            
            # Check values are valid cosine similarities (between 0 and 1)
            similarities = stability_data['stability_matrices'][c]
            self.assertTrue(np.all(similarities >= 0))
            self.assertTrue(np.all(similarities <= 1))
    
    def test_eigenvalue_evolution(self):
        """Test eigenvalue evolution tracking"""
        # Create features for two checkpoints
        model1 = BilinearMLP(self.input_dim, self.hidden_dim, self.output_dim)
        model2 = BilinearMLP(self.input_dim, self.hidden_dim, self.output_dim)
        
        k = 5
        features1 = extract_features(model1, k=k)
        features2 = extract_features(model2, k=k)
        
        # Track eigenvalue evolution
        eigenvalue_data = eigenvalue_evolution([features1, features2], k=k)
        
        # Check structure
        self.assertIn('eigenvalue_trajectories', eigenvalue_data)
        self.assertIn('concentration_metrics', eigenvalue_data)
        
        # Check dimensions
        self.assertEqual(len(eigenvalue_data['eigenvalue_trajectories']), self.output_dim)
        self.assertEqual(len(eigenvalue_data['concentration_metrics']), self.output_dim)
        
        for c in range(self.output_dim):
            self.assertEqual(eigenvalue_data['eigenvalue_trajectories'][c].shape, (2, k))
            self.assertEqual(eigenvalue_data['concentration_metrics'][c].shape, (2, k))
            
            # Check concentration metrics are between 0 and 1
            concentrations = eigenvalue_data['concentration_metrics'][c]
            self.assertTrue(np.all(concentrations >= 0))
            self.assertTrue(np.all(concentrations <= 1))
            
            # Check concentration is increasing with more eigenvalues
            for i in range(2):
                self.assertTrue(np.all(concentrations[i, :-1] <= concentrations[i, 1:]))
    
    def test_truncated_model(self):
        """Test truncated model for feature importance verification"""
        # Create features
        k = 5
        features = extract_features(self.model, k=k)
        
        # Create truncated model
        truncated_model = TruncatedModel(self.model, features, k)
        
        # Check forward pass
        x = torch.randn(self.batch_size, self.input_dim)
        out = truncated_model(x)
        
        # Check output shape
        self.assertEqual(out.shape, (self.batch_size, self.output_dim))
    
    def test_check_hypothesis(self):
        """Test hypothesis checking logic"""
        # Import the function
        from bilinear_mlp_feature_emergence import check_hypothesis
        
        # Create dummy stability data
        stability_data = {
            'stabilization_points': {
                0: np.array([0, 0, 0, 0, 0, 6, 7, 8, 9, 10]),  # Top 5 stabilize early
                1: np.array([1, 1, 1, 1, 1, 6, 7, 8, 9, 10])
            }
        }
        
        # Create dummy eigenvalue data
        eigenvalue_data = {}
        
        # Create dummy importance data
        importance_data = {
            'minimal_representations': np.array([10, 8, 6, 5, 4])  # Decreasing over time
        }
        
        # Define checkpoint percentages
        checkpoint_percentages = [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        
        # Check hypothesis
        criteria_results = check_hypothesis(
            stability_data, eigenvalue_data, importance_data, checkpoint_percentages
        )
        
        # Check structure
        self.assertIn('top5_early_stabilization', criteria_results)
        self.assertIn('lower_late_evolution', criteria_results)
        self.assertIn('decreasing_minimal_representation', criteria_results)
        self.assertIn('hypothesis_supported', criteria_results)
        
        # Check specific results based on our dummy data
        self.assertTrue(criteria_results['top5_early_stabilization'])
        self.assertTrue(criteria_results['lower_late_evolution'])
        self.assertTrue(criteria_results['decreasing_minimal_representation'])
        self.assertTrue(criteria_results['hypothesis_supported'])

if __name__ == '__main__':
    unittest.main()