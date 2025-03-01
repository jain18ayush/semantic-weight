# Neural Network Weight Dynamics During Mono-to-Poly Semantic Transitions

This experiment investigates how neural network weights evolve when neurons transition from encoding single concepts (monosemantic) to encoding multiple concepts (polysemantic).

## Research Question
How do neural network weights evolve when neurons transition from monosemantic to polysemantic representations?

## Hypotheses
1. **Weight Distribution**: As neurons transition to polysemantic representations, their incoming weights become more distributed (less sparse).
   - Success Metric: Higher weight entropy in polysemantic neurons compared to monosemantic neurons.

2. **Weight Magnitude**: The magnitude of weight vectors increases to accommodate multiple feature representations.
   - Success Metric: Larger L2 norm of weight vectors in polysemantic neurons.

3. **Angular Distance**: The angular distance between weight vectors of different neurons decreases as they begin to capture overlapping concepts.
   - Success Metric: Decreased mean angular distance between neuron weight vectors in Phase 2.

## Dataset
The experiment uses a synthetic dataset with two independent feature dimensions:
- **Feature A**: Line orientation (vertical vs. horizontal)
- **Feature B**: Color (red vs. blue)

Key properties:
- Image size: 32×32 pixels (RGB)
- Training set: 5000 images
- Test set: 1000 images
- Random noise added for robustness

## Model Architecture
Simple feedforward neural network:
- Input layer: 3072 nodes (32×32×3 RGB pixels)
- Hidden layer: 10 neurons with ReLU activation
- Output layer: 2 nodes (binary classification)

## Experimental Design

### Phase 1: Monosemantic Training
- Task: Classify images based on line orientation only
- Output encoding: [1,0] = Vertical, [0,1] = Horizontal
- Color is randomly assigned and irrelevant
- Expected outcome: Neurons become orientation-selective

### Phase 2: Polysemantic Training
- Task: Classify specific orientation-color combinations
- Output encoding:
  - Class 1 [1,0] = "Vertical+Red OR Horizontal+Blue"
  - Class 2 [0,1] = "Vertical+Blue OR Horizontal+Red"
- Expected outcome: Some neurons transition to encode both features

## Implementation Details

### Requirements
```
torch>=1.7.0
numpy>=1.19.0
matplotlib>=3.3.0
pandas>=1.2.0
seaborn>=0.11.0
scikit-learn>=0.24.0
Pillow>=8.0.0
```

### Project Structure
```
mono_poly_semantics/
├── README.md
├── requirements.txt
├── src/
│   ├── dataset.py
│   ├── model.py
│   └── experiment.py
├── data/
│   └── dataset_images/
└── results/
    ├── checkpoints/
    ├── plots/
    └── metrics/
```

### Running the Experiment

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the full experiment:
```bash
python src/experiment.py
```

The experiment will:
- Generate synthetic datasets
- Train the model in two phases
- Save checkpoints and visualizations
- Calculate and log metrics

## Success Metrics & Analysis

### 1. Concept Response Selectivity (CRS)
- Measures how selectively a neuron responds to different concepts
- CRS = (max_response - second_max_response) / max_response
- Neurons with CRS < 0.3 are considered polysemantic

### 2. Weight Distribution Analysis
- Weight entropy calculated for each neuron
- Higher entropy indicates more distributed weights
- Tracked across training phases

### 3. Weight Magnitude Analysis
- L2 norm of weight vectors
- Tracked for each neuron throughout training

### 4. Angular Distance Analysis
- Cosine similarity between neuron weight vectors
- Converted to angular distance in degrees
- Mean distance tracked across phases

## Output Files

### Checkpoints
- `checkpoints/phase1/`: Model checkpoints during Phase 1
- `checkpoints/phase2/`: Model checkpoints during Phase 2

### Visualizations
- `plots/phase1_samples.png`: Example images from Phase 1
- `plots/phase2_samples.png`: Example images from Phase 2
- `plots/phase1_weights.png`: Weight visualizations after Phase 1
- `plots/phase2_weights.png`: Weight visualizations after Phase 2
- `plots/metrics_*.png`: Various metric plots

### Metrics
- `metrics/crs_phase1.csv`: CRS scores after Phase 1
- `metrics/crs_phase2.csv`: CRS scores after Phase 2
- `metrics/training_history.csv`: Training metrics over time

## Expected Results

1. After Phase 1:
   - Most neurons should show high CRS for orientation
   - Low CRS for color (random)
   - Relatively sparse weight distributions

2. After Phase 2:
   - Some neurons maintain orientation selectivity
   - Some neurons become polysemantic (low CRS for both features) [increased slightly]
   - Increased weight entropy in polysemantic neurons [decreased]
   - Larger weight magnitudes in polysemantic neurons [x]
   - Decreased angular distances between neurons [x]

## Contributing

To contribute to this experiment:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request with:
   - Clear description of changes
   - Updated tests if necessary
   - Any new visualizations or metrics

## License
MIT License

## Citation
If you use this code in your research, please cite:
```bibtex
@misc{mono2poly2025,
  title={Weight Dynamics During Mono-to-Poly Semantic Transitions},
  author={[Your Name]},
  year={2025},
  publisher={GitHub},
  howpublished={\url{https://github.com/yourusername/mono-poly-semantics}}
}
```