# Predictive-alerting-for-cloud-metrics

A JetBrains Internship project: predictive alerting system that expects incidents in cloud services based on historical metric data.

## Overview

This project builds a predictive alerting system for cloud metrics using a 1D CNN baseline model. The system predicts whether an incident will occur within the next H time steps based on the previous W metric values.

## Project Structure

```
├── src/
│   ├── data.py       # Data loading and preprocessing
│   ├── model.py      # Model definitions (1D CNN baseline)
│   ├── train.py      # Training utilities
│   └── evaluate.py   # Evaluation and threshold selection
├── notebooks/
│   ├── 01_eda.ipynb       # Exploratory data analysis
│   └── 02_baseline.ipynb  # Baseline experiments
├── data/              # Dataset (NAB format)
└── artifacts/         # Training run outputs

```

## Quick Start

### Install Dependencies

```bash
pip install -e .
```

### Run Baseline Experiments

Open and run `notebooks/02_baseline.ipynb` to:
- Train multiple CNN configurations
- Evaluate on validation set
- Compare results across runs
- Select best model

### Training a Model

```python
from src.train import train_model
from src.evaluate import evaluate_and_save

config = {
    'model_type': 'cnn1d',
    'window_size': 24,
    'filters': [32, 64],
    'kernel_size': 3,
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 50,
}

model, history, run_dir = train_model(
    X_train, y_train, X_val, y_val, config
)

metrics = evaluate_and_save(model, X_val, y_val, run_dir)
```

### Using Weights & Biases

To enable W&B logging:

```python
model, history, run_dir = train_model(
    X_train, y_train, X_val, y_val, config,
    use_wandb=True,
    project_name="predictive-alerting",
)
```

## Model Architecture

The baseline uses a 1D CNN with:
- Multiple convolutional layers with ReLU activation
- MaxPooling for temporal downsampling
- Dropout for regularization
- Dense layers for classification

## Data Format

- Input: `(n_samples, window_size, 1)` - sliding windows of metric values
- Output: `(n_samples,)` - binary labels (0=no incident, 1=incident)
- Data is normalized per series using z-score normalization

## Training Artifacts

Each run creates a directory in `artifacts/run_YYYYMMDD_HHMMSS/` containing:
- `model.pt` - Trained PyTorch model
- `config.json` - Hyperparameters
- `history.json` - Training/validation loss
- `metrics_val.json` - Validation metrics
- `threshold.json` - Optimal classification threshold

## Cross-Series Evaluation

The project uses cross-series evaluation to ensure models generalize:
- **Train**: Multiple time series
- **Validation**: Separate time series for threshold selection
- **Test**: Held-out series for final evaluation
