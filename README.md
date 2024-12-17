# LIME

## Overview
This project implements a machine learning pipeline for analyzing financial data using neural networks and random forests, with LIME (Local Interpretable Model-agnostic Explanations) for model interpretation.

## Project Structure
- `src/`: Source code
  - `data/`: Data processing modules
  - `models/`: ML model implementations
  - `analysis/`: Analysis utilities
  - `utils/`: Shared utilities
- `scripts/`: Training and analysis scripts
- `tests/`: Unit tests


## Usage
1. Data Processing:
```python
from src.data.data_pipeline import DataPipeline
pipeline = DataPipeline(file_path)
X_train, y_train, X_test, y_test = pipeline.LoadTrainTest(year, period)
```

2. Model Training:
```python
# Neural Network
python scripts/train_nn.py

# Random Forest
python scripts/train_rf.py
```

3. Analysis:
```python
# Run LIME analysis
python scripts/run_lime_analysis.py
```

