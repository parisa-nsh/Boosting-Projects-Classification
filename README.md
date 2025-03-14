# Boosting Projects Classification

This project implements and compares various boosting algorithms for classification tasks using popular machine learning libraries.

## Implemented Algorithms

- XGBoost
- LightGBM
- CatBoost
- Gradient Boosting (scikit-learn)

## Project Structure

```
.
├── data/
│   ├── raw/          # Raw data files
│   └── processed/    # Processed data files
├── src/
│   └── train.py      # Training script with boosting implementations
├── results/          # Model outputs and visualizations
├── logs/            # Training logs
└── requirements.txt  # Project dependencies
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
```

2. Activate the virtual environment:
- Windows:
```bash
venv\Scripts\activate
```
- Unix/MacOS:
```bash
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the training script:
```bash
python src/train.py
```

The script will:
1. Load the selected dataset (default: breast cancer)
2. Train multiple boosting models
3. Compare their performance
4. Generate evaluation metrics and visualizations

## Available Datasets

- Breast Cancer Wisconsin
- Iris
- Wine

## Results

The training script will output:
- Model accuracy scores
- Classification reports
- Feature importance plots (if applicable)
