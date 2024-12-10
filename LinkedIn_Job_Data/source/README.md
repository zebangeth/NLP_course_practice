# LinkedIn Job Data - Source Code Documentation

This directory contains the source code for analyzing and predicting salaries from job descriptions using both traditional ML and transformer-based approaches.

## Project Structure

```
src/
├── synthetic_data/        # Synthetic dataset generation and validation
├── real_data/            # Real LinkedIn data preprocessing and EDA
├── TFIDF/                # Traditional ML models using TF-IDF training and evaluation
└── transformer_fine_tune/ # Transformer model fine-tuning and evaluation
```

## Components

### 1. Synthetic Data Generation

The synthetic data generator (`synthetic_data_gen.py`) creates a controlled dataset with known patterns for initial model validation. It generates:

- Job titles with varying seniority levels
- Location information
- Industry classifications
- Required skills and categories
- Detailed job descriptions
- Salary ranges based on defined rules

### 2. Real Data Processing

The real data processing pipeline consists of two main notebooks:

- `data_preprocess.ipynb`: Handles data cleaning and standardization
  - Converts salaries to annual USD
  - Standardizes location information
  - Removes outliers and invalid entries
  
- `eda.ipynb`: Performs exploratory data analysis
  - Analyzes salary distributions
  - Examines job title frequencies
  - Studies geographical salary variations

### 3. Traditional ML Models (TF-IDF)

Two training notebooks implementing TF-IDF based approaches:

- `train_syn.ipynb`: Trains and evaluates models on synthetic data
- `train_real.ipynb`: Trains and evaluates models on real LinkedIn data

Models implemented:
- Ridge Regression
- Random Forest
- XGBoost
- LightGBM

### 4. Transformer Fine-tuning

The transformer approach uses fine-tuned language models with examples stored in JSONL format.
