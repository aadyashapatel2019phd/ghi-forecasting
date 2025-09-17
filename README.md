# Seasonal GHI Forecasting for Stand-Alone Photovoltaic Systems

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://img.shields.io/badge/DOI-10.xxxx/xxxx-blue.svg)](https://doi.org/10.xxxx/xxxx)

This repository contains the implementation of machine learning models for seasonal Global Horizontal Irradiance (GHI) forecasting across diverse climatic regions, as described in the paper "Forecasting and analyzing seasonal GHI for a SAPV system in extreme Indian climatic regions".

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Data Format](#data-format)
- [Model Details](#model-details)
- [Results Reproduction](#results-reproduction)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## Overview

This project implements three machine learning models for seasonal GHI prediction:

1. **Efficient Linear Regression (ELR)** - Linear model with regularization
2. **Regression Trees (RT)** - Decision tree-based approach
3. **Gaussian Process Regression (GPR)** - Non-parametric Bayesian method

The models are trained on meteorological data from the National Solar Radiation Database (NSRDB) and can predict seasonal GHI patterns for Stand-Alone Photovoltaic (SAPV) systems.

## Features

- **Multiple Model Support**: ELR, RT, and GPR implementations
- **Hyperparameter Optimization**: Automated tuning using Optuna
- **Comprehensive Evaluation**: Multiple metrics and diagnostic plots
- **Reproducible Results**: Exact reproduction of paper results
- **Flexible Configuration**: YAML-based configuration system
- **Command-Line Interface**: Easy-to-use CLI for all operations
- **Extensive Logging**: Detailed logging for debugging and monitoring
- **Data Validation**: Robust data quality checks and preprocessing

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Install from Source

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ghi-forecasting.git
cd ghi-forecasting
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Test with Pre-trained Model (Delhi Example)

To reproduce the paper's results for Delhi using pre-trained models:

```bash
# Test GPR model on Delhi data
python main.py --model GPR --mode test --pretrained models/trained_model.pkl --test_data data/testdata.xlsx

# Test all models and compare
python main.py --model ELR --mode test --pretrained models/elr_trained.pkl --test_data data/testdata.xlsx
python main.py --model RT --mode test --pretrained models/rt_trained.pkl --test_data data/testdata.xlsx
python main.py --model GPR --mode test --pretrained models/gpr_trained.pkl --test_data data/testdata.xlsx
```

### 2. Train New Models

```bash
# Train a new GPR model
python main.py --model GPR --mode train --data data/training_data.xlsx

# Train with hyperparameter optimization
python main.py --model GPR --mode train --data data/training_data.xlsx --optimize_hyperparams --n_trials 100
```

### 3. Train and Test Workflow

```bash
# Complete workflow: train on historical data, test on new location
python main.py --model GPR --mode train_test --data data/training_data.xlsx --test_data data/testdata.xlsx
```

## Usage

### Command-Line Interface

The main interface is through `main.py` with the following options:

```
python main.py [OPTIONS]

Options:
  --model {ELR,GPR,RT}          Model type (required)
  --mode {train,test,train_test} Execution mode (required)
  --data PATH                   Training data file (required)
  --test_data PATH              Test data file
  --pretrained PATH             Pre-trained model file
  --output_dir PATH             Output directory (default: results)
  --config PATH                 Configuration file (default: config/config.yaml)
  --optimize_hyperparams        Enable hyperparameter optimization
  --n_trials INT                Number of optimization trials (default: 60)
  --verbose                     Enable verbose logging
  --help                        Show help message
```

### Examples

#### Training Examples
```bash
# Basic training with default hyperparameters
python main.py --model GPR --mode train --data data/chennai_data.xlsx

# Training with hyperparameter optimization
python main.py --model RT --mode train --data data/jaisalmer_data.xlsx --optimize_hyperparams

# Custom output directory
python main.py --model ELR --mode train --data data/leh_data.xlsx --output_dir results/leh_experiment
```

#### Testing Examples
```bash
# Test pre-trained model
python main.py --model GPR --mode test --pretrained models/gpr_chennai.pkl --test_data data/delhi_test.xlsx

# Test with custom configuration
python main.py --model RT --mode test --pretrained models/rt_model.pkl --test_data data/test.xlsx --config config/custom.yaml
```

#### Complete Workflow
```bash
# Train on multiple locations, test on Delhi
python main.py --model GPR --mode train_test --data data/multi_location_training.xlsx --test_data data/delhi_test.xlsx
```

## Data Format

### Input Data Requirements

The system expects Excel files (.xlsx) with normalized meteorological data. Based on the paper's feature selection, the following columns are required:

#### Required Columns
- `T norm`: Temperature (normalized)
- `CDHI norm`: Clearsky Diffuse Horizontal Irradiance (normalized)
- `CDNI norm`: Clearsky Direct Normal Irradiance (normalized)
- `CGHI norm`: Clearsky Global Horizontal Irradiance (normalized)
- `DHI norm`: Diffuse Horizontal Irradiance (normalized)
- `DNI norm`: Direct Normal Irradiance (normalized)
- `RH norm`: Relative Humidity (normalized)
- `SZA norm`: Solar Zenith Angle (normalized)
- `GHI norm`: Global Horizontal Irradiance (normalized) - Target variable

#### Data Preprocessing

The original paper used min-max normalization:
```
X_norm = (X - X_min) / (X_max - X_min)
```

All values should be in the range [0, 1].

### Data Sources

The paper used data from the [National Solar Radiation Database (NSRDB)](https://nsrdb.nrel.gov/data-viewer):
- **Dataset**: Europe, Africa & Asia (15, 30, 60 min / 4 km / 2017–2019)
- **Locations**: Chennai, Jaisalmer, Leh, Mawsynram
- **Test Location**: Delhi
- **Time Period**: 2017-2019 (training), 2019 (testing)

## Model Details

### Efficient Linear Regression (ELR)

Linear regression with regularization options:
- **Default**: Ridge regression (λ = 0.09694)
- **Alternatives**: Lasso regression, SVM-based regression
- **Paper Performance**: RMSE = 0.1070, R² = 0.8135

### Regression Trees (RT)

Decision tree-based regression:
- **Default**: min_samples_leaf = 3
- **Paper Performance**: RMSE = 0.0128, R² = 0.9973

### Gaussian Process Regression (GPR)

Non-parametric Bayesian approach:
- **Default Kernel**: Isotropic Matérn 5/2 (scale = 5.9895)
- **Noise Level**: σ = 0.00010961
- **Paper Performance**: RMSE = 0.0030, R² = 0.9999

## Results Reproduction

### Reproducing Paper Results

The repository includes pre-trained models and test data to exactly reproduce the paper's results:

1. **Load Pre-trained Models**: Models trained on Chennai, Jaisalmer, Leh, and Mawsynram (2017-2019)
2. **Test on Delhi**: Evaluate performance on Delhi test data
3. **Compare Results**: Generate comparison tables and plots

```bash
# Reproduce main results table (Table 5 equivalent)
python scripts/reproduce_results.py --output results/reproduction

# Generate diagnostic plots (Figures 7-8 equivalent)
python scripts/generate_diagnostics.py --model GPR --data data/testdata.xlsx
```

### Expected Results (Delhi Test Data)

Based on the paper's methodology, expected performance on Delhi:

| Model | RMSE   | MAE    | R²     |
|-------|--------|--------|--------|
| ELR   | ~0.100 | ~0.080 | ~0.85  |
| RT    | ~0.015 | ~0.008 | ~0.997 |
| GPR   | ~0.004 | ~0.003 | ~0.999 |

## Configuration

### Configuration File Structure

The system uses YAML configuration files. See `config/config.yaml` for the complete structure:

```yaml
models:
  GPR:
    default_params:
      basis_function: "linear"
      kernel_function: "Isotropic Matern 5/2" 
      kernel_scale: 5.9895
      sigma: 0.00010961
      standardize: false

optimization:
  n_trials: 60
  cv_folds: 5
  scoring: "neg_mean_squared_error"
```

### Custom Configuration

Create custom configurations for different experiments:

```yaml
# config/high_precision.yaml
optimization:
  n_trials: 200
  cv_folds: 10

evaluation:
  plot_dpi: 600
```

Use with:
```bash
python main.py --config config/high_precision.yaml --model GPR --mode train --data data/training.xlsx
```

## Output Structure

The system generates organized outputs:

```
results/
├── models/
│   ├── GPR_trained_model.pkl
│   ├── RT_trained_model.pkl
│   └── ELR_trained_model.pkl
├── plots/
│   ├── actual_vs_predicted.png
│   ├── residual_scatter.png
│   ├── residual_histogram.png
│   ├── qq_plot.png
│   └── time_series_comparison.png
├── logs/
│   └── main.log
├── GPR_test_results.json
├── model_comparison.csv
└── model_comparison.png
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Run tests: `python -m pytest tests/`
5. Submit a pull request

### Code Style

- Follow PEP 8 style guidelines
- Add docstrings to all functions and classes
- Include type hints where appropriate
- Write unit tests for new functionality

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{patel2024ghi,
  title={Forecasting and analyzing seasonal GHI for a SAPV system in extreme Indian climatic regions},
  author={Patel, Aadyasha and Gnana Swathika, O. V.},
  journal={Journal Name},
  year={2024},
  publisher={Publisher}
}
```

## Data Availability

The datasets used in this research are available through the NSRDB repository:
- **Source**: [NSRDB Data Viewer](https://nsrdb.nrel.gov/data-viewer)
- **Dataset**: Europe, Africa & Asia (15, 30, 60 min / 4 km / 2017–2019)

## Authors

- **Aadyasha Patel** - School of Electrical Engineering, Vellore Institute of Technology, Chennai, India
- **O. V. Gnana Swathika** - Centre for Smart Grid Technologies, School of Electrical Engineering, Vellore Institute of Technology, Chennai, India

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- National Solar Radiation Database (NSRDB) for providing the meteorological data
- Vellore Institute of Technology for research support
- The open-source community for the excellent tools and libraries used in this project

## Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/yourusername/ghi-forecasting/issues) page
2. Review the documentation
3. Contact the authors

---

**Keywords**: GHI forecasting, seasonal GHI, stand-alone photovoltaic, gaussian process regression, machine learning, solar energy, renewable energy forecasting