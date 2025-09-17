# Data Directory

This directory contains the datasets used for training and testing the GHI forecasting models.

## Data Files

### Training Data
- `training_data.xlsx` - Combined training data from Chennai, Jaisalmer, Leh, and Mawsynram (2017-2019)
- `trained.xlsx` - Pre-processed training data used in the original paper

### Test Data
- `testdata.xlsx` - Delhi test data for model evaluation
- `delhi_test.xlsx` - Alternative test dataset for Delhi

## Data Format

All data files should be Excel (.xlsx) format with the following normalized columns:

### Required Columns
- `T norm` - Temperature (normalized)
- `CDHI norm` - Clearsky Diffuse Horizontal Irradiance (normalized)
- `CDNI norm` - Clearsky Direct Normal Irradiance (normalized) 
- `CGHI norm` - Clearsky Global Horizontal Irradiance (normalized)
- `DHI norm` - Diffuse Horizontal Irradiance (normalized)
- `DNI norm` - Direct Normal Irradiance (normalized)
- `RH norm` - Relative Humidity (normalized)
- `SZA norm` - Solar Zenith Angle (normalized)
- `GHI norm` - Global Horizontal Irradiance (normalized) **[Target Variable]**

## Data Sources

The original data comes from the National Solar Radiation Database (NSRDB):
- **URL**: https://nsrdb.nrel.gov/data-viewer
- **Dataset**: Europe, Africa & Asia (15, 30, 60 min / 4 km / 2017–2019)

### Locations Used

#### Training Locations (2017-2019)
1. **Chennai, Tamil Nadu** - Coastal tropical savanna (Köppen: As)
2. **Jaisalmer, Rajasthan** - Hot desert region (Köppen: BWh)  
3. **Leh, Ladakh** - Cold desert region (Köppen: BWk)
4. **Mawsynram, Meghalaya** - Wet mountain region (Köppen: Cwb)

#### Test Location
- **Delhi** - Semi-arid climate (Köppen: BSh)

## Data Preprocessing

All data has been preprocessed using min-max normalization:

```
X_norm = (X - X_min) / (X_max - X_min)
```

Where:
- `X` = original value
- `X_min` = minimum value in the dataset
- `X_max` = maximum value in the dataset

This ensures all values are in the range [0, 1].

## Feature Selection

Based on the paper's correlation analysis and mutual information, eight key meteorological features were selected from the original 15 parameters:

### Selected Features (Input)
1. Temperature (T norm)
2. Clearsky DHI (CDHI norm)
3. Clearsky DNI (CDNI norm)
4. Clearsky GHI (CGHI norm)
5. Diffuse Horizontal Irradiance (DHI norm)
6. Direct Normal Irradiance (DNI norm)
7. Relative Humidity (RH norm)
8. Solar Zenith Angle (SZA norm)

### Target Variable
- Global Horizontal Irradiance (GHI norm)

## Data Quality

All datasets have been validated for:
- Missing values (removed)
- Outliers (identified using 3σ rule)
- Normalization consistency (values in [0,1])
- Temporal consistency (hourly data)

## Usage

### Loading Data in Python

```python
import pandas as pd
from src.utils.data_loader import DataLoader

# Initialize data loader
loader = DataLoader(logger)

# Load training data
train_data = loader.load_data('data/training_data.xlsx')

# Load test data  
test_data = loader.load_data('data/testdata.xlsx')
```

### Data Validation

The system automatically validates data quality:
- Checks for required columns
- Validates normalization ranges
- Identifies missing values
- Reports data statistics

## File Sizes

Due to GitHub file size limitations, large data files may be stored using Git LFS:
- `.xlsx` files > 25MB use Git LFS
- Download instructions provided in individual file READMEs

## Citation

If you use this data, please cite both our paper and the original NSRDB:

**Our Paper:**
```bibtex
@article{patel2024ghi,
  title={Forecasting and analyzing seasonal GHI for a SAPV system in extreme Indian climatic regions},
  author={Patel, Aadyasha and Gnana Swathika, O. V.},
  year={2024}
}
```

**NSRDB:**
```bibtex
@article{sengupta2018nsrdb,
  title={The national solar radiation data base (NSRDB)},
  author={Sengupta, Manajit and others},
  journal={Renewable and Sustainable Energy Reviews},
  volume={89},
  pages={51--60},
  year={2018}
}
```

## Contact

For data-related questions:
- Check the main README.md
- Open an issue on GitHub
- Contact the authors