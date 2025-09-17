"""
Data Loading Utilities
Handles loading and basic validation of Excel data files
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List


class DataLoader:
    """Handles loading of Excel data files with validation."""
    
    def __init__(self, logger):
        """Initialize data loader."""
        self.logger = logger
        
        # Required columns based on the paper's feature selection
        self.required_columns = [
            'T norm', 'CDHI norm', 'CDNI norm', 'CGHI norm', 
            'DHI norm', 'DNI norm', 'RH norm', 'SZA norm', 'GHI norm'
        ]
        
        # Original feature names for reference
        self.original_features = {
            'T norm': 'Temperature (normalized)',
            'CDHI norm': 'Clearsky DHI (normalized)',
            'CDNI norm': 'Clearsky DNI (normalized)', 
            'CGHI norm': 'Clearsky GHI (normalized)',
            'DHI norm': 'Diffuse Horizontal Irradiance (normalized)',
            'DNI norm': 'Direct Normal Irradiance (normalized)',
            'RH norm': 'Relative Humidity (normalized)',
            'SZA norm': 'Solar Zenith Angle (normalized)',
            'GHI norm': 'Global Horizontal Irradiance (normalized)'
        }
    
    def load_data(self, file_path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from Excel file with validation.
        
        Args:
            file_path: Path to Excel file
            sheet_name: Specific sheet name to load (optional)
            
        Returns:
            Loaded and validated DataFrame
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        self.logger.info(f"Loading data from: {file_path}")
        
        try:
            # Load Excel file
            xls = pd.ExcelFile(file_path)
            
            # Determine which sheet to use
            if sheet_name:
                if sheet_name not in xls.sheet_names:
                    raise ValueError(f"Sheet '{sheet_name}' not found in {file_path}")
                df = xls.parse(sheet_name)
                self.logger.info(f"Using specified sheet: {sheet_name}")
            else:
                # Try common sheet names
                priority_sheets = ['Train', 'Training', 'Data', 'Sheet1']
                sheet_to_use = None
                
                for sheet in priority_sheets:
                    if sheet in xls.sheet_names:
                        sheet_to_use = sheet
                        break
                
                if sheet_to_use is None:
                    sheet_to_use = xls.sheet_names[0]
                
                df = xls.parse(sheet_to_use)
                self.logger.info(f"Using sheet: {sheet_to_use}")
        
        except Exception as e:
            raise ValueError(f"Error reading Excel file {file_path}: {str(e)}")
        
        # Validate data
        df = self._validate_data(df, file_path)
        
        self.logger.info(f"Successfully loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
        
        return df
    
    def _validate_data(self, df: pd.DataFrame, file_path: Path) -> pd.DataFrame:
        """Validate loaded data."""
        
        # Check for required columns
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            self.logger.info(f"Available columns: {list(df.columns)}")
            raise ValueError(f"Missing required columns in {file_path}: {missing_columns}")
        
        # Check data types and ranges
        for col in self.required_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                self.logger.warning(f"Column {col} is not numeric, attempting conversion")
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    raise ValueError(f"Cannot convert column {col} to numeric")
        
        # Check for missing values
        missing_data = df[self.required_columns].isnull().sum()
        if missing_data.sum() > 0:
            self.logger.warning("Missing values found:")
            for col, count in missing_data.items():
                if count > 0:
                    self.logger.warning(f"  {col}: {count} missing values")
            
            # Remove rows with missing values
            df_clean = df[self.required_columns].dropna()
            self.logger.info(f"Removed {len(df) - len(df_clean)} rows with missing values")
            df = df_clean
        
        # Check for normalized data (should be between 0 and 1)
        for col in self.required_columns:
            col_min, col_max = df[col].min(), df[col].max()
            if col_min < -0.1 or col_max > 1.1:  # Allow small tolerance
                self.logger.warning(f"Column {col} may not be properly normalized (min: {col_min:.3f}, max: {col_max:.3f})")
        
        # Check for infinite values
        inf_counts = np.isinf(df[self.required_columns]).sum().sum()
        if inf_counts > 0:
            self.logger.warning(f"Found {inf_counts} infinite values, removing affected rows")
            df = df[~np.isinf(df[self.required_columns]).any(axis=1)]
        
        # Basic statistics
        self.logger.info("Data validation summary:")
        self.logger.info(f"  Final shape: {df.shape}")
        self.logger.info(f"  GHI range: [{df['GHI norm'].min():.4f}, {df['GHI norm'].max():.4f}]")
        self.logger.info(f"  GHI mean: {df['GHI norm'].mean():.4f}")
        
        if len(df) < 100:
            self.logger.warning("Dataset has fewer than 100 samples, results may be unreliable")
        
        return df
    
    def get_feature_info(self) -> dict:
        """Get information about the features used in the model."""
        return {
            'required_columns': self.required_columns,
            'feature_descriptions': self.original_features,
            'input_features': self.required_columns[:-1],  # All except target
            'target_variable': self.required_columns[-1]   # GHI norm
        }
    
    def validate_test_data_compatibility(self, train_data: pd.DataFrame, 
                                       test_data: pd.DataFrame) -> bool:
        """
        Validate that test data is compatible with training data.
        
        Args:
            train_data: Training dataset
            test_data: Test dataset
            
        Returns:
            True if compatible, raises ValueError if not
        """
        
        # Check columns
        train_cols = set(train_data.columns)
        test_cols = set(test_data.columns)
        
        missing_in_test = train_cols - test_cols
        if missing_in_test:
            raise ValueError(f"Test data missing columns present in training data: {missing_in_test}")
        
        # Check data ranges (test data should be within reasonable bounds of training data)
        for col in self.required_columns:
            if col in train_data.columns and col in test_data.columns:
                train_min, train_max = train_data[col].min(), train_data[col].max()
                test_min, test_max = test_data[col].min(), test_data[col].max()
                
                # Allow some tolerance for extrapolation
                tolerance = 0.2 * (train_max - train_min)
                
                if test_min < (train_min - tolerance) or test_max > (train_max + tolerance):
                    self.logger.warning(
                        f"Test data for {col} extends beyond training range: "
                        f"train [{train_min:.3f}, {train_max:.3f}], "
                        f"test [{test_min:.3f}, {test_max:.3f}]"
                    )
        
        self.logger.info("Test data compatibility check passed")
        return True