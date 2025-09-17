"""
Configuration Management
Handles loading and management of configuration settings
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """Configuration manager for the GHI forecasting application."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration."""
        self.config_data = self._load_default_config()
        
        if config_path and os.path.exists(config_path):
            self._load_config_file(config_path)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration values."""
        return {
            'data': {
                'required_columns': [
                    'T norm', 'CDHI norm', 'CDNI norm', 'CGHI norm', 
                    'DHI norm', 'DNI norm', 'RH norm', 'SZA norm', 'GHI norm'
                ],
                'input_features': [
                    'T norm', 'CDHI norm', 'CDNI norm', 'CGHI norm', 
                    'DHI norm', 'DNI norm', 'RH norm', 'SZA norm'
                ],
                'target_column': 'GHI norm',
                'test_size': 0.2,
                'random_state': 42
            },
            'models': {
                'ELR': {
                    'default_params': {
                        'learner': 'least_squares',
                        'regularization': 'ridge', 
                        'lambda': 0.09694,
                        'beta_tolerance': 0.0001,
                        'standardize': False
                    },
                    'optimization_ranges': {
                        'lambda': {'min': 2.7902e-9, 'max': 27.9018, 'log': True},
                        'learner': ['least_squares', 'svm'],
                        'regularization': ['ridge', 'lasso'],
                        'standardize': [True, False]
                    }
                },
                'RT': {
                    'default_params': {
                        'min_samples_leaf': 3,
                        'standardize': False
                    },
                    'optimization_ranges': {
                        'min_samples_leaf': {'min': 1, 'max': 1792},
                        'standardize': [True, False]
                    }
                },
                'GPR': {
                    'default_params': {
                        'basis_function': 'linear',
                        'kernel_function': 'Isotropic Matern 5/2',
                        'kernel_scale': 5.9895,
                        'sigma': 0.00010961,
                        'standardize': False
                    },
                    'optimization_ranges': {
                        'basis_function': ['constant', 'zero', 'linear'],
                        'kernel_function': [
                            'Nonisotropic Exponential', 'Nonisotropic Matern 3/2', 
                            'Nonisotropic Matern 5/2', 'Nonisotropic Rational Quadratic',
                            'Nonisotropic Squared Exponential', 'Isotropic Exponential',
                            'Isotropic Matern 3/2', 'Isotropic Matern 5/2',