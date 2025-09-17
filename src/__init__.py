"""
GHI Forecasting Package
Seasonal Global Horizontal Irradiance forecasting for Stand-Alone Photovoltaic Systems
"""

__version__ = "1.0.0"
__author__ = "Aadyasha Patel, O. V. Gnana Swathika"
__email__ = "gnanaswathika.ov@vit.ac.in"

# src/utils/__init__.py content would be:
# """Utilities package for GHI forecasting."""

# Core modules
from .model_trainer import ModelTrainer
from .model_evaluator import ModelEvaluator

# Utilities  
from .utils.data_loader import DataLoader
from .utils.preprocessor import DataPreprocessor
from .utils.config import Config
from .utils.logger import setup_logger

__all__ = [
    "ModelTrainer",
    "ModelEvaluator", 
    "DataLoader",
    "DataPreprocessor",
    "Config",
    "setup_logger"
]