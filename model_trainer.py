"""
Model Training Module
Handles training of ELR, GPR, and RT models with hyperparameter optimization
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Model imports
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import LinearSVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ConstantKernel
from sklearn.tree import DecisionTreeRegressor

from utils.preprocessor import DataPreprocessor


class ModelTrainer:
    """Handles training of machine learning models for GHI forecasting."""
    
    def __init__(self, config, logger):
        """Initialize model trainer."""
        self.config = config
        self.logger = logger
        self.preprocessor = DataPreprocessor(logger)
        
        # Default hyperparameters from the paper
        self.default_hyperparams = {
            'ELR': {
                'learner': 'least_squares',
                'regularization': 'ridge',
                'lambda': 0.09694,
                'beta_tolerance': 0.0001,
                'standardize': False
            },
            'RT': {
                'min_samples_leaf': 3,
                'standardize': False
            },
            'GPR': {
                'basis_function': 'linear',
                'kernel_function': 'Isotropic Matern 5/2',
                'kernel_scale': 5.9895,
                'sigma': 0.00010961,
                'standardize': False
            }
        }
    
    def train_model(self, model_type: str, data: pd.DataFrame, 
                   optimize_hyperparams: bool = False, n_trials: int = 60) -> Tuple[Any, Dict]:
        """
        Train a model with the specified type.
        
        Args:
            model_type: Type of model ('ELR', 'GPR', 'RT')
            data: Training data
            optimize_hyperparams: Whether to optimize hyperparameters
            n_trials: Number of optimization trials
            
        Returns:
            Tuple of (trained_model, training_metrics)
        """
        self.logger.info(f"Starting training for {model_type} model")
        
        # Preprocess data
        X, y = self.preprocessor.prepare_features_target(data)
        
        if optimize_hyperparams:
            self.logger.info("Performing hyperparameter optimization...")
            best_params = self._optimize_hyperparameters(model_type, X, y, n_trials)
        else:
            self.logger.info("Using default hyperparameters from paper")
            best_params = self.default_hyperparams[model_type].copy()
        
        # Train final model with best/default parameters
        model = self._build_model(model_type, best_params)
        
        # Apply standardization if needed
        if best_params.get('standardize', False):
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model.fit(X_scaled, y)
            # Store scaler with model for later use
            model.scaler = scaler
        else:
            model.fit(X, y)
            model.scaler = None
        
        # Calculate training metrics
        if hasattr(model, 'scaler') and model.scaler is not None:
            y_pred = model.predict(model.scaler.transform(X))
        else:
            y_pred = model.predict(X)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred)
        }
        
        # Store metadata
        model.model_type = model_type
        model.hyperparameters = best_params
        model.feature_names = self.preprocessor.get_feature_names()
        
        self.logger.info(f"Training completed. RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")
        
        return model, metrics
    
    def _optimize_hyperparameters(self, model_type: str, X: np.ndarray, y: np.ndarray, 
                                n_trials: int) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna."""
        
        def objective(trial):
            try:
                params = self._suggest_hyperparameters(trial, model_type)
                model = self._build_model(model_type, params)
                
                # Apply standardization if needed
                X_use = X
                if params.get('standardize', False):
                    scaler = StandardScaler()
                    X_use = scaler.fit_transform(X)
                
                # Use cross-validation for more robust evaluation
                cv_scores = cross_val_score(model, X_use, y, cv=5, 
                                          scoring='neg_mean_squared_error', n_jobs=-1)
                rmse = np.sqrt(-cv_scores.mean())
                
                return rmse
                
            except Exception as e:
                self.logger.warning(f"Trial failed: {str(e)}")
                return float('inf')
        
        # Create study and optimize
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        if study.best_trial.value == float('inf'):
            self.logger.warning("All optimization trials failed. Using default parameters.")
            return self.default_hyperparams[model_type].copy()
        
        best_params = study.best_params
        self.logger.info(f"Best hyperparameters: {best_params}")
        self.logger.info(f"Best RMSE: {study.best_value:.4f}")
        
        return best_params
    
    def _suggest_hyperparameters(self, trial, model_type: str) -> Dict[str, Any]:
        """Suggest hyperparameters for optimization based on model type."""
        
        if model_type == 'ELR':
            return {
                'learner': trial.suggest_categorical('learner', ['least_squares', 'svm']),
                'regularization': trial.suggest_categorical('regularization', ['ridge', 'lasso']),
                'lambda': trial.suggest_float('lambda', 2.7902e-9, 27.9018, log=True),
                'beta_tolerance': 0.0001,
                'standardize': trial.suggest_categorical('standardize', [True, False])
            }
        
        elif model_type == 'RT':
            return {
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 1792),
                'standardize': trial.suggest_categorical('standardize', [True, False])
            }
        
        elif model_type == 'GPR':
            return {
                'basis_function': trial.suggest_categorical('basis_function', 
                                                          ['constant', 'zero', 'linear']),
                'kernel_function': trial.suggest_categorical('kernel_function', [
                    'Nonisotropic Exponential', 'Nonisotropic Matern 3/2', 'Nonisotropic Matern 5/2',
                    'Nonisotropic Rational Quadratic', 'Nonisotropic Squared Exponential',
                    'Isotropic Exponential', 'Isotropic Matern 3/2', 'Isotropic Matern 5/2',
                    'Isotropic Rational Quadratic', 'Isotropic Squared Exponential'
                ]),
                'kernel_scale': trial.suggest_float('kernel_scale', 0.0001, 1000, log=True),
                'sigma': trial.suggest_float('sigma', 0.0001, 2.665, log=True),
                'standardize': trial.suggest_categorical('standardize', [True, False])
            }
    
    def _build_model(self, model_type: str, params: Dict[str, Any]):
        """Build model with specified parameters."""
        
        if model_type == 'ELR':
            return self._build_elr_model(params)
        elif model_type == 'RT':
            return self._build_rt_model(params)
        elif model_type == 'GPR':
            return self._build_gpr_model(params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _build_elr_model(self, params: Dict[str, Any]):
        """Build Efficient Linear Regression model."""
        
        if params['learner'] == 'least_squares':
            if params['regularization'] == 'ridge':
                return Ridge(
                    alpha=params['lambda'],
                    tol=params['beta_tolerance'],
                    solver='auto',
                    random_state=42
                )
            else:  # lasso
                return Lasso(
                    alpha=params['lambda'],
                    tol=params['beta_tolerance'],
                    random_state=42
                )
        else:  # svm
            C = 1 / params['lambda'] if params['lambda'] > 0 else 1e5
            return LinearSVR(
                C=C,
                tol=params['beta_tolerance'],
                random_state=42,
                max_iter=10000
            )
    
    def _build_rt_model(self, params: Dict[str, Any]):
        """Build Regression Tree model."""
        return DecisionTreeRegressor(
            min_samples_leaf=params['min_samples_leaf'],
            splitter='best',
            random_state=42
        )
    
    def _build_gpr_model(self, params: Dict[str, Any]):
        """Build Gaussian Process Regression model."""
        
        # Signal variance (from paper)
        signal_std = 0.18845
        signal_variance = signal_std ** 2
        
        kernel_scale = params['kernel_scale']
        kernel_name = params['kernel_function']
        
        # Choose base kernel
        if "Exponential" in kernel_name:
            base_kernel = RBF(length_scale=kernel_scale)
        elif "Matern 3/2" in kernel_name:
            base_kernel = Matern(length_scale=kernel_scale, nu=1.5)
        elif "Matern 5/2" in kernel_name:
            base_kernel = Matern(length_scale=kernel_scale, nu=2.5)
        elif "Rational Quadratic" in kernel_name:
            base_kernel = RationalQuadratic(length_scale=kernel_scale)
        elif "Squared Exponential" in kernel_name:
            base_kernel = RBF(length_scale=kernel_scale)
        else:
            base_kernel = RBF(length_scale=kernel_scale)
        
        kernel = ConstantKernel(constant_value=signal_variance) * base_kernel
        
        return GaussianProcessRegressor(
            kernel=kernel,
            alpha=params['sigma'] ** 2,
            optimizer=None,
            normalize_y=True
        )
    
    def save_model(self, model, filepath: Path):
        """Save trained model to file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        
        self.logger.info(f"Model saved to: {filepath}")
    
    def load_model(self, filepath: Path):
        """Load trained model from file."""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        self.logger.info(f"Model loaded from: {filepath}")
        return model