"""
Model Evaluation Module
Handles evaluation of trained models on test data with comprehensive metrics and visualizations
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Tuple
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from utils.preprocessor import DataPreprocessor


class ModelEvaluator:
    """Handles evaluation of trained models on test data."""
    
    def __init__(self, config, logger):
        """Initialize model evaluator."""
        self.config = config
        self.logger = logger
        self.preprocessor = DataPreprocessor(logger)
    
    def evaluate_model(self, model, test_data: pd.DataFrame, model_type: str) -> Dict[str, Any]:
        """
        Evaluate a trained model on test data.
        
        Args:
            model: Trained model
            test_data: Test dataset
            model_type: Type of model for documentation
            
        Returns:
            Dictionary containing evaluation results
        """
        self.logger.info(f"Starting evaluation of {model_type} model")
        
        # Prepare test data
        X_test, y_test = self.preprocessor.prepare_features_target(test_data)
        
        # Make predictions
        if hasattr(model, 'scaler') and model.scaler is not None:
            X_test_scaled = model.scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled)
        else:
            y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred)
        
        # Perform diagnostic validation
        diagnostics = self._diagnostic_validation(y_test, y_pred)
        
        # Create visualizations
        plots = self._create_evaluation_plots(y_test, y_pred, model_type)
        
        # Compile results
        results = {
            'model_type': model_type,
            'model_hyperparameters': getattr(model, 'hyperparameters', {}),
            'metrics': metrics,
            'diagnostics': diagnostics,
            'plots': plots,
            'test_data_info': {
                'n_samples': len(y_test),
                'features': getattr(model, 'feature_names', []),
                'target_statistics': {
                    'mean': float(np.mean(y_test)),
                    'std': float(np.std(y_test)),
                    'min': float(np.min(y_test)),
                    'max': float(np.max(y_test))
                }
            }
        }
        
        self.logger.info(f"Evaluation completed. RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")
        
        return results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics."""
        
        metrics = {
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'r2': float(r2_score(y_true, y_pred)),
            'mse': float(mean_squared_error(y_true, y_pred)),
            'mape': float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),  # Mean Absolute Percentage Error
            'max_error': float(np.max(np.abs(y_true - y_pred))),
            'explained_variance': float(1 - (np.var(y_true - y_pred) / np.var(y_true)))
        }
        
        return metrics
    
    def _diagnostic_validation(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Perform diagnostic validation similar to the paper."""
        
        residuals = y_true - y_pred
        
        # Test for normality of residuals
        normality_stat, normality_p = stats.normaltest(residuals)
        
        # Test for homoscedasticity (Breusch-Pagan test approximation)
        correlation_coeff, correlation_p = stats.pearsonr(y_pred, np.abs(residuals))
        
        # Calculate residual statistics
        residual_stats = {
            'mean': float(np.mean(residuals)),
            'std': float(np.std(residuals)),
            'skewness': float(stats.skew(residuals)),
            'kurtosis': float(stats.kurtosis(residuals)),
            'min': float(np.min(residuals)),
            'max': float(np.max(residuals))
        }
        
        diagnostics = {
            'residual_statistics': residual_stats,
            'normality_test': {
                'statistic': float(normality_stat),
                'p_value': float(normality_p),
                'is_normal': normality_p > 0.05
            },
            'homoscedasticity_test': {
                'correlation_with_predictions': float(correlation_coeff),
                'p_value': float(correlation_p),
                'is_homoscedastic': abs(correlation_coeff) < 0.1
            }
        }
        
        return diagnostics
    
    def _create_evaluation_plots(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               model_type: str) -> Dict[str, str]:
        """Create evaluation plots and return their file paths."""
        
        plots = {}
        residuals = y_true - y_pred
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Actual vs Predicted Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_true, y_pred, alpha=0.6, s=20)
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        ax.set_xlabel('Actual GHI')
        ax.set_ylabel('Predicted GHI')
        ax.set_title(f'{model_type}: Actual vs Predicted GHI')
        ax.grid(True, alpha=0.3)
        
        # Add R² annotation
        r2 = r2_score(y_true, y_pred)
        ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes, 
                bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
        
        plot_path = Path('results/plots/actual_vs_predicted.png')
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots['actual_vs_predicted'] = str(plot_path)
        
        # 2. Residual Scatter Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(y_pred, residuals, alpha=0.6, s=20)
        ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Predicted GHI')
        ax.set_ylabel('Residuals')
        ax.set_title(f'{model_type}: Residual Scatter Plot')
        ax.grid(True, alpha=0.3)
        
        plot_path = Path('results/plots/residual_scatter.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots['residual_scatter'] = str(plot_path)
        
        # 3. Histogram of Residuals
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(residuals, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Overlay normal distribution
        mu, sigma = np.mean(residuals), np.std(residuals)
        x = np.linspace(residuals.min(), residuals.max(), 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal Distribution')
        
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Density')
        ax.set_title(f'{model_type}: Histogram of Residuals')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plot_path = Path('results/plots/residual_histogram.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots['residual_histogram'] = str(plot_path)
        
        # 4. Q-Q Plot of Residuals
        fig, ax = plt.subplots(figsize=(8, 6))
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title(f'{model_type}: Q-Q Plot of Residuals')
        ax.grid(True, alpha=0.3)
        
        plot_path = Path('results/plots/qq_plot.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plots['qq_plot'] = str(plot_path)
        
        # 5. Time Series Plot (if applicable)
        if len(y_true) > 100:  # Only create if we have enough data points
            fig, ax = plt.subplots(figsize=(12, 6))
            indices = np.arange(len(y_true))
            ax.plot(indices, y_true, label='Actual', alpha=0.7, linewidth=1)
            ax.plot(indices, y_pred, label='Predicted', alpha=0.7, linewidth=1)
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('GHI Value')
            ax.set_title(f'{model_type}: Time Series Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plot_path = Path('results/plots/time_series_comparison.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plots['time_series_comparison'] = str(plot_path)
        
        return plots
    
    def save_results(self, results: Dict[str, Any], filepath: Path):
        """Save evaluation results to JSON file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4, default=str)
        
        self.logger.info(f"Results saved to: {filepath}")
    
    def create_comparison_report(self, results_list: list, output_path: Path):
        """Create a comparison report for multiple models."""
        
        if not results_list:
            self.logger.warning("No results provided for comparison report")
            return
        
        # Create comparison table
        comparison_data = []
        for result in results_list:
            model_data = {
                'Model': result['model_type'],
                'RMSE': result['metrics']['rmse'],
                'MAE': result['metrics']['mae'],
                'R²': result['metrics']['r2'],
                'MAPE': result['metrics']['mape'],
                'Max Error': result['metrics']['max_error']
            }
            comparison_data.append(model_data)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create comparison visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        # RMSE comparison
        axes[0, 0].bar(comparison_df['Model'], comparison_df['RMSE'], color='skyblue')
        axes[0, 0].set_title('Root Mean Square Error')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # MAE comparison
        axes[0, 1].bar(comparison_df['Model'], comparison_df['MAE'], color='lightgreen')
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # R² comparison
        axes[1, 0].bar(comparison_df['Model'], comparison_df['R²'], color='salmon')
        axes[1, 0].set_title('Coefficient of Determination')
        axes[1, 0].set_ylabel('R²')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # MAPE comparison
        axes[1, 1].bar(comparison_df['Model'], comparison_df['MAPE'], color='gold')
        axes[1, 1].set_title('Mean Absolute Percentage Error')
        axes[1, 1].set_ylabel('MAPE (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        plot_path = output_path.parent / 'model_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save comparison data
        comparison_df.to_csv(output_path.parent / 'model_comparison.csv', index=False)
        
        # Create detailed report
        report = {
            'comparison_summary': comparison_df.to_dict('records'),
            'best_model': {
                'lowest_rmse': comparison_df.loc[comparison_df['RMSE'].idxmin(), 'Model'],
                'lowest_mae': comparison_df.loc[comparison_df['MAE'].idxmin(), 'Model'],
                'highest_r2': comparison_df.loc[comparison_df['R²'].idxmax(), 'Model']
            },
            'detailed_results': results_list
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=4, default=str)
        
        self.logger.info(f"Comparison report saved to: {output_path}")
        
        return report