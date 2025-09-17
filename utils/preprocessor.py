"""
Data Preprocessing Utilities
Handles feature extraction, data preparation, and preprocessing steps
"""

import numpy as np
import pandas as pd
from typing import Tuple, List


class DataPreprocessor:
    """Handles preprocessing of meteorological data for GHI forecasting."""
    
    def __init__(self, logger):
        """Initialize preprocessor."""
        self.logger = logger
        
        # Feature names based on the paper's feature selection results
        self.input_features = [
            'T norm', 'CDHI norm', 'CDNI norm', 'CGHI norm', 
            'DHI norm', 'DNI norm', 'RH norm', 'SZA norm'
        ]
        self.target_feature = 'GHI norm'
        
        # Feature descriptions from the paper
        self.feature_descriptions = {
            'T norm': 'Temperature (normalized)',
            'CDHI norm': 'Clearsky DHI (normalized)', 
            'CDNI norm': 'Clearsky DNI (normalized)',
            'CGHI norm': 'Clearsky GHI (normalized)',
            'DHI norm': 'Diffuse Horizontal Irradiance (normalized)',
            'DNI norm': 'Direct Normal Irradiance (normalized)',
            'RH norm': 'Relative Humidity (normalized)',
            'SZA norm': 'Solar Zenith Angle (normalized)',
            'GHI norm': 'Global Horizontal Irradiance (normalized) - Target'
        }
    
    def prepare_features_target(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features and target from dataset.
        
        Args:
            data: Input dataframe containing normalized features
            
        Returns:
            Tuple of (features, target) as numpy arrays
        """
        self.logger.info("Preparing features and target from dataset")
        
        # Validate required columns
        missing_features = [col for col in self.input_features if col not in data.columns]
        if missing_features:
            raise ValueError(f"Missing required feature columns: {missing_features}")
        
        if self.target_feature not in data.columns:
            raise ValueError(f"Missing target column: {self.target_feature}")
        
        # Extract features and target
        X = data[self.input_features].values
        y = data[self.target_feature].values
        
        # Basic validation
        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            self.logger.warning("NaN values found in features or target")
            
        if np.any(np.isinf(X)) or np.any(np.isinf(y)):
            self.logger.warning("Infinite values found in features or target")
        
        self.logger.info(f"Prepared features shape: {X.shape}")
        self.logger.info(f"Target shape: {y.shape}")
        self.logger.info(f"Feature range: [{X.min():.4f}, {X.max():.4f}]")
        self.logger.info(f"Target range: [{y.min():.4f}, {y.max():.4f}]")
        
        return X, y
    
    def get_feature_names(self) -> List[str]:
        """Get list of input feature names."""
        return self.input_features.copy()
    
    def get_feature_descriptions(self) -> dict:
        """Get feature descriptions."""
        return self.feature_descriptions.copy()
    
    def validate_data_quality(self, data: pd.DataFrame) -> dict:
        """
        Validate data quality and return summary statistics.
        
        Args:
            data: Input dataframe
            
        Returns:
            Dictionary with quality metrics
        """
        self.logger.info("Validating data quality")
        
        quality_report = {}
        
        # Check each feature
        for feature in self.input_features + [self.target_feature]:
            if feature not in data.columns:
                quality_report[feature] = {"status": "missing"}
                continue
            
            col_data = data[feature]
            
            feature_report = {
                "status": "ok",
                "count": len(col_data),
                "missing": col_data.isnull().sum(),
                "infinite": np.isinf(col_data).sum(),
                "mean": float(col_data.mean()),
                "std": float(col_data.std()),
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "q25": float(col_data.quantile(0.25)),
                "q75": float(col_data.quantile(0.75))
            }
            
            # Check normalization (data should be roughly between 0 and 1)
            if feature_report["min"] < -0.1 or feature_report["max"] > 1.1:
                feature_report["normalization_warning"] = True
                self.logger.warning(f"{feature} may not be properly normalized")
            
            # Check for outliers (values beyond 3 standard deviations)
            z_scores = np.abs((col_data - feature_report["mean"]) / feature_report["std"])
            outliers = (z_scores > 3).sum()
            feature_report["outliers"] = int(outliers)
            
            if outliers > len(col_data) * 0.05:  # More than 5% outliers
                feature_report["outlier_warning"] = True
                self.logger.warning(f"{feature} has {outliers} outliers ({outliers/len(col_data)*100:.1f}%)")
            
            quality_report[feature] = feature_report
        
        # Overall data quality
        total_samples = len(data)
        valid_samples = len(data.dropna(subset=self.input_features + [self.target_feature]))
        
        quality_report["overall"] = {
            "total_samples": total_samples,
            "valid_samples": valid_samples,
            "data_completeness": valid_samples / total_samples if total_samples > 0 else 0,
            "feature_count": len(self.input_features),
            "target_available": self.target_feature in data.columns
        }
        
        self.logger.info(f"Data quality validation completed:")
        self.logger.info(f"  Total samples: {total_samples}")
        self.logger.info(f"  Valid samples: {valid_samples}")
        self.logger.info(f"  Completeness: {quality_report['overall']['data_completeness']:.1%}")
        
        return quality_report
    
    def get_correlation_analysis(self, data: pd.DataFrame) -> dict:
        """
        Perform correlation analysis between features and target.
        
        Args:
            data: Input dataframe
            
        Returns:
            Dictionary with correlation results
        """
        self.logger.info("Performing correlation analysis")
        
        if self.target_feature not in data.columns:
            raise ValueError(f"Target column {self.target_feature} not found")
        
        correlations = {}
        
        # Calculate Spearman correlation (as used in the paper)
        for feature in self.input_features:
            if feature in data.columns:
                corr_coef = data[feature].corr(data[self.target_feature], method='spearman')
                correlations[feature] = {
                    'spearman_correlation': float(corr_coef),
                    'correlation_strength': self._classify_correlation_strength(abs(corr_coef))
                }
        
        # Sort by absolute correlation
        sorted_features = sorted(
            correlations.items(), 
            key=lambda x: abs(x[1]['spearman_correlation']), 
            reverse=True
        )
        
        correlation_report = {
            'correlations': correlations,
            'sorted_by_strength': sorted_features,
            'strong_correlations': [f for f, c in correlations.items() 
                                  if abs(c['spearman_correlation']) > 0.7],
            'weak_correlations': [f for f, c in correlations.items() 
                                if abs(c['spearman_correlation']) < 0.3]
        }
        
        self.logger.info("Correlation analysis completed:")
        self.logger.info(f"  Strong correlations: {len(correlation_report['strong_correlations'])}")
        self.logger.info(f"  Weak correlations: {len(correlation_report['weak_correlations'])}")
        
        return correlation_report
    
    def _classify_correlation_strength(self, corr_abs: float) -> str:
        """Classify correlation strength based on absolute value."""
        if corr_abs >= 0.7:
            return "strong"
        elif corr_abs >= 0.5:
            return "moderate"
        elif corr_abs >= 0.3:
            return "weak"
        else:
            return "very_weak"
    
    def create_data_summary(self, data: pd.DataFrame) -> dict:
        """
        Create comprehensive data summary.
        
        Args:
            data: Input dataframe
            
        Returns:
            Dictionary with data summary
        """
        quality_report = self.validate_data_quality(data)
        correlation_report = self.get_correlation_analysis(data)
        
        summary = {
            'dataset_info': {
                'shape': data.shape,
                'features': self.input_features,
                'target': self.target_feature,
                'memory_usage_mb': data.memory_usage(deep=True).sum() / (1024 * 1024)
            },
            'quality_report': quality_report,
            'correlation_analysis': correlation_report,
            'feature_descriptions': self.feature_descriptions
        }
        
        return summary