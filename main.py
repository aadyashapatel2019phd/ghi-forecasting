#!/usr/bin/env python3
"""
Seasonal GHI Forecasting for Stand-Alone Photovoltaic Systems
Main execution script for training and testing machine learning models

Authors: Aadyasha Patel, O. V. Gnana Swathika
Institution: Vellore Institute of Technology, Chennai, India
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator
from utils.data_loader import DataLoader
from utils.config import Config
from utils.logger import setup_logger


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Seasonal GHI Forecasting Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a new GPR model
  python main.py --model GPR --mode train --data data/training_data.xlsx
  
  # Use pre-trained model to test on new location
  python main.py --model GPR --mode test --pretrained models/trained_model.pkl --data data/test_data.xlsx
  
  # Train and immediately test
  python main.py --model RT --mode train_test --data data/training_data.xlsx --test_data data/test_data.xlsx
        """
    )
    
    parser.add_argument(
        "--model", 
        choices=["ELR", "GPR", "RT"],
        required=True,
        help="Model type: ELR (Efficient Linear Regression), GPR (Gaussian Process Regression), RT (Regression Trees)"
    )
    
    parser.add_argument(
        "--mode",
        choices=["train", "test", "train_test"],
        required=True,
        help="Execution mode: train (train new model), test (use pretrained model), train_test (train then test)"
    )
    
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to training data file (Excel format)"
    )
    
    parser.add_argument(
        "--test_data",
        type=str,
        help="Path to test data file (required for test and train_test modes)"
    )
    
    parser.add_argument(
        "--pretrained",
        type=str,
        help="Path to pre-trained model file (required for test mode)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for results and models (default: results)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file (default: config/config.yaml)"
    )
    
    parser.add_argument(
        "--optimize_hyperparams",
        action="store_true",
        help="Perform hyperparameter optimization (default: use predefined hyperparameters)"
    )
    
    parser.add_argument(
        "--n_trials",
        type=int,
        default=60,
        help="Number of optimization trials (default: 60)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def validate_arguments(args):
    """Validate command line arguments."""
    errors = []
    
    # Check file paths
    if not os.path.exists(args.data):
        errors.append(f"Training data file not found: {args.data}")
    
    if args.mode in ["test", "train_test"] and args.test_data and not os.path.exists(args.test_data):
        errors.append(f"Test data file not found: {args.test_data}")
    
    if args.mode == "test" and not args.pretrained:
        errors.append("Pre-trained model path is required for test mode")
    
    if args.pretrained and not os.path.exists(args.pretrained):
        errors.append(f"Pre-trained model file not found: {args.pretrained}")
    
    # Mode-specific validations
    if args.mode == "train_test" and not args.test_data:
        errors.append("Test data is required for train_test mode")
    
    if errors:
        print("Validation errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)


def main():
    """Main execution function."""
    args = parse_arguments()
    validate_arguments(args)
    
    # Setup logging
    logger = setup_logger(
        name="ghi_forecasting",
        log_file=Path(args.output_dir) / "logs" / "main.log",
        level=logging.DEBUG if args.verbose else logging.INFO
    )
    
    logger.info("=" * 60)
    logger.info("Starting Seasonal GHI Forecasting")
    logger.info(f"Model: {args.model}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Data: {args.data}")
    logger.info("=" * 60)
    
    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(Path(args.output_dir) / "models", exist_ok=True)
        os.makedirs(Path(args.output_dir) / "logs", exist_ok=True)
        
        # Load configuration
        config = Config(args.config if os.path.exists(args.config) else None)
        
        # Initialize components
        data_loader = DataLoader(logger)
        model_trainer = ModelTrainer(config, logger)
        model_evaluator = ModelEvaluator(config, logger)
        
        # Execute based on mode
        if args.mode == "train":
            logger.info("Training new model...")
            
            # Load training data
            train_data = data_loader.load_data(args.data)
            logger.info(f"Loaded training data: {train_data.shape}")
            
            # Train model
            model, metrics = model_trainer.train_model(
                model_type=args.model,
                data=train_data,
                optimize_hyperparams=args.optimize_hyperparams,
                n_trials=args.n_trials
            )
            
            # Save model
            model_path = Path(args.output_dir) / "models" / f"{args.model}_trained_model.pkl"
            model_trainer.save_model(model, model_path)
            logger.info(f"Model saved to: {model_path}")
            
            # Display training results
            print(f"\nTraining completed successfully!")
            print(f"Model saved to: {model_path}")
            print(f"Training RMSE: {metrics.get('rmse', 'N/A'):.4f}")
            print(f"Training R²: {metrics.get('r2', 'N/A'):.4f}")
            
        elif args.mode == "test":
            logger.info("Testing with pre-trained model...")
            
            # Load test data
            test_data = data_loader.load_data(args.test_data)
            logger.info(f"Loaded test data: {test_data.shape}")
            
            # Load pre-trained model
            model = model_trainer.load_model(args.pretrained)
            logger.info(f"Loaded pre-trained model from: {args.pretrained}")
            
            # Evaluate model
            results = model_evaluator.evaluate_model(model, test_data, args.model)
            
            # Save results
            results_path = Path(args.output_dir) / f"{args.model}_test_results.json"
            model_evaluator.save_results(results, results_path)
            
            # Display results
            print(f"\nTesting completed successfully!")
            print(f"Results saved to: {results_path}")
            print(f"Test RMSE: {results['metrics']['rmse']:.4f}")
            print(f"Test MAE: {results['metrics']['mae']:.4f}")
            print(f"Test R²: {results['metrics']['r2']:.4f}")
            
        elif args.mode == "train_test":
            logger.info("Training new model and testing...")
            
            # Load data
            train_data = data_loader.load_data(args.data)
            test_data = data_loader.load_data(args.test_data)
            logger.info(f"Loaded training data: {train_data.shape}")
            logger.info(f"Loaded test data: {test_data.shape}")
            
            # Train model
            model, train_metrics = model_trainer.train_model(
                model_type=args.model,
                data=train_data,
                optimize_hyperparams=args.optimize_hyperparams,
                n_trials=args.n_trials
            )
            
            # Save model
            model_path = Path(args.output_dir) / "models" / f"{args.model}_trained_model.pkl"
            model_trainer.save_model(model, model_path)
            
            # Test model
            test_results = model_evaluator.evaluate_model(model, test_data, args.model)
            
            # Save results
            results_path = Path(args.output_dir) / f"{args.model}_train_test_results.json"
            combined_results = {
                "training_metrics": train_metrics,
                "test_results": test_results
            }
            model_evaluator.save_results(combined_results, results_path)
            
            # Display results
            print(f"\nTraining and testing completed successfully!")
            print(f"Model saved to: {model_path}")
            print(f"Results saved to: {results_path}")
            print(f"\nTraining Results:")
            print(f"  RMSE: {train_metrics.get('rmse', 'N/A'):.4f}")
            print(f"  R²: {train_metrics.get('r2', 'N/A'):.4f}")
            print(f"\nTest Results:")
            print(f"  RMSE: {test_results['metrics']['rmse']:.4f}")
            print(f"  MAE: {test_results['metrics']['mae']:.4f}")
            print(f"  R²: {test_results['metrics']['r2']:.4f}")
        
        logger.info("Execution completed successfully")
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}", exc_info=True)
        print(f"\nError: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()