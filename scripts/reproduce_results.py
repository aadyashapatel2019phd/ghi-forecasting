#!/usr/bin/env python3
"""
Results Reproduction Script
Reproduces the main results from the paper using pre-trained models and test data
"""

import argparse
import sys
import os
import json
import pandas as pd
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from model_trainer import ModelTrainer
from model_evaluator import ModelEvaluator
from utils.data_loader import DataLoader
from utils.config import Config
from utils.logger import setup_logger


def reproduce_paper_results():
    """Reproduce the main results table from the paper."""
    
    # Setup
    logger = setup_logger("reproduce_results", log_file=Path("results/reproduction.log"))
    config = Config("config/config.yaml")
    
    data_loader = DataLoader(logger)
    trainer = ModelTrainer(config, logger)
    evaluator = ModelEvaluator(config, logger)
    
    # Model paths (these should be created from the original training data)
    model_paths = {
        'ELR': 'models/elr_trained.pkl',
        'RT': 'models/rt_trained.pkl', 
        'GPR': 'models/gpr_trained.pkl'
    }
    
    # Test data path
    test_data_path = 'data/testdata.xlsx'
    
    if not os.path.exists(test_data_path):
        logger.error(f"Test data not found: {test_data_path}")
        logger.info("Please ensure the test data (testdata.xlsx) is in the data/ directory")
        return
    
    logger.info("Loading test data...")
    test_data = data_loader.load_data(test_data_path)
    
    # Test each model
    results = {}
    
    for model_name, model_path in model_paths.items():
        logger.info(f"Testing {model_name} model...")
        
        if not os.path.exists(model_path):
            logger.warning(f"Pre-trained model not found: {model_path}")
            logger.info(f"Skipping {model_name} model")
            continue
            
        # Load model and evaluate
        try:
            model = trainer.load_model(model_path)
            result = evaluator.evaluate_model(model, test_data, model_name)
            results[model_name] = result
            
            logger.info(f"{model_name} Results:")
            logger.info(f"  RMSE: {result['metrics']['rmse']:.4f}")
            logger.info(f"  MAE: {result['metrics']['mae']:.4f}")
            logger.info(f"  R²: {result['metrics']['r2']:.4f}")
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {str(e)}")
            continue
    
    if not results:
        logger.error("No models could be evaluated. Please check model files.")
        return
    
    # Create comparison table
    comparison_data = []
    for model_name, result in results.items():
        comparison_data.append({
            'Model': model_name,
            'RMSE': result['metrics']['rmse'],
            'MAE': result['metrics']['mae'],
            'R²': result['metrics']['r2'],
            'MAPE': result['metrics']['mape']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save results
    output_dir = Path("results/reproduction")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save comparison table
    comparison_df.to_csv(output_dir / "model_comparison_delhi.csv", index=False)
    
    # Save detailed results
    with open(output_dir / "detailed_results.json", 'w') as f:
        json.dump(results, f, indent=4, default=str)
    
    # Create comparison report
    evaluator.create_comparison_report(list(results.values()), output_dir / "comparison_report.json")
    
    # Print results table
    print("\n" + "="*60)
    print("REPRODUCTION RESULTS - DELHI TEST DATA")
    print("="*60)
    print(comparison_df.to_string(index=False, float_format='%.4f'))
    print("="*60)
    
    # Compare with paper results (expected values)
    paper_results = {
        'ELR': {'RMSE': 0.1070, 'MAE': 0.0864, 'R²': 0.8135},
        'RT': {'RMSE': 0.0128, 'MAE': 0.0077, 'R²': 0.9973}, 
        'GPR': {'RMSE': 0.0030, 'MAE': 0.0022, 'R²': 0.9999}
    }
    
    print("\nCOMPARISON WITH PAPER RESULTS:")
    print("-"*60)
    for model in comparison_df['Model']:
        if model in paper_results:
            actual = comparison_df[comparison_df['Model'] == model].iloc[0]
            expected = paper_results[model]
            
            print(f"\n{model} Model:")
            print(f"  RMSE - Paper: {expected['RMSE']:.4f}, Actual: {actual['RMSE']:.4f}")
            print(f"  MAE  - Paper: {expected['MAE']:.4f}, Actual: {actual['MAE']:.4f}")
            print(f"  R²   - Paper: {expected['R²']:.4f}, Actual: {actual['R²']:.4f}")
    
    logger.info(f"Results saved to: {output_dir}")
    print(f"\nDetailed results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce paper results")
    parser.add_argument("--output", default="results/reproduction", help="Output directory")
    
    args = parser.parse_args()
    reproduce_paper_results()