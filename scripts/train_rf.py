#!/usr/bin/env python

import argparse
import logging
import sys
from pathlib import Path
import time
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from src.config.config import Config
from src.data.data_pipeline import DataPipeline
from src.models.random_forest import StockReturnRF

def setup_logging(config: Config) -> logging.Logger:
    """Set up logging configuration."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = config.logs_dir / f"train_rf_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Random Forest Model')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--start-year', type=int, help='Start year for training')
    parser.add_argument('--end-year', type=int, help='End year for training')
    parser.add_argument('--n-jobs', type=int, default=-1, help='Number of jobs for parallel processing')
    return parser.parse_args()

def main():
    """Main training script."""
    args = parse_args()
    
    # Load configuration
    config = Config(args.config if args.config else None)
    logger = setup_logging(config)
    
    # Initialize data pipeline
    try:
        pipeline = DataPipeline(config.data.file_path)
        logger.info("Data pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize data pipeline: {str(e)}")
        sys.exit(1)
    
    # Set year range
    start_year = args.start_year if args.start_year else config.data.min_year
    end_year = args.end_year if args.end_year else config.data.max_year
    
    # Initialize model
    model = StockReturnRF(model_path=str(config.models_dir))
    
    # Train models for each year
    for year in range(start_year, end_year + 1):
        logger.info(f"Starting training for year {year}")
        start_time = time.time()
        
        try:
            # Load data
            X_train, y_train, X_test, y_test = pipeline.load_train_test(
                year - config.analysis.test_period,
                config.analysis.test_period
            )
            logger.info(f"Data loaded for year {year}")
            
            # Tune hyperparameters
            logger.info("Starting hyperparameter tuning")
            model.tune_hyperparameters(
                X=X_train,
                y=y_train,
                train_indices=range(len(X_train) - len(X_test)),
                validation_indices=range(len(X_train) - len(X_test), len(X_train))
            )
            
            # Train final model
            model.train(X_train, y_train)
            
            # Save model and feature importance
            model.save_model(year)
            
            # Log results
            end_time = time.time()
            training_time = end_time - start_time
            logger.info(f"Year {year} training completed in {training_time:.2f} seconds")
            logger.info(f"Best parameters: {model.best_params}")
            
            # Log feature importance
            importance_df = model.get_feature_importance()
            logger.info("Top 10 most important features:")
            logger.info(importance_df.head(10).to_string())
            
        except Exception as e:
            logger.error(f"Error training model for year {year}: {str(e)}")
            continue

if __name__ == "__main__":
    main()