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
from src.models.neural_network import StockReturnNN

def setup_logging(config: Config) -> logging.Logger:
    """Set up logging configuration."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = config.logs_dir / f"train_nn_{timestamp}.log"
    
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
    parser = argparse.ArgumentParser(description='Train Neural Network Model')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--start-year', type=int, help='Start year for training')
    parser.add_argument('--end-year', type=int, help='End year for training')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device number')
    return parser.parse_args()

def main():
    """Main training script."""
    args = parse_args()
    
    # Load configuration
    config = Config(args.config if args.config else None)
    logger = setup_logging(config)
    
    # Set up GPU
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
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
            
            # Initialize and train model
            model = StockReturnNN(input_dim=len(pipeline.features))
            training_history = model.train(
                X_train=X_train,
                y_train=y_train,
                X_val=X_test,
                y_val=y_test,
                year=year
            )
            
            # Save model
            model.save_model(year)
            
            # Log training results
            end_time = time.time()
            training_time = end_time - start_time
            logger.info(f"Year {year} training completed in {training_time:.2f} seconds")
            logger.info(f"Best hyperparameters: {training_history['best_hyperparameters']}")
            logger.info(f"Final validation R2: {training_history['history']['val_r2_metrics'][-1]:.4f}")
            
        except Exception as e:
            logger.error(f"Error training model for year {year}: {str(e)}")
            continue

if __name__ == "__main__":
    main()