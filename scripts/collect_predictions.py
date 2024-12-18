#!/usr/bin/env python
# scripts/collect_predictions.py

import argparse
import sys
from pathlib import Path
import logging

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.config.config import Config
from src.data.data_pipeline import DataPipeline
from src.analysis.prediction_collector import PredictionCollector
from src.utils.metrics import r2_metrics_tf

def setup_logging() -> logging.Logger:
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('prediction_collection.log')
        ]
    )
    return logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Collect Model Predictions')
    parser.add_argument('--start-year', type=int, default=1989, 
                       help='Start year for predictions (default: 1989)')
    parser.add_argument('--end-year', type=int, default=2021,
                       help='End year for predictions (default: 2021)')
    parser.add_argument('--output-name', type=str, default='oos_predictions',
                       help='Base name for output file (default: oos_predictions)')
    return parser.parse_args()

def main():
    """Main function to run prediction collection."""
    args = parse_args()
    logger = setup_logging()
    
    try:
        # Initialize configuration
        config = Config()
        logger.info(f"Configuration loaded successfully. Data path: {config.data.file_path}")
        
        # Initialize data pipeline with the correct path
        dataset = DataPipeline(config.data.file_path)
        logger.info("Data pipeline initialized")
        
        # Log data information
        logger.info(f"Data shape: {dataset.data.shape}")
        logger.info(f"Number of features: {len(dataset.features)}")
        
        # Initialize collector
        collector = PredictionCollector(config)
        logger.info("Prediction collector initialized")
        
        # Collect predictions
        logger.info(f"Collecting predictions for years {args.start_year}-{args.end_year}")
        results = collector.collect_predictions(
            dataset=dataset,
            start_year=args.start_year,
            end_year=args.end_year,
            custom_objects={'r2_metrics': r2_metrics_tf}
        )
        
        # Save results
        collector.save_results(results, args.output_name)
        logger.info("Prediction collection completed successfully")
        
    except Exception as e:
        logger.error(f"Error in prediction collection: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()