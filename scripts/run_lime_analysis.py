#!/usr/bin/env python

import argparse
import logging
import sys
from pathlib import Path
import time
from datetime import datetime
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from src.config.config import Config
from src.data.data_pipeline import DataPipeline
from src.models.lime_explainer import LIMEExplainer
from tensorflow.keras.models import load_model
from joblib import load

def setup_logging(config: Config) -> logging.Logger:
    """Set up logging configuration."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = config.logs_dir / f"lime_analysis_{timestamp}.log"
    
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
    parser = argparse.ArgumentParser(description='Run LIME Analysis')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--model-type', type=str, required=True, choices=['nn', 'rf'],
                      help='Model type (neural network or random forest)')
    parser.add_argument('--year', type=int, required=True, help='Year to analyze')
    parser.add_argument('--batch-size', type=int, help='Batch size for LIME analysis')
    return parser.parse_args()

def load_model_predictor(config: Config, model_type: str, year: int):
    """Load the appropriate model and return its prediction function."""
    if model_type == 'nn':
        model = load_model(
            config.get_model_path('NN3', year) / f'NN3_{year}.keras',
            custom_objects={'r2_metrics_tf': None}  # Add your custom metrics here
        )
        return lambda x: model.predict(x, verbose=0)
    else:
        model = load(config.get_model_path('RF', year) / f'rf_model_{year}.joblib')
        return model.predict

def main():
    """Main LIME analysis script."""
    args = parse_args()
    
    # Load configuration
    config = Config(args.config if args.config else None)
    logger = setup_logging(config)
    
    try:
        # Initialize data pipeline
        pipeline = DataPipeline(config.data.file_path)
        logger.info("Data pipeline initialized successfully")
        
        # Load data for the specified year
        X_train, y_train, X_test, y_test = pipeline.load_train_test(
            args.year,
            config.analysis.test_period
        )
        logger.info(f"Data loaded for year {args.year}")
        
        # Load model predictor
        model_predict = load_model_predictor(config, args.model_type, args.year)
        logger.info(f"Model loaded for year {args.year}")
        
        # Initialize LIME explainer
        batch_size = args.batch_size if args.batch_size else config.lime.batch_size
        explainer = LIMEExplainer(
            model_predict=model_predict,
            training_data=X_train,
            num_samples=config.lime.num_samples,
            kernel_width=config.lime.kernel_width,
            verbose=config.lime.verbose
        )
        
        # Run LIME analysis for each feature
        for feature in pipeline.features:
            logger.info(f"Starting LIME analysis for feature {feature}")
            start_time = time.time()
            
            try:
                # Generate explanations
                explanations = explainer.explain_instances(
                    X_test,
                    batch_size=batch_size
                )
                
                # Create results DataFrame
                results_df = pd.DataFrame({
                    'permno': range(len(explanations)),
                    'yyyymm': args.year * 100 + 1,  # Adjust as needed
                    feature: X_test[feature],
                    'coefficient': [exp.coefficients[0] for exp in explanations],
                    't_statistic': [exp.t_statistics[0] for exp in explanations],
                    'p_value': [exp.p_values[0] for exp in explanations]
                })
                
                # Save results
                output_dir = config.results_dir / f"{args.model_type.upper()}_model"
                output_dir.mkdir(parents=True, exist_ok=True)
                results_df.to_csv(output_dir / f"{feature}.csv", index=False)
                
                end_time = time.time()
                logger.info(f"Feature {feature} analysis completed in {end_time - start_time:.2f} seconds")
                
            except Exception as e:
                logger.error(f"Error analyzing feature {feature}: {str(e)}")
                continue
        
    except Exception as e:
        logger.error(f"Fatal error in LIME analysis: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()