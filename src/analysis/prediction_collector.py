# src/analysis/prediction_collector.py

# src/analysis/prediction_collector.py

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import tensorflow as tf
from joblib import load
from tqdm import tqdm
import logging

class PredictionCollector:
    """
    Collects and combines predictions from multiple models.
    """
    
    def __init__(self, config):
        """
        Initialize the prediction collector.
        
        Args:
            config: Configuration object containing paths and settings
        """
        self.config = config
        self.base_path = Path(config.data.file_path)
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def collect_predictions(
        self,
        dataset,
        start_year: int,
        end_year: int,
        custom_objects: Optional[Dict] = None
    ) -> pd.DataFrame:
        """Collect and combine predictions from multiple models."""
        test_mask = (dataset.data['yyyymm'] >= start_year * 100) & \
                   (dataset.data['yyyymm'] < (end_year + 1) * 100)
        results = dataset.data[['yyyymm', 'permno', 'me', 'exret']][test_mask].copy()
        
        # Generate predictions for NN3
        self.logger.info("Collecting NN3 predictions...")
        nn3_predictions = []
        for year in tqdm(range(start_year, end_year + 1)):
            
            # Construct full path to NN3 model
            model_path = self.base_path / 'NN3_model' / f'NN3_{year}.keras'
            self.logger.info(f"Loading NN3 model from: {model_path}")
            
            model = tf.keras.models.load_model(
                model_path,
                custom_objects=custom_objects
            )
            X_test, y, _ = dataset.load_one_year_data(year)  # Fixed method name
            y_predict = model.predict(X_test, verbose=0)
            nn3_predictions.extend(y_predict)
            
        
        results['pred_nn3'] = np.array(nn3_predictions).reshape(-1)
        
        # Generate predictions for RF
        self.logger.info("Collecting RF predictions...")
        rf_predictions = []
        for year in tqdm(range(start_year, end_year + 1)):
            # try:
            # Construct full path to RF model
            model_path = self.base_path / 'RF_model' / f'RF_{year}.joblib'

            model = load(model_path)
            self.logger.info(f"Loading RF model from: {model_path}")
            X_test, y, _ = dataset.load_one_year_data(year)  # Fixed method name
            y_predict = model.predict(X_test)
            rf_predictions.extend(y_predict)
            # except Exception as e:
            #     self.logger.error(f"Error processing RF for year {year}: {str(e)}")
            #     X_test, y, _ = dataset.load_one_year_data(year)  # Fixed method name
            #     rf_predictions.extend([np.nan] * len(X_test))
        
        results['pred_rf'] = np.array(rf_predictions).reshape(-1)
        
        return results
    
    def save_results(
        self,
        results: pd.DataFrame,
        analysis_name: str = 'model_predictions'
    ) -> None:
        """Save combined predictions to file."""
        # Create predictions directory in the base path
        output_dir = self.base_path / 'predictions'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save with timestamp
        timestamp = pd.Timestamp.now().strftime('%Y%m%d')
        filename = f'{analysis_name}_{timestamp}.csv'
        
        results.to_csv(output_dir / filename, index=False)
        self.logger.info(f"Results saved to {output_dir / filename}")