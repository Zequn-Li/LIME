import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from joblib import dump, load
from typing import Dict, Tuple, Optional, Any
import logging

class StockReturnRF:
    """
    Random Forest model for stock return prediction.
    Implements hyperparameter tuning and model persistence.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the Random Forest model.
        
        Args:
            model_path (str): Path for saving/loading models
        """
        self.model_path = model_path
        self.model = None
        self.best_params = None
        self.feature_importance = None
        self.is_fitted = False
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _calculate_feature_importance(self) -> None:
        """Calculate and store feature importance scores."""
        if not self.is_fitted:
            raise ValueError("Model not trained yet")
            
        if not hasattr(self.model, 'feature_names_in_'):
            # If feature names aren't available, use indices
            feature_names = [f'feature_{i}' for i in range(len(self.model.feature_importances_))]
        else:
            feature_names = self.model.feature_names_in_

        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

    def train(self, 
             X: np.ndarray,
             y: np.ndarray,
             hyperparams: Optional[Dict] = None) -> None:
        """
        Train the Random Forest model.
        
        Args:
            X: Feature matrix
            y: Target vector
            hyperparams: Optional dictionary of hyperparameters
        """
        self.logger.info("Training Random Forest model...")
        
        if hyperparams is not None:
            self.model = RandomForestRegressor(**hyperparams, random_state=42)
        elif self.model is None:
            self.model = RandomForestRegressor(random_state=42)
            
        # Convert feature names to list if X is a DataFrame
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X = X.values
        else:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            
        self.model.fit(X, y)
        self.is_fitted = True
        self._calculate_feature_importance()
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not trained yet")
            
        return self.model.predict(X)

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Returns:
            DataFrame with feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model not trained yet. Please train the model first.")
            
        if self.feature_importance is None:
            self._calculate_feature_importance()
            
        return self.feature_importance.copy()

    def save_model(self, year: int) -> None:
        """
        Save the trained model and feature importance.
        
        Args:
            year: Year identifier for the model
        """
        if not self.is_fitted:
            raise ValueError("No model to save. Please train the model first.")
            
        model_file = f"{self.model_path}/rf_model_{year}.joblib"
        importance_file = f"{self.model_path}/rf_importance_{year}.csv"
        
        dump(self.model, model_file)
        if self.feature_importance is not None:
            self.feature_importance.to_csv(importance_file)
            
        self.logger.info(f"Model saved to {model_file}")

    def load_model(self, year: int) -> None:
        """
        Load a saved model.
        
        Args:
            year: Year identifier for the model
        """
        model_file = f"{self.model_path}/rf_model_{year}.joblib"
        importance_file = f"{self.model_path}/rf_importance_{year}.csv"
        
        try:
            self.model = load(model_file)
            if os.path.exists(importance_file):
                self.feature_importance = pd.read_csv(importance_file)
            self.logger.info(f"Model loaded from {model_file}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {model_file}")

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Returns:
            DataFrame with feature importance scores
        """
        if self.feature_importance is None:
            raise ValueError("Feature importance not calculated yet")
            
        return self.feature_importance

# Example usage:
"""
# Initialize model
rf_model = StockReturnRF(model_path='/path/to/models')

# For hyperparameter tuning
rf_model.tune_hyperparameters(X, y, train_idx, val_idx)

# For training with specific hyperparameters
hyperparams = {
    'n_estimators': 500,
    'max_depth': 10,
    'max_features': 0.3,
    'max_samples': 0.5,
    'min_samples_leaf': 0.01
}
rf_model.train(X, y, hyperparams)

# Save the model
rf_model.save_model(year=2020)

# Load the model
rf_model.load_model(year=2020)

# Make predictions
predictions = rf_model.predict(X_test)

# Get feature importance
importance_df = rf_model.get_feature_importance()
"""