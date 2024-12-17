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
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def _get_param_grid(self) -> Dict[str, Any]:
        """
        Define the hyperparameter grid for random search.
        
        Returns:
            Dict containing hyperparameter ranges
        """
        return {
            'n_estimators': [500],  # Fixed as per requirement
            'max_depth': [3, 5, 10, 15, 20],
            'max_features': [0.2, 0.3, 0.5, 0.8],
            'max_samples': [0.2, 0.3, 0.5, 0.8],
            'min_samples_leaf': [0.001, 0.005, 0.01, 0.05, 0.1],
        }

    def tune_hyperparameters(self, 
                           X: np.ndarray,
                           y: np.ndarray,
                           train_indices: np.ndarray,
                           validation_indices: np.ndarray,
                           n_iter: int = 5) -> RandomForestRegressor:
        """
        Perform hyperparameter tuning using RandomizedSearchCV.
        
        Args:
            X: Feature matrix
            y: Target vector
            train_indices: Indices for training data
            validation_indices: Indices for validation data
            n_iter: Number of random combinations to try
            
        Returns:
            Tuned RandomForestRegressor
        """
        self.logger.info("Starting hyperparameter tuning...")
        
        # Create PredefinedSplit object
        test_fold = np.zeros(len(X))
        test_fold[validation_indices] = -1
        ps = PredefinedSplit(test_fold)

        # Initialize base model
        base_model = RandomForestRegressor(random_state=42)

        # Set up RandomizedSearchCV
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=self._get_param_grid(),
            n_iter=n_iter,
            cv=ps,
            verbose=2,
            random_state=42,
            n_jobs=-1
        )

        # Perform search
        random_search.fit(X, y)
        
        self.best_params = random_search.best_params_
        self.model = random_search.best_estimator_
        
        self.logger.info(f"Best parameters found: {self.best_params}")
        self.logger.info(f"Best CV score: {random_search.best_score_}")
        
        return self.model

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
            
        self.model.fit(X, y)
        self._calculate_feature_importance()
        
    def _calculate_feature_importance(self) -> None:
        """Calculate and store feature importance scores."""
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        self.feature_importance = pd.DataFrame({
            'feature': self.model.feature_names_in_,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        return self.model.predict(X)

    def save_model(self, year: int) -> None:
        """
        Save the trained model and feature importance.
        
        Args:
            year: Year identifier for the model
        """
        if self.model is None:
            raise ValueError("No model to save")
            
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