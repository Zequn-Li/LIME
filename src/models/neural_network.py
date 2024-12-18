import tensorflow as tf
import numpy as np
from tensorflow.keras import Input, layers, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
from typing import Tuple, Dict, Any
from tensorflow.keras import regularizers

from src.utils.metrics import r2_metrics_tf

class StockReturnNN:
    """
    Neural Network model for stock return prediction.
    Implements a three-layer architecture with hyperparameter tuning.
    """
    
    def __init__(self, input_dim: int):
        """
        Initialize the neural network model.
        
        Args:
            input_dim (int): Number of input features
        """
        self.input_dim = input_dim
        self.model = None
        self.tuner = None

    def build_model(self, hp: kt.HyperParameters) -> Model:
        """
        Build the neural network model with hyperparameters.
        
        Args:
            hp (kt.HyperParameters): Hyperparameters for model construction
            
        Returns:
            Model: Compiled Keras model
        """
        # Hyperparameters to tune
        l1_reg = hp.Choice('l1', values=[1e-3, 1e-4, 1e-5])
        learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])
        dropout_rate = hp.Choice('dropout', values=[0.0, 0.3, 0.5])
        units = hp.Int('units', min_value=8, max_value=32, step=8)

        # Model architecture
        input_tensor = Input(shape=(self.input_dim,))
        x = layers.Dense(
            units=32,
            activation='relu',
            activity_regularizer=regularizers.L1(l1_reg)
        )(input_tensor)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(
            units=16,
            activation='relu',
            activity_regularizer=regularizers.L1(l1_reg)
        )(x)
        x = layers.Dense(units=8, activation='relu')(x)
        output_tensor = layers.Dense(1)(x)

        model = Model(input_tensor, output_tensor)
        
        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mean_squared_error',
            metrics=[r2_metrics_tf]
        )
        
        return model

    def create_tuner(self, year: int) -> None:
        """
        Create a hyperparameter tuner.
        
        Args:
            year (int): Year for model identification
        """
        self.tuner = kt.Hyperband(
            self.build_model,
            objective='val_loss',
            max_epochs=100,
            factor=3,
            directory='model_tuning',
            project_name=f'nn_tuning_{year}'
        )

    def train(self, 
              X_train: tf.Tensor,
              y_train: tf.Tensor,
              X_val: tf.Tensor,
              y_val: tf.Tensor,
              year: int) -> Dict[str, Any]:
        """
        Train the model with hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            year: Training year
            
        Returns:
            Dict containing best hyperparameters and training history
        """
        # Create tuner if not exists
        if self.tuner is None:
            self.create_tuner(year)

        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_r2_metrics',
            patience=3,
            mode='max'
        )

        # Search for best hyperparameters
        self.tuner.search(
            X_train, y_train,
            epochs=100,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping]
        )

        # Get best hyperparameters
        best_hps = self.tuner.get_best_hyperparameters()[0]

        # Train final model
        self.model = self.build_model(best_hps)
        history = self.model.fit(
            X_train, y_train,
            epochs=100,
            validation_split=0.2,
            callbacks=[early_stopping]
        )

        return {
            'best_hyperparameters': best_hps.values,
            'history': history.history
        }

    def save_model(self, year: int) -> None:
        """
        Save the trained model.
        
        Args:
            year (int): Year identifier for the model
        """
        if self.model is not None:
            self.model.save(f'models/NN3_{year}.keras')

    def load_model(self, year: int) -> None:
        """
        Load a trained model.
        
        Args:
            year (int): Year identifier for the model
        """
        self.model = tf.keras.models.load_model(
            f'models/NN3_{year}.keras',
            custom_objects={'r2_metrics': r2_metrics_tf}
        )

    def predict(self, X: tf.Tensor) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input features
            
        Returns:
            np.ndarray: Model predictions
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        return self.model.predict(X, verbose=0)