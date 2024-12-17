import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import t
from typing import Tuple, List, Callable, Optional
import logging
from tqdm import tqdm
import gc
import psutil
import os
from dataclasses import dataclass

@dataclass
class LIMEResult:
    """
    Data class to store LIME explanation results.
    """
    coefficients: np.ndarray
    t_statistics: np.ndarray
    p_values: np.ndarray

class MemoryMonitor:
    """
    Utility class to monitor memory usage during LIME explanations.
    """
    @staticmethod
    def get_memory_usage() -> float:
        """
        Get current memory usage in MB.
        
        Returns:
            float: Current memory usage in MB
        """
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024

    @staticmethod
    def log_memory_usage(message: str = "") -> None:
        """
        Log current memory usage with optional message.
        
        Args:
            message (str): Optional context message
        """
        memory_mb = MemoryMonitor.get_memory_usage()
        logging.info(f"Memory usage {message}: {memory_mb:.2f} MB")

class LIMEExplainer:
    """
    Local Interpretable Model-agnostic Explanations (LIME) implementation
    for explaining model predictions.
    """
    
    def __init__(
        self,
        model_predict: Callable,
        training_data: np.ndarray,
        num_samples: int = 5000,
        kernel_width: float = 1.0,
        verbose: bool = False
    ):
        """
        Initialize LIME explainer.
        
        Args:
            model_predict: Function that takes features and returns predictions
            training_data: Training data for scaling reference
            num_samples: Number of samples for local approximation
            kernel_width: Kernel width for similarity computation
            verbose: Whether to print progress and memory information
        """
        self.model_predict = model_predict
        self.training_data = training_data
        self.num_samples = num_samples
        self.kernel_width = kernel_width
        self.num_features = training_data.shape[1]
        self.scaler = StandardScaler(with_mean=False)
        self.scaler.fit(training_data)
        self.verbose = verbose
        
        # Set up logging
        logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
        self.logger = logging.getLogger(__name__)

    def _generate_perturbed_samples(
        self,
        instance: np.ndarray,
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate perturbed samples around an instance.
        
        Args:
            instance: Instance to explain
            batch_size: Optional batch size for memory efficiency
            
        Returns:
            np.ndarray: Perturbed samples
        """
        if batch_size is None:
            batch_size = self.num_samples
            
        samples = np.random.normal(
            0, 1,
            (batch_size, self.num_features)
        )
        
        # Scale samples using pre-fitted scaler
        scaled_samples = samples * self.scaler.scale_
        perturbed = scaled_samples + instance
        
        return perturbed

    def _compute_kernel_weights(
        self,
        instance: np.ndarray,
        perturbed_samples: np.ndarray
    ) -> np.ndarray:
        """
        Compute kernel-based weights for perturbed samples.
        
        Args:
            instance: Original instance
            perturbed_samples: Generated perturbed samples
            
        Returns:
            np.ndarray: Sample weights
        """
        distances = np.linalg.norm(
            perturbed_samples - instance.reshape(1, -1),
            axis=1
        )
        weights = np.exp(-(distances ** 2) / (self.kernel_width ** 2))
        return weights

    def _fit_local_model(
        self,
        perturbed_samples: np.ndarray,
        predictions: np.ndarray,
        weights: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit weighted linear regression model locally.
        
        Args:
            perturbed_samples: Generated samples
            predictions: Model predictions for samples
            weights: Sample weights
            
        Returns:
            Tuple containing coefficients, t-statistics, and p-values
        """
        model = LinearRegression()
        model.fit(perturbed_samples, predictions, sample_weight=weights)
        
        # Compute statistics
        predictions_local = model.predict(perturbed_samples)
        residuals = predictions - predictions_local
        mse = np.sum(weights * residuals ** 2) / (self.num_samples - self.num_features - 1)
        
        # Calculate standard errors
        X_weighted = perturbed_samples * np.sqrt(weights).reshape(-1, 1)
        covariance = np.linalg.inv(X_weighted.T @ X_weighted) * mse
        std_errors = np.sqrt(np.diag(covariance))
        
        # Calculate t-statistics and p-values
        t_stats = model.coef_ / std_errors
        p_values = 2 * (1 - t.cdf(np.abs(t_stats), self.num_samples - 2))
        
        return model.coef_, t_stats, p_values

    def explain_instance(
        self,
        instance: np.ndarray,
        batch_size: Optional[int] = None
    ) -> LIMEResult:
        """
        Generate LIME explanation for a single instance.
        
        Args:
            instance: Instance to explain
            batch_size: Optional batch size for memory efficiency
            
        Returns:
            LIMEResult containing coefficients, t-statistics, and p-values
        """
        if self.verbose:
            MemoryMonitor.log_memory_usage("Before perturbation")
            
        # Generate perturbed samples
        perturbed_samples = self._generate_perturbed_samples(instance, batch_size)
        
        if self.verbose:
            MemoryMonitor.log_memory_usage("After perturbation")
            
        # Get model predictions
        predictions = self.model_predict(perturbed_samples)
        
        if self.verbose:
            MemoryMonitor.log_memory_usage("After predictions")
            
        # Compute kernel weights
        weights = self._compute_kernel_weights(instance, perturbed_samples)
        
        # Fit local model and get statistics
        coefficients, t_stats, p_values = self._fit_local_model(
            perturbed_samples, predictions, weights
        )
        
        if self.verbose:
            MemoryMonitor.log_memory_usage("After local model fitting")
            
        # Clean up
        del perturbed_samples, predictions, weights
        gc.collect()
        
        return LIMEResult(coefficients, t_stats, p_values)

    def explain_instances(
        self,
        instances: np.ndarray,
        batch_size: Optional[int] = None
    ) -> List[LIMEResult]:
        """
        Generate LIME explanations for multiple instances.
        
        Args:
            instances: Instances to explain
            batch_size: Optional batch size for memory efficiency
            
        Returns:
            List of LIMEResult objects
        """
        explanations = []
        
        for idx in tqdm(range(len(instances)), desc="Generating LIME explanations"):
            if self.verbose:
                self.logger.info(f"Processing instance {idx + 1}/{len(instances)}")
                
            result = self.explain_instance(instances[idx], batch_size)
            explanations.append(result)
            
            if self.verbose:
                MemoryMonitor.log_memory_usage(f"After instance {idx + 1}")
                
            # Force garbage collection
            gc.collect()
            
        return explanations

# Example usage:
"""
# Initialize explainer
explainer = LIMEExplainer(
    model_predict=model.predict,
    training_data=X_train,
    num_samples=5000,
    verbose=True
)

# Explain a single instance
explanation = explainer.explain_instance(X_test[0])

# Explain multiple instances with batch processing
explanations = explainer.explain_instances(
    X_test,
    batch_size=1000
)

# Access results
coefficients = explanation.coefficients
t_stats = explanation.t_statistics
p_values = explanation.p_values
"""