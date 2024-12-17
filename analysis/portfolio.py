import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from typing import Dict, Tuple, Optional, Union, List
from dataclasses import dataclass
import logging
from pathlib import Path

@dataclass
class StatisticalResult:
    """Data class to store statistical analysis results."""
    mean: float
    standard_error: float
    t_statistic: float
    p_value: float
    sharpe_ratio: float

class PortfolioAnalysis:
    """
    A comprehensive toolkit for financial portfolio analysis, including:
    - Newey-West adjusted statistics
    - Portfolio performance metrics
    - Statistical hypothesis testing
    """
    
    def __init__(self, data_path: Union[str, Path]):
        """
        Initialize PortfolioAnalysis.
        
        Args:
            data_path: Path to data files
        """
        self.data_path = Path(data_path)
        self.logger = self._setup_logger()

    @staticmethod
    def _setup_logger() -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def newey_west_statistics(
        self,
        returns: pd.Series,
        lags: Optional[int] = None
    ) -> StatisticalResult:
        """
        Calculate Newey-West adjusted statistics for a return series.
        
        Args:
            returns: Series of returns
            lags: Number of lags for Newey-West adjustment. If None, computed automatically.
            
        Returns:
            StatisticalResult containing computed statistics
        """
        # Convert to numpy array and remove any missing values
        y = returns.dropna().values
        x = np.ones_like(y)
        nobs = len(y)
        
        # Compute optimal lag length if not provided
        if lags is None:
            lags = int(np.ceil(12 * np.power(nobs / 100, 1/4)))
            
        try:
            # Fit OLS model with Newey-West adjustment
            model = sm.OLS(y, x)
            results = model.fit(
                cov_type='HAC',
                cov_kwds={'maxlags': lags, 'use_correction': True}
            )
            
            # Calculate statistics
            mean = results.params[0]
            std_error = results.bse[0]
            t_stat = results.tvalues[0]
            p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), nobs - 1))
            sharpe = np.sqrt(12) * mean / np.std(y)
            
            return StatisticalResult(
                mean=mean,
                standard_error=std_error,
                t_statistic=t_stat,
                p_value=p_value,
                sharpe_ratio=sharpe
            )
            
        except Exception as e:
            self.logger.error(f"Error in Newey-West calculation: {str(e)}")
            raise

    def compute_portfolio_statistics(
        self,
        returns_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute statistical measures for multiple portfolios.
        
        Args:
            returns_df: DataFrame where each column represents a portfolio's returns
            
        Returns:
            DataFrame with statistical measures for each portfolio
        """
        results = []
        
        for column in returns_df.columns:
            try:
                stats = self.newey_west_statistics(returns_df[column])
                results.append({
                    'Portfolio': column,
                    'Mean Return (%)': stats.mean * 100,
                    'Std Error (%)': stats.standard_error * 100,
                    't-statistic': stats.t_statistic,
                    'p-value': stats.p_value,
                    'Sharpe Ratio': stats.sharpe_ratio
                })
            except Exception as e:
                self.logger.error(f"Error processing portfolio {column}: {str(e)}")
                
        return pd.DataFrame(results).set_index('Portfolio')

    def multiple_testing_adjustment(
        self,
        p_values: pd.Series,
        method: str = 'holm',
        alpha: float = 0.05
    ) -> pd.Series:
        """
        Apply multiple testing adjustment to p-values.
        
        Args:
            p_values: Series of p-values
            method: Adjustment method ('holm' or 'bhy')
            alpha: Significance level
            
        Returns:
            Series of adjusted p-values
        """
        sorted_idx = p_values.argsort()
        p_values_sorted = p_values.iloc[sorted_idx]
        n = len(p_values)
        
        if method == 'holm':
            # Holm's adjustment
            adjusted = pd.Series(
                [min(1, (n - i) * p) for i, p in enumerate(p_values_sorted)],
                index=p_values_sorted.index
            )
        else:  # BHY adjustment
            # Calculate c(m)
            c_m = np.sum(1 / np.arange(1, n + 1))
            
            # Calculate adjusted values
            adjusted = pd.Series(index=p_values_sorted.index)
            prev_value = 0
            for i, p in enumerate(p_values_sorted):
                value = min(1, c_m * n / (n - i) * p)
                adjusted.iloc[i] = max(value, prev_value)
                prev_value = adjusted.iloc[i]
                
        # Restore original order
        return adjusted.reindex(p_values.index)

    def calculate_rolling_statistics(
        self,
        returns: pd.Series,
        window: int,
        min_periods: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Calculate rolling statistical measures.
        
        Args:
            returns: Series of returns
            window: Rolling window size
            min_periods: Minimum number of observations required
            
        Returns:
            DataFrame with rolling statistics
        """
        if min_periods is None:
            min_periods = window // 2
            
        rolling_stats = pd.DataFrame(index=returns.index)
        
        # Rolling mean
        rolling_stats['mean'] = returns.rolling(
            window=window,
            min_periods=min_periods
        ).mean()
        
        # Rolling standard deviation
        rolling_stats['std'] = returns.rolling(
            window=window,
            min_periods=min_periods
        ).std()
        
        # Rolling Sharpe ratio
        rolling_stats['sharpe'] = (
            rolling_stats['mean'] / rolling_stats['std'] * np.sqrt(12)
        )
        
        # Rolling skewness and kurtosis
        rolling_stats['skew'] = returns.rolling(
            window=window,
            min_periods=min_periods
        ).skew()
        
        rolling_stats['kurt'] = returns.rolling(
            window=window,
            min_periods=min_periods
        ).kurt()
        
        return rolling_stats

    def performance_attribution(
        self,
        returns: pd.Series,
        factors: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Perform return attribution analysis using factor models.
        
        Args:
            returns: Series of portfolio returns
            factors: DataFrame of factor returns
            
        Returns:
            Dictionary containing alpha, factor exposures, and R-squared
        """
        try:
            # Add constant to factors for alpha estimation
            X = sm.add_constant(factors)
            
            # Fit regression model
            model = sm.OLS(returns, X)
            results = model.fit(cov_type='HAC', cov_kwds={'maxlags': 12})
            
            # Create attribution dictionary
            attribution = {
                'alpha': results.params['const'],
                'alpha_t_stat': results.tvalues['const'],
                'r_squared': results.rsquared
            }
            
            # Add factor exposures
            for factor in factors.columns:
                attribution[f'beta_{factor}'] = results.params[factor]
                attribution[f't_stat_{factor}'] = results.tvalues[factor]
                
            return attribution
            
        except Exception as e:
            self.logger.error(f"Error in performance attribution: {str(e)}")
            raise

# Example usage:
"""
# Initialize analysis
analyzer = PortfolioAnalysis(data_path='path/to/data')

# Calculate Newey-West statistics
stats = analyzer.newey_west_statistics(portfolio_returns)

# Compute portfolio statistics
portfolio_stats = analyzer.compute_portfolio_statistics(all_portfolio_returns)

# Adjust for multiple testing
adjusted_p_values = analyzer.multiple_testing_adjustment(
    p_values,
    method='holm',
    alpha=0.05
)

# Calculate rolling statistics
rolling_stats = analyzer.calculate_rolling_statistics(
    returns,
    window=12,
    min_periods=6
)

# Perform attribution analysis
attribution = analyzer.performance_attribution(
    portfolio_returns,
    factor_returns
)
"""