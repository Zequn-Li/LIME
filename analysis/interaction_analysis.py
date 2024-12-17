import pandas as pd
import numpy as np
import statsmodels.api as sm
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

@dataclass
class InteractionResult:
    """Data class to store interaction analysis results."""
    feature1: str
    feature2: str
    coefficient: float
    t_statistic: float
    p_value: float
    std_error: float

class InteractionAnalyzer:
    """
    Analyzes interactions between features using LIME coefficients.
    Implements methods for computing, visualizing, and testing feature interactions.
    """
    
    def __init__(self, data_path: Union[str, Path], model_name: str):
        """
        Initialize the interaction analyzer.
        
        Args:
            data_path: Path to data directory
            model_name: Name of the model being analyzed
        """
        self.data_path = Path(data_path)
        self.model_name = model_name
        self.features = self._load_feature_list()
        self.logger = self._setup_logger()
        self.min_year = 1989  # Minimum year for analysis
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def _load_feature_list(self) -> List[str]:
        """Load the list of features to analyze."""
        return [
            'absacc', 'acc', 'age', 'agr', 'beta', 'betasq', 'bm', 'bm_ia',
            'cashdebt', 'cashpr', 'cfp', 'cfp_ia', 'chatoia', 'chcsho', 'chempia',
            'chinv', 'chmom', 'chpmia', 'currat', 'depr', 'dolvol', 'dy', 'egr',
            'ep', 'gma', 'grcapx', 'grltnoa', 'herf', 'hire', 'idiovol', 'ill',
            'indmom', 'invest', 'lev', 'lgr', 'maxret', 'mom12m', 'mom1m',
            'mom36m', 'mom6m', 'mvel1', 'mve_ia', 'operprof', 'orgcap',
            'pchcapx_ia', 'pchcurrat', 'pchdepr', 'pchgm_pchsale', 'pchquick',
            'pchsale_pchinvt', 'pchsale_pchxsga', 'pchsaleinv', 'pctacc',
            'pricedelay', 'ps', 'rd_mve', 'rd_sale', 'retvol', 'roic',
            'salecash', 'saleinv', 'salerec', 'sgr', 'sp', 'std_dolvol',
            'std_turn', 'tang', 'tb', 'turn', 'zerotrade'
        ]

    def load_lime_data(self, feature: str) -> pd.DataFrame:
        """
        Load LIME results for a specific feature.
        
        Args:
            feature: Feature name
            
        Returns:
            DataFrame containing LIME results
        """
        file_path = self.data_path / f'{self.model_name}/{feature}.csv'
        lime_data = pd.read_csv(file_path)
        lime_data = lime_data[lime_data[feature] != 0]  # Filter zero values
        lime_data['yyyymm'] = pd.to_numeric(lime_data['yyyymm'])
        lime_data = lime_data[lime_data['yyyymm'] > self.min_year * 100]
        return lime_data

    def compute_newey_west(self, series: pd.Series) -> Tuple[float, float, float, float]:
        """
        Compute Newey-West adjusted statistics.
        
        Args:
            series: Time series data
            
        Returns:
            Tuple of coefficient, standard error, t-statistic, and p-value
        """
        y = series.values
        x = np.ones_like(y)
        nobs = len(y)
        
        # Compute optimal lag length
        lags = int(np.ceil(12 * np.power(nobs / 100, 1/4)))
        
        model = sm.OLS(y, x)
        results = model.fit(cov_type='HAC', cov_kwds={'maxlags': lags, 'use_correction': True})
        
        coef = results.params[0]
        stderr = results.bse[0]
        t_stat = results.tvalues[0]
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), nobs - 1))
        
        return coef, stderr, t_stat, p_value

    def analyze_interactions(self) -> pd.DataFrame:
        """
        Analyze interactions between all pairs of features.
        
        Returns:
            DataFrame containing interaction analysis results
        """
        self.logger.info("Starting interaction analysis...")
        results = []
        
        for feature1 in tqdm(self.features, desc="Analyzing interactions"):
            lime_data1 = self.load_lime_data(feature1)
            
            for feature2 in self.features:
                if feature2 == feature1:
                    continue
                    
                try:
                    lime_data2 = self.load_lime_data(feature2)
                    merged_data = pd.merge(
                        lime_data1[['coefficient', 'yyyymm', feature1]],
                        lime_data2[[feature2, 'yyyymm']],
                        on='yyyymm'
                    )
                    
                    # Compute interaction regression by year
                    yearly_coefs = merged_data.groupby('yyyymm').apply(
                        lambda x: sm.OLS(
                            x['coefficient'],
                            sm.add_constant(x[[feature1, feature2]])
                        ).fit().params[feature2]
                    )
                    
                    # Compute Newey-West statistics
                    coef, stderr, t_stat, p_value = self.compute_newey_west(yearly_coefs)
                    
                    results.append(InteractionResult(
                        feature1=feature1,
                        feature2=feature2,
                        coefficient=coef,
                        t_statistic=t_stat,
                        p_value=p_value,
                        std_error=stderr
                    ))
                    
                except Exception as e:
                    self.logger.error(f"Error analyzing interaction {feature1}-{feature2}: {str(e)}")
        
        return pd.DataFrame([vars(r) for r in results])

    def create_interaction_heatmap(
        self,
        interaction_results: pd.DataFrame,
        metric: str = 't_statistic',
        threshold: Optional[float] = None
    ) -> None:
        """
        Create a heatmap visualization of feature interactions.
        
        Args:
            interaction_results: DataFrame with interaction analysis results
            metric: Metric to visualize ('t_statistic', 'coefficient', or 'p_value')
            threshold: Optional significance threshold for highlighting
        """
        # Pivot data for heatmap
        heatmap_data = interaction_results.pivot(
            index='feature1',
            columns='feature2',
            values=metric
        )
        
        # Create figure
        plt.figure(figsize=(15, 12))
        
        # Create mask for insignificant values if threshold is provided
        mask = None
        if threshold is not None and 'p_value' in interaction_results:
            p_values = interaction_results.pivot(
                index='feature1',
                columns='feature2',
                values='p_value'
            )
            mask = p_values > threshold
        
        # Plot heatmap
        sns.heatmap(
            heatmap_data,
            mask=mask,
            cmap='RdBu_r',
            center=0,
            annot=True,
            fmt='.2f',
            cbar_kws={'label': metric}
        )
        
        plt.title(f'Feature Interaction {metric.replace("_", " ").title()}')
        plt.tight_layout()
        
        # Save plot
        output_dir = self.data_path / self.model_name / 'plots'
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / f'interaction_{metric}_heatmap.png')
        plt.close()

    def save_results(self, results: pd.DataFrame) -> None:
        """
        Save interaction analysis results.
        
        Args:
            results: DataFrame containing interaction analysis results
        """
        output_dir = self.data_path / self.model_name
        output_file = output_dir / 'interaction_analysis.csv'
        results.to_csv(output_file, index=False)
        self.logger.info(f"Results saved to {output_file}")

    def summarize_significant_interactions(
        self,
        results: pd.DataFrame,
        alpha: float = 0.05
    ) -> pd.DataFrame:
        """
        Summarize significant feature interactions.
        
        Args:
            results: DataFrame with interaction analysis results
            alpha: Significance level
            
        Returns:
            DataFrame containing only significant interactions
        """
        significant = results[results['p_value'] < alpha].copy()
        significant['abs_t_stat'] = np.abs(significant['t_statistic'])
        significant = significant.sort_values('abs_t_stat', ascending=False)
        significant = significant.drop('abs_t_stat', axis=1)
        return significant

# Example usage:
"""
# Initialize analyzer
analyzer = InteractionAnalyzer(
    data_path='path/to/data',
    model_name='RF_model'
)

# Perform analysis
interaction_results = analyzer.analyze_interactions()

# Create visualizations
analyzer.create_interaction_heatmap(
    interaction_results,
    metric='t_statistic',
    threshold=0.05
)

# Get significant interactions
significant_interactions = analyzer.summarize_significant_interactions(
    interaction_results,
    alpha=0.05
)

# Save results
analyzer.save_results(interaction_results)
"""