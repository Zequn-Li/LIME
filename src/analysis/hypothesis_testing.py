import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import logging

@dataclass
class TestResult:
    """Data class to store statistical test results."""
    feature: str
    test_statistic: float
    p_value: float
    coefficient: Optional[float] = None
    additional_info: Optional[Dict] = None

class LIMEHypothesisTester:
    """
    Class for conducting hypothesis tests on LIME explanations.
    Implements tests for linearity, heterogeneity, and independence.
    """
    
    def __init__(self, data_path: Union[str, Path], model_name: str):
        """
        Initialize the hypothesis tester.
        
        Args:
            data_path: Path to data directory
            model_name: Name of the model being tested
        """
        self.data_path = Path(data_path)
        self.model_name = model_name
        self.features = self._load_feature_list()
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def _load_feature_list(self) -> List[str]:
        """Load the list of features to test."""
        # Standard feature list used in the project
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

    def _load_lime_results(self, feature: str) -> pd.DataFrame:
        """
        Load LIME results for a specific feature.
        
        Args:
            feature: Feature name
            
        Returns:
            DataFrame containing LIME results
        """
        file_path = self.data_path / self.model_name / f'{feature}.csv'
        lime_results = pd.read_csv(file_path)
        lime_results['yyyymm'] = pd.to_datetime(lime_results['yyyymm'].astype(str), format='%Y%m')
        return lime_results

    def test_linearity(self, alpha: float = 0.05) -> pd.DataFrame:
        """
        Test the linearity hypothesis for all features.
        H0: LIME coefficients are zero
        
        Args:
            alpha: Significance level
            
        Returns:
            DataFrame containing test results
        """
        self.logger.info("Starting linearity tests...")
        results = []
        
        for feature in tqdm(self.features, desc="Testing linearity"):
            try:
                lime_data = self._load_lime_results(feature)
                
                # Perform t-test for each year
                yearly_results = lime_data.groupby(
                    lime_data['yyyymm'].dt.year
                )['coefficient'].apply(
                    lambda x: pd.Series(stats.ttest_1samp(x, 0))
                )
                
                # Aggregate results
                mean_t = yearly_results.apply(lambda x: x.statistic).mean()
                mean_p = yearly_results.apply(lambda x: x.pvalue).mean()
                
                results.append(TestResult(
                    feature=feature,
                    test_statistic=mean_t,
                    p_value=mean_p,
                    coefficient=lime_data['coefficient'].mean()
                ))
                
            except Exception as e:
                self.logger.error(f"Error testing linearity for {feature}: {str(e)}")
        
        return pd.DataFrame([vars(r) for r in results])

    def test_heterogeneity(self) -> pd.DataFrame:
        """
        Test the heterogeneity hypothesis.
        H0: LIME coefficients are constant over feature values
        
        Returns:
            DataFrame containing test results
        """
        self.logger.info("Starting heterogeneity tests...")
        results = []
        
        for feature in tqdm(self.features, desc="Testing heterogeneity"):
            try:
                lime_data = self._load_lime_results(feature)
                
                # Perform regression for each year
                def ols_test(group: pd.DataFrame) -> Tuple[float, float]:
                    model = sm.OLS(
                        group['coefficient'],
                        sm.add_constant(group[feature])
                    ).fit()
                    return model.fvalue, model.f_pvalue
                
                yearly_results = lime_data.groupby(
                    lime_data['yyyymm'].dt.year
                ).apply(ols_test)
                
                # Aggregate results
                mean_f = yearly_results.apply(lambda x: x[0]).mean()
                mean_p = yearly_results.apply(lambda x: x[1]).mean()
                
                results.append(TestResult(
                    feature=feature,
                    test_statistic=mean_f,
                    p_value=mean_p
                ))
                
            except Exception as e:
                self.logger.error(f"Error testing heterogeneity for {feature}: {str(e)}")
        
        return pd.DataFrame([vars(r) for r in results])

    def test_independence(self, dataset: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Test the independence hypothesis.
        H0: LIME coefficients are independent of other features
        
        Args:
            dataset: DataFrame containing feature values
            
        Returns:
            Dictionary containing test results and interaction matrices
        """
        self.logger.info("Starting independence tests...")
        f_values = pd.DataFrame(index=self.features, columns=self.features)
        p_values = pd.DataFrame(index=self.features, columns=self.features)
        
        for feature in tqdm(self.features, desc="Testing independence"):
            try:
                lime_data = self._load_lime_results(feature)
                
                # Merge with dataset
                merged_data = lime_data.merge(
                    dataset[['yyyymm', 'permno'] + self.features],
                    on=['yyyymm', 'permno']
                )
                
                # Test independence with each other feature
                for other_feature in self.features:
                    if other_feature == feature:
                        continue
                        
                    def independence_test(group: pd.DataFrame) -> Tuple[float, float]:
                        # Full model
                        full_model = sm.OLS(
                            group['coefficient'],
                            sm.add_constant(group[self.features])
                        ).fit()
                        
                        # Restricted model
                        restricted_model = sm.OLS(
                            group['coefficient'],
                            sm.add_constant(group[feature])
                        ).fit()
                        
                        # Calculate F-statistic
                        n = len(group)
                        k_full = len(self.features)
                        k_restricted = 1
                        f_stat = ((restricted_model.ssr - full_model.ssr) / (k_full - k_restricted)) / \
                                (full_model.ssr / (n - k_full))
                        p_value = 1 - stats.f.cdf(f_stat, k_full - k_restricted, n - k_full)
                        
                        return f_stat, p_value
                    
                    yearly_results = merged_data.groupby(
                        merged_data['yyyymm'].dt.year
                    ).apply(independence_test)
                    
                    f_values.loc[feature, other_feature] = yearly_results.apply(
                        lambda x: x[0]
                    ).mean()
                    p_values.loc[feature, other_feature] = yearly_results.apply(
                        lambda x: x[1]
                    ).mean()
                    
            except Exception as e:
                self.logger.error(f"Error testing independence for {feature}: {str(e)}")
        
        return {
            'f_values': f_values,
            'p_values': p_values
        }

    def save_results(self, results: Dict[str, pd.DataFrame]) -> None:
        """
        Save test results to files.
        
        Args:
            results: Dictionary containing test results
        """
        output_dir = self.data_path / self.model_name / 'hypothesis_tests'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for name, df in results.items():
            df.to_csv(output_dir / f'{name}.csv')
            self.logger.info(f"Saved {name} to {output_dir / f'{name}.csv'}")

# Example usage:
"""
# Initialize tester
tester = LIMEHypothesisTester(
    data_path='path/to/data',
    model_name='RF_model'
)

# Run linearity tests
linearity_results = tester.test_linearity()

# Run heterogeneity tests
heterogeneity_results = tester.test_heterogeneity()

# Run independence tests
independence_results = tester.test_independence(dataset)

# Save all results
tester.save_results({
    'linearity': linearity_results,
    'heterogeneity': heterogeneity_results,
    'independence_f': independence_results['f_values'],
    'independence_p': independence_results['p_values']
})
"""