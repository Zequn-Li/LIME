from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
import yaml
import os
from datetime import datetime

@dataclass
class DataConfig:
    """Configuration for data processing and storage."""
    file_path: Path
    start_date: str = '1963-07-01'
    end_date: str = '2021-12-31'
    wrds_username: str = None
    min_year: int = 1987
    max_year: int = 2021

@dataclass
class ModelConfig:
    """Configuration for model parameters."""
    # Neural Network parameters
    nn_params: Dict[str, Any] = None
    epochs: int = 100
    batch_size: int = 32
    validation_split: float = 0.2
    early_stopping_patience: int = 3
    
    # Random Forest parameters
    rf_params: Dict[str, Any] = None
    n_estimators: int = 500
    max_depth_range: List[int] = None
    max_features_range: List[float] = None
    max_samples_range: List[float] = None
    min_samples_leaf_range: List[float] = None
    
    def __post_init__(self):
        if self.nn_params is None:
            self.nn_params = {
                'l1_range': [1e-3, 1e-4, 1e-5],
                'learning_rate_range': [1e-2, 1e-3, 1e-4, 1e-5],
                'dropout_range': [0.0, 0.3, 0.5],
                'units_range': {'min': 8, 'max': 32, 'step': 8}
            }
        
        if self.rf_params is None:
            self.rf_params = {
                'n_estimators': 500,
                'max_depth_range': [3, 5, 10, 15, 20],
                'max_features_range': [0.2, 0.3, 0.5, 0.8],
                'max_samples_range': [0.2, 0.3, 0.5, 0.8],
                'min_samples_leaf_range': [0.001, 0.005, 0.01, 0.05, 0.1]
            }

@dataclass
class LIMEConfig:
    """Configuration for LIME analysis."""
    num_samples: int = 5000
    kernel_width: float = 1.0
    batch_size: int = 1000
    memory_threshold_mb: int = 1000
    verbose: bool = False

@dataclass
class AnalysisConfig:
    """Configuration for statistical analysis."""
    significance_level: float = 0.05
    test_period: int = 12
    rolling_window: int = 12
    newey_west_lags: int = None  # If None, computed automatically
    multiple_testing_method: str = 'holm'  # 'holm' or 'bhy'

class Config:
    """Main configuration class that manages all settings."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Optional path to yaml configuration file
        """
        # Set up base paths
        self.project_root = Path(__file__).parent.parent.parent
        self.data_dir = self.project_root / 'data'
        self.models_dir = self.project_root / 'models'
        self.results_dir = self.project_root / 'results'
        self.logs_dir = self.project_root / 'logs'

        # Create required directories
        self._create_directories()

        # Initialize configurations
        self.data = DataConfig(file_path=self.data_dir)
        self.model = ModelConfig()
        self.lime = LIMEConfig()
        self.analysis = AnalysisConfig()

        # Load custom config if provided
        if config_path:
            self.load_config(config_path)

        # Features list
        self.features = [
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

    def _create_directories(self) -> None:
        """Create necessary project directories."""
        for directory in [self.data_dir, self.models_dir, self.results_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def load_config(self, config_path: str) -> None:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
            
        # Update configurations
        if 'data' in config_data:
            self.data = DataConfig(**config_data['data'])
        if 'model' in config_data:
            self.model = ModelConfig(**config_data['model'])
        if 'lime' in config_data:
            self.lime = LIMEConfig(**config_data['lime'])
        if 'analysis' in config_data:
            self.analysis = AnalysisConfig(**config_data['analysis'])

    def save_config(self, save_path: str) -> None:
        """
        Save current configuration to YAML file.
        
        Args:
            save_path: Path to save configuration file
        """
        config_dict = {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'lime': self.lime.__dict__,
            'analysis': self.analysis.__dict__
        }
        
        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def get_model_path(self, model_name: str, year: int) -> Path:
        """Get path for model files."""
        return self.models_dir / f"{model_name}_{year}"

    def get_results_path(self, analysis_name: str) -> Path:
        """Get path for analysis results."""
        return self.results_dir / analysis_name

    def get_log_path(self) -> Path:
        """Get path for log file with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.logs_dir / f"log_{timestamp}.txt"

# Example configuration YAML file:
"""
data:
  file_path: /path/to/data
  start_date: '1963-07-01'
  end_date: '2021-12-31'
  wrds_username: username

model:
  nn_params:
    l1_range: [0.001, 0.0001, 0.00001]
    learning_rate_range: [0.01, 0.001, 0.0001]
    dropout_range: [0.0, 0.3, 0.5]
    units_range:
      min: 8
      max: 32
      step: 8

lime:
  num_samples: 5000
  kernel_width: 1.0
  batch_size: 1000
  verbose: true

analysis:
  significance_level: 0.05
  test_period: 12
  multiple_testing_method: 'holm'
"""

# Example usage:
"""
# Initialize with default settings
config = Config()

# Or load from YAML file
config = Config('config.yaml')

# Access configurations
data_path = config.data.file_path
num_samples = config.lime.num_samples
sig_level = config.analysis.significance_level

# Save current configuration
config.save_config('current_config.yaml')
"""