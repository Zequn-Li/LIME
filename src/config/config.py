# src/config/config.py

from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
import yaml
import os
from datetime import datetime

@dataclass
class DataConfig:
    """Configuration for data processing and storage."""
    file_path: Path = Path("/Users/zequnli/Documents/zequnli/Education/2019SIT/ResearchData/LIMEdata")
    start_date: str = '1963-07-01'
    end_date: str = '2021-12-31'
    wrds_username: str = None
    min_year: int = 1987
    max_year: int = 2021

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
        
        # Initialize data configuration
        self.data = DataConfig()
        
        # Set up external data and model paths
        self.base_data_path = self.data.file_path
        self.models_dir = self.base_data_path  # Parent directory containing model folders
        
        # Set up project-specific directories for logs and results
        self.results_dir = self.project_root / 'results'
        self.logs_dir = self.project_root / 'logs'
        
        # Create required directories
        self._create_directories()

        # Load custom config if provided
        if config_path:
            self.load_config(config_path)

    def _create_directories(self) -> None:
        """Create necessary project directories."""
        for directory in [self.results_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def get_model_path(self, model_type: str) -> Path:
        """Get path for specific model type."""
        if model_type == 'NN3':
            return self.base_data_path / 'NN3_model'
        elif model_type == 'RF':
            return self.base_data_path / 'RF_model'
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def get_results_path(self, analysis_name: str) -> Path:
        """Get path for analysis results."""
        return self.results_dir / analysis_name

    def get_log_path(self) -> Path:
        """Get path for log file with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.logs_dir / f"log_{timestamp}.txt"

    def load_config(self, config_path: str) -> None:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
            
        if 'data' in config_data:
            self.data = DataConfig(**config_data['data'])
            self.base_data_path = self.data.file_path

    def save_config(self, save_path: str) -> None:
        """Save current configuration to YAML file."""
        config_dict = {
            'data': self.data.__dict__,
        }
        
        with open(save_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

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