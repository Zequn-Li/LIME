import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import logging

sys.path.append(str(Path(__file__).parent.parent))

from src.config.config import Config
from src.data.data_pipeline import DataPipeline
from src.models.neural_network import StockReturnNN
from src.models.lime_explainer import LIMEExplainer
from src.analysis.hypothesis_testing import LIMEHypothesisTester

def setup_logging():
    """Set up logging for end-to-end test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def run_end_to_end_test():
    """Run a basic end-to-end test of the pipeline."""
    logger = setup_logging()
    logger.info("Starting end-to-end test")
    
    try:
        # Initialize configuration
        config = Config()
        logger.info("Configuration loaded")
        
        # Create sample data
        n_samples = 1000
        n_features = 79
        sample_data = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        sample_data['exret'] = np.random.randn(n_samples)
        sample_data['yyyymm'] = np.random.randint(198700, 202100, n_samples)
        sample_data['permno'] = np.random.randint(10000, 99999, n_samples)
        sample_data['me'] = np.random.rand(n_samples) * 1000
        
        # Save sample data
        sample_data.to_csv(config.data_dir / 'mldata.csv', index=False)
        logger.info("Sample data created and saved")
        
        # Test data pipeline
        pipeline = DataPipeline(config.data_dir)
        X_train, y_train, X_test, y_test = pipeline.load_train_test(2000, 1)
        logger.info("Data pipeline tested successfully")
        
        # Test neural network
        model = StockReturnNN(input_dim=79)
        history = model.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_test,
            y_val=y_test,
            year=2000
        )
        logger.info("Neural network trained successfully")
        
        # Test LIME
        explainer = LIMEExplainer(
            model_predict=model.predict,
            training_data=X_train,
            num_samples=100,
            verbose=True
        )
        explanation = explainer.explain_instance(X_test[0])
        logger.info("LIME explanation generated successfully")
        
        # Test hypothesis testing
        tester = LIMEHypothesisTester(
            data_path=config.data_dir,
            model_name='test_model'
        )
        test_results = tester.test_linearity()
        logger.info("Hypothesis testing completed successfully")
        
        logger.info("End-to-end test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"End-to-end test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = run_end_to_end_test()
    sys.exit(0 if success else 1)