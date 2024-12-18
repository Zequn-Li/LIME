import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.config.config import Config
from src.data.data_pipeline import DataPipeline
from src.models.neural_network import StockReturnNN
from src.models.random_forest import StockReturnRF
from src.models.lime_explainer import LIMEExplainer
from src.analysis.portfolio import PortfolioAnalysis
from src.analysis.hypothesis_testing import LIMEHypothesisTester
from src.analysis.interaction_analysis import InteractionAnalyzer

class TestDataPipeline(unittest.TestCase):
    """Test data pipeline functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        cls.config = Config()
        
        # Create sample data
        np.random.seed(42)
        n_samples = 1000
        n_features = 79
        
        # Create sample DataFrame
        cls.sample_data = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        cls.sample_data['exret'] = np.random.randn(n_samples)
        cls.sample_data['yyyymm'] = np.random.randint(198700, 202100, n_samples)
        cls.sample_data['permno'] = np.random.randint(10000, 99999, n_samples)
        cls.sample_data['me'] = np.random.rand(n_samples) * 1000
        
        # Save sample data
        cls.sample_data.to_csv(cls.config.data_dir / 'mldata.csv', index=False)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        # Remove test files
        if hasattr(cls, 'config'):
            test_file = cls.config.data_dir / 'mldata.csv'
            if test_file.exists():
                test_file.unlink()
            
            test_model_dir = cls.config.data_dir / 'test_model'
            if test_model_dir.exists():
                for f in test_model_dir.glob('*'):
                    f.unlink()
                test_model_dir.rmdir()

    def test_data_loading(self):
        """Test data loading functionality."""
        pipeline = DataPipeline(self.config.data_dir)
        self.assertIsNotNone(pipeline.data)
        self.assertEqual(len(pipeline.features), 79)

    def test_train_test_split(self):
        """Test train-test split functionality."""
        pipeline = DataPipeline(self.config.data_dir)
        X_train, y_train, X_test, y_test = pipeline.load_train_test(2000, 1)
        
        self.assertIsNotNone(X_train)
        self.assertIsNotNone(y_train)
        self.assertEqual(len(X_train), len(y_train))

class TestModels(unittest.TestCase):
    """Test model functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data for models."""
        cls.config = Config()
        # Create test data as DataFrame with feature names
        feature_names = [f'feature_{i}' for i in range(79)]
        cls.X = pd.DataFrame(
            np.random.randn(1000, 79),
            columns=feature_names
        )
        cls.y = np.random.randn(1000)

    def test_random_forest(self):
        """Test random forest model."""
        model = StockReturnRF(model_path=str(self.config.models_dir))
        
        # Train the model
        model.train(self.X, self.y)
        
        # Test predictions
        predictions = model.predict(self.X.iloc[:10])
        self.assertEqual(len(predictions), 10)
        
        # Test feature importance
        importance_df = model.get_feature_importance()
        self.assertEqual(len(importance_df), 79)  # Number of features
        self.assertTrue('feature' in importance_df.columns)
        self.assertTrue('importance' in importance_df.columns)

class TestLIME(unittest.TestCase):
    """Test LIME functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data for LIME."""
        cls.X = np.random.randn(100, 79)
        cls.dummy_model = lambda x: np.sum(x, axis=1)

    def test_lime_explainer(self):
        """Test LIME explainer."""
        # Make sure dummy_model accepts the right number of arguments
        self.dummy_model = lambda x: np.sum(x, axis=1) if len(x.shape) > 1 else np.sum(x)
        
        explainer = LIMEExplainer(
            model_predict=self.dummy_model,
            training_data=self.X,
            num_samples=100,
            verbose=True
        )
        explanation = explainer.explain_instance(self.X[0])
        self.assertIsNotNone(explanation)

class TestAnalysis(unittest.TestCase):
    """Test analysis functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data for analysis."""
        cls.config = Config()
        cls.returns = pd.Series(np.random.randn(100))
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        # Remove test files
        if hasattr(cls, 'config'):
            test_file = cls.config.data_dir / 'mldata.csv'
            if test_file.exists():
                test_file.unlink()
            
            test_model_dir = cls.config.data_dir / 'test_model'
            if test_model_dir.exists():
                for f in test_model_dir.glob('*'):
                    f.unlink()
                test_model_dir.rmdir()

    def test_portfolio_analysis(self):
        """Test portfolio analysis."""
        analyzer = PortfolioAnalysis(self.config.data_dir)
        stats = analyzer.newey_west_statistics(self.returns)
        self.assertIsNotNone(stats)

    def test_hypothesis_testing(self):
        """Test hypothesis testing."""
        # Create directory first
        test_model_dir = self.config.data_dir / 'test_model'
        test_model_dir.mkdir(parents=True, exist_ok=True)
        
        tester = LIMEHypothesisTester(
            data_path=self.config.data_dir,
            model_name='test_model'
        )
        # Create dummy test data
        results = pd.DataFrame({
            'coefficient': np.random.randn(100),
            'feature1': np.random.randn(100),
            'yyyymm': np.random.randint(200001, 202012, 100)
        })
        results.to_csv(test_model_dir / 'feature1.csv')
        test_results = tester.test_linearity()
        self.assertIsNotNone(test_results)

def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestDataPipeline))
    suite.addTests(loader.loadTestsFromTestCase(TestModels))
    suite.addTests(loader.loadTestsFromTestCase(TestLIME))
    suite.addTests(loader.loadTestsFromTestCase(TestAnalysis))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

if __name__ == '__main__':
    run_tests()