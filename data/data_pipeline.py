import pandas as pd
import numpy as np
from typing import Tuple, Optional, List, Dict

class DataPipeline:
    """
    A data pipeline for processing and managing financial data.
    
    Attributes:
        file_path (str): Path to the data files
        data (pd.DataFrame): The loaded dataset
        features (List[str]): List of feature names excluding target variables
    """
    
    def __init__(self, file_path: str):
        """
        Initialize the DataPipeline.
        
        Args:
            file_path (str): Path to the data files
        """
        self.file_path = file_path
        self.data = pd.read_csv(file_path + 'mldata.csv')
        
        # Define features excluding target and identifier columns
        self.features = self.data.columns.to_list()
        excluded_cols = ['exret', 'yyyymm', 'permno', 'me']
        self.features = [col for col in self.features if col not in excluded_cols]

    def load_train_test(self, test_year: int, test_period: int) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Load training and test data for a specific time period.
        
        Args:
            test_year (int): The starting year for the test period
            test_period (int): Number of years in the test period
            
        Returns:
            Tuple containing:
            - X_train: Training features
            - y_train: Training target
            - X_test: Test features
            - y_test: Test target
        """
        # Get test data for specified period
        test = self.data[(self.data['yyyymm'] >= test_year*100) & 
                        (self.data['yyyymm'] < (test_year+test_period)*100)]
        
        # Fill missing values
        test = test.fillna(0)
        
        # Split features and target
        X_test = test[self.features]
        y_test = test['exret']
        
        return X_test, y_test, X_test, y_test

    def load_one_year_data(self, year: int) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Load data for a specific year.
        
        Args:
            year (int): The year to load data for
            
        Returns:
            Tuple containing:
            - X: Features for the year
            - y: Target variable for the year
            - metadata: DataFrame with yyyymm, permno, exret, and me
        """
        one_year = self.data[
            (self.data['yyyymm'] > year*100) & 
            (self.data['yyyymm'] < (year+1)*100)
        ]
        one_year = one_year.fillna(0)
        
        X = one_year[self.features]
        y = one_year['exret']
        metadata = one_year[['yyyymm', 'permno', 'exret', 'me']]
        
        return X, y, metadata

    def load_one_month_data(self, year: int, month: int) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """
        Load data for a specific month.
        
        Args:
            year (int): The year
            month (int): The month (1-12)
            
        Returns:
            Tuple containing:
            - X: Features for the month
            - y: Target variable for the month
            - metadata: DataFrame with yyyymm, permno, exret, and me
        """
        yyyymm = year*100 + month
        one_month = self.data[self.data['yyyymm'] == yyyymm]
        one_month = one_month.fillna(0)
        
        X = one_month[self.features]
        y = one_month['exret']
        metadata = one_month[['yyyymm', 'permno', 'exret', 'me']]
        
        return X, y, metadata