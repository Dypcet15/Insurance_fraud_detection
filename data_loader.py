"""
Data loading and preprocessing module for insurance fraud detection
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Load and prepare data for model training and prediction"""
    
    def __init__(self, filepath, target_column='is_fraud'):
        """Initialize DataLoader
        
        Args:
            filepath (str): Path to the CSV file
            target_column (str): Name of the target column
        """
        self.filepath = filepath
        self.target_column = target_column
        self.data = None
        self.X = None
        self.y = None
        
    def load_data(self):
        """Load data from CSV file"""
        try:
            self.data = pd.read_csv(self.filepath)
            logger.info(f"Data loaded successfully. Shape: {self.data.shape}")
            return self.data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def check_missing_values(self):
        """Check for missing values in the dataset"""
        missing = self.data.isnull().sum()
        if missing.sum() > 0:
            logger.warning(f"Missing values found:\n{missing[missing > 0]}")
        return missing
    
    def remove_duplicates(self):
        """Remove duplicate rows"""
        initial_shape = self.data.shape[0]
        self.data = self.data.drop_duplicates()
        removed = initial_shape - self.data.shape[0]
        logger.info(f"Removed {removed} duplicate rows")
        return self.data
    
    def split_data(self, test_size=0.2, random_state=42):
        """Split data into training and testing sets
        
        Args:
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
        """
        if self.target_column not in self.data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in data")
        
        self.y = self.data[self.target_column]
        self.X = self.data.drop(columns=[self.target_column])
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def get_data_info(self):
        """Print detailed information about the dataset"""
        if self.data is not None:
            logger.info(f"\nDataset Info:")
            logger.info(f"Shape: {self.data.shape}")
            logger.info(f"Columns: {self.data.columns.tolist()}")
            logger.info(f"Data types:\n{self.data.dtypes}")
            logger.info(f"Null values:\n{self.data.isnull().sum()}")
        return self.data.info() if self.data is not None else None
