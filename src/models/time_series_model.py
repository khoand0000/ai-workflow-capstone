import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from fbprophet import Prophet
import logging
from typing import Dict, Any, Tuple, List
import joblib
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeSeriesModel:
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize time series model.
        
        Args:
            model_type (str): Type of model to use ('random_forest', 'linear', 'arima', 'prophet')
        """
        self.model_type = model_type
        self.model = None
        self.features = None
        self.target = 'Revenue'
        
    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Train the model.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
        """
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X, y)
            
        elif self.model_type == 'linear':
            self.model = LinearRegression()
            self.model.fit(X, y)
            
        elif self.model_type == 'arima':
            # ARIMA requires different data format
            self.model = ARIMA(y, order=(5,1,0))
            self.model = self.model.fit()
            
        elif self.model_type == 'prophet':
            # Prophet requires specific data format
            df = pd.DataFrame({
                'ds': X.index,
                'y': y
            })
            self.model = Prophet()
            self.model.fit(df)
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
        logger.info(f"Trained {self.model_type} model")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X (pd.DataFrame): Feature matrix
            
        Returns:
            np.ndarray: Predictions
        """
        if self.model_type in ['random_forest', 'linear']:
            return self.model.predict(X)
            
        elif self.model_type == 'arima':
            return self.model.forecast(steps=len(X))
            
        elif self.model_type == 'prophet':
            future = pd.DataFrame({'ds': X.index})
            forecast = self.model.predict(future)
            return forecast['yhat'].values
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
            
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): True values
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        y_pred = self.predict(X)
        
        metrics = {
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred)
        }
        
        return metrics
        
    def save(self, path: str):
        """
        Save model to file.
        
        Args:
            path (str): Path to save model
        """
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
            
        joblib.dump(self.model, path)
        logger.info(f"Saved model to {path}")
        
    def load(self, path: str):
        """
        Load model from file.
        
        Args:
            path (str): Path to load model from
        """
        self.model = joblib.load(path)
        logger.info(f"Loaded model from {path}")

def compare_models(X: pd.DataFrame, y: pd.Series, 
                  model_types: List[str] = ['random_forest', 'linear', 'arima', 'prophet']) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple models.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        model_types (List[str]): List of model types to compare
        
    Returns:
        Dict[str, Dict[str, float]]: Evaluation metrics for each model
    """
    results = {}
    
    for model_type in model_types:
        logger.info(f"Training and evaluating {model_type} model")
        model = TimeSeriesModel(model_type=model_type)
        model.train(X, y)
        metrics = model.evaluate(X, y)
        results[model_type] = metrics
        
    return results 