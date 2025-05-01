import unittest
import pandas as pd
import numpy as np
from src.models.time_series_model import TimeSeriesModel, compare_models
from src.data.data_ingestion import read_retail_data, prepare_time_series_data

class TestTimeSeriesModel(unittest.TestCase):
    def setUp(self):
        # Create sample data for testing
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.X = pd.DataFrame({
            'Date': dates,
            'Revenue_lag_1': np.random.randn(100),
            'Revenue_lag_7': np.random.randn(100),
            'Revenue_lag_30': np.random.randn(100),
            'DayOfWeek': np.random.randint(0, 7, 100),
            'Month': np.random.randint(1, 13, 100)
        })
        self.y = pd.Series(np.random.randn(100))
        
    def test_model_initialization(self):
        # Test initialization with different model types
        for model_type in ['random_forest', 'linear', 'arima', 'prophet']:
            model = TimeSeriesModel(model_type=model_type)
            self.assertEqual(model.model_type, model_type)
            
    def test_model_training(self):
        # Test training for each model type
        for model_type in ['random_forest', 'linear', 'arima', 'prophet']:
            model = TimeSeriesModel(model_type=model_type)
            model.train(self.X, self.y)
            self.assertIsNotNone(model.model)
            
    def test_model_prediction(self):
        # Test prediction for each model type
        for model_type in ['random_forest', 'linear', 'arima', 'prophet']:
            model = TimeSeriesModel(model_type=model_type)
            model.train(self.X, self.y)
            predictions = model.predict(self.X)
            self.assertEqual(len(predictions), len(self.X))
            
    def test_model_evaluation(self):
        # Test evaluation for each model type
        for model_type in ['random_forest', 'linear', 'arima', 'prophet']:
            model = TimeSeriesModel(model_type=model_type)
            model.train(self.X, self.y)
            metrics = model.evaluate(self.X, self.y)
            self.assertIn('mse', metrics)
            self.assertIn('rmse', metrics)
            self.assertIn('mae', metrics)
            
    def test_model_save_load(self):
        # Test saving and loading model
        model = TimeSeriesModel(model_type='random_forest')
        model.train(self.X, self.y)
        
        # Save model
        model.save('test_model.joblib')
        
        # Load model
        new_model = TimeSeriesModel(model_type='random_forest')
        new_model.load('test_model.joblib')
        
        # Compare predictions
        original_predictions = model.predict(self.X)
        loaded_predictions = new_model.predict(self.X)
        np.testing.assert_array_almost_equal(original_predictions, loaded_predictions)
        
    def test_model_comparison(self):
        # Test comparing multiple models
        results = compare_models(self.X, self.y)
        self.assertIn('random_forest', results)
        self.assertIn('linear', results)
        self.assertIn('arima', results)
        self.assertIn('prophet', results)

if __name__ == '__main__':
    unittest.main() 