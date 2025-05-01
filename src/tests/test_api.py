import unittest
import json
from src.api.app import app
import pandas as pd
import numpy as np
import os

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        
    def test_train_endpoint(self):
        # Test training with default parameters
        response = self.app.post('/train', 
                               data=json.dumps({}),
                               content_type='application/json')
        self.assertEqual(response.status_code, 200)
        
        # Test training with specific parameters
        response = self.app.post('/train',
                               data=json.dumps({
                                   'model_type': 'random_forest',
                                   'country': 'United Kingdom'
                               }),
                               content_type='application/json')
        self.assertEqual(response.status_code, 200)
        
    def test_predict_endpoint(self):
        # First train the model
        self.app.post('/train',
                     data=json.dumps({}),
                     content_type='application/json')
        
        # Test prediction
        response = self.app.post('/predict',
                               data=json.dumps({
                                   'date': '2023-01-01',
                                   'country': 'United Kingdom'
                               }),
                               content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('prediction', data)
        
    def test_logs_endpoint(self):
        response = self.app.get('/logs')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('logs', data)

if __name__ == '__main__':
    unittest.main() 