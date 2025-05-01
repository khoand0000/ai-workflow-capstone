import unittest
import pandas as pd
import numpy as np
from src.data.data_ingestion import read_retail_data, aggregate_daily_revenue, prepare_time_series_data

class TestDataIngestion(unittest.TestCase):
    def setUp(self):
        # Create sample data for testing
        self.sample_data = pd.DataFrame({
            'InvoiceDate': pd.date_range(start='2023-01-01', periods=100, freq='D'),
            'CustomerID': np.random.randint(1000, 2000, 100),
            'Quantity': np.random.randint(1, 10, 100),
            'UnitPrice': np.random.uniform(1, 100, 100),
            'Country': np.random.choice(['UK', 'US', 'Germany'], 100)
        })
        
    def test_read_retail_data(self):
        # Test reading and preprocessing data
        df = read_retail_data('data/Online Retail.xlsx')
        
        # Check if required columns exist
        self.assertIn('InvoiceDate', df.columns)
        self.assertIn('CustomerID', df.columns)
        self.assertIn('Quantity', df.columns)
        self.assertIn('UnitPrice', df.columns)
        
        # Check if date features are created
        self.assertIn('Year', df.columns)
        self.assertIn('Month', df.columns)
        self.assertIn('Day', df.columns)
        self.assertIn('DayOfWeek', df.columns)
        
    def test_aggregate_daily_revenue(self):
        # Test daily revenue aggregation
        df = self.sample_data.copy()
        df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
        
        daily_revenue = aggregate_daily_revenue(df)
        
        # Check if output has correct columns
        self.assertIn('Date', daily_revenue.columns)
        self.assertIn('Revenue', daily_revenue.columns)
        
        # Check if revenue is calculated correctly
        expected_revenue = df.groupby('InvoiceDate')['TotalPrice'].sum()
        pd.testing.assert_series_equal(daily_revenue.set_index('Date')['Revenue'], expected_revenue)
        
    def test_prepare_time_series_data(self):
        # Test time series data preparation
        df = self.sample_data.copy()
        df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
        
        # Test without country filter
        time_series_data = prepare_time_series_data(df)
        self.assertIn('Date', time_series_data.columns)
        self.assertIn('Revenue', time_series_data.columns)
        self.assertIn('DayOfWeek', time_series_data.columns)
        self.assertIn('Month', time_series_data.columns)
        self.assertIn('Year', time_series_data.columns)
        
        # Test with country filter
        time_series_data_uk = prepare_time_series_data(df, country='UK')
        self.assertTrue(all(time_series_data_uk['Country'] == 'UK'))
        
        # Check if lag features are created
        for lag in [1, 7, 30]:
            self.assertIn(f'Revenue_lag_{lag}', time_series_data.columns)
            
        # Check if rolling features are created
        for window in [7, 30]:
            self.assertIn(f'Revenue_rolling_mean_{window}', time_series_data.columns)
            self.assertIn(f'Revenue_rolling_std_{window}', time_series_data.columns)

if __name__ == '__main__':
    unittest.main() 