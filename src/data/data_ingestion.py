import pandas as pd
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_retail_data(file_path):
    """
    Read and preprocess the retail data from Excel file.
    
    Args:
        file_path (str): Path to the Excel file
        
    Returns:
        pd.DataFrame: Preprocessed retail data
    """
    try:
        logger.info(f"Reading data from {file_path}")
        df = pd.read_excel(file_path)
        
        # Basic preprocessing
        df = df.dropna(subset=['CustomerID'])
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
        
        # Remove negative quantities (returns)
        df = df[df['Quantity'] > 0]
        
        # Create date features
        df['Year'] = df['InvoiceDate'].dt.year
        df['Month'] = df['InvoiceDate'].dt.month
        df['Day'] = df['InvoiceDate'].dt.day
        df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
        
        logger.info("Data preprocessing completed successfully")
        return df
        
    except Exception as e:
        logger.error(f"Error reading data: {str(e)}")
        raise

def aggregate_daily_revenue(df):
    """
    Aggregate revenue by day.
    
    Args:
        df (pd.DataFrame): Preprocessed retail data
        
    Returns:
        pd.DataFrame: Daily revenue data
    """
    daily_revenue = df.groupby('InvoiceDate')['TotalPrice'].sum().reset_index()
    daily_revenue.columns = ['Date', 'Revenue']
    return daily_revenue

def prepare_time_series_data(df, country=None):
    """
    Prepare time series data for modeling.
    
    Args:
        df (pd.DataFrame): Preprocessed retail data
        country (str, optional): Country to filter by
        
    Returns:
        pd.DataFrame: Time series data ready for modeling
    """
    if country:
        df = df[df['Country'] == country]
    
    # Aggregate by day
    daily_data = aggregate_daily_revenue(df)
    
    # Create features
    daily_data['DayOfWeek'] = daily_data['Date'].dt.dayofweek
    daily_data['Month'] = daily_data['Date'].dt.month
    daily_data['Year'] = daily_data['Date'].dt.year
    
    # Create lag features
    for lag in [1, 7, 30]:
        daily_data[f'Revenue_lag_{lag}'] = daily_data['Revenue'].shift(lag)
    
    # Create rolling features
    for window in [7, 30]:
        daily_data[f'Revenue_rolling_mean_{window}'] = daily_data['Revenue'].rolling(window=window).mean()
        daily_data[f'Revenue_rolling_std_{window}'] = daily_data['Revenue'].rolling(window=window).std()
    
    return daily_data.dropna() 