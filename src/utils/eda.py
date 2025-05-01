import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def plot_daily_revenue(df: pd.DataFrame, title: str = "Daily Revenue Over Time"):
    """
    Plot daily revenue over time.
    
    Args:
        df (pd.DataFrame): DataFrame with 'Date' and 'Revenue' columns
        title (str): Plot title
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Revenue'])
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Revenue')
    plt.grid(True)
    plt.tight_layout()
    return plt

def plot_revenue_by_country(df: pd.DataFrame, top_n: int = 10):
    """
    Plot revenue by country.
    
    Args:
        df (pd.DataFrame): DataFrame with 'Country' and 'TotalPrice' columns
        top_n (int): Number of top countries to show
    """
    country_revenue = df.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False)
    top_countries = country_revenue.head(top_n)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_countries.index, y=top_countries.values)
    plt.title(f'Top {top_n} Countries by Revenue')
    plt.xlabel('Country')
    plt.ylabel('Total Revenue')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt

def plot_seasonal_decomposition(df: pd.DataFrame, period: int = 30):
    """
    Plot seasonal decomposition of revenue.
    
    Args:
        df (pd.DataFrame): DataFrame with 'Date' and 'Revenue' columns
        period (int): Period for seasonal decomposition
    """
    from statsmodels.tsa.seasonal import seasonal_decompose
    
    # Ensure data is sorted by date
    df = df.sort_values('Date')
    
    # Set date as index
    df = df.set_index('Date')
    
    # Perform decomposition
    decomposition = seasonal_decompose(df['Revenue'], period=period)
    
    # Plot
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 8))
    
    decomposition.observed.plot(ax=ax1)
    ax1.set_title('Observed')
    
    decomposition.trend.plot(ax=ax2)
    ax2.set_title('Trend')
    
    decomposition.seasonal.plot(ax=ax3)
    ax3.set_title('Seasonal')
    
    decomposition.resid.plot(ax=ax4)
    ax4.set_title('Residual')
    
    plt.tight_layout()
    return plt

def plot_correlation_heatmap(df: pd.DataFrame, features: Optional[List[str]] = None):
    """
    Plot correlation heatmap of features.
    
    Args:
        df (pd.DataFrame): DataFrame with features
        features (List[str], optional): List of features to include in heatmap
    """
    if features is None:
        features = df.columns.tolist()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[features].corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    return plt

def save_plots(plots: List[plt.Figure], output_dir: str = "plots"):
    """
    Save plots to output directory.
    
    Args:
        plots (List[plt.Figure]): List of matplotlib figures
        output_dir (str): Output directory path
    """
    import os
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, plot in enumerate(plots):
        plot.savefig(f"{output_dir}/plot_{i+1}.png")
        plt.close(plot) 