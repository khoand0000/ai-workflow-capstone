import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelMonitor:
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize model monitor.
        
        Args:
            log_dir (str): Directory to store monitoring logs
        """
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
    def log_prediction(self, date: str, country: str, 
                      prediction: float, actual: Optional[float] = None):
        """
        Log a prediction.
        
        Args:
            date (str): Date of prediction
            country (str): Country for prediction
            prediction (float): Predicted value
            actual (float, optional): Actual value if available
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'date': date,
            'country': country,
            'prediction': prediction,
            'actual': actual
        }
        
        # Save to CSV
        log_file = os.path.join(self.log_dir, f"predictions_{country}.csv")
        df = pd.DataFrame([log_entry])
        
        if os.path.exists(log_file):
            df.to_csv(log_file, mode='a', header=False, index=False)
        else:
            df.to_csv(log_file, index=False)
            
        logger.info(f"Logged prediction for {date}, {country}")
        
    def calculate_metrics(self, country: Optional[str] = None) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Args:
            country (str, optional): Country to filter by
            
        Returns:
            Dict[str, float]: Performance metrics
        """
        metrics = {}
        
        if country:
            log_files = [os.path.join(self.log_dir, f"predictions_{country}.csv")]
        else:
            log_files = [f for f in os.listdir(self.log_dir) if f.startswith('predictions_')]
            
        for log_file in log_files:
            df = pd.read_csv(log_file)
            
            if 'actual' in df.columns and not df['actual'].isna().all():
                metrics[os.path.basename(log_file)] = {
                    'mse': np.mean((df['prediction'] - df['actual'])**2),
                    'rmse': np.sqrt(np.mean((df['prediction'] - df['actual'])**2)),
                    'mae': np.mean(np.abs(df['prediction'] - df['actual']))
                }
                
        return metrics
        
    def plot_performance(self, country: Optional[str] = None):
        """
        Plot model performance.
        
        Args:
            country (str, optional): Country to filter by
        """
        if country:
            log_files = [os.path.join(self.log_dir, f"predictions_{country}.csv")]
        else:
            log_files = [f for f in os.listdir(self.log_dir) if f.startswith('predictions_')]
            
        for log_file in log_files:
            df = pd.read_csv(log_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            plt.figure(figsize=(12, 6))
            plt.plot(df['timestamp'], df['prediction'], label='Prediction')
            
            if 'actual' in df.columns and not df['actual'].isna().all():
                plt.plot(df['timestamp'], df['actual'], label='Actual')
                
            plt.title(f'Model Performance - {os.path.basename(log_file)}')
            plt.xlabel('Date')
            plt.ylabel('Revenue')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            # Save plot
            plot_file = os.path.join(self.log_dir, f"performance_{os.path.basename(log_file).replace('.csv', '.png')}")
            plt.savefig(plot_file)
            plt.close()
            
            logger.info(f"Saved performance plot to {plot_file}")
            
    def generate_report(self, output_file: str = "performance_report.html"):
        """
        Generate HTML performance report.
        
        Args:
            output_file (str): Output file path
        """
        metrics = self.calculate_metrics()
        
        html = """
        <html>
        <head>
            <title>Model Performance Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                img { max-width: 100%; height: auto; }
            </style>
        </head>
        <body>
            <h1>Model Performance Report</h1>
            <h2>Metrics</h2>
            <table>
                <tr>
                    <th>Country</th>
                    <th>MSE</th>
                    <th>RMSE</th>
                    <th>MAE</th>
                </tr>
        """
        
        for country, country_metrics in metrics.items():
            html += f"""
                <tr>
                    <td>{country}</td>
                    <td>{country_metrics['mse']:.2f}</td>
                    <td>{country_metrics['rmse']:.2f}</td>
                    <td>{country_metrics['mae']:.2f}</td>
                </tr>
            """
            
        html += """
            </table>
            <h2>Performance Plots</h2>
        """
        
        for log_file in os.listdir(self.log_dir):
            if log_file.startswith('predictions_'):
                plot_file = f"performance_{log_file.replace('.csv', '.png')}"
                if os.path.exists(os.path.join(self.log_dir, plot_file)):
                    html += f"""
                        <h3>{log_file}</h3>
                        <img src="{plot_file}" alt="Performance Plot">
                    """
                    
        html += """
        </body>
        </html>
        """
        
        with open(os.path.join(self.log_dir, output_file), 'w') as f:
            f.write(html)
            
        logger.info(f"Generated performance report: {output_file}") 