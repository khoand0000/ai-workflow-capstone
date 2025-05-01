from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os
from typing import Dict, Any
import joblib

from src.models.time_series_model import TimeSeriesModel
from src.data.data_ingestion import read_retail_data, prepare_time_series_data

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='api.log'
)
logger = logging.getLogger(__name__)

# Global variables
model = None
data_path = "data/Online Retail.xlsx"

@app.route('/train', methods=['POST'])
def train():
    """
    Train the model with specified parameters.
    """
    try:
        global model
        
        # Get parameters from request
        params = request.get_json()
        model_type = params.get('model_type', 'random_forest')
        country = params.get('country', None)
        
        # Read and prepare data
        df = read_retail_data(data_path)
        time_series_data = prepare_time_series_data(df, country)
        
        # Prepare features and target
        X = time_series_data.drop(['Date', 'Revenue'], axis=1)
        y = time_series_data['Revenue']
        
        # Train model
        model = TimeSeriesModel(model_type=model_type)
        model.train(X, y)
        
        # Save model
        model.save(f"models/{model_type}_model.joblib")
        
        # Log training
        logger.info(f"Trained {model_type} model for country: {country}")
        
        return jsonify({
            "status": "success",
            "message": f"Model trained successfully with type: {model_type}"
        })
        
    except Exception as e:
        logger.error(f"Error in training: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """
    Make predictions for a specific date and country.
    """
    try:
        global model
        
        if model is None:
            return jsonify({
                "status": "error",
                "message": "Model not trained. Please train the model first."
            }), 400
            
        # Get parameters from request
        params = request.get_json()
        date = params.get('date')
        country = params.get('country', None)
        
        # Read and prepare data
        df = read_retail_data(data_path)
        time_series_data = prepare_time_series_data(df, country)
        
        # Prepare features for prediction
        X = time_series_data.drop(['Date', 'Revenue'], axis=1)
        
        # Make prediction
        prediction = model.predict(X)
        
        # Log prediction
        logger.info(f"Made prediction for date: {date}, country: {country}")
        
        return jsonify({
            "status": "success",
            "prediction": float(prediction[-1]),
            "date": date,
            "country": country
        })
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/logs', methods=['GET'])
def get_logs():
    """
    Get API logs.
    """
    try:
        with open('api.log', 'r') as f:
            logs = f.readlines()
            
        return jsonify({
            "status": "success",
            "logs": logs
        })
        
    except Exception as e:
        logger.error(f"Error reading logs: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) 