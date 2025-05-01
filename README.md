# Online Retail Revenue Prediction

This project implements a time series forecasting system for online retail revenue prediction. It includes data ingestion, exploratory data analysis, model training, and a REST API for predictions.

## Project Structure

```
.
├── src/
│   ├── api/
│   │   └── app.py              # Flask API implementation
│   ├── data/
│   │   └── data_ingestion.py   # Data loading and preprocessing
│   ├── models/
│   │   └── time_series_model.py # Time series model implementation
│   ├── tests/
│   │   ├── test_api.py         # API tests
│   │   ├── test_model.py       # Model tests
│   │   └── test_data_ingestion.py # Data ingestion tests
│   └── utils/
│       ├── eda.py              # Exploratory data analysis
│       └── monitor.py          # Model performance monitoring
├── data/
│   └── Online Retail.xlsx      # Input data
├── models/                     # Saved models
├── logs/                       # Prediction logs
├── plots/                      # Generated plots
├── Dockerfile                  # Docker configuration
├── requirements.txt            # Python dependencies
└── run_tests.py               # Test runner
```

## Features

- Data ingestion and preprocessing
- Exploratory data analysis with visualizations
- Multiple time series models (Random Forest, Linear Regression, ARIMA, Prophet)
- REST API for model training and predictions
- Model performance monitoring and reporting
- Unit tests for all components
- Docker containerization

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd online-retail-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place the Online Retail.xlsx file in the data/ directory.

## Usage

### Running the API

```bash
python src/api/app.py
```

The API will be available at http://localhost:5000 with the following endpoints:

- POST /train - Train the model
- POST /predict - Make predictions
- GET /logs - Get API logs

### Running Tests

```bash
python run_tests.py
```

### Running with Docker

1. Build the Docker image:
```bash
docker build -t retail-prediction .
```

2. Run the container:
```bash
docker run -p 5000:5000 retail-prediction
```

## Model Monitoring

The system includes a monitoring module that:

- Logs all predictions
- Calculates performance metrics (MSE, RMSE, MAE)
- Generates performance plots
- Creates HTML reports

To generate a performance report:
```python
from src.utils.monitor import ModelMonitor

monitor = ModelMonitor()
monitor.plot_performance()
monitor.generate_report()
```

## Requirements

- Python 3.9+
- See requirements.txt for Python dependencies

## License

This project is licensed under the MIT License - see the LICENSE file for details. 