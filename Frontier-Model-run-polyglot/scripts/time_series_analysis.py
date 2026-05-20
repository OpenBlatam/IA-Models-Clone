#!/usr/bin/env python3
"""
Advanced Time Series Analysis System for Frontier Model Training
Provides comprehensive time series algorithms, forecasting, and anomaly detection capabilities.
"""

import os
import json
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
import sqlite3
from contextlib import contextmanager
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.exponential_smoothing import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HW
import prophet
from prophet import Prophet
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import joblib
import pickle
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

console = Console()

class TSTask(Enum):
    """Time series tasks."""
    FORECASTING = "forecasting"
    ANOMALY_DETECTION = "anomaly_detection"
    TREND_ANALYSIS = "trend_analysis"
    SEASONALITY_ANALYSIS = "seasonality_analysis"
    DECOMPOSITION = "decomposition"
    STATIONARITY_TESTING = "stationarity_testing"
    AUTOCORRELATION_ANALYSIS = "autocorrelation_analysis"
    CROSS_CORRELATION_ANALYSIS = "cross_correlation_analysis"
    REGIME_DETECTION = "regime_detection"
    VOLATILITY_MODELING = "volatility_modeling"
    COINTEGRATION_ANALYSIS = "cointegration_analysis"
    GRANGER_CAUSALITY = "granger_causality"
    SPECTRAL_ANALYSIS = "spectral_analysis"
    WAVELET_ANALYSIS = "wavelet_analysis"
    CHANGE_POINT_DETECTION = "change_point_detection"

class TSModel(Enum):
    """Time series models."""
    # Statistical models
    ARIMA = "arima"
    SARIMA = "sarima"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    HOLT_WINTERS = "holt_winters"
    GARCH = "garch"
    VAR = "var"
    VECM = "vecm"
    
    # Machine learning models
    LINEAR_REGRESSION = "linear_regression"
    RIDGE_REGRESSION = "ridge_regression"
    LASSO_REGRESSION = "lasso_regression"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    SVR = "svr"
    
    # Deep learning models
    LSTM = "lstm"
    GRU = "gru"
    CNN_LSTM = "cnn_lstm"
    TRANSFORMER = "transformer"
    WAVENET = "wavenet"
    TCN = "tcn"
    
    # Specialized models
    PROPHET = "prophet"
    TBATS = "tbats"
    AUTO_ARIMA = "auto_arima"
    ETS = "ets"

class TSDecomposition(Enum):
    """Time series decomposition methods."""
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"
    STL = "stl"
    X13ARIMA = "x13arima"
    SEASONAL_DECOMPOSE = "seasonal_decompose"

class TSStationarityTest(Enum):
    """Stationarity tests."""
    ADF = "adf"  # Augmented Dickey-Fuller
    KPSS = "kpss"  # Kwiatkowski-Phillips-Schmidt-Shin
    PP = "pp"  # Phillips-Perron
    ZA = "za"  # Zivot-Andrews

class TSAnomalyMethod(Enum):
    """Anomaly detection methods."""
    STATISTICAL = "statistical"
    ISOLATION_FOREST = "isolation_forest"
    ONE_CLASS_SVM = "one_class_svm"
    LOCAL_OUTLIER_FACTOR = "local_outlier_factor"
    DBSCAN = "dbscan"
    LSTM_AUTOENCODER = "lstm_autoencoder"
    VAR_AUTOENCODER = "var_autoencoder"
    PROPHET_ANOMALY = "prophet_anomaly"

@dataclass
class TSConfig:
    """Time series configuration."""
    task: TSTask = TSTask.FORECASTING
    model: TSModel = TSModel.ARIMA
    forecast_horizon: int = 30
    lookback_window: int = 100
    seasonality_period: int = 12
    decomposition_method: TSDecomposition = TSDecomposition.ADDITIVE
    stationarity_test: TSStationarityTest = TSStationarityTest.ADF
    anomaly_method: TSAnomalyMethod = TSAnomalyMethod.STATISTICAL
    enable_differencing: bool = True
    enable_seasonal_adjustment: bool = True
    enable_outlier_detection: bool = True
    enable_feature_engineering: bool = True
    enable_cross_validation: bool = True
    enable_hyperparameter_tuning: bool = True
    enable_model_ensemble: bool = True
    enable_uncertainty_quantification: bool = True
    enable_visualization: bool = True
    device: str = "auto"

@dataclass
class TimeSeriesData:
    """Time series data container."""
    data_id: str
    timestamps: List[datetime]
    values: List[float]
    metadata: Dict[str, Any] = None
    processed_data: Optional[pd.DataFrame] = None
    features: Optional[np.ndarray] = None

@dataclass
class TSModelResult:
    """Time series model result."""
    result_id: str
    task: TSTask
    model: TSModel
    performance_metrics: Dict[str, float]
    forecasts: Optional[List[float]] = None
    anomalies: Optional[List[int]] = None
    decomposition: Optional[Dict[str, List[float]]] = None
    model_state: Dict[str, Any] = None
    created_at: datetime = None

class TimeSeriesProcessor:
    """Time series data processor."""
    
    def __init__(self, config: TSConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def process_time_series(self, data: TimeSeriesData) -> pd.DataFrame:
        """Process time series data."""
        console.print("[blue]Processing time series data...[/blue]")
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': data.timestamps,
            'value': data.values
        })
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Detect and handle outliers
        if self.config.enable_outlier_detection:
            df = self._handle_outliers(df)
        
        # Check stationarity
        if self.config.enable_differencing:
            df = self._make_stationary(df)
        
        # Seasonal adjustment
        if self.config.enable_seasonal_adjustment:
            df = self._seasonal_adjustment(df)
        
        # Feature engineering
        if self.config.enable_feature_engineering:
            df = self._engineer_features(df)
        
        data.processed_data = df
        console.print("[green]Time series processing completed[/green]")
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in time series."""
        # Forward fill, then backward fill
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # If still missing values, interpolate
        if df.isnull().any().any():
            df = df.interpolate(method='linear')
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and handle outliers."""
        # Use IQR method
        Q1 = df['value'].quantile(0.25)
        Q3 = df['value'].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap outliers
        df['value'] = df['value'].clip(lower=lower_bound, upper=upper_bound)
        
        return df
    
    def _make_stationary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make time series stationary."""
        # Test stationarity
        adf_result = adfuller(df['value'])
        
        if adf_result[1] > 0.05:  # Not stationary
            # Apply differencing
            df['value'] = df['value'].diff()
            df = df.dropna()
            
            # Test again
            adf_result = adfuller(df['value'])
            if adf_result[1] > 0.05:
                # Apply second differencing
                df['value'] = df['value'].diff()
                df = df.dropna()
        
        return df
    
    def _seasonal_adjustment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply seasonal adjustment."""
        try:
            # Seasonal decomposition
            decomposition = seasonal_decompose(df['value'], model=self.config.decomposition_method.value, period=self.config.seasonality_period)
            
            # Remove seasonal component
            df['value'] = df['value'] - decomposition.seasonal
            
        except Exception as e:
            self.logger.warning(f"Seasonal adjustment failed: {e}")
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for time series."""
        # Lag features
        for lag in [1, 2, 3, 7, 14, 30]:
            df[f'lag_{lag}'] = df['value'].shift(lag)
        
        # Rolling statistics
        for window in [7, 14, 30]:
            df[f'rolling_mean_{window}'] = df['value'].rolling(window=window).mean()
            df[f'rolling_std_{window}'] = df['value'].rolling(window=window).std()
            df[f'rolling_min_{window}'] = df['value'].rolling(window=window).min()
            df[f'rolling_max_{window}'] = df['value'].rolling(window=window).max()
        
        # Time-based features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        df['year'] = df.index.year
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df

class TSModelFactory:
    """Time series model factory."""
    
    def __init__(self, config: TSConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
    
    def create_model(self, data: pd.DataFrame) -> Any:
        """Create time series model."""
        console.print(f"[blue]Creating {self.config.model.value} model for {self.config.task.value}...[/blue]")
        
        try:
            if self.config.task == TSTask.FORECASTING:
                return self._create_forecasting_model(data)
            elif self.config.task == TSTask.ANOMALY_DETECTION:
                return self._create_anomaly_model(data)
            elif self.config.task == TSTask.DECOMPOSITION:
                return self._create_decomposition_model(data)
            else:
                return self._create_forecasting_model(data)
                
        except Exception as e:
            self.logger.error(f"Model creation failed: {e}")
            return self._create_fallback_model(data)
    
    def _create_forecasting_model(self, data: pd.DataFrame) -> Any:
        """Create forecasting model."""
        if self.config.model == TSModel.ARIMA:
            return self._create_arima_model(data)
        elif self.config.model == TSModel.EXPONENTIAL_SMOOTHING:
            return self._create_exponential_smoothing_model(data)
        elif self.config.model == TSModel.PROPHET:
            return self._create_prophet_model(data)
        elif self.config.model == TSModel.LSTM:
            return self._create_lstm_model(data)
        elif self.config.model == TSModel.XGBOOST:
            return self._create_xgboost_model(data)
        else:
            return self._create_arima_model(data)
    
    def _create_arima_model(self, data: pd.DataFrame) -> Any:
        """Create ARIMA model."""
        # Auto ARIMA
        try:
            from pmdarima import auto_arima
            model = auto_arima(data['value'], seasonal=True, m=self.config.seasonality_period)
        except ImportError:
            # Fallback to manual ARIMA
            model = ARIMA(data['value'], order=(1, 1, 1))
            model = model.fit()
        
        return model
    
    def _create_exponential_smoothing_model(self, data: pd.DataFrame) -> Any:
        """Create exponential smoothing model."""
        model = ExponentialSmoothing(
            data['value'],
            trend='add',
            seasonal='add',
            seasonal_periods=self.config.seasonality_period
        )
        return model.fit()
    
    def _create_prophet_model(self, data: pd.DataFrame) -> Any:
        """Create Prophet model."""
        # Prepare data for Prophet
        prophet_data = pd.DataFrame({
            'ds': data.index,
            'y': data['value']
        })
        
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True
        )
        model.fit(prophet_data)
        
        return model
    
    def _create_lstm_model(self, data: pd.DataFrame) -> Any:
        """Create LSTM model."""
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, output_size):
                super().__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, output_size)
            
            def forward(self, x):
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
                
                out, _ = self.lstm(x, (h0, c0))
                out = self.fc(out[:, -1, :])
                return out
        
        # Determine input size based on features
        feature_cols = [col for col in data.columns if col != 'value']
        input_size = len(feature_cols) + 1  # +1 for the value column
        
        model = LSTMModel(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            output_size=1
        )
        
        return model.to(self.device)
    
    def _create_xgboost_model(self, data: pd.DataFrame) -> Any:
        """Create XGBoost model."""
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        return model
    
    def _create_anomaly_model(self, data: pd.DataFrame) -> Any:
        """Create anomaly detection model."""
        if self.config.anomaly_method == TSAnomalyMethod.STATISTICAL:
            return self._create_statistical_anomaly_model(data)
        elif self.config.anomaly_method == TSAnomalyMethod.ISOLATION_FOREST:
            return self._create_isolation_forest_model(data)
        elif self.config.anomaly_method == TSAnomalyMethod.LSTM_AUTOENCODER:
            return self._create_lstm_autoencoder_model(data)
        else:
            return self._create_statistical_anomaly_model(data)
    
    def _create_statistical_anomaly_model(self, data: pd.DataFrame) -> Any:
        """Create statistical anomaly detection model."""
        # Z-score based anomaly detection
        mean = data['value'].mean()
        std = data['value'].std()
        threshold = 3  # 3-sigma rule
        
        anomalies = np.abs(data['value'] - mean) > threshold * std
        return {'anomalies': anomalies, 'mean': mean, 'std': std, 'threshold': threshold}
    
    def _create_isolation_forest_model(self, data: pd.DataFrame) -> Any:
        """Create Isolation Forest model."""
        from sklearn.ensemble import IsolationForest
        
        model = IsolationForest(contamination=0.1, random_state=42)
        anomalies = model.fit_predict(data[['value']])
        
        return {'model': model, 'anomalies': anomalies == -1}
    
    def _create_lstm_autoencoder_model(self, data: pd.DataFrame) -> Any:
        """Create LSTM autoencoder model."""
        class LSTMAutoencoder(nn.Module):
            def __init__(self, input_size, hidden_size):
                super().__init__()
                self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
                self.decoder = nn.LSTM(hidden_size, input_size, batch_first=True)
            
            def forward(self, x):
                encoded, _ = self.encoder(x)
                decoded, _ = self.decoder(encoded)
                return decoded
        
        model = LSTMAutoencoder(input_size=1, hidden_size=32)
        return model.to(self.device)
    
    def _create_decomposition_model(self, data: pd.DataFrame) -> Any:
        """Create decomposition model."""
        decomposition = seasonal_decompose(
            data['value'],
            model=self.config.decomposition_method.value,
            period=self.config.seasonality_period
        )
        
        return {
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'residual': decomposition.resid,
            'observed': decomposition.observed
        }
    
    def _create_fallback_model(self, data: pd.DataFrame) -> Any:
        """Create fallback model."""
        return self._create_arima_model(data)

class TSTrainer:
    """Time series training engine."""
    
    def __init__(self, config: TSConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
    
    def train_model(self, model: Any, data: pd.DataFrame) -> Dict[str, Any]:
        """Train time series model."""
        console.print(f"[blue]Training {self.config.model.value} model...[/blue]")
        
        try:
            if self.config.task == TSTask.FORECASTING:
                return self._train_forecasting_model(model, data)
            elif self.config.task == TSTask.ANOMALY_DETECTION:
                return self._train_anomaly_model(model, data)
            elif self.config.task == TSTask.DECOMPOSITION:
                return self._train_decomposition_model(model, data)
            else:
                return self._train_forecasting_model(model, data)
                
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return {'error': str(e)}
    
    def _train_forecasting_model(self, model: Any, data: pd.DataFrame) -> Dict[str, Any]:
        """Train forecasting model."""
        if isinstance(model, dict) and 'anomalies' in model:
            # Statistical anomaly model - no training needed
            return {'model': model}
        
        # Split data
        split_idx = int(0.8 * len(data))
        train_data = data[:split_idx]
        test_data = data[split_idx:]
        
        if self.config.model == TSModel.LSTM:
            return self._train_lstm_model(model, train_data, test_data)
        elif self.config.model == TSModel.XGBOOST:
            return self._train_xgboost_model(model, train_data, test_data)
        else:
            # Statistical models - already fitted
            return {'model': model}
    
    def _train_lstm_model(self, model: nn.Module, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Train LSTM model."""
        # Prepare data
        feature_cols = [col for col in train_data.columns if col != 'value']
        X_train = train_data[feature_cols].values
        y_train = train_data['value'].values
        
        X_test = test_data[feature_cols].values
        y_test = test_data['value'].values
        
        # Create sequences
        def create_sequences(X, y, seq_length):
            X_seq, y_seq = [], []
            for i in range(seq_length, len(X)):
                X_seq.append(X[i-seq_length:i])
                y_seq.append(y[i])
            return np.array(X_seq), np.array(y_seq)
        
        seq_length = self.config.lookback_window
        X_train_seq, y_train_seq = create_sequences(X_train, y_train, seq_length)
        X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_length)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_seq).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train_seq).to(self.device)
        X_test_tensor = torch.FloatTensor(X_test_seq).to(self.device)
        y_test_tensor = torch.FloatTensor(y_test_seq).to(self.device)
        
        # Training
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs.squeeze(), y_train_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                console.print(f"[blue]Epoch {epoch}, Loss: {loss.item():.4f}[/blue]")
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            train_pred = model(X_train_tensor).squeeze().cpu().numpy()
            test_pred = model(X_test_tensor).squeeze().cpu().numpy()
        
        train_mse = mean_squared_error(y_train_seq, train_pred)
        test_mse = mean_squared_error(y_test_seq, test_pred)
        
        return {
            'model': model,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'predictions': test_pred
        }
    
    def _train_xgboost_model(self, model: Any, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Train XGBoost model."""
        # Prepare data
        feature_cols = [col for col in train_data.columns if col != 'value']
        X_train = train_data[feature_cols]
        y_train = train_data['value']
        
        X_test = test_data[feature_cols]
        y_test = test_data['value']
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        return {
            'model': model,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'predictions': test_pred
        }
    
    def _train_anomaly_model(self, model: Any, data: pd.DataFrame) -> Dict[str, Any]:
        """Train anomaly detection model."""
        if isinstance(model, dict) and 'anomalies' in model:
            # Statistical model - already "trained"
            return {'model': model}
        elif isinstance(model, dict) and 'model' in model:
            # Isolation Forest - already fitted
            return {'model': model}
        else:
            # LSTM Autoencoder
            return self._train_lstm_autoencoder(model, data)
    
    def _train_lstm_autoencoder(self, model: nn.Module, data: pd.DataFrame) -> Dict[str, Any]:
        """Train LSTM autoencoder."""
        # Prepare data
        values = data['value'].values.reshape(-1, 1)
        
        # Create sequences
        def create_sequences(data, seq_length):
            sequences = []
            for i in range(seq_length, len(data)):
                sequences.append(data[i-seq_length:i])
            return np.array(sequences)
        
        seq_length = self.config.lookback_window
        sequences = create_sequences(values, seq_length)
        
        # Convert to tensors
        sequences_tensor = torch.FloatTensor(sequences).to(self.device)
        
        # Training
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            reconstructed = model(sequences_tensor)
            loss = criterion(reconstructed, sequences_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                console.print(f"[blue]Epoch {epoch}, Loss: {loss.item():.4f}[/blue]")
        
        return {'model': model}
    
    def _train_decomposition_model(self, model: Any, data: pd.DataFrame) -> Dict[str, Any]:
        """Train decomposition model."""
        # Decomposition models don't need training
        return {'model': model}

class TSSystem:
    """Main time series system."""
    
    def __init__(self, config: TSConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.processor = TimeSeriesProcessor(config)
        self.model_factory = TSModelFactory(config)
        self.trainer = TSTrainer(config)
        
        # Initialize database
        self.db_path = self._init_database()
        
        # Results storage
        self.ts_results: Dict[str, TSModelResult] = {}
    
    def _init_database(self) -> str:
        """Initialize time series database."""
        db_path = Path("./time_series_analysis.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ts_models (
                    model_id TEXT PRIMARY KEY,
                    task TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    performance_metrics TEXT NOT NULL,
                    forecasts TEXT,
                    anomalies TEXT,
                    decomposition TEXT,
                    model_state TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """)
        
        return str(db_path)
    
    def run_ts_experiment(self, data: TimeSeriesData) -> TSModelResult:
        """Run complete time series experiment."""
        console.print(f"[blue]Starting TS experiment with {self.config.task.value}...[/blue]")
        
        start_time = time.time()
        result_id = f"ts_exp_{int(time.time())}"
        
        # Process data
        processed_data = self.processor.process_time_series(data)
        
        # Create model
        model = self.model_factory.create_model(processed_data)
        
        # Train model
        training_result = self.trainer.train_model(model, processed_data)
        
        # Generate results based on task
        if self.config.task == TSTask.FORECASTING:
            forecasts = self._generate_forecasts(training_result['model'], processed_data)
            performance_metrics = self._evaluate_forecasts(forecasts, processed_data)
        elif self.config.task == TSTask.ANOMALY_DETECTION:
            anomalies = self._detect_anomalies(training_result['model'], processed_data)
            performance_metrics = self._evaluate_anomalies(anomalies, processed_data)
        elif self.config.task == TSTask.DECOMPOSITION:
            decomposition = self._perform_decomposition(training_result['model'], processed_data)
            performance_metrics = {'decomposition_completed': True}
        else:
            forecasts = None
            anomalies = None
            decomposition = None
            performance_metrics = {}
        
        # Create TS result
        ts_result = TSModelResult(
            result_id=result_id,
            task=self.config.task,
            model=self.config.model,
            performance_metrics=performance_metrics,
            forecasts=forecasts,
            anomalies=anomalies,
            decomposition=decomposition,
            model_state={
                'data_length': len(processed_data),
                'features_count': len(processed_data.columns),
                'training_time': time.time() - start_time
            },
            created_at=datetime.now()
        )
        
        # Store result
        self.ts_results[result_id] = ts_result
        
        # Save to database
        self._save_ts_result(ts_result)
        
        experiment_time = time.time() - start_time
        console.print(f"[green]TS experiment completed in {experiment_time:.2f} seconds[/green]")
        
        return ts_result
    
    def _generate_forecasts(self, model: Any, data: pd.DataFrame) -> List[float]:
        """Generate forecasts."""
        try:
            if hasattr(model, 'forecast'):
                # Statistical models
                forecasts = model.forecast(steps=self.config.forecast_horizon)
                return forecasts.tolist() if hasattr(forecasts, 'tolist') else list(forecasts)
            elif hasattr(model, 'predict'):
                # Prophet model
                future = model.make_future_dataframe(periods=self.config.forecast_horizon)
                forecast = model.predict(future)
                return forecast['yhat'].tail(self.config.forecast_horizon).tolist()
            else:
                # Deep learning models
                return self._generate_dl_forecasts(model, data)
        except Exception as e:
            self.logger.error(f"Forecast generation failed: {e}")
            return [0.0] * self.config.forecast_horizon
    
    def _generate_dl_forecasts(self, model: nn.Module, data: pd.DataFrame) -> List[float]:
        """Generate forecasts using deep learning model."""
        model.eval()
        forecasts = []
        
        # Use last lookback_window points to predict next point
        last_sequence = data.tail(self.config.lookback_window)
        feature_cols = [col for col in last_sequence.columns if col != 'value']
        
        current_sequence = last_sequence[feature_cols].values
        
        with torch.no_grad():
            for _ in range(self.config.forecast_horizon):
                # Prepare input
                input_tensor = torch.FloatTensor(current_sequence).unsqueeze(0).to(self.device)
                
                # Predict
                prediction = model(input_tensor).squeeze().cpu().numpy()
                forecasts.append(float(prediction))
                
                # Update sequence for next prediction
                current_sequence = np.roll(current_sequence, -1, axis=0)
                current_sequence[-1] = prediction
        
        return forecasts
    
    def _detect_anomalies(self, model: Any, data: pd.DataFrame) -> List[int]:
        """Detect anomalies."""
        try:
            if isinstance(model, dict) and 'anomalies' in model:
                # Statistical model
                anomaly_indices = np.where(model['anomalies'])[0].tolist()
                return anomaly_indices
            elif isinstance(model, dict) and 'model' in model:
                # Isolation Forest
                anomaly_indices = np.where(model['anomalies'])[0].tolist()
                return anomaly_indices
            else:
                # LSTM Autoencoder
                return self._detect_dl_anomalies(model, data)
        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {e}")
            return []
    
    def _detect_dl_anomalies(self, model: nn.Module, data: pd.DataFrame) -> List[int]:
        """Detect anomalies using deep learning model."""
        model.eval()
        reconstruction_errors = []
        
        # Create sequences
        values = data['value'].values.reshape(-1, 1)
        seq_length = self.config.lookback_window
        
        for i in range(seq_length, len(values)):
            sequence = values[i-seq_length:i]
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                reconstructed = model(sequence_tensor)
                error = torch.mean((sequence_tensor - reconstructed) ** 2).item()
                reconstruction_errors.append(error)
        
        # Find anomalies based on reconstruction error
        threshold = np.mean(reconstruction_errors) + 2 * np.std(reconstruction_errors)
        anomaly_indices = [i + seq_length for i, error in enumerate(reconstruction_errors) if error > threshold]
        
        return anomaly_indices
    
    def _perform_decomposition(self, model: Any, data: pd.DataFrame) -> Dict[str, List[float]]:
        """Perform time series decomposition."""
        try:
            decomposition = seasonal_decompose(
                data['value'],
                model=self.config.decomposition_method.value,
                period=self.config.seasonality_period
            )
            
            return {
                'trend': decomposition.trend.dropna().tolist(),
                'seasonal': decomposition.seasonal.dropna().tolist(),
                'residual': decomposition.resid.dropna().tolist(),
                'observed': decomposition.observed.dropna().tolist()
            }
        except Exception as e:
            self.logger.error(f"Decomposition failed: {e}")
            return {}
    
    def _evaluate_forecasts(self, forecasts: List[float], data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate forecast performance."""
        if not forecasts:
            return {'mse': 0.0, 'mae': 0.0, 'r2': 0.0}
        
        # Use last part of data for evaluation
        actual_values = data['value'].tail(len(forecasts)).values
        
        mse = mean_squared_error(actual_values, forecasts)
        mae = mean_absolute_error(actual_values, forecasts)
        r2 = r2_score(actual_values, forecasts)
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'rmse': np.sqrt(mse)
        }
    
    def _evaluate_anomalies(self, anomalies: List[int], data: pd.DataFrame) -> Dict[str, float]:
        """Evaluate anomaly detection performance."""
        if not anomalies:
            return {'anomaly_count': 0, 'anomaly_rate': 0.0}
        
        anomaly_rate = len(anomalies) / len(data)
        
        return {
            'anomaly_count': len(anomalies),
            'anomaly_rate': anomaly_rate,
            'total_points': len(data)
        }
    
    def _save_ts_result(self, result: TSModelResult):
        """Save TS result to database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO ts_models 
                (model_id, task, model_name, performance_metrics,
                 forecasts, anomalies, decomposition, model_state, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.result_id,
                result.task.value,
                result.model.value,
                json.dumps(result.performance_metrics),
                json.dumps(result.forecasts) if result.forecasts else None,
                json.dumps(result.anomalies) if result.anomalies else None,
                json.dumps(result.decomposition) if result.decomposition else None,
                json.dumps(result.model_state),
                result.created_at.isoformat()
            ))
    
    def visualize_ts_results(self, result: TSModelResult, 
                           output_path: str = None) -> str:
        """Visualize time series results."""
        if output_path is None:
            output_path = f"ts_analysis_{result.result_id}.png"
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Performance metrics
        performance_metrics = result.performance_metrics
        metric_names = list(performance_metrics.keys())
        metric_values = list(performance_metrics.values())
        
        axes[0, 0].bar(metric_names, metric_values)
        axes[0, 0].set_title('Performance Metrics')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Forecasts
        if result.forecasts:
            axes[0, 1].plot(result.forecasts)
            axes[0, 1].set_title('Forecasts')
            axes[0, 1].set_xlabel('Time Steps')
            axes[0, 1].set_ylabel('Value')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Anomalies
        if result.anomalies:
            axes[1, 0].scatter(result.anomalies, [1] * len(result.anomalies), c='red', alpha=0.7)
            axes[1, 0].set_title('Anomalies Detected')
            axes[1, 0].set_xlabel('Time Index')
            axes[1, 0].set_ylabel('Anomaly')
        
        # Decomposition
        if result.decomposition:
            decomposition = result.decomposition
            if 'trend' in decomposition:
                axes[1, 1].plot(decomposition['trend'][:100], label='Trend')
                axes[1, 1].plot(decomposition['seasonal'][:100], label='Seasonal')
                axes[1, 1].plot(decomposition['residual'][:100], label='Residual')
                axes[1, 1].set_title('Time Series Decomposition')
                axes[1, 1].set_xlabel('Time Steps')
                axes[1, 1].set_ylabel('Value')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        console.print(f"[green]TS visualization saved: {output_path}[/green]")
        return output_path
    
    def get_ts_summary(self) -> Dict[str, Any]:
        """Get time series system summary."""
        if not self.ts_results:
            return {'total_experiments': 0}
        
        total_experiments = len(self.ts_results)
        
        # Calculate average metrics
        mse_scores = [result.performance_metrics.get('mse', 0) for result in self.ts_results.values()]
        r2_scores = [result.performance_metrics.get('r2', 0) for result in self.ts_results.values()]
        
        avg_mse = np.mean(mse_scores) if mse_scores else 0
        avg_r2 = np.mean(r2_scores) if r2_scores else 0
        
        # Best performing experiment
        best_result = min(self.ts_results.values(), 
                         key=lambda x: x.performance_metrics.get('mse', float('inf')))
        
        return {
            'total_experiments': total_experiments,
            'average_mse': avg_mse,
            'average_r2': avg_r2,
            'best_mse': best_result.performance_metrics.get('mse', 0),
            'best_experiment_id': best_result.result_id,
            'tasks_used': list(set(result.task.value for result in self.ts_results.values())),
            'models_used': list(set(result.model.value for result in self.ts_results.values()))
        }

def main():
    """Main function for TS CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Time Series Analysis System")
    parser.add_argument("--task", type=str,
                       choices=["forecasting", "anomaly_detection", "decomposition"],
                       default="forecasting", help="Time series task")
    parser.add_argument("--model", type=str,
                       choices=["arima", "exponential_smoothing", "prophet", "lstm", "xgboost"],
                       default="arima", help="Time series model")
    parser.add_argument("--forecast-horizon", type=int, default=30,
                       help="Forecast horizon")
    parser.add_argument("--lookback-window", type=int, default=100,
                       help="Lookback window")
    parser.add_argument("--seasonality-period", type=int, default=12,
                       help="Seasonality period")
    parser.add_argument("--decomposition-method", type=str,
                       choices=["additive", "multiplicative", "stl"],
                       default="additive", help="Decomposition method")
    parser.add_argument("--anomaly-method", type=str,
                       choices=["statistical", "isolation_forest", "lstm_autoencoder"],
                       default="statistical", help="Anomaly detection method")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Create TS configuration
    config = TSConfig(
        task=TSTask(args.task),
        model=TSModel(args.model),
        forecast_horizon=args.forecast_horizon,
        lookback_window=args.lookback_window,
        seasonality_period=args.seasonality_period,
        decomposition_method=TSDecomposition(args.decomposition_method),
        anomaly_method=TSAnomalyMethod(args.anomaly_method),
        device=args.device
    )
    
    # Create TS system
    ts_system = TSSystem(config)
    
    # Create sample time series data
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    values = np.sin(np.arange(len(dates)) * 2 * np.pi / 365) + np.random.normal(0, 0.1, len(dates))
    
    sample_data = TimeSeriesData(
        data_id="sample_ts",
        timestamps=dates.tolist(),
        values=values.tolist(),
        metadata={'frequency': 'daily', 'length': len(dates)}
    )
    
    # Run TS experiment
    result = ts_system.run_ts_experiment(sample_data)
    
    # Show results
    console.print(f"[green]TS experiment completed[/green]")
    console.print(f"[blue]Task: {result.task.value}[/blue]")
    console.print(f"[blue]Model: {result.model.value}[/blue]")
    console.print(f"[blue]Performance: {result.performance_metrics}[/blue]")
    
    # Create visualization
    ts_system.visualize_ts_results(result)
    
    # Show summary
    summary = ts_system.get_ts_summary()
    console.print(f"[blue]Summary: {summary}[/blue]")

if __name__ == "__main__":
    main()
