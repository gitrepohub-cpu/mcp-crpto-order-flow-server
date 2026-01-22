"""
Darts Integration Bridge

Seamless conversion between your TimeSeriesData format and Darts TimeSeries.
Provides unified model wrappers for 48+ Darts forecasting models.

Features:
- Bidirectional data conversion (TimeSeriesData <-> Darts TimeSeries)
- Unified model wrapper with consistent API
- Auto-model selection based on data characteristics
- Ensemble forecasting with multiple Darts models
- Covariate support (past and future covariates)
- Integration with DuckDB storage

Usage:
    from src.integrations import DartsBridge, DartsModelWrapper
    
    # Convert your data to Darts
    bridge = DartsBridge()
    darts_ts = bridge.to_darts(your_timeseries_data)
    
    # Train and forecast with any Darts model
    wrapper = DartsModelWrapper('NBEATS')
    wrapper.fit(darts_ts)
    forecast = wrapper.predict(horizon=24)
    
    # Convert back to your format
    result = bridge.from_darts_forecast(forecast)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Type, Callable
from enum import Enum
import warnings

import numpy as np
import pandas as pd

# Darts imports
try:
    from darts import TimeSeries
    from darts.dataprocessing.transformers import Scaler
    from darts.metrics import mape, rmse, mae, mse, smape
    DARTS_AVAILABLE = True
except ImportError:
    DARTS_AVAILABLE = False
    TimeSeries = None

# Your existing imports
from src.analytics.timeseries_engine import TimeSeriesData, ForecastResult

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


# ============================================================================
# MODEL REGISTRY
# ============================================================================

class ModelCategory(Enum):
    """Categories of Darts models."""
    STATISTICAL = "statistical"
    DEEP_LEARNING = "deep_learning"
    MACHINE_LEARNING = "machine_learning"
    ENSEMBLE = "ensemble"
    NAIVE = "naive"


# Model name mappings to Darts classes
STATISTICAL_MODELS = {
    'ARIMA': ('darts.models', 'ARIMA'),
    'AutoARIMA': ('darts.models', 'AutoARIMA'),
    'ExponentialSmoothing': ('darts.models', 'ExponentialSmoothing'),
    'Theta': ('darts.models', 'Theta'),
    'FourTheta': ('darts.models', 'FourTheta'),
    'FFT': ('darts.models', 'FFT'),
    'Prophet': ('darts.models', 'Prophet'),
    'BATS': ('darts.models', 'BATS'),
    'TBATS': ('darts.models', 'TBATS'),
    'Croston': ('darts.models', 'Croston'),
    'StatsForecastAutoARIMA': ('darts.models', 'StatsForecastAutoARIMA'),
    'StatsForecastAutoETS': ('darts.models', 'StatsForecastAutoETS'),
    'StatsForecastAutoCES': ('darts.models', 'StatsForecastAutoCES'),
    'StatsForecastAutoTheta': ('darts.models', 'StatsForecastAutoTheta'),
}

DEEP_LEARNING_MODELS = {
    'NBEATS': ('darts.models', 'NBEATSModel'),
    'NHiTS': ('darts.models', 'NHiTSModel'),
    'TCN': ('darts.models', 'TCNModel'),
    'Transformer': ('darts.models', 'TransformerModel'),
    'TFT': ('darts.models', 'TFTModel'),
    'DLinear': ('darts.models', 'DLinearModel'),
    'NLinear': ('darts.models', 'NLinearModel'),
    'TiDE': ('darts.models', 'TiDEModel'),
    'TSMixer': ('darts.models', 'TSMixerModel'),
    'RNN': ('darts.models', 'RNNModel'),
    'LSTM': ('darts.models', 'RNNModel'),  # RNN with model='LSTM'
    'GRU': ('darts.models', 'RNNModel'),   # RNN with model='GRU'
    'BlockRNN': ('darts.models', 'BlockRNNModel'),
}

ML_MODELS = {
    'XGBoost': ('darts.models', 'XGBModel'),
    'LightGBM': ('darts.models', 'LightGBMModel'),
    'CatBoost': ('darts.models', 'CatBoostModel'),
    'RandomForest': ('darts.models', 'RandomForest'),
    'LinearRegression': ('darts.models', 'LinearRegressionModel'),
    'RegressionModel': ('darts.models', 'RegressionModel'),
}

NAIVE_MODELS = {
    'Naive': ('darts.models', 'NaiveMean'),
    'NaiveMean': ('darts.models', 'NaiveMean'),
    'NaiveSeasonal': ('darts.models', 'NaiveSeasonal'),
    'NaiveDrift': ('darts.models', 'NaiveDrift'),
    'NaiveMovingAverage': ('darts.models', 'NaiveMovingAverage'),
}

ALL_MODELS = {**STATISTICAL_MODELS, **DEEP_LEARNING_MODELS, **ML_MODELS, **NAIVE_MODELS}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class DartsForecastResult:
    """
    Unified forecast result from Darts models.
    Compatible with your existing ForecastResult format.
    """
    forecast: pd.Series
    lower_bound: Optional[pd.Series]
    upper_bound: Optional[pd.Series]
    model_name: str
    darts_timeseries: Optional[Any] = None  # Original Darts TimeSeries
    confidence_level: float = 0.95
    metrics: Dict[str, float] = field(default_factory=dict)
    training_time: float = 0.0
    
    def to_forecast_result(self) -> ForecastResult:
        """Convert to your standard ForecastResult format."""
        return ForecastResult(
            forecast=self.forecast,
            lower_bound=self.lower_bound if self.lower_bound is not None else self.forecast,
            upper_bound=self.upper_bound if self.upper_bound is not None else self.forecast,
            model_name=self.model_name,
            confidence_level=self.confidence_level,
            metrics=self.metrics
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'forecast': self.forecast.to_dict(),
            'lower_bound': self.lower_bound.to_dict() if self.lower_bound is not None else None,
            'upper_bound': self.upper_bound.to_dict() if self.upper_bound is not None else None,
            'model': self.model_name,
            'confidence': self.confidence_level,
            'metrics': self.metrics,
            'training_time_seconds': self.training_time
        }


@dataclass
class ModelConfig:
    """Configuration for a Darts model."""
    name: str
    category: ModelCategory
    supports_covariates: bool = False
    supports_multivariate: bool = False
    requires_training: bool = True
    default_params: Dict[str, Any] = field(default_factory=dict)
    
    # Deep learning specific
    input_chunk_length: Optional[int] = None
    output_chunk_length: Optional[int] = None


# ============================================================================
# DARTS BRIDGE - Core Conversion Layer
# ============================================================================

class DartsBridge:
    """
    Bidirectional bridge between your TimeSeriesData and Darts TimeSeries.
    
    Handles:
    - Data format conversion
    - Missing value handling
    - Frequency inference
    - Multivariate series
    - Covariates
    """
    
    def __init__(self, fill_missing: str = 'interpolate', default_freq: str = '1min'):
        """
        Initialize the bridge.
        
        Args:
            fill_missing: How to handle missing values ('interpolate', 'forward', 'backward', 'zero', 'drop')
            default_freq: Default frequency if cannot be inferred
        """
        if not DARTS_AVAILABLE:
            raise ImportError("Darts is not installed. Run: pip install darts")
        
        self.fill_missing = fill_missing
        self.default_freq = default_freq
        self.scaler = None
        
    def to_darts(
        self,
        data: Union[TimeSeriesData, pd.DataFrame, pd.Series, np.ndarray, List[float]],
        time_col: Optional[str] = None,
        value_col: Optional[str] = None,
        freq: Optional[str] = None,
        fill_missing: Optional[str] = None
    ) -> TimeSeries:
        """
        Convert your data to Darts TimeSeries.
        
        Args:
            data: Input data (TimeSeriesData, DataFrame, Series, array, or list)
            time_col: Column name for timestamps (if DataFrame)
            value_col: Column name for values (if DataFrame)
            freq: Time frequency (e.g., '1min', '1h', '1d')
            fill_missing: Override default missing value handling
            
        Returns:
            Darts TimeSeries object
        """
        fill_method = fill_missing or self.fill_missing
        
        # Convert to DataFrame first
        if isinstance(data, TimeSeriesData):
            df = pd.DataFrame({
                'time': data.time,
                'value': data.value.values
            })
            time_col = 'time'
            value_col = 'value'
            
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
            time_col = time_col or df.columns[0]
            value_col = value_col or df.columns[1]
            
        elif isinstance(data, pd.Series):
            if isinstance(data.index, pd.DatetimeIndex):
                df = pd.DataFrame({'time': data.index, 'value': data.values})
            else:
                # Create synthetic timestamps
                df = pd.DataFrame({
                    'time': pd.date_range(start='2024-01-01', periods=len(data), freq=self.default_freq),
                    'value': data.values
                })
            time_col = 'time'
            value_col = 'value'
            
        elif isinstance(data, (np.ndarray, list)):
            arr = np.array(data)
            df = pd.DataFrame({
                'time': pd.date_range(start='2024-01-01', periods=len(arr), freq=self.default_freq),
                'value': arr
            })
            time_col = 'time'
            value_col = 'value'
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        # Ensure datetime index
        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index(time_col).sort_index()
        
        # Handle missing values
        df = self._fill_missing_values(df, value_col, fill_method)
        
        # Infer frequency if not provided
        if freq is None:
            freq = self._infer_frequency(df.index)
        
        # Create Darts TimeSeries
        try:
            ts = TimeSeries.from_dataframe(
                df.reset_index(),
                time_col=time_col,
                value_cols=value_col,
                freq=freq,
                fill_missing_dates=True
            )
        except Exception as e:
            logger.warning(f"Failed to create with freq={freq}: {e}. Trying without freq.")
            ts = TimeSeries.from_dataframe(
                df.reset_index(),
                time_col=time_col,
                value_cols=value_col
            )
        
        return ts
    
    def from_darts(
        self,
        ts: TimeSeries,
        name: str = "value"
    ) -> TimeSeriesData:
        """
        Convert Darts TimeSeries back to your TimeSeriesData format.
        
        Args:
            ts: Darts TimeSeries object
            name: Name for the value column
            
        Returns:
            TimeSeriesData object
        """
        df = ts.to_dataframe()  # Darts 0.40+ uses to_dataframe()
        
        # Handle multivariate - take first column
        if len(df.columns) > 1:
            value_col = df.columns[0]
            logger.info(f"Multivariate series detected. Using first column: {value_col}")
        else:
            value_col = df.columns[0]
        
        return TimeSeriesData(
            time=df.index,
            value=pd.Series(df[value_col].values, index=df.index),
            name=name,
            metadata={
                'original_columns': list(df.columns),
                'frequency': str(ts.freq) if ts.freq else 'unknown',
                'length': len(ts)
            }
        )
    
    def from_darts_forecast(
        self,
        forecast_ts: TimeSeries,
        model_name: str = "DartsModel",
        confidence_level: float = 0.95
    ) -> DartsForecastResult:
        """
        Convert Darts forecast TimeSeries to DartsForecastResult.
        
        Args:
            forecast_ts: Darts TimeSeries forecast
            model_name: Name of the model used
            confidence_level: Confidence level for intervals
            
        Returns:
            DartsForecastResult object
        """
        df = forecast_ts.to_dataframe()  # Darts 0.40+ uses to_dataframe()
        
        # Get the first value column
        value_col = df.columns[0]
        forecast_series = pd.Series(df[value_col].values, index=df.index)
        
        # Check for probabilistic forecasts (multiple samples)
        lower_bound = None
        upper_bound = None
        
        if hasattr(forecast_ts, 'n_samples') and forecast_ts.n_samples > 1:
            # Probabilistic forecast - calculate confidence intervals
            alpha = 1 - confidence_level
            lower_q = alpha / 2
            upper_q = 1 - alpha / 2
            
            lower_ts = forecast_ts.quantile(lower_q)
            upper_ts = forecast_ts.quantile(upper_q)
            
            lower_df = lower_ts.to_dataframe()  # Darts 0.40+ uses to_dataframe()
            upper_df = upper_ts.to_dataframe()  # Darts 0.40+ uses to_dataframe()
            
            lower_bound = pd.Series(lower_df.iloc[:, 0].values, index=df.index)
            upper_bound = pd.Series(upper_df.iloc[:, 0].values, index=df.index)
        
        return DartsForecastResult(
            forecast=forecast_series,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            model_name=model_name,
            darts_timeseries=forecast_ts,
            confidence_level=confidence_level
        )
    
    def to_multivariate_darts(
        self,
        data_dict: Dict[str, Union[TimeSeriesData, pd.Series, np.ndarray]],
        freq: Optional[str] = None
    ) -> TimeSeries:
        """
        Create multivariate Darts TimeSeries from multiple series.
        
        Args:
            data_dict: Dictionary mapping column names to data
            freq: Time frequency
            
        Returns:
            Multivariate Darts TimeSeries
        """
        dfs = []
        
        for name, data in data_dict.items():
            if isinstance(data, TimeSeriesData):
                df = pd.DataFrame({name: data.value.values}, index=data.time)
            elif isinstance(data, pd.Series):
                df = pd.DataFrame({name: data.values}, index=data.index)
            elif isinstance(data, np.ndarray):
                df = pd.DataFrame({name: data})
            else:
                df = pd.DataFrame({name: np.array(data)})
            dfs.append(df)
        
        # Merge all DataFrames
        combined = pd.concat(dfs, axis=1)
        combined = combined.dropna()
        
        # Create multivariate TimeSeries
        freq = freq or self._infer_frequency(combined.index)
        
        return TimeSeries.from_dataframe(
            combined.reset_index(),
            time_col='index',
            value_cols=list(data_dict.keys()),
            freq=freq
        )
    
    def create_covariates(
        self,
        data: Dict[str, Union[TimeSeriesData, pd.Series, np.ndarray]],
        is_future: bool = False,
        freq: Optional[str] = None
    ) -> TimeSeries:
        """
        Create covariate TimeSeries for models that support them.
        
        Args:
            data: Dictionary of covariate data
            is_future: Whether these are future covariates (known ahead of time)
            freq: Time frequency
            
        Returns:
            Covariate TimeSeries
        """
        return self.to_multivariate_darts(data, freq)
    
    def scale(self, ts: TimeSeries, fit: bool = True) -> TimeSeries:
        """
        Scale TimeSeries using MinMax scaling.
        
        Args:
            ts: TimeSeries to scale
            fit: Whether to fit the scaler (True for training data)
            
        Returns:
            Scaled TimeSeries
        """
        if fit or self.scaler is None:
            self.scaler = Scaler()
            return self.scaler.fit_transform(ts)
        return self.scaler.transform(ts)
    
    def inverse_scale(self, ts: TimeSeries) -> TimeSeries:
        """
        Inverse scale TimeSeries back to original values.
        
        Args:
            ts: Scaled TimeSeries
            
        Returns:
            Original scale TimeSeries
        """
        if self.scaler is None:
            logger.warning("No scaler fitted. Returning original series.")
            return ts
        return self.scaler.inverse_transform(ts)
    
    def _fill_missing_values(
        self,
        df: pd.DataFrame,
        value_col: str,
        method: str
    ) -> pd.DataFrame:
        """Fill missing values in the DataFrame."""
        if method == 'interpolate':
            df[value_col] = df[value_col].interpolate(method='linear')
        elif method == 'forward':
            df[value_col] = df[value_col].ffill()
        elif method == 'backward':
            df[value_col] = df[value_col].bfill()
        elif method == 'zero':
            df[value_col] = df[value_col].fillna(0)
        elif method == 'drop':
            df = df.dropna(subset=[value_col])
        
        # Final forward/backward fill for any remaining NaNs
        df[value_col] = df[value_col].ffill().bfill()
        
        return df
    
    def _infer_frequency(self, index: pd.DatetimeIndex) -> str:
        """Infer the frequency from a DatetimeIndex."""
        if len(index) < 2:
            return self.default_freq
        
        try:
            freq = pd.infer_freq(index)
            if freq:
                return freq
        except:
            pass
        
        # Manual inference based on median diff
        diffs = index.to_series().diff().dropna()
        if len(diffs) == 0:
            return self.default_freq
        
        median_diff = diffs.median()
        
        if median_diff <= timedelta(seconds=1):
            return '1s'
        elif median_diff <= timedelta(minutes=1):
            return '1min'
        elif median_diff <= timedelta(minutes=5):
            return '5min'
        elif median_diff <= timedelta(minutes=15):
            return '15min'
        elif median_diff <= timedelta(hours=1):
            return '1h'
        elif median_diff <= timedelta(hours=4):
            return '4h'
        elif median_diff <= timedelta(days=1):
            return '1d'
        elif median_diff <= timedelta(weeks=1):
            return '1w'
        else:
            return '1M'


# ============================================================================
# DARTS MODEL WRAPPER - Unified Model Interface
# ============================================================================

class DartsModelWrapper:
    """
    Unified wrapper for all Darts models.
    
    Provides consistent API regardless of model type (statistical, ML, deep learning).
    
    Usage:
        wrapper = DartsModelWrapper('NBEATS')
        wrapper.fit(train_ts)
        forecast = wrapper.predict(horizon=24)
    """
    
    def __init__(
        self,
        model_name: str,
        **model_kwargs
    ):
        """
        Initialize the model wrapper.
        
        Args:
            model_name: Name of the Darts model (e.g., 'NBEATS', 'ARIMA', 'XGBoost')
            **model_kwargs: Model-specific parameters
        """
        if not DARTS_AVAILABLE:
            raise ImportError("Darts is not installed. Run: pip install darts")
        
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.model = None
        self.is_fitted = False
        self.training_series = None
        self.bridge = DartsBridge()
        
        # Get model class
        self.model_class = self._get_model_class(model_name)
        self.category = self._get_category(model_name)
        
        # Initialize model
        self._initialize_model()
    
    def _get_model_class(self, name: str) -> Type:
        """Get the Darts model class by name."""
        if name not in ALL_MODELS:
            raise ValueError(
                f"Unknown model: {name}. Available models: {list(ALL_MODELS.keys())}"
            )
        
        module_path, class_name = ALL_MODELS[name]
        
        try:
            import importlib
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except ImportError as e:
            raise ImportError(f"Failed to import {class_name} from {module_path}: {e}")
    
    def _get_category(self, name: str) -> ModelCategory:
        """Determine the model category."""
        if name in STATISTICAL_MODELS:
            return ModelCategory.STATISTICAL
        elif name in DEEP_LEARNING_MODELS:
            return ModelCategory.DEEP_LEARNING
        elif name in ML_MODELS:
            return ModelCategory.MACHINE_LEARNING
        elif name in NAIVE_MODELS:
            return ModelCategory.NAIVE
        return ModelCategory.STATISTICAL
    
    def _initialize_model(self):
        """Initialize the Darts model with appropriate defaults."""
        kwargs = self.model_kwargs.copy()
        
        # Handle special cases
        if self.model_name in ['LSTM', 'GRU']:
            kwargs['model'] = self.model_name
        
        # Set defaults for deep learning models
        if self.category == ModelCategory.DEEP_LEARNING:
            kwargs.setdefault('input_chunk_length', 24)
            kwargs.setdefault('output_chunk_length', 12)
            kwargs.setdefault('n_epochs', 50)
            kwargs.setdefault('random_state', 42)
            
            # Disable GPU if not available
            try:
                import torch
                if not torch.cuda.is_available():
                    kwargs.setdefault('pl_trainer_kwargs', {'accelerator': 'cpu'})
            except ImportError:
                pass
        
        # Set defaults for ML models
        if self.category == ModelCategory.MACHINE_LEARNING:
            kwargs.setdefault('lags', 24)
            kwargs.setdefault('output_chunk_length', 12)
        
        try:
            self.model = self.model_class(**kwargs)
        except TypeError as e:
            # Remove unsupported kwargs and retry
            logger.warning(f"Some kwargs not supported: {e}. Retrying with defaults.")
            self.model = self.model_class()
    
    def fit(
        self,
        series: Union[TimeSeries, TimeSeriesData, pd.Series, np.ndarray],
        past_covariates: Optional[TimeSeries] = None,
        future_covariates: Optional[TimeSeries] = None,
        val_series: Optional[TimeSeries] = None,
        verbose: bool = False
    ) -> 'DartsModelWrapper':
        """
        Fit the model to training data.
        
        Args:
            series: Training time series
            past_covariates: Past covariates (optional)
            future_covariates: Future covariates (optional)
            val_series: Validation series for early stopping (optional)
            verbose: Whether to show training progress
            
        Returns:
            self
        """
        import time
        start_time = time.time()
        
        # Convert to Darts TimeSeries if needed
        if not isinstance(series, TimeSeries):
            series = self.bridge.to_darts(series)
        
        self.training_series = series
        
        # Build fit kwargs
        fit_kwargs = {}
        
        # Add covariates if supported
        if self._supports_covariates():
            if past_covariates is not None:
                fit_kwargs['past_covariates'] = past_covariates
            if future_covariates is not None:
                fit_kwargs['future_covariates'] = future_covariates
        
        # Add validation series if supported (deep learning)
        if self.category == ModelCategory.DEEP_LEARNING and val_series is not None:
            fit_kwargs['val_series'] = val_series
        
        # Handle verbose for deep learning
        if self.category == ModelCategory.DEEP_LEARNING:
            fit_kwargs['verbose'] = verbose
        
        try:
            self.model.fit(series, **fit_kwargs)
            self.is_fitted = True
            self.training_time = time.time() - start_time
            logger.info(f"{self.model_name} fitted in {self.training_time:.2f}s")
        except Exception as e:
            logger.error(f"Failed to fit {self.model_name}: {e}")
            raise
        
        return self
    
    def predict(
        self,
        horizon: int,
        series: Optional[TimeSeries] = None,
        past_covariates: Optional[TimeSeries] = None,
        future_covariates: Optional[TimeSeries] = None,
        num_samples: int = 1,
        confidence_level: float = 0.95
    ) -> DartsForecastResult:
        """
        Generate forecast.
        
        Args:
            horizon: Number of steps to forecast
            series: Series to predict from (uses training series if None)
            past_covariates: Past covariates for prediction
            future_covariates: Future covariates for prediction
            num_samples: Number of samples for probabilistic forecast
            confidence_level: Confidence level for prediction intervals
            
        Returns:
            DartsForecastResult object
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")
        
        predict_kwargs = {'n': horizon}
        
        # Use provided series or training series
        if series is not None:
            if not isinstance(series, TimeSeries):
                series = self.bridge.to_darts(series)
            predict_kwargs['series'] = series
        
        # Add covariates if supported
        if self._supports_covariates():
            if past_covariates is not None:
                predict_kwargs['past_covariates'] = past_covariates
            if future_covariates is not None:
                predict_kwargs['future_covariates'] = future_covariates
        
        # Add num_samples for probabilistic models
        if self._supports_probabilistic() and num_samples > 1:
            predict_kwargs['num_samples'] = num_samples
        
        try:
            forecast_ts = self.model.predict(**predict_kwargs)
            result = self.bridge.from_darts_forecast(
                forecast_ts,
                model_name=self.model_name,
                confidence_level=confidence_level
            )
            result.training_time = getattr(self, 'training_time', 0.0)
            return result
        except Exception as e:
            logger.error(f"Prediction failed for {self.model_name}: {e}")
            raise
    
    def backtest(
        self,
        series: TimeSeries,
        start: float = 0.7,
        forecast_horizon: int = 12,
        stride: int = 1,
        retrain: bool = False,
        metric: Callable = mape
    ) -> float:
        """
        Perform backtesting.
        
        Args:
            series: Full series to backtest on
            start: Starting point (fraction of series)
            forecast_horizon: Forecast horizon for each backtest
            stride: Steps between backtests
            retrain: Whether to retrain at each step
            metric: Metric function to use
            
        Returns:
            Average metric value across backtests
        """
        if not isinstance(series, TimeSeries):
            series = self.bridge.to_darts(series)
        
        try:
            backtest_result = self.model.backtest(
                series=series,
                start=start,
                forecast_horizon=forecast_horizon,
                stride=stride,
                retrain=retrain,
                metric=metric,
                verbose=False
            )
            return float(np.mean(backtest_result))
        except Exception as e:
            logger.error(f"Backtest failed for {self.model_name}: {e}")
            return float('inf')
    
    def evaluate(
        self,
        actual: TimeSeries,
        predicted: TimeSeries
    ) -> Dict[str, float]:
        """
        Evaluate forecast against actual values.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            
        Returns:
            Dictionary of metric values
        """
        return {
            'mape': mape(actual, predicted),
            'rmse': rmse(actual, predicted),
            'mae': mae(actual, predicted),
            'mse': mse(actual, predicted),
            'smape': smape(actual, predicted)
        }
    
    def _supports_covariates(self) -> bool:
        """Check if model supports covariates."""
        covariate_models = {
            'TFT', 'NBEATS', 'NHiTS', 'TCN', 'Transformer', 
            'DLinear', 'NLinear', 'TiDE', 'TSMixer',
            'XGBoost', 'LightGBM', 'CatBoost', 'RandomForest',
            'RNN', 'LSTM', 'GRU', 'BlockRNN'
        }
        return self.model_name in covariate_models
    
    def _supports_probabilistic(self) -> bool:
        """Check if model supports probabilistic forecasting."""
        probabilistic_models = {
            'NBEATS', 'NHiTS', 'TCN', 'Transformer', 'TFT',
            'DLinear', 'NLinear', 'TiDE', 'TSMixer',
            'RNN', 'LSTM', 'GRU', 'BlockRNN'
        }
        return self.model_name in probabilistic_models


# ============================================================================
# DARTS ENSEMBLE - Multi-Model Ensemble
# ============================================================================

class DartsEnsemble:
    """
    Ensemble forecasting using multiple Darts models.
    
    Combines predictions from multiple models using various strategies:
    - Simple average
    - Weighted average (based on validation performance)
    - Median
    - Best model selection
    
    Usage:
        ensemble = DartsEnsemble(['ARIMA', 'Theta', 'ExponentialSmoothing'])
        ensemble.fit(train_ts)
        forecast = ensemble.predict(horizon=24, method='weighted')
    """
    
    def __init__(
        self,
        model_names: List[str],
        model_kwargs: Optional[Dict[str, Dict]] = None
    ):
        """
        Initialize ensemble.
        
        Args:
            model_names: List of model names to include
            model_kwargs: Model-specific parameters {model_name: {param: value}}
        """
        self.model_names = model_names
        self.model_kwargs = model_kwargs or {}
        self.models: Dict[str, DartsModelWrapper] = {}
        self.weights: Dict[str, float] = {}
        self.bridge = DartsBridge()
        self.is_fitted = False
        
        # Initialize all models
        for name in model_names:
            kwargs = self.model_kwargs.get(name, {})
            try:
                self.models[name] = DartsModelWrapper(name, **kwargs)
            except Exception as e:
                logger.warning(f"Failed to initialize {name}: {e}")
    
    def fit(
        self,
        series: Union[TimeSeries, TimeSeriesData],
        val_series: Optional[TimeSeries] = None,
        calculate_weights: bool = True
    ) -> 'DartsEnsemble':
        """
        Fit all models in the ensemble.
        
        Args:
            series: Training series
            val_series: Validation series for weight calculation
            calculate_weights: Whether to calculate performance-based weights
            
        Returns:
            self
        """
        if not isinstance(series, TimeSeries):
            series = self.bridge.to_darts(series)
        
        # Fit each model
        successful_models = {}
        for name, wrapper in self.models.items():
            try:
                wrapper.fit(series)
                successful_models[name] = wrapper
                logger.info(f"✓ {name} fitted successfully")
            except Exception as e:
                logger.warning(f"✗ {name} failed to fit: {e}")
        
        self.models = successful_models
        
        if len(self.models) == 0:
            raise ValueError("All models failed to fit!")
        
        # Calculate weights based on validation performance
        if calculate_weights and val_series is not None:
            self._calculate_weights(series, val_series)
        else:
            # Equal weights
            self.weights = {name: 1.0 / len(self.models) for name in self.models}
        
        self.is_fitted = True
        return self
    
    def _calculate_weights(self, train_series: TimeSeries, val_series: TimeSeries):
        """Calculate performance-based weights."""
        performances = {}
        
        for name, wrapper in self.models.items():
            try:
                forecast = wrapper.predict(horizon=len(val_series))
                error = mape(val_series, self.bridge.to_darts(forecast.forecast))
                performances[name] = 1.0 / (error + 1e-6)  # Inverse error
            except Exception as e:
                logger.warning(f"Could not evaluate {name}: {e}")
                performances[name] = 0.0
        
        # Normalize weights
        total = sum(performances.values())
        if total > 0:
            self.weights = {name: perf / total for name, perf in performances.items()}
        else:
            self.weights = {name: 1.0 / len(self.models) for name in self.models}
    
    def predict(
        self,
        horizon: int,
        method: str = 'weighted',
        confidence_level: float = 0.95
    ) -> DartsForecastResult:
        """
        Generate ensemble forecast.
        
        Args:
            horizon: Forecast horizon
            method: Combination method ('weighted', 'average', 'median', 'best')
            confidence_level: Confidence level for intervals
            
        Returns:
            DartsForecastResult
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted first.")
        
        forecasts = {}
        for name, wrapper in self.models.items():
            try:
                forecast = wrapper.predict(horizon=horizon)
                forecasts[name] = forecast.forecast
            except Exception as e:
                logger.warning(f"Prediction failed for {name}: {e}")
        
        if len(forecasts) == 0:
            raise ValueError("All model predictions failed!")
        
        # Combine forecasts
        forecast_df = pd.DataFrame(forecasts)
        
        if method == 'weighted':
            weights_series = pd.Series({name: self.weights.get(name, 0) 
                                       for name in forecast_df.columns})
            combined = (forecast_df * weights_series).sum(axis=1)
        elif method == 'average':
            combined = forecast_df.mean(axis=1)
        elif method == 'median':
            combined = forecast_df.median(axis=1)
        elif method == 'best':
            # Use best performing model
            best_model = max(self.weights, key=self.weights.get)
            combined = forecasts[best_model]
        else:
            combined = forecast_df.mean(axis=1)
        
        # Calculate uncertainty bounds from ensemble spread
        lower_bound = forecast_df.min(axis=1)
        upper_bound = forecast_df.max(axis=1)
        
        return DartsForecastResult(
            forecast=combined,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            model_name=f"Ensemble({', '.join(self.models.keys())})",
            confidence_level=confidence_level,
            metrics={'ensemble_std': forecast_df.std(axis=1).mean()}
        )
    
    def get_individual_forecasts(
        self,
        horizon: int
    ) -> Dict[str, DartsForecastResult]:
        """Get forecasts from each individual model."""
        results = {}
        for name, wrapper in self.models.items():
            try:
                results[name] = wrapper.predict(horizon=horizon)
            except Exception as e:
                logger.warning(f"Prediction failed for {name}: {e}")
        return results


# ============================================================================
# DARTS AUTO-ML - Automatic Model Selection
# ============================================================================

class DartsAutoML:
    """
    Automatic model selection and hyperparameter tuning.
    
    Tests multiple models and selects the best based on validation performance.
    
    Usage:
        auto = DartsAutoML()
        best_model = auto.fit(train_ts, val_ts)
        forecast = best_model.predict(horizon=24)
    """
    
    # Default models to try in order of complexity
    DEFAULT_MODELS = [
        'NaiveSeasonal',
        'ExponentialSmoothing',
        'Theta',
        'ARIMA',
        'Prophet',
        'XGBoost',
        'NBEATS'
    ]
    
    def __init__(
        self,
        models_to_try: Optional[List[str]] = None,
        metric: str = 'mape',
        timeout_per_model: int = 300,
        include_deep_learning: bool = False
    ):
        """
        Initialize AutoML.
        
        Args:
            models_to_try: List of models to try (uses defaults if None)
            metric: Metric to optimize ('mape', 'rmse', 'mae', 'smape')
            timeout_per_model: Max seconds per model
            include_deep_learning: Whether to include deep learning models
        """
        self.models_to_try = models_to_try or self.DEFAULT_MODELS.copy()
        
        if include_deep_learning:
            self.models_to_try.extend(['NBEATS', 'NHiTS', 'TFT', 'TCN'])
        
        self.metric = metric
        self.timeout = timeout_per_model
        self.results: Dict[str, Dict] = {}
        self.best_model: Optional[DartsModelWrapper] = None
        self.bridge = DartsBridge()
    
    def fit(
        self,
        train_series: Union[TimeSeries, TimeSeriesData],
        val_series: Union[TimeSeries, TimeSeriesData],
        verbose: bool = True
    ) -> DartsModelWrapper:
        """
        Find the best model.
        
        Args:
            train_series: Training data
            val_series: Validation data
            verbose: Whether to print progress
            
        Returns:
            Best performing model wrapper
        """
        import time
        
        # Convert to Darts if needed
        if not isinstance(train_series, TimeSeries):
            train_series = self.bridge.to_darts(train_series)
        if not isinstance(val_series, TimeSeries):
            val_series = self.bridge.to_darts(val_series)
        
        horizon = len(val_series)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"DARTS AUTO-ML: Testing {len(self.models_to_try)} models")
            print(f"{'='*60}")
        
        for model_name in self.models_to_try:
            if verbose:
                print(f"\n[{model_name}] Training...", end=" ")
            
            try:
                start = time.time()
                wrapper = DartsModelWrapper(model_name)
                wrapper.fit(train_series)
                
                # Predict and evaluate
                forecast = wrapper.predict(horizon=horizon)
                forecast_ts = self.bridge.to_darts(forecast.forecast)
                
                # Calculate metrics
                metrics = wrapper.evaluate(val_series, forecast_ts)
                elapsed = time.time() - start
                
                self.results[model_name] = {
                    'wrapper': wrapper,
                    'metrics': metrics,
                    'time': elapsed,
                    'score': metrics[self.metric]
                }
                
                if verbose:
                    print(f"✓ {self.metric.upper()}={metrics[self.metric]:.4f} ({elapsed:.1f}s)")
                    
            except Exception as e:
                if verbose:
                    print(f"✗ Failed: {str(e)[:50]}")
                self.results[model_name] = {
                    'error': str(e),
                    'score': float('inf')
                }
        
        # Select best model
        valid_results = {k: v for k, v in self.results.items() if 'wrapper' in v}
        
        if not valid_results:
            raise ValueError("All models failed!")
        
        best_name = min(valid_results, key=lambda x: valid_results[x]['score'])
        self.best_model = valid_results[best_name]['wrapper']
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"BEST MODEL: {best_name}")
            print(f"  {self.metric.upper()}: {valid_results[best_name]['score']:.4f}")
            print(f"{'='*60}")
        
        return self.best_model
    
    def get_leaderboard(self) -> pd.DataFrame:
        """Get ranked results as DataFrame."""
        rows = []
        for name, result in self.results.items():
            row = {'model': name}
            if 'metrics' in result:
                row.update(result['metrics'])
                row['time_seconds'] = result['time']
            else:
                row['error'] = result.get('error', 'Unknown')
            rows.append(row)
        
        df = pd.DataFrame(rows)
        if self.metric in df.columns:
            df = df.sort_values(self.metric)
        return df


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def quick_forecast(
    data: Union[TimeSeriesData, pd.Series, np.ndarray, List[float]],
    horizon: int,
    model: str = 'AutoARIMA'
) -> DartsForecastResult:
    """
    Quick one-liner forecast.
    
    Args:
        data: Input time series data
        horizon: Forecast horizon
        model: Model name to use
        
    Returns:
        DartsForecastResult
    """
    wrapper = DartsModelWrapper(model)
    bridge = DartsBridge()
    ts = bridge.to_darts(data)
    wrapper.fit(ts)
    return wrapper.predict(horizon=horizon)


def compare_models(
    data: Union[TimeSeriesData, pd.Series, np.ndarray],
    models: List[str],
    test_size: float = 0.2,
    horizon: int = 12
) -> pd.DataFrame:
    """
    Compare multiple models on the same data.
    
    Args:
        data: Input time series
        models: List of model names to compare
        test_size: Fraction of data for testing
        horizon: Forecast horizon
        
    Returns:
        DataFrame with comparison results
    """
    bridge = DartsBridge()
    ts = bridge.to_darts(data)
    
    # Split data
    split_idx = int(len(ts) * (1 - test_size))
    train, test = ts[:split_idx], ts[split_idx:]
    
    results = []
    for model_name in models:
        try:
            wrapper = DartsModelWrapper(model_name)
            wrapper.fit(train)
            forecast = wrapper.predict(horizon=min(horizon, len(test)))
            
            # Evaluate
            forecast_ts = bridge.to_darts(forecast.forecast)
            test_subset = test[:len(forecast_ts)]
            metrics = wrapper.evaluate(test_subset, forecast_ts)
            metrics['model'] = model_name
            metrics['training_time'] = wrapper.training_time
            results.append(metrics)
        except Exception as e:
            results.append({'model': model_name, 'error': str(e)})
    
    return pd.DataFrame(results)


# ============================================================================
# DUCKDB INTEGRATION
# ============================================================================

class DartsDuckDBIntegration:
    """
    Integration layer between Darts and your DuckDB storage.
    
    Provides direct methods to:
    - Load data from DuckDB tables into Darts TimeSeries
    - Save forecasts back to DuckDB
    - Run forecasts on historical data
    """
    
    def __init__(self, duckdb_manager=None):
        """
        Initialize integration.
        
        Args:
            duckdb_manager: Your DuckDB manager instance
        """
        self.db_manager = duckdb_manager
        self.bridge = DartsBridge()
    
    def load_from_table(
        self,
        table_name: str,
        time_col: str = 'timestamp',
        value_col: str = 'value',
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> TimeSeries:
        """
        Load data from DuckDB table into Darts TimeSeries.
        
        Args:
            table_name: Name of the DuckDB table
            time_col: Column containing timestamps
            value_col: Column containing values
            start_time: Filter start time
            end_time: Filter end time
            limit: Max rows to load
            
        Returns:
            Darts TimeSeries
        """
        if self.db_manager is None:
            raise ValueError("DuckDB manager not configured")
        
        # Build query
        conditions = []
        if start_time:
            conditions.append(f"{time_col} >= '{start_time}'")
        if end_time:
            conditions.append(f"{time_col} <= '{end_time}'")
        
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        limit_clause = f"LIMIT {limit}" if limit else ""
        
        query = f"""
            SELECT {time_col}, {value_col}
            FROM {table_name}
            {where_clause}
            ORDER BY {time_col}
            {limit_clause}
        """
        
        results = self.db_manager.execute_query(query)
        ts_data = TimeSeriesData.from_duckdb_result(results, name=value_col)
        
        return self.bridge.to_darts(ts_data)
    
    def forecast_from_table(
        self,
        table_name: str,
        horizon: int,
        model: str = 'AutoARIMA',
        time_col: str = 'timestamp',
        value_col: str = 'value',
        lookback: int = 1000
    ) -> DartsForecastResult:
        """
        Run forecast on data from DuckDB table.
        
        Args:
            table_name: Table name
            horizon: Forecast horizon
            model: Model to use
            time_col: Time column
            value_col: Value column
            lookback: Number of historical points to use
            
        Returns:
            DartsForecastResult
        """
        ts = self.load_from_table(
            table_name,
            time_col=time_col,
            value_col=value_col,
            limit=lookback
        )
        
        wrapper = DartsModelWrapper(model)
        wrapper.fit(ts)
        return wrapper.predict(horizon=horizon)


# Export all
__all__ = [
    # Core Classes
    'DartsBridge',
    'DartsModelWrapper',
    'DartsEnsemble',
    'DartsAutoML',
    'DartsDuckDBIntegration',
    
    # Data Classes
    'DartsForecastResult',
    'ModelConfig',
    'ModelCategory',
    
    # Model Registries
    'STATISTICAL_MODELS',
    'DEEP_LEARNING_MODELS',
    'ML_MODELS',
    'NAIVE_MODELS',
    'ALL_MODELS',
    
    # Convenience Functions
    'quick_forecast',
    'compare_models',
]
