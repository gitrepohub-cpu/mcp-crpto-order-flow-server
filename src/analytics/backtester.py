"""
Backtesting Framework for Time Series Forecasting.

This module provides comprehensive backtesting capabilities similar to Kats:
- Walk-forward validation
- Expanding window validation
- Sliding window validation
- Multiple evaluation metrics (MAPE, RMSE, MAE, MASE, etc.)
- Cross-validation
- Performance visualization support

Essential for validating forecasting models before production deployment.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging
import warnings

logger = logging.getLogger(__name__)


class BacktestMethod(Enum):
    """Backtesting methods."""
    WALK_FORWARD = "walk_forward"  # Expanding window, fixed horizon
    SLIDING_WINDOW = "sliding_window"  # Fixed window, slides forward
    EXPANDING_WINDOW = "expanding_window"  # Growing window
    CROSS_VALIDATION = "cross_validation"  # K-fold style


@dataclass
class ForecastMetrics:
    """Standard forecasting evaluation metrics."""
    mape: float  # Mean Absolute Percentage Error
    rmse: float  # Root Mean Squared Error
    mae: float   # Mean Absolute Error
    mase: float  # Mean Absolute Scaled Error
    smape: float  # Symmetric MAPE
    mse: float   # Mean Squared Error
    r2: float    # R-squared
    bias: float  # Mean Error (bias)
    
    # Additional metrics
    max_error: float  # Maximum absolute error
    median_ae: float  # Median Absolute Error
    coverage: Optional[float] = None  # Prediction interval coverage
    
    # Directional accuracy
    direction_accuracy: Optional[float] = None  # % correct direction predictions
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'mape': self.mape,
            'rmse': self.rmse,
            'mae': self.mae,
            'mase': self.mase,
            'smape': self.smape,
            'mse': self.mse,
            'r2': self.r2,
            'bias': self.bias,
            'max_error': self.max_error,
            'median_ae': self.median_ae,
            'coverage': self.coverage,
            'direction_accuracy': self.direction_accuracy
        }


@dataclass
class BacktestFold:
    """Results from a single backtest fold."""
    fold_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    y_true: np.ndarray
    y_pred: np.ndarray
    y_pred_lower: Optional[np.ndarray] = None
    y_pred_upper: Optional[np.ndarray] = None
    metrics: Optional[ForecastMetrics] = None
    model_name: Optional[str] = None
    fit_time: float = 0.0
    predict_time: float = 0.0


@dataclass
class BacktestResult:
    """Complete backtesting results."""
    method: BacktestMethod
    model_name: str
    n_folds: int
    horizon: int
    
    # Aggregated metrics
    mean_metrics: ForecastMetrics
    std_metrics: Dict[str, float]
    
    # Per-fold results
    folds: List[BacktestFold]
    
    # Summary
    total_fit_time: float
    total_predict_time: float
    
    # Predictions across all folds
    all_actuals: np.ndarray
    all_predictions: np.ndarray
    
    # Additional info
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def summary(self) -> str:
        """Generate summary string."""
        return f"""
Backtest Results: {self.model_name}
{'='*50}
Method: {self.method.value}
Folds: {self.n_folds}
Horizon: {self.horizon}

Mean Metrics:
  MAPE:  {self.mean_metrics.mape:.2f}%
  RMSE:  {self.mean_metrics.rmse:.4f}
  MAE:   {self.mean_metrics.mae:.4f}
  MASE:  {self.mean_metrics.mase:.4f}
  R²:    {self.mean_metrics.r2:.4f}

Total Time: {self.total_fit_time + self.total_predict_time:.2f}s
"""


class MetricsCalculator:
    """Calculator for forecasting evaluation metrics."""
    
    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error."""
        mask = y_true != 0
        if not np.any(mask):
            return float('inf')
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error."""
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error."""
        return float(np.mean(np.abs(y_true - y_pred)))
    
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Squared Error."""
        return float(np.mean((y_true - y_pred) ** 2))
    
    @staticmethod
    def mase(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_train: Optional[np.ndarray] = None,
        seasonality: int = 1
    ) -> float:
        """
        Mean Absolute Scaled Error.
        
        Scales MAE by the in-sample naive forecast error.
        """
        if y_train is None:
            # Use seasonal naive on test set
            y_train = y_true
        
        # Naive forecast error
        if len(y_train) <= seasonality:
            naive_error = np.mean(np.abs(np.diff(y_train)))
        else:
            naive_error = np.mean(np.abs(y_train[seasonality:] - y_train[:-seasonality]))
        
        if naive_error == 0:
            return float('inf')
        
        mae = np.mean(np.abs(y_true - y_pred))
        return float(mae / naive_error)
    
    @staticmethod
    def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Symmetric Mean Absolute Percentage Error."""
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask = denominator != 0
        if not np.any(mask):
            return float('inf')
        return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100)
    
    @staticmethod
    def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R-squared (coefficient of determination)."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return 0.0
        return float(1 - ss_res / ss_tot)
    
    @staticmethod
    def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Error (bias)."""
        return float(np.mean(y_pred - y_true))
    
    @staticmethod
    def max_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Maximum Absolute Error."""
        return float(np.max(np.abs(y_true - y_pred)))
    
    @staticmethod
    def median_ae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Median Absolute Error."""
        return float(np.median(np.abs(y_true - y_pred)))
    
    @staticmethod
    def coverage(
        y_true: np.ndarray,
        y_pred_lower: np.ndarray,
        y_pred_upper: np.ndarray
    ) -> float:
        """Prediction interval coverage."""
        in_interval = (y_true >= y_pred_lower) & (y_true <= y_pred_upper)
        return float(np.mean(in_interval) * 100)
    
    @staticmethod
    def direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Directional accuracy (percentage of correct direction predictions).
        
        Useful for financial forecasting.
        """
        if len(y_true) < 2:
            return 0.0
        
        actual_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        
        correct = actual_direction == pred_direction
        return float(np.mean(correct) * 100)
    
    @classmethod
    def calculate_all(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_train: Optional[np.ndarray] = None,
        y_pred_lower: Optional[np.ndarray] = None,
        y_pred_upper: Optional[np.ndarray] = None,
        seasonality: int = 1
    ) -> ForecastMetrics:
        """Calculate all metrics."""
        # Handle NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) == 0:
            return ForecastMetrics(
                mape=float('inf'), rmse=float('inf'), mae=float('inf'),
                mase=float('inf'), smape=float('inf'), mse=float('inf'),
                r2=0.0, bias=0.0, max_error=float('inf'), median_ae=float('inf')
            )
        
        coverage_val = None
        if y_pred_lower is not None and y_pred_upper is not None:
            y_pred_lower = y_pred_lower[mask]
            y_pred_upper = y_pred_upper[mask]
            coverage_val = cls.coverage(y_true, y_pred_lower, y_pred_upper)
        
        direction_acc = None
        if len(y_true) >= 2:
            direction_acc = cls.direction_accuracy(y_true, y_pred)
        
        return ForecastMetrics(
            mape=cls.mape(y_true, y_pred),
            rmse=cls.rmse(y_true, y_pred),
            mae=cls.mae(y_true, y_pred),
            mase=cls.mase(y_true, y_pred, y_train, seasonality),
            smape=cls.smape(y_true, y_pred),
            mse=cls.mse(y_true, y_pred),
            r2=cls.r2(y_true, y_pred),
            bias=cls.bias(y_true, y_pred),
            max_error=cls.max_error(y_true, y_pred),
            median_ae=cls.median_ae(y_true, y_pred),
            coverage=coverage_val,
            direction_accuracy=direction_acc
        )


class Backtester:
    """
    Time Series Backtesting Framework.
    
    Provides comprehensive backtesting capabilities for validating
    forecasting models before production deployment.
    
    Supports multiple backtesting methods:
    - Walk-forward (expanding training window)
    - Sliding window (fixed-size training window)
    - Cross-validation (K-fold style)
    
    Example usage:
        # Define a forecasting function
        def forecast_fn(train_data, steps):
            from statsmodels.tsa.arima.model import ARIMA
            model = ARIMA(train_data, order=(1,1,1))
            fit = model.fit()
            return fit.forecast(steps=steps)
        
        # Run backtest
        backtester = Backtester(
            forecast_fn=forecast_fn,
            horizon=30,
            n_folds=5
        )
        
        result = backtester.run(data)
        print(result.summary())
    """
    
    def __init__(
        self,
        forecast_fn: Callable[[np.ndarray, int], np.ndarray],
        horizon: int = 30,
        n_folds: int = 5,
        method: BacktestMethod = BacktestMethod.WALK_FORWARD,
        initial_train_size: Optional[int] = None,
        step_size: Optional[int] = None,
        window_size: Optional[int] = None,
        seasonality: int = 1,
        model_name: str = "Model"
    ):
        """
        Initialize Backtester.
        
        Args:
            forecast_fn: Function that takes (train_data, steps) and returns predictions
            horizon: Forecast horizon (number of steps to predict)
            n_folds: Number of backtest folds
            method: Backtesting method
            initial_train_size: Initial training set size (default: 50% of data)
            step_size: Steps between folds (default: horizon)
            window_size: Training window size for sliding window method
            seasonality: Seasonality period for MASE calculation
            model_name: Name of the model being tested
        """
        self.forecast_fn = forecast_fn
        self.horizon = horizon
        self.n_folds = n_folds
        self.method = method
        self.initial_train_size = initial_train_size
        self.step_size = step_size or horizon
        self.window_size = window_size
        self.seasonality = seasonality
        self.model_name = model_name
        
        self.metrics_calculator = MetricsCalculator()
    
    def run(
        self,
        data: Union[np.ndarray, pd.Series, pd.DataFrame],
        value_col: Optional[str] = None
    ) -> BacktestResult:
        """
        Run backtesting.
        
        Args:
            data: Time series data
            value_col: Column name if DataFrame
        
        Returns:
            BacktestResult with all metrics and fold results
        """
        # Convert to numpy
        values = self._to_array(data, value_col)
        n = len(values)
        
        # Determine fold parameters
        if self.initial_train_size is None:
            initial_train_size = n // 2
        else:
            initial_train_size = self.initial_train_size
        
        # Validate parameters
        min_required = initial_train_size + self.horizon
        if n < min_required:
            raise ValueError(
                f"Data too short: {n} points, need at least {min_required}"
            )
        
        # Generate fold indices
        folds = self._generate_folds(n, initial_train_size)
        
        if len(folds) == 0:
            raise ValueError("Could not generate any backtest folds")
        
        # Run backtesting
        fold_results = []
        all_actuals = []
        all_predictions = []
        total_fit_time = 0.0
        total_predict_time = 0.0
        
        for fold_id, (train_start, train_end, test_start, test_end) in enumerate(folds):
            train_data = values[train_start:train_end]
            test_data = values[test_start:test_end]
            
            # Generate forecast
            import time
            
            try:
                fit_start = time.time()
                # Note: forecast_fn should handle fitting internally
                fit_end = time.time()
                
                predict_start = time.time()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    y_pred = self.forecast_fn(train_data, len(test_data))
                predict_end = time.time()
                
                y_pred = np.array(y_pred).flatten()[:len(test_data)]
                
                fit_time = fit_end - fit_start
                predict_time = predict_end - predict_start
            except Exception as e:
                logger.warning(f"Forecast failed for fold {fold_id}: {e}")
                y_pred = np.full(len(test_data), np.nan)
                fit_time = 0
                predict_time = 0
            
            # Calculate metrics
            metrics = self.metrics_calculator.calculate_all(
                test_data, y_pred, train_data, seasonality=self.seasonality
            )
            
            fold_result = BacktestFold(
                fold_id=fold_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                y_true=test_data,
                y_pred=y_pred,
                metrics=metrics,
                model_name=self.model_name,
                fit_time=fit_time,
                predict_time=predict_time
            )
            fold_results.append(fold_result)
            
            all_actuals.extend(test_data)
            all_predictions.extend(y_pred)
            total_fit_time += fit_time
            total_predict_time += predict_time
        
        # Aggregate metrics
        mean_metrics, std_metrics = self._aggregate_metrics(fold_results)
        
        return BacktestResult(
            method=self.method,
            model_name=self.model_name,
            n_folds=len(fold_results),
            horizon=self.horizon,
            mean_metrics=mean_metrics,
            std_metrics=std_metrics,
            folds=fold_results,
            total_fit_time=total_fit_time,
            total_predict_time=total_predict_time,
            all_actuals=np.array(all_actuals),
            all_predictions=np.array(all_predictions),
            metadata={
                "data_length": n,
                "initial_train_size": initial_train_size,
                "step_size": self.step_size
            }
        )
    
    def _to_array(
        self,
        data: Union[np.ndarray, pd.Series, pd.DataFrame],
        value_col: Optional[str] = None
    ) -> np.ndarray:
        """Convert input to numpy array."""
        if isinstance(data, np.ndarray):
            return data.flatten()
        elif isinstance(data, pd.Series):
            return data.values
        elif isinstance(data, pd.DataFrame):
            if value_col and value_col in data.columns:
                return data[value_col].values
            for col in ['value', 'close', 'price', 'y']:
                if col in data.columns:
                    return data[col].values
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                return data[numeric_cols[0]].values
            raise ValueError("Could not find numeric column")
        else:
            return np.array(data).flatten()
    
    def _generate_folds(
        self,
        n: int,
        initial_train_size: int
    ) -> List[Tuple[int, int, int, int]]:
        """Generate fold indices based on backtesting method."""
        folds = []
        
        if self.method == BacktestMethod.WALK_FORWARD:
            # Expanding window
            train_end = initial_train_size
            for fold in range(self.n_folds):
                test_start = train_end
                test_end = min(test_start + self.horizon, n)
                
                if test_end <= test_start:
                    break
                
                folds.append((0, train_end, test_start, test_end))
                train_end = min(train_end + self.step_size, n - self.horizon)
                
                if train_end >= n - self.horizon:
                    break
        
        elif self.method == BacktestMethod.SLIDING_WINDOW:
            # Fixed window size
            window_size = self.window_size or initial_train_size
            train_start = 0
            
            for fold in range(self.n_folds):
                train_end = train_start + window_size
                test_start = train_end
                test_end = min(test_start + self.horizon, n)
                
                if test_end <= test_start:
                    break
                
                folds.append((train_start, train_end, test_start, test_end))
                train_start += self.step_size
                
                if train_start + window_size >= n - self.horizon:
                    break
        
        elif self.method == BacktestMethod.EXPANDING_WINDOW:
            # Same as walk forward
            train_end = initial_train_size
            for fold in range(self.n_folds):
                test_start = train_end
                test_end = min(test_start + self.horizon, n)
                
                if test_end <= test_start:
                    break
                
                folds.append((0, train_end, test_start, test_end))
                train_end = test_end
                
                if train_end >= n:
                    break
        
        elif self.method == BacktestMethod.CROSS_VALIDATION:
            # Time series cross-validation (blocked)
            fold_size = (n - initial_train_size) // self.n_folds
            
            for fold in range(self.n_folds):
                train_end = initial_train_size + fold * fold_size
                test_start = train_end
                test_end = min(test_start + min(self.horizon, fold_size), n)
                
                if test_end <= test_start:
                    break
                
                folds.append((0, train_end, test_start, test_end))
        
        return folds[:self.n_folds]
    
    def _aggregate_metrics(
        self,
        fold_results: List[BacktestFold]
    ) -> Tuple[ForecastMetrics, Dict[str, float]]:
        """Aggregate metrics across folds."""
        metrics_list = [f.metrics for f in fold_results if f.metrics is not None]
        
        if not metrics_list:
            return ForecastMetrics(
                mape=float('nan'), rmse=float('nan'), mae=float('nan'),
                mase=float('nan'), smape=float('nan'), mse=float('nan'),
                r2=float('nan'), bias=float('nan'), max_error=float('nan'),
                median_ae=float('nan')
            ), {}
        
        # Calculate means
        mean_metrics = ForecastMetrics(
            mape=np.mean([m.mape for m in metrics_list if np.isfinite(m.mape)]),
            rmse=np.mean([m.rmse for m in metrics_list if np.isfinite(m.rmse)]),
            mae=np.mean([m.mae for m in metrics_list if np.isfinite(m.mae)]),
            mase=np.mean([m.mase for m in metrics_list if np.isfinite(m.mase)]),
            smape=np.mean([m.smape for m in metrics_list if np.isfinite(m.smape)]),
            mse=np.mean([m.mse for m in metrics_list if np.isfinite(m.mse)]),
            r2=np.mean([m.r2 for m in metrics_list if np.isfinite(m.r2)]),
            bias=np.mean([m.bias for m in metrics_list if np.isfinite(m.bias)]),
            max_error=np.mean([m.max_error for m in metrics_list if np.isfinite(m.max_error)]),
            median_ae=np.mean([m.median_ae for m in metrics_list if np.isfinite(m.median_ae)]),
            coverage=np.mean([m.coverage for m in metrics_list if m.coverage is not None]),
            direction_accuracy=np.mean([
                m.direction_accuracy for m in metrics_list
                if m.direction_accuracy is not None
            ]) if any(m.direction_accuracy is not None for m in metrics_list) else None
        )
        
        # Calculate standard deviations
        std_metrics = {
            'mape_std': np.std([m.mape for m in metrics_list if np.isfinite(m.mape)]),
            'rmse_std': np.std([m.rmse for m in metrics_list if np.isfinite(m.rmse)]),
            'mae_std': np.std([m.mae for m in metrics_list if np.isfinite(m.mae)]),
            'mase_std': np.std([m.mase for m in metrics_list if np.isfinite(m.mase)]),
            'r2_std': np.std([m.r2 for m in metrics_list if np.isfinite(m.r2)])
        }
        
        return mean_metrics, std_metrics


def backtest(
    data: Union[np.ndarray, pd.Series],
    forecast_fn: Callable[[np.ndarray, int], np.ndarray],
    horizon: int = 30,
    n_folds: int = 5,
    method: BacktestMethod = BacktestMethod.WALK_FORWARD,
    model_name: str = "Model"
) -> BacktestResult:
    """
    Quick function for backtesting.
    
    Args:
        data: Time series data
        forecast_fn: Forecasting function (train_data, steps) -> predictions
        horizon: Forecast horizon
        n_folds: Number of backtest folds
        method: Backtesting method
        model_name: Name of model
    
    Returns:
        BacktestResult
    """
    backtester = Backtester(
        forecast_fn=forecast_fn,
        horizon=horizon,
        n_folds=n_folds,
        method=method,
        model_name=model_name
    )
    return backtester.run(data)


def compare_models(
    data: Union[np.ndarray, pd.Series],
    forecast_fns: Dict[str, Callable[[np.ndarray, int], np.ndarray]],
    horizon: int = 30,
    n_folds: int = 5,
    method: BacktestMethod = BacktestMethod.WALK_FORWARD
) -> pd.DataFrame:
    """
    Compare multiple models using backtesting.
    
    Args:
        data: Time series data
        forecast_fns: Dictionary of {model_name: forecast_function}
        horizon: Forecast horizon
        n_folds: Number of backtest folds
        method: Backtesting method
    
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    for model_name, forecast_fn in forecast_fns.items():
        try:
            result = backtest(
                data, forecast_fn,
                horizon=horizon,
                n_folds=n_folds,
                method=method,
                model_name=model_name
            )
            
            results.append({
                'model': model_name,
                'mape': result.mean_metrics.mape,
                'rmse': result.mean_metrics.rmse,
                'mae': result.mean_metrics.mae,
                'mase': result.mean_metrics.mase,
                'r2': result.mean_metrics.r2,
                'direction_accuracy': result.mean_metrics.direction_accuracy,
                'total_time': result.total_fit_time + result.total_predict_time
            })
        except Exception as e:
            logger.warning(f"Backtest failed for {model_name}: {e}")
            results.append({
                'model': model_name,
                'mape': float('nan'),
                'rmse': float('nan'),
                'mae': float('nan'),
                'mase': float('nan'),
                'r2': float('nan'),
                'direction_accuracy': float('nan'),
                'total_time': float('nan')
            })
    
    return pd.DataFrame(results).sort_values('mape')


# Convenience class for quick metrics calculation
def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: Optional[np.ndarray] = None
) -> ForecastMetrics:
    """
    Quick function to calculate all forecast metrics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        y_train: Training data (for MASE calculation)
    
    Returns:
        ForecastMetrics
    """
    return MetricsCalculator.calculate_all(y_true, y_pred, y_train)
