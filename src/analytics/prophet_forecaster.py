"""
Prophet Forecaster Module.

This module provides Facebook/Meta Prophet integration for time series forecasting.
Prophet is an industry-standard model for:
- Automatic seasonality detection (yearly, weekly, daily)
- Holiday effects
- Robust to missing data and outliers
- Changepoint detection

Note: Requires 'prophet' package: pip install prophet
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import warnings

logger = logging.getLogger(__name__)

# Check if Prophet is available
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.warning("Prophet not installed. Install with: pip install prophet")


@dataclass
class ProphetForecastResult:
    """Result from Prophet forecasting."""
    forecast: pd.DataFrame  # Full forecast DataFrame with yhat, yhat_lower, yhat_upper
    predictions: np.ndarray  # Just the yhat values
    lower_bound: np.ndarray  # Lower confidence interval
    upper_bound: np.ndarray  # Upper confidence interval
    model_params: Dict[str, Any]  # Model parameters used
    components: Optional[pd.DataFrame]  # Trend, seasonality components
    changepoints: Optional[List[datetime]]  # Detected changepoints
    metrics: Dict[str, float]  # Model metrics (if historical data available)


class ProphetForecaster:
    """
    Prophet-based Time Series Forecaster.
    
    Implements Facebook/Meta's Prophet model for time series forecasting.
    Prophet is particularly good at:
    - Handling multiple seasonality (daily, weekly, yearly)
    - Automatic changepoint detection
    - Robust to missing data and outliers
    - Including holiday effects
    
    Example usage:
        forecaster = ProphetForecaster()
        
        # Prepare data (must have 'ds' and 'y' columns)
        df = pd.DataFrame({
            'ds': pd.date_range('2024-01-01', periods=100),
            'y': np.random.randn(100).cumsum()
        })
        
        # Fit and forecast
        result = forecaster.fit_predict(df, periods=30)
        print(result.predictions)
    """
    
    def __init__(
        self,
        seasonality_mode: str = 'multiplicative',
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0,
        yearly_seasonality: Union[bool, int] = 'auto',
        weekly_seasonality: Union[bool, int] = 'auto',
        daily_seasonality: Union[bool, int] = 'auto',
        interval_width: float = 0.95,
        growth: str = 'linear',
        n_changepoints: int = 25,
        holidays: Optional[pd.DataFrame] = None,
        **kwargs
    ):
        """
        Initialize ProphetForecaster.
        
        Args:
            seasonality_mode: 'additive' or 'multiplicative'
            changepoint_prior_scale: Flexibility of automatic changepoint selection
                                    (larger = more changepoints)
            seasonality_prior_scale: Strength of seasonality model
            yearly_seasonality: Include yearly seasonality (True/False/int for Fourier order)
            weekly_seasonality: Include weekly seasonality
            daily_seasonality: Include daily seasonality
            interval_width: Width of uncertainty intervals
            growth: 'linear' or 'logistic'
            n_changepoints: Number of potential changepoints
            holidays: DataFrame of holidays to include
            **kwargs: Additional Prophet parameters
        """
        if not PROPHET_AVAILABLE:
            raise ImportError(
                "Prophet is not installed. Install with: pip install prophet"
            )
        
        self.params = {
            'seasonality_mode': seasonality_mode,
            'changepoint_prior_scale': changepoint_prior_scale,
            'seasonality_prior_scale': seasonality_prior_scale,
            'yearly_seasonality': yearly_seasonality,
            'weekly_seasonality': weekly_seasonality,
            'daily_seasonality': daily_seasonality,
            'interval_width': interval_width,
            'growth': growth,
            'n_changepoints': n_changepoints,
            **kwargs
        }
        
        self.holidays = holidays
        self.model: Optional[Prophet] = None
        self._is_fitted = False
        self._train_data: Optional[pd.DataFrame] = None
    
    def _prepare_data(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        time_col: Optional[str] = None,
        value_col: Optional[str] = None,
        start_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Prepare data for Prophet (requires 'ds' and 'y' columns).
        
        Args:
            data: Input data
            time_col: Name of time column
            value_col: Name of value column
            start_date: Start date if creating dates from array
        
        Returns:
            DataFrame with 'ds' and 'y' columns
        """
        if isinstance(data, np.ndarray):
            # Create DataFrame from array
            n = len(data)
            if start_date is None:
                start_date = datetime.now() - timedelta(days=n)
            
            return pd.DataFrame({
                'ds': pd.date_range(start=start_date, periods=n, freq='D'),
                'y': data.flatten()
            })
        
        df = data.copy()
        
        # Handle column naming
        if 'ds' not in df.columns:
            if time_col and time_col in df.columns:
                df['ds'] = pd.to_datetime(df[time_col])
            elif 'timestamp' in df.columns:
                df['ds'] = pd.to_datetime(df['timestamp'])
            elif 'time' in df.columns:
                df['ds'] = pd.to_datetime(df['time'])
            elif 'date' in df.columns:
                df['ds'] = pd.to_datetime(df['date'])
            else:
                # Use index if it's datetime
                if isinstance(df.index, pd.DatetimeIndex):
                    df['ds'] = df.index
                else:
                    # Create dates
                    n = len(df)
                    start_date = start_date or (datetime.now() - timedelta(days=n))
                    df['ds'] = pd.date_range(start=start_date, periods=n, freq='D')
        
        if 'y' not in df.columns:
            if value_col and value_col in df.columns:
                df['y'] = df[value_col]
            elif 'value' in df.columns:
                df['y'] = df['value']
            elif 'close' in df.columns:
                df['y'] = df['close']
            elif 'price' in df.columns:
                df['y'] = df['price']
            else:
                # Use first numeric column
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    df['y'] = df[numeric_cols[0]]
                else:
                    raise ValueError("Could not find value column")
        
        return df[['ds', 'y']].dropna()
    
    def fit(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        time_col: Optional[str] = None,
        value_col: Optional[str] = None,
        start_date: Optional[datetime] = None
    ) -> 'ProphetForecaster':
        """
        Fit Prophet model to data.
        
        Args:
            data: Training data
            time_col: Name of time column
            value_col: Name of value column
            start_date: Start date if creating from array
        
        Returns:
            Self for chaining
        """
        # Prepare data
        df = self._prepare_data(data, time_col, value_col, start_date)
        self._train_data = df
        
        # Initialize Prophet model
        self.model = Prophet(**self.params)
        
        # Add holidays if provided
        if self.holidays is not None:
            self.model.add_country_holidays(country_name='US')  # Or use custom holidays
        
        # Fit model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(df)
        
        self._is_fitted = True
        return self
    
    def predict(
        self,
        periods: int,
        freq: str = 'D',
        include_history: bool = False
    ) -> ProphetForecastResult:
        """
        Generate forecasts for future periods.
        
        Args:
            periods: Number of periods to forecast
            freq: Frequency string (D=daily, H=hourly, etc.)
            include_history: Include historical predictions
        
        Returns:
            ProphetForecastResult with forecasts and components
        """
        if not self._is_fitted or self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Create future DataFrame
        future = self.model.make_future_dataframe(
            periods=periods,
            freq=freq,
            include_history=include_history
        )
        
        # Generate forecast
        forecast = self.model.predict(future)
        
        # Extract predictions for forecast period only
        if include_history:
            pred_df = forecast
        else:
            pred_df = forecast.tail(periods)
        
        predictions = pred_df['yhat'].values
        lower_bound = pred_df['yhat_lower'].values
        upper_bound = pred_df['yhat_upper'].values
        
        # Get changepoints
        changepoints = list(self.model.changepoints) if hasattr(self.model, 'changepoints') else None
        
        # Calculate metrics if we have historical data
        metrics = {}
        if self._train_data is not None and include_history:
            historical = forecast[forecast['ds'].isin(self._train_data['ds'])]
            if len(historical) > 0:
                y_true = self._train_data['y'].values
                y_pred = historical['yhat'].values[:len(y_true)]
                
                metrics = {
                    'mape': self._calculate_mape(y_true, y_pred),
                    'rmse': self._calculate_rmse(y_true, y_pred),
                    'mae': self._calculate_mae(y_true, y_pred)
                }
        
        return ProphetForecastResult(
            forecast=forecast,
            predictions=predictions,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            model_params=self.params,
            components=None,  # Can extract with model.plot_components()
            changepoints=changepoints,
            metrics=metrics
        )
    
    def fit_predict(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        periods: int,
        time_col: Optional[str] = None,
        value_col: Optional[str] = None,
        freq: str = 'D',
        start_date: Optional[datetime] = None
    ) -> ProphetForecastResult:
        """
        Fit model and generate forecasts in one step.
        
        Args:
            data: Training data
            periods: Number of periods to forecast
            time_col: Name of time column
            value_col: Name of value column
            freq: Frequency string
            start_date: Start date if creating from array
        
        Returns:
            ProphetForecastResult with forecasts
        """
        self.fit(data, time_col, value_col, start_date)
        return self.predict(periods, freq)
    
    def add_seasonality(
        self,
        name: str,
        period: float,
        fourier_order: int,
        prior_scale: Optional[float] = None,
        mode: Optional[str] = None
    ) -> 'ProphetForecaster':
        """
        Add custom seasonality before fitting.
        
        Args:
            name: Name of seasonality component
            period: Period of seasonality in days
            fourier_order: Number of Fourier terms
            prior_scale: Strength of seasonality
            mode: 'additive' or 'multiplicative'
        
        Returns:
            Self for chaining
        """
        if self._is_fitted:
            raise ValueError("Cannot add seasonality after fitting. Create new instance.")
        
        if self.model is None:
            self.model = Prophet(**self.params)
        
        self.model.add_seasonality(
            name=name,
            period=period,
            fourier_order=fourier_order,
            prior_scale=prior_scale,
            mode=mode
        )
        
        return self
    
    def add_regressor(
        self,
        name: str,
        prior_scale: Optional[float] = None,
        standardize: str = 'auto',
        mode: Optional[str] = None
    ) -> 'ProphetForecaster':
        """
        Add external regressor before fitting.
        
        Args:
            name: Name of regressor column
            prior_scale: Strength of regressor
            standardize: 'auto', True, or False
            mode: 'additive' or 'multiplicative'
        
        Returns:
            Self for chaining
        """
        if self._is_fitted:
            raise ValueError("Cannot add regressor after fitting. Create new instance.")
        
        if self.model is None:
            self.model = Prophet(**self.params)
        
        self.model.add_regressor(
            name=name,
            prior_scale=prior_scale,
            standardize=standardize,
            mode=mode
        )
        
        return self
    
    def get_changepoints(self) -> List[datetime]:
        """Get detected changepoints from fitted model."""
        if not self._is_fitted or self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return list(self.model.changepoints)
    
    def get_seasonality_components(
        self,
        periods: int = 0,
        freq: str = 'D'
    ) -> pd.DataFrame:
        """
        Get seasonality components from fitted model.
        
        Args:
            periods: Additional future periods (0 for history only)
            freq: Frequency string
        
        Returns:
            DataFrame with trend and seasonality components
        """
        if not self._is_fitted or self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        forecast = self.model.predict(future)
        
        # Extract components
        components = ['ds', 'trend']
        
        if 'yearly' in forecast.columns:
            components.append('yearly')
        if 'weekly' in forecast.columns:
            components.append('weekly')
        if 'daily' in forecast.columns:
            components.append('daily')
        
        return forecast[components]
    
    def cross_validate(
        self,
        initial: str = '730 days',
        period: str = '180 days',
        horizon: str = '365 days'
    ) -> pd.DataFrame:
        """
        Perform cross-validation on fitted model.
        
        Args:
            initial: Training period size
            period: Period between cutoffs
            horizon: Forecast horizon
        
        Returns:
            DataFrame with cross-validation results
        """
        if not self._is_fitted or self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        from prophet.diagnostics import cross_validation
        
        cv_results = cross_validation(
            self.model,
            initial=initial,
            period=period,
            horizon=horizon
        )
        
        return cv_results
    
    def performance_metrics(self, cv_results: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Calculate performance metrics from cross-validation.
        
        Args:
            cv_results: Cross-validation results (will run CV if not provided)
        
        Returns:
            DataFrame with performance metrics
        """
        if cv_results is None:
            cv_results = self.cross_validate()
        
        from prophet.diagnostics import performance_metrics
        
        return performance_metrics(cv_results)
    
    @staticmethod
    def _calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        mask = y_true != 0
        if not np.any(mask):
            return float('inf')
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    
    @staticmethod
    def _calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Squared Error."""
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    @staticmethod
    def _calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        return float(np.mean(np.abs(y_true - y_pred)))


class ProphetTrendDetector:
    """
    Prophet-based trend and changepoint detector.
    
    Uses Prophet's built-in changepoint detection to identify
    structural changes in time series data.
    """
    
    def __init__(
        self,
        changepoint_prior_scale: float = 0.05,
        n_changepoints: int = 25,
        changepoint_range: float = 0.8
    ):
        """
        Initialize ProphetTrendDetector.
        
        Args:
            changepoint_prior_scale: Flexibility of changepoint selection
            n_changepoints: Number of potential changepoints
            changepoint_range: Proportion of history for changepoints
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet not installed. Install with: pip install prophet")
        
        self.changepoint_prior_scale = changepoint_prior_scale
        self.n_changepoints = n_changepoints
        self.changepoint_range = changepoint_range
        self.model: Optional[Prophet] = None
    
    def detect(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        time_col: Optional[str] = None,
        value_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Detect trend and changepoints in data.
        
        Args:
            data: Time series data
            time_col: Name of time column
            value_col: Name of value column
        
        Returns:
            Dictionary with changepoints and trend analysis
        """
        # Prepare data
        forecaster = ProphetForecaster(
            changepoint_prior_scale=self.changepoint_prior_scale,
            n_changepoints=self.n_changepoints,
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False
        )
        
        # Fit model
        forecaster.fit(data, time_col, value_col)
        
        # Get changepoints
        changepoints = forecaster.get_changepoints()
        
        # Get trend components
        components = forecaster.get_seasonality_components()
        
        # Analyze trend changes
        trend_changes = []
        if len(changepoints) > 0:
            trend = components['trend'].values
            for cp in changepoints:
                idx = (components['ds'] == cp).argmax()
                if idx > 0 and idx < len(trend) - 1:
                    before_slope = trend[idx] - trend[idx - 1]
                    after_slope = trend[idx + 1] - trend[idx]
                    trend_changes.append({
                        'date': cp,
                        'index': int(idx),
                        'slope_before': float(before_slope),
                        'slope_after': float(after_slope),
                        'slope_change': float(after_slope - before_slope)
                    })
        
        return {
            'changepoints': [str(cp) for cp in changepoints],
            'n_changepoints': len(changepoints),
            'trend_changes': trend_changes,
            'trend': components['trend'].values.tolist(),
            'dates': components['ds'].dt.strftime('%Y-%m-%d').tolist()
        }


# Convenience functions
def prophet_forecast(
    data: Union[pd.DataFrame, np.ndarray],
    periods: int,
    time_col: Optional[str] = None,
    value_col: Optional[str] = None,
    **kwargs
) -> ProphetForecastResult:
    """
    Quick function for Prophet forecasting.
    
    Args:
        data: Time series data
        periods: Number of periods to forecast
        time_col: Name of time column
        value_col: Name of value column
        **kwargs: Additional Prophet parameters
    
    Returns:
        ProphetForecastResult
    """
    forecaster = ProphetForecaster(**kwargs)
    return forecaster.fit_predict(data, periods, time_col, value_col)


def detect_trend_changes(
    data: Union[pd.DataFrame, np.ndarray],
    time_col: Optional[str] = None,
    value_col: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Quick function to detect trend changes using Prophet.
    
    Args:
        data: Time series data
        time_col: Name of time column
        value_col: Name of value column
        **kwargs: Additional Prophet parameters
    
    Returns:
        Dictionary with changepoints and trend analysis
    """
    detector = ProphetTrendDetector(**kwargs)
    return detector.detect(data, time_col, value_col)


# Check availability
def is_prophet_available() -> bool:
    """Check if Prophet is available."""
    return PROPHET_AVAILABLE
