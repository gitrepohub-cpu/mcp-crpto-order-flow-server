"""
Time Series Analytics Engine

Provides Kats-like time series analysis capabilities using statsmodels, scipy, and scikit-learn.
Designed to work with the existing feature calculator framework and DuckDB storage.

Capabilities:
- Forecasting (ARIMA, Exponential Smoothing, Theta)
- Anomaly Detection (Z-score, IQR, Isolation Forest, CUSUM)
- Change Point Detection (PELT, Binary Segmentation, CUSUM)
- Feature Extraction (40+ statistical features)
- Seasonality Analysis
- Regime Detection
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
import warnings

import numpy as np
import pandas as pd
from scipy import stats, signal
from scipy.fft import fft

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class TimeSeriesData:
    """
    Standard time series data container.
    Compatible with institutional calculations that have timestamps.
    """
    time: pd.DatetimeIndex
    value: pd.Series
    name: str = "value"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        time_col: str = "timestamp",
        value_col: str = "value",
        name: Optional[str] = None
    ) -> "TimeSeriesData":
        """Create from DataFrame - works with institutional calculations."""
        time = pd.DatetimeIndex(df[time_col])
        value = pd.Series(df[value_col].values, index=time)
        return cls(time=time, value=value, name=name or value_col)
    
    @classmethod
    def from_duckdb_result(
        cls,
        results: List[tuple],
        time_idx: int = 0,
        value_idx: int = 1,
        name: str = "value"
    ) -> "TimeSeriesData":
        """Create from DuckDB query results."""
        if not results:
            return cls(
                time=pd.DatetimeIndex([]),
                value=pd.Series([], dtype=float),
                name=name
            )
        
        times = [r[time_idx] for r in results]
        values = [r[value_idx] for r in results]
        
        time_index = pd.DatetimeIndex(times)
        value_series = pd.Series(values, index=time_index, dtype=float)
        
        return cls(time=time_index, value=value_series, name=name)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        return pd.DataFrame({
            'time': self.time,
            self.name: self.value.values
        })
    
    def __len__(self):
        return len(self.value)
    
    def resample(self, freq: str, agg: str = 'mean') -> "TimeSeriesData":
        """Resample to different frequency."""
        df = self.to_dataframe().set_index('time')
        if agg == 'mean':
            resampled = df.resample(freq).mean()
        elif agg == 'sum':
            resampled = df.resample(freq).sum()
        elif agg == 'last':
            resampled = df.resample(freq).last()
        elif agg == 'first':
            resampled = df.resample(freq).first()
        else:
            resampled = df.resample(freq).mean()
        
        resampled = resampled.dropna()
        return TimeSeriesData(
            time=resampled.index,
            value=resampled[self.name],
            name=self.name,
            metadata=self.metadata
        )


@dataclass
class ForecastResult:
    """Container for forecast results."""
    forecast: pd.Series
    lower_bound: pd.Series
    upper_bound: pd.Series
    model_name: str
    confidence_level: float = 0.95
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'forecast': self.forecast.to_dict(),
            'lower_bound': self.lower_bound.to_dict(),
            'upper_bound': self.upper_bound.to_dict(),
            'model': self.model_name,
            'confidence': self.confidence_level,
            'metrics': self.metrics
        }


@dataclass
class AnomalyResult:
    """Container for anomaly detection results."""
    is_anomaly: pd.Series
    scores: pd.Series
    threshold: float
    method: str
    anomaly_indices: List[int] = field(default_factory=list)
    
    def get_anomaly_timestamps(self) -> List[datetime]:
        """Get timestamps of anomalies."""
        return list(self.is_anomaly[self.is_anomaly].index)
    
    def to_dict(self) -> Dict:
        return {
            'anomaly_count': int(self.is_anomaly.sum()),
            'anomaly_rate': float(self.is_anomaly.mean()),
            'threshold': self.threshold,
            'method': self.method,
            'anomaly_timestamps': [str(t) for t in self.get_anomaly_timestamps()[-10:]]
        }


@dataclass
class ChangePointResult:
    """Container for change point detection results."""
    change_points: List[int]
    timestamps: List[datetime]
    confidence_scores: List[float]
    method: str
    segments: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'change_point_count': len(self.change_points),
            'change_points': [
                {
                    'index': cp,
                    'timestamp': str(ts),
                    'confidence': conf
                }
                for cp, ts, conf in zip(self.change_points, self.timestamps, self.confidence_scores)
            ],
            'method': self.method,
            'segments': self.segments
        }


class MarketRegime(Enum):
    """Market regime classification."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"


@dataclass
class RegimeResult:
    """Container for regime detection results."""
    current_regime: MarketRegime
    confidence: float
    regime_history: List[Tuple[datetime, MarketRegime]]
    regime_durations: Dict[MarketRegime, float]
    transition_matrix: Dict[str, Dict[str, float]]
    
    def to_dict(self) -> Dict:
        return {
            'current_regime': self.current_regime.value,
            'confidence': self.confidence,
            'recent_regimes': [
                {'timestamp': str(t), 'regime': r.value}
                for t, r in self.regime_history[-10:]
            ],
            'regime_distribution': {
                r.value: d for r, d in self.regime_durations.items()
            }
        }


# ============================================================================
# TIME SERIES ANALYTICS ENGINE
# ============================================================================

class TimeSeriesEngine:
    """
    Main time series analytics engine.
    
    Provides forecasting, anomaly detection, change point detection,
    feature extraction, and regime detection.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        logger.info("TimeSeriesEngine initialized")
    
    # ========================================================================
    # FORECASTING
    # ========================================================================
    
    def forecast_arima(
        self,
        ts: TimeSeriesData,
        steps: int = 24,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        confidence: float = 0.95
    ) -> ForecastResult:
        """
        Forecast using ARIMA/SARIMA model.
        
        Args:
            ts: Time series data
            steps: Number of steps to forecast
            order: ARIMA order (p, d, q)
            seasonal_order: Seasonal order (P, D, Q, s) or None
            confidence: Confidence level for intervals
        
        Returns:
            ForecastResult with predictions and confidence intervals
        """
        try:
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            
            values = ts.value.dropna()
            
            if len(values) < 10:
                raise ValueError("Insufficient data for ARIMA (need at least 10 points)")
            
            if seasonal_order:
                model = SARIMAX(values, order=order, seasonal_order=seasonal_order)
            else:
                model = ARIMA(values, order=order)
            
            # Fit model (suppress warnings with warnings filter instead of deprecated disp param)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted = model.fit()
            
            # Forecast
            forecast_result = fitted.get_forecast(steps=steps)
            forecast = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int(alpha=1-confidence)
            
            # Generate future timestamps
            last_time = ts.time[-1]
            freq = pd.infer_freq(ts.time) or 'H'
            future_times = pd.date_range(start=last_time, periods=steps+1, freq=freq)[1:]
            
            forecast.index = future_times
            lower = pd.Series(conf_int.iloc[:, 0].values, index=future_times)
            upper = pd.Series(conf_int.iloc[:, 1].values, index=future_times)
            
            return ForecastResult(
                forecast=forecast,
                lower_bound=lower,
                upper_bound=upper,
                model_name="ARIMA" if not seasonal_order else "SARIMA",
                confidence_level=confidence,
                metrics={
                    'aic': fitted.aic,
                    'bic': fitted.bic,
                }
            )
            
        except Exception as e:
            logger.error(f"ARIMA forecast failed: {e}")
            raise
    
    def forecast_exponential_smoothing(
        self,
        ts: TimeSeriesData,
        steps: int = 24,
        seasonal_periods: Optional[int] = None,
        trend: str = 'add',
        seasonal: Optional[str] = 'add',
        confidence: float = 0.95
    ) -> ForecastResult:
        """
        Forecast using Exponential Smoothing (Holt-Winters).
        
        Args:
            ts: Time series data
            steps: Number of steps to forecast
            seasonal_periods: Number of periods in seasonal cycle
            trend: 'add', 'mul', or None
            seasonal: 'add', 'mul', or None
            confidence: Confidence level
        
        Returns:
            ForecastResult
        """
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            
            values = ts.value.dropna()
            
            if len(values) < 10:
                raise ValueError("Insufficient data for Exponential Smoothing")
            
            model = ExponentialSmoothing(
                values,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods
            )
            
            fitted = model.fit()
            forecast = fitted.forecast(steps)
            
            # Estimate confidence intervals using residual std
            residuals = fitted.resid
            std_resid = residuals.std()
            z_score = stats.norm.ppf((1 + confidence) / 2)
            
            # Generate future timestamps
            last_time = ts.time[-1]
            freq = pd.infer_freq(ts.time) or 'H'
            future_times = pd.date_range(start=last_time, periods=steps+1, freq=freq)[1:]
            
            forecast.index = future_times
            lower = forecast - z_score * std_resid
            upper = forecast + z_score * std_resid
            
            return ForecastResult(
                forecast=forecast,
                lower_bound=lower,
                upper_bound=upper,
                model_name="ExponentialSmoothing",
                confidence_level=confidence,
                metrics={
                    'sse': fitted.sse,
                }
            )
            
        except Exception as e:
            logger.error(f"Exponential Smoothing forecast failed: {e}")
            raise
    
    def forecast_theta(
        self,
        ts: TimeSeriesData,
        steps: int = 24,
        confidence: float = 0.95
    ) -> ForecastResult:
        """
        Forecast using Theta method.
        Simple but effective for many time series.
        
        Args:
            ts: Time series data
            steps: Steps to forecast
            confidence: Confidence level
        
        Returns:
            ForecastResult
        """
        try:
            from statsmodels.tsa.forecasting.theta import ThetaModel
            
            values = ts.value.dropna()
            
            if len(values) < 10:
                raise ValueError("Insufficient data for Theta model")
            
            model = ThetaModel(values)
            fitted = model.fit()
            forecast = fitted.forecast(steps)
            
            # Prediction intervals
            pred_int = fitted.prediction_intervals(steps, alpha=1-confidence)
            
            # Generate future timestamps
            last_time = ts.time[-1]
            freq = pd.infer_freq(ts.time) or 'H'
            future_times = pd.date_range(start=last_time, periods=steps+1, freq=freq)[1:]
            
            forecast.index = future_times
            lower = pd.Series(pred_int['lower'].values, index=future_times)
            upper = pd.Series(pred_int['upper'].values, index=future_times)
            
            return ForecastResult(
                forecast=forecast,
                lower_bound=lower,
                upper_bound=upper,
                model_name="Theta",
                confidence_level=confidence,
                metrics={}
            )
            
        except Exception as e:
            logger.error(f"Theta forecast failed: {e}")
            raise
    
    def auto_forecast(
        self,
        ts: TimeSeriesData,
        steps: int = 24,
        confidence: float = 0.95
    ) -> ForecastResult:
        """
        Automatically select best forecasting model.
        Tries multiple models and returns best based on AIC/BIC.
        
        Args:
            ts: Time series data
            steps: Steps to forecast
            confidence: Confidence level
        
        Returns:
            ForecastResult from best model
        """
        results = []
        
        # Try ARIMA
        try:
            result = self.forecast_arima(ts, steps, confidence=confidence)
            results.append(('ARIMA', result, result.metrics.get('aic', float('inf'))))
        except Exception as e:
            logger.debug(f"ARIMA forecasting failed: {e}")
        
        # Try Exponential Smoothing
        try:
            result = self.forecast_exponential_smoothing(ts, steps, confidence=confidence)
            results.append(('ExpSmooth', result, result.metrics.get('sse', float('inf'))))
        except Exception as e:
            logger.debug(f"Exponential Smoothing forecasting failed: {e}")
        
        # Try Theta
        try:
            result = self.forecast_theta(ts, steps, confidence=confidence)
            results.append(('Theta', result, 0))  # Theta doesn't have AIC
        except Exception as e:
            logger.debug(f"Theta forecasting failed: {e}")
        
        if not results:
            raise ValueError("All forecasting methods failed")
        
        # Return best result (lowest AIC/metric)
        best = min(results, key=lambda x: x[2])
        logger.info(f"Auto-selected model: {best[0]}")
        return best[1]
    
    # ========================================================================
    # ANOMALY DETECTION
    # ========================================================================
    
    def detect_anomalies_zscore(
        self,
        ts: TimeSeriesData,
        threshold: float = 3.0,
        window: Optional[int] = None
    ) -> AnomalyResult:
        """
        Detect anomalies using Z-score method.
        
        Args:
            ts: Time series data
            threshold: Z-score threshold for anomaly
            window: Rolling window for local z-score (None for global)
        
        Returns:
            AnomalyResult
        """
        values = ts.value.dropna()
        
        if window:
            # Rolling z-score
            rolling_mean = values.rolling(window=window).mean()
            rolling_std = values.rolling(window=window).std()
            z_scores = (values - rolling_mean) / rolling_std
        else:
            # Global z-score
            mean = values.mean()
            std = values.std()
            z_scores = (values - mean) / std
        
        is_anomaly = z_scores.abs() > threshold
        
        return AnomalyResult(
            is_anomaly=is_anomaly,
            scores=z_scores,
            threshold=threshold,
            method="z-score" + (f"_rolling_{window}" if window else "_global"),
            anomaly_indices=list(is_anomaly[is_anomaly].index)
        )
    
    def detect_anomalies_iqr(
        self,
        ts: TimeSeriesData,
        multiplier: float = 1.5,
        window: Optional[int] = None
    ) -> AnomalyResult:
        """
        Detect anomalies using Interquartile Range (IQR) method.
        
        Args:
            ts: Time series data
            multiplier: IQR multiplier for bounds
            window: Rolling window (None for global)
        
        Returns:
            AnomalyResult
        """
        values = ts.value.dropna()
        
        if window:
            q1 = values.rolling(window=window).quantile(0.25)
            q3 = values.rolling(window=window).quantile(0.75)
        else:
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
        
        iqr = q3 - q1
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        is_anomaly = (values < lower_bound) | (values > upper_bound)
        scores = pd.Series(0.0, index=values.index)
        scores[values < lower_bound] = (lower_bound - values[values < lower_bound]) / iqr if isinstance(iqr, float) else (lower_bound - values[values < lower_bound]) / iqr[values < lower_bound]
        scores[values > upper_bound] = (values[values > upper_bound] - upper_bound) / iqr if isinstance(iqr, float) else (values[values > upper_bound] - upper_bound) / iqr[values > upper_bound]
        
        return AnomalyResult(
            is_anomaly=is_anomaly,
            scores=scores,
            threshold=multiplier,
            method="iqr" + (f"_rolling_{window}" if window else "_global"),
            anomaly_indices=list(is_anomaly[is_anomaly].index)
        )
    
    def detect_anomalies_isolation_forest(
        self,
        ts: TimeSeriesData,
        contamination: float = 0.05,
        n_estimators: int = 100
    ) -> AnomalyResult:
        """
        Detect anomalies using Isolation Forest.
        
        Args:
            ts: Time series data
            contamination: Expected proportion of anomalies
            n_estimators: Number of trees
        
        Returns:
            AnomalyResult
        """
        try:
            from sklearn.ensemble import IsolationForest
            
            values = ts.value.dropna()
            X = values.values.reshape(-1, 1)
            
            model = IsolationForest(
                contamination=contamination,
                n_estimators=n_estimators,
                random_state=42
            )
            
            predictions = model.fit_predict(X)
            scores = model.score_samples(X)
            
            is_anomaly = pd.Series(predictions == -1, index=values.index)
            score_series = pd.Series(-scores, index=values.index)  # Negate so higher = more anomalous
            
            return AnomalyResult(
                is_anomaly=is_anomaly,
                scores=score_series,
                threshold=contamination,
                method="isolation_forest",
                anomaly_indices=list(is_anomaly[is_anomaly].index)
            )
            
        except Exception as e:
            logger.error(f"Isolation Forest failed: {e}")
            raise
    
    def detect_anomalies_cusum(
        self,
        ts: TimeSeriesData,
        threshold: float = 5.0,
        drift: float = 0.0
    ) -> AnomalyResult:
        """
        Detect anomalies using CUSUM (Cumulative Sum) method.
        Good for detecting shifts in mean.
        
        Args:
            ts: Time series data
            threshold: Detection threshold
            drift: Drift allowance
        
        Returns:
            AnomalyResult
        """
        values = ts.value.dropna()
        mean = values.mean()
        std = values.std()
        
        # Standardize
        z = (values - mean) / std
        
        # CUSUM
        cusum_pos = np.zeros(len(z))
        cusum_neg = np.zeros(len(z))
        
        for i in range(1, len(z)):
            cusum_pos[i] = max(0, cusum_pos[i-1] + z.iloc[i] - drift)
            cusum_neg[i] = max(0, cusum_neg[i-1] - z.iloc[i] - drift)
        
        scores = pd.Series(np.maximum(cusum_pos, cusum_neg), index=values.index)
        is_anomaly = scores > threshold
        
        return AnomalyResult(
            is_anomaly=is_anomaly,
            scores=scores,
            threshold=threshold,
            method="cusum",
            anomaly_indices=list(is_anomaly[is_anomaly].index)
        )
    
    # ========================================================================
    # CHANGE POINT DETECTION
    # ========================================================================
    
    def detect_change_points_cusum(
        self,
        ts: TimeSeriesData,
        threshold: float = 5.0
    ) -> ChangePointResult:
        """
        Detect change points using CUSUM method.
        
        Args:
            ts: Time series data
            threshold: Detection threshold
        
        Returns:
            ChangePointResult
        """
        values = ts.value.dropna()
        mean = values.mean()
        std = values.std()
        
        z = (values - mean) / std
        
        cusum_pos = np.zeros(len(z))
        cusum_neg = np.zeros(len(z))
        
        change_points = []
        
        for i in range(1, len(z)):
            cusum_pos[i] = max(0, cusum_pos[i-1] + z.iloc[i])
            cusum_neg[i] = max(0, cusum_neg[i-1] - z.iloc[i])
            
            if cusum_pos[i] > threshold or cusum_neg[i] > threshold:
                change_points.append(i)
                cusum_pos[i] = 0
                cusum_neg[i] = 0
        
        timestamps = [ts.time[cp] for cp in change_points]
        confidences = [1.0] * len(change_points)  # CUSUM doesn't provide confidence
        
        return ChangePointResult(
            change_points=change_points,
            timestamps=timestamps,
            confidence_scores=confidences,
            method="cusum",
            segments=self._compute_segments(ts, change_points)
        )
    
    def detect_change_points_binary_segmentation(
        self,
        ts: TimeSeriesData,
        min_segment_length: int = 10,
        penalty: float = 3.0
    ) -> ChangePointResult:
        """
        Detect change points using Binary Segmentation.
        
        Args:
            ts: Time series data
            min_segment_length: Minimum segment length
            penalty: Penalty for adding change points
        
        Returns:
            ChangePointResult
        """
        values = ts.value.dropna().values
        n = len(values)
        
        def segment_cost(start: int, end: int) -> float:
            """Cost of a segment (variance)."""
            if end - start < 2:
                return 0
            segment = values[start:end]
            return len(segment) * np.var(segment)
        
        def find_best_split(start: int, end: int) -> Tuple[int, float]:
            """Find best split point."""
            if end - start < 2 * min_segment_length:
                return -1, float('inf')
            
            best_split = -1
            best_cost = segment_cost(start, end)
            
            for split in range(start + min_segment_length, end - min_segment_length + 1):
                cost = segment_cost(start, split) + segment_cost(split, end) + penalty
                if cost < best_cost:
                    best_cost = cost
                    best_split = split
            
            return best_split, best_cost
        
        change_points = []
        
        def recursive_split(start: int, end: int):
            split, cost = find_best_split(start, end)
            if split > 0:
                change_points.append(split)
                recursive_split(start, split)
                recursive_split(split, end)
        
        recursive_split(0, n)
        change_points = sorted(change_points)
        
        timestamps = [ts.time[cp] for cp in change_points]
        confidences = [0.8] * len(change_points)  # Fixed confidence
        
        return ChangePointResult(
            change_points=change_points,
            timestamps=timestamps,
            confidence_scores=confidences,
            method="binary_segmentation",
            segments=self._compute_segments(ts, change_points)
        )
    
    def _compute_segments(
        self,
        ts: TimeSeriesData,
        change_points: List[int]
    ) -> List[Dict]:
        """Compute segment statistics."""
        values = ts.value.dropna()
        segments = []
        
        boundaries = [0] + change_points + [len(values)]
        
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            segment_values = values.iloc[start:end]
            
            segments.append({
                'start_idx': start,
                'end_idx': end,
                'start_time': str(ts.time[start]),
                'end_time': str(ts.time[min(end-1, len(ts.time)-1)]),
                'mean': float(segment_values.mean()),
                'std': float(segment_values.std()),
                'length': end - start
            })
        
        return segments
    
    # ========================================================================
    # FEATURE EXTRACTION
    # ========================================================================
    
    def extract_features(
        self,
        ts: TimeSeriesData,
        include_spectral: bool = True
    ) -> Dict[str, float]:
        """
        Extract 40+ statistical features from time series.
        
        Args:
            ts: Time series data
            include_spectral: Include spectral/frequency features
        
        Returns:
            Dictionary of feature names to values
        """
        values = ts.value.dropna()
        n = len(values)
        
        if n < 10:
            return {'error': 'Insufficient data'}
        
        features = {}
        
        # Basic statistics
        features['mean'] = float(values.mean())
        features['std'] = float(values.std())
        features['variance'] = float(values.var())
        features['min'] = float(values.min())
        features['max'] = float(values.max())
        features['range'] = features['max'] - features['min']
        features['median'] = float(values.median())
        features['skewness'] = float(stats.skew(values))
        features['kurtosis'] = float(stats.kurtosis(values))
        
        # Percentiles
        features['p10'] = float(values.quantile(0.1))
        features['p25'] = float(values.quantile(0.25))
        features['p75'] = float(values.quantile(0.75))
        features['p90'] = float(values.quantile(0.9))
        features['iqr'] = features['p75'] - features['p25']
        
        # Trend features
        x = np.arange(n)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        features['trend_slope'] = float(slope)
        features['trend_r_squared'] = float(r_value ** 2)
        features['trend_p_value'] = float(p_value)
        
        # Autocorrelation features
        try:
            from statsmodels.tsa.stattools import acf, pacf
            
            acf_values = acf(values, nlags=min(10, n//2), fft=True)
            features['acf_lag1'] = float(acf_values[1]) if len(acf_values) > 1 else 0
            features['acf_lag2'] = float(acf_values[2]) if len(acf_values) > 2 else 0
            features['acf_lag5'] = float(acf_values[5]) if len(acf_values) > 5 else 0
            features['acf_lag10'] = float(acf_values[10]) if len(acf_values) > 10 else 0
            
            if n > 20:
                pacf_values = pacf(values, nlags=min(10, n//3))
                features['pacf_lag1'] = float(pacf_values[1]) if len(pacf_values) > 1 else 0
                features['pacf_lag5'] = float(pacf_values[5]) if len(pacf_values) > 5 else 0
        except Exception as e:
            logger.debug(f"ACF/PACF calculation failed: {e}")
            features['acf_lag1'] = 0
            features['acf_lag2'] = 0
            features['acf_lag5'] = 0
            features['acf_lag10'] = 0
            features['pacf_lag1'] = 0
            features['pacf_lag5'] = 0
        
        # Stationarity
        try:
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(values, maxlag=min(10, n//3))
            features['adf_statistic'] = float(adf_result[0])
            features['adf_pvalue'] = float(adf_result[1])
            features['is_stationary'] = float(adf_result[1] < 0.05)
        except Exception as e:
            logger.debug(f"ADF stationarity test failed: {e}")
            features['adf_statistic'] = 0
            features['adf_pvalue'] = 1
            features['is_stationary'] = 0
        
        # Volatility features
        returns = values.pct_change().dropna()
        if len(returns) > 1:
            features['volatility'] = float(returns.std() * np.sqrt(252))  # Annualized
            features['return_mean'] = float(returns.mean())
            features['return_skew'] = float(stats.skew(returns))
            features['return_kurtosis'] = float(stats.kurtosis(returns))
        
        # Complexity features
        features['sample_entropy'] = self._sample_entropy(values.values)
        features['hurst_exponent'] = self._hurst_exponent(values.values)
        
        # Level shift features
        diff = values.diff().dropna()
        features['diff_mean'] = float(diff.mean())
        features['diff_std'] = float(diff.std())
        features['zero_crossings'] = int((diff.shift(1) * diff < 0).sum())
        
        # Spectral features
        if include_spectral and n > 20:
            try:
                fft_values = np.abs(fft(values.values))[:n//2]
                freqs = np.fft.fftfreq(n)[:n//2]
                
                fft_sum = np.sum(fft_values)
                features['spectral_centroid'] = float(np.sum(freqs * fft_values) / fft_sum) if fft_sum > 0 else 0
                features['spectral_energy'] = float(np.sum(fft_values ** 2))
                features['dominant_frequency'] = float(freqs[np.argmax(fft_values)]) if len(fft_values) > 0 else 0
            except Exception as e:
                logger.debug(f"Spectral features calculation failed: {e}")
                features['spectral_centroid'] = 0
                features['spectral_energy'] = 0
                features['dominant_frequency'] = 0
        
        return features
    
    def _sample_entropy(self, data: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Calculate sample entropy."""
        n = len(data)
        if n < m + 2:
            return 0.0
        
        r_threshold = r * np.std(data)
        
        def count_matches(template_length):
            count = 0
            for i in range(n - template_length):
                for j in range(i + 1, n - template_length):
                    if np.max(np.abs(data[i:i+template_length] - data[j:j+template_length])) < r_threshold:
                        count += 1
            return count
        
        A = count_matches(m + 1)
        B = count_matches(m)
        
        if B == 0:
            return 0.0
        
        return -np.log(A / B) if A > 0 else 0.0
    
    def _hurst_exponent(self, data: np.ndarray) -> float:
        """Estimate Hurst exponent using R/S analysis."""
        n = len(data)
        if n < 20:
            return 0.5
        
        max_k = min(n // 4, 100)
        if max_k < 2:
            return 0.5
        
        rs_values = []
        n_values = []
        
        for k in range(2, max_k):
            # Divide into k segments
            seg_len = n // k
            if seg_len < 2:
                break
            
            rs_seg = []
            for i in range(k):
                segment = data[i*seg_len:(i+1)*seg_len]
                mean_seg = np.mean(segment)
                dev = segment - mean_seg
                cumsum = np.cumsum(dev)
                r = np.max(cumsum) - np.min(cumsum)
                s = np.std(segment)
                if s > 0:
                    rs_seg.append(r / s)
            
            if rs_seg:
                rs_values.append(np.mean(rs_seg))
                n_values.append(seg_len)
        
        if len(rs_values) < 2:
            return 0.5
        
        # Linear regression on log-log
        log_n = np.log(n_values)
        log_rs = np.log(rs_values)
        
        slope, _, _, _, _ = stats.linregress(log_n, log_rs)
        return float(np.clip(slope, 0, 1))
    
    # ========================================================================
    # SEASONALITY ANALYSIS
    # ========================================================================
    
    def detect_seasonality(
        self,
        ts: TimeSeriesData,
        max_period: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Detect seasonality patterns.
        
        Args:
            ts: Time series data
            max_period: Maximum period to check
        
        Returns:
            Dictionary with seasonality info
        """
        values = ts.value.dropna()
        n = len(values)
        
        if n < 20:
            return {'seasonal': False, 'error': 'Insufficient data'}
        
        max_period = max_period or min(n // 2, 100)
        
        # Use FFT to find dominant frequencies
        fft_values = np.abs(fft(values.values))[:n//2]
        freqs = np.fft.fftfreq(n, d=1)[:n//2]
        
        # Find peaks
        peaks_idx = np.argsort(fft_values)[-5:]  # Top 5 peaks
        
        dominant_periods = []
        for idx in peaks_idx:
            if freqs[idx] > 0:
                period = int(1 / freqs[idx])
                if 2 <= period <= max_period:
                    strength = fft_values[idx] / np.sum(fft_values)
                    dominant_periods.append({
                        'period': period,
                        'strength': float(strength),
                        'frequency': float(freqs[idx])
                    })
        
        # Sort by strength
        dominant_periods.sort(key=lambda x: x['strength'], reverse=True)
        
        # Check if seasonality is significant
        has_seasonality = len(dominant_periods) > 0 and dominant_periods[0]['strength'] > 0.1
        
        return {
            'seasonal': has_seasonality,
            'dominant_periods': dominant_periods[:3],
            'primary_period': dominant_periods[0]['period'] if dominant_periods else None,
            'seasonality_strength': dominant_periods[0]['strength'] if dominant_periods else 0
        }
    
    def decompose_seasonality(
        self,
        ts: TimeSeriesData,
        period: Optional[int] = None,
        model: str = 'additive'
    ) -> Dict[str, pd.Series]:
        """
        Decompose time series into trend, seasonal, and residual.
        
        Args:
            ts: Time series data
            period: Seasonal period (auto-detected if None)
            model: 'additive' or 'multiplicative'
        
        Returns:
            Dictionary with trend, seasonal, residual components
        """
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            values = ts.value.dropna()
            
            if period is None:
                seasonality = self.detect_seasonality(ts)
                period = seasonality.get('primary_period', 24)
            
            if period is None or len(values) < 2 * period:
                return {'error': 'Cannot decompose - insufficient data or period'}
            
            decomposition = seasonal_decompose(values, model=model, period=period)
            
            return {
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid,
                'period': period,
                'model': model
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    # ========================================================================
    # REGIME DETECTION
    # ========================================================================
    
    def detect_regime(
        self,
        ts: TimeSeriesData,
        lookback: int = 50,
        volatility_window: int = 20
    ) -> RegimeResult:
        """
        Detect market regime (trending, ranging, volatile).
        
        Args:
            ts: Time series data
            lookback: Lookback period for regime detection
            volatility_window: Window for volatility calculation
        
        Returns:
            RegimeResult
        """
        values = ts.value.dropna()
        n = len(values)
        
        if n < lookback:
            lookback = n // 2
        
        # Calculate rolling metrics
        returns = values.pct_change().dropna()
        rolling_vol = returns.rolling(window=volatility_window).std() * np.sqrt(252)
        rolling_mean = values.rolling(window=lookback).mean()
        
        # Regime classification
        regimes = []
        
        for i in range(lookback, n):
            window_values = values.iloc[i-lookback:i]
            window_returns = returns.iloc[max(0, i-lookback):i]
            
            # Calculate metrics
            trend_slope = np.polyfit(range(lookback), window_values, 1)[0]
            volatility = window_returns.std() * np.sqrt(252) if len(window_returns) > 1 else 0
            vol_percentile = (rolling_vol.iloc[:i] < volatility).mean() if i > volatility_window else 0.5
            
            # Classify regime
            if vol_percentile > 0.9:
                regime = MarketRegime.HIGH_VOLATILITY
            elif vol_percentile < 0.1:
                regime = MarketRegime.LOW_VOLATILITY
            elif trend_slope > 0.001 * window_values.mean():
                regime = MarketRegime.TRENDING_UP
            elif trend_slope < -0.001 * window_values.mean():
                regime = MarketRegime.TRENDING_DOWN
            else:
                regime = MarketRegime.RANGING
            
            regimes.append((ts.time[i], regime))
        
        # Current regime
        current_regime = regimes[-1][1] if regimes else MarketRegime.RANGING
        
        # Calculate regime durations
        regime_counts = {}
        for _, regime in regimes:
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        total = len(regimes)
        regime_durations = {r: c / total for r, c in regime_counts.items()}
        
        # Calculate transition matrix
        transitions = {}
        for i in range(1, len(regimes)):
            from_regime = regimes[i-1][1].value
            to_regime = regimes[i][1].value
            
            if from_regime not in transitions:
                transitions[from_regime] = {}
            transitions[from_regime][to_regime] = transitions[from_regime].get(to_regime, 0) + 1
        
        # Normalize
        for from_r in transitions:
            total = sum(transitions[from_r].values())
            for to_r in transitions[from_r]:
                transitions[from_r][to_r] /= total
        
        # Confidence based on regime stability
        recent_regimes = [r for _, r in regimes[-10:]]
        confidence = recent_regimes.count(current_regime) / len(recent_regimes) if recent_regimes else 0.5
        
        return RegimeResult(
            current_regime=current_regime,
            confidence=confidence,
            regime_history=regimes[-50:],
            regime_durations=regime_durations,
            transition_matrix=transitions
        )


# ============================================================================
# SINGLETON ACCESSOR
# ============================================================================

_engine: Optional[TimeSeriesEngine] = None


def get_timeseries_engine() -> TimeSeriesEngine:
    """Get the global TimeSeriesEngine instance."""
    global _engine
    if _engine is None:
        _engine = TimeSeriesEngine()
    return _engine
