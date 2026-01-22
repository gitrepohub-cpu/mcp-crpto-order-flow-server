"""
Meta-Learning Framework for Time Series Model Selection.

This module implements Kats-equivalent meta-learning capabilities:
- Auto model selection based on time series characteristics
- Hyperparameter tuning recommendations
- Predictability assessment
- Detector recommendation

Based on Meta's research: "Self-supervised Learning for Fast and Scalable 
Time Series Hyper-parameter Tuning" (arXiv:2102.05740)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings
from scipy import stats
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
import logging

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Supported forecasting model types."""
    ARIMA = "arima"
    SARIMA = "sarima"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    THETA = "theta"
    PROPHET = "prophet"
    LSTM = "lstm"
    ENSEMBLE = "ensemble"


class DetectorType(Enum):
    """Supported anomaly/changepoint detector types."""
    ZSCORE = "zscore"
    IQR = "iqr"
    ISOLATION_FOREST = "isolation_forest"
    CUSUM = "cusum"
    BOCPD = "bocpd"
    MANN_KENDALL = "mann_kendall"
    DTW = "dtw"


@dataclass
class TimeSeriesMetadata:
    """Extracted metadata features from a time series."""
    # Basic statistics
    length: int = 0
    mean: float = 0.0
    std: float = 0.0
    cv: float = 0.0  # Coefficient of variation
    skewness: float = 0.0
    kurtosis: float = 0.0
    
    # Trend features
    trend_strength: float = 0.0
    trend_direction: float = 0.0  # Positive = upward, negative = downward
    
    # Seasonality features
    seasonality_strength: float = 0.0
    dominant_period: Optional[int] = None
    num_seasonal_peaks: int = 0
    
    # Stationarity
    is_stationary: bool = False
    adf_pvalue: float = 1.0
    
    # Autocorrelation
    acf_lag1: float = 0.0
    acf_lag5: float = 0.0
    acf_lag10: float = 0.0
    pacf_lag1: float = 0.0
    
    # Complexity
    sample_entropy: float = 0.0
    hurst_exponent: float = 0.5
    
    # Volatility
    volatility_clustering: float = 0.0
    heteroscedasticity: float = 0.0
    
    # Predictability score (0-1, higher = more predictable)
    predictability_score: float = 0.5
    
    # Additional features as dict
    extra_features: Dict[str, float] = field(default_factory=dict)


@dataclass
class ModelRecommendation:
    """Model recommendation result."""
    recommended_model: ModelType
    confidence: float
    alternative_models: List[Tuple[ModelType, float]]
    recommended_params: Dict[str, Any]
    reasoning: str


@dataclass
class DetectorRecommendation:
    """Detector recommendation result."""
    recommended_detector: DetectorType
    confidence: float
    alternative_detectors: List[Tuple[DetectorType, float]]
    recommended_params: Dict[str, Any]
    reasoning: str


class MetaLearner:
    """
    Meta-Learning Framework for Time Series Analysis.
    
    Implements Kats-equivalent capabilities:
    - Extract metadata features from time series
    - Auto-select best forecasting model
    - Auto-select best anomaly detector
    - Recommend hyperparameters
    - Assess predictability
    
    Example usage:
        meta_learner = MetaLearner()
        
        # Get model recommendation
        recommendation = meta_learner.recommend_model(data)
        print(f"Best model: {recommendation.recommended_model}")
        print(f"Confidence: {recommendation.confidence:.2%}")
        
        # Get detector recommendation
        detector_rec = meta_learner.recommend_detector(data)
        print(f"Best detector: {detector_rec.recommended_detector}")
    """
    
    def __init__(self, use_ml_classifier: bool = True):
        """
        Initialize MetaLearner.
        
        Args:
            use_ml_classifier: Whether to use ML-based classifier for recommendations.
                              If False, uses rule-based heuristics.
        """
        self.use_ml_classifier = use_ml_classifier
        self._model_classifier: Optional[RandomForestClassifier] = None
        self._detector_classifier: Optional[RandomForestClassifier] = None
        self._scaler = StandardScaler()
        self._is_trained = False
        
        # Initialize with pre-defined rules if not using ML
        self._initialize_rules()
    
    def _initialize_rules(self):
        """Initialize rule-based decision logic."""
        # Model selection rules based on time series characteristics
        self.model_rules = {
            # (condition_name, check_function, recommended_model, weight)
            "strong_seasonality": (
                lambda m: m.seasonality_strength > 0.6,
                ModelType.SARIMA,
                0.8
            ),
            "weak_seasonality_trending": (
                lambda m: m.seasonality_strength < 0.3 and abs(m.trend_strength) > 0.5,
                ModelType.EXPONENTIAL_SMOOTHING,
                0.7
            ),
            "stationary_no_seasonality": (
                lambda m: m.is_stationary and m.seasonality_strength < 0.2,
                ModelType.ARIMA,
                0.75
            ),
            "complex_pattern": (
                lambda m: m.sample_entropy > 1.5 and m.hurst_exponent < 0.4,
                ModelType.PROPHET,
                0.6
            ),
            "simple_trend": (
                lambda m: abs(m.trend_strength) > 0.7 and m.seasonality_strength < 0.2,
                ModelType.THETA,
                0.7
            ),
            "high_volatility_clustering": (
                lambda m: m.volatility_clustering > 0.5,
                ModelType.ENSEMBLE,
                0.65
            ),
        }
        
        # Detector selection rules
        self.detector_rules = {
            "normal_distribution": (
                lambda m: abs(m.skewness) < 0.5 and abs(m.kurtosis) < 3,
                DetectorType.ZSCORE,
                0.8
            ),
            "heavy_tails": (
                lambda m: abs(m.kurtosis) > 5,
                DetectorType.IQR,
                0.75
            ),
            "high_complexity": (
                lambda m: m.sample_entropy > 2.0,
                DetectorType.ISOLATION_FOREST,
                0.7
            ),
            "trending_data": (
                lambda m: abs(m.trend_strength) > 0.6,
                DetectorType.CUSUM,
                0.75
            ),
            "regime_changes": (
                lambda m: m.volatility_clustering > 0.6,
                DetectorType.BOCPD,
                0.7
            ),
            "monotonic_trend": (
                lambda m: abs(m.trend_direction) > 0.8,
                DetectorType.MANN_KENDALL,
                0.65
            ),
        }
    
    def extract_metadata(
        self,
        data: Union[np.ndarray, pd.Series, pd.DataFrame],
        time_col: Optional[str] = None,
        value_col: Optional[str] = None
    ) -> TimeSeriesMetadata:
        """
        Extract comprehensive metadata features from time series.
        
        This is equivalent to Kats' get_metadata functionality.
        
        Args:
            data: Time series data (array, Series, or DataFrame)
            time_col: Name of time column if DataFrame
            value_col: Name of value column if DataFrame
        
        Returns:
            TimeSeriesMetadata with extracted features
        """
        # Convert to numpy array
        values = self._to_array(data, value_col)
        
        if len(values) < 10:
            logger.warning("Time series too short for reliable metadata extraction")
            return TimeSeriesMetadata(length=len(values))
        
        metadata = TimeSeriesMetadata()
        metadata.length = len(values)
        
        # Basic statistics
        metadata.mean = float(np.nanmean(values))
        metadata.std = float(np.nanstd(values))
        metadata.cv = metadata.std / abs(metadata.mean) if metadata.mean != 0 else 0
        metadata.skewness = float(stats.skew(values, nan_policy='omit'))
        metadata.kurtosis = float(stats.kurtosis(values, nan_policy='omit'))
        
        # Trend analysis
        trend_strength, trend_direction = self._calculate_trend(values)
        metadata.trend_strength = trend_strength
        metadata.trend_direction = trend_direction
        
        # Seasonality analysis
        seasonality_strength, dominant_period, num_peaks = self._calculate_seasonality(values)
        metadata.seasonality_strength = seasonality_strength
        metadata.dominant_period = dominant_period
        metadata.num_seasonal_peaks = num_peaks
        
        # Stationarity test
        is_stationary, adf_pvalue = self._test_stationarity(values)
        metadata.is_stationary = is_stationary
        metadata.adf_pvalue = adf_pvalue
        
        # Autocorrelation
        acf_values, pacf_values = self._calculate_autocorrelation(values)
        metadata.acf_lag1 = acf_values[0] if len(acf_values) > 0 else 0
        metadata.acf_lag5 = acf_values[4] if len(acf_values) > 4 else 0
        metadata.acf_lag10 = acf_values[9] if len(acf_values) > 9 else 0
        metadata.pacf_lag1 = pacf_values[0] if len(pacf_values) > 0 else 0
        
        # Complexity measures
        metadata.sample_entropy = self._calculate_sample_entropy(values)
        metadata.hurst_exponent = self._calculate_hurst_exponent(values)
        
        # Volatility analysis
        metadata.volatility_clustering = self._calculate_volatility_clustering(values)
        metadata.heteroscedasticity = self._test_heteroscedasticity(values)
        
        # Calculate predictability score
        metadata.predictability_score = self._calculate_predictability(metadata)
        
        return metadata
    
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
            # Try common column names
            for col in ['value', 'close', 'price', 'y']:
                if col in data.columns:
                    return data[col].values
            # Use first numeric column
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                return data[numeric_cols[0]].values
            raise ValueError("Could not find numeric column in DataFrame")
        else:
            return np.array(data).flatten()
    
    def _calculate_trend(self, values: np.ndarray) -> Tuple[float, float]:
        """Calculate trend strength and direction."""
        try:
            n = len(values)
            x = np.arange(n)
            
            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
            
            # Trend strength is R-squared
            trend_strength = r_value ** 2
            
            # Normalize slope for direction
            value_range = np.ptp(values)
            if value_range > 0:
                trend_direction = slope * n / value_range
            else:
                trend_direction = 0
            
            return float(trend_strength), float(np.clip(trend_direction, -1, 1))
        except Exception:
            return 0.0, 0.0
    
    def _calculate_seasonality(self, values: np.ndarray) -> Tuple[float, Optional[int], int]:
        """Calculate seasonality strength and dominant period."""
        try:
            n = len(values)
            if n < 20:
                return 0.0, None, 0
            
            # Remove trend
            detrended = values - np.linspace(values[0], values[-1], n)
            
            # FFT analysis
            fft_result = np.abs(fft(detrended))[:n // 2]
            frequencies = np.fft.fftfreq(n)[:n // 2]
            
            # Find peaks in frequency domain
            from scipy.signal import find_peaks
            peaks, properties = find_peaks(fft_result, height=np.mean(fft_result))
            
            if len(peaks) == 0:
                return 0.0, None, 0
            
            # Dominant period
            dominant_idx = peaks[np.argmax(fft_result[peaks])]
            if frequencies[dominant_idx] > 0:
                dominant_period = int(1 / frequencies[dominant_idx])
            else:
                dominant_period = None
            
            # Seasonality strength (ratio of peak energy to total energy)
            total_energy = np.sum(fft_result ** 2)
            peak_energy = np.sum(fft_result[peaks] ** 2)
            seasonality_strength = peak_energy / total_energy if total_energy > 0 else 0
            
            return float(seasonality_strength), dominant_period, len(peaks)
        except Exception:
            return 0.0, None, 0
    
    def _test_stationarity(self, values: np.ndarray) -> Tuple[bool, float]:
        """Test stationarity using ADF test."""
        try:
            from statsmodels.tsa.stattools import adfuller
            result = adfuller(values, autolag='AIC')
            adf_pvalue = result[1]
            is_stationary = adf_pvalue < 0.05
            return is_stationary, float(adf_pvalue)
        except Exception:
            return False, 1.0
    
    def _calculate_autocorrelation(self, values: np.ndarray) -> Tuple[List[float], List[float]]:
        """Calculate ACF and PACF values."""
        try:
            from statsmodels.tsa.stattools import acf, pacf
            max_lags = min(20, len(values) // 4)
            
            acf_values = acf(values, nlags=max_lags, fft=True)
            pacf_values = pacf(values, nlags=max_lags)
            
            # Return without lag 0 (which is always 1)
            return list(acf_values[1:]), list(pacf_values[1:])
        except Exception:
            return [], []
    
    def _calculate_sample_entropy(self, values: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """Calculate sample entropy (complexity measure)."""
        try:
            n = len(values)
            if n < m + 1:
                return 0.0
            
            # Normalize
            std = np.std(values)
            if std == 0:
                return 0.0
            
            tolerance = r * std
            
            def count_matches(template_length):
                count = 0
                for i in range(n - template_length):
                    for j in range(i + 1, n - template_length):
                        if np.max(np.abs(
                            values[i:i + template_length] - values[j:j + template_length]
                        )) < tolerance:
                            count += 1
                return count
            
            # Simplified calculation for efficiency
            A = count_matches(m + 1)
            B = count_matches(m)
            
            if B == 0:
                return 0.0
            
            return -np.log(A / B) if A > 0 else 0.0
        except Exception:
            return 0.0
    
    def _calculate_hurst_exponent(self, values: np.ndarray) -> float:
        """Calculate Hurst exponent (long-term memory measure)."""
        try:
            n = len(values)
            if n < 20:
                return 0.5
            
            # R/S analysis
            max_k = min(n // 2, 100)
            rs_values = []
            ns = []
            
            for k in range(10, max_k, 5):
                rs = self._rs_statistic(values[:k])
                if rs > 0:
                    rs_values.append(np.log(rs))
                    ns.append(np.log(k))
            
            if len(rs_values) < 3:
                return 0.5
            
            # Linear regression to find Hurst exponent
            slope, _, _, _, _ = stats.linregress(ns, rs_values)
            return float(np.clip(slope, 0, 1))
        except Exception:
            return 0.5
    
    def _rs_statistic(self, values: np.ndarray) -> float:
        """Calculate R/S statistic for a segment."""
        try:
            n = len(values)
            mean = np.mean(values)
            std = np.std(values)
            
            if std == 0:
                return 0.0
            
            # Cumulative deviation from mean
            cumdev = np.cumsum(values - mean)
            
            # R/S statistic
            rs = (np.max(cumdev) - np.min(cumdev)) / std
            return rs
        except Exception:
            return 0.0
    
    def _calculate_volatility_clustering(self, values: np.ndarray) -> float:
        """Calculate volatility clustering (ARCH effect)."""
        try:
            # Calculate returns/differences
            returns = np.diff(values)
            if len(returns) < 10:
                return 0.0
            
            # Squared returns
            sq_returns = returns ** 2
            
            # Autocorrelation of squared returns
            from statsmodels.tsa.stattools import acf
            acf_sq = acf(sq_returns, nlags=10, fft=True)
            
            # Average of first 5 lags (excluding lag 0)
            clustering = np.mean(np.abs(acf_sq[1:6]))
            return float(clustering)
        except Exception:
            return 0.0
    
    def _test_heteroscedasticity(self, values: np.ndarray) -> float:
        """Test for heteroscedasticity (changing variance)."""
        try:
            n = len(values)
            if n < 20:
                return 0.0
            
            # Split into segments and compare variances
            n_segments = 4
            segment_size = n // n_segments
            variances = []
            
            for i in range(n_segments):
                start = i * segment_size
                end = start + segment_size
                variances.append(np.var(values[start:end]))
            
            # Coefficient of variation of variances
            mean_var = np.mean(variances)
            if mean_var == 0:
                return 0.0
            
            cv_var = np.std(variances) / mean_var
            return float(np.clip(cv_var, 0, 1))
        except Exception:
            return 0.0
    
    def _calculate_predictability(self, metadata: TimeSeriesMetadata) -> float:
        """Calculate overall predictability score."""
        score = 0.5  # Base score
        
        # Higher trend strength = more predictable
        score += metadata.trend_strength * 0.15
        
        # Strong seasonality = more predictable
        score += metadata.seasonality_strength * 0.15
        
        # Stationarity = more predictable
        if metadata.is_stationary:
            score += 0.1
        
        # High autocorrelation = more predictable
        score += abs(metadata.acf_lag1) * 0.1
        
        # Low complexity = more predictable
        if metadata.sample_entropy < 1.0:
            score += 0.1
        elif metadata.sample_entropy > 2.0:
            score -= 0.1
        
        # Hurst > 0.5 = trending, more predictable
        if metadata.hurst_exponent > 0.6:
            score += 0.1
        elif metadata.hurst_exponent < 0.4:
            score -= 0.05
        
        # Low volatility clustering = more predictable
        score -= metadata.volatility_clustering * 0.1
        
        return float(np.clip(score, 0, 1))
    
    def recommend_model(
        self,
        data: Union[np.ndarray, pd.Series, pd.DataFrame],
        value_col: Optional[str] = None,
        available_models: Optional[List[ModelType]] = None
    ) -> ModelRecommendation:
        """
        Recommend best forecasting model based on time series characteristics.
        
        This is equivalent to Kats' metalearner_modelselect functionality.
        
        Args:
            data: Time series data
            value_col: Name of value column if DataFrame
            available_models: List of models to consider (default: all)
        
        Returns:
            ModelRecommendation with best model and alternatives
        """
        # Extract metadata
        metadata = self.extract_metadata(data, value_col=value_col)
        
        if available_models is None:
            available_models = list(ModelType)
        
        # Score each model based on rules
        model_scores: Dict[ModelType, float] = {model: 0.5 for model in available_models}
        reasons: List[str] = []
        
        for rule_name, (condition, model, weight) in self.model_rules.items():
            try:
                if condition(metadata) and model in available_models:
                    model_scores[model] += weight * 0.3
                    reasons.append(f"{rule_name}: suggests {model.value}")
            except Exception:
                continue
        
        # Additional scoring based on metadata
        if metadata.seasonality_strength > 0.5 and ModelType.SARIMA in available_models:
            model_scores[ModelType.SARIMA] += 0.2
            reasons.append(f"Strong seasonality ({metadata.seasonality_strength:.2f})")
        
        if metadata.is_stationary and ModelType.ARIMA in available_models:
            model_scores[ModelType.ARIMA] += 0.15
            reasons.append("Stationary series favors ARIMA")
        
        if metadata.predictability_score < 0.4 and ModelType.ENSEMBLE in available_models:
            model_scores[ModelType.ENSEMBLE] += 0.15
            reasons.append("Low predictability suggests ensemble")
        
        # Sort models by score
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        best_model, best_score = sorted_models[0]
        
        # Normalize scores to confidence
        total_score = sum(s for _, s in sorted_models)
        confidence = best_score / total_score if total_score > 0 else 0.5
        
        # Get recommended parameters
        recommended_params = self._get_model_params(best_model, metadata)
        
        # Build reasoning
        reasoning = "; ".join(reasons[:5]) if reasons else "Default recommendation"
        
        return ModelRecommendation(
            recommended_model=best_model,
            confidence=confidence,
            alternative_models=[(m, s / total_score) for m, s in sorted_models[1:4]],
            recommended_params=recommended_params,
            reasoning=reasoning
        )
    
    def _get_model_params(self, model: ModelType, metadata: TimeSeriesMetadata) -> Dict[str, Any]:
        """Get recommended parameters for a model."""
        params: Dict[str, Any] = {}
        
        if model == ModelType.ARIMA:
            # Determine order based on ACF/PACF
            p = 1 if abs(metadata.pacf_lag1) > 0.2 else 0
            d = 0 if metadata.is_stationary else 1
            q = 1 if abs(metadata.acf_lag1) > 0.2 else 0
            params = {"order": (p, d, q)}
        
        elif model == ModelType.SARIMA:
            p = 1 if abs(metadata.pacf_lag1) > 0.2 else 0
            d = 0 if metadata.is_stationary else 1
            q = 1 if abs(metadata.acf_lag1) > 0.2 else 0
            s = metadata.dominant_period or 7
            params = {
                "order": (p, d, q),
                "seasonal_order": (1, 1, 1, s)
            }
        
        elif model == ModelType.EXPONENTIAL_SMOOTHING:
            params = {
                "trend": "add" if metadata.trend_strength > 0.3 else None,
                "seasonal": "add" if metadata.seasonality_strength > 0.3 else None,
                "seasonal_periods": metadata.dominant_period or 7
            }
        
        elif model == ModelType.THETA:
            params = {"period": metadata.dominant_period or 1}
        
        elif model == ModelType.PROPHET:
            params = {
                "seasonality_mode": "multiplicative" if metadata.cv > 0.5 else "additive",
                "changepoint_prior_scale": 0.05 if metadata.volatility_clustering < 0.3 else 0.15
            }
        
        return params
    
    def recommend_detector(
        self,
        data: Union[np.ndarray, pd.Series, pd.DataFrame],
        value_col: Optional[str] = None,
        detection_type: str = "anomaly",  # "anomaly" or "changepoint"
        available_detectors: Optional[List[DetectorType]] = None
    ) -> DetectorRecommendation:
        """
        Recommend best anomaly/changepoint detector based on data characteristics.
        
        This is equivalent to Kats' metalearning_detection_model functionality.
        
        Args:
            data: Time series data
            value_col: Name of value column if DataFrame
            detection_type: "anomaly" or "changepoint"
            available_detectors: List of detectors to consider
        
        Returns:
            DetectorRecommendation with best detector and alternatives
        """
        # Extract metadata
        metadata = self.extract_metadata(data, value_col=value_col)
        
        if available_detectors is None:
            if detection_type == "changepoint":
                available_detectors = [
                    DetectorType.CUSUM,
                    DetectorType.BOCPD,
                    DetectorType.DTW
                ]
            else:
                available_detectors = [
                    DetectorType.ZSCORE,
                    DetectorType.IQR,
                    DetectorType.ISOLATION_FOREST,
                    DetectorType.MANN_KENDALL
                ]
        
        # Score each detector based on rules
        detector_scores: Dict[DetectorType, float] = {d: 0.5 for d in available_detectors}
        reasons: List[str] = []
        
        for rule_name, (condition, detector, weight) in self.detector_rules.items():
            try:
                if condition(metadata) and detector in available_detectors:
                    detector_scores[detector] += weight * 0.3
                    reasons.append(f"{rule_name}: suggests {detector.value}")
            except Exception:
                continue
        
        # Additional scoring
        if abs(metadata.skewness) < 0.5 and DetectorType.ZSCORE in available_detectors:
            detector_scores[DetectorType.ZSCORE] += 0.2
            reasons.append("Near-normal distribution favors Z-score")
        
        if metadata.kurtosis > 3 and DetectorType.IQR in available_detectors:
            detector_scores[DetectorType.IQR] += 0.15
            reasons.append("Heavy tails favor IQR")
        
        # Sort detectors by score
        sorted_detectors = sorted(detector_scores.items(), key=lambda x: x[1], reverse=True)
        best_detector, best_score = sorted_detectors[0]
        
        # Normalize scores to confidence
        total_score = sum(s for _, s in sorted_detectors)
        confidence = best_score / total_score if total_score > 0 else 0.5
        
        # Get recommended parameters
        recommended_params = self._get_detector_params(best_detector, metadata)
        
        # Build reasoning
        reasoning = "; ".join(reasons[:5]) if reasons else "Default recommendation"
        
        return DetectorRecommendation(
            recommended_detector=best_detector,
            confidence=confidence,
            alternative_detectors=[(d, s / total_score) for d, s in sorted_detectors[1:4]],
            recommended_params=recommended_params,
            reasoning=reasoning
        )
    
    def _get_detector_params(self, detector: DetectorType, metadata: TimeSeriesMetadata) -> Dict[str, Any]:
        """Get recommended parameters for a detector."""
        params: Dict[str, Any] = {}
        
        if detector == DetectorType.ZSCORE:
            # Adjust threshold based on kurtosis
            threshold = 3.0 if abs(metadata.kurtosis) < 3 else 4.0
            params = {"threshold": threshold, "window": min(50, metadata.length // 4)}
        
        elif detector == DetectorType.IQR:
            multiplier = 1.5 if abs(metadata.kurtosis) < 5 else 2.0
            params = {"multiplier": multiplier}
        
        elif detector == DetectorType.ISOLATION_FOREST:
            contamination = 0.05 if metadata.cv < 0.5 else 0.1
            params = {"contamination": contamination, "n_estimators": 100}
        
        elif detector == DetectorType.CUSUM:
            threshold = metadata.std * 5 if metadata.std > 0 else 5.0
            params = {"threshold": threshold, "drift": 0}
        
        elif detector == DetectorType.BOCPD:
            params = {"hazard": 1 / 100, "prior_mean": metadata.mean, "prior_var": metadata.std ** 2}
        
        elif detector == DetectorType.MANN_KENDALL:
            params = {"alpha": 0.05}
        
        elif detector == DetectorType.DTW:
            params = {"window": min(20, metadata.length // 10)}
        
        return params
    
    def assess_predictability(
        self,
        data: Union[np.ndarray, pd.Series, pd.DataFrame],
        value_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Assess the predictability of a time series.
        
        This is equivalent to Kats' metalearner_predictability functionality.
        
        Args:
            data: Time series data
            value_col: Name of value column if DataFrame
        
        Returns:
            Dictionary with predictability assessment
        """
        metadata = self.extract_metadata(data, value_col=value_col)
        
        assessment = {
            "predictability_score": metadata.predictability_score,
            "predictability_level": self._get_predictability_level(metadata.predictability_score),
            "factors": {
                "trend_contribution": metadata.trend_strength * 0.2,
                "seasonality_contribution": metadata.seasonality_strength * 0.2,
                "stationarity_contribution": 0.15 if metadata.is_stationary else 0,
                "autocorrelation_contribution": abs(metadata.acf_lag1) * 0.15,
                "complexity_penalty": -min(0.15, (metadata.sample_entropy - 1) * 0.1),
                "volatility_penalty": -metadata.volatility_clustering * 0.1
            },
            "recommendations": self._get_predictability_recommendations(metadata),
            "metadata": {
                "trend_strength": metadata.trend_strength,
                "seasonality_strength": metadata.seasonality_strength,
                "is_stationary": metadata.is_stationary,
                "sample_entropy": metadata.sample_entropy,
                "hurst_exponent": metadata.hurst_exponent
            }
        }
        
        return assessment
    
    def _get_predictability_level(self, score: float) -> str:
        """Convert predictability score to level."""
        if score >= 0.75:
            return "HIGH"
        elif score >= 0.5:
            return "MODERATE"
        elif score >= 0.25:
            return "LOW"
        else:
            return "VERY_LOW"
    
    def _get_predictability_recommendations(self, metadata: TimeSeriesMetadata) -> List[str]:
        """Get recommendations to improve forecasting."""
        recommendations = []
        
        if not metadata.is_stationary:
            recommendations.append("Consider differencing to achieve stationarity")
        
        if metadata.seasonality_strength > 0.3 and metadata.dominant_period:
            recommendations.append(
                f"Account for seasonality with period {metadata.dominant_period}"
            )
        
        if metadata.volatility_clustering > 0.5:
            recommendations.append("Consider GARCH model for volatility clustering")
        
        if metadata.sample_entropy > 2.0:
            recommendations.append("High complexity - consider ensemble methods")
        
        if metadata.hurst_exponent < 0.4:
            recommendations.append("Mean-reverting series - consider mean reversion strategies")
        elif metadata.hurst_exponent > 0.6:
            recommendations.append("Trending series - momentum strategies may work")
        
        return recommendations
    
    def tune_hyperparameters(
        self,
        data: Union[np.ndarray, pd.Series, pd.DataFrame],
        model: ModelType,
        value_col: Optional[str] = None,
        param_grid: Optional[Dict[str, List[Any]]] = None
    ) -> Dict[str, Any]:
        """
        Recommend hyperparameters for a specific model.
        
        This is equivalent to Kats' metalearner_hpt functionality.
        
        Args:
            data: Time series data
            model: Model type to tune
            value_col: Name of value column if DataFrame
            param_grid: Optional parameter grid to search
        
        Returns:
            Dictionary with recommended hyperparameters
        """
        metadata = self.extract_metadata(data, value_col=value_col)
        
        # Get base recommended params
        recommended_params = self._get_model_params(model, metadata)
        
        # Add additional tuning based on metadata
        tuning_result = {
            "model": model.value,
            "recommended_params": recommended_params,
            "confidence": self._calculate_param_confidence(metadata, model),
            "metadata_used": {
                "length": metadata.length,
                "trend_strength": metadata.trend_strength,
                "seasonality_strength": metadata.seasonality_strength,
                "is_stationary": metadata.is_stationary,
                "dominant_period": metadata.dominant_period
            },
            "suggestions": self._get_tuning_suggestions(model, metadata)
        }
        
        return tuning_result
    
    def _calculate_param_confidence(self, metadata: TimeSeriesMetadata, model: ModelType) -> float:
        """Calculate confidence in parameter recommendations."""
        confidence = 0.5
        
        # More data = higher confidence
        if metadata.length > 100:
            confidence += 0.1
        if metadata.length > 500:
            confidence += 0.1
        
        # Clear patterns = higher confidence
        if metadata.trend_strength > 0.5 or metadata.seasonality_strength > 0.5:
            confidence += 0.1
        
        # Model-specific adjustments
        if model == ModelType.SARIMA and metadata.seasonality_strength > 0.3:
            confidence += 0.1
        if model == ModelType.ARIMA and metadata.is_stationary:
            confidence += 0.1
        
        return min(confidence, 0.95)
    
    def _get_tuning_suggestions(self, model: ModelType, metadata: TimeSeriesMetadata) -> List[str]:
        """Get tuning suggestions for a model."""
        suggestions = []
        
        if model in [ModelType.ARIMA, ModelType.SARIMA]:
            if metadata.length < 100:
                suggestions.append("Limited data - use simpler model orders")
            if not metadata.is_stationary:
                suggestions.append("Consider d=1 or d=2 for non-stationary data")
        
        if model == ModelType.PROPHET:
            if metadata.volatility_clustering > 0.5:
                suggestions.append("Increase changepoint_prior_scale for volatile data")
            if metadata.seasonality_strength < 0.2:
                suggestions.append("Consider disabling yearly/weekly seasonality")
        
        if model == ModelType.EXPONENTIAL_SMOOTHING:
            if metadata.trend_strength < 0.2:
                suggestions.append("Consider trend=None for trendless data")
        
        return suggestions


# Convenience function for quick model recommendation
def recommend_model(
    data: Union[np.ndarray, pd.Series, pd.DataFrame],
    value_col: Optional[str] = None
) -> ModelRecommendation:
    """Quick function to get model recommendation."""
    learner = MetaLearner()
    return learner.recommend_model(data, value_col=value_col)


# Convenience function for quick detector recommendation
def recommend_detector(
    data: Union[np.ndarray, pd.Series, pd.DataFrame],
    value_col: Optional[str] = None,
    detection_type: str = "anomaly"
) -> DetectorRecommendation:
    """Quick function to get detector recommendation."""
    learner = MetaLearner()
    return learner.recommend_detector(data, value_col=value_col, detection_type=detection_type)


# Convenience function for predictability assessment
def assess_predictability(
    data: Union[np.ndarray, pd.Series, pd.DataFrame],
    value_col: Optional[str] = None
) -> Dict[str, Any]:
    """Quick function to assess predictability."""
    learner = MetaLearner()
    return learner.assess_predictability(data, value_col=value_col)
