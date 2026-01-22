"""
Advanced Time Series Detectors.

This module provides advanced anomaly and changepoint detection algorithms
similar to Kats:
- BOCPD (Bayesian Online Changepoint Detection)
- Mann-Kendall Trend Test
- DTW (Dynamic Time Warping) Distance for pattern matching
- Robust Z-Score Detector
- Seasonal Hybrid ESD (Twitter's algorithm)

These complement the basic detectors in TimeSeriesEngine.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from scipy import stats
from scipy.special import gammaln
import logging

logger = logging.getLogger(__name__)


@dataclass
class ChangePoint:
    """Represents a detected changepoint."""
    index: int  # Index in the time series
    timestamp: Optional[Any] = None  # Associated timestamp if available
    confidence: float = 0.0  # Confidence score [0, 1]
    severity: float = 0.0  # Magnitude of change
    direction: str = "unknown"  # "increase", "decrease", "unknown"
    cp_type: str = "level_shift"  # "level_shift", "trend_change", "variance_change"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class TrendResult:
    """Result from Mann-Kendall trend test."""
    trend: str  # "increasing", "decreasing", "no_trend"
    h: bool  # Reject null hypothesis (trend exists)
    p_value: float  # P-value
    z_score: float  # Z statistic
    tau: float  # Kendall tau
    s: float  # Mann-Kendall S statistic
    slope: float  # Sen's slope
    intercept: float  # Intercept
    confidence_interval: Tuple[float, float]  # 95% CI for slope


@dataclass
class DTWResult:
    """Result from DTW distance calculation."""
    distance: float  # DTW distance
    path: List[Tuple[int, int]]  # Alignment path
    normalized_distance: float  # Normalized by path length
    is_similar: bool  # Whether patterns are similar (below threshold)


class BOCPD:
    """
    Bayesian Online Changepoint Detection.
    
    Implements the algorithm from Adams & MacKay (2007).
    Detects changepoints in real-time by computing the probability
    distribution over run lengths.
    
    Suitable for:
    - Online/streaming changepoint detection
    - Multiple changepoint detection
    - Level shifts and distribution changes
    
    Example:
        detector = BOCPD(hazard_rate=1/100)
        changepoints = detector.detect(data)
    """
    
    def __init__(
        self,
        hazard_rate: float = 0.01,
        model: str = "gaussian",
        threshold: float = 0.5,
        min_run_length: int = 10
    ):
        """
        Initialize BOCPD detector.
        
        Args:
            hazard_rate: Prior probability of changepoint (lambda / timescale)
            model: Likelihood model ("gaussian", "student_t")
            threshold: Probability threshold for declaring changepoint
            min_run_length: Minimum run length before allowing changepoint
        """
        self.hazard_rate = hazard_rate
        self.model = model
        self.threshold = threshold
        self.min_run_length = min_run_length
        
        # Prior parameters (Normal-Inverse-Gamma for Gaussian model)
        self.mu0 = 0.0
        self.kappa0 = 1.0
        self.alpha0 = 1.0
        self.beta0 = 1.0
    
    def detect(
        self,
        data: Union[np.ndarray, pd.Series],
        return_probabilities: bool = False
    ) -> Union[List[ChangePoint], Tuple[List[ChangePoint], np.ndarray]]:
        """
        Detect changepoints in the data.
        
        Args:
            data: Time series data
            return_probabilities: If True, also return run length probabilities
        
        Returns:
            List of ChangePoint objects (and optionally run length matrix)
        """
        if isinstance(data, pd.Series):
            data = data.values
        data = np.array(data).flatten()
        n = len(data)
        
        if n < self.min_run_length * 2:
            return [] if not return_probabilities else ([], np.array([]))
        
        # Initialize run length probability matrix
        # R[t, r] = P(run_length = r at time t)
        R = np.zeros((n + 1, n + 1))
        R[0, 0] = 1.0
        
        # Store message passing (sufficient statistics)
        mu_params = np.zeros((n + 1, n + 1))
        kappa_params = np.zeros((n + 1, n + 1)) + self.kappa0
        alpha_params = np.zeros((n + 1, n + 1)) + self.alpha0
        beta_params = np.zeros((n + 1, n + 1)) + self.beta0
        
        # Initialize
        mu_params[0, 0] = self.mu0
        
        # Changepoint probabilities
        cp_probs = np.zeros(n)
        
        for t in range(n):
            x = data[t]
            
            # Evaluate predictive probabilities
            pred_probs = self._predictive_prob(
                x, mu_params[t, :t+1], kappa_params[t, :t+1],
                alpha_params[t, :t+1], beta_params[t, :t+1]
            )
            
            # Hazard function (constant hazard)
            H = self.hazard_rate
            
            # Growth probabilities (no changepoint)
            growth = R[t, :t+1] * pred_probs * (1 - H)
            
            # Changepoint probability (reset to run length 0)
            cp = np.sum(R[t, :t+1] * pred_probs * H)
            
            # Update run length distribution
            R[t+1, 1:t+2] = growth
            R[t+1, 0] = cp
            
            # Normalize
            R[t+1, :t+2] /= np.sum(R[t+1, :t+2]) + 1e-10
            
            # Record changepoint probability
            cp_probs[t] = R[t+1, 0]
            
            # Update sufficient statistics
            for r in range(t + 2):
                if r == 0:
                    # Reset statistics
                    mu_params[t+1, 0] = self.mu0
                    kappa_params[t+1, 0] = self.kappa0
                    alpha_params[t+1, 0] = self.alpha0
                    beta_params[t+1, 0] = self.beta0
                else:
                    # Update statistics
                    kappa_new = kappa_params[t, r-1] + 1
                    mu_new = (kappa_params[t, r-1] * mu_params[t, r-1] + x) / kappa_new
                    alpha_new = alpha_params[t, r-1] + 0.5
                    beta_new = (
                        beta_params[t, r-1] + 
                        0.5 * kappa_params[t, r-1] * (x - mu_params[t, r-1])**2 / kappa_new
                    )
                    
                    mu_params[t+1, r] = mu_new
                    kappa_params[t+1, r] = kappa_new
                    alpha_params[t+1, r] = alpha_new
                    beta_params[t+1, r] = beta_new
        
        # Find changepoints
        changepoints = self._extract_changepoints(data, cp_probs)
        
        if return_probabilities:
            return changepoints, cp_probs
        return changepoints
    
    def _predictive_prob(
        self,
        x: float,
        mu: np.ndarray,
        kappa: np.ndarray,
        alpha: np.ndarray,
        beta: np.ndarray
    ) -> np.ndarray:
        """Compute predictive probabilities under Student-t distribution."""
        # Student-t parameters
        df = 2 * alpha
        mu_pred = mu
        var = beta * (kappa + 1) / (alpha * kappa)
        
        # Avoid numerical issues
        var = np.maximum(var, 1e-10)
        
        # Student-t log PDF
        log_prob = (
            gammaln((df + 1) / 2) - gammaln(df / 2) -
            0.5 * np.log(np.pi * df * var) -
            ((df + 1) / 2) * np.log(1 + (x - mu_pred)**2 / (df * var))
        )
        
        return np.exp(log_prob)
    
    def _extract_changepoints(
        self,
        data: np.ndarray,
        cp_probs: np.ndarray
    ) -> List[ChangePoint]:
        """Extract changepoints from probability sequence."""
        changepoints = []
        n = len(data)
        
        # Find peaks above threshold
        for i in range(self.min_run_length, n - self.min_run_length):
            if cp_probs[i] > self.threshold:
                # Check if it's a local maximum
                window = cp_probs[max(0, i-5):min(n, i+6)]
                if cp_probs[i] >= np.max(window):
                    # Determine direction
                    before = np.mean(data[max(0, i-self.min_run_length):i])
                    after = np.mean(data[i:min(n, i+self.min_run_length)])
                    
                    direction = "increase" if after > before else "decrease"
                    severity = abs(after - before) / (np.std(data) + 1e-10)
                    
                    changepoints.append(ChangePoint(
                        index=i,
                        confidence=float(cp_probs[i]),
                        severity=float(severity),
                        direction=direction,
                        cp_type="level_shift"
                    ))
        
        return changepoints


class MannKendall:
    """
    Mann-Kendall Trend Test.
    
    Non-parametric test for detecting monotonic trends in time series.
    Includes Sen's slope estimator for trend magnitude.
    
    Variants:
    - Original Mann-Kendall
    - Seasonal Mann-Kendall (for periodic data)
    - Modified MK (accounts for autocorrelation)
    
    Example:
        mk = MannKendall()
        result = mk.test(data)
        print(f"Trend: {result.trend}, p-value: {result.p_value:.4f}")
    """
    
    def __init__(
        self,
        alpha: float = 0.05,
        method: str = "original"
    ):
        """
        Initialize Mann-Kendall test.
        
        Args:
            alpha: Significance level
            method: "original", "seasonal", or "modified"
        """
        self.alpha = alpha
        self.method = method
    
    def test(
        self,
        data: Union[np.ndarray, pd.Series],
        period: Optional[int] = None
    ) -> TrendResult:
        """
        Perform Mann-Kendall trend test.
        
        Args:
            data: Time series data
            period: Seasonal period (required for seasonal method)
        
        Returns:
            TrendResult with test statistics
        """
        if isinstance(data, pd.Series):
            data = data.values
        data = np.array(data).flatten()
        n = len(data)
        
        if n < 4:
            return TrendResult(
                trend="no_trend", h=False, p_value=1.0,
                z_score=0.0, tau=0.0, s=0.0,
                slope=0.0, intercept=0.0,
                confidence_interval=(0.0, 0.0)
            )
        
        if self.method == "seasonal" and period:
            return self._seasonal_mk(data, period)
        elif self.method == "modified":
            return self._modified_mk(data)
        else:
            return self._original_mk(data)
    
    def _original_mk(self, data: np.ndarray) -> TrendResult:
        """Original Mann-Kendall test."""
        n = len(data)
        
        # Calculate S statistic
        s = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                s += np.sign(data[j] - data[i])
        
        # Calculate variance of S
        # Account for ties
        unique, counts = np.unique(data, return_counts=True)
        tie_correction = np.sum(counts * (counts - 1) * (2 * counts + 5))
        
        var_s = (n * (n - 1) * (2 * n + 5) - tie_correction) / 18
        
        # Calculate Z score
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0
        
        # Calculate p-value (two-sided)
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        # Kendall's tau
        tau = 2 * s / (n * (n - 1))
        
        # Sen's slope
        slopes = []
        for i in range(n - 1):
            for j in range(i + 1, n):
                if data[j] != data[i]:
                    slopes.append((data[j] - data[i]) / (j - i))
        
        slope = np.median(slopes) if slopes else 0.0
        
        # Intercept (median of y - slope * x)
        x = np.arange(n)
        intercept = np.median(data - slope * x)
        
        # Confidence interval for slope
        ci = self._slope_confidence_interval(slopes, var_s)
        
        # Determine trend
        h = p_value < self.alpha
        if h:
            trend = "increasing" if z > 0 else "decreasing"
        else:
            trend = "no_trend"
        
        return TrendResult(
            trend=trend,
            h=h,
            p_value=p_value,
            z_score=z,
            tau=tau,
            s=s,
            slope=slope,
            intercept=intercept,
            confidence_interval=ci
        )
    
    def _seasonal_mk(self, data: np.ndarray, period: int) -> TrendResult:
        """Seasonal Mann-Kendall test."""
        n = len(data)
        n_seasons = n // period
        
        if n_seasons < 2:
            return self._original_mk(data)
        
        # Calculate S for each season
        s_total = 0
        var_s_total = 0
        all_slopes = []
        
        for season in range(period):
            # Extract seasonal subseries
            seasonal_data = data[season::period]
            m = len(seasonal_data)
            
            if m < 2:
                continue
            
            # S for this season
            s = 0
            for i in range(m - 1):
                for j in range(i + 1, m):
                    s += np.sign(seasonal_data[j] - seasonal_data[i])
            s_total += s
            
            # Variance for this season
            var_s_total += m * (m - 1) * (2 * m + 5) / 18
            
            # Slopes
            for i in range(m - 1):
                for j in range(i + 1, m):
                    if seasonal_data[j] != seasonal_data[i]:
                        all_slopes.append(
                            (seasonal_data[j] - seasonal_data[i]) / ((j - i) * period)
                        )
        
        # Z score
        if s_total > 0:
            z = (s_total - 1) / np.sqrt(var_s_total)
        elif s_total < 0:
            z = (s_total + 1) / np.sqrt(var_s_total)
        else:
            z = 0
        
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        # Slope and intercept
        slope = np.median(all_slopes) if all_slopes else 0.0
        x = np.arange(n)
        intercept = np.median(data - slope * x)
        
        # Result
        h = p_value < self.alpha
        if h:
            trend = "increasing" if z > 0 else "decreasing"
        else:
            trend = "no_trend"
        
        tau = s_total / (period * n_seasons * (n_seasons - 1) / 2)
        
        return TrendResult(
            trend=trend, h=h, p_value=p_value, z_score=z,
            tau=tau, s=s_total, slope=slope, intercept=intercept,
            confidence_interval=(slope * 0.9, slope * 1.1)  # Approximate
        )
    
    def _modified_mk(self, data: np.ndarray) -> TrendResult:
        """Modified Mann-Kendall (corrects for autocorrelation)."""
        n = len(data)
        
        # First get original results
        original = self._original_mk(data)
        
        # Calculate autocorrelation correction
        ranks = stats.rankdata(data)
        
        # Calculate autocorrelations
        n_s_ratio = 1.0
        for lag in range(1, min(10, n // 4)):
            acf = np.corrcoef(ranks[:-lag], ranks[lag:])[0, 1]
            if np.isfinite(acf):
                n_s_ratio += 2 * (n - lag) * (n - lag - 1) * (n - lag - 2) * acf / (n * (n - 1) * (n - 2))
        
        n_s_ratio = max(n_s_ratio, 1.0)  # Ensure positive
        
        # Adjust variance
        var_s_original = (n * (n - 1) * (2 * n + 5)) / 18
        var_s_corrected = var_s_original / n_s_ratio
        
        # Recalculate z
        s = original.s
        if s > 0:
            z = (s - 1) / np.sqrt(var_s_corrected)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s_corrected)
        else:
            z = 0
        
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        h = p_value < self.alpha
        if h:
            trend = "increasing" if z > 0 else "decreasing"
        else:
            trend = "no_trend"
        
        return TrendResult(
            trend=trend, h=h, p_value=p_value, z_score=z,
            tau=original.tau, s=original.s, slope=original.slope,
            intercept=original.intercept,
            confidence_interval=original.confidence_interval
        )
    
    def _slope_confidence_interval(
        self,
        slopes: List[float],
        var_s: float,
        alpha: float = 0.05
    ) -> Tuple[float, float]:
        """Calculate confidence interval for Sen's slope."""
        if not slopes:
            return (0.0, 0.0)
        
        slopes = sorted(slopes)
        n_slopes = len(slopes)
        
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        c = z_alpha * np.sqrt(var_s)
        
        lower_idx = int((n_slopes - c) / 2)
        upper_idx = int((n_slopes + c) / 2)
        
        lower_idx = max(0, min(lower_idx, n_slopes - 1))
        upper_idx = max(0, min(upper_idx, n_slopes - 1))
        
        return (slopes[lower_idx], slopes[upper_idx])


class DTWDistance:
    """
    Dynamic Time Warping Distance Calculator.
    
    DTW measures similarity between two time series that may vary
    in speed/timing. Useful for:
    - Pattern matching
    - Similarity search
    - Changepoint detection via pattern comparison
    
    Example:
        dtw = DTWDistance()
        result = dtw.compute(series1, series2)
        print(f"DTW distance: {result.distance}")
    """
    
    def __init__(
        self,
        window: Optional[int] = None,
        step_pattern: str = "symmetric2",
        normalize: bool = True
    ):
        """
        Initialize DTW calculator.
        
        Args:
            window: Sakoe-Chiba band width (None = no constraint)
            step_pattern: "symmetric1", "symmetric2", or "asymmetric"
            normalize: Whether to normalize by path length
        """
        self.window = window
        self.step_pattern = step_pattern
        self.normalize = normalize
    
    def compute(
        self,
        s1: Union[np.ndarray, pd.Series],
        s2: Union[np.ndarray, pd.Series],
        return_path: bool = True
    ) -> DTWResult:
        """
        Compute DTW distance between two series.
        
        Args:
            s1: First time series
            s2: Second time series
            return_path: Whether to compute alignment path
        
        Returns:
            DTWResult with distance and path
        """
        if isinstance(s1, pd.Series):
            s1 = s1.values
        if isinstance(s2, pd.Series):
            s2 = s2.values
        
        s1 = np.array(s1).flatten()
        s2 = np.array(s2).flatten()
        
        n, m = len(s1), len(s2)
        
        # Initialize cost matrix
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0
        
        # Compute window constraints
        if self.window:
            window = max(self.window, abs(n - m))
        else:
            window = max(n, m)
        
        # Fill cost matrix
        for i in range(1, n + 1):
            j_start = max(1, i - window)
            j_end = min(m + 1, i + window + 1)
            
            for j in range(j_start, j_end):
                cost = abs(s1[i-1] - s2[j-1])
                
                if self.step_pattern == "symmetric2":
                    dtw_matrix[i, j] = cost + min(
                        dtw_matrix[i-1, j],      # insertion
                        dtw_matrix[i, j-1],      # deletion
                        dtw_matrix[i-1, j-1]     # match
                    )
                elif self.step_pattern == "asymmetric":
                    dtw_matrix[i, j] = cost + min(
                        dtw_matrix[i-1, j],
                        dtw_matrix[i-1, j-1]
                    )
                else:  # symmetric1
                    dtw_matrix[i, j] = cost + min(
                        dtw_matrix[i-1, j] + cost,
                        dtw_matrix[i, j-1] + cost,
                        dtw_matrix[i-1, j-1] + cost
                    )
        
        distance = dtw_matrix[n, m]
        
        # Backtrack to find path
        path = []
        if return_path:
            path = self._backtrack(dtw_matrix, n, m)
        
        # Normalize
        path_length = len(path) if path else n + m
        normalized_distance = distance / path_length if self.normalize else distance
        
        # Threshold for similarity (heuristic)
        threshold = np.std(np.concatenate([s1, s2])) * 2
        is_similar = normalized_distance < threshold
        
        return DTWResult(
            distance=distance,
            path=path,
            normalized_distance=normalized_distance,
            is_similar=is_similar
        )
    
    def _backtrack(
        self,
        dtw_matrix: np.ndarray,
        n: int,
        m: int
    ) -> List[Tuple[int, int]]:
        """Backtrack to find alignment path."""
        path = [(n-1, m-1)]
        i, j = n, m
        
        while i > 1 or j > 1:
            if i == 1:
                j -= 1
            elif j == 1:
                i -= 1
            else:
                candidates = [
                    (dtw_matrix[i-1, j], (i-1, j)),
                    (dtw_matrix[i, j-1], (i, j-1)),
                    (dtw_matrix[i-1, j-1], (i-1, j-1))
                ]
                _, (i, j) = min(candidates, key=lambda x: x[0])
            
            path.append((i-1, j-1))
        
        return list(reversed(path))
    
    def find_similar_patterns(
        self,
        query: np.ndarray,
        database: np.ndarray,
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Find similar patterns in a longer time series.
        
        Args:
            query: Query pattern
            database: Time series to search in
            top_k: Number of matches to return
        
        Returns:
            List of (start_index, distance) tuples
        """
        query_len = len(query)
        db_len = len(database)
        
        if query_len > db_len:
            return []
        
        distances = []
        
        for i in range(db_len - query_len + 1):
            window = database[i:i + query_len]
            result = self.compute(query, window, return_path=False)
            distances.append((i, result.normalized_distance))
        
        # Sort by distance
        distances.sort(key=lambda x: x[1])
        
        return distances[:top_k]


class RobustZScore:
    """
    Robust Z-Score Anomaly Detector.
    
    Uses median and MAD (Median Absolute Deviation) instead of
    mean and standard deviation for robustness to outliers.
    
    Example:
        detector = RobustZScore(threshold=3.5)
        anomalies = detector.detect(data)
    """
    
    def __init__(
        self,
        threshold: float = 3.5,
        window_size: Optional[int] = None
    ):
        """
        Initialize detector.
        
        Args:
            threshold: Z-score threshold for anomaly
            window_size: Rolling window size (None = use all data)
        """
        self.threshold = threshold
        self.window_size = window_size
    
    def detect(
        self,
        data: Union[np.ndarray, pd.Series]
    ) -> List[int]:
        """
        Detect anomalies.
        
        Args:
            data: Time series data
        
        Returns:
            List of anomaly indices
        """
        if isinstance(data, pd.Series):
            data = data.values
        data = np.array(data).flatten()
        
        if self.window_size:
            return self._rolling_detect(data)
        else:
            return self._global_detect(data)
    
    def _global_detect(self, data: np.ndarray) -> List[int]:
        """Detect anomalies using global statistics."""
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        
        # MAD to std conversion factor
        mad_std = mad * 1.4826
        
        if mad_std == 0:
            return []
        
        z_scores = np.abs((data - median) / mad_std)
        return list(np.where(z_scores > self.threshold)[0])
    
    def _rolling_detect(self, data: np.ndarray) -> List[int]:
        """Detect anomalies using rolling window."""
        anomalies = []
        n = len(data)
        
        for i in range(n):
            start = max(0, i - self.window_size)
            window = data[start:i+1]
            
            if len(window) < 3:
                continue
            
            median = np.median(window)
            mad = np.median(np.abs(window - median))
            mad_std = mad * 1.4826
            
            if mad_std > 0:
                z_score = abs((data[i] - median) / mad_std)
                if z_score > self.threshold:
                    anomalies.append(i)
        
        return anomalies


class SeasonalESD:
    """
    Seasonal Hybrid ESD (Extreme Studentized Deviate).
    
    Based on Twitter's anomaly detection algorithm.
    Handles seasonal patterns in data.
    
    Example:
        detector = SeasonalESD(seasonality=24, max_anomalies=0.1)
        anomalies = detector.detect(hourly_data)
    """
    
    def __init__(
        self,
        seasonality: int = 7,
        max_anomalies: float = 0.1,
        alpha: float = 0.05
    ):
        """
        Initialize detector.
        
        Args:
            seasonality: Seasonal period
            max_anomalies: Maximum fraction of points to mark as anomalies
            alpha: Significance level
        """
        self.seasonality = seasonality
        self.max_anomalies = max_anomalies
        self.alpha = alpha
    
    def detect(
        self,
        data: Union[np.ndarray, pd.Series]
    ) -> List[int]:
        """
        Detect anomalies using Seasonal ESD.
        
        Args:
            data: Time series data
        
        Returns:
            List of anomaly indices
        """
        if isinstance(data, pd.Series):
            data = data.values
        data = np.array(data).flatten()
        n = len(data)
        
        if n < self.seasonality * 2:
            # Fall back to simple ESD
            return self._simple_esd(data)
        
        # Remove seasonal component using STL-like decomposition
        residuals = self._deseasonalize(data)
        
        # Apply Generalized ESD to residuals
        max_outliers = int(n * self.max_anomalies)
        return self._generalized_esd(residuals, max_outliers)
    
    def _deseasonalize(self, data: np.ndarray) -> np.ndarray:
        """Remove seasonal component using median-based approach."""
        n = len(data)
        seasonal = np.zeros(n)
        
        # Calculate seasonal pattern (median of each season)
        for s in range(self.seasonality):
            indices = range(s, n, self.seasonality)
            seasonal_vals = data[list(indices)]
            seasonal_median = np.median(seasonal_vals)
            seasonal[list(indices)] = seasonal_median
        
        return data - seasonal
    
    def _generalized_esd(
        self,
        data: np.ndarray,
        max_outliers: int
    ) -> List[int]:
        """Generalized ESD test."""
        n = len(data)
        outliers = []
        
        # Work with copy
        remaining = data.copy()
        original_indices = list(range(n))
        
        for i in range(max_outliers):
            if len(remaining) < 3:
                break
            
            # Calculate test statistic
            mean = np.mean(remaining)
            std = np.std(remaining, ddof=1)
            
            if std == 0:
                break
            
            # Find maximum deviation
            deviations = np.abs(remaining - mean)
            max_idx = np.argmax(deviations)
            test_stat = deviations[max_idx] / std
            
            # Calculate critical value
            n_i = len(remaining)
            t_crit = stats.t.ppf(1 - self.alpha / (2 * n_i), n_i - 2)
            critical = (n_i - 1) * t_crit / np.sqrt((n_i - 2 + t_crit**2) * n_i)
            
            if test_stat > critical:
                outliers.append(original_indices[max_idx])
                remaining = np.delete(remaining, max_idx)
                del original_indices[max_idx]
            else:
                break
        
        return sorted(outliers)
    
    def _simple_esd(self, data: np.ndarray) -> List[int]:
        """Simple ESD test for short series."""
        n = len(data)
        max_outliers = int(n * self.max_anomalies)
        return self._generalized_esd(data, max_outliers)


# Convenience functions

def detect_changepoints_bocpd(
    data: Union[np.ndarray, pd.Series],
    hazard_rate: float = 0.01,
    threshold: float = 0.5
) -> List[ChangePoint]:
    """
    Quick function for BOCPD changepoint detection.
    
    Args:
        data: Time series
        hazard_rate: Prior probability of changepoint
        threshold: Confidence threshold
    
    Returns:
        List of ChangePoint objects
    """
    detector = BOCPD(hazard_rate=hazard_rate, threshold=threshold)
    return detector.detect(data)


def test_trend(
    data: Union[np.ndarray, pd.Series],
    method: str = "original",
    alpha: float = 0.05
) -> TrendResult:
    """
    Quick function for Mann-Kendall trend test.
    
    Args:
        data: Time series
        method: "original", "seasonal", or "modified"
        alpha: Significance level
    
    Returns:
        TrendResult with statistics
    """
    mk = MannKendall(alpha=alpha, method=method)
    return mk.test(data)


def dtw_distance(
    s1: Union[np.ndarray, pd.Series],
    s2: Union[np.ndarray, pd.Series]
) -> float:
    """
    Quick function for DTW distance.
    
    Args:
        s1: First series
        s2: Second series
    
    Returns:
        Normalized DTW distance
    """
    dtw = DTWDistance(normalize=True)
    result = dtw.compute(s1, s2, return_path=False)
    return result.normalized_distance
