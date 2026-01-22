"""
Model Drift Detection and Monitoring

Production-grade drift detection for forecasting models:
- Data drift detection (input distribution changes)
- Concept drift detection (prediction relationship changes)
- Performance degradation monitoring
- Automatic retraining triggers
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class DriftType(Enum):
    """Types of drift detected"""
    NONE = "none"
    DATA_DRIFT = "data_drift"           # Input distribution changed
    CONCEPT_DRIFT = "concept_drift"     # Target relationship changed
    PERFORMANCE_DRIFT = "performance"   # Model accuracy degraded
    SUDDEN_DRIFT = "sudden"             # Abrupt change
    GRADUAL_DRIFT = "gradual"           # Slow change over time


class DriftSeverity(Enum):
    """Severity levels for drift"""
    NONE = "none"
    LOW = "low"           # Monitor but no action needed
    MEDIUM = "medium"     # Consider retraining soon
    HIGH = "high"         # Retrain recommended
    CRITICAL = "critical" # Immediate retraining required


@dataclass
class DriftReport:
    """Comprehensive drift detection report"""
    timestamp: float
    model_name: str
    drift_detected: bool
    drift_type: DriftType
    severity: DriftSeverity
    confidence: float  # 0-1 confidence in detection
    details: Dict[str, Any]
    recommendation: str
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class PerformanceWindow:
    """Rolling window of performance metrics"""
    errors: List[float]
    timestamps: List[float]
    predictions: List[float]
    actuals: List[float]


class StatisticalDriftDetector:
    """
    Statistical tests for detecting distribution drift.
    
    Uses:
    - Kolmogorov-Smirnov test for distribution comparison
    - Population Stability Index (PSI) for binned comparison
    - Jensen-Shannon divergence for probability distributions
    """
    
    @staticmethod
    def ks_test(
        reference: np.ndarray, 
        current: np.ndarray,
        significance: float = 0.05
    ) -> Tuple[bool, float, float]:
        """
        Kolmogorov-Smirnov test for distribution drift.
        
        Args:
            reference: Reference (training) distribution
            current: Current (production) distribution
            significance: Significance level (default 0.05)
        
        Returns:
            (drift_detected, statistic, p_value)
        """
        statistic, p_value = stats.ks_2samp(reference, current)
        drift_detected = p_value < significance
        return drift_detected, statistic, p_value
    
    @staticmethod
    def psi(
        reference: np.ndarray, 
        current: np.ndarray,
        n_bins: int = 10,
        threshold: float = 0.25
    ) -> Tuple[bool, float]:
        """
        Population Stability Index for drift detection.
        
        PSI < 0.1: No drift
        PSI 0.1-0.25: Moderate drift
        PSI > 0.25: Significant drift
        
        Args:
            reference: Reference distribution
            current: Current distribution
            n_bins: Number of bins for histogram
            threshold: PSI threshold for drift (default 0.25)
        
        Returns:
            (drift_detected, psi_value)
        """
        # Create bins from reference distribution
        _, bin_edges = np.histogram(reference, bins=n_bins)
        
        # Calculate frequencies
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        curr_counts, _ = np.histogram(current, bins=bin_edges)
        
        # Convert to proportions (add small epsilon to avoid division by zero)
        eps = 1e-10
        ref_pct = (ref_counts + eps) / (len(reference) + eps * n_bins)
        curr_pct = (curr_counts + eps) / (len(current) + eps * n_bins)
        
        # Calculate PSI
        psi_value = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
        
        drift_detected = psi_value > threshold
        return drift_detected, psi_value
    
    @staticmethod
    def js_divergence(
        reference: np.ndarray, 
        current: np.ndarray,
        n_bins: int = 10,
        threshold: float = 0.1
    ) -> Tuple[bool, float]:
        """
        Jensen-Shannon divergence for drift detection.
        
        Symmetric measure of distribution difference.
        Range: [0, 1] where 0 = identical distributions
        
        Args:
            reference: Reference distribution
            current: Current distribution
            n_bins: Number of bins
            threshold: JS threshold for drift
        
        Returns:
            (drift_detected, js_value)
        """
        # Create histogram bins
        all_data = np.concatenate([reference, current])
        _, bin_edges = np.histogram(all_data, bins=n_bins)
        
        # Calculate probability distributions
        eps = 1e-10
        ref_hist, _ = np.histogram(reference, bins=bin_edges, density=True)
        curr_hist, _ = np.histogram(current, bins=bin_edges, density=True)
        
        ref_prob = ref_hist + eps
        curr_prob = curr_hist + eps
        
        # Normalize
        ref_prob = ref_prob / ref_prob.sum()
        curr_prob = curr_prob / curr_prob.sum()
        
        # Calculate JS divergence
        m = 0.5 * (ref_prob + curr_prob)
        js_value = 0.5 * (
            np.sum(ref_prob * np.log(ref_prob / m)) +
            np.sum(curr_prob * np.log(curr_prob / m))
        )
        
        drift_detected = js_value > threshold
        return drift_detected, js_value


class ADWINDriftDetector:
    """
    ADWIN (ADaptive WINdowing) for concept drift detection.
    
    Maintains a variable-length window of recent data.
    Detects changes by comparing sub-windows.
    
    Reference: Bifet & Gavalda, "Learning from Time-Changing Data 
    with Adaptive Windowing"
    """
    
    def __init__(
        self,
        delta: float = 0.002,
        max_window_size: int = 1000
    ):
        """
        Initialize ADWIN detector.
        
        Args:
            delta: Confidence parameter (smaller = more sensitive)
            max_window_size: Maximum window size
        """
        self.delta = delta
        self.max_window_size = max_window_size
        self.window: deque = deque(maxlen=max_window_size)
        self.total = 0.0
        self.variance = 0.0
        self.n = 0
    
    def add_element(self, value: float) -> bool:
        """
        Add element and check for drift.
        
        Args:
            value: New observation
        
        Returns:
            True if drift detected
        """
        self.window.append(value)
        self.n += 1
        self.total += value
        
        if self.n < 2:
            return False
        
        # Update variance (Welford's algorithm)
        mean = self.total / self.n
        self.variance += (value - mean) ** 2
        
        # Check for drift
        return self._detect_change()
    
    def _detect_change(self) -> bool:
        """Check if there's a significant change in the window"""
        if len(self.window) < 10:
            return False
        
        window_array = np.array(self.window)
        n = len(window_array)
        
        # Try different split points
        for i in range(10, n - 10):
            n0 = i
            n1 = n - i
            
            mean0 = np.mean(window_array[:i])
            mean1 = np.mean(window_array[i:])
            
            # Hoeffding bound
            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            epsilon = np.sqrt(2.0 * m * np.log(2.0 / self.delta)) / m
            
            if abs(mean0 - mean1) > epsilon:
                # Drift detected - shrink window
                self._shrink_window(i)
                return True
        
        return False
    
    def _shrink_window(self, split_point: int):
        """Remove old data from window after drift"""
        for _ in range(split_point):
            if self.window:
                removed = self.window.popleft()
                self.total -= removed
                self.n -= 1
    
    def reset(self):
        """Reset the detector"""
        self.window.clear()
        self.total = 0.0
        self.variance = 0.0
        self.n = 0


class PageHinkleyDetector:
    """
    Page-Hinkley test for change detection.
    
    Detects changes in the mean of a Gaussian distribution.
    Good for detecting gradual drift.
    """
    
    def __init__(
        self,
        delta: float = 0.005,
        threshold: float = 50.0,
        alpha: float = 0.9999
    ):
        """
        Initialize Page-Hinkley detector.
        
        Args:
            delta: Minimum amplitude of change to detect
            threshold: Detection threshold
            alpha: Forgetting factor for mean estimation
        """
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        
        self.sum = 0.0
        self.mean = 0.0
        self.n = 0
        self.min_sum = float('inf')
    
    def add_element(self, value: float) -> bool:
        """
        Add element and check for drift.
        
        Args:
            value: New observation
        
        Returns:
            True if drift detected
        """
        self.n += 1
        
        # Update mean with forgetting
        self.mean = self.alpha * self.mean + (1 - self.alpha) * value
        
        # Update sum
        self.sum += value - self.mean - self.delta
        self.min_sum = min(self.min_sum, self.sum)
        
        # Check for drift
        if self.sum - self.min_sum > self.threshold:
            self.reset()
            return True
        
        return False
    
    def reset(self):
        """Reset the detector"""
        self.sum = 0.0
        self.min_sum = float('inf')
        # Keep mean for continuity


class ModelDriftMonitor:
    """
    Comprehensive model drift monitoring.
    
    Combines multiple drift detection methods:
    - Data drift (input features)
    - Prediction drift (model outputs)
    - Performance drift (error metrics)
    - Concept drift (input-output relationship)
    
    Usage:
        monitor = ModelDriftMonitor('my_model')
        
        # Store reference distribution
        monitor.set_reference(train_predictions, train_actuals)
        
        # In production, check for drift
        report = monitor.check_drift(new_predictions, new_actuals)
        if report.drift_detected:
            print(f"Drift detected: {report.drift_type}")
    """
    
    def __init__(
        self,
        model_name: str,
        window_size: int = 100,
        performance_threshold: float = 0.2,  # 20% degradation
        data_drift_threshold: float = 0.25,   # PSI threshold
        concept_drift_delta: float = 0.002    # ADWIN delta
    ):
        """
        Initialize drift monitor.
        
        Args:
            model_name: Name of the model being monitored
            window_size: Size of rolling performance window
            performance_threshold: Relative degradation threshold
            data_drift_threshold: PSI threshold for data drift
            concept_drift_delta: ADWIN sensitivity parameter
        """
        self.model_name = model_name
        self.window_size = window_size
        self.performance_threshold = performance_threshold
        self.data_drift_threshold = data_drift_threshold
        
        # Reference distributions (from training)
        self.reference_predictions: Optional[np.ndarray] = None
        self.reference_errors: Optional[np.ndarray] = None
        self.reference_mape: Optional[float] = None
        
        # Rolling performance window
        self.performance_window = PerformanceWindow(
            errors=[],
            timestamps=[],
            predictions=[],
            actuals=[]
        )
        
        # Drift detectors
        self.adwin = ADWINDriftDetector(delta=concept_drift_delta)
        self.page_hinkley = PageHinkleyDetector()
        self.stat_detector = StatisticalDriftDetector()
        
        # History
        self.drift_history: List[DriftReport] = []
    
    def set_reference(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray
    ):
        """
        Set reference distributions from training/validation data.
        
        Args:
            predictions: Model predictions on validation set
            actuals: Actual values from validation set
        """
        self.reference_predictions = np.asarray(predictions).flatten()
        actuals = np.asarray(actuals).flatten()
        
        # Calculate reference errors
        self.reference_errors = np.abs(self.reference_predictions - actuals)
        
        # Calculate reference MAPE
        mask = actuals != 0
        if mask.any():
            self.reference_mape = np.mean(
                np.abs((actuals[mask] - self.reference_predictions[mask]) / actuals[mask])
            ) * 100
        else:
            self.reference_mape = np.mean(self.reference_errors)
        
        logger.info(
            f"Reference set for {self.model_name}: "
            f"MAPE={self.reference_mape:.2f}%, "
            f"samples={len(self.reference_predictions)}"
        )
    
    def add_observation(
        self,
        prediction: float,
        actual: float,
        timestamp: Optional[float] = None
    ) -> Optional[DriftReport]:
        """
        Add a single prediction/actual pair and check for drift.
        
        Args:
            prediction: Model prediction
            actual: Actual observed value
            timestamp: Observation timestamp
        
        Returns:
            DriftReport if drift detected, None otherwise
        """
        timestamp = timestamp or time.time()
        error = abs(prediction - actual)
        
        # Add to rolling window
        self.performance_window.errors.append(error)
        self.performance_window.timestamps.append(timestamp)
        self.performance_window.predictions.append(prediction)
        self.performance_window.actuals.append(actual)
        
        # Trim window
        if len(self.performance_window.errors) > self.window_size:
            self.performance_window.errors.pop(0)
            self.performance_window.timestamps.pop(0)
            self.performance_window.predictions.pop(0)
            self.performance_window.actuals.pop(0)
        
        # Check ADWIN for sudden drift
        if self.adwin.add_element(error):
            return self._create_report(
                DriftType.SUDDEN_DRIFT,
                DriftSeverity.HIGH,
                confidence=0.9,
                details={'detector': 'adwin', 'trigger': 'error_change'}
            )
        
        # Check Page-Hinkley for gradual drift
        if self.page_hinkley.add_element(error):
            return self._create_report(
                DriftType.GRADUAL_DRIFT,
                DriftSeverity.MEDIUM,
                confidence=0.8,
                details={'detector': 'page_hinkley', 'trigger': 'mean_shift'}
            )
        
        return None
    
    def check_drift(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        features: Optional[np.ndarray] = None
    ) -> DriftReport:
        """
        Comprehensive drift check on a batch of data.
        
        Args:
            predictions: Recent model predictions
            actuals: Actual observed values
            features: Input features (optional, for data drift)
        
        Returns:
            DriftReport with full analysis
        """
        predictions = np.asarray(predictions).flatten()
        actuals = np.asarray(actuals).flatten()
        
        if self.reference_predictions is None:
            return self._create_report(
                DriftType.NONE,
                DriftSeverity.NONE,
                confidence=0.0,
                details={'error': 'No reference set'}
            )
        
        drift_signals = []
        details = {}
        
        # 1. Check prediction distribution drift
        pred_drift, pred_psi = self.stat_detector.psi(
            self.reference_predictions,
            predictions,
            threshold=self.data_drift_threshold
        )
        details['prediction_psi'] = pred_psi
        
        if pred_drift:
            drift_signals.append(('prediction_drift', pred_psi))
        
        # 2. Check error distribution drift
        current_errors = np.abs(predictions - actuals)
        error_drift, error_psi = self.stat_detector.psi(
            self.reference_errors,
            current_errors,
            threshold=self.data_drift_threshold
        )
        details['error_psi'] = error_psi
        
        if error_drift:
            drift_signals.append(('error_drift', error_psi))
        
        # 3. Check performance degradation
        mask = actuals != 0
        if mask.any():
            current_mape = np.mean(
                np.abs((actuals[mask] - predictions[mask]) / actuals[mask])
            ) * 100
        else:
            current_mape = np.mean(current_errors)
        
        details['current_mape'] = current_mape
        details['reference_mape'] = self.reference_mape
        
        if self.reference_mape > 0:
            degradation = (current_mape - self.reference_mape) / self.reference_mape
            details['degradation_pct'] = degradation * 100
            
            if degradation > self.performance_threshold:
                drift_signals.append(('performance_degradation', degradation))
        
        # 4. Statistical test for distribution change
        ks_drift, ks_stat, ks_pvalue = self.stat_detector.ks_test(
            self.reference_errors,
            current_errors
        )
        details['ks_statistic'] = ks_stat
        details['ks_pvalue'] = ks_pvalue
        
        if ks_drift:
            drift_signals.append(('ks_test', ks_stat))
        
        # Determine overall drift status
        if not drift_signals:
            return self._create_report(
                DriftType.NONE,
                DriftSeverity.NONE,
                confidence=1.0 - max(pred_psi, error_psi),
                details=details
            )
        
        # Determine severity based on number and strength of signals
        n_signals = len(drift_signals)
        max_signal = max(s[1] for s in drift_signals)
        
        if n_signals >= 3 or max_signal > 0.5:
            severity = DriftSeverity.CRITICAL
        elif n_signals >= 2 or max_signal > 0.35:
            severity = DriftSeverity.HIGH
        elif max_signal > 0.25:
            severity = DriftSeverity.MEDIUM
        else:
            severity = DriftSeverity.LOW
        
        # Determine drift type
        if 'performance_degradation' in [s[0] for s in drift_signals]:
            drift_type = DriftType.PERFORMANCE_DRIFT
        elif 'error_drift' in [s[0] for s in drift_signals]:
            drift_type = DriftType.CONCEPT_DRIFT
        else:
            drift_type = DriftType.DATA_DRIFT
        
        details['drift_signals'] = drift_signals
        
        report = self._create_report(
            drift_type=drift_type,
            severity=severity,
            confidence=min(1.0, n_signals * 0.3 + max_signal),
            details=details
        )
        
        self.drift_history.append(report)
        return report
    
    def _create_report(
        self,
        drift_type: DriftType,
        severity: DriftSeverity,
        confidence: float,
        details: Dict[str, Any]
    ) -> DriftReport:
        """Create a drift report with recommendation"""
        
        # Generate recommendation
        if severity == DriftSeverity.NONE:
            recommendation = "No action needed. Model performing within expected bounds."
        elif severity == DriftSeverity.LOW:
            recommendation = "Monitor closely. Consider retraining if drift persists."
        elif severity == DriftSeverity.MEDIUM:
            recommendation = "Schedule model retraining soon. Performance may degrade."
        elif severity == DriftSeverity.HIGH:
            recommendation = "Retrain model recommended. Significant drift detected."
        else:  # CRITICAL
            recommendation = "URGENT: Retrain model immediately. Critical performance degradation."
        
        # Add metrics summary
        metrics = {
            'reference_mape': self.reference_mape or 0,
            'current_window_size': len(self.performance_window.errors),
        }
        
        if self.performance_window.errors:
            metrics['recent_mean_error'] = np.mean(self.performance_window.errors)
        
        return DriftReport(
            timestamp=time.time(),
            model_name=self.model_name,
            drift_detected=drift_type != DriftType.NONE,
            drift_type=drift_type,
            severity=severity,
            confidence=confidence,
            details=details,
            recommendation=recommendation,
            metrics=metrics
        )
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall model health summary"""
        recent_drifts = [
            d for d in self.drift_history 
            if time.time() - d.timestamp < 86400  # Last 24 hours
        ]
        
        return {
            'model_name': self.model_name,
            'reference_mape': self.reference_mape,
            'current_window_size': len(self.performance_window.errors),
            'recent_mean_error': (
                np.mean(self.performance_window.errors) 
                if self.performance_window.errors else None
            ),
            'drift_events_24h': len(recent_drifts),
            'last_drift': (
                self.drift_history[-1].timestamp 
                if self.drift_history else None
            ),
            'status': self._get_health_status(recent_drifts)
        }
    
    def _get_health_status(self, recent_drifts: List[DriftReport]) -> str:
        """Determine overall health status"""
        if not recent_drifts:
            return "healthy"
        
        critical_count = sum(
            1 for d in recent_drifts 
            if d.severity == DriftSeverity.CRITICAL
        )
        high_count = sum(
            1 for d in recent_drifts 
            if d.severity == DriftSeverity.HIGH
        )
        
        if critical_count > 0:
            return "critical"
        elif high_count > 2:
            return "degraded"
        elif high_count > 0:
            return "warning"
        else:
            return "healthy"


# Convenience functions
def create_drift_monitor(model_name: str) -> ModelDriftMonitor:
    """Create a drift monitor for a model"""
    return ModelDriftMonitor(model_name)


def check_model_drift(
    model_name: str,
    reference_predictions: np.ndarray,
    reference_actuals: np.ndarray,
    current_predictions: np.ndarray,
    current_actuals: np.ndarray
) -> DriftReport:
    """
    Quick drift check between reference and current data.
    
    Args:
        model_name: Name of the model
        reference_predictions: Predictions from reference period
        reference_actuals: Actuals from reference period
        current_predictions: Recent predictions
        current_actuals: Recent actuals
    
    Returns:
        DriftReport with analysis
    """
    monitor = ModelDriftMonitor(model_name)
    monitor.set_reference(reference_predictions, reference_actuals)
    return monitor.check_drift(current_predictions, current_actuals)
