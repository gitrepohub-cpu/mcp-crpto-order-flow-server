"""
Time Series Cross-Validation for Financial Forecasting

Production-grade cross-validation with:
- Purged K-Fold (prevents data leakage)
- Walk-forward validation with embargo gaps
- Combinatorial purged CV
- Block time series splits
"""

import logging
from typing import List, Tuple, Optional, Iterator, Dict, Any, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
from enum import Enum

logger = logging.getLogger(__name__)


class CVStrategy(Enum):
    """Cross-validation strategies for time series"""
    EXPANDING_WINDOW = "expanding_window"    # Train on all past data
    SLIDING_WINDOW = "sliding_window"        # Fixed-size training window
    PURGED_KFOLD = "purged_kfold"           # K-fold with purging
    BLOCKED = "blocked"                      # Non-overlapping blocks


@dataclass
class CVFold:
    """A single cross-validation fold"""
    fold_index: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    embargo_size: int = 0


@dataclass
class CVResult:
    """Result of cross-validation evaluation"""
    model_name: str
    cv_strategy: str
    n_folds: int
    scores: List[float]
    mean_score: float
    std_score: float
    metric_name: str
    fold_details: List[Dict[str, Any]]


class TimeSeriesCrossValidator:
    """
    Time series cross-validation with financial-grade safeguards.
    
    Prevents data leakage through:
    - Embargo gaps between train/test
    - Purging overlapping samples
    - Forward-only validation (no future data in training)
    
    Usage:
        cv = TimeSeriesCrossValidator(
            n_splits=5,
            embargo_pct=0.01,
            purge_pct=0.01
        )
        
        for train_idx, test_idx in cv.split(data):
            train_data = data[train_idx]
            test_data = data[test_idx]
            # Train and evaluate model
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        test_size: Optional[int] = None,
        test_pct: float = 0.2,
        embargo_pct: float = 0.01,
        purge_pct: float = 0.01,
        gap: int = 0,
        strategy: CVStrategy = CVStrategy.EXPANDING_WINDOW
    ):
        """
        Initialize cross-validator.
        
        Args:
            n_splits: Number of CV folds
            test_size: Fixed test size (overrides test_pct)
            test_pct: Percentage of data for test set
            embargo_pct: Percentage of data to exclude after test
            purge_pct: Percentage of data to purge around test
            gap: Fixed gap between train and test (periods)
            strategy: CV strategy to use
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.test_pct = test_pct
        self.embargo_pct = embargo_pct
        self.purge_pct = purge_pct
        self.gap = gap
        self.strategy = strategy
    
    def split(
        self, 
        X: Union[np.ndarray, pd.DataFrame, pd.Series],
        y: Optional[np.ndarray] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for each fold.
        
        Args:
            X: Data array or DataFrame (used for length)
            y: Ignored (for sklearn compatibility)
        
        Yields:
            Tuple of (train_indices, test_indices)
        """
        n_samples = len(X)
        
        if self.strategy == CVStrategy.EXPANDING_WINDOW:
            yield from self._expanding_window_split(n_samples)
        elif self.strategy == CVStrategy.SLIDING_WINDOW:
            yield from self._sliding_window_split(n_samples)
        elif self.strategy == CVStrategy.PURGED_KFOLD:
            yield from self._purged_kfold_split(n_samples)
        elif self.strategy == CVStrategy.BLOCKED:
            yield from self._blocked_split(n_samples)
        else:
            yield from self._expanding_window_split(n_samples)
    
    def get_folds(
        self, 
        X: Union[np.ndarray, pd.DataFrame, pd.Series]
    ) -> List[CVFold]:
        """
        Get detailed fold information.
        
        Args:
            X: Data array or DataFrame
        
        Returns:
            List of CVFold objects with full details
        """
        folds = []
        n_samples = len(X)
        embargo_size = int(n_samples * self.embargo_pct)
        
        for i, (train_idx, test_idx) in enumerate(self.split(X)):
            folds.append(CVFold(
                fold_index=i,
                train_indices=train_idx,
                test_indices=test_idx,
                train_start=train_idx[0],
                train_end=train_idx[-1],
                test_start=test_idx[0],
                test_end=test_idx[-1],
                embargo_size=embargo_size
            ))
        
        return folds
    
    def _expanding_window_split(
        self, 
        n_samples: int
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Expanding window CV: train on all past, test on next period.
        
        Train: [0, t)  ->  Test: [t, t+test_size)
        Each fold expands training window
        """
        test_size = self.test_size or int(n_samples * self.test_pct)
        embargo = int(n_samples * self.embargo_pct)
        
        # Calculate fold boundaries
        min_train_size = max(test_size * 2, n_samples // (self.n_splits + 1))
        
        for i in range(self.n_splits):
            # Test start position (moves forward each fold)
            test_start = min_train_size + i * test_size
            test_end = min(test_start + test_size, n_samples)
            
            if test_end > n_samples:
                break
            
            # Training ends before test (with gap and embargo)
            train_end = max(0, test_start - self.gap - embargo)
            
            if train_end < min_train_size // 2:
                continue
            
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_end)
            
            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx
    
    def _sliding_window_split(
        self, 
        n_samples: int
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Sliding window CV: fixed-size training window.
        
        Train: [t-window, t)  ->  Test: [t, t+test_size)
        Window slides forward each fold
        """
        test_size = self.test_size or int(n_samples * self.test_pct)
        train_size = int(n_samples * (1 - self.test_pct) / self.n_splits)
        embargo = int(n_samples * self.embargo_pct)
        
        step = (n_samples - train_size - test_size) // self.n_splits
        
        for i in range(self.n_splits):
            train_start = i * step
            train_end = train_start + train_size
            
            test_start = train_end + self.gap + embargo
            test_end = min(test_start + test_size, n_samples)
            
            if test_end > n_samples:
                break
            
            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)
            
            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx
    
    def _purged_kfold_split(
        self, 
        n_samples: int
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Purged K-Fold CV: removes samples that could leak information.
        
        Purges samples within purge_pct of test boundaries.
        Adds embargo gap after test set.
        """
        fold_size = n_samples // self.n_splits
        purge_size = int(n_samples * self.purge_pct)
        embargo_size = int(n_samples * self.embargo_pct)
        
        for i in range(self.n_splits):
            # Test set boundaries
            test_start = i * fold_size
            test_end = min((i + 1) * fold_size, n_samples)
            
            # Define purge boundaries
            purge_before_start = max(0, test_start - purge_size)
            purge_after_end = min(n_samples, test_end + embargo_size)
            
            # Training indices: everything except test + purge zones
            train_before = np.arange(0, purge_before_start)
            train_after = np.arange(purge_after_end, n_samples)
            
            # Only use training data BEFORE test (time series constraint)
            train_idx = train_before
            test_idx = np.arange(test_start, test_end)
            
            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx
    
    def _blocked_split(
        self, 
        n_samples: int
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Blocked time series split: non-overlapping contiguous blocks.
        
        Divides data into blocks, uses first N-1 blocks for train,
        last block for test, then shifts.
        """
        block_size = n_samples // (self.n_splits + 1)
        embargo = int(n_samples * self.embargo_pct)
        
        for i in range(self.n_splits):
            # Training: all blocks up to i+1
            train_end = (i + 1) * block_size - embargo
            
            # Test: block i+1
            test_start = (i + 1) * block_size
            test_end = min((i + 2) * block_size, n_samples)
            
            train_idx = np.arange(0, max(0, train_end))
            test_idx = np.arange(test_start, test_end)
            
            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx


class WalkForwardValidator:
    """
    Walk-forward validation with anchored or rolling origin.
    
    Mimics real trading: train on past, predict future, move forward.
    
    Usage:
        wfv = WalkForwardValidator(
            train_size=100,
            test_size=20,
            step_size=10
        )
        
        for train_idx, test_idx in wfv.split(data):
            # Train and evaluate
    """
    
    def __init__(
        self,
        train_size: Optional[int] = None,
        test_size: int = 24,
        step_size: Optional[int] = None,
        min_train_size: int = 100,
        expanding: bool = True,
        embargo: int = 0
    ):
        """
        Initialize walk-forward validator.
        
        Args:
            train_size: Fixed training window (None for expanding)
            test_size: Size of test window
            step_size: Steps between folds (default: test_size)
            min_train_size: Minimum training samples
            expanding: Use expanding window (else rolling)
            embargo: Gap between train and test
        """
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size or test_size
        self.min_train_size = min_train_size
        self.expanding = expanding
        self.embargo = embargo
    
    def split(
        self, 
        X: Union[np.ndarray, pd.DataFrame]
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test indices for walk-forward validation"""
        n_samples = len(X)
        
        # Start position
        origin = self.min_train_size
        
        while origin + self.embargo + self.test_size <= n_samples:
            # Training window
            if self.expanding:
                train_start = 0
            else:
                train_start = max(0, origin - (self.train_size or self.min_train_size))
            
            train_end = origin
            
            # Test window (after embargo)
            test_start = origin + self.embargo
            test_end = min(test_start + self.test_size, n_samples)
            
            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)
            
            yield train_idx, test_idx
            
            # Move origin forward
            origin += self.step_size
    
    def get_n_splits(self, X: Union[np.ndarray, pd.DataFrame]) -> int:
        """Calculate number of splits"""
        n_samples = len(X)
        usable = n_samples - self.min_train_size - self.embargo - self.test_size
        return max(1, usable // self.step_size + 1)


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation (CPCV).
    
    Most rigorous CV for financial data:
    - Tests all possible train/test combinations
    - Purges overlapping samples
    - Prevents all forms of leakage
    
    Reference: "Advances in Financial Machine Learning" by López de Prado
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        n_test_splits: int = 2,
        purge_pct: float = 0.01,
        embargo_pct: float = 0.01
    ):
        """
        Initialize CPCV.
        
        Args:
            n_splits: Total number of groups
            n_test_splits: Number of groups in each test set
            purge_pct: Samples to purge around boundaries
            embargo_pct: Gap after test set
        """
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.purge_pct = purge_pct
        self.embargo_pct = embargo_pct
    
    def split(
        self, 
        X: Union[np.ndarray, pd.DataFrame]
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate combinatorial train/test splits"""
        from itertools import combinations
        
        n_samples = len(X)
        group_size = n_samples // self.n_splits
        purge_size = int(n_samples * self.purge_pct)
        embargo_size = int(n_samples * self.embargo_pct)
        
        # Create groups
        groups = []
        for i in range(self.n_splits):
            start = i * group_size
            end = (i + 1) * group_size if i < self.n_splits - 1 else n_samples
            groups.append((start, end))
        
        # Generate all combinations of test groups
        for test_groups in combinations(range(self.n_splits), self.n_test_splits):
            # Test indices
            test_idx = []
            for g in test_groups:
                start, end = groups[g]
                test_idx.extend(range(start, end))
            test_idx = np.array(sorted(test_idx))
            
            # Find boundaries for purging
            test_start = test_idx[0]
            test_end = test_idx[-1]
            
            # Training indices: before test, with purging
            train_end = max(0, test_start - purge_size - embargo_size)
            train_idx = np.arange(0, train_end)
            
            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx
    
    def get_n_splits(self) -> int:
        """Calculate total number of combinations"""
        from math import comb
        return comb(self.n_splits, self.n_test_splits)


def cross_validate_forecast(
    model,
    series,  # Darts TimeSeries
    cv: Union[TimeSeriesCrossValidator, WalkForwardValidator],
    metric: str = 'mape',
    forecast_horizon: Optional[int] = None
) -> CVResult:
    """
    Cross-validate a forecasting model.
    
    Args:
        model: Darts model or DartsModelWrapper
        series: Darts TimeSeries
        cv: Cross-validator instance
        metric: Evaluation metric ('mape', 'rmse', 'mae', 'smape')
        forecast_horizon: Steps to forecast (default: test size)
    
    Returns:
        CVResult with scores and details
    """
    from src.integrations.darts_bridge import DartsBridge
    
    bridge = DartsBridge()
    data = bridge.from_darts(series)
    
    scores = []
    fold_details = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(data)):
        try:
            # Get train/test series
            train_data = data[train_idx]
            test_data = data[test_idx]
            
            train_series = bridge.to_darts(train_data)
            
            # Train model
            model.fit(train_series)
            
            # Forecast
            horizon = forecast_horizon or len(test_idx)
            predictions = model.predict(n=horizon)
            pred_values = predictions.values().flatten()[:len(test_idx)]
            
            # Calculate metric
            score = _calculate_cv_metric(test_data, pred_values, metric)
            scores.append(score)
            
            fold_details.append({
                'fold': fold_idx,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'score': score
            })
            
            logger.debug(f"Fold {fold_idx}: {metric}={score:.4f}")
            
        except Exception as e:
            logger.warning(f"Fold {fold_idx} failed: {e}")
            continue
    
    if not scores:
        raise ValueError("All CV folds failed")
    
    # Get model name
    model_name = getattr(model, 'model_name', str(type(model).__name__))
    cv_strategy = getattr(cv, 'strategy', type(cv).__name__)
    if hasattr(cv_strategy, 'value'):
        cv_strategy = cv_strategy.value
    
    return CVResult(
        model_name=model_name,
        cv_strategy=str(cv_strategy),
        n_folds=len(scores),
        scores=scores,
        mean_score=np.mean(scores),
        std_score=np.std(scores),
        metric_name=metric,
        fold_details=fold_details
    )


def _calculate_cv_metric(
    actual: np.ndarray, 
    predicted: np.ndarray, 
    metric: str
) -> float:
    """Calculate forecasting metric for CV"""
    actual = np.asarray(actual).flatten()
    predicted = np.asarray(predicted).flatten()
    
    # Handle length mismatch
    min_len = min(len(actual), len(predicted))
    actual = actual[:min_len]
    predicted = predicted[:min_len]
    
    if metric == 'mape':
        mask = actual != 0
        if not mask.any():
            return float('inf')
        return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    elif metric == 'rmse':
        return np.sqrt(np.mean((actual - predicted) ** 2))
    elif metric == 'mae':
        return np.mean(np.abs(actual - predicted))
    elif metric == 'smape':
        return np.mean(2 * np.abs(actual - predicted) / 
                      (np.abs(actual) + np.abs(predicted) + 1e-8)) * 100
    else:
        mask = actual != 0
        return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


# Convenience functions
def expanding_window_cv(n_splits: int = 5, embargo_pct: float = 0.01) -> TimeSeriesCrossValidator:
    """Create expanding window cross-validator"""
    return TimeSeriesCrossValidator(
        n_splits=n_splits,
        embargo_pct=embargo_pct,
        strategy=CVStrategy.EXPANDING_WINDOW
    )


def purged_kfold_cv(n_splits: int = 5, purge_pct: float = 0.01) -> TimeSeriesCrossValidator:
    """Create purged k-fold cross-validator"""
    return TimeSeriesCrossValidator(
        n_splits=n_splits,
        purge_pct=purge_pct,
        strategy=CVStrategy.PURGED_KFOLD
    )


def walk_forward_cv(
    train_size: int = 168, 
    test_size: int = 24,
    expanding: bool = True
) -> WalkForwardValidator:
    """Create walk-forward cross-validator"""
    return WalkForwardValidator(
        train_size=train_size,
        test_size=test_size,
        expanding=expanding
    )
