"""
Ensemble Forecasting Methods.

This module provides ensemble forecasting capabilities similar to Kats:
- Weighted Average Ensemble
- Median Ensemble
- Stacking Ensemble
- Model Selection Ensemble

Ensemble methods typically improve forecast accuracy by 10-30% over single models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import warnings

logger = logging.getLogger(__name__)


class EnsembleMethod(Enum):
    """Supported ensemble methods."""
    SIMPLE_AVERAGE = "simple_average"
    WEIGHTED_AVERAGE = "weighted_average"
    MEDIAN = "median"
    TRIMMED_MEAN = "trimmed_mean"
    BEST_MODEL = "best_model"
    STACKING = "stacking"
    BAYESIAN = "bayesian"


@dataclass
class ForecastResult:
    """Individual forecast result from a model."""
    model_name: str
    predictions: np.ndarray
    lower_bound: Optional[np.ndarray] = None
    upper_bound: Optional[np.ndarray] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnsembleForecastResult:
    """Result from ensemble forecasting."""
    predictions: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    method: EnsembleMethod
    weights: Dict[str, float]
    individual_forecasts: List[ForecastResult]
    ensemble_metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseForecaster(ABC):
    """Abstract base class for forecasters in ensemble."""
    
    @abstractmethod
    def fit(self, data: np.ndarray, **kwargs) -> 'BaseForecaster':
        """Fit the model."""
        pass
    
    @abstractmethod
    def predict(self, steps: int, **kwargs) -> np.ndarray:
        """Generate predictions."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Model name."""
        pass


class ARIMAForecaster(BaseForecaster):
    """ARIMA model wrapper for ensemble."""
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1)):
        self.order = order
        self.model = None
        self._fitted_values = None
    
    @property
    def name(self) -> str:
        return f"ARIMA{self.order}"
    
    def fit(self, data: np.ndarray, **kwargs) -> 'ARIMAForecaster':
        try:
            from statsmodels.tsa.arima.model import ARIMA
            self.model = ARIMA(data, order=self.order)
            self._fit_result = self.model.fit()
            self._fitted_values = self._fit_result.fittedvalues
        except Exception as e:
            logger.warning(f"ARIMA fit failed: {e}")
            self._fit_result = None
        return self
    
    def predict(self, steps: int, **kwargs) -> np.ndarray:
        if self._fit_result is None:
            return np.full(steps, np.nan)
        try:
            forecast = self._fit_result.forecast(steps=steps)
            return np.array(forecast)
        except Exception as e:
            logger.warning(f"ARIMA predict failed: {e}")
            return np.full(steps, np.nan)


class ExponentialSmoothingForecaster(BaseForecaster):
    """Exponential Smoothing model wrapper for ensemble."""
    
    def __init__(
        self,
        trend: Optional[str] = 'add',
        seasonal: Optional[str] = None,
        seasonal_periods: int = 7
    ):
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self._fit_result = None
    
    @property
    def name(self) -> str:
        return f"ExpSmooth(trend={self.trend})"
    
    def fit(self, data: np.ndarray, **kwargs) -> 'ExponentialSmoothingForecaster':
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            
            model = ExponentialSmoothing(
                data,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods if self.seasonal else None
            )
            self._fit_result = model.fit()
        except Exception as e:
            logger.warning(f"ExponentialSmoothing fit failed: {e}")
            self._fit_result = None
        return self
    
    def predict(self, steps: int, **kwargs) -> np.ndarray:
        if self._fit_result is None:
            return np.full(steps, np.nan)
        try:
            forecast = self._fit_result.forecast(steps=steps)
            return np.array(forecast)
        except Exception as e:
            logger.warning(f"ExponentialSmoothing predict failed: {e}")
            return np.full(steps, np.nan)


class ThetaForecaster(BaseForecaster):
    """Theta model wrapper for ensemble."""
    
    def __init__(self, period: int = 1):
        self.period = period
        self._fit_result = None
    
    @property
    def name(self) -> str:
        return "Theta"
    
    def fit(self, data: np.ndarray, **kwargs) -> 'ThetaForecaster':
        try:
            from statsmodels.tsa.forecasting.theta import ThetaModel
            model = ThetaModel(data, period=self.period)
            self._fit_result = model.fit()
        except Exception as e:
            logger.warning(f"Theta fit failed: {e}")
            self._fit_result = None
        return self
    
    def predict(self, steps: int, **kwargs) -> np.ndarray:
        if self._fit_result is None:
            return np.full(steps, np.nan)
        try:
            forecast = self._fit_result.forecast(steps=steps)
            return np.array(forecast)
        except Exception as e:
            logger.warning(f"Theta predict failed: {e}")
            return np.full(steps, np.nan)


class NaiveForecaster(BaseForecaster):
    """Naive forecaster (last value or seasonal naive)."""
    
    def __init__(self, seasonal_period: Optional[int] = None):
        self.seasonal_period = seasonal_period
        self._last_values = None
    
    @property
    def name(self) -> str:
        if self.seasonal_period:
            return f"SeasonalNaive({self.seasonal_period})"
        return "Naive"
    
    def fit(self, data: np.ndarray, **kwargs) -> 'NaiveForecaster':
        if self.seasonal_period:
            self._last_values = data[-self.seasonal_period:]
        else:
            self._last_values = np.array([data[-1]])
        return self
    
    def predict(self, steps: int, **kwargs) -> np.ndarray:
        if self._last_values is None:
            return np.full(steps, np.nan)
        
        if self.seasonal_period:
            # Repeat seasonal pattern
            repeats = (steps // self.seasonal_period) + 1
            forecast = np.tile(self._last_values, repeats)[:steps]
        else:
            # Constant forecast
            forecast = np.full(steps, self._last_values[0])
        
        return forecast


class EnsembleForecaster:
    """
    Ensemble Forecaster combining multiple models.
    
    Supports multiple ensemble methods:
    - Simple Average: Equal weights for all models
    - Weighted Average: Weights based on historical accuracy
    - Median: Robust to outliers
    - Trimmed Mean: Removes extreme predictions
    - Best Model: Selects single best model
    - Stacking: Meta-learner combines predictions
    
    Example usage:
        ensemble = EnsembleForecaster()
        
        # Add models
        ensemble.add_model(ARIMAForecaster((1,1,1)))
        ensemble.add_model(ExponentialSmoothingForecaster())
        ensemble.add_model(ThetaForecaster())
        
        # Fit and predict
        result = ensemble.fit_predict(data, periods=30)
        print(f"Ensemble predictions: {result.predictions}")
        print(f"Model weights: {result.weights}")
    """
    
    def __init__(
        self,
        method: EnsembleMethod = EnsembleMethod.WEIGHTED_AVERAGE,
        confidence_level: float = 0.95,
        parallel: bool = True,
        n_jobs: int = -1
    ):
        """
        Initialize EnsembleForecaster.
        
        Args:
            method: Ensemble combination method
            confidence_level: Confidence level for intervals
            parallel: Whether to fit models in parallel
            n_jobs: Number of parallel jobs (-1 for all CPUs)
        """
        self.method = method
        self.confidence_level = confidence_level
        self.parallel = parallel
        self.n_jobs = n_jobs
        
        self.models: List[BaseForecaster] = []
        self.weights: Dict[str, float] = {}
        self._is_fitted = False
        self._train_data: Optional[np.ndarray] = None
        self._validation_errors: Dict[str, float] = {}
    
    def add_model(self, model: BaseForecaster, weight: Optional[float] = None) -> 'EnsembleForecaster':
        """
        Add a model to the ensemble.
        
        Args:
            model: Forecaster model
            weight: Optional fixed weight (if None, will be computed)
        
        Returns:
            Self for chaining
        """
        self.models.append(model)
        if weight is not None:
            self.weights[model.name] = weight
        return self
    
    def add_default_models(self) -> 'EnsembleForecaster':
        """Add a default set of models to the ensemble."""
        self.add_model(ARIMAForecaster((1, 1, 1)))
        self.add_model(ARIMAForecaster((2, 1, 2)))
        self.add_model(ExponentialSmoothingForecaster(trend='add'))
        self.add_model(ExponentialSmoothingForecaster(trend='mul'))
        self.add_model(ThetaForecaster())
        self.add_model(NaiveForecaster())
        return self
    
    def fit(
        self,
        data: Union[np.ndarray, pd.Series],
        validation_split: float = 0.2
    ) -> 'EnsembleForecaster':
        """
        Fit all models in the ensemble.
        
        Args:
            data: Training data
            validation_split: Fraction of data for validation (to compute weights)
        
        Returns:
            Self for chaining
        """
        # Convert to numpy
        if isinstance(data, pd.Series):
            data = data.values
        
        self._train_data = data
        
        # Split for validation
        n = len(data)
        split_idx = int(n * (1 - validation_split))
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        val_steps = len(val_data)
        
        # Fit models
        if self.parallel and len(self.models) > 1:
            self._fit_parallel(train_data)
        else:
            self._fit_sequential(train_data)
        
        # Compute validation errors and weights
        if val_steps > 0 and self.method in [
            EnsembleMethod.WEIGHTED_AVERAGE,
            EnsembleMethod.BEST_MODEL
        ]:
            self._compute_weights(train_data, val_data, val_steps)
        else:
            # Equal weights
            for model in self.models:
                self.weights[model.name] = 1.0 / len(self.models)
        
        self._is_fitted = True
        return self
    
    def _fit_sequential(self, data: np.ndarray):
        """Fit models sequentially."""
        for model in self.models:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(data)
            except Exception as e:
                logger.warning(f"Failed to fit {model.name}: {e}")
    
    def _fit_parallel(self, data: np.ndarray):
        """Fit models in parallel."""
        n_jobs = self.n_jobs if self.n_jobs > 0 else len(self.models)
        
        def fit_model(model):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(data)
                return model.name, True
            except Exception as e:
                logger.warning(f"Failed to fit {model.name}: {e}")
                return model.name, False
        
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(fit_model, model) for model in self.models]
            for future in as_completed(futures):
                name, success = future.result()
                if not success:
                    logger.warning(f"Model {name} failed to fit")
    
    def _compute_weights(
        self,
        train_data: np.ndarray,
        val_data: np.ndarray,
        val_steps: int
    ):
        """Compute model weights based on validation performance."""
        errors = {}
        
        for model in self.models:
            try:
                # Re-fit on training data
                model.fit(train_data)
                
                # Predict validation period
                pred = model.predict(val_steps)
                
                if np.any(np.isnan(pred)):
                    errors[model.name] = float('inf')
                else:
                    # Calculate RMSE
                    rmse = np.sqrt(np.mean((val_data - pred) ** 2))
                    errors[model.name] = rmse
            except Exception as e:
                logger.warning(f"Validation failed for {model.name}: {e}")
                errors[model.name] = float('inf')
        
        self._validation_errors = errors
        
        # Compute weights (inverse of errors)
        valid_errors = {k: v for k, v in errors.items() if v < float('inf')}
        
        if not valid_errors:
            # Fall back to equal weights
            for model in self.models:
                self.weights[model.name] = 1.0 / len(self.models)
            return
        
        if self.method == EnsembleMethod.BEST_MODEL:
            # Winner takes all
            best_model = min(valid_errors, key=valid_errors.get)
            for model in self.models:
                self.weights[model.name] = 1.0 if model.name == best_model else 0.0
        else:
            # Inverse error weighting
            min_error = min(valid_errors.values())
            max_error = max(valid_errors.values())
            
            # Normalize errors to [0, 1] and invert
            for model in self.models:
                if model.name in valid_errors:
                    error = valid_errors[model.name]
                    if max_error > min_error:
                        normalized = (error - min_error) / (max_error - min_error)
                        self.weights[model.name] = 1 - normalized + 0.1  # Add smoothing
                    else:
                        self.weights[model.name] = 1.0
                else:
                    self.weights[model.name] = 0.0
            
            # Normalize weights to sum to 1
            total_weight = sum(self.weights.values())
            if total_weight > 0:
                for name in self.weights:
                    self.weights[name] /= total_weight
    
    def predict(self, steps: int) -> EnsembleForecastResult:
        """
        Generate ensemble forecast.
        
        Args:
            steps: Number of steps to forecast
        
        Returns:
            EnsembleForecastResult with predictions and metadata
        """
        if not self._is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        # Get individual forecasts
        individual_forecasts = []
        predictions_matrix = []
        
        for model in self.models:
            try:
                pred = model.predict(steps)
                weight = self.weights.get(model.name, 1.0 / len(self.models))
                
                forecast_result = ForecastResult(
                    model_name=model.name,
                    predictions=pred,
                    weight=weight,
                    metrics={"validation_rmse": self._validation_errors.get(model.name, np.nan)}
                )
                individual_forecasts.append(forecast_result)
                
                if not np.any(np.isnan(pred)) and weight > 0:
                    predictions_matrix.append((pred, weight))
            except Exception as e:
                logger.warning(f"Prediction failed for {model.name}: {e}")
        
        if not predictions_matrix:
            raise ValueError("All models failed to generate predictions")
        
        # Combine predictions based on method
        predictions, lower_bound, upper_bound = self._combine_predictions(
            predictions_matrix, steps
        )
        
        # Calculate ensemble metrics
        ensemble_metrics = self._calculate_ensemble_metrics(individual_forecasts)
        
        return EnsembleForecastResult(
            predictions=predictions,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            method=self.method,
            weights=self.weights.copy(),
            individual_forecasts=individual_forecasts,
            ensemble_metrics=ensemble_metrics,
            metadata={
                "n_models": len(self.models),
                "n_successful": len(predictions_matrix),
                "validation_errors": self._validation_errors.copy()
            }
        )
    
    def _combine_predictions(
        self,
        predictions_matrix: List[Tuple[np.ndarray, float]],
        steps: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Combine individual predictions using the ensemble method."""
        preds = np.array([p[0] for p in predictions_matrix])
        weights = np.array([p[1] for p in predictions_matrix])
        
        if self.method == EnsembleMethod.SIMPLE_AVERAGE:
            combined = np.mean(preds, axis=0)
        
        elif self.method == EnsembleMethod.WEIGHTED_AVERAGE:
            # Normalize weights
            weights = weights / weights.sum()
            combined = np.average(preds, axis=0, weights=weights)
        
        elif self.method == EnsembleMethod.MEDIAN:
            combined = np.median(preds, axis=0)
        
        elif self.method == EnsembleMethod.TRIMMED_MEAN:
            # Remove top and bottom 10%
            n_models = len(preds)
            trim = max(1, n_models // 10)
            sorted_preds = np.sort(preds, axis=0)
            combined = np.mean(sorted_preds[trim:-trim or None], axis=0)
        
        elif self.method == EnsembleMethod.BEST_MODEL:
            best_idx = np.argmax(weights)
            combined = preds[best_idx]
        
        else:
            # Default to weighted average
            weights = weights / weights.sum()
            combined = np.average(preds, axis=0, weights=weights)
        
        # Calculate confidence intervals from prediction spread
        std = np.std(preds, axis=0)
        z_score = 1.96 if self.confidence_level == 0.95 else 2.576
        lower_bound = combined - z_score * std
        upper_bound = combined + z_score * std
        
        return combined, lower_bound, upper_bound
    
    def _calculate_ensemble_metrics(
        self,
        individual_forecasts: List[ForecastResult]
    ) -> Dict[str, float]:
        """Calculate ensemble-level metrics."""
        valid_forecasts = [
            f for f in individual_forecasts
            if not np.any(np.isnan(f.predictions))
        ]
        
        if not valid_forecasts:
            return {}
        
        # Diversity metric (average pairwise correlation)
        if len(valid_forecasts) >= 2:
            correlations = []
            for i, f1 in enumerate(valid_forecasts):
                for f2 in valid_forecasts[i+1:]:
                    corr = np.corrcoef(f1.predictions, f2.predictions)[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
            
            diversity = 1 - np.mean(correlations) if correlations else 0
        else:
            diversity = 0
        
        # Average validation RMSE
        valid_rmses = [
            f.metrics.get("validation_rmse", np.nan)
            for f in valid_forecasts
            if not np.isnan(f.metrics.get("validation_rmse", np.nan))
        ]
        avg_rmse = np.mean(valid_rmses) if valid_rmses else np.nan
        
        return {
            "diversity_score": float(diversity),
            "n_models_used": len(valid_forecasts),
            "avg_validation_rmse": float(avg_rmse) if not np.isnan(avg_rmse) else None,
            "weight_concentration": float(np.max(list(self.weights.values())))
        }
    
    def fit_predict(
        self,
        data: Union[np.ndarray, pd.Series],
        steps: int,
        validation_split: float = 0.2
    ) -> EnsembleForecastResult:
        """
        Fit ensemble and generate predictions in one step.
        
        Args:
            data: Training data
            steps: Number of steps to forecast
            validation_split: Fraction for validation
        
        Returns:
            EnsembleForecastResult
        """
        self.fit(data, validation_split)
        return self.predict(steps)


class MedianEnsemble(EnsembleForecaster):
    """Convenience class for median ensemble."""
    
    def __init__(self, **kwargs):
        super().__init__(method=EnsembleMethod.MEDIAN, **kwargs)


class WeightedEnsemble(EnsembleForecaster):
    """Convenience class for weighted average ensemble."""
    
    def __init__(self, **kwargs):
        super().__init__(method=EnsembleMethod.WEIGHTED_AVERAGE, **kwargs)


class StackingEnsemble(EnsembleForecaster):
    """
    Stacking ensemble using a meta-learner.
    
    The meta-learner learns how to combine base model predictions
    using a second-level model (e.g., linear regression).
    """
    
    def __init__(
        self,
        meta_learner: Optional[Any] = None,
        **kwargs
    ):
        """
        Initialize StackingEnsemble.
        
        Args:
            meta_learner: Scikit-learn compatible regressor for stacking
            **kwargs: Additional arguments for EnsembleForecaster
        """
        super().__init__(method=EnsembleMethod.STACKING, **kwargs)
        
        if meta_learner is None:
            from sklearn.linear_model import Ridge
            self.meta_learner = Ridge(alpha=1.0)
        else:
            self.meta_learner = meta_learner
        
        self._meta_fitted = False
    
    def fit(
        self,
        data: Union[np.ndarray, pd.Series],
        validation_split: float = 0.2
    ) -> 'StackingEnsemble':
        """Fit base models and meta-learner."""
        # Convert to numpy
        if isinstance(data, pd.Series):
            data = data.values
        
        self._train_data = data
        
        # Split data
        n = len(data)
        split_idx = int(n * (1 - validation_split))
        train_data = data[:split_idx]
        val_data = data[split_idx:]
        val_steps = len(val_data)
        
        # Fit base models on training data
        for model in self.models:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(train_data)
            except Exception as e:
                logger.warning(f"Failed to fit {model.name}: {e}")
        
        # Generate predictions for validation set to train meta-learner
        if val_steps > 0:
            X_meta = []
            for model in self.models:
                try:
                    pred = model.predict(val_steps)
                    if not np.any(np.isnan(pred)):
                        X_meta.append(pred)
                except Exception:
                    pass
            
            if X_meta:
                X_meta = np.column_stack(X_meta)
                
                # Fit meta-learner
                try:
                    self.meta_learner.fit(X_meta, val_data)
                    self._meta_fitted = True
                except Exception as e:
                    logger.warning(f"Meta-learner fit failed: {e}")
                    self._meta_fitted = False
        
        # Re-fit base models on full data
        for model in self.models:
            try:
                model.fit(data)
            except Exception:
                pass
        
        self._is_fitted = True
        return self
    
    def predict(self, steps: int) -> EnsembleForecastResult:
        """Generate stacking ensemble predictions."""
        if not self._is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")
        
        # Get base model predictions
        individual_forecasts = []
        X_stack = []
        
        for model in self.models:
            try:
                pred = model.predict(steps)
                
                forecast_result = ForecastResult(
                    model_name=model.name,
                    predictions=pred,
                    weight=1.0 / len(self.models)
                )
                individual_forecasts.append(forecast_result)
                
                if not np.any(np.isnan(pred)):
                    X_stack.append(pred)
            except Exception as e:
                logger.warning(f"Prediction failed for {model.name}: {e}")
        
        if not X_stack:
            raise ValueError("All base models failed")
        
        X_stack = np.column_stack(X_stack)
        
        # Use meta-learner if fitted, otherwise fall back to average
        if self._meta_fitted:
            try:
                predictions = self.meta_learner.predict(X_stack)
            except Exception:
                predictions = np.mean(X_stack, axis=1)
        else:
            predictions = np.mean(X_stack, axis=1)
        
        # Calculate uncertainty from base model spread
        std = np.std(X_stack, axis=1)
        z_score = 1.96
        lower_bound = predictions - z_score * std
        upper_bound = predictions + z_score * std
        
        return EnsembleForecastResult(
            predictions=predictions,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            method=self.method,
            weights={m.name: 1/len(self.models) for m in self.models},
            individual_forecasts=individual_forecasts,
            ensemble_metrics={"stacking_used": self._meta_fitted}
        )


# Convenience functions
def ensemble_forecast(
    data: Union[np.ndarray, pd.Series],
    steps: int,
    method: EnsembleMethod = EnsembleMethod.WEIGHTED_AVERAGE,
    models: Optional[List[BaseForecaster]] = None,
    validation_split: float = 0.2
) -> EnsembleForecastResult:
    """
    Quick function for ensemble forecasting.
    
    Args:
        data: Time series data
        steps: Number of steps to forecast
        method: Ensemble method
        models: List of models (if None, uses default set)
        validation_split: Fraction for validation
    
    Returns:
        EnsembleForecastResult
    """
    ensemble = EnsembleForecaster(method=method)
    
    if models:
        for model in models:
            ensemble.add_model(model)
    else:
        ensemble.add_default_models()
    
    return ensemble.fit_predict(data, steps, validation_split)


def weighted_ensemble_forecast(
    data: Union[np.ndarray, pd.Series],
    steps: int,
    validation_split: float = 0.2
) -> EnsembleForecastResult:
    """Quick function for weighted average ensemble."""
    return ensemble_forecast(
        data, steps,
        method=EnsembleMethod.WEIGHTED_AVERAGE,
        validation_split=validation_split
    )


def median_ensemble_forecast(
    data: Union[np.ndarray, pd.Series],
    steps: int
) -> EnsembleForecastResult:
    """Quick function for median ensemble."""
    return ensemble_forecast(data, steps, method=EnsembleMethod.MEDIAN)
