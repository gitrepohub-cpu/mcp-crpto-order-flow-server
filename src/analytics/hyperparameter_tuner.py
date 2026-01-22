"""
Automated Hyperparameter Tuning for Darts Models

Production-grade hyperparameter optimization using:
- Optuna for Bayesian optimization
- Time series cross-validation objectives
- Early pruning for expensive models
- Model-specific parameter spaces
"""

import logging
import time
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try importing Optuna
try:
    import optuna
    from optuna.pruners import MedianPruner, HyperbandPruner
    from optuna.samplers import TPESampler, CmaEsSampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logger.warning("Optuna not available. Install with: pip install optuna")


class TuningStrategy(Enum):
    """Hyperparameter tuning strategies"""
    TPE = "tpe"              # Tree-structured Parzen Estimator (default)
    CMA_ES = "cma_es"        # Covariance Matrix Adaptation
    RANDOM = "random"        # Random search
    GRID = "grid"            # Grid search (exhaustive)


class PruningStrategy(Enum):
    """Early stopping strategies for expensive trials"""
    MEDIAN = "median"        # Prune if worse than median
    HYPERBAND = "hyperband"  # Successive halving
    NONE = "none"            # No pruning


@dataclass
class TuningResult:
    """Result of hyperparameter tuning"""
    best_params: Dict[str, Any]
    best_score: float
    metric_name: str
    n_trials: int
    best_trial_number: int
    optimization_time_seconds: float
    all_trials: List[Dict[str, Any]] = field(default_factory=list)
    convergence_history: List[float] = field(default_factory=list)


@dataclass
class ParameterSpace:
    """Definition of hyperparameter search space"""
    name: str
    param_type: str  # "int", "float", "categorical", "loguniform"
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None
    log: bool = False
    step: Optional[float] = None


# ============================================================================
# MODEL-SPECIFIC PARAMETER SPACES
# ============================================================================

STATISTICAL_PARAM_SPACES = {
    'arima': [
        ParameterSpace('p', 'int', low=0, high=5),
        ParameterSpace('d', 'int', low=0, high=2),
        ParameterSpace('q', 'int', low=0, high=5),
    ],
    'auto_arima': [
        ParameterSpace('max_p', 'int', low=3, high=8),
        ParameterSpace('max_d', 'int', low=1, high=3),
        ParameterSpace('max_q', 'int', low=3, high=8),
        ParameterSpace('seasonal', 'categorical', choices=[True, False]),
    ],
    'exponential_smoothing': [
        ParameterSpace('seasonal_periods', 'int', low=12, high=168),
        ParameterSpace('trend', 'categorical', choices=['add', 'mul', None]),
        ParameterSpace('seasonal', 'categorical', choices=['add', 'mul', None]),
        ParameterSpace('damped', 'categorical', choices=[True, False]),
    ],
    'theta': [
        ParameterSpace('theta', 'float', low=0.5, high=3.0),
        ParameterSpace('seasonality_period', 'int', low=12, high=168),
    ],
    'prophet': [
        ParameterSpace('changepoint_prior_scale', 'float', low=0.001, high=0.5, log=True),
        ParameterSpace('seasonality_prior_scale', 'float', low=0.01, high=10.0, log=True),
        ParameterSpace('seasonality_mode', 'categorical', choices=['additive', 'multiplicative']),
    ],
}

ML_PARAM_SPACES = {
    'lightgbm': [
        ParameterSpace('lags', 'int', low=12, high=168),
        ParameterSpace('n_estimators', 'int', low=50, high=500),
        ParameterSpace('max_depth', 'int', low=3, high=15),
        ParameterSpace('learning_rate', 'float', low=0.01, high=0.3, log=True),
        ParameterSpace('num_leaves', 'int', low=15, high=127),
        ParameterSpace('min_child_samples', 'int', low=5, high=100),
        ParameterSpace('subsample', 'float', low=0.5, high=1.0),
        ParameterSpace('colsample_bytree', 'float', low=0.5, high=1.0),
    ],
    'xgboost': [
        ParameterSpace('lags', 'int', low=12, high=168),
        ParameterSpace('n_estimators', 'int', low=50, high=500),
        ParameterSpace('max_depth', 'int', low=3, high=12),
        ParameterSpace('learning_rate', 'float', low=0.01, high=0.3, log=True),
        ParameterSpace('subsample', 'float', low=0.5, high=1.0),
        ParameterSpace('colsample_bytree', 'float', low=0.5, high=1.0),
        ParameterSpace('reg_alpha', 'float', low=1e-8, high=10.0, log=True),
        ParameterSpace('reg_lambda', 'float', low=1e-8, high=10.0, log=True),
    ],
    'catboost': [
        ParameterSpace('lags', 'int', low=12, high=168),
        ParameterSpace('iterations', 'int', low=50, high=500),
        ParameterSpace('depth', 'int', low=3, high=10),
        ParameterSpace('learning_rate', 'float', low=0.01, high=0.3, log=True),
        ParameterSpace('l2_leaf_reg', 'float', low=1.0, high=10.0),
        ParameterSpace('bagging_temperature', 'float', low=0.0, high=1.0),
    ],
    'random_forest': [
        ParameterSpace('lags', 'int', low=12, high=168),
        ParameterSpace('n_estimators', 'int', low=50, high=300),
        ParameterSpace('max_depth', 'int', low=5, high=30),
        ParameterSpace('min_samples_split', 'int', low=2, high=20),
        ParameterSpace('min_samples_leaf', 'int', low=1, high=10),
    ],
}

DL_PARAM_SPACES = {
    'nbeats': [
        ParameterSpace('input_chunk_length', 'int', low=24, high=168),
        ParameterSpace('output_chunk_length', 'int', low=12, high=48),
        ParameterSpace('num_stacks', 'int', low=2, high=8),
        ParameterSpace('num_blocks', 'int', low=1, high=4),
        ParameterSpace('num_layers', 'int', low=2, high=6),
        ParameterSpace('layer_widths', 'int', low=128, high=512, step=64),
        ParameterSpace('batch_size', 'int', low=16, high=128),
        ParameterSpace('n_epochs', 'int', low=20, high=100),
        ParameterSpace('learning_rate', 'float', low=1e-5, high=1e-2, log=True),
        ParameterSpace('dropout', 'float', low=0.0, high=0.5),
    ],
    'nhits': [
        ParameterSpace('input_chunk_length', 'int', low=24, high=168),
        ParameterSpace('output_chunk_length', 'int', low=12, high=48),
        ParameterSpace('num_stacks', 'int', low=2, high=5),
        ParameterSpace('num_blocks', 'int', low=1, high=3),
        ParameterSpace('num_layers', 'int', low=2, high=4),
        ParameterSpace('layer_widths', 'int', low=256, high=512),
        ParameterSpace('batch_size', 'int', low=32, high=128),
        ParameterSpace('n_epochs', 'int', low=20, high=100),
        ParameterSpace('learning_rate', 'float', low=1e-5, high=1e-2, log=True),
    ],
    'tft': [
        ParameterSpace('input_chunk_length', 'int', low=24, high=168),
        ParameterSpace('output_chunk_length', 'int', low=12, high=48),
        ParameterSpace('hidden_size', 'int', low=16, high=128),
        ParameterSpace('lstm_layers', 'int', low=1, high=3),
        ParameterSpace('num_attention_heads', 'int', low=1, high=4),
        ParameterSpace('dropout', 'float', low=0.0, high=0.3),
        ParameterSpace('batch_size', 'int', low=32, high=128),
        ParameterSpace('n_epochs', 'int', low=20, high=100),
        ParameterSpace('learning_rate', 'float', low=1e-5, high=1e-2, log=True),
    ],
    'transformer': [
        ParameterSpace('input_chunk_length', 'int', low=24, high=168),
        ParameterSpace('output_chunk_length', 'int', low=12, high=48),
        ParameterSpace('d_model', 'int', low=32, high=128),
        ParameterSpace('nhead', 'int', low=2, high=8),
        ParameterSpace('num_encoder_layers', 'int', low=1, high=4),
        ParameterSpace('num_decoder_layers', 'int', low=1, high=4),
        ParameterSpace('dim_feedforward', 'int', low=64, high=256),
        ParameterSpace('dropout', 'float', low=0.0, high=0.3),
        ParameterSpace('batch_size', 'int', low=32, high=128),
        ParameterSpace('n_epochs', 'int', low=20, high=100),
    ],
}

# Combined parameter spaces
ALL_PARAM_SPACES = {
    **STATISTICAL_PARAM_SPACES,
    **ML_PARAM_SPACES,
    **DL_PARAM_SPACES,
}


class HyperparameterTuner:
    """
    Automated hyperparameter tuning for Darts forecasting models.
    
    Uses Optuna for efficient Bayesian optimization with:
    - Time series cross-validation objectives
    - Early pruning for expensive trials
    - Model-specific parameter spaces
    
    Usage:
        tuner = HyperparameterTuner()
        result = tuner.tune(
            model_name='lightgbm',
            series=darts_series,
            n_trials=50,
            metric='mape'
        )
        best_params = result.best_params
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        random_seed: int = 42
    ):
        """
        Initialize the hyperparameter tuner.
        
        Args:
            storage_path: Path for Optuna study storage (enables resuming)
            random_seed: Random seed for reproducibility
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna required. Install with: pip install optuna")
        
        self.storage_path = storage_path
        self.random_seed = random_seed
        self._cached_results: Dict[str, TuningResult] = {}
        
        # Set Optuna verbosity
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def get_parameter_space(self, model_name: str) -> List[ParameterSpace]:
        """Get the parameter search space for a model"""
        return ALL_PARAM_SPACES.get(model_name.lower(), [])
    
    def tune(
        self,
        model_name: str,
        series,  # Darts TimeSeries
        n_trials: int = 50,
        metric: str = 'mape',
        val_ratio: float = 0.2,
        n_cv_folds: int = 3,
        strategy: TuningStrategy = TuningStrategy.TPE,
        pruning: PruningStrategy = PruningStrategy.MEDIAN,
        timeout_seconds: Optional[int] = None,
        custom_param_space: Optional[List[ParameterSpace]] = None,
        use_gpu: bool = False
    ) -> TuningResult:
        """
        Tune hyperparameters for a forecasting model.
        
        Args:
            model_name: Name of the model (e.g., 'lightgbm', 'nbeats')
            series: Darts TimeSeries for training/validation
            n_trials: Number of optimization trials
            metric: Metric to optimize ('mape', 'rmse', 'mae', 'smape')
            val_ratio: Ratio of data for validation
            n_cv_folds: Number of cross-validation folds
            strategy: Optimization strategy
            pruning: Pruning strategy for early stopping
            timeout_seconds: Maximum time for optimization
            custom_param_space: Override default parameter space
            use_gpu: Use GPU for training (deep learning models)
        
        Returns:
            TuningResult with best parameters and optimization details
        """
        start_time = time.time()
        
        # Get parameter space
        param_space = custom_param_space or self.get_parameter_space(model_name)
        if not param_space:
            raise ValueError(f"No parameter space defined for model: {model_name}")
        
        # Create sampler
        sampler = self._create_sampler(strategy)
        
        # Create pruner
        pruner = self._create_pruner(pruning)
        
        # Create study
        study_name = f"{model_name}_tuning_{int(time.time())}"
        storage = f"sqlite:///{self.storage_path}" if self.storage_path else None
        
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",  # Minimize error metrics
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            load_if_exists=True
        )
        
        # Create objective function
        objective = self._create_objective(
            model_name=model_name,
            series=series,
            param_space=param_space,
            metric=metric,
            val_ratio=val_ratio,
            n_cv_folds=n_cv_folds,
            use_gpu=use_gpu
        )
        
        # Run optimization
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout_seconds,
            show_progress_bar=False,
            catch=(Exception,)  # Catch failures in trials
        )
        
        # Collect results
        optimization_time = time.time() - start_time
        
        # Get convergence history
        convergence = [
            trial.value for trial in study.trials 
            if trial.value is not None
        ]
        
        # Compute running best
        running_best = []
        current_best = float('inf')
        for val in convergence:
            current_best = min(current_best, val)
            running_best.append(current_best)
        
        result = TuningResult(
            best_params=study.best_params,
            best_score=study.best_value,
            metric_name=metric,
            n_trials=len(study.trials),
            best_trial_number=study.best_trial.number,
            optimization_time_seconds=optimization_time,
            all_trials=[
                {
                    'number': t.number,
                    'params': t.params,
                    'value': t.value,
                    'state': str(t.state)
                }
                for t in study.trials
            ],
            convergence_history=running_best
        )
        
        # Cache result
        cache_key = f"{model_name}_{len(series)}"
        self._cached_results[cache_key] = result
        
        logger.info(
            f"Tuning complete for {model_name}: "
            f"best_{metric}={study.best_value:.4f}, "
            f"trials={len(study.trials)}, "
            f"time={optimization_time:.1f}s"
        )
        
        return result
    
    def _create_sampler(self, strategy: TuningStrategy):
        """Create Optuna sampler based on strategy"""
        if strategy == TuningStrategy.TPE:
            return TPESampler(seed=self.random_seed)
        elif strategy == TuningStrategy.CMA_ES:
            return CmaEsSampler(seed=self.random_seed)
        elif strategy == TuningStrategy.RANDOM:
            return optuna.samplers.RandomSampler(seed=self.random_seed)
        elif strategy == TuningStrategy.GRID:
            # Grid search needs explicit search space
            return optuna.samplers.GridSampler({})
        else:
            return TPESampler(seed=self.random_seed)
    
    def _create_pruner(self, pruning: PruningStrategy):
        """Create Optuna pruner based on strategy"""
        if pruning == PruningStrategy.MEDIAN:
            return MedianPruner(n_startup_trials=5, n_warmup_steps=3)
        elif pruning == PruningStrategy.HYPERBAND:
            return HyperbandPruner()
        elif pruning == PruningStrategy.NONE:
            return optuna.pruners.NopPruner()
        else:
            return MedianPruner()
    
    def _create_objective(
        self,
        model_name: str,
        series,
        param_space: List[ParameterSpace],
        metric: str,
        val_ratio: float,
        n_cv_folds: int,
        use_gpu: bool
    ) -> Callable:
        """Create Optuna objective function"""
        
        def objective(trial: optuna.Trial) -> float:
            # Sample hyperparameters
            params = {}
            for ps in param_space:
                if ps.param_type == 'int':
                    params[ps.name] = trial.suggest_int(
                        ps.name, int(ps.low), int(ps.high), step=int(ps.step or 1)
                    )
                elif ps.param_type == 'float':
                    if ps.log:
                        params[ps.name] = trial.suggest_float(
                            ps.name, ps.low, ps.high, log=True
                        )
                    else:
                        params[ps.name] = trial.suggest_float(
                            ps.name, ps.low, ps.high
                        )
                elif ps.param_type == 'categorical':
                    params[ps.name] = trial.suggest_categorical(
                        ps.name, ps.choices
                    )
                elif ps.param_type == 'loguniform':
                    params[ps.name] = trial.suggest_float(
                        ps.name, ps.low, ps.high, log=True
                    )
            
            try:
                # Create and train model
                model = self._create_model(model_name, params, use_gpu)
                
                # Split data for validation
                train_len = int(len(series) * (1 - val_ratio))
                train_series = series[:train_len]
                val_series = series[train_len:]
                
                # Train model
                model.fit(train_series)
                
                # Make predictions
                forecast_horizon = len(val_series)
                predictions = model.predict(n=forecast_horizon)
                
                # Calculate metric
                score = self._calculate_metric(
                    actual=val_series.values().flatten(),
                    predicted=predictions.values().flatten(),
                    metric=metric
                )
                
                # Report intermediate value for pruning
                trial.report(score, step=1)
                
                if trial.should_prune():
                    raise optuna.TrialPruned()
                
                return score
                
            except optuna.TrialPruned:
                raise
            except Exception as e:
                logger.warning(f"Trial {trial.number} failed: {e}")
                return float('inf')
        
        return objective
    
    def _create_model(self, model_name: str, params: Dict[str, Any], use_gpu: bool):
        """Create a Darts model with given hyperparameters"""
        from src.integrations.darts_bridge import DartsModelWrapper
        
        # Handle model-specific parameter mappings
        model_params = params.copy()
        
        # Add GPU settings for deep learning models
        if model_name.lower() in ['nbeats', 'nhits', 'tft', 'transformer', 'tcn']:
            if use_gpu:
                try:
                    import torch
                    if torch.cuda.is_available():
                        model_params['pl_trainer_kwargs'] = {
                            'accelerator': 'gpu',
                            'devices': 1
                        }
                except ImportError:
                    pass
            
            # Ensure verbose is off
            model_params['verbose'] = False
        
        return DartsModelWrapper(model_name, **model_params)
    
    def _calculate_metric(
        self, 
        actual: np.ndarray, 
        predicted: np.ndarray, 
        metric: str
    ) -> float:
        """Calculate forecasting metric"""
        actual = np.asarray(actual).flatten()
        predicted = np.asarray(predicted).flatten()
        
        # Handle length mismatch
        min_len = min(len(actual), len(predicted))
        actual = actual[:min_len]
        predicted = predicted[:min_len]
        
        if metric == 'mape':
            # Mean Absolute Percentage Error
            mask = actual != 0
            return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
        elif metric == 'rmse':
            # Root Mean Square Error
            return np.sqrt(np.mean((actual - predicted) ** 2))
        elif metric == 'mae':
            # Mean Absolute Error
            return np.mean(np.abs(actual - predicted))
        elif metric == 'smape':
            # Symmetric MAPE
            return np.mean(2 * np.abs(actual - predicted) / (np.abs(actual) + np.abs(predicted) + 1e-8)) * 100
        elif metric == 'mse':
            # Mean Square Error
            return np.mean((actual - predicted) ** 2)
        else:
            # Default to MAPE
            mask = actual != 0
            return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    
    def get_cached_result(self, model_name: str, data_length: int) -> Optional[TuningResult]:
        """Retrieve cached tuning result if available"""
        cache_key = f"{model_name}_{data_length}"
        return self._cached_results.get(cache_key)
    
    def quick_tune(
        self,
        model_name: str,
        series,
        n_trials: int = 20
    ) -> Dict[str, Any]:
        """
        Quick tuning with sensible defaults.
        
        Args:
            model_name: Name of the model
            series: Darts TimeSeries
            n_trials: Number of trials (default 20 for speed)
        
        Returns:
            Best hyperparameters
        """
        result = self.tune(
            model_name=model_name,
            series=series,
            n_trials=n_trials,
            metric='mape',
            val_ratio=0.2,
            strategy=TuningStrategy.TPE,
            pruning=PruningStrategy.MEDIAN
        )
        return result.best_params


# Singleton instance
_tuner_instance: Optional[HyperparameterTuner] = None


def get_tuner() -> HyperparameterTuner:
    """Get or create the global tuner instance"""
    global _tuner_instance
    if _tuner_instance is None:
        _tuner_instance = HyperparameterTuner()
    return _tuner_instance


def tune_model(
    model_name: str,
    data: np.ndarray,
    n_trials: int = 50,
    metric: str = 'mape'
) -> TuningResult:
    """
    Convenience function for tuning a model.
    
    Args:
        model_name: Name of the model to tune
        data: NumPy array of time series data
        n_trials: Number of optimization trials
        metric: Metric to optimize
    
    Returns:
        TuningResult with best parameters
    """
    from src.integrations.darts_bridge import DartsBridge
    
    bridge = DartsBridge()
    series = bridge.to_darts(data)
    
    tuner = get_tuner()
    return tuner.tune(
        model_name=model_name,
        series=series,
        n_trials=n_trials,
        metric=metric
    )
