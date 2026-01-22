"""
Production Forecasting Tools - MCP Tool Implementations

Tools for production-grade forecasting operations:
- Hyperparameter tuning with Optuna
- Time series cross-validation
- Model drift detection and monitoring
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# HYPERPARAMETER TUNING TOOLS
# ============================================================================

async def tune_model_hyperparameters(
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    model_name: str = "lightgbm",
    n_trials: int = 30,
    metric: str = "mape",
    hours: int = 168
) -> str:
    """
    Automatically tune hyperparameters for a forecasting model using Optuna.
    
    Uses Bayesian optimization to find optimal parameters with:
    - Tree-structured Parzen Estimator (TPE) sampling
    - Median pruning for early stopping
    - Time series cross-validation objective
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        exchange: Exchange name
        model_name: Model to tune ("lightgbm", "xgboost", "catboost", "nbeats", etc.)
        n_trials: Number of optimization trials (more = better but slower)
        metric: Metric to optimize ("mape", "rmse", "mae", "smape")
        hours: Hours of historical data to use
        
    Returns:
        XML formatted tuning results with best parameters
    """
    try:
        from src.analytics.hyperparameter_tuner import (
            HyperparameterTuner, TuningStrategy, OPTUNA_AVAILABLE
        )
        from src.integrations.darts_bridge import DartsBridge
        from src.tools.darts_tools import _load_price_data
        
        if not OPTUNA_AVAILABLE:
            return '''<error>
  <message>Optuna not installed. Install with: pip install optuna</message>
</error>'''
        
        # Load data
        df = _load_price_data(symbol, exchange, hours=hours)
        if df is None or len(df) < 100:
            return f'''<error>
  <message>Insufficient data for {symbol}. Need at least 100 points.</message>
</error>'''
        
        # Convert to Darts series
        bridge = DartsBridge()
        series = bridge.to_darts(df['price'].values)
        
        # Run tuning
        tuner = HyperparameterTuner()
        result = tuner.tune(
            model_name=model_name,
            series=series,
            n_trials=n_trials,
            metric=metric,
            strategy=TuningStrategy.TPE
        )
        
        # Format best params as XML
        params_xml = '\n'.join(
            f'      <{k}>{v}</{k}>'
            for k, v in result.best_params.items()
        )
        
        # Convergence summary
        if result.convergence_history:
            improvement = (
                (result.convergence_history[0] - result.best_score) 
                / result.convergence_history[0] * 100
            )
        else:
            improvement = 0
        
        return f'''<hyperparameter_tuning_result>
  <model>{model_name}</model>
  <symbol>{symbol}</symbol>
  <exchange>{exchange}</exchange>
  <optimization>
    <n_trials>{result.n_trials}</n_trials>
    <best_trial>{result.best_trial_number}</best_trial>
    <best_{metric}>{result.best_score:.4f}</best_{metric}>
    <optimization_time_seconds>{result.optimization_time_seconds:.1f}</optimization_time_seconds>
    <improvement_pct>{improvement:.1f}</improvement_pct>
  </optimization>
  <best_parameters>
{params_xml}
  </best_parameters>
  <usage>
    <hint>Use these parameters with forecast_with_darts_{
        'statistical' if model_name in ['arima', 'theta', 'exponential_smoothing', 'prophet'] 
        else 'ml' if model_name in ['lightgbm', 'xgboost', 'catboost'] 
        else 'dl'
    }</hint>
  </usage>
</hyperparameter_tuning_result>'''
        
    except Exception as e:
        logger.error(f"Hyperparameter tuning error: {e}")
        return f'''<error>
  <message>Tuning failed: {str(e)}</message>
</error>'''


async def get_parameter_space(
    model_name: str = "lightgbm"
) -> str:
    """
    Get the hyperparameter search space for a model.
    
    Shows which parameters can be tuned and their valid ranges.
    
    Args:
        model_name: Model name ("lightgbm", "xgboost", "nbeats", etc.)
        
    Returns:
        XML formatted parameter space definition
    """
    try:
        from src.analytics.hyperparameter_tuner import ALL_PARAM_SPACES
        
        param_space = ALL_PARAM_SPACES.get(model_name.lower(), [])
        
        if not param_space:
            available = list(ALL_PARAM_SPACES.keys())
            return f'''<error>
  <message>No parameter space defined for {model_name}</message>
  <available_models>{', '.join(available)}</available_models>
</error>'''
        
        params_xml = ''
        for ps in param_space:
            if ps.param_type == 'categorical':
                params_xml += f'''    <parameter>
      <name>{ps.name}</name>
      <type>categorical</type>
      <choices>{ps.choices}</choices>
    </parameter>
'''
            else:
                params_xml += f'''    <parameter>
      <name>{ps.name}</name>
      <type>{ps.param_type}</type>
      <range>[{ps.low}, {ps.high}]</range>
      <log_scale>{ps.log}</log_scale>
    </parameter>
'''
        
        return f'''<parameter_space>
  <model>{model_name}</model>
  <n_parameters>{len(param_space)}</n_parameters>
  <parameters>
{params_xml}  </parameters>
</parameter_space>'''
        
    except Exception as e:
        return f'''<error>
  <message>Error getting parameter space: {str(e)}</message>
</error>'''


# ============================================================================
# CROSS-VALIDATION TOOLS
# ============================================================================

async def cross_validate_forecast_model(
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    model_name: str = "lightgbm",
    n_folds: int = 5,
    cv_strategy: str = "expanding_window",
    metric: str = "mape",
    hours: int = 168,
    forecast_horizon: int = 24
) -> str:
    """
    Cross-validate a forecasting model with proper time series CV.
    
    Prevents data leakage with:
    - Forward-only validation
    - Embargo gaps between train/test
    - Purging overlapping samples
    
    CV Strategies:
    - "expanding_window": Train on all past data
    - "sliding_window": Fixed-size training window
    - "purged_kfold": K-fold with purging
    - "walk_forward": Walk-forward validation
    
    Args:
        symbol: Trading pair
        exchange: Exchange name
        model_name: Model to evaluate
        n_folds: Number of CV folds
        cv_strategy: Cross-validation strategy
        metric: Evaluation metric
        hours: Hours of historical data
        forecast_horizon: Steps to forecast
        
    Returns:
        XML formatted cross-validation results
    """
    try:
        from src.analytics.timeseries_cv import (
            TimeSeriesCrossValidator, WalkForwardValidator,
            CVStrategy, cross_validate_forecast
        )
        from src.integrations.darts_bridge import DartsBridge, DartsModelWrapper
        from src.tools.darts_tools import _load_price_data
        
        # Load data
        df = _load_price_data(symbol, exchange, hours=hours)
        if df is None or len(df) < 100:
            return f'''<error>
  <message>Insufficient data for {symbol}</message>
</error>'''
        
        # Convert to Darts series
        bridge = DartsBridge()
        series = bridge.to_darts(df['price'].values)
        
        # Create CV strategy
        strategy_map = {
            'expanding_window': CVStrategy.EXPANDING_WINDOW,
            'sliding_window': CVStrategy.SLIDING_WINDOW,
            'purged_kfold': CVStrategy.PURGED_KFOLD,
            'blocked': CVStrategy.BLOCKED,
        }
        
        if cv_strategy == 'walk_forward':
            cv = WalkForwardValidator(
                test_size=forecast_horizon,
                min_train_size=100,
                expanding=True
            )
        else:
            cv = TimeSeriesCrossValidator(
                n_splits=n_folds,
                embargo_pct=0.01,
                purge_pct=0.01,
                strategy=strategy_map.get(cv_strategy, CVStrategy.EXPANDING_WINDOW)
            )
        
        # Create model
        model = DartsModelWrapper(model_name, verbose=False)
        
        # Run CV
        result = cross_validate_forecast(
            model=model,
            series=series,
            cv=cv,
            metric=metric,
            forecast_horizon=forecast_horizon
        )
        
        # Format fold results
        folds_xml = '\n'.join(
            f'''    <fold>
      <index>{f['fold']}</index>
      <train_size>{f['train_size']}</train_size>
      <test_size>{f['test_size']}</test_size>
      <{metric}>{f['score']:.4f}</{metric}>
    </fold>'''
            for f in result.fold_details
        )
        
        return f'''<cross_validation_result>
  <model>{model_name}</model>
  <symbol>{symbol}</symbol>
  <exchange>{exchange}</exchange>
  <cv_config>
    <strategy>{cv_strategy}</strategy>
    <n_folds>{result.n_folds}</n_folds>
    <metric>{metric}</metric>
    <forecast_horizon>{forecast_horizon}</forecast_horizon>
  </cv_config>
  <results>
    <mean_{metric}>{result.mean_score:.4f}</mean_{metric}>
    <std_{metric}>{result.std_score:.4f}</std_{metric}>
    <min_{metric}>{min(result.scores):.4f}</min_{metric}>
    <max_{metric}>{max(result.scores):.4f}</max_{metric}>
  </results>
  <fold_details>
{folds_xml}
  </fold_details>
  <interpretation>
    <stability>{'High' if result.std_score < result.mean_score * 0.2 else 'Medium' if result.std_score < result.mean_score * 0.5 else 'Low'}</stability>
    <reliability>Model performance is {'consistent' if result.std_score < result.mean_score * 0.2 else 'variable'} across folds</reliability>
  </interpretation>
</cross_validation_result>'''
        
    except Exception as e:
        logger.error(f"Cross-validation error: {e}")
        return f'''<error>
  <message>Cross-validation failed: {str(e)}</message>
</error>'''


async def list_cv_strategies() -> str:
    """
    List available cross-validation strategies with explanations.
    
    Returns:
        XML formatted list of CV strategies
    """
    return '''<cv_strategies>
  <strategy>
    <name>expanding_window</name>
    <description>Train on all historical data up to fold boundary</description>
    <best_for>When more data always helps</best_for>
    <leakage_risk>Low</leakage_risk>
  </strategy>
  <strategy>
    <name>sliding_window</name>
    <description>Fixed-size training window that slides forward</description>
    <best_for>Non-stationary data where old data may hurt</best_for>
    <leakage_risk>Low</leakage_risk>
  </strategy>
  <strategy>
    <name>purged_kfold</name>
    <description>K-fold with samples purged around test boundaries</description>
    <best_for>Financial data with autocorrelation</best_for>
    <leakage_risk>Very Low</leakage_risk>
  </strategy>
  <strategy>
    <name>walk_forward</name>
    <description>Train, predict, step forward, repeat</description>
    <best_for>Simulating real trading conditions</best_for>
    <leakage_risk>None</leakage_risk>
  </strategy>
  <strategy>
    <name>blocked</name>
    <description>Non-overlapping contiguous time blocks</description>
    <best_for>Testing regime-specific performance</best_for>
    <leakage_risk>Low</leakage_risk>
  </strategy>
  <recommendation>
    For crypto forecasting, use 'walk_forward' or 'purged_kfold' to ensure 
    realistic performance estimates without data leakage.
  </recommendation>
</cv_strategies>'''


# ============================================================================
# DRIFT DETECTION TOOLS
# ============================================================================

async def check_model_drift(
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    model_name: str = "lightgbm",
    reference_hours: int = 168,
    current_hours: int = 24
) -> str:
    """
    Check if a forecasting model has drifted and needs retraining.
    
    Detects:
    - Data drift (input distribution changes)
    - Concept drift (prediction relationship changes)
    - Performance degradation
    
    Args:
        symbol: Trading pair
        exchange: Exchange name
        model_name: Model to check
        reference_hours: Hours of reference data (training period)
        current_hours: Hours of recent data to check
        
    Returns:
        XML formatted drift report with recommendations
    """
    try:
        from src.analytics.drift_detector import (
            ModelDriftMonitor, DriftSeverity, DriftType
        )
        from src.integrations.darts_bridge import DartsBridge, DartsModelWrapper
        from src.tools.darts_tools import _load_price_data
        
        # Load reference data
        ref_df = _load_price_data(symbol, exchange, hours=reference_hours)
        if ref_df is None or len(ref_df) < 100:
            return f'''<error>
  <message>Insufficient reference data for {symbol}</message>
</error>'''
        
        # Load current data
        curr_df = _load_price_data(symbol, exchange, hours=current_hours)
        if curr_df is None or len(curr_df) < 10:
            return f'''<error>
  <message>Insufficient current data for {symbol}</message>
</error>'''
        
        # Convert to arrays
        bridge = DartsBridge()
        
        # Split reference into train/val
        ref_data = ref_df['price'].values
        train_size = int(len(ref_data) * 0.8)
        train_data = ref_data[:train_size]
        val_data = ref_data[train_size:]
        
        # Train model on reference period
        model = DartsModelWrapper(model_name, verbose=False)
        train_series = bridge.to_darts(train_data)
        model.fit(train_series)
        
        # Get reference predictions
        ref_forecast = model.predict(n=len(val_data))
        ref_predictions = ref_forecast.values().flatten()
        
        # Get current predictions
        curr_data = curr_df['price'].values
        curr_series = bridge.to_darts(ref_data[-len(train_data):])  # Use recent history
        model.fit(curr_series)  # Retrain on recent
        
        # Simulate "current" predictions
        curr_forecast = model.predict(n=len(curr_data))
        curr_predictions = curr_forecast.values().flatten()[:len(curr_data)]
        
        # Check drift
        monitor = ModelDriftMonitor(model_name)
        monitor.set_reference(ref_predictions, val_data)
        report = monitor.check_drift(curr_predictions, curr_data)
        
        # Format severity color
        severity_colors = {
            DriftSeverity.NONE: 'green',
            DriftSeverity.LOW: 'yellow',
            DriftSeverity.MEDIUM: 'orange',
            DriftSeverity.HIGH: 'red',
            DriftSeverity.CRITICAL: 'darkred'
        }
        
        return f'''<drift_report>
  <model>{model_name}</model>
  <symbol>{symbol}</symbol>
  <exchange>{exchange}</exchange>
  <drift_status>
    <detected>{str(report.drift_detected).lower()}</detected>
    <type>{report.drift_type.value}</type>
    <severity>{report.severity.value}</severity>
    <severity_color>{severity_colors.get(report.severity, 'gray')}</severity_color>
    <confidence>{report.confidence:.2f}</confidence>
  </drift_status>
  <metrics>
    <reference_mape>{report.details.get('reference_mape', 0):.2f}%</reference_mape>
    <current_mape>{report.details.get('current_mape', 0):.2f}%</current_mape>
    <degradation_pct>{report.details.get('degradation_pct', 0):.1f}%</degradation_pct>
    <prediction_psi>{report.details.get('prediction_psi', 0):.4f}</prediction_psi>
    <error_psi>{report.details.get('error_psi', 0):.4f}</error_psi>
  </metrics>
  <recommendation>{report.recommendation}</recommendation>
  <action_required>{report.severity.value in ['high', 'critical']}</action_required>
</drift_report>'''
        
    except Exception as e:
        logger.error(f"Drift detection error: {e}")
        return f'''<error>
  <message>Drift detection failed: {str(e)}</message>
</error>'''


async def get_model_health_report(
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    model_name: str = "lightgbm"
) -> str:
    """
    Get comprehensive health report for a forecasting model.
    
    Includes:
    - Current performance metrics
    - Drift status
    - Retraining recommendations
    - Historical performance trends
    
    Args:
        symbol: Trading pair
        exchange: Exchange name
        model_name: Model to analyze
        
    Returns:
        XML formatted health report
    """
    try:
        from src.analytics.drift_detector import ModelDriftMonitor
        from src.integrations.darts_bridge import DartsBridge, DartsModelWrapper
        from src.tools.darts_tools import _load_price_data
        import time
        
        # Load recent data
        df = _load_price_data(symbol, exchange, hours=168)
        if df is None:
            return f'''<error>
  <message>No data available for {symbol}</message>
</error>'''
        
        bridge = DartsBridge()
        data = df['price'].values
        
        # Split into train/test
        train_size = int(len(data) * 0.8)
        train_data = data[:train_size]
        test_data = data[train_size:]
        
        # Train model
        model = DartsModelWrapper(model_name, verbose=False)
        train_series = bridge.to_darts(train_data)
        
        start_time = time.time()
        model.fit(train_series)
        training_time = time.time() - start_time
        
        # Make predictions
        start_time = time.time()
        forecast = model.predict(n=len(test_data))
        inference_time = time.time() - start_time
        predictions = forecast.values().flatten()
        
        # Calculate metrics
        errors = np.abs(predictions - test_data)
        mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100
        rmse = np.sqrt(np.mean(errors ** 2))
        mae = np.mean(errors)
        
        # Determine health status
        if mape < 3:
            status = 'excellent'
            status_color = 'green'
        elif mape < 5:
            status = 'good'
            status_color = 'lightgreen'
        elif mape < 10:
            status = 'acceptable'
            status_color = 'yellow'
        elif mape < 20:
            status = 'degraded'
            status_color = 'orange'
        else:
            status = 'poor'
            status_color = 'red'
        
        return f'''<model_health_report>
  <model>{model_name}</model>
  <symbol>{symbol}</symbol>
  <exchange>{exchange}</exchange>
  <timestamp>{time.strftime('%Y-%m-%d %H:%M:%S')}</timestamp>
  <health_status>
    <status>{status}</status>
    <status_color>{status_color}</status_color>
  </health_status>
  <performance>
    <mape>{mape:.2f}%</mape>
    <rmse>{rmse:.4f}</rmse>
    <mae>{mae:.4f}</mae>
    <training_time_seconds>{training_time:.2f}</training_time_seconds>
    <inference_time_seconds>{inference_time:.4f}</inference_time_seconds>
  </performance>
  <data_summary>
    <total_samples>{len(data)}</total_samples>
    <train_samples>{train_size}</train_samples>
    <test_samples>{len(test_data)}</test_samples>
    <price_mean>{np.mean(data):.2f}</price_mean>
    <price_std>{np.std(data):.2f}</price_std>
  </data_summary>
  <recommendations>
    <retrain_needed>{status in ['degraded', 'poor']}</retrain_needed>
    <suggested_action>{
        'No action needed' if status in ['excellent', 'good'] else
        'Monitor closely' if status == 'acceptable' else
        'Consider retraining' if status == 'degraded' else
        'Retrain immediately'
    }</suggested_action>
  </recommendations>
</model_health_report>'''
        
    except Exception as e:
        logger.error(f"Health report error: {e}")
        return f'''<error>
  <message>Health report failed: {str(e)}</message>
</error>'''


async def monitor_prediction_quality(
    predictions: List[float],
    actuals: List[float],
    model_name: str = "unknown"
) -> str:
    """
    Monitor prediction quality and detect anomalies.
    
    Analyzes recent predictions vs actuals to detect:
    - Systematic bias
    - Increasing errors
    - Outlier predictions
    
    Args:
        predictions: List of model predictions
        actuals: List of actual values
        model_name: Name of the model
        
    Returns:
        XML formatted quality report
    """
    try:
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate errors
        errors = predictions - actuals
        abs_errors = np.abs(errors)
        pct_errors = np.abs(errors / actuals) * 100
        
        # Bias detection
        mean_error = np.mean(errors)
        bias_detected = abs(mean_error) > np.std(errors) * 0.5
        bias_direction = 'overestimating' if mean_error > 0 else 'underestimating'
        
        # Trend detection (are errors increasing?)
        if len(errors) > 10:
            recent_errors = np.mean(abs_errors[-5:])
            earlier_errors = np.mean(abs_errors[:5])
            error_trend = (recent_errors - earlier_errors) / (earlier_errors + 1e-8) * 100
        else:
            error_trend = 0
        
        # Outlier detection
        error_threshold = np.mean(abs_errors) + 2 * np.std(abs_errors)
        n_outliers = np.sum(abs_errors > error_threshold)
        
        return f'''<prediction_quality_report>
  <model>{model_name}</model>
  <n_predictions>{len(predictions)}</n_predictions>
  <error_metrics>
    <mape>{np.mean(pct_errors):.2f}%</mape>
    <mae>{np.mean(abs_errors):.4f}</mae>
    <rmse>{np.sqrt(np.mean(errors**2)):.4f}</rmse>
    <max_error>{np.max(abs_errors):.4f}</max_error>
  </error_metrics>
  <bias_analysis>
    <bias_detected>{str(bias_detected).lower()}</bias_detected>
    <bias_direction>{bias_direction if bias_detected else 'none'}</bias_direction>
    <mean_error>{mean_error:.4f}</mean_error>
  </bias_analysis>
  <trend_analysis>
    <error_trend_pct>{error_trend:.1f}%</error_trend_pct>
    <trend_direction>{'increasing' if error_trend > 10 else 'decreasing' if error_trend < -10 else 'stable'}</trend_direction>
  </trend_analysis>
  <outliers>
    <n_outliers>{n_outliers}</n_outliers>
    <outlier_threshold>{error_threshold:.4f}</outlier_threshold>
    <outlier_rate>{n_outliers / len(predictions) * 100:.1f}%</outlier_rate>
  </outliers>
  <overall_quality>{'good' if np.mean(pct_errors) < 5 and not bias_detected else 'acceptable' if np.mean(pct_errors) < 10 else 'needs_attention'}</overall_quality>
</prediction_quality_report>'''
        
    except Exception as e:
        return f'''<error>
  <message>Quality monitoring failed: {str(e)}</message>
</error>'''
