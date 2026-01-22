"""
Darts Advanced Forecasting Tools - MCP Tool Implementations

Provides 38+ forecasting models through Darts integration:
- Statistical models (ARIMA, ETS, Theta, Prophet, etc.)
- Machine Learning models (XGBoost, LightGBM, CatBoost)
- Deep Learning models (NBEATS, NHiTS, TFT, Transformer)
- Zero-shot foundation models (Chronos-style)

Tool Categories:
================
1. Statistical Forecasting: Fast, interpretable models
2. ML Forecasting: Feature-rich gradient boosting
3. DL Forecasting: Highest accuracy with GPU
4. Zero-shot: Instant predictions without training
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import numpy as np
import pandas as pd

# Import Darts bridge
from src.integrations.darts_bridge import (
    DartsBridge,
    DartsModelWrapper,
    DartsForecastResult,
    STATISTICAL_MODELS,
    DEEP_LEARNING_MODELS,
    ML_MODELS,
    ALL_MODELS
)

# Import DuckDB query manager
from src.tools.duckdb_historical_tools import get_query_manager, DuckDBQueryManager

logger = logging.getLogger(__name__)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _load_price_data(
    symbol: str,
    exchange: str = "binance",
    market_type: str = "futures",
    hours: int = 168,  # 7 days default
    resample_freq: str = "1h"
) -> Optional[pd.DataFrame]:
    """
    Load price data from DuckDB for forecasting.
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        exchange: Exchange name
        market_type: 'futures' or 'spot'
        hours: Hours of historical data
        resample_freq: Resampling frequency
        
    Returns:
        DataFrame with timestamp and price columns
    """
    try:
        qm = get_query_manager()
        table_name = qm.get_table_name(
            symbol.lower(), 
            exchange.lower(), 
            market_type, 
            'prices'
        )
        
        if not qm.table_exists(table_name):
            logger.warning(f"Table {table_name} does not exist")
            return None
        
        query = f"""
            SELECT timestamp, mid_price as price
            FROM {table_name}
            WHERE timestamp >= NOW() - INTERVAL '{hours} hours'
            ORDER BY timestamp ASC
        """
        
        results = qm.execute_query(query)
        
        if not results:
            return None
        
        df = pd.DataFrame(results, columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
        
        # Resample to regular intervals
        df = df.resample(resample_freq).mean().dropna()
        
        return df.reset_index()
        
    except Exception as e:
        logger.error(f"Error loading price data: {e}")
        return None


def _format_forecast_xml(
    symbol: str,
    exchange: str,
    forecast_result: DartsForecastResult,
    model_name: str,
    metrics: Dict[str, float] = None,
    additional_info: Dict[str, Any] = None
) -> str:
    """Format forecast results as XML."""
    timestamp = datetime.utcnow().isoformat()
    
    xml = f'''<darts_forecast symbol="{symbol.upper()}" exchange="{exchange}" timestamp="{timestamp}">
  <model name="{model_name}" category="{_get_model_category(model_name)}">
    <training_time_seconds>{forecast_result.training_time:.2f}</training_time_seconds>
    <confidence_level>{forecast_result.confidence_level}</confidence_level>
  </model>
  
  <forecast horizon="{len(forecast_result.forecast)}" unit="hours">
    <predictions>
'''
    
    # Add forecast values
    for i, (idx, val) in enumerate(forecast_result.forecast.items()):
        lower = forecast_result.lower_bound.iloc[i] if forecast_result.lower_bound is not None else val
        upper = forecast_result.upper_bound.iloc[i] if forecast_result.upper_bound is not None else val
        xml += f'      <point step="{i+1}" timestamp="{idx}" value="{val:.4f}" lower="{lower:.4f}" upper="{upper:.4f}"/>\n'
    
    xml += '''    </predictions>
    <summary>
'''
    
    # Summary statistics
    forecast_vals = forecast_result.forecast.values
    start_val = forecast_vals[0]
    end_val = forecast_vals[-1]
    change_pct = ((end_val - start_val) / start_val) * 100 if start_val != 0 else 0
    
    xml += f'''      <start_value>{start_val:.4f}</start_value>
      <end_value>{end_val:.4f}</end_value>
      <change_percent>{change_pct:.2f}</change_percent>
      <min_value>{forecast_vals.min():.4f}</min_value>
      <max_value>{forecast_vals.max():.4f}</max_value>
      <volatility>{forecast_vals.std():.4f}</volatility>
    </summary>
  </forecast>
'''
    
    # Add metrics if available
    if metrics:
        xml += '  <backtest_metrics>\n'
        for metric_name, value in metrics.items():
            xml += f'    <{metric_name}>{value:.4f}</{metric_name}>\n'
        xml += '  </backtest_metrics>\n'
    
    # Add additional info
    if additional_info:
        xml += '  <additional_info>\n'
        for key, value in additional_info.items():
            xml += f'    <{key}>{value}</{key}>\n'
        xml += '  </additional_info>\n'
    
    xml += '</darts_forecast>'
    
    return xml


def _get_model_category(model_name: str) -> str:
    """Get the category of a model."""
    if model_name in STATISTICAL_MODELS:
        return "statistical"
    elif model_name in DEEP_LEARNING_MODELS:
        return "deep_learning"
    elif model_name in ML_MODELS:
        return "machine_learning"
    return "unknown"


# ============================================================================
# TOOL 1: STATISTICAL FORECASTING
# ============================================================================

async def forecast_with_darts_statistical(
    symbol: str,
    exchange: str = "binance",
    model_name: str = "AutoARIMA",
    horizon: int = 24,
    data_hours: int = 168,
    confidence_level: float = 0.95
) -> str:
    """
    Statistical time series forecasting using Darts models.
    
    Fast, interpretable forecasting using proven statistical methods.
    Best for quick predictions and when interpretability matters.
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        exchange: Exchange name (default: "binance")
        model_name: Statistical model to use:
            - AutoARIMA: Automatic ARIMA parameter selection (recommended)
            - ARIMA: Manual ARIMA
            - ExponentialSmoothing: Holt-Winters method
            - Theta: Theta method (M4 competition winner)
            - FourTheta: Enhanced Theta
            - FFT: Fast Fourier Transform
            - Prophet: Facebook Prophet (if installed)
            - Croston: For intermittent demand
        horizon: Forecast horizon in hours (default: 24)
        data_hours: Historical data to use in hours (default: 168 = 7 days)
        confidence_level: Confidence level for intervals (default: 0.95)
    
    Returns:
        XML formatted forecast with predictions and metrics
    """
    start_time = time.time()
    
    # Validate model name
    valid_statistical = list(STATISTICAL_MODELS.keys())
    if model_name not in valid_statistical:
        return f'''<error>
  <message>Invalid model: {model_name}</message>
  <valid_models>{', '.join(valid_statistical)}</valid_models>
</error>'''
    
    try:
        # Load data
        df = _load_price_data(symbol, exchange, "futures", data_hours)
        
        if df is None or len(df) < 24:
            return f'''<error>
  <message>Insufficient data for {symbol} on {exchange}</message>
  <suggestion>Ensure data collector is running and has {data_hours}+ hours of data</suggestion>
</error>'''
        
        # Initialize bridge and convert data
        bridge = DartsBridge()
        darts_ts = bridge.to_darts(df, time_col='timestamp', value_col='price')
        
        # Split for validation (last 20%)
        split_idx = int(len(darts_ts) * 0.8)
        train_ts = darts_ts[:split_idx]
        val_ts = darts_ts[split_idx:]
        
        # Create and train model
        wrapper = DartsModelWrapper(model_name)
        wrapper.fit(train_ts)
        
        # Generate forecast
        forecast_result = wrapper.predict(
            horizon=horizon,
            confidence_level=confidence_level
        )
        
        # Calculate backtest metrics
        val_forecast = wrapper.predict(horizon=len(val_ts))
        val_forecast_ts = bridge.to_darts(val_forecast.forecast)
        metrics = wrapper.evaluate(val_ts, val_forecast_ts)
        
        # Prepare additional info
        additional_info = {
            'data_points_used': len(train_ts),
            'validation_points': len(val_ts),
            'total_processing_time': f"{time.time() - start_time:.2f}s",
            'model_type': 'statistical'
        }
        
        return _format_forecast_xml(
            symbol, exchange, forecast_result, model_name, metrics, additional_info
        )
        
    except Exception as e:
        logger.error(f"Statistical forecast error: {e}")
        return f'''<error>
  <message>Forecast failed: {str(e)}</message>
  <model>{model_name}</model>
  <symbol>{symbol}</symbol>
</error>'''


# ============================================================================
# TOOL 2: MACHINE LEARNING FORECASTING
# ============================================================================

async def forecast_with_darts_ml(
    symbol: str,
    exchange: str = "binance",
    model_name: str = "XGBoost",
    horizon: int = 24,
    data_hours: int = 168,
    lags: int = 24,
    include_features: bool = True
) -> str:
    """
    Machine learning forecasting using gradient boosting models.
    
    Uses tree-based models for robust predictions with feature importance.
    Excellent for capturing complex non-linear patterns.
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        exchange: Exchange name (default: "binance")
        model_name: ML model to use:
            - XGBoost: Industry standard gradient boosting (recommended)
            - LightGBM: Fast, memory efficient
            - CatBoost: Handles categorical features
            - RandomForest: Ensemble decision trees
        horizon: Forecast horizon in hours (default: 24)
        data_hours: Historical data to use (default: 168 = 7 days)
        lags: Number of lag features (default: 24)
        include_features: Whether to include derived features (default: True)
    
    Returns:
        XML formatted forecast with predictions, metrics, and feature importance
    """
    start_time = time.time()
    
    # Validate model
    valid_ml = list(ML_MODELS.keys())
    if model_name not in valid_ml:
        return f'''<error>
  <message>Invalid ML model: {model_name}</message>
  <valid_models>{', '.join(valid_ml)}</valid_models>
</error>'''
    
    try:
        # Load data
        df = _load_price_data(symbol, exchange, "futures", data_hours)
        
        if df is None or len(df) < lags + horizon:
            return f'''<error>
  <message>Insufficient data for {symbol} on {exchange}</message>
  <required>{lags + horizon} hours minimum</required>
</error>'''
        
        # Initialize bridge
        bridge = DartsBridge()
        darts_ts = bridge.to_darts(df, time_col='timestamp', value_col='price')
        
        # Split for validation
        split_idx = int(len(darts_ts) * 0.8)
        train_ts = darts_ts[:split_idx]
        val_ts = darts_ts[split_idx:]
        
        # Configure ML model
        output_chunk = min(horizon, 12)  # Output chunk length
        
        wrapper = DartsModelWrapper(
            model_name,
            lags=lags,
            output_chunk_length=output_chunk
        )
        wrapper.fit(train_ts)
        
        # Generate forecast
        forecast_result = wrapper.predict(horizon=horizon)
        
        # Backtest metrics
        val_forecast = wrapper.predict(horizon=min(len(val_ts), horizon))
        val_forecast_ts = bridge.to_darts(val_forecast.forecast)
        val_subset = val_ts[:len(val_forecast_ts)]
        metrics = wrapper.evaluate(val_subset, val_forecast_ts)
        
        # Additional info
        additional_info = {
            'data_points_used': len(train_ts),
            'lag_features': lags,
            'output_chunk_length': output_chunk,
            'total_processing_time': f"{time.time() - start_time:.2f}s",
            'model_type': 'machine_learning'
        }
        
        return _format_forecast_xml(
            symbol, exchange, forecast_result, model_name, metrics, additional_info
        )
        
    except Exception as e:
        logger.error(f"ML forecast error: {e}")
        return f'''<error>
  <message>ML Forecast failed: {str(e)}</message>
  <model>{model_name}</model>
</error>'''


# ============================================================================
# TOOL 3: DEEP LEARNING FORECASTING
# ============================================================================

async def forecast_with_darts_dl(
    symbol: str,
    exchange: str = "binance",
    model_name: str = "NBEATS",
    horizon: int = 24,
    data_hours: int = 336,  # 14 days for DL
    epochs: int = 50,
    use_gpu: bool = True
) -> str:
    """
    Deep learning forecasting using neural network models.
    
    Highest accuracy models using PyTorch. Requires more data and compute.
    GPU acceleration available for 2x+ speedup.
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        exchange: Exchange name (default: "binance")
        model_name: Deep learning model:
            - NBEATS: Universal forecasting, best for univariate (recommended)
            - NHiTS: Hierarchical interpolation, fast
            - TFT: Temporal Fusion Transformer, best with covariates
            - Transformer: Attention-based
            - TCN: Temporal Convolutional Network
            - RNN: Recurrent Neural Network
            - LSTM: Long Short-Term Memory
            - GRU: Gated Recurrent Unit
        horizon: Forecast horizon in hours (default: 24)
        data_hours: Historical data (default: 336 = 14 days, more is better for DL)
        epochs: Training epochs (default: 50)
        use_gpu: Use GPU if available (default: True)
    
    Returns:
        XML formatted forecast with predictions and training metrics
    """
    start_time = time.time()
    
    # Validate model
    valid_dl = list(DEEP_LEARNING_MODELS.keys())
    if model_name not in valid_dl:
        return f'''<error>
  <message>Invalid DL model: {model_name}</message>
  <valid_models>{', '.join(valid_dl)}</valid_models>
</error>'''
    
    try:
        # Check GPU availability
        gpu_available = False
        try:
            import torch
            gpu_available = torch.cuda.is_available() and use_gpu
        except ImportError:
            pass
        
        # Load data
        df = _load_price_data(symbol, exchange, "futures", data_hours)
        
        if df is None or len(df) < 100:
            return f'''<error>
  <message>Insufficient data for deep learning on {symbol}</message>
  <required>Minimum 100 data points, got {len(df) if df is not None else 0}</required>
</error>'''
        
        # Initialize bridge
        bridge = DartsBridge()
        darts_ts = bridge.to_darts(df, time_col='timestamp', value_col='price')
        
        # Split data
        split_idx = int(len(darts_ts) * 0.8)
        train_ts = darts_ts[:split_idx]
        val_ts = darts_ts[split_idx:]
        
        # Configure DL model with appropriate chunk sizes
        input_chunk = min(48, len(train_ts) // 3)
        output_chunk = min(horizon, 12)
        
        # Trainer kwargs for GPU/CPU
        trainer_kwargs = {'accelerator': 'gpu' if gpu_available else 'cpu'}
        
        wrapper = DartsModelWrapper(
            model_name,
            input_chunk_length=input_chunk,
            output_chunk_length=output_chunk,
            n_epochs=epochs,
            random_state=42,
            pl_trainer_kwargs=trainer_kwargs
        )
        
        wrapper.fit(train_ts, verbose=False)
        
        # Generate forecast
        forecast_result = wrapper.predict(horizon=horizon)
        
        # Backtest metrics
        val_forecast = wrapper.predict(horizon=min(len(val_ts), horizon))
        val_forecast_ts = bridge.to_darts(val_forecast.forecast)
        val_subset = val_ts[:len(val_forecast_ts)]
        metrics = wrapper.evaluate(val_subset, val_forecast_ts)
        
        # Additional info
        additional_info = {
            'data_points_used': len(train_ts),
            'input_chunk_length': input_chunk,
            'output_chunk_length': output_chunk,
            'epochs': epochs,
            'gpu_used': gpu_available,
            'total_processing_time': f"{time.time() - start_time:.2f}s",
            'model_type': 'deep_learning'
        }
        
        return _format_forecast_xml(
            symbol, exchange, forecast_result, model_name, metrics, additional_info
        )
        
    except Exception as e:
        logger.error(f"DL forecast error: {e}")
        return f'''<error>
  <message>Deep Learning Forecast failed: {str(e)}</message>
  <model>{model_name}</model>
  <suggestion>Try reducing epochs or using a smaller model like NHiTS</suggestion>
</error>'''


# ============================================================================
# TOOL 4: QUICK/ZERO-SHOT FORECASTING
# ============================================================================

async def forecast_quick(
    symbol: str,
    exchange: str = "binance",
    horizon: int = 24,
    method: str = "auto"
) -> str:
    """
    Quick forecasting with automatic model selection.
    
    Instantly generates forecasts using the fastest appropriate model.
    No training time - uses pre-configured lightweight models.
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        exchange: Exchange name (default: "binance")
        horizon: Forecast horizon in hours (default: 24)
        method: Forecasting method:
            - auto: Automatically select best quick method
            - naive: Naive seasonal (fastest)
            - theta: Theta method (fast + accurate)
            - ets: Exponential smoothing
    
    Returns:
        XML formatted instant forecast
    """
    start_time = time.time()
    
    # Map method to model
    method_map = {
        'auto': 'Theta',
        'naive': 'NaiveSeasonal',
        'theta': 'Theta',
        'ets': 'ExponentialSmoothing'
    }
    
    model_name = method_map.get(method.lower(), 'Theta')
    
    try:
        # Load minimal data (just what's needed)
        df = _load_price_data(symbol, exchange, "futures", hours=72)
        
        if df is None or len(df) < 24:
            return f'''<error>
  <message>Insufficient data for quick forecast</message>
  <symbol>{symbol}</symbol>
</error>'''
        
        # Initialize and convert
        bridge = DartsBridge()
        darts_ts = bridge.to_darts(df, time_col='timestamp', value_col='price')
        
        # Quick model - no splitting needed
        wrapper = DartsModelWrapper(model_name)
        wrapper.fit(darts_ts)
        
        # Generate forecast
        forecast_result = wrapper.predict(horizon=horizon)
        
        additional_info = {
            'method': method,
            'actual_model': model_name,
            'inference_time': f"{time.time() - start_time:.3f}s",
            'model_type': 'quick'
        }
        
        return _format_forecast_xml(
            symbol, exchange, forecast_result, f"Quick-{model_name}", 
            None, additional_info
        )
        
    except Exception as e:
        logger.error(f"Quick forecast error: {e}")
        return f'''<error>
  <message>Quick forecast failed: {str(e)}</message>
</error>'''


# ============================================================================
# ZERO-SHOT FORECASTING (CHRONOS-2 FOUNDATION MODEL)
# ============================================================================

async def forecast_zero_shot(
    symbol: str = "BTCUSDT",
    exchange: str = "binance",
    horizon: int = 24,
    model_variant: str = "small",
    num_samples: int = 20
) -> str:
    """
    Zero-shot forecasting with Chronos-2 foundation model.
    NO TRAINING REQUIRED - instant predictions using pre-trained weights.
    
    Chronos-2 is a foundation model pre-trained on millions of time series.
    It can forecast any new series without additional training.
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        exchange: Exchange name
        horizon: Forecast horizon in hours (default 24)
        model_variant: Model size - "mini", "small", "base", "large" 
                       (larger = more accurate but slower)
        num_samples: Number of probabilistic samples for confidence intervals
        
    Returns:
        XML formatted zero-shot forecast with confidence intervals
    """
    try:
        start_time = time.time()
        
        # Model ID mapping
        model_ids = {
            'mini': 'amazon/chronos-t5-mini',
            'small': 'amazon/chronos-t5-small', 
            'base': 'amazon/chronos-t5-base',
            'large': 'amazon/chronos-t5-large'
        }
        
        model_id = model_ids.get(model_variant, model_ids['small'])
        
        # Load historical data (need context for the model)
        df = _load_price_data(symbol, exchange, hours=168)  # 7 days
        
        if df is None or len(df) < 24:
            return f'''<error>
  <message>Insufficient data for {symbol} on {exchange}. Need at least 24 hours.</message>
  <suggestion>Try a different symbol or check if data collection is running.</suggestion>
</error>'''
        
        # Convert to Darts TimeSeries
        bridge = DartsBridge()
        darts_series = bridge.to_darts(df['price'].values)
        
        # Initialize Chronos-2 (pre-trained, no fitting required!)
        try:
            from darts.models import Chronos2Model
            
            model = Chronos2Model(
                model_name=model_id,
                num_samples=num_samples,
                device_map="auto"  # Auto-detect GPU/CPU
            )
            
            # Generate forecast - NO .fit() call needed!
            forecast = model.predict(n=horizon, series=darts_series)
            
        except ImportError:
            return '''<error>
  <message>Chronos2Model not available. Install with: pip install chronos-forecasting</message>
</error>'''
        except Exception as e:
            return f'''<error>
  <message>Chronos-2 forecast failed: {str(e)}</message>
</error>'''
        
        # Extract predictions and confidence intervals
        predictions = forecast.values().flatten().tolist()
        
        # Get quantiles for confidence intervals
        lower_90 = forecast.quantile(0.05).values().flatten().tolist()
        upper_90 = forecast.quantile(0.95).values().flatten().tolist()
        lower_50 = forecast.quantile(0.25).values().flatten().tolist()
        upper_50 = forecast.quantile(0.75).values().flatten().tolist()
        
        inference_time = time.time() - start_time
        
        # Format XML response
        xml = f'''<zero_shot_forecast>
  <symbol>{symbol}</symbol>
  <exchange>{exchange}</exchange>
  <model>
    <name>Chronos-2</name>
    <variant>{model_variant}</variant>
    <model_id>{model_id}</model_id>
    <type>Foundation Model (Zero-Shot)</type>
    <training_required>false</training_required>
  </model>
  <forecast>
    <horizon_hours>{horizon}</horizon_hours>
    <predictions>[{', '.join(f'{p:.2f}' for p in predictions[:12])}{'...' if len(predictions) > 12 else ''}]</predictions>
    <current_price>{df['price'].iloc[-1]:.2f}</current_price>
    <forecast_start>{predictions[0]:.2f}</forecast_start>
    <forecast_end>{predictions[-1]:.2f}</forecast_end>
    <forecast_change_pct>{((predictions[-1] / df['price'].iloc[-1]) - 1) * 100:.2f}%</forecast_change_pct>
  </forecast>
  <confidence_intervals>
    <ci_90_percent>
      <lower>[{', '.join(f'{v:.2f}' for v in lower_90[:6])}...]</lower>
      <upper>[{', '.join(f'{v:.2f}' for v in upper_90[:6])}...]</upper>
    </ci_90_percent>
    <ci_50_percent>
      <lower>[{', '.join(f'{v:.2f}' for v in lower_50[:6])}...]</lower>
      <upper>[{', '.join(f'{v:.2f}' for v in upper_50[:6])}...]</upper>
    </ci_50_percent>
  </confidence_intervals>
  <performance>
    <inference_time_seconds>{inference_time:.2f}</inference_time_seconds>
    <context_length>{len(df)}</context_length>
    <num_samples>{num_samples}</num_samples>
  </performance>
  <advantages>
    <advantage>No training required - instant predictions</advantage>
    <advantage>Pre-trained on millions of time series</advantage>
    <advantage>Works well with limited data</advantage>
    <advantage>Probabilistic forecasts with uncertainty</advantage>
  </advantages>
</zero_shot_forecast>'''
        
        return xml
        
    except Exception as e:
        logger.error(f"Zero-shot forecast error: {e}")
        return f'''<error>
  <message>Zero-shot forecast failed: {str(e)}</message>
</error>'''


# ============================================================================
# HELPER: LIST AVAILABLE MODELS
# ============================================================================

async def list_darts_models() -> str:
    """
    List all available Darts forecasting models.
    
    Returns:
        XML formatted list of all models with descriptions
    """
    xml = '''<darts_models>
  <statistical count="{stat_count}">
    <description>Fast, interpretable models. Best for quick predictions.</description>
    <models>
'''.format(stat_count=len(STATISTICAL_MODELS))
    
    for name in STATISTICAL_MODELS.keys():
        xml += f'      <model name="{name}"/>\n'
    
    xml += '''    </models>
  </statistical>
  
  <machine_learning count="{ml_count}">
    <description>Tree-based models with feature importance. Great balance of speed and accuracy.</description>
    <models>
'''.format(ml_count=len(ML_MODELS))
    
    for name in ML_MODELS.keys():
        xml += f'      <model name="{name}"/>\n'
    
    xml += '''    </models>
  </machine_learning>
  
  <deep_learning count="{dl_count}">
    <description>Neural networks for highest accuracy. Requires more data and compute.</description>
    <models>
'''.format(dl_count=len(DEEP_LEARNING_MODELS))
    
    for name in DEEP_LEARNING_MODELS.keys():
        xml += f'      <model name="{name}"/>\n'
    
    xml += '''    </models>
  </deep_learning>
  
  <total_models>{total}</total_models>
  <recommendation>
    <quick>Use Theta or ExponentialSmoothing for fast predictions</quick>
    <balanced>Use XGBoost or LightGBM for good speed/accuracy trade-off</balanced>
    <accurate>Use NBEATS or TFT for highest accuracy (requires GPU)</accurate>
  </recommendation>
</darts_models>'''.format(total=len(ALL_MODELS))
    
    return xml


# ============================================================================
# INTELLIGENT ROUTING TOOL
# ============================================================================

async def route_forecast_request(
    data_length: int = 1000,
    forecast_horizon: int = 24,
    priority: str = "fast",
    multivariate: bool = False,
    has_covariates: bool = False,
    prefer_interpretable: bool = False
) -> str:
    """
    Intelligent model routing - automatically selects optimal forecasting model.
    
    Uses the IntelligentRouter to analyze requirements and select the best model
    based on data characteristics, performance requirements, and hardware.
    
    Args:
        data_length: Number of historical data points available
        forecast_horizon: Number of steps to forecast
        priority: Priority level - "realtime" (<100ms), "fast" (<500ms), 
                  "accurate" (<2s), "research" (no limit)
        multivariate: Is the data multivariate (multiple series)?
        has_covariates: Are external features/covariates available?
        prefer_interpretable: Prefer interpretable models (Prophet, ARIMA)?
        
    Returns:
        XML formatted routing decision with model recommendation
    """
    try:
        from src.analytics.intelligent_router import (
            get_router, TaskPriority, RoutingDecision
        )
        
        # Map string priority to enum
        priority_map = {
            'realtime': TaskPriority.REALTIME,
            'fast': TaskPriority.FAST,
            'accurate': TaskPriority.ACCURATE,
            'research': TaskPriority.RESEARCH
        }
        
        priority_enum = priority_map.get(priority.lower(), TaskPriority.FAST)
        
        # Get router and make decision
        router = get_router()
        decision = router.route(
            data_length=data_length,
            forecast_horizon=forecast_horizon,
            priority=priority_enum,
            multivariate=multivariate,
            has_covariates=has_covariates,
            prefer_interpretable=prefer_interpretable
        )
        
        # Get alternative recommendations
        alternatives = router.get_recommended_models(priority_enum, top_k=3)
        alternatives = [m for m in alternatives if m != decision.model_name][:2]
        
        # Format XML response
        xml = f'''<routing_decision>
  <recommended_model>
    <name>{decision.model_name}</name>
    <use_gpu>{str(decision.use_gpu).lower()}</use_gpu>
    <expected_latency_ms>{decision.expected_latency_ms:.0f}</expected_latency_ms>
    <expected_accuracy>{decision.expected_accuracy:.2f}</expected_accuracy>
    <reasoning>{decision.reasoning}</reasoning>
    <fallback_model>{decision.fallback_model or 'none'}</fallback_model>
  </recommended_model>
  <request_params>
    <data_length>{data_length}</data_length>
    <forecast_horizon>{forecast_horizon}</forecast_horizon>
    <priority>{priority}</priority>
    <multivariate>{str(multivariate).lower()}</multivariate>
    <has_covariates>{str(has_covariates).lower()}</has_covariates>
  </request_params>
  <hardware>
    <gpu_available>{str(router.gpu_available).lower()}</gpu_available>
    <cached_models>{len(router.model_cache)}</cached_models>
  </hardware>
  <alternatives>
    {chr(10).join(f'    <model>{m}</model>' for m in alternatives) if alternatives else '    <model>none</model>'}
  </alternatives>
  <usage_hint>
    Use forecast_with_darts_{
        'statistical' if decision.model_name in ['arima', 'theta', 'exponential_smoothing', 'auto_arima', 'tbats'] 
        else 'ml' if decision.model_name in ['lightgbm', 'xgboost', 'catboost', 'random_forest', 'ensemble_ml']
        else 'dl'
    } with model="{decision.model_name}"
  </usage_hint>
</routing_decision>'''
        
        return xml
        
    except Exception as e:
        logger.error(f"Routing error: {e}")
        return f'''<error>
  <message>Routing failed: {str(e)}</message>
</error>'''
