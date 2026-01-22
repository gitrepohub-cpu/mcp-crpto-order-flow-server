"""
Darts Model Comparison & Selection Tools - MCP Tool Implementations

Tools for comparing multiple models and automatic model selection.
Essential for finding the best model for each trading pair.

Tool Categories:
================
1. Model Comparison: Side-by-side comparison of all models
2. Auto Selection: Intelligent model recommendation
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

# Import Darts components
from src.integrations.darts_bridge import (
    DartsBridge,
    DartsModelWrapper,
    DartsAutoML,
    DartsForecastResult,
    STATISTICAL_MODELS,
    DEEP_LEARNING_MODELS,
    ML_MODELS,
    ALL_MODELS
)

# Import data loading
from src.tools.darts_tools import _load_price_data

logger = logging.getLogger(__name__)


# ============================================================================
# TOOL 8: COMPARE ALL MODELS
# ============================================================================

async def compare_all_models(
    symbol: str,
    exchange: str = "binance",
    horizon: int = 24,
    models: str = "fast",
    metric: str = "mape",
    return_top_n: int = 5,
    data_hours: int = 168
) -> str:
    """
    Compare multiple Darts models side-by-side.
    
    Tests multiple models and ranks them by performance.
    Essential for finding the best model for a specific trading pair.
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        exchange: Exchange name (default: "binance")
        horizon: Forecast horizon in hours (default: 24)
        models: Model set to compare:
            - fast: Quick models only (Theta, ETS, ARIMA) ~30s
            - statistical: All statistical models ~60s
            - ml: Machine learning models ~90s
            - all: All models (may take several minutes)
        metric: Primary ranking metric ('mape', 'rmse', 'mae', 'smape')
        return_top_n: Number of top models to show in detail (default: 5)
        data_hours: Historical data hours (default: 168)
    
    Returns:
        XML formatted model comparison with rankings and recommendations
    """
    start_time = time.time()
    
    # Determine which models to test
    if models == "fast":
        models_to_test = ['NaiveSeasonal', 'Theta', 'ExponentialSmoothing', 'FFT']
    elif models == "statistical":
        models_to_test = list(STATISTICAL_MODELS.keys())
    elif models == "ml":
        models_to_test = list(ML_MODELS.keys())
    elif models == "all":
        models_to_test = list(STATISTICAL_MODELS.keys()) + list(ML_MODELS.keys())
    else:
        models_to_test = ['Theta', 'ExponentialSmoothing', 'XGBoost']
    
    try:
        # Load data
        df = _load_price_data(symbol, exchange, "futures", data_hours)
        
        if df is None or len(df) < 72:
            return f'''<error>
  <message>Insufficient data for model comparison</message>
  <symbol>{symbol}</symbol>
</error>'''
        
        # Initialize bridge
        bridge = DartsBridge()
        darts_ts = bridge.to_darts(df, time_col='timestamp', value_col='price')
        
        # Split data for evaluation
        split_idx = int(len(darts_ts) * 0.8)
        train_ts = darts_ts[:split_idx]
        val_ts = darts_ts[split_idx:]
        
        # Test each model
        results = []
        
        for model_name in models_to_test:
            model_start = time.time()
            
            try:
                # Create wrapper with appropriate settings
                if model_name in ML_MODELS:
                    wrapper = DartsModelWrapper(model_name, lags=24, output_chunk_length=12)
                else:
                    wrapper = DartsModelWrapper(model_name)
                
                # Train
                wrapper.fit(train_ts)
                train_time = time.time() - model_start
                
                # Predict and evaluate
                val_pred = wrapper.predict(horizon=min(len(val_ts), horizon))
                val_pred_ts = bridge.to_darts(val_pred.forecast)
                val_subset = val_ts[:len(val_pred_ts)]
                
                metrics = wrapper.evaluate(val_subset, val_pred_ts)
                
                results.append({
                    'model': model_name,
                    'mape': metrics.get('mape', float('inf')),
                    'rmse': metrics.get('rmse', float('inf')),
                    'mae': metrics.get('mae', float('inf')),
                    'smape': metrics.get('smape', float('inf')),
                    'train_time': train_time,
                    'status': 'success'
                })
                
            except Exception as e:
                results.append({
                    'model': model_name,
                    'mape': float('inf'),
                    'rmse': float('inf'),
                    'mae': float('inf'),
                    'smape': float('inf'),
                    'train_time': time.time() - model_start,
                    'status': f'failed: {str(e)[:50]}'
                })
        
        # Sort by selected metric
        valid_results = [r for r in results if r['status'] == 'success']
        valid_results.sort(key=lambda x: x.get(metric, float('inf')))
        
        processing_time = time.time() - start_time
        
        # Build XML response
        xml = f'''<darts_model_comparison symbol="{symbol.upper()}" exchange="{exchange}" timestamp="{datetime.utcnow().isoformat()}">
  <comparison_settings>
    <models_tested>{len(results)}</models_tested>
    <models_successful>{len(valid_results)}</models_successful>
    <horizon>{horizon}</horizon>
    <ranking_metric>{metric}</ranking_metric>
    <total_time_seconds>{processing_time:.1f}</total_time_seconds>
  </comparison_settings>
  
  <rankings>
'''
        
        # Add top models with full details
        for rank, result in enumerate(valid_results[:return_top_n], 1):
            xml += f'''    <model rank="{rank}">
      <name>{result['model']}</name>
      <mape>{result['mape']:.4f}</mape>
      <rmse>{result['rmse']:.6f}</rmse>
      <mae>{result['mae']:.6f}</mae>
      <smape>{result['smape']:.4f}</smape>
      <train_time_seconds>{result['train_time']:.2f}</train_time_seconds>
    </model>
'''
        
        xml += '''  </rankings>
  
  <all_results>
'''
        
        # Add all results in compact form
        for result in results:
            status = "✓" if result['status'] == 'success' else "✗"
            xml += f'    <model name="{result["model"]}" {metric}="{result.get(metric, "N/A")}" status="{status}"/>\n'
        
        xml += '''  </all_results>
  
  <recommendations>
'''
        
        # Add recommendations
        if valid_results:
            best = valid_results[0]
            xml += f'''    <best_model>{best['model']}</best_model>
    <best_{metric}>{best[metric]:.4f}</best_{metric}>
'''
            
            # Find fastest model
            fastest = min(valid_results, key=lambda x: x['train_time'])
            xml += f'''    <fastest_model>{fastest['model']}</fastest_model>
    <fastest_time>{fastest['train_time']:.2f}s</fastest_time>
'''
            
            # Recommend based on use case
            if best['train_time'] < 1.0:
                xml += '    <production_recommendation>Best model is fast enough for production</production_recommendation>\n'
            else:
                fast_good = [r for r in valid_results if r['train_time'] < 2.0]
                if fast_good:
                    xml += f'    <production_recommendation>Consider {fast_good[0]["model"]} for faster production use</production_recommendation>\n'
        
        xml += '''  </recommendations>
</darts_model_comparison>'''
        
        return xml
        
    except Exception as e:
        logger.error(f"Model comparison error: {e}")
        return f'''<error>
  <message>Model comparison failed: {str(e)}</message>
</error>'''


# ============================================================================
# TOOL 9: AUTO MODEL SELECTION
# ============================================================================

async def auto_model_select(
    symbol: str,
    exchange: str = "binance",
    horizon: int = 24,
    priority: str = "accuracy",
    include_forecast: bool = True,
    data_hours: int = 168
) -> str:
    """
    Automatic model selection using DartsAutoML.
    
    Intelligently selects the best model based on data characteristics
    and specified priorities. Also considers hardware availability.
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        exchange: Exchange name (default: "binance")
        horizon: Forecast horizon in hours (default: 24)
        priority: Selection priority:
            - accuracy: Choose best accuracy (may be slower)
            - speed: Choose fastest model (< 1s inference)
            - balanced: Optimize accuracy/speed trade-off
        include_forecast: Whether to include forecast in response (default: True)
        data_hours: Historical data hours (default: 168)
    
    Returns:
        XML formatted auto-selected model with forecast
    """
    start_time = time.time()
    
    # Determine models based on priority
    if priority == "speed":
        models_to_try = ['NaiveSeasonal', 'Theta', 'ExponentialSmoothing']
    elif priority == "accuracy":
        models_to_try = ['Theta', 'ExponentialSmoothing', 'ARIMA', 'XGBoost']
    else:  # balanced
        models_to_try = ['Theta', 'ExponentialSmoothing', 'XGBoost']
    
    try:
        # Load data
        df = _load_price_data(symbol, exchange, "futures", data_hours)
        
        if df is None or len(df) < 72:
            return f'''<error>
  <message>Insufficient data for auto model selection</message>
  <symbol>{symbol}</symbol>
</error>'''
        
        # Initialize bridge
        bridge = DartsBridge()
        darts_ts = bridge.to_darts(df, time_col='timestamp', value_col='price')
        
        # Split for evaluation
        split_idx = int(len(darts_ts) * 0.8)
        train_ts = darts_ts[:split_idx]
        val_ts = darts_ts[split_idx:]
        
        # Use DartsAutoML
        automl = DartsAutoML(
            models_to_try=models_to_try,
            metric='mape'
        )
        
        best_wrapper = automl.fit(train_ts, val_ts, verbose=False)
        
        # Get leaderboard
        leaderboard = automl.get_leaderboard()
        
        # Generate forecast with best model if requested
        forecast_xml = ""
        if include_forecast and automl.best_model is not None:
            # Retrain on full data
            automl.best_model.fit(darts_ts)
            forecast_result = automl.best_model.predict(horizon=horizon)
            
            forecast_xml = '''
  <forecast horizon="{horizon}" unit="hours">
    <predictions>
'''.format(horizon=len(forecast_result.forecast))
            
            for i, (idx, val) in enumerate(forecast_result.forecast.items()):
                forecast_xml += f'      <point step="{i+1}" value="{val:.4f}"/>\n'
            
            forecast_xml += '''    </predictions>
    <summary>
      <start_value>{start:.4f}</start_value>
      <end_value>{end:.4f}</end_value>
      <change_percent>{change:.2f}</change_percent>
    </summary>
  </forecast>
'''.format(
                start=forecast_result.forecast.iloc[0],
                end=forecast_result.forecast.iloc[-1],
                change=((forecast_result.forecast.iloc[-1] - forecast_result.forecast.iloc[0]) / 
                       forecast_result.forecast.iloc[0] * 100)
            )
        
        processing_time = time.time() - start_time
        
        # Build XML response
        xml = f'''<darts_auto_select symbol="{symbol.upper()}" exchange="{exchange}" timestamp="{datetime.utcnow().isoformat()}">
  <auto_selection>
    <priority>{priority}</priority>
    <models_evaluated>{len(automl.results)}</models_evaluated>
    <total_time_seconds>{processing_time:.1f}</total_time_seconds>
  </auto_selection>
  
  <selected_model>
    <name>{automl.best_model.model_name if automl.best_model else 'None'}</name>
    <category>{_get_category(automl.best_model.model_name) if automl.best_model else 'None'}</category>
    <selection_reason>Best {priority} among {len(models_to_try)} tested models</selection_reason>
  </selected_model>
  
  <evaluation_results>
'''
        
        # Add leaderboard
        for _, row in leaderboard.iterrows():
            if 'error' not in str(row.get('mape', '')):
                xml += f'    <model name="{row["model"]}" mape="{row.get("mape", "N/A")}" time="{row.get("time_seconds", "N/A")}s"/>\n'
        
        xml += '''  </evaluation_results>
'''
        
        xml += forecast_xml
        
        xml += '''  <recommendations>
    <primary>Use {model} for {priority} priority forecasting</primary>
    <alternative>Consider ensemble for production reliability</alternative>
  </recommendations>
</darts_auto_select>'''.format(
            model=automl.best_model.model_name if automl.best_model else 'N/A',
            priority=priority
        )
        
        return xml
        
    except Exception as e:
        logger.error(f"Auto model selection error: {e}")
        return f'''<error>
  <message>Auto model selection failed: {str(e)}</message>
</error>'''


def _get_category(model_name: str) -> str:
    """Get model category."""
    if model_name in STATISTICAL_MODELS:
        return "statistical"
    elif model_name in ML_MODELS:
        return "machine_learning"
    elif model_name in DEEP_LEARNING_MODELS:
        return "deep_learning"
    return "unknown"
