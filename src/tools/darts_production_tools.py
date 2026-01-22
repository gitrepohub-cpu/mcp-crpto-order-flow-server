"""
Darts Production Tools - MCP Tool Implementations

Tools for production-grade backtesting and model validation.
Essential for validating model performance before live trading.

Tool Categories:
================
1. Backtesting: Walk-forward validation
2. Performance Analysis: Risk-adjusted metrics
"""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd

# Import Darts components
from src.integrations.darts_bridge import (
    DartsBridge,
    DartsModelWrapper,
    ALL_MODELS,
    STATISTICAL_MODELS,
    ML_MODELS
)

# Import data loading
from src.tools.darts_tools import _load_price_data

logger = logging.getLogger(__name__)


# ============================================================================
# TOOL 12: PRODUCTION BACKTESTING
# ============================================================================

async def backtest_model(
    symbol: str,
    exchange: str = "binance",
    model_name: str = "Theta",
    horizon: int = 24,
    data_hours: int = 720,
    train_ratio: float = 0.8,
    walk_forward_steps: int = 5
) -> str:
    """
    Production-grade walk-forward backtesting.
    
    Tests model performance by simulating real trading conditions:
    - Trains on historical data
    - Predicts forward
    - Measures actual vs predicted performance
    - Repeats with expanding window
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        exchange: Exchange name (default: "binance")
        model_name: Model to backtest (any Darts model)
        horizon: Forecast horizon in hours (default: 24)
        data_hours: Total historical data hours (default: 720 = 30 days)
        train_ratio: Initial training data ratio (default: 0.8)
        walk_forward_steps: Number of walk-forward iterations (default: 5)
    
    Returns:
        XML formatted backtest results with performance metrics
    """
    start_time = time.time()
    
    if model_name not in ALL_MODELS:
        return f'''<error>
  <message>Unknown model: {model_name}</message>
  <available_models>{', '.join(sorted(ALL_MODELS))}</available_models>
</error>'''
    
    try:
        # Load data
        df = _load_price_data(symbol, exchange, "futures", data_hours)
        
        if df is None or len(df) < 96:
            return f'''<error>
  <message>Insufficient data for backtesting</message>
  <minimum_required>96 hours</minimum_required>
  <available>{len(df) if df is not None else 0} hours</available>
</error>'''
        
        # Initialize bridge
        bridge = DartsBridge()
        darts_ts = bridge.to_darts(df, time_col='timestamp', value_col='price')
        
        # Run walk-forward backtest
        backtest_results = _run_walk_forward_backtest(
            darts_ts=darts_ts,
            prices=df['price'].values,
            model_name=model_name,
            horizon=horizon,
            train_ratio=train_ratio,
            steps=walk_forward_steps
        )
        
        # Calculate aggregate metrics
        aggregate_metrics = _calculate_aggregate_metrics(backtest_results)
        
        # Performance grade
        grade = _grade_performance(aggregate_metrics)
        
        processing_time = time.time() - start_time
        
        # Build XML response
        xml = f'''<darts_backtest_results symbol="{symbol.upper()}" exchange="{exchange}" timestamp="{datetime.utcnow().isoformat()}">
  <configuration>
    <model>{model_name}</model>
    <horizon_hours>{horizon}</horizon_hours>
    <total_data_hours>{data_hours}</total_data_hours>
    <train_ratio>{train_ratio}</train_ratio>
    <walk_forward_steps>{walk_forward_steps}</walk_forward_steps>
    <processing_time_seconds>{processing_time:.2f}</processing_time_seconds>
  </configuration>
  
  <aggregate_performance>
    <mae>{aggregate_metrics['mae']:.4f}</mae>
    <rmse>{aggregate_metrics['rmse']:.4f}</rmse>
    <mape>{aggregate_metrics['mape']:.2f}%</mape>
    <direction_accuracy>{aggregate_metrics['direction_accuracy']:.1f}%</direction_accuracy>
    <max_error>{aggregate_metrics['max_error']:.4f}</max_error>
    <sharpe_ratio>{aggregate_metrics['sharpe_ratio']:.3f}</sharpe_ratio>
  </aggregate_performance>
  
  <performance_grade>
    <overall>{grade['overall']}</overall>
    <accuracy_grade>{grade['accuracy']}</accuracy_grade>
    <direction_grade>{grade['direction']}</direction_grade>
    <risk_grade>{grade['risk']}</risk_grade>
    <recommendation>{grade['recommendation']}</recommendation>
  </performance_grade>
  
  <walk_forward_iterations>
'''
        
        for i, result in enumerate(backtest_results, 1):
            xml += f'''    <iteration number="{i}">
      <train_size>{result['train_size']}</train_size>
      <test_size>{result['test_size']}</test_size>
      <mae>{result['mae']:.4f}</mae>
      <mape>{result['mape']:.2f}%</mape>
      <direction_accuracy>{result['direction_accuracy']:.1f}%</direction_accuracy>
      <predicted_direction>{'UP' if result['predicted_end'] > result['predicted_start'] else 'DOWN'}</predicted_direction>
      <actual_direction>{'UP' if result['actual_end'] > result['actual_start'] else 'DOWN'}</actual_direction>
    </iteration>
'''
        
        xml += '''  </walk_forward_iterations>
  
  <trading_simulation>
'''
        
        # Simulate trading
        trading_results = _simulate_trading(backtest_results)
        
        xml += f'''    <total_trades>{trading_results['total_trades']}</total_trades>
    <winning_trades>{trading_results['winning_trades']}</winning_trades>
    <losing_trades>{trading_results['losing_trades']}</losing_trades>
    <win_rate>{trading_results['win_rate']:.1f}%</win_rate>
    <total_return>{trading_results['total_return']:.2f}%</total_return>
    <max_drawdown>{trading_results['max_drawdown']:.2f}%</max_drawdown>
    <profit_factor>{trading_results['profit_factor']:.2f}</profit_factor>
  </trading_simulation>
  
  <summary>
    <conclusion>{_generate_backtest_conclusion(grade, aggregate_metrics, trading_results)}</conclusion>
    <use_case>{_suggest_use_case(grade, model_name)}</use_case>
  </summary>
</darts_backtest_results>'''
        
        return xml
        
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        return f'''<error>
  <message>Backtest failed: {str(e)}</message>
</error>'''


def _run_walk_forward_backtest(
    darts_ts: Any,
    prices: np.ndarray,
    model_name: str,
    horizon: int,
    train_ratio: float,
    steps: int
) -> List[Dict[str, Any]]:
    """Run walk-forward backtesting."""
    results = []
    total_len = len(prices)
    initial_train_size = int(total_len * train_ratio)
    test_size = (total_len - initial_train_size) // steps
    
    for step in range(steps):
        # Calculate split points
        train_end = initial_train_size + (step * test_size)
        test_end = min(train_end + horizon, total_len)
        
        if test_end > total_len or train_end >= total_len:
            break
        
        # Split data
        train_prices = prices[:train_end]
        test_prices = prices[train_end:test_end]
        
        if len(test_prices) < 2:
            continue
        
        # Create and train model
        bridge = DartsBridge()
        train_df = pd.DataFrame({
            'timestamp': pd.date_range(end=datetime.utcnow(), periods=len(train_prices), freq='h'),
            'price': train_prices
        })
        train_ts = bridge.to_darts(train_df, time_col='timestamp', value_col='price')
        
        if model_name in ML_MODELS:
            wrapper = DartsModelWrapper(model_name, lags=24, output_chunk_length=12)
        else:
            wrapper = DartsModelWrapper(model_name)
        
        try:
            wrapper.fit(train_ts)
            forecast_result = wrapper.predict(horizon=len(test_prices))
            predicted = forecast_result.forecast.values
        except Exception as e:
            logger.warning(f"Model fit failed in step {step}: {e}")
            continue
        
        # Calculate metrics
        actual = test_prices[:len(predicted)]
        
        mae = np.mean(np.abs(predicted - actual))
        mape = np.mean(np.abs((predicted - actual) / actual)) * 100
        
        # Direction accuracy
        pred_direction = 1 if predicted[-1] > predicted[0] else -1
        actual_direction = 1 if actual[-1] > actual[0] else -1
        direction_correct = pred_direction == actual_direction
        
        results.append({
            'step': step,
            'train_size': train_end,
            'test_size': len(actual),
            'mae': mae,
            'mape': mape,
            'direction_accuracy': 100.0 if direction_correct else 0.0,
            'predicted_start': predicted[0],
            'predicted_end': predicted[-1],
            'actual_start': actual[0],
            'actual_end': actual[-1],
            'predicted_return': (predicted[-1] - predicted[0]) / predicted[0] * 100,
            'actual_return': (actual[-1] - actual[0]) / actual[0] * 100
        })
    
    return results


def _calculate_aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate aggregate metrics from backtest results."""
    if not results:
        return {
            'mae': 0, 'rmse': 0, 'mape': 0, 
            'direction_accuracy': 0, 'max_error': 0, 'sharpe_ratio': 0
        }
    
    maes = [r['mae'] for r in results]
    mapes = [r['mape'] for r in results]
    directions = [r['direction_accuracy'] for r in results]
    
    # Calculate returns for Sharpe ratio
    actual_returns = [r['actual_return'] for r in results]
    
    return {
        'mae': np.mean(maes),
        'rmse': np.sqrt(np.mean(np.array(maes) ** 2)),
        'mape': np.mean(mapes),
        'direction_accuracy': np.mean(directions),
        'max_error': max(maes),
        'sharpe_ratio': np.mean(actual_returns) / (np.std(actual_returns) + 0.001) if actual_returns else 0
    }


def _grade_performance(metrics: Dict[str, float]) -> Dict[str, str]:
    """Grade the performance based on metrics."""
    # Accuracy grade (based on MAPE)
    mape = metrics['mape']
    if mape < 1:
        accuracy = 'A'
    elif mape < 2:
        accuracy = 'B'
    elif mape < 5:
        accuracy = 'C'
    else:
        accuracy = 'D'
    
    # Direction grade
    dir_acc = metrics['direction_accuracy']
    if dir_acc >= 80:
        direction = 'A'
    elif dir_acc >= 60:
        direction = 'B'
    elif dir_acc >= 50:
        direction = 'C'
    else:
        direction = 'D'
    
    # Risk grade (based on max error / mae ratio)
    risk_ratio = metrics['max_error'] / (metrics['mae'] + 0.001)
    if risk_ratio < 1.5:
        risk = 'A'
    elif risk_ratio < 2:
        risk = 'B'
    elif risk_ratio < 3:
        risk = 'C'
    else:
        risk = 'D'
    
    # Overall grade
    grades = {'A': 4, 'B': 3, 'C': 2, 'D': 1}
    avg = (grades[accuracy] + grades[direction] + grades[risk]) / 3
    
    if avg >= 3.5:
        overall = 'A'
        rec = "Model performs well - suitable for production with monitoring"
    elif avg >= 2.5:
        overall = 'B'
        rec = "Model performs reasonably - consider using with other signals"
    elif avg >= 1.5:
        overall = 'C'
        rec = "Model has limitations - use with caution and ensemble with others"
    else:
        overall = 'D'
        rec = "Model underperforms - not recommended for this symbol/timeframe"
    
    return {
        'overall': overall,
        'accuracy': accuracy,
        'direction': direction,
        'risk': risk,
        'recommendation': rec
    }


def _simulate_trading(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Simulate trading based on predictions."""
    if not results:
        return {
            'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
            'win_rate': 0, 'total_return': 0, 'max_drawdown': 0, 'profit_factor': 1
        }
    
    total_trades = len(results)
    winning = 0
    returns = []
    
    for r in results:
        # Assume we trade in predicted direction
        pred_dir = 1 if r['predicted_end'] > r['predicted_start'] else -1
        actual_return = r['actual_return']
        
        # If we bet on up and price went up, or bet on down and price went down
        trade_return = actual_return * pred_dir
        returns.append(trade_return)
        
        if trade_return > 0:
            winning += 1
    
    losing = total_trades - winning
    
    # Calculate max drawdown
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
    
    # Profit factor
    gains = sum(r for r in returns if r > 0)
    losses = abs(sum(r for r in returns if r < 0))
    profit_factor = gains / (losses + 0.001)
    
    return {
        'total_trades': total_trades,
        'winning_trades': winning,
        'losing_trades': losing,
        'win_rate': (winning / total_trades * 100) if total_trades > 0 else 0,
        'total_return': sum(returns),
        'max_drawdown': max_drawdown,
        'profit_factor': profit_factor
    }


def _generate_backtest_conclusion(
    grade: Dict[str, str], 
    metrics: Dict[str, float],
    trading: Dict[str, Any]
) -> str:
    """Generate conclusion text."""
    overall = grade['overall']
    
    if overall == 'A':
        return f"Excellent performance with {metrics['direction_accuracy']:.0f}% direction accuracy and {trading['win_rate']:.0f}% win rate. Model is suitable for production use."
    elif overall == 'B':
        return f"Good performance with {metrics['direction_accuracy']:.0f}% direction accuracy. Consider combining with other signals for better reliability."
    elif overall == 'C':
        return f"Moderate performance with {metrics['mape']:.1f}% average error. Model may work in certain market conditions but requires careful monitoring."
    else:
        return f"Model underperformed with only {metrics['direction_accuracy']:.0f}% direction accuracy. Consider trying different models or parameters."


def _suggest_use_case(grade: Dict[str, str], model_name: str) -> str:
    """Suggest appropriate use case."""
    overall = grade['overall']
    
    if overall in ['A', 'B']:
        if model_name in ML_MODELS:
            return "Use for short-term momentum signals with feature-based decision support"
        else:
            return "Use for trend-following strategies and medium-term position sizing"
    else:
        return "Use only as one input among many signals - not recommended as primary indicator"


# ============================================================================
# BONUS: MULTI-MODEL BACKTEST COMPARISON
# ============================================================================

async def compare_models_backtest(
    symbol: str,
    exchange: str = "binance",
    models: Optional[List[str]] = None,
    horizon: int = 24,
    data_hours: int = 720
) -> str:
    """
    Compare multiple models using backtesting.
    
    Runs backtests for multiple models and ranks them by performance.
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        exchange: Exchange name (default: "binance")
        models: List of models to compare (default: top 5 models)
        horizon: Forecast horizon in hours (default: 24)
        data_hours: Historical data hours (default: 720)
    
    Returns:
        XML formatted comparison with rankings
    """
    start_time = time.time()
    
    # Default model selection
    if models is None:
        models = ['Theta', 'ExponentialSmoothing', 'XGBoost', 'Prophet', 'NaiveDrift']
    
    # Validate models
    invalid_models = [m for m in models if m not in ALL_MODELS]
    if invalid_models:
        return f'''<error>
  <message>Unknown models: {', '.join(invalid_models)}</message>
</error>'''
    
    try:
        # Load data
        df = _load_price_data(symbol, exchange, "futures", data_hours)
        
        if df is None or len(df) < 96:
            return f'''<error>
  <message>Insufficient data for comparison</message>
</error>'''
        
        # Run backtests for each model
        model_results = []
        
        for model_name in models:
            try:
                bridge = DartsBridge()
                darts_ts = bridge.to_darts(df, time_col='timestamp', value_col='price')
                
                results = _run_walk_forward_backtest(
                    darts_ts=darts_ts,
                    prices=df['price'].values,
                    model_name=model_name,
                    horizon=horizon,
                    train_ratio=0.8,
                    steps=3  # Fewer steps for speed
                )
                
                metrics = _calculate_aggregate_metrics(results)
                grade = _grade_performance(metrics)
                
                model_results.append({
                    'model': model_name,
                    'metrics': metrics,
                    'grade': grade['overall']
                })
                
            except Exception as e:
                logger.warning(f"Failed to backtest {model_name}: {e}")
                model_results.append({
                    'model': model_name,
                    'metrics': None,
                    'grade': 'F',
                    'error': str(e)
                })
        
        # Rank models
        valid_results = [r for r in model_results if r['metrics'] is not None]
        ranked = sorted(valid_results, key=lambda x: (
            -x['metrics']['direction_accuracy'],
            x['metrics']['mape']
        ))
        
        processing_time = time.time() - start_time
        
        # Build XML
        xml = f'''<darts_model_comparison symbol="{symbol.upper()}" exchange="{exchange}" timestamp="{datetime.utcnow().isoformat()}">
  <configuration>
    <models_tested>{len(models)}</models_tested>
    <horizon_hours>{horizon}</horizon_hours>
    <data_hours>{data_hours}</data_hours>
    <processing_time_seconds>{processing_time:.2f}</processing_time_seconds>
  </configuration>
  
  <rankings>
'''
        
        for rank, result in enumerate(ranked, 1):
            m = result['metrics']
            xml += f'''    <model rank="{rank}">
      <name>{result['model']}</name>
      <grade>{result['grade']}</grade>
      <direction_accuracy>{m['direction_accuracy']:.1f}%</direction_accuracy>
      <mape>{m['mape']:.2f}%</mape>
      <sharpe_ratio>{m['sharpe_ratio']:.3f}</sharpe_ratio>
    </model>
'''
        
        xml += '''  </rankings>
  
  <recommendation>
'''
        
        if ranked:
            best = ranked[0]
            xml += f'''    <best_model>{best['model']}</best_model>
    <reason>Highest direction accuracy ({best['metrics']['direction_accuracy']:.1f}%) with acceptable error ({best['metrics']['mape']:.2f}% MAPE)</reason>
'''
        
        xml += '''  </recommendation>
</darts_model_comparison>'''
        
        return xml
        
    except Exception as e:
        logger.error(f"Model comparison error: {e}")
        return f'''<error>
  <message>Model comparison failed: {str(e)}</message>
</error>'''
