"""
Price Forecasting Calculator

Uses ARIMA, Exponential Smoothing, and Theta models to forecast
cryptocurrency prices from historical DuckDB data.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from src.features.base import FeatureCalculator, FeatureResult
from src.features.utils import generate_signal
from src.analytics.timeseries_engine import (
    get_timeseries_engine,
    TimeSeriesData,
    ForecastResult
)

logger = logging.getLogger(__name__)


class PriceForecastCalculator(FeatureCalculator):
    """
    Forecast cryptocurrency prices using time series models.
    
    Models:
        - ARIMA (Auto-Regressive Integrated Moving Average)
        - Exponential Smoothing (Holt-Winters)
        - Theta method
        - Auto-selection (best model)
    """
    
    name = "price_forecast"
    description = "Forecast future prices using ARIMA, Exponential Smoothing, and Theta models"
    category = "forecasting"
    version = "1.0.0"
    
    async def calculate(
        self,
        symbol: str,
        exchange: Optional[str] = None,
        hours: int = 168,
        forecast_steps: int = 24,
        model: str = "auto",
        resample_freq: str = "1h",
        confidence: float = 0.95,
        **params
    ) -> FeatureResult:
        """
        Forecast prices.
        
        Args:
            symbol: Trading pair
            exchange: Exchange (default: binance)
            hours: Hours of history to use
            forecast_steps: Number of steps to forecast
            model: 'arima', 'exp_smoothing', 'theta', or 'auto'
            resample_freq: Resample frequency ('1h', '4h', '1d')
            confidence: Confidence level for intervals
        """
        symbol_lower = symbol.lower()
        exc = (exchange or 'binance').lower()
        
        engine = get_timeseries_engine()
        
        # Get historical price data
        table_name = self.get_table_name(symbol_lower, exc, 'futures', 'prices')
        
        if not self.table_exists(table_name):
            return self.create_result(
                symbol=symbol,
                exchanges=[exc],
                data={},
                errors=[f"No price data found for {symbol} on {exc}"]
            )
        
        try:
            query = f"""
                SELECT timestamp, mid_price
                FROM {table_name}
                WHERE timestamp >= NOW() - INTERVAL '{hours} hours'
                ORDER BY timestamp
            """
            
            results = self.execute_query(query)
            
            if len(results) < 50:
                return self.create_result(
                    symbol=symbol,
                    exchanges=[exc],
                    data={},
                    errors=["Insufficient data for forecasting (need at least 50 points)"]
                )
            
            # Convert to TimeSeriesData
            ts = TimeSeriesData.from_duckdb_result(results, name="price")
            
            # Resample to desired frequency
            ts = ts.resample(resample_freq)
            
            # Run forecast
            if model == "arima":
                forecast_result = engine.forecast_arima(ts, forecast_steps, confidence=confidence)
            elif model == "exp_smoothing":
                forecast_result = engine.forecast_exponential_smoothing(ts, forecast_steps, confidence=confidence)
            elif model == "theta":
                forecast_result = engine.forecast_theta(ts, forecast_steps, confidence=confidence)
            else:
                forecast_result = engine.auto_forecast(ts, forecast_steps, confidence=confidence)
            
            # Current price
            current_price = ts.value.iloc[-1]
            
            # Forecast direction and magnitude
            final_forecast = forecast_result.forecast.iloc[-1]
            price_change = final_forecast - current_price
            price_change_pct = (price_change / current_price) * 100
            
            # Generate signals
            signals = []
            if price_change_pct > 2:
                signals.append(generate_signal(
                    'BULLISH', min(price_change_pct / 10, 1.0),
                    f"Price forecast: +{price_change_pct:.2f}% ({forecast_result.model_name})",
                    {'forecast_pct': price_change_pct}
                ))
            elif price_change_pct < -2:
                signals.append(generate_signal(
                    'BEARISH', min(abs(price_change_pct) / 10, 1.0),
                    f"Price forecast: {price_change_pct:.2f}% ({forecast_result.model_name})",
                    {'forecast_pct': price_change_pct}
                ))
            
            # Build forecast series for output
            forecast_data = []
            for i, (idx, val) in enumerate(forecast_result.forecast.items()):
                forecast_data.append({
                    'timestamp': str(idx),
                    'forecast': val,
                    'lower': forecast_result.lower_bound.iloc[i],
                    'upper': forecast_result.upper_bound.iloc[i]
                })
            
            return self.create_result(
                symbol=symbol,
                exchanges=[exc],
                data={
                    'current_price': current_price,
                    'model_used': forecast_result.model_name,
                    'forecast_summary': {
                        'final_forecast': final_forecast,
                        'price_change': price_change,
                        'price_change_pct': price_change_pct,
                        'direction': 'UP' if price_change > 0 else 'DOWN'
                    },
                    'confidence_level': confidence,
                    'metrics': forecast_result.metrics,
                    'forecast_series': forecast_data,
                    'input_data': {
                        'points_used': len(ts),
                        'resample_freq': resample_freq,
                        'history_hours': hours
                    }
                },
                metadata={
                    'model': model,
                    'forecast_steps': forecast_steps,
                    'confidence': confidence
                },
                signals=signals
            )
            
        except Exception as e:
            logger.error(f"Price forecast failed: {e}")
            return self.create_result(
                symbol=symbol,
                exchanges=[exc],
                data={},
                errors=[str(e)]
            )
    
    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            'symbol': {
                'type': 'str',
                'default': 'BTCUSDT',
                'description': 'Trading pair to forecast',
                'required': True
            },
            'exchange': {
                'type': 'str',
                'default': 'binance',
                'description': 'Exchange for historical data',
                'required': False
            },
            'hours': {
                'type': 'int',
                'default': 168,
                'description': 'Hours of history to use (default 7 days)',
                'required': False
            },
            'forecast_steps': {
                'type': 'int',
                'default': 24,
                'description': 'Number of periods to forecast',
                'required': False
            },
            'model': {
                'type': 'str',
                'default': 'auto',
                'description': 'Model: arima, exp_smoothing, theta, or auto',
                'required': False
            },
            'resample_freq': {
                'type': 'str',
                'default': '1h',
                'description': 'Resample frequency (1h, 4h, 1d)',
                'required': False
            },
            'confidence': {
                'type': 'float',
                'default': 0.95,
                'description': 'Confidence level for prediction intervals',
                'required': False
            }
        }
