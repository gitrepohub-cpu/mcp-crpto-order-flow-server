"""
Funding Rate Forecast Calculator

Forecast funding rates using time series analysis.
"""

import logging
from typing import Dict, List, Optional, Any

from src.features.base import FeatureCalculator, FeatureResult
from src.features.utils import generate_signal
from src.analytics.timeseries_engine import (
    get_timeseries_engine,
    TimeSeriesData
)

logger = logging.getLogger(__name__)


class FundingForecastCalculator(FeatureCalculator):
    """
    Forecast funding rates for perpetual futures.
    
    Uses multiple models:
        - ARIMA for trend/cycle capture
        - Exponential Smoothing for smoothness
        - Auto-selection for best model
    
    Provides arbitrage signals when predicted funding diverges.
    """
    
    name = "funding_forecast"
    description = "Forecast funding rates using time series analysis"
    category = "forecasting"
    version = "1.0.0"
    
    async def calculate(
        self,
        symbol: str,
        exchange: Optional[str] = None,
        hours: int = 168,
        forecast_periods: int = 8,
        model: str = "auto",
        confidence: float = 0.95,
        include_seasonality: bool = True,
        **params
    ) -> FeatureResult:
        """
        Forecast funding rates.
        
        Args:
            symbol: Trading pair
            exchange: Exchange
            hours: Hours of historical data (default 7 days)
            forecast_periods: Number of periods to forecast (8h periods)
            model: arima, exponential_smoothing, or auto
            confidence: Confidence interval (default 95%)
            include_seasonality: Include seasonality analysis
        """
        symbol_lower = symbol.lower()
        exc = (exchange or 'binance').lower()
        
        engine = get_timeseries_engine()
        
        table_name = self.get_table_name(symbol_lower, exc, 'futures', 'funding_rates')
        
        if not self.table_exists(table_name):
            return self.create_result(
                symbol=symbol,
                exchanges=[exc],
                data={},
                errors=[f"No funding rate data found for {symbol}"]
            )
        
        try:
            query = f"""
                SELECT timestamp, funding_rate as value
                FROM {table_name}
                WHERE timestamp >= NOW() - INTERVAL '{hours} hours'
                ORDER BY timestamp
            """
            
            results = self.execute_query(query)
            
            if len(results) < 20:
                return self.create_result(
                    symbol=symbol,
                    exchanges=[exc],
                    data={},
                    errors=["Insufficient funding rate data for forecasting"]
                )
            
            ts = TimeSeriesData.from_duckdb_result(results, name='funding_rate')
            # Funding rates are typically 8h intervals, don't resample
            
            # Generate forecast
            if model == 'auto':
                forecast_result = engine.auto_forecast(
                    ts,
                    forecast_steps=forecast_periods,
                    confidence=confidence
                )
            elif model == 'arima':
                forecast_result = engine.forecast_arima(
                    ts,
                    forecast_steps=forecast_periods,
                    confidence=confidence
                )
            elif model == 'exponential_smoothing':
                forecast_result = engine.forecast_exponential_smoothing(
                    ts,
                    forecast_steps=forecast_periods,
                    confidence=confidence
                )
            else:
                forecast_result = engine.auto_forecast(
                    ts,
                    forecast_steps=forecast_periods,
                    confidence=confidence
                )
            
            # Analyze forecast for trading signals
            forecast_analysis = self._analyze_forecast(
                ts,
                forecast_result,
                forecast_periods
            )
            
            # Optional seasonality analysis
            seasonality_info = {}
            if include_seasonality:
                try:
                    seasonality = engine.detect_seasonality(ts, top_n=3)
                    seasonality_info = {
                        'has_seasonality': seasonality.get('has_seasonality', False),
                        'dominant_period': seasonality.get('dominant_period'),
                        'interpretation': self._interpret_funding_seasonality(
                            seasonality.get('dominant_period')
                        )
                    }
                except Exception as e:
                    logger.warning(f"Seasonality analysis failed: {e}")
            
            # Generate signals
            signals = self._generate_funding_signals(
                ts,
                forecast_result,
                forecast_analysis
            )
            
            return self.create_result(
                symbol=symbol,
                exchanges=[exc],
                data={
                    'forecast': forecast_result.to_dict(),
                    'analysis': forecast_analysis,
                    'seasonality': seasonality_info,
                    'current_funding': {
                        'last_rate': float(ts.values[-1]) if len(ts.values) > 0 else None,
                        'mean_rate': float(ts.values.mean()),
                        'std_rate': float(ts.values.std()),
                        'min_rate': float(ts.values.min()),
                        'max_rate': float(ts.values.max())
                    },
                    'data_stats': {
                        'observations': len(ts),
                        'time_range_hours': hours,
                        'forecast_periods': forecast_periods
                    }
                },
                metadata={
                    'model': forecast_result.model_name,
                    'confidence': confidence
                },
                signals=signals
            )
            
        except Exception as e:
            logger.error(f"Funding forecast failed: {e}")
            return self.create_result(
                symbol=symbol,
                exchanges=[exc],
                data={},
                errors=[str(e)]
            )
    
    def _analyze_forecast(
        self,
        ts: TimeSeriesData,
        forecast,
        periods: int
    ) -> Dict:
        """Analyze forecast for trading insights."""
        import numpy as np
        
        analysis = {}
        
        # Current vs forecast comparison
        current = ts.values[-1] if len(ts.values) > 0 else 0
        forecast_mean = np.mean(forecast.forecast)
        forecast_end = forecast.forecast[-1] if len(forecast.forecast) > 0 else 0
        
        analysis['direction'] = 'increasing' if forecast_end > current else 'decreasing' if forecast_end < current else 'stable'
        analysis['change_pct'] = ((forecast_end - current) / abs(current) * 100) if current != 0 else 0
        analysis['forecast_mean'] = float(forecast_mean)
        analysis['forecast_range'] = {
            'min': float(np.min(forecast.forecast)),
            'max': float(np.max(forecast.forecast))
        }
        
        # Annualized funding
        # Funding is typically 8h, so 3 per day, 1095 per year
        annualized_current = current * 1095 * 100  # As percentage
        annualized_forecast = forecast_mean * 1095 * 100
        
        analysis['annualized'] = {
            'current_apr': annualized_current,
            'forecast_apr': annualized_forecast
        }
        
        # Trading implications
        analysis['implications'] = []
        
        if forecast_mean > 0.0003:  # >0.03% or >32.85% APR
            analysis['implications'].append(
                "High positive funding - longs paying shorts, consider short bias"
            )
        elif forecast_mean < -0.0003:
            analysis['implications'].append(
                "High negative funding - shorts paying longs, consider long bias"
            )
        
        if analysis['direction'] == 'increasing' and current > 0:
            analysis['implications'].append(
                "Funding increasing positive - market getting more bullish/overleveraged"
            )
        elif analysis['direction'] == 'decreasing' and current < 0:
            analysis['implications'].append(
                "Funding decreasing negative - short squeeze risk reducing"
            )
        
        return analysis
    
    def _interpret_funding_seasonality(self, period: Optional[float]) -> str:
        """Interpret funding rate seasonality."""
        if period is None:
            return "No clear seasonality detected"
        
        # Funding is typically 8h intervals
        if 2.5 <= period <= 3.5:  # ~3 periods = 24h
            return "Daily cycle detected - funding resets daily"
        elif 20 <= period <= 22:  # ~21 periods = 7 days
            return "Weekly cycle detected - weekly pattern in funding"
        elif 0.8 <= period <= 1.2:
            return "Each funding period varies - high variability"
        else:
            return f"Cycle of approximately {period * 8:.0f} hours detected"
    
    def _generate_funding_signals(
        self,
        ts: TimeSeriesData,
        forecast,
        analysis: Dict
    ) -> List[Dict]:
        """Generate trading signals from funding forecast."""
        import numpy as np
        
        signals = []
        
        current = ts.values[-1] if len(ts.values) > 0 else 0
        forecast_mean = analysis.get('forecast_mean', 0)
        annualized = analysis.get('annualized', {})
        
        # Extreme funding signal
        if abs(current) > 0.001:  # >0.1% or >109.5% APR
            signal_type = 'WARNING'
            direction = "positive" if current > 0 else "negative"
            signals.append(generate_signal(
                signal_type, min(abs(current) * 500, 1.0),
                f"Extreme {direction} funding: {current:.4%} ({annualized.get('current_apr', 0):.1f}% APR)",
                {'funding_rate': current, 'apr': annualized.get('current_apr')}
            ))
        
        # Funding direction change signal
        if (current > 0 and forecast_mean < 0) or (current < 0 and forecast_mean > 0):
            signals.append(generate_signal(
                'INFO', 0.7,
                f"Funding direction change predicted: {current:.4%} → {forecast_mean:.4%}",
                {'current': current, 'forecast': forecast_mean}
            ))
        
        # Cash-and-carry opportunity
        if forecast_mean > 0.0002:  # >21.9% APR
            signals.append(generate_signal(
                'BULLISH', min(forecast_mean * 2000, 1.0),
                f"Cash-and-carry opportunity: collect {annualized.get('forecast_apr', 0):.1f}% APR by shorting perp",
                {'strategy': 'short_perp_long_spot', 'apr': annualized.get('forecast_apr')}
            ))
        elif forecast_mean < -0.0002:
            signals.append(generate_signal(
                'BEARISH', min(abs(forecast_mean) * 2000, 1.0),
                f"Reverse cash-carry: collect {abs(annualized.get('forecast_apr', 0)):.1f}% APR by longing perp",
                {'strategy': 'long_perp_short_spot', 'apr': abs(annualized.get('forecast_apr'))}
            ))
        
        # High forecast uncertainty
        if forecast.lower_bound is not None and forecast.upper_bound is not None:
            uncertainty = np.mean(forecast.upper_bound) - np.mean(forecast.lower_bound)
            if uncertainty > 0.001:  # Wide confidence interval
                signals.append(generate_signal(
                    'WARNING', min(uncertainty * 500, 1.0),
                    f"High funding uncertainty: ±{uncertainty:.4%} range",
                    {'uncertainty': uncertainty}
                ))
        
        return signals
    
    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        return {
            'symbol': {
                'type': 'str',
                'default': 'BTCUSDT',
                'description': 'Trading pair',
                'required': True
            },
            'exchange': {
                'type': 'str',
                'default': 'binance',
                'description': 'Exchange',
                'required': False
            },
            'hours': {
                'type': 'int',
                'default': 168,
                'description': 'Hours of historical data',
                'required': False
            },
            'forecast_periods': {
                'type': 'int',
                'default': 8,
                'description': 'Number of 8h periods to forecast',
                'required': False
            },
            'model': {
                'type': 'str',
                'default': 'auto',
                'description': 'Model: arima, exponential_smoothing, auto',
                'required': False
            },
            'confidence': {
                'type': 'float',
                'default': 0.95,
                'description': 'Confidence interval',
                'required': False
            },
            'include_seasonality': {
                'type': 'bool',
                'default': True,
                'description': 'Include seasonality analysis',
                'required': False
            }
        }
