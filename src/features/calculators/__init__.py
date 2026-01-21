"""
Feature Calculators Package

Add your feature calculation scripts here. They will be auto-discovered
and registered as MCP tools.

ALL CALCULATORS USE THE TIME SERIES ENGINE FOR ANALYSIS:
    - The TimeSeriesEngine is available via self.timeseries_engine
    - Use self.create_timeseries_data() to convert DuckDB results
    - Available methods: forecast, detect_anomalies, detect_change_points,
      extract_features, detect_seasonality, detect_regime

To create a new calculator:
    1. Create a new .py file in this directory
    2. Import and subclass FeatureCalculator from src.features.base
    3. Implement the required methods
    4. USE self.timeseries_engine for advanced time series analysis
    5. The system auto-discovers and registers it

Example (my_feature.py):
    
    from src.features.base import FeatureCalculator, FeatureResult
    
    class MyFeatureCalculator(FeatureCalculator):
        name = "my_feature"
        description = "Calculates my custom market feature"
        category = "custom"
        version = "1.0.0"
        
        async def calculate(self, symbol, exchange=None, **params):
            # Query data from DuckDB
            results = self.execute_query("SELECT timestamp, value FROM ...")
            
            # USE TIME SERIES ENGINE FOR ANALYSIS
            ts_data = self.create_timeseries_data(results, name='my_metric')
            
            # Forecasting
            forecast = self.timeseries_engine.auto_forecast(ts_data, forecast_steps=24)
            
            # Anomaly detection
            anomalies = self.timeseries_engine.detect_anomalies_zscore(ts_data)
            
            # Feature extraction
            features = self.timeseries_engine.extract_features(ts_data)
            
            # Regime detection
            regime = self.timeseries_engine.detect_regime(ts_data)
            
            data = {
                'value': 123,
                'forecast': list(forecast.forecast),
                'features': features,
                'regime': regime.current_regime.value
            }
            return self.create_result(symbol, [exchange or 'all'], data)
        
        def get_parameters(self):
            return {
                'symbol': {'type': 'str', 'required': True},
                'hours': {'type': 'int', 'default': 24, 'description': 'Hours of data'},
            }

Available TimeSeriesEngine Methods:
    FORECASTING:
        - forecast_arima(ts_data, forecast_steps, confidence)
        - forecast_exponential_smoothing(ts_data, forecast_steps, confidence)
        - forecast_theta(ts_data, forecast_steps)
        - auto_forecast(ts_data, forecast_steps, confidence)
    
    ANOMALY DETECTION:
        - detect_anomalies_zscore(ts_data, threshold)
        - detect_anomalies_iqr(ts_data, multiplier)
        - detect_anomalies_isolation_forest(ts_data, contamination)
        - detect_anomalies_cusum(ts_data, threshold)
    
    CHANGE POINT DETECTION:
        - detect_change_points_cusum(ts_data, threshold)
        - detect_change_points_binary_segmentation(ts_data, max_changepoints)
    
    FEATURE EXTRACTION:
        - extract_features(ts_data, include_advanced)  # 40+ features
    
    SEASONALITY:
        - detect_seasonality(ts_data, top_n)
        - decompose_seasonality(ts_data, period)
    
    REGIME DETECTION:
        - detect_regime(ts_data, lookback, volatility_threshold)
"""
