"""
Feature Calculators Package

Add your feature calculation scripts here. They will be auto-discovered
and registered as MCP tools.

To create a new calculator:
    1. Create a new .py file in this directory
    2. Import and subclass FeatureCalculator from src.features.base
    3. Implement the required methods
    4. The system auto-discovers and registers it

Example (my_feature.py):
    
    from src.features.base import FeatureCalculator, FeatureResult
    
    class MyFeatureCalculator(FeatureCalculator):
        name = "my_feature"
        description = "Calculates my custom market feature"
        category = "custom"
        version = "1.0.0"
        
        async def calculate(self, symbol, exchange=None, **params):
            # Your calculation logic here
            data = {'value': 123}
            return self.create_result(symbol, [exchange or 'all'], data)
        
        def get_parameters(self):
            return {
                'hours': {'type': 'int', 'default': 24, 'description': 'Hours of data'},
            }
"""
