"""
Advanced Feature Calculation Framework

This package provides a plugin-based architecture for calculating advanced market features
from stored DuckDB data. New feature scripts can be added to the `calculators/` directory
and will be automatically discovered and registered as MCP tools.

Architecture:
    - base.py: Base classes for feature calculators
    - registry.py: Plugin discovery and registration system
    - calculators/: Directory for feature calculation scripts
    - utils.py: Shared utilities for feature calculations

Usage:
    1. Create a new calculator by subclassing FeatureCalculator in calculators/
    2. Implement the required methods (calculate, get_schema)
    3. The system auto-discovers and registers it as an MCP tool

Example:
    from src.features.base import FeatureCalculator
    
    class MyFeatureCalculator(FeatureCalculator):
        name = "my_feature"
        description = "Calculates my custom feature"
        
        async def calculate(self, symbol, exchange=None, **params):
            # Your calculation logic
            return result
"""

from .base import FeatureCalculator, FeatureResult
from .registry import FeatureRegistry, get_registry

__all__ = [
    'FeatureCalculator',
    'FeatureResult', 
    'FeatureRegistry',
    'get_registry',
]
