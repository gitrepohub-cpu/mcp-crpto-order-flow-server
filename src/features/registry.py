"""
Feature Calculator Registry and Auto-Discovery System

Automatically discovers and registers feature calculators from the calculators/ directory.
Provides methods to generate MCP tool wrappers dynamically.
"""

import importlib
import inspect
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Type, Callable, Any

from .base import FeatureCalculator, FeatureResult

logger = logging.getLogger(__name__)


class FeatureRegistry:
    """
    Registry for feature calculators with auto-discovery.
    
    Features:
        - Auto-discovers calculators in calculators/ directory
        - Validates calculator implementations
        - Generates MCP tool wrappers dynamically
        - Supports hot-reload of new calculators
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._calculators: Dict[str, FeatureCalculator] = {}
        self._calculator_classes: Dict[str, Type[FeatureCalculator]] = {}
        self._initialized = True
        
        # Auto-discover on initialization
        self.discover_calculators()
    
    def discover_calculators(self, force_reload: bool = False) -> int:
        """
        Discover and register all calculators in the calculators/ directory.
        
        Args:
            force_reload: If True, reload previously discovered calculators
        
        Returns:
            Number of calculators discovered
        """
        if not force_reload and self._calculators:
            return len(self._calculators)
        
        calculators_dir = Path(__file__).parent / "calculators"
        
        if not calculators_dir.exists():
            logger.warning(f"Calculators directory not found: {calculators_dir}")
            calculators_dir.mkdir(exist_ok=True)
            return 0
        
        discovered = 0
        
        for file_path in calculators_dir.glob("*.py"):
            if file_path.name.startswith("_"):
                continue
            
            module_name = file_path.stem
            full_module_name = f"src.features.calculators.{module_name}"
            
            try:
                # Import or reload the module
                if full_module_name in sys.modules and force_reload:
                    module = importlib.reload(sys.modules[full_module_name])
                else:
                    module = importlib.import_module(full_module_name)
                
                # Find all FeatureCalculator subclasses
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (issubclass(obj, FeatureCalculator) and 
                        obj is not FeatureCalculator and
                        hasattr(obj, 'name') and obj.name != 'base_calculator'):
                        
                        try:
                            self.register(obj)
                            discovered += 1
                            logger.info(f"Discovered calculator: {obj.name} from {module_name}")
                        except Exception as e:
                            logger.error(f"Failed to register {name}: {e}")
                
            except Exception as e:
                logger.error(f"Failed to import {module_name}: {e}")
        
        logger.info(f"Discovered {discovered} feature calculators")
        return discovered
    
    @property
    def calculators(self) -> Dict[str, FeatureCalculator]:
        """Access registered calculators."""
        return self._calculators
    
    def register(self, calculator_class: Type[FeatureCalculator]) -> None:
        """
        Register a calculator class.
        
        Args:
            calculator_class: The calculator class to register
        
        Raises:
            ValueError: If calculator is invalid or already registered
        """
        # Validate
        if not issubclass(calculator_class, FeatureCalculator):
            raise ValueError(f"{calculator_class} must be a FeatureCalculator subclass")
        
        if not hasattr(calculator_class, 'name') or calculator_class.name == 'base_calculator':
            raise ValueError(f"{calculator_class} must have a unique 'name' attribute")
        
        name = calculator_class.name
        
        if name in self._calculators and not isinstance(self._calculators[name], calculator_class):
            logger.warning(f"Overwriting existing calculator: {name}")
        
        # Instantiate and store
        instance = calculator_class()
        self._calculators[name] = instance
        self._calculator_classes[name] = calculator_class
    
    def get(self, name: str) -> Optional[FeatureCalculator]:
        """Get a calculator by name."""
        return self._calculators.get(name)
    
    def list_calculators(self) -> List[Dict[str, Any]]:
        """List all registered calculators with their metadata."""
        return [
            {
                'name': calc.name,
                'description': calc.description,
                'category': calc.category,
                'version': calc.version,
                'parameters': calc.get_parameters()
            }
            for calc in self._calculators.values()
        ]
    
    def get_by_category(self, category: str) -> List[FeatureCalculator]:
        """Get all calculators in a category."""
        return [
            calc for calc in self._calculators.values()
            if calc.category == category
        ]
    
    def get_categories(self) -> List[str]:
        """Get list of all categories."""
        return list(set(calc.category for calc in self._calculators.values()))
    
    async def calculate(
        self,
        calculator_name: str,
        symbol: str,
        exchange: Optional[str] = None,
        hours: int = 24,
        extra_params: Optional[Dict[str, Any]] = None
    ) -> FeatureResult:
        """
        Run a calculation using a registered calculator.
        
        Args:
            calculator_name: Name of the calculator to use
            symbol: Trading pair
            exchange: Specific exchange or None for all
            hours: Hours of historical data
            extra_params: Calculator-specific parameters
        
        Returns:
            FeatureResult from the calculator
        
        Raises:
            KeyError: If calculator not found
        """
        calculator = self._calculators.get(calculator_name)
        if not calculator:
            raise KeyError(f"Calculator not found: {calculator_name}")
        
        params = extra_params or {}
        params['hours'] = hours
        
        return await calculator.calculate(symbol=symbol, exchange=exchange, **params)
    
    def register_all_with_mcp(self, mcp) -> int:
        """
        Register all calculators as MCP tools.
        
        Note: Since FastMCP doesn't support **kwargs, we register a generic
        dispatcher tool that can call any calculator.
        
        Args:
            mcp: The FastMCP instance
        
        Returns:
            Number of tools registered
        """
        registered = 0
        
        for name, calculator in self._calculators.items():
            try:
                # Create a closure to capture calculator reference
                self._register_calculator_tool(mcp, name, calculator)
                registered += 1
                logger.info(f"Registered MCP tool: calculate_{name}")
            except Exception as e:
                logger.error(f"Failed to register MCP tool for {name}: {e}")
        
        return registered
    
    def _register_calculator_tool(self, mcp, name: str, calculator: FeatureCalculator):
        """Register a single calculator as an MCP tool."""
        # Build description with parameters
        params = calculator.get_parameters()
        param_list = []
        for pname, pschema in params.items():
            ptype = pschema.get('type', 'str')
            pdefault = pschema.get('default', 'None')
            pdesc = pschema.get('description', '')
            param_list.append(f"    {pname} ({ptype}, default={pdefault}): {pdesc}")
        
        description = f"""{calculator.description}

Category: {calculator.category}
Version: {calculator.version}

Parameters:
{chr(10).join(param_list)}

Returns: XML with calculated feature data, signals, and metadata."""

        # Create tool function with explicit parameters (no **kwargs)
        # Use exec to dynamically create function with proper signature
        func_code = f'''
async def calculate_{name}(
    symbol: str = "BTCUSDT",
    exchange: str = None,
    hours: int = 24
) -> str:
    """{description}"""
    try:
        result = await _calculator.calculate(
            symbol=symbol,
            exchange=exchange,
            hours=hours
        )
        return result.to_xml()
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f"Error in {name}: {{e}}", exc_info=True)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<error type="CALCULATION_FAILED">
  <calculator>{name}</calculator>
  <message>{{str(e)}}</message>
</error>"""
'''
        
        # Execute the function definition with calculator in scope
        local_vars = {'_calculator': calculator}
        exec(func_code, globals(), local_vars)
        tool_func = local_vars[f'calculate_{name}']
        
        # Register with MCP
        mcp.tool()(tool_func)


# Global registry instance
_registry: Optional[FeatureRegistry] = None


def get_registry() -> FeatureRegistry:
    """Get the global feature registry instance."""
    global _registry
    if _registry is None:
        _registry = FeatureRegistry()
    return _registry
