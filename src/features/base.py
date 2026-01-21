"""
Base Classes for Feature Calculators

Provides the foundation for creating pluggable feature calculation scripts.
All calculators use the TimeSeriesEngine for advanced time series analysis.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from xml.sax.saxutils import escape as xml_escape

logger = logging.getLogger(__name__)


@dataclass
class FeatureResult:
    """
    Standardized result container for feature calculations.
    
    Attributes:
        name: Feature calculator name
        symbol: Trading pair analyzed
        exchanges: List of exchanges included
        timestamp: When calculation was performed
        data: The calculated feature data
        metadata: Additional context (parameters used, data ranges, etc.)
        signals: Trading signals derived from the feature (optional)
        errors: Any errors encountered during calculation
    """
    name: str
    symbol: str
    exchanges: List[str]
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    signals: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'symbol': self.symbol,
            'exchanges': self.exchanges,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data,
            'metadata': self.metadata,
            'signals': self.signals,
            'errors': self.errors
        }
    
    def to_xml(self) -> str:
        """Convert to XML format for MCP tools."""
        xml_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<feature_result name="{xml_escape(self.name)}">',
            f'  <symbol>{xml_escape(self.symbol)}</symbol>',
            f'  <timestamp>{xml_escape(self.timestamp.isoformat())}</timestamp>',
            f'  <exchanges>{", ".join(self.exchanges)}</exchanges>',
        ]
        
        # Metadata
        if self.metadata:
            xml_parts.append('  <metadata>')
            for key, value in self.metadata.items():
                xml_parts.append(f'    <{key}>{xml_escape(str(value))}</{key}>')
            xml_parts.append('  </metadata>')
        
        # Data
        xml_parts.append('  <data>')
        xml_parts.append(self._dict_to_xml(self.data, indent=4))
        xml_parts.append('  </data>')
        
        # Signals
        if self.signals:
            xml_parts.append('  <signals>')
            for signal in self.signals:
                xml_parts.append('    <signal>')
                for key, value in signal.items():
                    xml_parts.append(f'      <{key}>{xml_escape(str(value))}</{key}>')
                xml_parts.append('    </signal>')
            xml_parts.append('  </signals>')
        
        # Errors
        if self.errors:
            xml_parts.append('  <errors>')
            for error in self.errors:
                xml_parts.append(f'    <error>{xml_escape(error)}</error>')
            xml_parts.append('  </errors>')
        
        xml_parts.append('</feature_result>')
        return '\n'.join(xml_parts)
    
    def _dict_to_xml(self, data: Any, indent: int = 0) -> str:
        """Recursively convert dict to XML."""
        prefix = ' ' * indent
        lines = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                safe_key = str(key).replace(' ', '_').replace('-', '_')
                if isinstance(value, (dict, list)):
                    lines.append(f'{prefix}<{safe_key}>')
                    lines.append(self._dict_to_xml(value, indent + 2))
                    lines.append(f'{prefix}</{safe_key}>')
                elif isinstance(value, float):
                    lines.append(f'{prefix}<{safe_key}>{value:.6f}</{safe_key}>')
                else:
                    lines.append(f'{prefix}<{safe_key}>{xml_escape(str(value))}</{safe_key}>')
        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    lines.append(f'{prefix}<item index="{i}">')
                    lines.append(self._dict_to_xml(item, indent + 2))
                    lines.append(f'{prefix}</item>')
                else:
                    lines.append(f'{prefix}<item>{xml_escape(str(item))}</item>')
        else:
            lines.append(f'{prefix}{xml_escape(str(data))}')
        
        return '\n'.join(lines)


class FeatureCalculator(ABC):
    """
    Abstract base class for all feature calculators.
    
    Subclass this to create new feature calculation scripts.
    The system will auto-discover classes that inherit from this.
    
    Required class attributes:
        name: Unique identifier for this calculator
        description: Human-readable description for MCP tool
        category: Category for grouping (e.g., 'order_flow', 'volatility')
    
    Required methods:
        calculate(): Perform the feature calculation
        get_parameters(): Return parameter schema for MCP tool
    """
    
    # Required class attributes (override in subclasses)
    name: str = "base_calculator"
    description: str = "Base feature calculator"
    category: str = "general"
    version: str = "1.0.0"
    
    # Default parameters
    default_params: Dict[str, Any] = {
        'hours': 24,
        'exchange': None,
        'market_type': 'futures'
    }
    
    def __init__(self):
        """Initialize the calculator with database connection and TimeSeries engine."""
        self._qm = None
        self._ts_engine = None
    
    @property
    def query_manager(self):
        """Lazy-load the DuckDB query manager."""
        if self._qm is None:
            from src.tools.duckdb_historical_tools import get_query_manager
            self._qm = get_query_manager()
        return self._qm
    
    @property
    def timeseries_engine(self):
        """
        Lazy-load the TimeSeriesEngine for advanced time series analysis.
        
        All calculators should use this for:
            - Forecasting (ARIMA, Exponential Smoothing, Theta)
            - Anomaly Detection (Z-score, IQR, Isolation Forest, CUSUM)
            - Change Point Detection (CUSUM, Binary Segmentation)
            - Feature Extraction (40+ statistical features)
            - Seasonality Analysis (FFT, decomposition)
            - Regime Detection (market regime classification)
        """
        if self._ts_engine is None:
            from src.analytics.timeseries_engine import get_timeseries_engine
            self._ts_engine = get_timeseries_engine()
        return self._ts_engine
    
    def create_timeseries_data(self, results: List[tuple], name: str = "value"):
        """
        Create TimeSeriesData from DuckDB query results.
        
        Args:
            results: Query results [(timestamp, value), ...]
            name: Name for the time series
        
        Returns:
            TimeSeriesData object for use with TimeSeriesEngine
        """
        from src.analytics.timeseries_engine import TimeSeriesData
        return TimeSeriesData.from_duckdb_result(results, name=name)
    
    @abstractmethod
    async def calculate(
        self,
        symbol: str,
        exchange: Optional[str] = None,
        **params
    ) -> FeatureResult:
        """
        Perform the feature calculation.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            exchange: Specific exchange or None for all
            **params: Additional parameters specific to this calculator
        
        Returns:
            FeatureResult with calculated data
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        """
        Return the parameter schema for this calculator.
        
        Used to generate MCP tool parameters dynamically.
        
        Returns:
            Dict mapping parameter names to their schemas:
            {
                'param_name': {
                    'type': 'str' | 'int' | 'float' | 'bool',
                    'default': <default_value>,
                    'description': 'Parameter description',
                    'required': True | False
                }
            }
        """
        pass
    
    def get_mcp_description(self) -> str:
        """Generate the full MCP tool description."""
        params = self.get_parameters()
        param_docs = []
        for name, schema in params.items():
            param_docs.append(f"        {name}: {schema.get('description', 'No description')}")
        
        return f"""{self.description}

    Category: {self.category}
    Version: {self.version}
    
    Args:
{chr(10).join(param_docs)}
    
    Returns:
        XML with calculated feature data, signals, and metadata.
"""
    
    # Utility methods for subclasses
    
    def get_table_name(self, symbol: str, exchange: str, market_type: str, stream: str) -> str:
        """Generate DuckDB table name."""
        return f"{symbol.lower()}_{exchange.lower()}_{market_type}_{stream}"
    
    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in DuckDB."""
        return self.query_manager.table_exists(table_name)
    
    def execute_query(self, query: str) -> List[tuple]:
        """Execute a query on DuckDB."""
        return self.query_manager.execute_query(query)
    
    def get_exchanges(self, market_type: str = 'futures') -> List[str]:
        """Get list of exchanges for a market type."""
        from src.tools.duckdb_historical_tools import EXCHANGES
        return EXCHANGES.get(market_type, EXCHANGES['futures'])
    
    def create_result(
        self,
        symbol: str,
        exchanges: List[str],
        data: Dict[str, Any],
        metadata: Optional[Dict] = None,
        signals: Optional[List[Dict]] = None,
        errors: Optional[List[str]] = None
    ) -> FeatureResult:
        """Helper to create a FeatureResult."""
        return FeatureResult(
            name=self.name,
            symbol=symbol,
            exchanges=exchanges,
            timestamp=datetime.utcnow(),
            data=data,
            metadata=metadata or {},
            signals=signals or [],
            errors=errors or []
        )
