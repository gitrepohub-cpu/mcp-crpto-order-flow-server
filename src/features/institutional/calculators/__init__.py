"""
📊 Institutional Feature Calculators Package
============================================

Per-stream feature calculators for real-time streaming data.

Phase 1 Calculators:
- PricesFeatureCalculator: Microprice, spread dynamics, pressure
- OrderbookFeatureCalculator: Liquidity structure, absorption, dynamics
- TradesFeatureCalculator: CVD, delta, whale detection
- FundingFeatureCalculator: Rate analysis, skew, momentum
- OIFeatureCalculator: OI delta, leverage, liquidation risk

Phase 2 Calculators:
- LiquidationsFeatureCalculator: Cascade detection, clusters, exhaustion
- MarkPricesFeatureCalculator: Basis, index divergence, dislocation
- TickerFeatureCalculator: Volume analysis, range, volatility
"""

from .prices_calculator import PricesFeatureCalculator
from .orderbook_calculator import OrderbookFeatureCalculator
from .trades_calculator import TradesFeatureCalculator
from .funding_calculator import FundingFeatureCalculator, FundingArbitrageCalculator
from .oi_calculator import OIFeatureCalculator

# Phase 2 calculators
from .liquidations_calculator import LiquidationsFeatureCalculator
from .mark_prices_calculator import MarkPricesFeatureCalculator, BasisArbitrageCalculator
from .ticker_calculator import TickerFeatureCalculator

__all__ = [
    # Phase 1
    'PricesFeatureCalculator',
    'OrderbookFeatureCalculator',
    'TradesFeatureCalculator',
    'FundingFeatureCalculator',
    'FundingArbitrageCalculator',
    'OIFeatureCalculator',
    # Phase 2
    'LiquidationsFeatureCalculator',
    'MarkPricesFeatureCalculator',
    'BasisArbitrageCalculator',
    'TickerFeatureCalculator',
]

# Auto-register calculators with registry
def register_all_calculators():
    """Register all calculator classes with the institutional registry."""
    from ..base import institutional_feature_registry
    
    # Phase 1
    institutional_feature_registry.register_class('prices', PricesFeatureCalculator)
    institutional_feature_registry.register_class('orderbook', OrderbookFeatureCalculator)
    institutional_feature_registry.register_class('trades', TradesFeatureCalculator)
    institutional_feature_registry.register_class('funding', FundingFeatureCalculator)
    institutional_feature_registry.register_class('oi', OIFeatureCalculator)
    
    # Phase 2
    institutional_feature_registry.register_class('liquidations', LiquidationsFeatureCalculator)
    institutional_feature_registry.register_class('mark_prices', MarkPricesFeatureCalculator)
    institutional_feature_registry.register_class('ticker', TickerFeatureCalculator)

# Register on import
register_all_calculators()
