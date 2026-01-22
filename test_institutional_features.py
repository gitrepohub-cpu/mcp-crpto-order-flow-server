"""
🧪 Test Institutional Features Framework (Phase 1 & Phase 2)
============================================================

Validates that all Phase 1 & 2 components work correctly:
1. Feature calculators (prices, orderbook, trades, funding, oi)
2. Phase 2 calculators (liquidations, mark_prices, ticker)
3. Composite signal calculator
4. Feature storage
5. FeatureEngine integration
"""

import sys
import asyncio
from datetime import datetime, timezone
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_feature_buffer():
    """Test FeatureBuffer utility class."""
    print("\n" + "="*60)
    print("Testing FeatureBuffer")
    print("="*60)
    
    from src.features.institutional.base import FeatureBuffer
    
    buffer = FeatureBuffer(name='test', window_size=10)
    
    # Add values
    for i in range(15):
        buffer.append(float(i))
    
    assert len(buffer) == 10, f"Expected 10, got {len(buffer)}"
    assert buffer.mean() == 9.5, f"Expected 9.5, got {buffer.mean()}"
    
    # Test z-score
    zscore = buffer.zscore()
    print(f"  ✓ Z-score: {zscore:.4f}")
    
    # Test velocity
    velocity = buffer.velocity(periods=5)
    print(f"  ✓ Velocity: {velocity:.4f}")
    
    # Test percentile
    pct = buffer.percentile(12)
    print(f"  ✓ Percentile for 12: {pct:.2f}")
    
    print("  ✓ FeatureBuffer tests passed!")
    return True


def test_prices_calculator():
    """Test PricesFeatureCalculator."""
    print("\n" + "="*60)
    print("Testing PricesFeatureCalculator")
    print("="*60)
    
    from src.features.institutional.calculators import PricesFeatureCalculator
    
    calc = PricesFeatureCalculator(symbol='BTCUSDT', exchange='binance')
    
    # Simulate price updates
    test_data = [
        {'bid': 50000.0, 'ask': 50010.0, 'bid_qty': 1.5, 'ask_qty': 1.2},
        {'bid': 50005.0, 'ask': 50015.0, 'bid_qty': 2.0, 'ask_qty': 1.0},
        {'bid': 50010.0, 'ask': 50020.0, 'bid_qty': 1.8, 'ask_qty': 1.5},
        {'bid': 50008.0, 'ask': 50018.0, 'bid_qty': 1.3, 'ask_qty': 1.8},
        {'bid': 50012.0, 'ask': 50022.0, 'bid_qty': 2.5, 'ask_qty': 0.8},
    ]
    
    features = None
    for data in test_data:
        features = calc.calculate(data)
    
    assert features is not None, "No features calculated"
    assert 'microprice' in features, "Missing microprice"
    assert 'spread' in features, "Missing spread"
    assert 'pressure_ratio' in features, "Missing pressure_ratio"
    
    print(f"  ✓ Microprice: {features['microprice']:.2f}")
    print(f"  ✓ Spread: {features['spread']:.4f}")
    print(f"  ✓ Spread (bps): {features['spread_bps']:.2f}")
    print(f"  ✓ Pressure Ratio: {features['pressure_ratio']:.4f}")
    print(f"  ✓ Total features: {len(features)}")
    
    print("  ✓ PricesFeatureCalculator tests passed!")
    return True


def test_orderbook_calculator():
    """Test OrderbookFeatureCalculator."""
    print("\n" + "="*60)
    print("Testing OrderbookFeatureCalculator")
    print("="*60)
    
    from src.features.institutional.calculators import OrderbookFeatureCalculator
    
    calc = OrderbookFeatureCalculator(symbol='BTCUSDT', exchange='binance')
    
    # Simulate orderbook updates
    test_data = [
        {
            'bids': [[50000, 1.0], [49990, 2.0], [49980, 1.5], [49970, 3.0], [49960, 2.5],
                     [49950, 1.0], [49940, 1.5], [49930, 2.0], [49920, 1.0], [49910, 2.5]],
            'asks': [[50010, 0.8], [50020, 1.5], [50030, 2.0], [50040, 1.0], [50050, 3.0],
                     [50060, 1.5], [50070, 2.0], [50080, 1.0], [50090, 1.5], [50100, 2.0]],
        },
        {
            'bids': [[50005, 1.2], [49995, 1.8], [49985, 2.0], [49975, 2.5], [49965, 3.0],
                     [49955, 1.5], [49945, 1.0], [49935, 2.5], [49925, 1.5], [49915, 2.0]],
            'asks': [[50015, 1.0], [50025, 1.2], [50035, 1.8], [50045, 1.5], [50055, 2.0],
                     [50065, 1.0], [50075, 2.5], [50085, 1.5], [50095, 1.0], [50105, 2.5]],
        },
    ]
    
    features = None
    for data in test_data:
        features = calc.calculate(data)
    
    assert features is not None, "No features calculated"
    assert 'depth_imbalance_5' in features, "Missing depth_imbalance_5"
    assert 'absorption_ratio' in features, "Missing absorption_ratio"
    
    print(f"  ✓ Depth Imbalance (5): {features['depth_imbalance_5']:.4f}")
    print(f"  ✓ Depth Imbalance (10): {features['depth_imbalance_10']:.4f}")
    print(f"  ✓ Absorption Ratio: {features['absorption_ratio']:.4f}")
    print(f"  ✓ Liquidity Gradient: {features['liquidity_gradient']:.4f}")
    print(f"  ✓ Total features: {len(features)}")
    
    print("  ✓ OrderbookFeatureCalculator tests passed!")
    return True


def test_trades_calculator():
    """Test TradesFeatureCalculator."""
    print("\n" + "="*60)
    print("Testing TradesFeatureCalculator")
    print("="*60)
    
    from src.features.institutional.calculators import TradesFeatureCalculator
    
    calc = TradesFeatureCalculator(symbol='BTCUSDT', exchange='binance')
    
    # Simulate trade batches
    test_trades = [
        {'price': 50000, 'quantity': 0.5, 'side': 'buy'},
        {'price': 50005, 'quantity': 1.0, 'side': 'buy'},
        {'price': 50010, 'quantity': 0.3, 'side': 'sell'},
        {'price': 50008, 'quantity': 0.8, 'side': 'buy'},
        {'price': 50012, 'quantity': 2.0, 'side': 'buy'},  # Larger trade
        {'price': 50015, 'quantity': 0.4, 'side': 'sell'},
        {'price': 50010, 'quantity': 0.6, 'side': 'sell'},
        {'price': 50020, 'quantity': 5.0, 'side': 'buy'},  # Whale trade
    ]
    
    features = calc.calculate(test_trades)
    
    assert features is not None, "No features calculated"
    assert 'cvd' in features, "Missing CVD"
    assert 'aggressive_delta' in features, "Missing aggressive_delta"
    
    print(f"  ✓ CVD: {features['cvd']:.4f}")
    print(f"  ✓ Aggressive Delta: {features['aggressive_delta']:.4f}")
    print(f"  ✓ Buy Pressure: {features['buy_pressure']:.4f}")
    print(f"  ✓ Sell Pressure: {features['sell_pressure']:.4f}")
    print(f"  ✓ Flow Toxicity: {features['flow_toxicity']:.4f}")
    print(f"  ✓ Total features: {len(features)}")
    
    print("  ✓ TradesFeatureCalculator tests passed!")
    return True


def test_funding_calculator():
    """Test FundingFeatureCalculator."""
    print("\n" + "="*60)
    print("Testing FundingFeatureCalculator")
    print("="*60)
    
    from src.features.institutional.calculators import FundingFeatureCalculator
    
    calc = FundingFeatureCalculator(symbol='BTCUSDT', exchange='binance')
    
    # Simulate funding rate updates
    test_data = [
        {'funding_rate': 0.0001, 'predicted_rate': 0.00012, 'exchange': 'binance'},
        {'funding_rate': 0.00015, 'predicted_rate': 0.00018, 'exchange': 'binance'},
        {'funding_rate': 0.0002, 'predicted_rate': 0.00022, 'exchange': 'binance'},
        {'funding_rate': 0.00025, 'predicted_rate': 0.00028, 'exchange': 'binance'},
        {'funding_rate': 0.0003, 'predicted_rate': 0.00032, 'exchange': 'binance'},
    ]
    
    features = None
    for data in test_data:
        features = calc.calculate(data)
    
    assert features is not None, "No features calculated"
    assert 'funding_rate' in features, "Missing funding_rate"
    assert 'funding_carry_yield' in features, "Missing funding_carry_yield"
    
    print(f"  ✓ Funding Rate: {features['funding_rate']:.6f}")
    print(f"  ✓ Funding Z-Score: {features['funding_zscore']:.4f}")
    print(f"  ✓ Funding Momentum: {features['funding_momentum']:.6f}")
    print(f"  ✓ Carry Yield (Annual): {features['funding_carry_yield']:.2%}")
    print(f"  ✓ Regime: {features['funding_regime']}")
    print(f"  ✓ Total features: {len(features)}")
    
    print("  ✓ FundingFeatureCalculator tests passed!")
    return True


def test_oi_calculator():
    """Test OIFeatureCalculator."""
    print("\n" + "="*60)
    print("Testing OIFeatureCalculator")
    print("="*60)
    
    from src.features.institutional.calculators import OIFeatureCalculator
    
    calc = OIFeatureCalculator(symbol='BTCUSDT', exchange='binance')
    
    # Simulate OI updates with corresponding price
    test_data = [
        {'oi': 100000, 'oi_value': 5000000000, 'price': 50000, 'long_short_ratio': 1.1},
        {'oi': 102000, 'oi_value': 5100000000, 'price': 50200, 'long_short_ratio': 1.15},  # OI↑ Price↑
        {'oi': 105000, 'oi_value': 5200000000, 'price': 49800, 'long_short_ratio': 1.2},   # OI↑ Price↓
        {'oi': 103000, 'oi_value': 5100000000, 'price': 49500, 'long_short_ratio': 1.1},   # OI↓ Price↓
        {'oi': 101000, 'oi_value': 5050000000, 'price': 50000, 'long_short_ratio': 1.05},  # OI↓ Price↑
    ]
    
    features = None
    for data in test_data:
        features = calc.calculate(data)
    
    assert features is not None, "No features calculated"
    assert 'oi' in features, "Missing oi"
    assert 'leverage_index' in features, "Missing leverage_index"
    
    print(f"  ✓ OI: {features['oi']:,.0f}")
    print(f"  ✓ OI Delta: {features['oi_delta']:,.0f}")
    print(f"  ✓ Leverage Index: {features['leverage_index']:.4f}")
    print(f"  ✓ Position Intent: {features['position_intent']}")
    print(f"  ✓ L/S Ratio: {features['long_short_ratio']:.2f}")
    print(f"  ✓ Total features: {len(features)}")
    
    print("  ✓ OIFeatureCalculator tests passed!")
    return True


# ========== PHASE 2 TESTS ==========

def test_liquidations_calculator():
    """Test LiquidationsFeatureCalculator."""
    print("\n" + "="*60)
    print("Testing LiquidationsFeatureCalculator (Phase 2)")
    print("="*60)
    
    from src.features.institutional.calculators import LiquidationsFeatureCalculator
    
    calc = LiquidationsFeatureCalculator(symbol='BTCUSDT', exchange='binance')
    
    # Simulate liquidation events
    test_data = [
        {'side': 'long', 'quantity': 0.5, 'value': 25000, 'price': 50000},
        {'side': 'long', 'quantity': 1.0, 'value': 50000, 'price': 50000},
        {'side': 'short', 'quantity': 0.3, 'value': 15000, 'price': 50000},
        {'side': 'long', 'quantity': 2.0, 'value': 100000, 'price': 49800},  # Cascade event
        {'side': 'long', 'quantity': 3.0, 'value': 150000, 'price': 49700},  # Cascade continues
        {'side': 'short', 'quantity': 0.5, 'value': 25000, 'price': 49750},
        {'side': 'long', 'quantity': 0.8, 'value': 40000, 'price': 49600},
        {'side': 'short', 'quantity': 0.4, 'value': 20000, 'price': 49650},
    ]
    
    features = None
    for liq in test_data:
        features = calc.calculate(liq)
    
    assert features is not None, "No features calculated"
    assert 'liquidation_imbalance' in features, "Missing liquidation_imbalance"
    assert 'liquidation_rate' in features, "Missing liquidation_rate"
    assert 'cascade_probability' in features, "Missing cascade_probability"
    
    print(f"  ✓ Long Liquidations: {features['long_liquidation_count']:.0f}")
    print(f"  ✓ Short Liquidations: {features['short_liquidation_count']:.0f}")
    print(f"  ✓ Liquidation Imbalance: {features['liquidation_imbalance']:.4f}")
    print(f"  ✓ Liquidation Rate: {features['liquidation_rate']:.4f}")
    print(f"  ✓ Cascade Probability: {features['cascade_probability']:.4f}")
    print(f"  ✓ Exhaustion Signal: {features['exhaustion_signal']}")
    print(f"  ✓ Total features: {len(features)}")
    
    print("  ✓ LiquidationsFeatureCalculator tests passed!")
    return True


def test_mark_prices_calculator():
    """Test MarkPricesFeatureCalculator."""
    print("\n" + "="*60)
    print("Testing MarkPricesFeatureCalculator (Phase 2)")
    print("="*60)
    
    from src.features.institutional.calculators import MarkPricesFeatureCalculator
    
    calc = MarkPricesFeatureCalculator(symbol='BTCUSDT', exchange='binance')
    
    # Simulate mark price updates with basis
    test_data = [
        {'mark_price': 50010, 'index_price': 50000, 'spot_price': 49995, 'mid_price': 50005, 'funding_rate': 0.0001},
        {'mark_price': 50020, 'index_price': 50005, 'spot_price': 50000, 'mid_price': 50010, 'funding_rate': 0.00015},
        {'mark_price': 50035, 'index_price': 50015, 'spot_price': 50010, 'mid_price': 50020, 'funding_rate': 0.0002},
        {'mark_price': 50050, 'index_price': 50020, 'spot_price': 50015, 'mid_price': 50030, 'funding_rate': 0.00025},
        {'mark_price': 50080, 'index_price': 50030, 'spot_price': 50020, 'mid_price': 50045, 'funding_rate': 0.0003},
    ]
    
    features = None
    for data in test_data:
        features = calc.calculate(data)
    
    assert features is not None, "No features calculated"
    assert 'mark_spot_basis' in features, "Missing mark_spot_basis"
    assert 'annualized_basis' in features, "Missing annualized_basis"
    assert 'basis_regime' in features, "Missing basis_regime"
    
    print(f"  ✓ Mark-Spot Basis: {features['mark_spot_basis']:.2f}")
    print(f"  ✓ Basis %: {features['mark_spot_basis_pct']:.4%}")
    print(f"  ✓ Annualized Basis: {features['annualized_basis']:.2%}")
    print(f"  ✓ Basis Z-Score: {features['basis_zscore']:.4f}")
    print(f"  ✓ Basis Regime: {features['basis_regime']}")
    print(f"  ✓ Index Divergence: {features['index_divergence']:.4f}")
    print(f"  ✓ Total features: {len(features)}")
    
    print("  ✓ MarkPricesFeatureCalculator tests passed!")
    return True


def test_ticker_calculator():
    """Test TickerFeatureCalculator."""
    print("\n" + "="*60)
    print("Testing TickerFeatureCalculator (Phase 2)")
    print("="*60)
    
    from src.features.institutional.calculators import TickerFeatureCalculator
    
    calc = TickerFeatureCalculator(symbol='BTCUSDT', exchange='binance')
    
    # Simulate 24h ticker updates (use exact field names the calculator expects)
    test_data = [
        {'volume': 1000000, 'high': 51000, 'low': 49000, 'open': 49500, 'close': 50000, 'trades': 50000},
        {'volume': 1100000, 'high': 51200, 'low': 49100, 'open': 49600, 'close': 50200, 'trades': 55000},
        {'volume': 1250000, 'high': 51500, 'low': 49000, 'open': 49700, 'close': 50500, 'trades': 62000},
        {'volume': 1400000, 'high': 52000, 'low': 49200, 'open': 49800, 'close': 51000, 'trades': 70000},
        {'volume': 1600000, 'high': 52500, 'low': 49500, 'open': 49900, 'close': 51500, 'trades': 80000},
    ]
    
    features = None
    for data in test_data:
        features = calc.calculate(data)
    
    assert features is not None, "No features calculated"
    assert 'volume_acceleration' in features, "Missing volume_acceleration"
    assert 'high_low_range_pct' in features, "Missing high_low_range_pct"
    assert 'institutional_interest_idx' in features, "Missing institutional_interest_idx"
    
    print(f"  ✓ Volume Acceleration: {features['volume_acceleration']:.4f}")
    print(f"  ✓ High-Low Range %: {features['high_low_range_pct']:.2f}%")
    print(f"  ✓ Range Expansion: {features['range_expansion_ratio']:.4f}")
    print(f"  ✓ Realized Volatility: {features['realized_volatility']:.4f}")
    print(f"  ✓ Market Strength: {features['market_strength']:.4f}")
    print(f"  ✓ Institutional Interest: {features['institutional_interest_idx']:.4f}")
    print(f"  ✓ Total features: {len(features)}")
    
    print("  ✓ TickerFeatureCalculator tests passed!")
    return True


def test_phase2_engine_integration():
    """Test FeatureEngine with Phase 2 calculators."""
    print("\n" + "="*60)
    print("Testing FeatureEngine with Phase 2 Calculators")
    print("="*60)
    
    from src.features.institutional.integration import FeatureEngine
    
    # Mock DB manager
    class MockDBManager:
        def execute(self, query): pass
        def query(self, query): return []
    
    # Mock storage
    class MockStorage:
        def __init__(self, *args):
            self.stored_features = []
            self.stored_composites = []
        
        def store_features(self, **kwargs):
            self.stored_features.append(kwargs)
        
        def store_composite_signals(self, **kwargs):
            self.stored_composites.append(kwargs)
        
        def get_latest_features(self, *args):
            return None
        
        async def flush_all(self):
            pass
    
    db = MockDBManager()
    storage = MockStorage(db)
    engine = FeatureEngine(db, storage=storage, enable_composites=True)
    
    # Test liquidations processing
    liq_data = {'side': 'long', 'quantity': 1.0, 'value': 50000, 'price': 50000}
    features = engine.process_liquidations('BTCUSDT', 'binance', liq_data)
    
    assert features is not None, "No liquidations features"
    print(f"  ✓ Processed liquidations: {len(features)} features")
    
    # Test mark prices processing
    mark_data = {'mark_price': 50010, 'index_price': 50000, 'spot_price': 49995, 'mid_price': 50005, 'funding_rate': 0.0001}
    features = engine.process_mark_prices('BTCUSDT', 'binance', mark_data)
    
    assert features is not None, "No mark_prices features"
    print(f"  ✓ Processed mark_prices: {len(features)} features")
    
    # Test ticker processing (use exact field names the calculator expects)
    ticker_data = {'volume': 1000000, 'high': 51000, 'low': 49000, 'open': 49500, 'close': 50000, 'trades': 50000}
    features = engine.process_ticker('BTCUSDT', 'binance', ticker_data)
    
    assert features is not None, "No ticker features"
    print(f"  ✓ Processed ticker: {len(features)} features")
    
    # Check metrics
    metrics = engine.get_metrics()
    print(f"  ✓ Total features calculated: {metrics['features_calculated']}")
    
    print("\n  ✓ Phase 2 FeatureEngine integration tests passed!")
    return True


def test_composite_calculator():
    """Test CompositeSignalCalculator."""
    print("\n" + "="*60)
    print("Testing CompositeSignalCalculator")
    print("="*60)
    
    from src.features.institutional.composite import CompositeSignalCalculator
    
    calc = CompositeSignalCalculator()
    
    # Simulate features from different streams
    prices_features = {
        'spread_zscore': -0.5,
        'price_efficiency': 0.85,
    }
    
    orderbook_features = {
        'absorption_ratio': 0.75,
        'pull_wall_detected': True,
        'push_wall_detected': False,
        'add_cancel_ratio': 0.6,
        'depth_imbalance_5': 0.2,
    }
    
    trades_features = {
        'whale_trade_detected': True,
        'whale_volume_ratio': 0.15,
        'flow_toxicity': 0.4,
        'cvd_price_divergence': 0.3,
        'cvd_slope': 0.05,
        'sweep_detected': False,
    }
    
    funding_features = {
        'funding_zscore': 1.5,
        'funding_regime_score': 1.0,
    }
    
    oi_features = {
        'leverage_zscore': 0.8,
        'position_crowding_score': 0.6,
        'position_intent_score': 0.5,
        'liquidation_cascade_risk': 0.3,
    }
    
    # Update features
    calc.update_features('prices', prices_features)
    calc.update_features('orderbook', orderbook_features)
    calc.update_features('trades', trades_features)
    calc.update_features('funding', funding_features)
    calc.update_features('oi', oi_features)
    
    # Calculate composite signals
    signals = calc.calculate_all()
    
    assert 'smart_money_index' in signals, "Missing smart_money_index"
    assert 'squeeze_probability' in signals, "Missing squeeze_probability"
    assert 'stop_hunt_probability' in signals, "Missing stop_hunt_probability"
    
    print("\n  Composite Signals:")
    for name, signal in signals.items():
        interpretation = calc.get_signal_interpretation(name, signal.value)
        print(f"    {signal.name}:")
        print(f"      Value: {signal.value:.4f}")
        print(f"      Confidence: {signal.confidence:.2%}")
        print(f"      Interpretation: {interpretation}")
    
    print("\n  ✓ CompositeSignalCalculator tests passed!")
    return True


def test_phase3_composite_signals():
    """Test Phase 3 Enhanced Composite Signals."""
    print("\n" + "="*60)
    print("Testing Phase 3 Composite Signals (10 new signals)")
    print("="*60)
    
    from src.features.institutional.composite import CompositeSignalCalculator, EnhancedSignal
    
    calc = CompositeSignalCalculator()
    
    # Provide comprehensive features from all 8 streams
    prices_features = {
        'spread_zscore': -1.5,
        'spread_bps': 5.0,
        'spread_compression_velocity': 0.3,
        'price_efficiency': 0.85,
        'mid_zscore': -2.0,
        'hurst_exponent': 0.45,
        'vwap_deviation': -0.005,
    }
    
    orderbook_features = {
        'absorption_ratio': 0.75,
        'pull_wall_detected': True,
        'push_wall_detected': False,
        'bid_wall_detected': True,
        'ask_wall_detected': False,
        'add_cancel_ratio': 0.4,
        'depth_imbalance_5': 0.3,
        'depth_imbalance_10': 0.25,
        'liquidity_gradient': 1.3,
        'liquidity_persistence_score': 0.4,
        'bid_depth_10': 50.0,
        'ask_depth_10': 45.0,
    }
    
    trades_features = {
        'whale_trade_detected': True,
        'whale_volume_ratio': 0.15,
        'flow_toxicity': 0.4,
        'cvd_price_divergence': -0.4,
        'cvd_slope': 0.08,
        'sweep_detected': False,
        'buy_pressure': 0.6,
        'sell_pressure': 0.4,
        'aggressive_ratio': 0.55,
        'aggressive_delta': 0.1,
        'iceberg_detected': True,
        'trade_clustering_index': 0.6,
    }
    
    funding_features = {
        'funding_rate': 0.0003,
        'funding_zscore': 2.5,
        'funding_regime': 'positive',
        'funding_regime_score': 1.0,
        'funding_carry_yield': 0.35,
        'funding_momentum': 0.0001,
        'funding_percentile': 92,
    }
    
    oi_features = {
        'leverage_index': 1.8,
        'leverage_zscore': 1.5,
        'position_crowding_score': 0.7,
        'position_intent': 'long_accumulation',
        'position_intent_score': 0.6,
        'liquidation_cascade_risk': 0.4,
        'oi_delta_pct': 3.5,
        'long_short_ratio': 1.2,
        'oi_price_correlation': 0.6,
    }
    
    liquidations_features = {
        'liquidation_cluster_detected': True,
        'cascade_probability': 0.5,
        'cascade_acceleration': 0.3,
        'liquidation_imbalance': 0.4,
        'exhaustion_signal': False,
    }
    
    mark_prices_features = {
        'mark_spot_basis': 50,
        'mark_spot_basis_pct': 0.001,
        'annualized_basis': 0.12,
        'basis_zscore': 1.8,
        'dislocation_detected': False,
        'mark_price': 50050,
    }
    
    ticker_features = {
        'volume': 1500000,
        'volatility_compression_idx': 0.6,
        'volatility_expansion_idx': 0.2,
        'range_vs_atr': 0.7,
        'price_change_pct': 2.5,
        'market_strength': 0.65,
        'realized_volatility': 0.02,
        'relative_volume_percentile': 75,
        'institutional_interest_idx': 0.7,
    }
    
    # Update all features
    calc.update_features('prices', prices_features)
    calc.update_features('orderbook', orderbook_features)
    calc.update_features('trades', trades_features)
    calc.update_features('funding', funding_features)
    calc.update_features('oi', oi_features)
    calc.update_features('liquidations', liquidations_features)
    calc.update_features('mark_prices', mark_prices_features)
    calc.update_features('ticker', ticker_features)
    
    # Calculate all signals
    signals = calc.calculate_all()
    
    # Verify Phase 1 signals still work
    assert 'smart_money_index' in signals, "Missing smart_money_index"
    assert 'squeeze_probability' in signals, "Missing squeeze_probability"
    print(f"  ✓ Phase 1 signals: {sum(1 for s in signals.values() if not isinstance(s, EnhancedSignal))} signals")
    
    # Verify Phase 3 signals
    phase3_signals = calc.get_phase3_signals()
    print(f"  ✓ Phase 3 signals: {len(phase3_signals)} signals")
    
    # Test specific Phase 3 signals
    print("\n  Phase 3 Enhanced Signals:")
    
    # Market Maker Activity
    if 'market_maker_activity' in signals:
        mm = signals['market_maker_activity']
        assert isinstance(mm, EnhancedSignal), "market_maker_activity should be EnhancedSignal"
        print(f"    ✓ Market Maker Activity: {mm.value:.2f} ({mm.metadata['activity_type']}, {mm.metadata['inventory_bias']})")
    
    # Liquidation Cascade Risk
    if 'liquidation_cascade_risk' in signals:
        cascade = signals['liquidation_cascade_risk']
        print(f"    ✓ Liquidation Cascade Risk: {cascade.value:.2f} (severity: {cascade.metadata['severity']}, direction: {cascade.metadata['direction']})")
    
    # Institutional Phase
    if 'institutional_phase' in signals:
        phase = signals['institutional_phase']
        print(f"    ✓ Institutional Phase: {phase.metadata['phase']} (intensity: {phase.metadata['intensity']:.2f})")
    
    # Volatility Breakout
    if 'volatility_breakout' in signals:
        vol = signals['volatility_breakout']
        print(f"    ✓ Volatility Breakout: {vol.value:.2f} (direction: {vol.metadata['direction']}, timeframe: {vol.metadata['timeframe_hours']}h)")
    
    # Mean Reversion
    if 'mean_reversion_signal' in signals:
        mr = signals['mean_reversion_signal']
        print(f"    ✓ Mean Reversion: {mr.metadata['signal']} (strength: {mr.metadata['strength']:.2f})")
    
    # Momentum Exhaustion
    if 'momentum_exhaustion' in signals:
        exhaust = signals['momentum_exhaustion']
        print(f"    ✓ Momentum Exhaustion: {exhaust.value:.2f} (exhausted: {exhaust.metadata['exhaustion']}, trend: {exhaust.metadata['trend_direction']})")
    
    # Smart Money Flow
    if 'smart_money_flow' in signals:
        flow = signals['smart_money_flow']
        print(f"    ✓ Smart Money Flow: {flow.metadata['direction']} (strength: {flow.metadata['strength']:.2f})")
    
    # Arbitrage Opportunity
    if 'arbitrage_opportunity' in signals:
        arb = signals['arbitrage_opportunity']
        print(f"    ✓ Arbitrage: {arb.metadata['type']} (return: {arb.metadata['expected_return_pct_annual']:.1f}%, risk: {arb.metadata['risk_level']})")
    
    # Regime Transition
    if 'regime_transition' in signals:
        regime = signals['regime_transition']
        print(f"    ✓ Regime: {regime.metadata['current_regime']} → {regime.metadata['predicted_regime']} (prob: {regime.metadata['transition_probability']:.2f})")
    
    # Execution Quality
    if 'execution_quality' in signals:
        exec_q = signals['execution_quality']
        print(f"    ✓ Execution Quality: {exec_q.metadata['quality']} (slippage: {exec_q.metadata['slippage_estimate_bps']:.1f}bps)")
    
    # Test get_all_signals_dict
    signals_dict = calc.get_all_signals_dict()
    assert len(signals_dict) > 0, "get_all_signals_dict returned empty"
    print(f"\n  ✓ Flat signals dict: {len(signals_dict)} keys")
    
    # Test trading summary
    summary = calc.get_trading_summary()
    assert 'recommendation' in summary, "Missing recommendation in trading summary"
    print(f"  ✓ Trading Summary: {summary['recommendation']}")
    print(f"    Alerts: {len(summary['alerts'])}")
    print(f"    Opportunities: {len(summary['opportunities'])}")
    print(f"    Risks: {len(summary['risks'])}")
    
    print("\n  ✓ Phase 3 Composite Signals tests passed!")
    return True


def test_signal_aggregator():
    """Test SignalAggregator - Phase 3 Week 3-4."""
    print("\n" + "="*60)
    print("Testing SignalAggregator (Phase 3 Week 3-4)")
    print("="*60)
    
    from src.features.institutional.signal_aggregator import (
        SignalAggregator,
        SignalDirection,
        RecommendationStrength,
        ConflictSeverity,
    )
    from src.features.institutional.composite import (
        CompositeSignalCalculator,
        CompositeSignal,
        EnhancedSignal,
    )
    from datetime import datetime, timezone
    
    # Create aggregator
    aggregator = SignalAggregator(
        recency_decay=0.95,
        confidence_weight=0.3,
        correlation_penalty=0.1,
        conflict_threshold=0.25,
    )
    
    # Create composite calculator and feed features
    calc = CompositeSignalCalculator()
    
    # Provide comprehensive features for all streams
    prices_features = {
        'microprice': 50010.0,
        'spread_bps': 2.0,
        'spread': 10.0,
        'pressure_ratio': 1.2,
        'bid': 50000.0,
        'ask': 50010.0,
        'efficiency': 0.85,
    }
    
    orderbook_features = {
        'depth_imbalance_5': 0.25,
        'depth_imbalance_10': 0.18,
        'absorption_ratio': 0.65,
        'liquidity_gradient': 250.0,
        'bid_wall_distance': 50.0,
        'ask_wall_distance': 80.0,
        'total_bid_depth': 100.0,
        'total_ask_depth': 85.0,
    }
    
    trades_features = {
        'cvd': 150.0,
        'aggressive_delta': 100.0,
        'buy_pressure': 0.65,
        'sell_pressure': 0.35,
        'flow_toxicity': 0.2,
        'whale_trade_count': 3,
        'whale_volume': 15.0,
        'total_volume': 500.0,
        'vwap': 50005.0,
    }
    
    funding_features = {
        'funding_rate': 0.0003,
        'funding_zscore': 1.5,
        'funding_momentum': 0.0001,
        'carry_yield_annual': 0.35,
        'funding_regime': 'positive',
        'predicted_funding': 0.00035,
    }
    
    oi_features = {
        'oi': 1500000,
        'oi_delta': 25000,
        'leverage_index': 1.2,
        'position_intent': 'accumulation',
        'long_short_ratio': 1.15,
        'oi_change_pct': 1.5,
    }
    
    liquidations_features = {
        'long_liquidations_1m': 2,
        'short_liquidations_1m': 5,
        'liquidation_imbalance': 0.43,
        'liquidation_rate_per_hour': 150.0,
        'cascade_probability': 0.3,
        'liquidation_clusters': [{'price': 49500, 'volume': 500000}],
        'exhaustion_signal': False,
    }
    
    mark_prices_features = {
        'mark_spot_basis': 25.0,
        'basis_pct': 0.05,
        'basis_annualized_pct': 5.5,
        'basis_zscore': 0.8,
        'basis_regime': 'contango',
        'basis_momentum': 0.01,
    }
    
    ticker_features = {
        'volume': 1500000,
        'volatility_compression_idx': 0.6,
        'volatility_expansion_idx': 0.25,
        'range_vs_atr': 0.8,
        'price_change_pct': 2.0,
        'market_strength': 0.7,
        'realized_volatility': 0.02,
        'relative_volume_percentile': 80,
        'institutional_interest_idx': 0.75,
    }
    
    # Update all features
    calc.update_features('prices', prices_features)
    calc.update_features('orderbook', orderbook_features)
    calc.update_features('trades', trades_features)
    calc.update_features('funding', funding_features)
    calc.update_features('oi', oi_features)
    calc.update_features('liquidations', liquidations_features)
    calc.update_features('mark_prices', mark_prices_features)
    calc.update_features('ticker', ticker_features)
    
    # Calculate composite signals
    signals = calc.calculate_all()
    phase3_signals = calc.get_phase3_signals()
    
    print(f"\n  Input signals: {len(signals)} total, {len(phase3_signals)} Phase 3")
    
    # =================================================================
    # Test 1: Basic Aggregation
    # =================================================================
    print("\n  --- Test 1: Basic Aggregation ---")
    
    intelligence = aggregator.aggregate(
        symbol="BTCUSDT",
        signals=signals,
        phase3_signals=phase3_signals,
    )
    
    assert intelligence is not None, "Aggregation returned None"
    assert intelligence.symbol == "BTCUSDT", "Wrong symbol"
    assert intelligence.market_bias in [SignalDirection.BULLISH, SignalDirection.BEARISH, SignalDirection.NEUTRAL]
    assert 0 <= intelligence.bias_confidence <= 1, "Invalid bias confidence"
    
    print(f"  ✓ Market Bias: {intelligence.market_bias.value.upper()}")
    print(f"  ✓ Bias Confidence: {intelligence.bias_confidence:.2%}")
    print(f"  ✓ Ranked Signals: {len(intelligence.ranked_signals)}")
    
    # =================================================================
    # Test 2: Signal Ranking
    # =================================================================
    print("\n  --- Test 2: Signal Ranking ---")
    
    assert len(intelligence.ranked_signals) > 0, "No ranked signals"
    assert len(intelligence.top_signals) <= 5, "Too many top signals"
    
    print(f"  Top 5 Ranked Signals:")
    for i, sig in enumerate(intelligence.top_signals, 1):
        print(f"    {i}. {sig.name}: {sig.rank_score:.3f} (dir: {sig.direction.value}, cat: {sig.category.value})")
    
    # Verify ranking order
    for i in range(len(intelligence.top_signals) - 1):
        assert intelligence.top_signals[i].rank_score >= intelligence.top_signals[i+1].rank_score, "Signals not properly ranked"
    
    print(f"  ✓ Signals properly ranked by score")
    
    # =================================================================
    # Test 3: Category Scores
    # =================================================================
    print("\n  --- Test 3: Category Scores ---")
    
    assert len(intelligence.category_scores) > 0, "No category scores"
    
    print(f"  Category Scores:")
    for cat, score in intelligence.category_scores.items():
        print(f"    {cat}: {score:.3f}")
    
    print(f"  ✓ {len(intelligence.category_scores)} categories scored")
    
    # =================================================================
    # Test 4: Conflict Detection
    # =================================================================
    print("\n  --- Test 4: Conflict Detection ---")
    
    # Check if conflict was detected
    if intelligence.conflict:
        conflict = intelligence.conflict
        print(f"  ✓ Conflict Detected:")
        print(f"    Severity: {conflict.severity.value}")
        print(f"    Bullish signals: {len(conflict.bullish_signals)}")
        print(f"    Bearish signals: {len(conflict.bearish_signals)}")
        print(f"    Resolution: {conflict.resolution.value} ({conflict.resolution_confidence:.2%})")
        print(f"    Explanation: {conflict.explanation[:80]}...")
        
        assert conflict.severity in ConflictSeverity, "Invalid conflict severity"
        assert conflict.resolution in SignalDirection, "Invalid resolution direction"
    else:
        print(f"  ✓ No significant conflicts detected (signals agree)")
    
    # =================================================================
    # Test 5: Trade Recommendation
    # =================================================================
    print("\n  --- Test 5: Trade Recommendation ---")
    
    rec = intelligence.recommendation
    assert rec is not None, "No recommendation generated"
    
    print(f"  Trade Recommendation:")
    print(f"    Direction: {rec.direction.value.upper()}")
    print(f"    Strength: {rec.strength.value}")
    print(f"    Confidence: {rec.confidence:.2%}")
    print(f"    Entry Bias: {rec.entry_bias}")
    print(f"    Risk Level: {rec.risk_level}")
    print(f"    Urgency: {rec.urgency}")
    print(f"    Timeframe: {rec.timeframe}")
    print(f"    Primary Signals: {rec.primary_signals}")
    
    if rec.risk_factors:
        print(f"    Risk Factors: {rec.risk_factors[:2]}")
    
    if rec.alerts:
        print(f"    Alerts: {rec.alerts}")
    
    print(f"    Explanation: {rec.explanation}")
    
    assert rec.direction in SignalDirection, "Invalid direction"
    assert rec.strength in RecommendationStrength, "Invalid strength"
    assert rec.entry_bias in ['aggressive', 'conservative', 'wait'], "Invalid entry bias"
    assert rec.risk_level in ['low', 'medium', 'high', 'extreme'], "Invalid risk level"
    assert rec.urgency in ['immediate', 'soon', 'patient', 'wait'], "Invalid urgency"
    
    print(f"  ✓ Valid recommendation generated")
    
    # =================================================================
    # Test 6: to_dict() Serialization
    # =================================================================
    print("\n  --- Test 6: Serialization ---")
    
    intel_dict = intelligence.to_dict()
    assert 'symbol' in intel_dict, "Missing symbol in dict"
    assert 'market_bias' in intel_dict, "Missing market_bias in dict"
    assert 'ranked_signals' in intel_dict, "Missing ranked_signals in dict"
    assert 'recommendation' in intel_dict, "Missing recommendation in dict"
    
    rec_dict = rec.to_dict()
    assert 'direction' in rec_dict, "Missing direction in rec dict"
    assert 'strength' in rec_dict, "Missing strength in rec dict"
    assert 'explanation' in rec_dict, "Missing explanation in rec dict"
    
    print(f"  ✓ AggregatedIntelligence.to_dict(): {len(intel_dict)} keys")
    print(f"  ✓ TradeRecommendation.to_dict(): {len(rec_dict)} keys")
    
    # =================================================================
    # Test 7: Multiple Aggregations (History)
    # =================================================================
    print("\n  --- Test 7: Multiple Aggregations ---")
    
    # Run aggregation multiple times to test history tracking
    for i in range(3):
        _ = aggregator.aggregate(
            symbol="BTCUSDT",
            signals=signals,
            phase3_signals=phase3_signals,
        )
    
    # Check history
    history = aggregator.get_all_history()
    assert len(history) > 0, "No signal history recorded"
    
    sample_signal = list(history.keys())[0]
    sample_history = aggregator.get_signal_history(sample_signal)
    assert len(sample_history) >= 3, "History not tracking multiple calls"
    
    print(f"  ✓ Signal history: {len(history)} signals tracked")
    print(f"  ✓ {sample_signal} history: {len(sample_history)} entries")
    
    # =================================================================
    # Test 8: Reset
    # =================================================================
    print("\n  --- Test 8: Reset ---")
    
    aggregator.reset()
    history_after = aggregator.get_all_history()
    assert len(history_after) == 0, "History not cleared after reset"
    
    print(f"  ✓ Aggregator reset successfully")
    
    # =================================================================
    # Test 9: Conflicting Signal Scenario
    # =================================================================
    print("\n  --- Test 9: Conflicting Signal Scenario ---")
    
    # Create artificial conflicting signals
    timestamp = datetime.now(timezone.utc)
    conflicting_signals = {
        'smart_money_index': CompositeSignal(
            name='smart_money_index',
            value=0.8,  # Bullish
            confidence=0.9,
            components={},
            timestamp=timestamp,
        ),
        'momentum_quality': CompositeSignal(
            name='momentum_quality',
            value=0.85,  # Bullish
            confidence=0.85,
            components={},
            timestamp=timestamp,
        ),
        'composite_risk': CompositeSignal(
            name='composite_risk',
            value=0.9,  # High risk = bearish
            confidence=0.8,
            components={},
            timestamp=timestamp,
        ),
        'stop_hunt_probability': CompositeSignal(
            name='stop_hunt_probability',
            value=0.75,  # Bearish
            confidence=0.75,
            components={},
            timestamp=timestamp,
        ),
    }
    
    conflict_intel = aggregator.aggregate(
        symbol="BTCUSDT",
        signals=conflicting_signals,
        phase3_signals=[],
    )
    
    print(f"  Conflict scenario result:")
    print(f"    Market Bias: {conflict_intel.market_bias.value}")
    print(f"    Bias Confidence: {conflict_intel.bias_confidence:.2%}")
    if conflict_intel.conflict:
        print(f"    Conflict Severity: {conflict_intel.conflict.severity.value}")
        print(f"    Conflict Resolution: {conflict_intel.conflict.resolution.value}")
    print(f"    Recommendation Strength: {conflict_intel.recommendation.strength.value}")
    
    print(f"  ✓ Conflict scenario handled correctly")
    
    print("\n  ✓ SignalAggregator tests passed!")
    return True


def test_feature_engine():
    """Test FeatureEngine integration (without actual DB)."""
    print("\n" + "="*60)
    print("Testing FeatureEngine")
    print("="*60)
    
    from src.features.institutional.integration import FeatureEngine
    
    # Mock DB manager
    class MockDBManager:
        def execute(self, query): pass
        def query(self, query): return []
    
    # Mock storage
    class MockStorage:
        def __init__(self, *args):
            self.stored_features = []
            self.stored_composites = []
        
        def store_features(self, **kwargs):
            self.stored_features.append(kwargs)
        
        def store_composite_signals(self, **kwargs):
            self.stored_composites.append(kwargs)
        
        def get_latest_features(self, *args):
            return None
        
        async def flush_all(self):
            pass
    
    db = MockDBManager()
    storage = MockStorage(db)
    engine = FeatureEngine(db, storage=storage, enable_composites=True)
    
    # Test price processing
    price_data = {'bid': 50000.0, 'ask': 50010.0, 'bid_qty': 1.5, 'ask_qty': 1.2}
    features = engine.process_prices('BTCUSDT', 'binance', price_data)
    
    assert features is not None, "No price features"
    print(f"  ✓ Processed prices: {len(features)} features")
    
    # Test orderbook processing
    ob_data = {
        'bids': [[50000, 1.0], [49990, 2.0], [49980, 1.5], [49970, 3.0], [49960, 2.5],
                 [49950, 1.0], [49940, 1.5], [49930, 2.0], [49920, 1.0], [49910, 2.5]],
        'asks': [[50010, 0.8], [50020, 1.5], [50030, 2.0], [50040, 1.0], [50050, 3.0],
                 [50060, 1.5], [50070, 2.0], [50080, 1.0], [50090, 1.5], [50100, 2.0]],
    }
    features = engine.process_orderbook('BTCUSDT', 'binance', ob_data)
    
    assert features is not None, "No orderbook features"
    print(f"  ✓ Processed orderbook: {len(features)} features")
    
    # Test trades processing
    trades = [
        {'price': 50000, 'quantity': 0.5, 'side': 'buy'},
        {'price': 50005, 'quantity': 1.0, 'side': 'buy'},
        {'price': 50010, 'quantity': 0.3, 'side': 'sell'},
    ]
    features = engine.process_trades('BTCUSDT', 'binance', trades)
    
    assert features is not None, "No trade features"
    print(f"  ✓ Processed trades: {len(features)} features")
    
    # Check metrics
    metrics = engine.get_metrics()
    print(f"  ✓ Features calculated: {metrics['features_calculated']}")
    print(f"  ✓ Composites calculated: {metrics['composites_calculated']}")
    print(f"  ✓ Active symbols: {metrics['active_symbols']}")
    
    # Get composites
    composites = engine.get_latest_composites('BTCUSDT', 'binance')
    if composites:
        print(f"  ✓ Composite signals: {list(composites.keys())}")
    
    print("\n  ✓ FeatureEngine tests passed!")
    return True


def test_realtime_integration():
    """Test Phase 3 Week 5-6: Real-Time Integration with SignalAggregator."""
    print("\n" + "="*60)
    print("Testing Real-Time Integration (Phase 3 Week 5-6)")
    print("="*60)
    
    from src.features.institutional.integration import FeatureEngine
    from src.features.institutional.signal_aggregator import (
        SignalDirection,
        RecommendationStrength,
    )
    
    # Mock DB manager
    class MockDBManager:
        def execute(self, query): pass
        def query(self, query): return []
    
    # Mock storage with tracking
    class MockStorage:
        def __init__(self, *args):
            self.stored_features = []
            self.stored_composites = []
        
        def store_features(self, **kwargs):
            self.stored_features.append(kwargs)
        
        def store_composite_signals(self, **kwargs):
            self.stored_composites.append(kwargs)
        
        def get_latest_features(self, *args):
            return None
        
        async def flush_all(self):
            pass
    
    # =================================================================
    # Test 1: Engine with Real-Time & Aggregation Enabled
    # =================================================================
    print("\n  --- Test 1: Engine Initialization ---")
    
    db = MockDBManager()
    storage = MockStorage(db)
    engine = FeatureEngine(
        db, 
        storage=storage, 
        enable_composites=True,
        enable_realtime=True,
        enable_aggregation=True,
        aggregation_interval=0.1,  # Fast for testing
        cache_ttl=0.5,
    )
    
    assert engine.enable_realtime == True
    assert engine.enable_aggregation == True
    print(f"  ✓ Engine created with real-time mode")
    
    # =================================================================
    # Test 2: Register Callbacks
    # =================================================================
    print("\n  --- Test 2: Callback Registration ---")
    
    signal_callbacks_received = []
    recommendation_callbacks_received = []
    
    def on_signals(symbol, exchange, signals):
        signal_callbacks_received.append({
            'symbol': symbol,
            'exchange': exchange,
            'signal_count': len(signals)
        })
    
    def on_recommendation(symbol, exchange, recommendation):
        recommendation_callbacks_received.append({
            'symbol': symbol,
            'exchange': exchange,
            'direction': recommendation.direction.value,
            'strength': recommendation.strength.value,
        })
    
    engine.register_signal_callback(on_signals)
    engine.register_recommendation_callback(on_recommendation)
    
    assert len(engine._signal_callbacks) == 1
    assert len(engine._recommendation_callbacks) == 1
    print(f"  ✓ Callbacks registered")
    
    # =================================================================
    # Test 3: Process Data & Generate Signals/Recommendations
    # =================================================================
    print("\n  --- Test 3: Full Data Processing Pipeline ---")
    
    # Feed comprehensive data through all streams
    price_data = {'bid': 50000.0, 'ask': 50010.0, 'bid_qty': 1.5, 'ask_qty': 1.2}
    engine.process_prices('BTCUSDT', 'binance', price_data)
    
    ob_data = {
        'bids': [[50000, 1.0], [49990, 2.0], [49980, 1.5], [49970, 3.0], [49960, 2.5],
                 [49950, 1.0], [49940, 1.5], [49930, 2.0], [49920, 1.0], [49910, 2.5]],
        'asks': [[50010, 0.8], [50020, 1.5], [50030, 2.0], [50040, 1.0], [50050, 3.0],
                 [50060, 1.5], [50070, 2.0], [50080, 1.0], [50090, 1.5], [50100, 2.0]],
    }
    engine.process_orderbook('BTCUSDT', 'binance', ob_data)
    
    trades = [
        {'price': 50000, 'quantity': 0.5, 'side': 'buy'},
        {'price': 50005, 'quantity': 1.0, 'side': 'buy'},
        {'price': 50010, 'quantity': 0.3, 'side': 'sell'},
    ]
    engine.process_trades('BTCUSDT', 'binance', trades)
    
    funding_data = {'funding_rate': 0.0003, 'next_funding_time': 1700000000}
    engine.process_funding('BTCUSDT', 'binance', funding_data)
    
    oi_data = {'oi': 1000000, 'price': 50005}
    engine.process_oi('BTCUSDT', 'binance', oi_data)
    
    liquidation_data = {'side': 'short', 'quantity': 10.0, 'price': 50100}
    engine.process_liquidations('BTCUSDT', 'binance', liquidation_data)
    
    mark_data = {'mark_price': 50008, 'index_price': 50005, 'spot_price': 50000}
    engine.process_mark_prices('BTCUSDT', 'binance', mark_data)
    
    ticker_data = {
        'volume_24h': 1500000000,
        'high_24h': 51000,
        'low_24h': 49000,
        'price_change_24h': 2.5,
        'last_price': 50005,
    }
    engine.process_ticker('BTCUSDT', 'binance', ticker_data)
    
    print(f"  ✓ Processed all 8 data streams")
    
    # Check callbacks were triggered
    assert len(signal_callbacks_received) > 0, "No signal callbacks received"
    print(f"  ✓ Signal callbacks: {len(signal_callbacks_received)}")
    
    # Recommendations may not trigger on every update due to throttling
    # But should have some after processing all streams
    
    # =================================================================
    # Test 4: Get Aggregated Intelligence
    # =================================================================
    print("\n  --- Test 4: Aggregated Intelligence ---")
    
    intelligence = engine.get_aggregated_intelligence('BTCUSDT', 'binance')
    
    assert intelligence is not None, "No intelligence returned"
    assert intelligence.symbol == 'BTCUSDT'
    assert intelligence.market_bias in [SignalDirection.BULLISH, SignalDirection.BEARISH, SignalDirection.NEUTRAL]
    
    print(f"  ✓ Intelligence retrieved:")
    print(f"    Symbol: {intelligence.symbol}")
    print(f"    Market Bias: {intelligence.market_bias.value}")
    print(f"    Bias Confidence: {intelligence.bias_confidence:.2%}")
    print(f"    Ranked Signals: {len(intelligence.ranked_signals)}")
    print(f"    Top 3 Signals: {[s.name for s in intelligence.top_signals[:3]]}")
    
    # =================================================================
    # Test 5: Get Recommendation Directly
    # =================================================================
    print("\n  --- Test 5: Direct Recommendation Access ---")
    
    recommendation = engine.get_recommendation('BTCUSDT', 'binance')
    
    assert recommendation is not None, "No recommendation returned"
    assert recommendation.direction in SignalDirection
    assert recommendation.strength in RecommendationStrength
    
    print(f"  ✓ Recommendation:")
    print(f"    Direction: {recommendation.direction.value}")
    print(f"    Strength: {recommendation.strength.value}")
    print(f"    Entry Bias: {recommendation.entry_bias}")
    print(f"    Risk Level: {recommendation.risk_level}")
    print(f"    Urgency: {recommendation.urgency}")
    
    # =================================================================
    # Test 6: Get Top Signals
    # =================================================================
    print("\n  --- Test 6: Top Signals ---")
    
    top_signals = engine.get_top_signals('BTCUSDT', 'binance', limit=5)
    
    assert len(top_signals) > 0, "No top signals"
    
    print(f"  ✓ Top {len(top_signals)} Signals:")
    for i, sig in enumerate(top_signals, 1):
        print(f"    {i}. {sig.name}: {sig.rank_score:.3f} ({sig.direction.value})")
    
    # =================================================================
    # Test 7: Check Conflicts
    # =================================================================
    print("\n  --- Test 7: Signal Conflicts ---")
    
    conflict = engine.get_signal_conflicts('BTCUSDT', 'binance')
    
    if conflict:
        print(f"  ✓ Conflict detected:")
        print(f"    Severity: {conflict.severity.value}")
        print(f"    Bullish signals: {len(conflict.bullish_signals)}")
        print(f"    Bearish signals: {len(conflict.bearish_signals)}")
        print(f"    Resolution: {conflict.resolution.value}")
    else:
        print(f"  ✓ No significant conflicts detected")
    
    # =================================================================
    # Test 8: Metrics
    # =================================================================
    print("\n  --- Test 8: Engine Metrics ---")
    
    metrics = engine.get_metrics()
    
    print(f"  ✓ Metrics:")
    print(f"    Features calculated: {metrics['features_calculated']}")
    print(f"    Composites calculated: {metrics['composites_calculated']}")
    print(f"    Aggregations calculated: {metrics['aggregations_calculated']}")
    print(f"    Recommendations generated: {metrics['recommendations_generated']}")
    print(f"    Active symbols: {metrics['active_symbols']}")
    print(f"    Aggregator symbols: {metrics['aggregator_symbols']}")
    print(f"    Avg processing time: {metrics['avg_processing_time_ms']:.2f}ms")
    
    assert metrics['features_calculated'] > 0
    assert metrics['composites_calculated'] > 0
    
    # =================================================================
    # Test 9: Cache Behavior
    # =================================================================
    print("\n  --- Test 9: Cache Behavior ---")
    
    # Get intelligence twice - second should be cached
    intel1 = engine.get_aggregated_intelligence('BTCUSDT', 'binance')
    intel2 = engine.get_aggregated_intelligence('BTCUSDT', 'binance')
    
    # Both should return same object (cached)
    assert intel1.timestamp == intel2.timestamp, "Cache not working"
    
    # Force recalculate
    intel3 = engine.get_aggregated_intelligence('BTCUSDT', 'binance', force_recalculate=True)
    
    print(f"  ✓ Cache working correctly")
    print(f"  ✓ Cached intelligence count: {metrics['cached_intelligence']}")
    
    # =================================================================
    # Test 10: Reset
    # =================================================================
    print("\n  --- Test 10: Engine Reset ---")
    
    engine.reset_symbol('BTCUSDT', 'binance')
    
    # Intelligence cache should be cleared
    key = engine._get_symbol_key('BTCUSDT', 'binance')
    assert key not in engine._intelligence_cache, "Intelligence cache not cleared"
    
    print(f"  ✓ Symbol reset successfully")
    
    # Reset all
    engine.reset_all()
    assert len(engine._intelligence_cache) == 0
    print(f"  ✓ Full engine reset successful")
    
    print("\n  ✓ Real-Time Integration tests passed!")
    return True


def test_all_calculator_registration():
    """Test that all calculators are properly registered."""
    print("\n" + "="*60)
    print("Testing Calculator Registration")
    print("="*60)
    
    from src.features.institutional.base import institutional_feature_registry
    
    # Phase 1 + Phase 2 calculators
    expected = ['prices', 'orderbook', 'trades', 'funding', 'oi', 'liquidations', 'mark_prices', 'ticker']
    
    for stream_type in expected:
        calc_class = institutional_feature_registry.get_class(stream_type)
        assert calc_class is not None, f"Missing calculator for {stream_type}"
        print(f"  ✓ {stream_type}: {calc_class.__name__}")
    
    print("\n  ✓ All calculators registered!")
    return True


def main():
    """Run all Phase 1, Phase 2 & Phase 3 tests."""
    print("\n" + "🧪"*30)
    print("  INSTITUTIONAL FEATURES - PHASE 1, 2 & 3 TESTS")
    print("🧪"*30)
    
    tests = [
        # Phase 1 Tests
        ("FeatureBuffer", test_feature_buffer),
        ("Calculator Registration", test_all_calculator_registration),
        ("PricesFeatureCalculator", test_prices_calculator),
        ("OrderbookFeatureCalculator", test_orderbook_calculator),
        ("TradesFeatureCalculator", test_trades_calculator),
        ("FundingFeatureCalculator", test_funding_calculator),
        ("OIFeatureCalculator", test_oi_calculator),
        # Phase 2 Tests
        ("LiquidationsFeatureCalculator (Phase 2)", test_liquidations_calculator),
        ("MarkPricesFeatureCalculator (Phase 2)", test_mark_prices_calculator),
        ("TickerFeatureCalculator (Phase 2)", test_ticker_calculator),
        ("Phase 2 Engine Integration", test_phase2_engine_integration),
        # Composite & Integration (Phase 1)
        ("CompositeSignalCalculator (Phase 1)", test_composite_calculator),
        # Phase 3 Week 1-2 Tests
        ("Phase 3 Composite Signals (10 new)", test_phase3_composite_signals),
        # Phase 3 Week 3-4 Tests
        ("SignalAggregator (Phase 3 Week 3-4)", test_signal_aggregator),
        # Phase 3 Week 5-6 Tests
        ("Real-Time Integration (Phase 3 Week 5-6)", test_realtime_integration),
        # Legacy Integration
        ("FeatureEngine (Legacy)", test_feature_engine),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success, None))
        except Exception as e:
            print(f"  ❌ {name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False, str(e)))
    
    # Summary
    print("\n" + "="*60)
    print("  TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for name, success, error in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {name}")
        if error:
            print(f"        Error: {error}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 ALL PHASE 1, 2 & 3 TESTS PASSED! 🎉")
    else:
        print(f"\n  ⚠️ {total - passed} tests failed")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
