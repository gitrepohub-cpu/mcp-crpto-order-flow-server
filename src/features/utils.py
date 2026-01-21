"""
Shared Utilities for Feature Calculators

Common functions used across multiple feature calculation scripts.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import statistics

logger = logging.getLogger(__name__)


# =============================================================================
# STATISTICAL UTILITIES
# =============================================================================

def calculate_zscore(value: float, mean: float, std: float) -> float:
    """Calculate z-score for a value."""
    if std == 0:
        return 0.0
    return (value - mean) / std


def calculate_percentile_rank(value: float, values: List[float]) -> float:
    """Calculate percentile rank of a value in a list."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    count_below = sum(1 for v in sorted_values if v < value)
    return (count_below / len(sorted_values)) * 100


def rolling_mean(values: List[float], window: int) -> List[float]:
    """Calculate rolling mean."""
    if len(values) < window:
        return []
    return [
        statistics.mean(values[i:i+window])
        for i in range(len(values) - window + 1)
    ]


def rolling_std(values: List[float], window: int) -> List[float]:
    """Calculate rolling standard deviation."""
    if len(values) < window:
        return []
    return [
        statistics.stdev(values[i:i+window]) if len(values[i:i+window]) > 1 else 0
        for i in range(len(values) - window + 1)
    ]


def exponential_moving_average(values: List[float], span: int) -> List[float]:
    """Calculate exponential moving average."""
    if not values:
        return []
    
    alpha = 2 / (span + 1)
    ema = [values[0]]
    
    for value in values[1:]:
        ema.append(alpha * value + (1 - alpha) * ema[-1])
    
    return ema


def calculate_volatility(prices: List[float], annualize: bool = True) -> float:
    """Calculate price volatility from returns."""
    if len(prices) < 2:
        return 0.0
    
    returns = [
        (prices[i] - prices[i-1]) / prices[i-1]
        for i in range(1, len(prices))
        if prices[i-1] != 0
    ]
    
    if not returns:
        return 0.0
    
    vol = statistics.stdev(returns) if len(returns) > 1 else 0
    
    if annualize:
        # Assuming 1-minute data, annualize
        vol *= (365 * 24 * 60) ** 0.5
    
    return vol


# =============================================================================
# MARKET ANALYSIS UTILITIES
# =============================================================================

def calculate_vwap(trades: List[Dict]) -> float:
    """
    Calculate Volume Weighted Average Price.
    
    Args:
        trades: List of trade dicts with 'price' and 'quantity' keys
    
    Returns:
        VWAP value
    """
    total_value = sum(t.get('price', 0) * t.get('quantity', 0) for t in trades)
    total_volume = sum(t.get('quantity', 0) for t in trades)
    
    if total_volume == 0:
        return 0.0
    
    return total_value / total_volume


def calculate_buy_sell_ratio(trades: List[Dict]) -> Dict[str, float]:
    """
    Calculate buy/sell volume ratio.
    
    Args:
        trades: List of trade dicts with 'side' and 'value' keys
    
    Returns:
        Dict with buy_volume, sell_volume, ratio, and net_flow
    """
    buy_volume = sum(t.get('value', 0) for t in trades if t.get('side') == 'buy')
    sell_volume = sum(t.get('value', 0) for t in trades if t.get('side') == 'sell')
    
    return {
        'buy_volume': buy_volume,
        'sell_volume': sell_volume,
        'ratio': buy_volume / sell_volume if sell_volume > 0 else float('inf'),
        'net_flow': buy_volume - sell_volume
    }


def detect_large_trades(
    trades: List[Dict],
    threshold_percentile: float = 95
) -> List[Dict]:
    """
    Detect unusually large trades (whale activity).
    
    Args:
        trades: List of trade dicts with 'value' key
        threshold_percentile: Percentile threshold for "large"
    
    Returns:
        List of large trades
    """
    if not trades:
        return []
    
    values = [t.get('value', 0) for t in trades]
    sorted_values = sorted(values)
    threshold_idx = int(len(sorted_values) * threshold_percentile / 100)
    threshold = sorted_values[threshold_idx] if threshold_idx < len(sorted_values) else 0
    
    return [t for t in trades if t.get('value', 0) >= threshold]


def calculate_orderbook_imbalance(
    bids: List[Tuple[float, float]],
    asks: List[Tuple[float, float]],
    depth_levels: int = 10
) -> Dict[str, float]:
    """
    Calculate orderbook imbalance metrics.
    
    Args:
        bids: List of (price, quantity) tuples
        asks: List of (price, quantity) tuples
        depth_levels: Number of levels to consider
    
    Returns:
        Dict with imbalance metrics
    """
    bid_depth = sum(q for _, q in bids[:depth_levels])
    ask_depth = sum(q for _, q in asks[:depth_levels])
    total_depth = bid_depth + ask_depth
    
    imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0
    
    return {
        'bid_depth': bid_depth,
        'ask_depth': ask_depth,
        'imbalance': imbalance,  # -1 to 1, positive = more bids
        'imbalance_pct': imbalance * 100,
        'bid_ask_ratio': bid_depth / ask_depth if ask_depth > 0 else float('inf')
    }


def calculate_spread_metrics(
    bid_price: float,
    ask_price: float,
    mid_price: float = None
) -> Dict[str, float]:
    """Calculate spread metrics."""
    if mid_price is None:
        mid_price = (bid_price + ask_price) / 2
    
    spread = ask_price - bid_price
    spread_bps = (spread / mid_price * 10000) if mid_price > 0 else 0
    
    return {
        'spread': spread,
        'spread_bps': spread_bps,
        'mid_price': mid_price,
        'bid_price': bid_price,
        'ask_price': ask_price
    }


# =============================================================================
# TIME SERIES UTILITIES  
# =============================================================================

def resample_to_intervals(
    data: List[Dict],
    timestamp_key: str = 'timestamp',
    interval_minutes: int = 5
) -> List[Dict]:
    """
    Resample time series data to fixed intervals.
    
    Args:
        data: List of dicts with timestamp key
        timestamp_key: Key containing timestamp
        interval_minutes: Target interval in minutes
    
    Returns:
        Resampled data
    """
    if not data:
        return []
    
    # Group by interval
    buckets: Dict[datetime, List[Dict]] = {}
    
    for item in data:
        ts = item.get(timestamp_key)
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        
        # Round down to interval
        bucket_ts = ts.replace(
            minute=(ts.minute // interval_minutes) * interval_minutes,
            second=0,
            microsecond=0
        )
        
        if bucket_ts not in buckets:
            buckets[bucket_ts] = []
        buckets[bucket_ts].append(item)
    
    return [
        {'timestamp': ts, 'items': items}
        for ts, items in sorted(buckets.items())
    ]


def calculate_ohlc(prices: List[float]) -> Dict[str, float]:
    """Calculate OHLC from price list."""
    if not prices:
        return {'open': 0, 'high': 0, 'low': 0, 'close': 0}
    
    return {
        'open': prices[0],
        'high': max(prices),
        'low': min(prices),
        'close': prices[-1]
    }


# =============================================================================
# SIGNAL GENERATION UTILITIES
# =============================================================================

def generate_signal(
    signal_type: str,
    strength: float,
    description: str,
    metadata: Dict = None
) -> Dict[str, Any]:
    """
    Generate a standardized trading signal.
    
    Args:
        signal_type: 'BULLISH', 'BEARISH', 'NEUTRAL', 'WARNING'
        strength: Signal strength 0-1
        description: Human-readable description
        metadata: Additional signal data
    
    Returns:
        Standardized signal dict
    """
    return {
        'type': signal_type,
        'strength': min(1.0, max(0.0, strength)),
        'description': description,
        'timestamp': datetime.utcnow().isoformat(),
        'metadata': metadata or {}
    }


def classify_market_regime(
    volatility: float,
    trend_strength: float,
    volume_profile: str
) -> str:
    """
    Classify market regime based on indicators.
    
    Returns:
        One of: TRENDING_UP, TRENDING_DOWN, RANGING, HIGH_VOL, BREAKOUT
    """
    if volatility > 0.5:  # High volatility threshold
        return 'HIGH_VOL'
    
    if abs(trend_strength) > 0.7:
        return 'TRENDING_UP' if trend_strength > 0 else 'TRENDING_DOWN'
    
    if abs(trend_strength) < 0.2:
        return 'RANGING'
    
    return 'TRANSITIONING'


# =============================================================================
# CROSS-EXCHANGE UTILITIES
# =============================================================================

def calculate_cross_exchange_spread(
    prices: Dict[str, float]
) -> Dict[str, Any]:
    """
    Calculate spread between exchanges.
    
    Args:
        prices: Dict mapping exchange names to prices
    
    Returns:
        Dict with max spread, best buy/sell exchanges
    """
    if len(prices) < 2:
        return {'max_spread_pct': 0, 'best_buy': None, 'best_sell': None}
    
    max_price = max(prices.values())
    min_price = min(prices.values())
    
    max_exchange = max(prices, key=prices.get)
    min_exchange = min(prices, key=prices.get)
    
    spread_pct = ((max_price - min_price) / min_price * 100) if min_price > 0 else 0
    
    return {
        'max_spread_pct': spread_pct,
        'max_spread_usd': max_price - min_price,
        'best_buy': min_exchange,
        'best_sell': max_exchange,
        'best_buy_price': min_price,
        'best_sell_price': max_price
    }


def detect_price_divergence(
    exchange_prices: Dict[str, List[float]],
    threshold_std: float = 2.0
) -> List[Dict]:
    """
    Detect significant price divergence between exchanges.
    
    Args:
        exchange_prices: Dict mapping exchanges to price lists
        threshold_std: Number of std deviations to consider divergent
    
    Returns:
        List of divergence events
    """
    if len(exchange_prices) < 2:
        return []
    
    divergences = []
    
    # Get all timestamps (assume aligned)
    num_points = min(len(p) for p in exchange_prices.values())
    
    for i in range(num_points):
        prices_at_point = {exc: prices[i] for exc, prices in exchange_prices.items()}
        all_prices = list(prices_at_point.values())
        
        if len(all_prices) < 2:
            continue
        
        mean_price = statistics.mean(all_prices)
        std_price = statistics.stdev(all_prices) if len(all_prices) > 1 else 0
        
        for exc, price in prices_at_point.items():
            if std_price > 0:
                zscore = (price - mean_price) / std_price
                if abs(zscore) > threshold_std:
                    divergences.append({
                        'index': i,
                        'exchange': exc,
                        'price': price,
                        'mean': mean_price,
                        'zscore': zscore,
                        'direction': 'HIGH' if zscore > 0 else 'LOW'
                    })
    
    return divergences
