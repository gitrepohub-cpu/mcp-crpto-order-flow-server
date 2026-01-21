"""
Live + Historical Combined Data Tools

MCP tools that combine real-time streaming data with historical DuckDB data
for comprehensive market analysis.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from xml.sax.saxutils import escape as xml_escape

# Import live stream client
import os
USE_DIRECT_MODE = os.environ.get("USE_DIRECT_EXCHANGES", "true").lower() in ("true", "1", "yes")

if USE_DIRECT_MODE:
    from src.storage.direct_exchange_client import get_direct_client as get_live_client
else:
    from src.storage.websocket_client import get_arbitrage_client as get_live_client

# Import historical query manager
from src.tools.duckdb_historical_tools import (
    get_query_manager,
    get_historical_prices,
    get_historical_trades,
    get_historical_funding_rates,
    get_historical_liquidations,
    get_historical_open_interest,
    SYMBOLS, EXCHANGES
)

logger = logging.getLogger(__name__)


async def _ensure_live_client_connected(client):
    """Ensure live client is connected."""
    is_connected = getattr(client, '_started', False) or getattr(client, 'connected', False)
    if not is_connected:
        connected = await client.connect()
        if not connected:
            return False
        await asyncio.sleep(2.0)
    return True


# ============================================================================
# COMBINED LIVE + HISTORICAL TOOLS
# ============================================================================

async def get_market_snapshot_full(
    symbol: str,
    include_historical: bool = True,
    historical_minutes: int = 60
) -> str:
    """
    Get comprehensive market snapshot combining LIVE and HISTORICAL data.
    
    Args:
        symbol: Trading pair (BTCUSDT, ETHUSDT, etc.)
        include_historical: Whether to include historical context
        historical_minutes: Minutes of historical data to include
    
    Returns:
        XML with live prices + historical context + analysis
    """
    symbol_upper = symbol.upper()
    symbol_lower = symbol.lower()
    
    result = {
        'symbol': symbol_upper,
        'timestamp': datetime.utcnow().isoformat(),
        'live_data': {},
        'historical_context': {},
        'analysis': {}
    }
    
    # Get LIVE data
    try:
        client = get_live_client()
        if await _ensure_live_client_connected(client):
            # Live prices
            live_prices = await client.get_prices_snapshot(symbol_upper)
            result['live_data']['prices'] = live_prices.get(symbol_upper, {})
            
            # Live orderbooks
            live_orderbooks = await client.get_orderbooks(symbol_upper)
            result['live_data']['orderbooks'] = live_orderbooks.get(symbol_upper, {})
            
            # Live funding rates
            live_funding = await client.get_funding_rates(symbol_upper)
            result['live_data']['funding_rates'] = live_funding.get(symbol_upper, {})
            
            # Recent arbitrage opportunities
            opportunities = await client.get_arbitrage_opportunities(symbol=symbol_upper, limit=5)
            result['live_data']['arbitrage_opportunities'] = opportunities
    except Exception as e:
        logger.warning(f"Error getting live data: {e}")
        result['live_data']['error'] = str(e)
    
    # Get HISTORICAL context
    if include_historical:
        try:
            qm = get_query_manager()
            
            # Historical price stats
            for exc in EXCHANGES['futures'][:3]:  # Top 3 exchanges
                table_name = qm.get_table_name(symbol_lower, exc, 'futures', 'prices')
                if qm.table_exists(table_name):
                    try:
                        stats_query = f"""
                            SELECT 
                                AVG(mid_price) as avg_price,
                                MIN(mid_price) as low_price,
                                MAX(mid_price) as high_price,
                                STDDEV(mid_price) as volatility,
                                COUNT(*) as tick_count
                            FROM {table_name}
                            WHERE timestamp >= NOW() - INTERVAL '{historical_minutes} minutes'
                        """
                        stats = qm.execute_query(stats_query)
                        if stats and stats[0][0]:
                            result['historical_context'][f'{exc}_stats'] = {
                                'avg_price': stats[0][0],
                                'low': stats[0][1],
                                'high': stats[0][2],
                                'volatility': stats[0][3],
                                'ticks': stats[0][4]
                            }
                    except Exception as e:
                        logger.debug(f"Error getting historical stats for {exc}: {e}")
            
            # Historical trade flow
            for exc in EXCHANGES['futures'][:2]:
                table_name = qm.get_table_name(symbol_lower, exc, 'futures', 'trades')
                if qm.table_exists(table_name):
                    try:
                        flow_query = f"""
                            SELECT 
                                SUM(CASE WHEN side = 'buy' THEN value ELSE 0 END) as buy_volume,
                                SUM(CASE WHEN side = 'sell' THEN value ELSE 0 END) as sell_volume,
                                COUNT(*) as trade_count
                            FROM {table_name}
                            WHERE timestamp >= NOW() - INTERVAL '{historical_minutes} minutes'
                        """
                        flow = qm.execute_query(flow_query)
                        if flow and flow[0][2]:
                            buy_vol = flow[0][0] or 0
                            sell_vol = flow[0][1] or 0
                            result['historical_context'][f'{exc}_flow'] = {
                                'buy_volume': buy_vol,
                                'sell_volume': sell_vol,
                                'net_flow': buy_vol - sell_vol,
                                'buy_sell_ratio': buy_vol / sell_vol if sell_vol > 0 else 0,
                                'trade_count': flow[0][2]
                            }
                    except Exception as e:
                        logger.debug(f"Error getting trade flow for {exc}: {e}")
            
            # Recent liquidations summary
            total_long_liqs = 0
            total_short_liqs = 0
            for exc in EXCHANGES['futures']:
                table_name = qm.get_table_name(symbol_lower, exc, 'futures', 'liquidations')
                if qm.table_exists(table_name):
                    try:
                        liq_query = f"""
                            SELECT 
                                SUM(CASE WHEN side = 'long' THEN value ELSE 0 END),
                                SUM(CASE WHEN side = 'short' THEN value ELSE 0 END)
                            FROM {table_name}
                            WHERE timestamp >= NOW() - INTERVAL '{historical_minutes} minutes'
                        """
                        liqs = qm.execute_query(liq_query)
                        if liqs:
                            total_long_liqs += liqs[0][0] or 0
                            total_short_liqs += liqs[0][1] or 0
                    except:
                        pass
            
            result['historical_context']['liquidations'] = {
                'long_liquidated': total_long_liqs,
                'short_liquidated': total_short_liqs,
                'net_bias': 'LONG PAIN' if total_long_liqs > total_short_liqs else 'SHORT PAIN'
            }
            
        except Exception as e:
            logger.warning(f"Error getting historical context: {e}")
            result['historical_context']['error'] = str(e)
    
    # Generate ANALYSIS
    result['analysis'] = _generate_market_analysis(result)
    
    return _format_full_snapshot_xml(result)


async def get_price_with_history(
    symbol: str,
    exchange: str = "binance",
    market_type: str = "futures",
    historical_minutes: int = 30
) -> str:
    """
    Get current price with historical price context.
    
    Args:
        symbol: Trading pair
        exchange: Exchange to query
        market_type: 'futures' or 'spot'
        historical_minutes: Minutes of history for context
    
    Returns:
        XML with current price + historical range + change analysis
    """
    symbol_upper = symbol.upper()
    symbol_lower = symbol.lower()
    exchange_lower = exchange.lower()
    
    result = {
        'symbol': symbol_upper,
        'exchange': exchange_lower,
        'market_type': market_type,
        'timestamp': datetime.utcnow().isoformat()
    }
    
    # Get LIVE price
    try:
        client = get_live_client()
        if await _ensure_live_client_connected(client):
            prices = await client.get_prices_snapshot(symbol_upper)
            if symbol_upper in prices:
                exc_key = f"{exchange_lower}_{market_type}"
                if exc_key in prices[symbol_upper]:
                    result['live_price'] = prices[symbol_upper][exc_key]
    except Exception as e:
        result['live_error'] = str(e)
    
    # Get HISTORICAL context
    try:
        qm = get_query_manager()
        table_name = qm.get_table_name(symbol_lower, exchange_lower, market_type, 'prices')
        
        if qm.table_exists(table_name):
            # Get historical stats
            stats_query = f"""
                SELECT 
                    FIRST(mid_price ORDER BY timestamp) as open_price,
                    LAST(mid_price ORDER BY timestamp) as close_price,
                    MIN(mid_price) as low_price,
                    MAX(mid_price) as high_price,
                    AVG(mid_price) as avg_price,
                    STDDEV(mid_price) as volatility,
                    COUNT(*) as tick_count
                FROM {table_name}
                WHERE timestamp >= NOW() - INTERVAL '{historical_minutes} minutes'
            """
            
            stats = qm.execute_query(stats_query)
            if stats and stats[0][0]:
                open_price = stats[0][0]
                close_price = stats[0][1]
                result['historical'] = {
                    'period_minutes': historical_minutes,
                    'open': open_price,
                    'high': stats[0][3],
                    'low': stats[0][2],
                    'close': close_price,
                    'average': stats[0][4],
                    'volatility': stats[0][5],
                    'ticks': stats[0][6],
                    'change': close_price - open_price,
                    'change_pct': ((close_price - open_price) / open_price * 100) if open_price else 0
                }
            
            # Get price percentiles
            percentile_query = f"""
                SELECT 
                    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY mid_price) as p25,
                    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY mid_price) as p50,
                    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY mid_price) as p75
                FROM {table_name}
                WHERE timestamp >= NOW() - INTERVAL '{historical_minutes} minutes'
            """
            
            percentiles = qm.execute_query(percentile_query)
            if percentiles and percentiles[0][0]:
                result['percentiles'] = {
                    'p25': percentiles[0][0],
                    'p50': percentiles[0][1],
                    'p75': percentiles[0][2]
                }
    except Exception as e:
        result['historical_error'] = str(e)
    
    return _format_price_with_history_xml(result)


async def get_funding_arbitrage_analysis(
    symbol: str,
    historical_hours: int = 24
) -> str:
    """
    Analyze funding rate arbitrage opportunities across exchanges.
    Combines live funding rates with historical patterns.
    
    Args:
        symbol: Trading pair
        historical_hours: Hours of historical data
    
    Returns:
        XML with funding arbitrage opportunities + historical patterns
    """
    symbol_upper = symbol.upper()
    symbol_lower = symbol.lower()
    
    result = {
        'symbol': symbol_upper,
        'timestamp': datetime.utcnow().isoformat(),
        'live_funding': {},
        'historical_funding': {},
        'arbitrage_opportunities': []
    }
    
    # Get LIVE funding rates
    try:
        client = get_live_client()
        if await _ensure_live_client_connected(client):
            funding = await client.get_funding_rates(symbol_upper)
            if symbol_upper in funding:
                result['live_funding'] = funding[symbol_upper]
    except Exception as e:
        result['live_error'] = str(e)
    
    # Get HISTORICAL funding patterns
    try:
        qm = get_query_manager()
        
        for exc in EXCHANGES['futures']:
            table_name = qm.get_table_name(symbol_lower, exc, 'futures', 'funding_rates')
            
            if qm.table_exists(table_name):
                try:
                    stats_query = f"""
                        SELECT 
                            AVG(funding_rate) as avg_rate,
                            MIN(funding_rate) as min_rate,
                            MAX(funding_rate) as max_rate,
                            STDDEV(funding_rate) as volatility,
                            SUM(funding_rate) as cumulative,
                            COUNT(*) as samples
                        FROM {table_name}
                        WHERE timestamp >= NOW() - INTERVAL '{historical_hours} hours'
                    """
                    
                    stats = qm.execute_query(stats_query)
                    if stats and stats[0][5]:  # Has samples
                        result['historical_funding'][exc] = {
                            'avg_rate': stats[0][0],
                            'min_rate': stats[0][1],
                            'max_rate': stats[0][2],
                            'volatility': stats[0][3],
                            'cumulative': stats[0][4],
                            'samples': stats[0][5],
                            'avg_annualized': stats[0][0] * 3 * 365 * 100 if stats[0][0] else 0
                        }
                except Exception as e:
                    logger.debug(f"Error getting funding history for {exc}: {e}")
    except Exception as e:
        result['historical_error'] = str(e)
    
    # Calculate funding arbitrage opportunities
    live_rates = result['live_funding']
    if len(live_rates) >= 2:
        exchanges = list(live_rates.keys())
        for i, exc1 in enumerate(exchanges):
            for exc2 in exchanges[i+1:]:
                rate1 = live_rates[exc1].get('funding_rate', 0) if isinstance(live_rates[exc1], dict) else live_rates[exc1]
                rate2 = live_rates[exc2].get('funding_rate', 0) if isinstance(live_rates[exc2], dict) else live_rates[exc2]
                
                if rate1 and rate2:
                    spread = abs(rate1 - rate2)
                    if spread > 0.0001:  # 0.01% minimum spread
                        long_exchange = exc1 if rate1 < rate2 else exc2
                        short_exchange = exc2 if rate1 < rate2 else exc1
                        result['arbitrage_opportunities'].append({
                            'long_exchange': long_exchange,
                            'short_exchange': short_exchange,
                            'rate_spread': spread,
                            'spread_pct': spread * 100,
                            'annualized_profit': spread * 3 * 365 * 100,
                            'strategy': f"Long {symbol_upper} on {long_exchange}, Short on {short_exchange}"
                        })
    
    # Sort by potential profit
    result['arbitrage_opportunities'].sort(key=lambda x: x['rate_spread'], reverse=True)
    
    return _format_funding_arbitrage_xml(result)


async def get_liquidation_heatmap(
    symbol: str,
    hours: int = 24,
    price_buckets: int = 20
) -> str:
    """
    Generate liquidation heatmap showing where liquidations occurred by price level.
    
    Args:
        symbol: Trading pair
        hours: Hours of historical data
        price_buckets: Number of price buckets for the heatmap
    
    Returns:
        XML heatmap of liquidations by price level
    """
    symbol_lower = symbol.lower()
    
    result = {
        'symbol': symbol.upper(),
        'timestamp': datetime.utcnow().isoformat(),
        'hours': hours,
        'heatmap': [],
        'summary': {}
    }
    
    try:
        qm = get_query_manager()
        
        all_liquidations = []
        
        for exc in EXCHANGES['futures']:
            table_name = qm.get_table_name(symbol_lower, exc, 'futures', 'liquidations')
            
            if qm.table_exists(table_name):
                try:
                    query = f"""
                        SELECT price, value, side, timestamp
                        FROM {table_name}
                        WHERE timestamp >= NOW() - INTERVAL '{hours} hours'
                    """
                    
                    liqs = qm.execute_query(query)
                    for liq in liqs:
                        all_liquidations.append({
                            'exchange': exc,
                            'price': liq[0],
                            'value': liq[1],
                            'side': liq[2],
                            'timestamp': str(liq[3])
                        })
                except Exception as e:
                    logger.debug(f"Error getting liquidations for {exc}: {e}")
        
        if all_liquidations:
            # Calculate price range
            prices = [l['price'] for l in all_liquidations]
            min_price = min(prices)
            max_price = max(prices)
            bucket_size = (max_price - min_price) / price_buckets if max_price > min_price else 1
            
            # Build heatmap
            buckets = {}
            for i in range(price_buckets):
                bucket_start = min_price + (i * bucket_size)
                bucket_end = bucket_start + bucket_size
                buckets[i] = {
                    'price_start': bucket_start,
                    'price_end': bucket_end,
                    'long_liquidations': 0,
                    'short_liquidations': 0,
                    'total_value': 0,
                    'count': 0
                }
            
            for liq in all_liquidations:
                bucket_idx = int((liq['price'] - min_price) / bucket_size)
                bucket_idx = min(bucket_idx, price_buckets - 1)
                
                buckets[bucket_idx]['count'] += 1
                buckets[bucket_idx]['total_value'] += liq['value']
                
                if liq['side'] == 'long':
                    buckets[bucket_idx]['long_liquidations'] += liq['value']
                else:
                    buckets[bucket_idx]['short_liquidations'] += liq['value']
            
            result['heatmap'] = list(buckets.values())
            
            # Summary
            total_long = sum(b['long_liquidations'] for b in buckets.values())
            total_short = sum(b['short_liquidations'] for b in buckets.values())
            result['summary'] = {
                'total_liquidations': len(all_liquidations),
                'total_long_value': total_long,
                'total_short_value': total_short,
                'price_range': f"${min_price:,.2f} - ${max_price:,.2f}",
                'hottest_zone': max(buckets.values(), key=lambda x: x['total_value'])
            }
    except Exception as e:
        result['error'] = str(e)
    
    return _format_liquidation_heatmap_xml(result)


async def compare_live_vs_historical(
    symbol: str,
    exchange: str = "binance",
    market_type: str = "futures"
) -> str:
    """
    Compare current live price against historical averages to detect anomalies.
    
    Args:
        symbol: Trading pair
        exchange: Exchange to analyze
        market_type: 'futures' or 'spot'
    
    Returns:
        XML comparison with deviation analysis
    """
    symbol_upper = symbol.upper()
    symbol_lower = symbol.lower()
    exchange_lower = exchange.lower()
    
    result = {
        'symbol': symbol_upper,
        'exchange': exchange_lower,
        'market_type': market_type,
        'timestamp': datetime.utcnow().isoformat(),
        'comparisons': []
    }
    
    # Get live price
    live_price = None
    try:
        client = get_live_client()
        if await _ensure_live_client_connected(client):
            prices = await client.get_prices_snapshot(symbol_upper)
            if symbol_upper in prices:
                exc_key = f"{exchange_lower}_{market_type}"
                if exc_key in prices[symbol_upper]:
                    price_data = prices[symbol_upper][exc_key]
                    live_price = price_data.get('mid_price') if isinstance(price_data, dict) else price_data
                    result['live_price'] = live_price
    except Exception as e:
        result['live_error'] = str(e)
    
    if live_price is None:
        return _format_comparison_xml(result)
    
    # Compare against different historical windows
    windows = [
        ('5m', 5),
        ('15m', 15),
        ('1h', 60),
        ('4h', 240),
        ('24h', 1440)
    ]
    
    try:
        qm = get_query_manager()
        table_name = qm.get_table_name(symbol_lower, exchange_lower, market_type, 'prices')
        
        if qm.table_exists(table_name):
            for label, minutes in windows:
                try:
                    query = f"""
                        SELECT 
                            AVG(mid_price) as avg_price,
                            STDDEV(mid_price) as stddev,
                            MIN(mid_price) as min_price,
                            MAX(mid_price) as max_price
                        FROM {table_name}
                        WHERE timestamp >= NOW() - INTERVAL '{minutes} minutes'
                    """
                    
                    stats = qm.execute_query(query)
                    if stats and stats[0][0]:
                        avg_price = stats[0][0]
                        stddev = stats[0][1] or 0
                        
                        deviation = live_price - avg_price
                        deviation_pct = (deviation / avg_price * 100) if avg_price else 0
                        z_score = (deviation / stddev) if stddev else 0
                        
                        result['comparisons'].append({
                            'window': label,
                            'minutes': minutes,
                            'avg_price': avg_price,
                            'stddev': stddev,
                            'min': stats[0][2],
                            'max': stats[0][3],
                            'deviation': deviation,
                            'deviation_pct': deviation_pct,
                            'z_score': z_score,
                            'signal': _get_deviation_signal(z_score)
                        })
                except Exception as e:
                    logger.debug(f"Error comparing {label}: {e}")
    except Exception as e:
        result['historical_error'] = str(e)
    
    return _format_comparison_xml(result)


def _get_deviation_signal(z_score: float) -> str:
    """Get signal based on z-score deviation."""
    if z_score > 2:
        return "SIGNIFICANTLY_HIGH"
    elif z_score > 1:
        return "MODERATELY_HIGH"
    elif z_score < -2:
        return "SIGNIFICANTLY_LOW"
    elif z_score < -1:
        return "MODERATELY_LOW"
    else:
        return "NORMAL"


def _generate_market_analysis(result: Dict) -> Dict:
    """Generate market analysis from combined data."""
    analysis = {
        'market_state': 'UNKNOWN',
        'signals': [],
        'recommendations': []
    }
    
    # Analyze trade flow
    for key, flow in result.get('historical_context', {}).items():
        if '_flow' in key and isinstance(flow, dict):
            ratio = flow.get('buy_sell_ratio', 1)
            if ratio > 1.5:
                analysis['signals'].append(f"Strong buying pressure on {key.split('_')[0]}")
            elif ratio < 0.67:
                analysis['signals'].append(f"Strong selling pressure on {key.split('_')[0]}")
    
    # Analyze liquidations
    liqs = result.get('historical_context', {}).get('liquidations', {})
    if liqs:
        if liqs.get('long_liquidated', 0) > liqs.get('short_liquidated', 0) * 2:
            analysis['signals'].append("Heavy long liquidations - possible capitulation")
            analysis['market_state'] = 'BEARISH_CAPITULATION'
        elif liqs.get('short_liquidated', 0) > liqs.get('long_liquidated', 0) * 2:
            analysis['signals'].append("Heavy short liquidations - possible short squeeze")
            analysis['market_state'] = 'BULLISH_SQUEEZE'
    
    # Check arbitrage opportunities
    arb_opps = result.get('live_data', {}).get('arbitrage_opportunities', [])
    if arb_opps:
        best_opp = max(arb_opps, key=lambda x: x.get('profit_pct', 0)) if arb_opps else None
        if best_opp and best_opp.get('profit_pct', 0) > 0.1:
            analysis['recommendations'].append(
                f"Arbitrage opportunity: {best_opp.get('profit_pct', 0):.3f}% profit"
            )
    
    return analysis


# ============================================================================
# XML FORMATTING FUNCTIONS
# ============================================================================

def _format_full_snapshot_xml(result: Dict) -> str:
    """Format full snapshot as XML."""
    xml_parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<market_snapshot_full>',
        f'  <symbol>{xml_escape(result["symbol"])}</symbol>',
        f'  <timestamp>{xml_escape(result["timestamp"])}</timestamp>',
        '',
        '  <live_data>'
    ]
    
    # Live prices
    prices = result.get('live_data', {}).get('prices', {})
    if prices:
        xml_parts.append('    <live_prices>')
        for exc, data in prices.items():
            if isinstance(data, dict):
                xml_parts.append(f'      <price exchange="{xml_escape(exc)}">')
                for k, v in data.items():
                    if isinstance(v, (int, float)):
                        xml_parts.append(f'        <{k}>{v:.4f}</{k}>')
                    else:
                        xml_parts.append(f'        <{k}>{xml_escape(str(v))}</{k}>')
                xml_parts.append('      </price>')
        xml_parts.append('    </live_prices>')
    
    # Live funding
    funding = result.get('live_data', {}).get('funding_rates', {})
    if funding:
        xml_parts.append('    <live_funding>')
        for exc, data in funding.items():
            rate = data.get('funding_rate', data) if isinstance(data, dict) else data
            xml_parts.append(f'      <funding exchange="{xml_escape(exc)}">{rate:.6f}</funding>')
        xml_parts.append('    </live_funding>')
    
    xml_parts.append('  </live_data>')
    xml_parts.append('')
    xml_parts.append('  <historical_context>')
    
    # Historical stats
    for key, data in result.get('historical_context', {}).items():
        if isinstance(data, dict) and not key.endswith('_error'):
            xml_parts.append(f'    <{key}>')
            for k, v in data.items():
                if isinstance(v, float):
                    xml_parts.append(f'      <{k}>{v:,.4f}</{k}>')
                else:
                    xml_parts.append(f'      <{k}>{xml_escape(str(v))}</{k}>')
            xml_parts.append(f'    </{key}>')
    
    xml_parts.append('  </historical_context>')
    xml_parts.append('')
    xml_parts.append('  <analysis>')
    
    analysis = result.get('analysis', {})
    xml_parts.append(f'    <market_state>{xml_escape(analysis.get("market_state", "UNKNOWN"))}</market_state>')
    
    if analysis.get('signals'):
        xml_parts.append('    <signals>')
        for sig in analysis['signals']:
            xml_parts.append(f'      <signal>{xml_escape(sig)}</signal>')
        xml_parts.append('    </signals>')
    
    if analysis.get('recommendations'):
        xml_parts.append('    <recommendations>')
        for rec in analysis['recommendations']:
            xml_parts.append(f'      <recommendation>{xml_escape(rec)}</recommendation>')
        xml_parts.append('    </recommendations>')
    
    xml_parts.append('  </analysis>')
    xml_parts.append('</market_snapshot_full>')
    
    return '\n'.join(xml_parts)


def _format_price_with_history_xml(result: Dict) -> str:
    """Format price with history as XML."""
    xml_parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<price_with_history>',
        f'  <symbol>{xml_escape(result["symbol"])}</symbol>',
        f'  <exchange>{xml_escape(result["exchange"])}</exchange>',
        f'  <market_type>{xml_escape(result["market_type"])}</market_type>',
        f'  <timestamp>{xml_escape(result["timestamp"])}</timestamp>'
    ]
    
    if 'live_price' in result:
        live = result['live_price']
        if isinstance(live, dict):
            xml_parts.append('  <live>')
            for k, v in live.items():
                xml_parts.append(f'    <{k}>{v:.4f if isinstance(v, float) else xml_escape(str(v))}</{k}>')
            xml_parts.append('  </live>')
        else:
            xml_parts.append(f'  <live_price>{live:.4f}</live_price>')
    
    if 'historical' in result:
        hist = result['historical']
        xml_parts.append('  <historical>')
        xml_parts.append(f'    <period_minutes>{hist["period_minutes"]}</period_minutes>')
        xml_parts.append(f'    <open>${hist["open"]:,.2f}</open>')
        xml_parts.append(f'    <high>${hist["high"]:,.2f}</high>')
        xml_parts.append(f'    <low>${hist["low"]:,.2f}</low>')
        xml_parts.append(f'    <close>${hist["close"]:,.2f}</close>')
        xml_parts.append(f'    <average>${hist["average"]:,.2f}</average>')
        xml_parts.append(f'    <change>${hist["change"]:,.2f}</change>')
        xml_parts.append(f'    <change_pct>{hist["change_pct"]:.2f}%</change_pct>')
        xml_parts.append(f'    <ticks>{hist["ticks"]}</ticks>')
        xml_parts.append('  </historical>')
    
    if 'percentiles' in result:
        pct = result['percentiles']
        xml_parts.append('  <percentiles>')
        xml_parts.append(f'    <p25>${pct["p25"]:,.2f}</p25>')
        xml_parts.append(f'    <p50>${pct["p50"]:,.2f}</p50>')
        xml_parts.append(f'    <p75>${pct["p75"]:,.2f}</p75>')
        xml_parts.append('  </percentiles>')
    
    xml_parts.append('</price_with_history>')
    return '\n'.join(xml_parts)


def _format_funding_arbitrage_xml(result: Dict) -> str:
    """Format funding arbitrage analysis as XML."""
    xml_parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<funding_arbitrage_analysis>',
        f'  <symbol>{xml_escape(result["symbol"])}</symbol>',
        f'  <timestamp>{xml_escape(result["timestamp"])}</timestamp>',
        '',
        '  <live_funding_rates>'
    ]
    
    for exc, data in result.get('live_funding', {}).items():
        rate = data.get('funding_rate', data) if isinstance(data, dict) else data
        xml_parts.append(f'    <rate exchange="{xml_escape(exc)}">{rate:.6f}</rate>')
    
    xml_parts.append('  </live_funding_rates>')
    xml_parts.append('')
    xml_parts.append('  <historical_patterns>')
    
    for exc, data in result.get('historical_funding', {}).items():
        xml_parts.append(f'    <exchange name="{xml_escape(exc)}">')
        xml_parts.append(f'      <avg_rate>{data["avg_rate"]:.6f}</avg_rate>')
        xml_parts.append(f'      <avg_annualized>{data["avg_annualized"]:.2f}%</avg_annualized>')
        xml_parts.append(f'      <min_rate>{data["min_rate"]:.6f}</min_rate>')
        xml_parts.append(f'      <max_rate>{data["max_rate"]:.6f}</max_rate>')
        xml_parts.append(f'      <cumulative>{data["cumulative"]:.6f}</cumulative>')
        xml_parts.append('    </exchange>')
    
    xml_parts.append('  </historical_patterns>')
    xml_parts.append('')
    xml_parts.append('  <arbitrage_opportunities>')
    
    for opp in result.get('arbitrage_opportunities', []):
        xml_parts.append('    <opportunity>')
        xml_parts.append(f'      <long_exchange>{xml_escape(opp["long_exchange"])}</long_exchange>')
        xml_parts.append(f'      <short_exchange>{xml_escape(opp["short_exchange"])}</short_exchange>')
        xml_parts.append(f'      <rate_spread>{opp["spread_pct"]:.4f}%</rate_spread>')
        xml_parts.append(f'      <annualized_profit>{opp["annualized_profit"]:.2f}%</annualized_profit>')
        xml_parts.append(f'      <strategy>{xml_escape(opp["strategy"])}</strategy>')
        xml_parts.append('    </opportunity>')
    
    xml_parts.append('  </arbitrage_opportunities>')
    xml_parts.append('</funding_arbitrage_analysis>')
    
    return '\n'.join(xml_parts)


def _format_liquidation_heatmap_xml(result: Dict) -> str:
    """Format liquidation heatmap as XML."""
    xml_parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<liquidation_heatmap>',
        f'  <symbol>{xml_escape(result["symbol"])}</symbol>',
        f'  <hours>{result["hours"]}</hours>',
        f'  <timestamp>{xml_escape(result["timestamp"])}</timestamp>'
    ]
    
    if 'summary' in result:
        s = result['summary']
        xml_parts.append('  <summary>')
        xml_parts.append(f'    <total_liquidations>{s.get("total_liquidations", 0)}</total_liquidations>')
        xml_parts.append(f'    <total_long_value>${s.get("total_long_value", 0):,.0f}</total_long_value>')
        xml_parts.append(f'    <total_short_value>${s.get("total_short_value", 0):,.0f}</total_short_value>')
        xml_parts.append(f'    <price_range>{xml_escape(str(s.get("price_range", "N/A")))}</price_range>')
        xml_parts.append('  </summary>')
    
    xml_parts.append('  <heatmap>')
    for bucket in result.get('heatmap', []):
        intensity = bucket['total_value'] / max(1, max(b['total_value'] for b in result.get('heatmap', [{'total_value': 1}])))
        xml_parts.append(f'    <zone intensity="{intensity:.2f}">')
        xml_parts.append(f'      <price_range>${bucket["price_start"]:,.0f} - ${bucket["price_end"]:,.0f}</price_range>')
        xml_parts.append(f'      <long_liquidations>${bucket["long_liquidations"]:,.0f}</long_liquidations>')
        xml_parts.append(f'      <short_liquidations>${bucket["short_liquidations"]:,.0f}</short_liquidations>')
        xml_parts.append(f'      <total>${bucket["total_value"]:,.0f}</total>')
        xml_parts.append(f'      <count>{bucket["count"]}</count>')
        xml_parts.append('    </zone>')
    xml_parts.append('  </heatmap>')
    
    xml_parts.append('</liquidation_heatmap>')
    return '\n'.join(xml_parts)


def _format_comparison_xml(result: Dict) -> str:
    """Format live vs historical comparison as XML."""
    xml_parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<live_vs_historical>',
        f'  <symbol>{xml_escape(result["symbol"])}</symbol>',
        f'  <exchange>{xml_escape(result["exchange"])}</exchange>',
        f'  <timestamp>{xml_escape(result["timestamp"])}</timestamp>'
    ]
    
    if 'live_price' in result:
        xml_parts.append(f'  <live_price>${result["live_price"]:,.2f}</live_price>')
    
    xml_parts.append('  <comparisons>')
    for comp in result.get('comparisons', []):
        xml_parts.append(f'    <window period="{xml_escape(comp["window"])}">')
        xml_parts.append(f'      <avg_price>${comp["avg_price"]:,.2f}</avg_price>')
        xml_parts.append(f'      <range>${comp["min"]:,.2f} - ${comp["max"]:,.2f}</range>')
        xml_parts.append(f'      <deviation>${comp["deviation"]:,.2f}</deviation>')
        xml_parts.append(f'      <deviation_pct>{comp["deviation_pct"]:.2f}%</deviation_pct>')
        xml_parts.append(f'      <z_score>{comp["z_score"]:.2f}</z_score>')
        xml_parts.append(f'      <signal>{xml_escape(comp["signal"])}</signal>')
        xml_parts.append('    </window>')
    xml_parts.append('  </comparisons>')
    
    xml_parts.append('</live_vs_historical>')
    return '\n'.join(xml_parts)
