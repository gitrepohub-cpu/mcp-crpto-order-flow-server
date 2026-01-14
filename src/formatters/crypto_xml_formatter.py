"""
XML formatter for crypto arbitrage data.
Produces LLM-optimized output format for MCP consumption.
"""

from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class CryptoArbitrageFormatter:
    """Formats crypto arbitrage data into XML for MCP consumption."""
    
    # Exchange short names for display
    SOURCE_NAMES = {
        "binance_futures": "Binance Futures",
        "binance_spot": "Binance Spot",
        "bybit_futures": "Bybit Futures",
        "bybit_spot": "Bybit Spot",
        "okx_futures": "OKX Futures",
        "kraken_futures": "Kraken Futures",
        "gate_futures": "Gate.io Futures",
        "hyperliquid_futures": "Hyperliquid",
        "paradex_futures": "Paradex",
        "pyth": "Pyth Oracle"
    }
    
    def format_arbitrage_analysis(self, context: Dict) -> str:
        """Format comprehensive arbitrage analysis."""
        symbol = context.get("symbol", "UNKNOWN")
        timestamp = context.get("timestamp", datetime.utcnow().isoformat())
        prices = context.get("prices", {})
        spreads = context.get("spreads", {})
        opportunities = context.get("opportunities", [])
        analysis = context.get("analysis", {})
        
        xml_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<crypto_arbitrage_analysis symbol="{symbol}" timestamp="{timestamp}">'
        ]
        
        # Market Overview
        xml_parts.append("  <market_overview>")
        xml_parts.append(f"    <status>{analysis.get('market_status', 'UNKNOWN')}</status>")
        xml_parts.append(f"    <spread_assessment>{self._escape_xml(analysis.get('spread_assessment', ''))}</spread_assessment>")
        xml_parts.append(f"    <recommendation>{self._escape_xml(analysis.get('recommendation', ''))}</recommendation>")
        
        if "price_range" in analysis:
            pr = analysis["price_range"]
            xml_parts.append("    <price_range>")
            xml_parts.append(f"      <min_price>{self._format_price(pr.get('min', 0))}</min_price>")
            xml_parts.append(f"      <max_price>{self._format_price(pr.get('max', 0))}</max_price>")
            xml_parts.append(f"      <spread_pct>{pr.get('spread_pct', 0):.4f}%</spread_pct>")
            xml_parts.append("    </price_range>")
        
        xml_parts.append("  </market_overview>")
        
        # Exchange Prices
        xml_parts.append(f"  <exchange_prices count=\"{len(prices)}\">")
        sorted_prices = sorted(
            prices.items(), 
            key=lambda x: x[1].get("price", 0) if isinstance(x[1], dict) else 0, 
            reverse=True
        )
        for source, data in sorted_prices:
            if isinstance(data, dict):
                price = data.get("price", 0)
                ts = data.get("timestamp", 0)
            else:
                price = data
                ts = 0
            display_name = self.SOURCE_NAMES.get(source, source)
            xml_parts.append(f"    <exchange name=\"{display_name}\" id=\"{source}\">")
            xml_parts.append(f"      <price>{self._format_price(price)}</price>")
            xml_parts.append(f"      <raw_price>{price}</raw_price>")
            xml_parts.append(f"      <timestamp>{ts}</timestamp>")
            xml_parts.append(f"    </exchange>")
        xml_parts.append("  </exchange_prices>")
        
        # Spread Summary (if available)
        if spreads:
            xml_parts.append(self._format_spread_section(spreads))
        
        # Recent Opportunities
        xml_parts.append(f"  <arbitrage_opportunities count=\"{len(opportunities)}\">")
        for opp in opportunities[:10]:
            xml_parts.append(self._format_opportunity(opp))
        xml_parts.append("  </arbitrage_opportunities>")
        
        # Key Insights
        insights = analysis.get("key_insights", [])
        if insights:
            xml_parts.append("  <key_insights>")
            for insight in insights:
                xml_parts.append(f"    <insight>{self._escape_xml(insight)}</insight>")
            xml_parts.append("  </key_insights>")
        
        # Opportunity Statistics
        if "opportunity_stats" in analysis:
            stats = analysis["opportunity_stats"]
            xml_parts.append("  <opportunity_statistics>")
            xml_parts.append(f"    <total_count>{stats.get('count', 0)}</total_count>")
            xml_parts.append(f"    <average_profit_pct>{stats.get('avg_profit_pct', 0):.4f}%</average_profit_pct>")
            xml_parts.append(f"    <best_profit_pct>{stats.get('best_profit_pct', 0):.4f}%</best_profit_pct>")
            xml_parts.append("  </opportunity_statistics>")
        
        xml_parts.append("</crypto_arbitrage_analysis>")
        
        return "\n".join(xml_parts)
    
    def format_prices(self, prices: Dict) -> str:
        """Format price data for multiple symbols."""
        xml_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<exchange_prices timestamp="{datetime.utcnow().isoformat()}">'
        ]
        
        for symbol, sources in prices.items():
            xml_parts.append(f"  <symbol name=\"{symbol}\" exchanges=\"{len(sources)}\">")
            
            # Sort by price descending
            sorted_sources = sorted(
                sources.items(),
                key=lambda x: x[1].get("price", 0) if isinstance(x[1], dict) else 0,
                reverse=True
            )
            
            for source, data in sorted_sources:
                if isinstance(data, dict):
                    price = data.get("price", 0)
                else:
                    price = data
                display_name = self.SOURCE_NAMES.get(source, source)
                xml_parts.append(f"    <exchange name=\"{display_name}\" id=\"{source}\" price=\"{self._format_price(price)}\" raw=\"{price}\" />")
            xml_parts.append("  </symbol>")
        
        xml_parts.append("</exchange_prices>")
        return "\n".join(xml_parts)
    
    def format_spread_matrix(self, symbol: str, spreads: Dict, prices: Dict) -> str:
        """Format spread matrix showing pairwise differences."""
        xml_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<spread_matrix symbol="{symbol}" timestamp="{datetime.utcnow().isoformat()}">'
        ]
        
        spread_data = spreads.get("spreads", spreads)
        
        # Find significant spreads (positive arbitrage opportunities)
        significant = []
        for source1, targets in spread_data.items():
            if not isinstance(targets, dict):
                continue
            for source2, spread_pct in targets.items():
                if isinstance(spread_pct, (int, float)) and spread_pct > 0.02:
                    significant.append({
                        "buy": source1,
                        "sell": source2,
                        "spread": spread_pct,
                        "buy_name": self.SOURCE_NAMES.get(source1, source1),
                        "sell_name": self.SOURCE_NAMES.get(source2, source2)
                    })
        
        significant.sort(key=lambda x: x["spread"], reverse=True)
        
        # Best opportunities section
        xml_parts.append(f"  <best_opportunities count=\"{len(significant)}\">")
        for s in significant[:15]:
            xml_parts.append(f"    <opportunity>")
            xml_parts.append(f"      <action>BUY on {s['buy_name']}</action>")
            xml_parts.append(f"      <action>SELL on {s['sell_name']}</action>")
            xml_parts.append(f"      <spread_pct>{s['spread']:.4f}%</spread_pct>")
            xml_parts.append(f"    </opportunity>")
        xml_parts.append("  </best_opportunities>")
        
        # Current prices for context
        if prices:
            xml_parts.append("  <current_prices>")
            for source, data in prices.items():
                if isinstance(data, dict):
                    price = data.get("price", 0)
                else:
                    price = data
                display_name = self.SOURCE_NAMES.get(source, source)
                xml_parts.append(f"    <price exchange=\"{display_name}\">{self._format_price(price)}</price>")
            xml_parts.append("  </current_prices>")
        
        # Full matrix (condensed)
        xml_parts.append("  <full_matrix>")
        for source1, targets in spread_data.items():
            if not isinstance(targets, dict):
                continue
            source1_name = self.SOURCE_NAMES.get(source1, source1)
            positive_spreads = [(s2, sp) for s2, sp in targets.items() 
                               if isinstance(sp, (int, float)) and sp > 0]
            if positive_spreads:
                xml_parts.append(f"    <from exchange=\"{source1_name}\">")
                for source2, spread_pct in sorted(positive_spreads, key=lambda x: x[1], reverse=True)[:5]:
                    source2_name = self.SOURCE_NAMES.get(source2, source2)
                    xml_parts.append(f"      <to exchange=\"{source2_name}\" spread=\"{spread_pct:.4f}%\" />")
                xml_parts.append("    </from>")
        xml_parts.append("  </full_matrix>")
        
        xml_parts.append("</spread_matrix>")
        return "\n".join(xml_parts)
    
    def format_opportunities(self, opportunities: List[Dict]) -> str:
        """Format list of arbitrage opportunities."""
        xml_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<arbitrage_opportunities count="{len(opportunities)}" timestamp="{datetime.utcnow().isoformat()}">'
        ]
        
        if not opportunities:
            xml_parts.append("  <message>No arbitrage opportunities detected recently</message>")
            xml_parts.append("  <suggestion>Market may be efficient or scanner needs more time to collect data</suggestion>")
        else:
            # Group by symbol
            by_symbol = {}
            for opp in opportunities:
                sym = opp.get("symbol", "UNKNOWN")
                if sym not in by_symbol:
                    by_symbol[sym] = []
                by_symbol[sym].append(opp)
            
            for symbol, opps in by_symbol.items():
                xml_parts.append(f"  <symbol name=\"{symbol}\" count=\"{len(opps)}\">")
                for opp in opps:
                    xml_parts.append(self._format_opportunity(opp, indent=4))
                xml_parts.append("  </symbol>")
        
        xml_parts.append("</arbitrage_opportunities>")
        return "\n".join(xml_parts)
    
    def format_health_status(self, health: Dict) -> str:
        """Format health check response."""
        connected = health.get("connected", False)
        status = health.get("status", "unhealthy" if not connected else "healthy")
        
        xml_parts = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<arbitrage_scanner_health status="{status.upper()}">',
            f"  <connection>",
            f"    <connected>{connected}</connected>",
            f"    <websocket_url>{health.get('ws_url', 'unknown')}</websocket_url>",
            f"    <reconnect_attempts>{health.get('reconnect_attempts', 0)}</reconnect_attempts>",
            f"  </connection>",
            f"  <data_status>",
            f"    <symbols_tracked>{', '.join(health.get('symbols_tracked', [])) or 'None'}</symbols_tracked>",
            f"    <opportunities_cached>{health.get('opportunities_cached', 0)}</opportunities_cached>",
            f"  </data_status>",
        ]
        
        sources = health.get("sources_per_symbol", {})
        if sources:
            xml_parts.append("  <sources_by_symbol>")
            for symbol, source_list in sources.items():
                xml_parts.append(f"    <symbol name=\"{symbol}\" sources=\"{len(source_list)}\">")
                for src in source_list:
                    display_name = self.SOURCE_NAMES.get(src, src)
                    xml_parts.append(f"      <source id=\"{src}\">{display_name}</source>")
                xml_parts.append(f"    </symbol>")
            xml_parts.append("  </sources_by_symbol>")
        
        if not connected:
            xml_parts.append("  <troubleshooting>")
            xml_parts.append("    <suggestion>Ensure the Go arbitrage scanner is running</suggestion>")
            xml_parts.append("    <suggestion>Check: cd crypto-futures-arbitrage-scanner &amp;&amp; go run main.go</suggestion>")
            xml_parts.append("    <suggestion>Verify ARBITRAGE_SCANNER_HOST and ARBITRAGE_SCANNER_PORT env vars</suggestion>")
            xml_parts.append("  </troubleshooting>")
        
        xml_parts.append(f"  <checked_at>{health.get('timestamp', datetime.utcnow().isoformat())}</checked_at>")
        xml_parts.append("</arbitrage_scanner_health>")
        return "\n".join(xml_parts)
    
    def _format_opportunity(self, opp: Dict, indent: int = 4) -> str:
        """Format a single opportunity."""
        spaces = " " * indent
        buy_source = opp.get('buy_source', '')
        sell_source = opp.get('sell_source', '')
        buy_name = self.SOURCE_NAMES.get(buy_source, buy_source)
        sell_name = self.SOURCE_NAMES.get(sell_source, sell_source)
        
        return f"""{spaces}<opportunity symbol="{opp.get('symbol', '')}">
{spaces}  <buy exchange="{buy_name}" price="{self._format_price(opp.get('buy_price', 0))}" />
{spaces}  <sell exchange="{sell_name}" price="{self._format_price(opp.get('sell_price', 0))}" />
{spaces}  <profit_pct>{opp.get('profit_pct', 0):.4f}%</profit_pct>
{spaces}  <detected_at>{opp.get('detected_at', '')}</detected_at>
{spaces}</opportunity>"""
    
    def _format_spread_section(self, spreads: Dict) -> str:
        """Format spreads as part of larger document."""
        spread_data = spreads.get("spreads", spreads)
        lines = ["  <spread_summary>"]
        
        # Count positive spreads and find max
        positive_count = 0
        max_spread = 0
        max_pair = None
        
        for source1, targets in spread_data.items():
            if not isinstance(targets, dict):
                continue
            for source2, spread in targets.items():
                if isinstance(spread, (int, float)) and spread > 0:
                    positive_count += 1
                    if spread > max_spread:
                        max_spread = spread
                        max_pair = (source1, source2)
        
        lines.append(f"    <positive_spread_count>{positive_count}</positive_spread_count>")
        lines.append(f"    <max_spread_pct>{max_spread:.4f}%</max_spread_pct>")
        
        if max_pair:
            buy_name = self.SOURCE_NAMES.get(max_pair[0], max_pair[0])
            sell_name = self.SOURCE_NAMES.get(max_pair[1], max_pair[1])
            lines.append(f"    <best_route>BUY {buy_name} → SELL {sell_name}</best_route>")
        
        lines.append("  </spread_summary>")
        
        return "\n".join(lines)
    
    def _format_price(self, price: float) -> str:
        """Format price with appropriate decimals."""
        if not price or price == 0:
            return "$0.00"
        if price >= 1000:
            return f"${price:,.2f}"
        elif price >= 100:
            return f"${price:,.3f}"
        elif price >= 10:
            return f"${price:,.4f}"
        elif price >= 1:
            return f"${price:,.5f}"
        else:
            return f"${price:,.6f}"
    
    def _escape_xml(self, text: str) -> str:
        """Escape special XML characters."""
        if not text:
            return ""
        return (text
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&apos;"))
