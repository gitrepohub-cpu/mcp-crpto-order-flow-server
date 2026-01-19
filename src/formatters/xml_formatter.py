"""XML formatter for options order flow data"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class OptionsMCPFormatter:
    """
    Formats options order flow data into MCP XML structure.
    """

    def __init__(self):
        """Initialize MCP Formatter"""
        self.logger = logging.getLogger('OptionsMCPFormatter')

    def format_comprehensive(self, context: Dict[str, Any]) -> str:
        """
        Format comprehensive options flow context for all monitored contracts

        Args:
            context (Dict): Comprehensive context from ContextBuilder

        Returns:
            str: MCP XML string
        """
        ticker = context.get('ticker', '')
        timestamp = context.get('timestamp', datetime.now().isoformat())
        monitored_contracts = context.get('monitored_contracts', [])
        global_summary = context.get('summary', {})

        # Start building the XML
        xml = f'<options_order_flow ticker="{ticker}" timestamp="{timestamp}">\n'

        # Add monitored contracts section
        contract_count = len(monitored_contracts)
        xml += f'  <monitored_contracts count="{contract_count}">\n'

        # Group contracts by expiration
        expiration_groups = {}
        for contract in monitored_contracts:
            exp = contract.get('expiration', 0)
            if exp not in expiration_groups:
                expiration_groups[exp] = []
            expiration_groups[exp].append(contract)

        # Process each expiration group
        for expiration, contracts in expiration_groups.items():
            exp_display = self._format_expiration_date(expiration)
            xml += f'    <expiration date="{exp_display}">\n'

            # Group contracts by strike
            strike_groups = {}
            for contract in contracts:
                strike = contract.get('strike', 0)
                option_type = contract.get('option_type', '')
                if strike not in strike_groups:
                    strike_groups[strike] = {}
                strike_groups[strike][option_type] = contract

            # Process each strike
            for strike, options in sorted(strike_groups.items()):
                strike_display = self._format_strike(strike)
                xml += f'      <strike price="{strike_display}">\n'

                # Process call option if present
                if 'C' in options:
                    xml += self._format_option_contract(options['C'], 'call')

                # Process put option if present
                if 'P' in options:
                    xml += self._format_option_contract(options['P'], 'put')

                xml += '      </strike>\n'

            xml += '    </expiration>\n'

        xml += '  </monitored_contracts>\n'

        # Add global summary
        xml += self._format_global_summary(global_summary)

        # Close options_order_flow tag
        xml += '</options_order_flow>\n'

        return xml

    def _format_option_contract(self, contract: Dict[str, Any], tag_name: str) -> str:
        """
        Format a single option contract

        Args:
            contract (Dict): Contract data
            tag_name (str): XML tag name ('call' or 'put')

        Returns:
            str: Formatted XML
        """
        xml = f'        <{tag_name}>\n'

        # Add metadata
        xml += '          <metadata>\n'
        xml += f'            <monitoring_started>{contract.get("monitoring_started", "")}</monitoring_started>\n'
        xml += f'            <last_activity>{contract.get("last_activity", "")}</last_activity>\n'
        xml += '          </metadata>\n'

        # Add current state
        current_state = contract.get('current_state', {})
        xml += '          <current_state>\n'
        xml += f'            <activity_level>{current_state.get("activity_level", "LOW")}</activity_level>\n'
        xml += f'            <dominant_direction>{current_state.get("dominant_direction", "NEUTRAL")}</dominant_direction>\n'
        xml += f'            <significance>{current_state.get("significance", "LOW")}</significance>\n'
        xml += f'            <recent_pattern_count>{current_state.get("recent_pattern_count", 0)}</recent_pattern_count>\n'
        
        # Add pattern types breakdown
        pattern_types = current_state.get('pattern_types', {})
        if pattern_types:
            xml += '            <pattern_types>\n'
            for p_type, count in pattern_types.items():
                xml += f'              <{p_type.lower()}>{count}</{p_type.lower()}>\n'
            xml += '            </pattern_types>\n'
        
        xml += '          </current_state>\n'

        # Add patterns
        patterns = contract.get('patterns', [])
        if patterns:
            xml += '          <patterns>\n'
            for pattern in patterns[:10]:  # Limit to 10 most recent
                xml += self._format_pattern(pattern)
            xml += '          </patterns>\n'
        else:
            xml += '          <patterns />\n'

        # Add historical summary if available
        historical = contract.get('historical_summary', {})
        if historical:
            xml += self._format_historical_summary(historical)

        xml += f'        </{tag_name}>\n'
        return xml

    def _format_pattern(self, pattern: Dict[str, Any]) -> str:
        """Format a single pattern"""
        pattern_type = pattern.get('type', 'UNKNOWN')
        data = pattern.get('data', {}) if pattern.get('data') else pattern  # BUG FIX: Use pattern itself if data is empty
        
        xml = f'            <pattern type="{pattern_type}">\n'

        # Add timestamp - BUG FIX: Check all possible timestamp sources with proper type handling
        timestamp = (
            data.get('start_time') or 
            data.get('time') or 
            pattern.get('timestamp') or  # gRPC returns timestamp directly
            pattern.get('detected_at') or 
            0
        )
        
        # BUG FIX: Handle string timestamps from gRPC (ISO format)
        if isinstance(timestamp, str) and timestamp:
            # Already ISO format from gRPC
            pass
        elif isinstance(timestamp, (int, float)) and timestamp > 0:
            # Handle epoch timestamps
            if timestamp > 1000000000000:  # Milliseconds
                timestamp = datetime.fromtimestamp(timestamp / 1000).isoformat()
            elif timestamp > 1000000000:  # Seconds
                timestamp = datetime.fromtimestamp(timestamp).isoformat()
            else:
                timestamp = datetime.now().isoformat()  # Invalid timestamp
        else:
            timestamp = datetime.now().isoformat()  # Default to now
        xml += f'              <timestamp>{timestamp}</timestamp>\n'

        # Add direction - BUG FIX: Check both data and pattern level
        direction = data.get('direction') or pattern.get('direction', 'NEUTRAL')
        xml += f'              <direction>{direction}</direction>\n'

        # Add volume - BUG FIX: Use total_volume from gRPC patterns
        volume = data.get('volume') or pattern.get('total_volume', 0)
        xml += f'              <volume>{volume}</volume>\n'

        # Add confidence - BUG FIX: Check pattern level too
        confidence = data.get('confidence') or pattern.get('confidence', 0)
        xml += f'              <confidence>{confidence:.2f}</confidence>\n'

        # Add pattern-specific fields - BUG FIX: Also check gRPC metrics dict
        metrics = pattern.get('metrics', {}) or {}
        
        if pattern_type == 'SWEEP':
            duration = data.get('duration') or pattern.get('duration_seconds', 0) or metrics.get('duration', 0)
            xml += f'              <duration>{duration}</duration>\n'
            imbalance = data.get('imbalance') or metrics.get('imbalance', 0)
            xml += f'              <imbalance>{imbalance:.2f}</imbalance>\n'
            
        elif pattern_type == 'BLOCK':
            avg_bid = data.get('avg_bid') or metrics.get('avg_bid', 0)
            avg_ask = data.get('avg_ask') or metrics.get('avg_ask', 0)
            xml += f'              <avg_bid>${avg_bid:.2f}</avg_bid>\n'
            xml += f'              <avg_ask>${avg_ask:.2f}</avg_ask>\n'
            
        elif pattern_type == 'UNUSUAL_VOLUME':
            vol_increase = data.get('volume_increase') or metrics.get('volume_increase', 0)
            baseline = data.get('baseline_volume') or metrics.get('baseline_volume', 0)
            avg_vol = data.get('avg_volume') or metrics.get('avg_volume', 0)
            xml += f'              <volume_increase>{vol_increase:.2f}x</volume_increase>\n'
            xml += f'              <baseline_volume>{baseline}</baseline_volume>\n'
            xml += f'              <avg_volume>{avg_vol}</avg_volume>\n'
        
        # BUG FIX: Add description if available
        description = pattern.get('description', '')
        if description:
            xml += f'              <description>{self._escape_xml(description)}</description>\n'

        xml += '            </pattern>\n'
        return xml

    def _format_historical_summary(self, historical: Dict[str, Any]) -> str:
        """Format historical summary section"""
        xml = '          <historical_summary>\n'

        # Add intervals
        intervals = historical.get('intervals', [])
        if intervals:
            xml += '            <intervals>\n'
            for interval in intervals[-6:]:  # Last 6 intervals (3 hours)
                xml += '              <interval'
                if 'start' in interval:
                    xml += f' start="{interval["start"]}"'
                if 'end' in interval:
                    xml += f' end="{interval["end"]}"'
                xml += '>\n'

                xml += f'                <pattern_count>{interval.get("pattern_count", 0)}</pattern_count>\n'
                xml += f'                <volume>{interval.get("volume", 0)}</volume>\n'
                xml += f'                <direction>{interval.get("direction", "NEUTRAL")}</direction>\n'

                xml += '              </interval>\n'
            xml += '            </intervals>\n'

        # Add key transitions
        transitions = historical.get('key_transitions', [])
        if transitions:
            xml += '            <key_transitions>\n'
            for transition in transitions[:5]:  # Limit to 5 most recent
                xml += f'              <transition time="{transition.get("time", "")}">\n'
                xml += f'                <from>{transition.get("from", "UNKNOWN")}</from>\n'
                xml += f'                <to>{transition.get("to", "UNKNOWN")}</to>\n'

                trigger = transition.get('trigger_pattern', {})
                if trigger:
                    xml += f'                <trigger_pattern type="{trigger.get("type", "UNKNOWN")}" volume="{trigger.get("volume", 0)}" />\n'

                xml += '              </transition>\n'
            xml += '            </key_transitions>\n'

        xml += '          </historical_summary>\n'
        return xml

    def _format_global_summary(self, summary: Dict[str, Any]) -> str:
        """
        Format global summary across all contracts

        Args:
            summary (Dict): Global summary

        Returns:
            str: Formatted XML
        """
        xml = '  <summary>\n'

        # Most active strikes
        active_strikes = summary.get('most_active_strikes', [])
        if active_strikes:
            xml += '    <most_active_strikes>\n'
            for strike in active_strikes:
                xml += f'      <strike price="{strike.get("price", "")}" '
                xml += f'option_type="{strike.get("option_type", "")}" '
                xml += f'activity_level="{strike.get("activity_level", "MEDIUM")}" />\n'
            xml += '    </most_active_strikes>\n'

        # Institutional bias
        bias = summary.get('institutional_bias', {})
        if bias:
            xml += '    <institutional_bias>\n'
            xml += f'      <direction>{bias.get("direction", "NEUTRAL")}</direction>\n'
            xml += f'      <confidence>{bias.get("confidence", 0):.2f}</confidence>\n'

            primary_contracts = bias.get('primary_contracts', [])
            if primary_contracts:
                xml += '      <primary_contracts>\n'
                for contract in primary_contracts:
                    xml += f'        <contract strike="{contract.get("strike", "")}" '
                    xml += f'option_type="{contract.get("option_type", "")}" '
                    xml += f'expiration="{contract.get("expiration", "")}" />\n'
                xml += '      </primary_contracts>\n'

            xml += '    </institutional_bias>\n'

        # Recent trend
        trend = summary.get('recent_trend', {})
        if trend:
            xml += '    <recent_trend>\n'
            xml += f'      <timespan>{trend.get("timespan", "")}</timespan>\n'
            xml += f'      <description>{self._escape_xml(trend.get("description", ""))}</description>\n'

            direction_change = trend.get('direction_change', {})
            if direction_change and direction_change.get('time'):
                xml += f'      <direction_change time="{direction_change.get("time", "")}" '
                xml += f'from="{direction_change.get("from", "")}" '
                xml += f'to="{direction_change.get("to", "")}" />\n'

            xml += '    </recent_trend>\n'

        xml += '  </summary>\n'
        return xml

    def _format_expiration_date(self, expiration: int) -> str:
        """
        Format expiration date for display

        Args:
            expiration (int): Expiration date (YYYYMMDD)

        Returns:
            str: Formatted expiration (YYYY-MM-DD)
        """
        if not expiration:
            return "Unknown"

        exp_str = str(expiration)
        if len(exp_str) == 8:
            return f"{exp_str[0:4]}-{exp_str[4:6]}-{exp_str[6:8]}"
        return str(expiration)

    def _format_strike(self, strike: float) -> str:
        """
        Format strike price for display

        Args:
            strike (float): Strike price

        Returns:
            str: Formatted strike price
        """
        return f"{strike:.2f}"

    def _escape_xml(self, text: str) -> str:
        """Escape special XML characters"""
        if not text:
            return ""
        return (text.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                    .replace('"', "&quot;")
                    .replace("'", "&apos;"))
