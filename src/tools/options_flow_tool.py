"""Options flow analysis tool for MCP"""

import asyncio
import logging
from typing import Optional
import nest_asyncio

# Enable nested event loops
nest_asyncio.apply()

from src.formatters.context_builder import OptionsContextBuilder
from src.formatters.xml_formatter import OptionsMCPFormatter
from src.storage.grpc_client import OptionsOrderFlowGRPCClient

logger = logging.getLogger(__name__)


def _run_async(coro):
    """Helper to run async code in potentially nested event loops"""
    try:
        loop = asyncio.get_running_loop()
        # If we're already in a running loop, we can await directly
        return coro
    except RuntimeError:
        # No running loop, create new one
        return asyncio.run(coro)


async def get_options_flow(ticker: str, history_minutes: int = 20) -> str:
    """
    Get comprehensive options order flow data for all monitored contracts of a ticker
    
    Args:
        ticker: Stock ticker symbol
        history_minutes: Minutes of history to include (default: 20)
        
    Returns:
        XML-formatted options flow analysis
    """
    grpc_client = None
    try:
        # BUG FIX: Validate ticker input
        if not ticker or not isinstance(ticker, str):
            return build_error_response("UNKNOWN", "Invalid ticker - must be a non-empty string")
        
        ticker = ticker.upper().strip()  # BUG FIX: Normalize ticker
        
        # BUG FIX: Validate history_minutes
        if not isinstance(history_minutes, int) or history_minutes < 1:
            history_minutes = 20
        history_minutes = min(history_minutes, 120)  # BUG FIX: Cap at 2 hours
        
        # Get gRPC client
        grpc_client = OptionsOrderFlowGRPCClient()
        
        # Get comprehensive snapshot from data broker with timeout
        try:
            snapshot = await asyncio.wait_for(
                grpc_client.get_options_order_flow_snapshot(
                    ticker=ticker,
                    expiration=None,  # All expirations
                    strikes=None,     # All strikes
                    option_types=None,  # All option types
                    history_seconds=history_minutes * 60,
                    include_patterns=True,
                    include_aggregations=True
                ),
                timeout=30.0  # 30 second timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"Timeout getting options flow for {ticker}")
            return build_error_response(ticker, "Request timeout - data broker not responding")
        except ConnectionError as e:
            logger.error(f"Connection error for {ticker}: {e}")
            return build_error_response(ticker, f"Connection error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error getting snapshot for {ticker}: {e}")
            return build_error_response(ticker, f"Connection error: {str(e)}")
        
        if not snapshot:
            return build_error_response(ticker, "Failed to get data from options order flow broker")
        
        # BUG FIX: Check for both 'error' status and 'success' status
        status = snapshot.get('status', '').lower()
        if status == 'error':
            return build_error_response(ticker, snapshot.get('message', 'Unknown error from data broker'))
        
        # BUG FIX: Verify we have actual data
        contracts = snapshot.get('contracts', [])
        patterns = snapshot.get('patterns', [])
        if not contracts and not patterns:
            logger.warning(f"No contracts or patterns found for {ticker}")
            # Still proceed - might be valid empty result
        
        # Create context builder (updated to use gRPC data)
        context_builder = OptionsContextBuilder(grpc_client)
        
        # Build comprehensive context from snapshot data
        context = await context_builder.build_comprehensive_response_from_snapshot(ticker, snapshot)
        
        # BUG FIX: Validate context before formatting
        if not context or 'error' in context:
            error_msg = context.get('error', 'Failed to build context') if context else 'Failed to build context'
            return build_error_response(ticker, error_msg)
        
        # Format as MCP XML
        formatter = OptionsMCPFormatter()
        mcp_xml = formatter.format_comprehensive(context)
        
        return mcp_xml
        
    except Exception as e:
        logger.exception(f"Error getting options flow for {ticker}: {e}")
        return build_error_response(ticker, str(e))
    finally:
        # BUG FIX: Always close gRPC client to prevent connection leaks
        if grpc_client:
            try:
                await grpc_client.close()
            except Exception as close_error:
                logger.warning(f"Error closing gRPC client: {close_error}")


def build_error_response(ticker: str, error_message: str) -> str:
    """Build error response in MCP format"""
    return f"""<options_order_flow ticker="{ticker}" error="true">
    <error_message>{error_message}</error_message>
    <possible_causes>
        <cause>No monitoring configured for this ticker</cause>
        <cause>Data broker not running</cause>
        <cause>Network connectivity issue</cause>
    </possible_causes>
    <suggestions>
        <suggestion>Configure monitoring using configure_options_monitoring_tool</suggestion>
        <suggestion>Verify the ticker symbol is correct</suggestion>
        <suggestion>Check if the mcp-trading-data-broker is running</suggestion>
        <suggestion>Verify network connectivity to data broker</suggestion>
    </suggestions>
</options_order_flow>"""
