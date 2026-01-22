"""
🎮 MCP Tools for Production Streaming Control
=============================================

Provides MCP tools for controlling the production streaming system:
- Start/stop streaming
- Get status and health
- Configure streaming parameters
- Monitor alerts
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Global controller reference
_controller = None
_controller_task = None


async def start_streaming(
    symbols: List[str] = None,
    exchanges: List[str] = None,
    config_path: str = "config/streaming_config.json"
) -> Dict[str, Any]:
    """
    Start the production streaming system.
    
    Initiates real-time data collection with integrated analytics:
    - Multi-exchange data ingestion
    - Automatic forecasting pipeline
    - Model drift detection
    - Health monitoring
    
    Args:
        symbols: List of trading pairs (e.g., ["BTCUSDT", "ETHUSDT"])
                 If None, uses config defaults
        exchanges: List of exchanges (e.g., ["binance", "bybit"])
                   If None, uses config defaults
        config_path: Path to streaming configuration file
    
    Returns:
        Dict with status and configuration details
        
    Example:
        >>> result = await start_streaming(
        ...     symbols=["BTCUSDT", "ETHUSDT"],
        ...     exchanges=["binance", "bybit"]
        ... )
        >>> print(result["status"])
        'STARTED'
    """
    global _controller, _controller_task
    
    try:
        from src.streaming.production_controller import ProductionStreamingController
        
        # Check if already running
        if _controller and _controller.is_running:
            return {
                "status": "ALREADY_RUNNING",
                "message": "Streaming system is already running",
                "current_config": _controller.get_status()
            }
        
        # Create new controller
        _controller = ProductionStreamingController(config_path)
        
        # Override config if provided
        if symbols:
            _controller.config["symbols"] = symbols
        if exchanges:
            _controller.config["exchanges"] = exchanges
        
        # Start in background task
        _controller_task = asyncio.create_task(_controller.start())
        
        # Give it a moment to start
        await asyncio.sleep(1)
        
        return {
            "status": "STARTED",
            "message": "Production streaming system started successfully",
            "config": {
                "symbols": _controller.config.get("symbols"),
                "exchanges": _controller.config.get("exchanges"),
                "market_type": _controller.config.get("market_type", "futures"),
                "forecast_interval_seconds": _controller.config.get("forecast_interval_seconds"),
                "auto_retrain": _controller.config.get("auto_retrain")
            },
            "streams_count": len(_controller.config.get("symbols", [])) * len(_controller.config.get("exchanges", []))
        }
        
    except Exception as e:
        logger.error(f"Failed to start streaming: {e}")
        return {
            "status": "ERROR",
            "message": f"Failed to start streaming: {str(e)}"
        }


async def stop_streaming() -> Dict[str, Any]:
    """
    Stop the production streaming system.
    
    Gracefully shuts down all streaming components:
    - Stops data collection
    - Flushes remaining buffers
    - Closes database connections
    
    Returns:
        Dict with shutdown status
        
    Example:
        >>> result = await stop_streaming()
        >>> print(result["status"])
        'STOPPED'
    """
    global _controller, _controller_task
    
    try:
        if not _controller:
            return {
                "status": "NOT_RUNNING",
                "message": "Streaming system is not running"
            }
        
        # Get final stats before shutdown
        final_stats = _controller.get_status()
        
        # Stop controller
        await _controller.stop()
        
        # Cancel task if exists
        if _controller_task and not _controller_task.done():
            _controller_task.cancel()
            try:
                await _controller_task
            except asyncio.CancelledError:
                pass
        
        _controller = None
        _controller_task = None
        
        return {
            "status": "STOPPED",
            "message": "Streaming system stopped successfully",
            "final_stats": final_stats
        }
        
    except Exception as e:
        logger.error(f"Error stopping streaming: {e}")
        return {
            "status": "ERROR",
            "message": f"Error stopping streaming: {str(e)}"
        }


async def get_streaming_status() -> Dict[str, Any]:
    """
    Get current streaming system status.
    
    Returns comprehensive status information:
    - Running state
    - Active symbols and exchanges
    - Ingestion rates
    - Forecast statistics
    - Error counts
    
    Returns:
        Dict with complete status information
        
    Example:
        >>> status = await get_streaming_status()
        >>> print(f"Records: {status['health']['records_ingested']}")
    """
    global _controller
    
    if not _controller:
        return {
            "status": "NOT_RUNNING",
            "message": "Streaming system is not running"
        }
    
    return _controller.get_status()


async def get_streaming_health() -> Dict[str, Any]:
    """
    Get detailed streaming system health report.
    
    Provides health assessment including:
    - Overall health status (HEALTHY/DEGRADED/STOPPED)
    - Identified issues
    - Detailed metrics
    - Active streams information
    
    Returns:
        Dict with health report
        
    Example:
        >>> health = await get_streaming_health()
        >>> print(f"Status: {health['status']}")
        >>> print(f"Issues: {health['issues']}")
    """
    global _controller
    
    if not _controller:
        return {
            "status": "NOT_RUNNING",
            "issues": ["Streaming system is not running"],
            "metrics": None
        }
    
    return _controller.get_health_report()


async def get_streaming_alerts(
    level: str = None,
    limit: int = 50
) -> Dict[str, Any]:
    """
    Get recent streaming system alerts.
    
    Args:
        level: Filter by alert level (INFO, LOW, MEDIUM, HIGH, CRITICAL)
               If None, returns all alerts
        limit: Maximum number of alerts to return
    
    Returns:
        Dict with alerts list
        
    Example:
        >>> alerts = await get_streaming_alerts(level="HIGH", limit=10)
        >>> print(f"Found {len(alerts['alerts'])} high-priority alerts")
    """
    global _controller
    
    if not _controller:
        return {
            "status": "NOT_RUNNING",
            "alerts": []
        }
    
    alerts = _controller.alerts[-limit:]
    
    if level:
        from src.streaming.production_controller import AlertLevel
        target_level = AlertLevel[level.upper()]
        alerts = [a for a in alerts if a.level == target_level]
    
    return {
        "status": "OK",
        "total_alerts": len(_controller.alerts),
        "alerts": [
            {
                "timestamp": a.timestamp.isoformat(),
                "level": a.level.value,
                "source": a.source,
                "message": a.message,
                "acknowledged": a.acknowledged
            }
            for a in alerts
        ]
    }


async def configure_streaming(
    forecast_interval: int = None,
    drift_check_interval: int = None,
    auto_retrain: bool = None,
    batch_size: int = None
) -> Dict[str, Any]:
    """
    Update streaming configuration on-the-fly.
    
    Allows modifying streaming parameters without restart:
    - Forecast generation interval
    - Drift check frequency
    - Auto-retraining behavior
    - Batch sizes
    
    Args:
        forecast_interval: Seconds between forecasts (default: 300)
        drift_check_interval: Seconds between drift checks (default: 600)
        auto_retrain: Enable automatic model retraining on drift
        batch_size: Records per database batch
    
    Returns:
        Dict with updated configuration
        
    Example:
        >>> result = await configure_streaming(
        ...     forecast_interval=600,
        ...     auto_retrain=True
        ... )
    """
    global _controller
    
    if not _controller:
        return {
            "status": "NOT_RUNNING",
            "message": "Start streaming first to configure"
        }
    
    updated = []
    
    if forecast_interval is not None:
        _controller.config["forecast_interval_seconds"] = forecast_interval
        updated.append(f"forecast_interval={forecast_interval}")
    
    if drift_check_interval is not None:
        _controller.config["drift_check_interval_seconds"] = drift_check_interval
        updated.append(f"drift_check_interval={drift_check_interval}")
    
    if auto_retrain is not None:
        _controller.config["auto_retrain"] = auto_retrain
        updated.append(f"auto_retrain={auto_retrain}")
    
    if batch_size is not None:
        _controller.config["batch_size"] = batch_size
        updated.append(f"batch_size={batch_size}")
    
    return {
        "status": "UPDATED",
        "changes": updated,
        "current_config": {
            "forecast_interval_seconds": _controller.config.get("forecast_interval_seconds"),
            "drift_check_interval_seconds": _controller.config.get("drift_check_interval_seconds"),
            "auto_retrain": _controller.config.get("auto_retrain"),
            "batch_size": _controller.config.get("batch_size")
        }
    }


async def get_realtime_analytics_status() -> Dict[str, Any]:
    """
    Get real-time analytics engine status.
    
    Returns information about the analytics pipeline:
    - Active streams
    - Forecast counts
    - Drift detection status
    - Model usage statistics
    
    Returns:
        Dict with analytics status
        
    Example:
        >>> status = await get_realtime_analytics_status()
        >>> print(f"Total forecasts: {status['total_forecasts']}")
    """
    try:
        from src.streaming.realtime_analytics import get_realtime_analytics
        
        analytics = get_realtime_analytics()
        return analytics.get_global_metrics()
        
    except Exception as e:
        return {
            "status": "ERROR",
            "message": str(e)
        }


async def get_stream_forecast(
    symbol: str,
    exchange: str,
    market_type: str = "futures"
) -> Dict[str, Any]:
    """
    Get latest forecast for a specific stream.
    
    Args:
        symbol: Trading pair (e.g., "BTCUSDT")
        exchange: Exchange name (e.g., "binance")
        market_type: Market type (default: "futures")
    
    Returns:
        Dict with forecast data or error
        
    Example:
        >>> forecast = await get_stream_forecast("BTCUSDT", "binance")
        >>> print(f"Predicted change: {forecast['predicted_change_pct']}%")
    """
    try:
        from src.streaming.realtime_analytics import get_realtime_analytics
        
        analytics = get_realtime_analytics()
        return analytics.get_stream_status(symbol, exchange, market_type)
        
    except Exception as e:
        return {
            "status": "ERROR",
            "message": str(e)
        }


# Export all tools
__all__ = [
    'start_streaming',
    'stop_streaming',
    'get_streaming_status',
    'get_streaming_health',
    'get_streaming_alerts',
    'configure_streaming',
    'get_realtime_analytics_status',
    'get_stream_forecast'
]
