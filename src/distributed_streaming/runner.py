"""
Distributed Streaming CLI Runner
================================
Command-line interface to start the distributed streaming system.

Usage:
    # Run all workers (9 symbols x 4 exchanges = 36 workers)
    python -m src.distributed_streaming.runner
    
    # Run specific symbols and exchanges
    python -m src.distributed_streaming.runner --symbols BTCUSDT ETHUSDT --exchanges binance_futures bybit
    
    # Run for 10 minutes
    python -m src.distributed_streaming.runner --duration 10
    
    # Custom database path
    python -m src.distributed_streaming.runner --db-path data/my_stream.duckdb
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.distributed_streaming.config import (
    ExchangeType,
    DataType,
    DistributedConfig,
    DEFAULT_SYMBOLS,
    DEFAULT_EXCHANGES,
    DEFAULT_DATA_TYPES,
)
from src.distributed_streaming.coordinator import StreamCoordinator


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Use safe encoding for Windows console
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # Simple format without emojis for Windows compatibility
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers = [handler]
    
    # Quiet noisy loggers
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Distributed Streaming - Isolated workers per symbol-exchange pair",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.distributed_streaming.runner
  python -m src.distributed_streaming.runner --symbols BTCUSDT ETHUSDT
  python -m src.distributed_streaming.runner --exchanges binance_futures bybit
  python -m src.distributed_streaming.runner --duration 10
        """
    )
    
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=None,
        help=f"Symbols to track (default: {', '.join(DEFAULT_SYMBOLS[:4])}...)"
    )
    
    parser.add_argument(
        "--exchanges",
        nargs="+",
        choices=[e.value for e in ExchangeType],
        default=None,
        help=f"Exchanges to use (default: {', '.join(e.value for e in DEFAULT_EXCHANGES)})"
    )
    
    parser.add_argument(
        "--data-types",
        nargs="+",
        choices=[d.value for d in DataType],
        default=None,
        help=f"Data types to collect (default: {', '.join(d.value for d in DEFAULT_DATA_TYPES)})"
    )
    
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Duration in minutes (default: run forever)"
    )
    
    parser.add_argument(
        "--db-path",
        type=str,
        default="data/distributed_streaming.duckdb",
        help="Database file path (default: data/distributed_streaming.duckdb)"
    )
    
    parser.add_argument(
        "--health-interval",
        type=float,
        default=30.0,
        help="Health check interval in seconds (default: 30)"
    )
    
    parser.add_argument(
        "--stats-interval",
        type=float,
        default=60.0,
        help="Stats report interval in seconds (default: 60)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)
    
    logger = logging.getLogger(__name__)
    
    # Build config
    symbols = args.symbols if args.symbols else DEFAULT_SYMBOLS
    exchanges = [ExchangeType(e) for e in args.exchanges] if args.exchanges else DEFAULT_EXCHANGES
    data_types = [DataType(d) for d in args.data_types] if args.data_types else DEFAULT_DATA_TYPES
    
    config = DistributedConfig(
        symbols=symbols,
        exchanges=exchanges,
        data_types=data_types,
        db_path=args.db_path,
        health_check_interval=args.health_interval,
        stats_report_interval=args.stats_interval,
    )
    
    # Summary
    logger.info("=" * 60)
    logger.info("DISTRIBUTED STREAMING SYSTEM")
    logger.info("=" * 60)
    logger.info(f"Symbols: {len(symbols)} ({', '.join(symbols[:4])}{'...' if len(symbols) > 4 else ''})")
    logger.info(f"Exchanges: {len(exchanges)} ({', '.join(e.value for e in exchanges)})")
    logger.info(f"Data types: {len(data_types)} ({', '.join(d.value for d in data_types)})")
    logger.info(f"Total workers: {config.total_workers}")
    logger.info(f"Database: {args.db_path}")
    logger.info(f"Duration: {args.duration} minutes" if args.duration else "Duration: Run until stopped")
    logger.info("=" * 60)
    
    # Create and run coordinator
    coordinator = StreamCoordinator(config)
    
    try:
        await coordinator.start(duration=args.duration)
    except asyncio.CancelledError:
        logger.info("Cancelled")
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await coordinator.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown complete.")
