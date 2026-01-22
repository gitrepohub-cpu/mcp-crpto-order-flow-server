#!/usr/bin/env python3
"""
🚀 Start Production Streaming
=============================

Easy-to-use script to start the production streaming system.

Usage:
    python start_streaming.py
    python start_streaming.py --symbols BTCUSDT ETHUSDT --exchanges binance bybit
    python start_streaming.py --config config/my_config.json

Features:
- Real-time data collection from multiple exchanges
- Automatic forecasting pipeline
- Model drift detection and auto-retraining
- Health monitoring and alerting
"""

import asyncio
import argparse
import signal
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('streaming.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


async def main(args):
    """Main entry point"""
    from src.streaming.production_controller import ProductionStreamingController
    
    # Create controller
    controller = ProductionStreamingController(config_path=args.config)
    
    # Override config if CLI args provided
    if args.symbols:
        controller.config["symbols"] = args.symbols
    if args.exchanges:
        controller.config["exchanges"] = args.exchanges
    if args.market_type:
        controller.config["market_type"] = args.market_type
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logger.info(f"\n🛑 Received signal {sig}, initiating shutdown...")
        asyncio.create_task(controller.stop())
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Print startup banner
    print("\n" + "=" * 70)
    print("🚀 PRODUCTION STREAMING SYSTEM")
    print("=" * 70)
    print(f"📊 Symbols: {controller.config.get('symbols', [])}")
    print(f"🏦 Exchanges: {controller.config.get('exchanges', [])}")
    print(f"📈 Market Type: {controller.config.get('market_type', 'futures')}")
    print(f"⏱️  Forecast Interval: {controller.config.get('forecast_interval_seconds', 300)}s")
    print(f"🔍 Drift Check Interval: {controller.config.get('drift_check_interval_seconds', 600)}s")
    print("=" * 70)
    print("\nPress Ctrl+C to stop streaming...\n")
    
    try:
        # Start streaming
        await controller.start()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        await controller.stop()
        print("\n✅ Streaming stopped gracefully")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Start Production Streaming System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python start_streaming.py
  python start_streaming.py --symbols BTCUSDT ETHUSDT
  python start_streaming.py --exchanges binance bybit okx
  python start_streaming.py --config config/streaming_config.json
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        default="config/streaming_config.json",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--symbols", "-s",
        nargs="+",
        help="Symbols to stream (e.g., BTCUSDT ETHUSDT)"
    )
    
    parser.add_argument(
        "--exchanges", "-e",
        nargs="+",
        help="Exchanges to connect (e.g., binance bybit okx)"
    )
    
    parser.add_argument(
        "--market-type", "-m",
        choices=["futures", "spot"],
        help="Market type (futures or spot)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check Python version
    if sys.version_info < (3, 9):
        print("❌ Python 3.9+ required")
        sys.exit(1)
    
    # Run
    asyncio.run(main(args))
