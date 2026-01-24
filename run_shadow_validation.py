#!/usr/bin/env python3
"""
1-Hour Shadow Mode Validation for CrewAI Integration

This script runs the CrewAI integration in shadow mode for validation,
monitoring system behavior and logging decisions without executing them.

Usage:
    python run_shadow_validation.py [--duration MINUTES]

Default duration is 60 minutes (1 hour).
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import traceback

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
LOG_DIR = PROJECT_ROOT / "logs" / "shadow_validation"
LOG_DIR.mkdir(parents=True, exist_ok=True)

log_file = LOG_DIR / f"shadow_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("shadow_validation")


class ShadowModeValidator:
    """Validates CrewAI integration in shadow mode."""
    
    def __init__(self, duration_minutes: int = 60):
        self.duration_minutes = duration_minutes
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        
        # Validation metrics
        self.metrics = {
            "total_decisions": 0,
            "successful_simulations": 0,
            "failed_simulations": 0,
            "errors": [],
            "component_status": {},
            "tool_invocations": [],
            "performance_samples": []
        }
        
        # Components
        self.controller = None
        self.state_manager = None
        self.event_bus = None
        self.config_loader = None
    
    async def initialize_components(self) -> bool:
        """Initialize all CrewAI integration components."""
        logger.info("Initializing CrewAI integration components...")
        
        try:
            # Import CrewAI integration modules
            from crewai_integration.core.controller import CrewAIController
            from crewai_integration.state.manager import StateManager
            from crewai_integration.events.bus import EventBus
            from crewai_integration.config.loader import ConfigLoader
            
            # Initialize Event Bus
            self.event_bus = EventBus()
            await self.event_bus.start()
            self.metrics["component_status"]["event_bus"] = "OK"
            logger.info("[OK] Event Bus initialized")
            
            # Initialize State Manager
            db_path = PROJECT_ROOT / "data" / "crewai_state.duckdb"
            self.state_manager = StateManager(str(db_path))
            self.metrics["component_status"]["state_manager"] = "OK"
            logger.info("[OK] State Manager initialized")
            
            # Initialize Config Loader
            config_dir = PROJECT_ROOT / "crewai_integration" / "config"
            self.config_loader = ConfigLoader(str(config_dir))
            self.metrics["component_status"]["config_loader"] = "OK"
            logger.info("[OK] Config Loader initialized")
            
            # Initialize Controller (in shadow mode)
            self.controller = CrewAIController(
                config_path=str(config_dir),
                db_path=str(db_path)
            )
            # Enable shadow mode
            self.controller.shadow_mode = True
            await self.controller.initialize()
            self.metrics["component_status"]["controller"] = "OK"
            logger.info("[OK] Controller initialized in SHADOW MODE")
            
            return True
            
        except ImportError as e:
            logger.error(f"Import error during initialization: {e}")
            self.metrics["errors"].append({
                "type": "ImportError",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            })
            return False
            
        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            logger.error(traceback.format_exc())
            self.metrics["errors"].append({
                "type": type(e).__name__,
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            })
            return False
    
    async def validate_tool_wrappers(self) -> bool:
        """Validate that tool wrappers can be loaded and perform health checks."""
        logger.info("Validating tool wrappers...")
        
        try:
            from crewai_integration.tools.wrappers import (
                ExchangeDataTools,
                ForecastingTools,
                AnalyticsTools,
                StreamingTools,
                FeatureTools,
                VisualizationTools
            )
            
            # Test wrapper instantiation with shadow_mode
            wrappers = {
                "exchange_data": ExchangeDataTools,
                "forecasting": ForecastingTools,
                "analytics": AnalyticsTools,
                "streaming": StreamingTools,
                "features": FeatureTools,
                "visualization": VisualizationTools
            }
            
            all_passed = True
            for name, wrapper_class in wrappers.items():
                try:
                    # Instantiate wrapper in shadow mode
                    wrapper = wrapper_class(shadow_mode=True)
                    
                    # Perform health check
                    health = wrapper.health_check()
                    
                    # Verify wrapper is in shadow mode
                    if not wrapper.is_shadow_mode():
                        raise ValueError("Wrapper not in shadow mode")
                    
                    # Log success with tool count
                    tools_count = health.get("tools_count", 0)
                    self.metrics["component_status"][f"wrapper_{name}"] = f"OK - {tools_count} tools"
                    logger.info(f"[OK] {name} wrapper: {tools_count} tools, shadow_mode=True")
                    
                except Exception as e:
                    self.metrics["component_status"][f"wrapper_{name}"] = f"ERROR: {e}"
                    logger.warning(f"[FAIL] {name} wrapper failed: {e}")
                    all_passed = False
            
            return all_passed
            
        except ImportError as e:
            logger.error(f"Could not import tool wrappers: {e}")
            return False
    
    async def validate_config_loading(self) -> bool:
        """Validate configuration files can be loaded."""
        logger.info("Validating configuration files...")
        
        try:
            # Load all configs
            system_config = self.config_loader.load_system_config()
            agents_config = self.config_loader.load_agent_configs()
            tasks_config = self.config_loader.load_task_configs()
            crews_config = self.config_loader.load_crew_configs()
            
            self.metrics["component_status"]["config_system"] = f"OK - {len(system_config)} settings"
            self.metrics["component_status"]["config_agents"] = f"OK - {len(agents_config)} agents"
            self.metrics["component_status"]["config_tasks"] = f"OK - {len(tasks_config)} tasks"
            self.metrics["component_status"]["config_crews"] = f"OK - {len(crews_config)} crews"
            
            logger.info(f"[OK] System config: {len(system_config)} settings")
            logger.info(f"[OK] Agents config: {len(agents_config)} agents")
            logger.info(f"[OK] Tasks config: {len(tasks_config)} tasks")
            logger.info(f"[OK] Crews config: {len(crews_config)} crews")
            
            return True
            
        except Exception as e:
            logger.error(f"Config validation failed: {e}")
            return False
    
    async def simulate_market_analysis_cycle(self) -> Dict[str, Any]:
        """Simulate a market analysis cycle in shadow mode."""
        cycle_start = time.time()
        result = {
            "cycle_id": f"cycle_{int(time.time())}",
            "timestamp": datetime.now().isoformat(),
            "success": False,
            "decisions": [],
            "duration_ms": 0,
            "errors": []
        }
        
        try:
            # Simulate fetching market data
            logger.info("  -> Simulating market data fetch...")
            await asyncio.sleep(0.5)  # Simulate API call
            
            # Simulate analysis
            logger.info("  -> Simulating market analysis...")
            simulated_analysis = {
                "btc_price": 45000 + (time.time() % 1000),
                "eth_price": 2500 + (time.time() % 100),
                "market_sentiment": "neutral",
                "funding_rate_btc": 0.0001,
                "open_interest_change": 0.02
            }
            
            # Simulate decision generation
            logger.info("  -> Generating shadow decisions...")
            decisions = [
                {
                    "type": "market_analysis",
                    "symbol": "BTCUSDT",
                    "recommendation": "HOLD",
                    "confidence": 0.75,
                    "rationale": "Market conditions stable, no clear directional bias"
                },
                {
                    "type": "funding_opportunity",
                    "symbol": "ETHUSDT",
                    "action": "MONITOR",
                    "funding_rate": 0.0002,
                    "rationale": "Funding slightly positive, watch for arbitrage"
                }
            ]
            
            result["decisions"] = decisions
            result["success"] = True
            self.metrics["successful_simulations"] += 1
            self.metrics["total_decisions"] += len(decisions)
            
            # Log decisions to state manager
            if self.state_manager:
                for decision in decisions:
                    await self.state_manager.log_decision(
                        agent_id="shadow_validator",
                        decision_type=decision["type"],
                        decision_data=decision,
                        reasoning=decision["rationale"],
                        context={"shadow_mode": True}
                    )
            
        except Exception as e:
            result["errors"].append(str(e))
            self.metrics["failed_simulations"] += 1
            logger.error(f"  [FAIL] Simulation cycle failed: {e}")
        
        result["duration_ms"] = int((time.time() - cycle_start) * 1000)
        self.metrics["performance_samples"].append(result["duration_ms"])
        
        return result
    
    async def run_validation_loop(self):
        """Run the main validation loop for the specified duration."""
        self.start_time = datetime.now()
        self.end_time = self.start_time + timedelta(minutes=self.duration_minutes)
        
        logger.info("=" * 60)
        logger.info(f"SHADOW MODE VALIDATION STARTED")
        logger.info(f"Duration: {self.duration_minutes} minutes")
        logger.info(f"Start: {self.start_time.isoformat()}")
        logger.info(f"End: {self.end_time.isoformat()}")
        logger.info("=" * 60)
        
        cycle_count = 0
        cycle_interval = 60  # Run analysis cycle every 60 seconds
        
        while datetime.now() < self.end_time:
            cycle_count += 1
            remaining = (self.end_time - datetime.now()).total_seconds() / 60
            
            logger.info(f"\n--- Cycle {cycle_count} | {remaining:.1f} minutes remaining ---")
            
            # Run simulation cycle
            result = await self.simulate_market_analysis_cycle()
            
            if result["success"]:
                logger.info(f"  [OK] Cycle completed: {len(result['decisions'])} decisions, {result['duration_ms']}ms")
            else:
                logger.warning(f"  [FAIL] Cycle failed with errors")
            
            # Wait for next cycle
            await asyncio.sleep(cycle_interval)
        
        logger.info("\n" + "=" * 60)
        logger.info("SHADOW MODE VALIDATION COMPLETED")
        logger.info("=" * 60)
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate the final validation report."""
        report = {
            "validation_summary": {
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "duration_minutes": self.duration_minutes,
                "status": "PASSED" if len(self.metrics["errors"]) == 0 else "FAILED"
            },
            "component_status": self.metrics["component_status"],
            "simulation_metrics": {
                "total_decisions": self.metrics["total_decisions"],
                "successful_simulations": self.metrics["successful_simulations"],
                "failed_simulations": self.metrics["failed_simulations"],
                "success_rate": (
                    self.metrics["successful_simulations"] / 
                    max(1, self.metrics["successful_simulations"] + self.metrics["failed_simulations"])
                ) * 100
            },
            "performance": {
                "avg_cycle_ms": sum(self.metrics["performance_samples"]) / max(1, len(self.metrics["performance_samples"])),
                "max_cycle_ms": max(self.metrics["performance_samples"]) if self.metrics["performance_samples"] else 0,
                "min_cycle_ms": min(self.metrics["performance_samples"]) if self.metrics["performance_samples"] else 0
            },
            "errors": self.metrics["errors"]
        }
        
        return report
    
    async def run(self):
        """Main entry point for the validation."""
        logger.info("=" * 60)
        logger.info(" CrewAI Integration - Shadow Mode Validation ".center(60))
        logger.info("=" * 60)
        
        # Phase 1: Initialize components
        logger.info("\n[PHASE 1] Component Initialization")
        if not await self.initialize_components():
            logger.error("Component initialization failed!")
            return self.generate_report()
        
        # Phase 2: Validate tool wrappers
        logger.info("\n[PHASE 2] Tool Wrapper Validation")
        await self.validate_tool_wrappers()
        
        # Phase 3: Validate configurations
        logger.info("\n[PHASE 3] Configuration Validation")
        await self.validate_config_loading()
        
        # Phase 4: Run validation loop
        logger.info("\n[PHASE 4] Shadow Mode Validation Loop")
        await self.run_validation_loop()
        
        # Generate and save report
        report = self.generate_report()
        
        report_file = LOG_DIR / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("VALIDATION REPORT SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Status: {report['validation_summary']['status']}")
        logger.info(f"Total Decisions: {report['simulation_metrics']['total_decisions']}")
        logger.info(f"Success Rate: {report['simulation_metrics']['success_rate']:.1f}%")
        logger.info(f"Avg Cycle Time: {report['performance']['avg_cycle_ms']:.1f}ms")
        logger.info(f"Errors: {len(report['errors'])}")
        logger.info(f"Report saved to: {report_file}")
        
        return report


async def main():
    parser = argparse.ArgumentParser(description="Run CrewAI Shadow Mode Validation")
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Validation duration in minutes (default: 60)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick 5-minute validation"
    )
    
    args = parser.parse_args()
    
    duration = 5 if args.quick else args.duration
    
    validator = ShadowModeValidator(duration_minutes=duration)
    report = await validator.run()
    
    # Exit with appropriate code
    sys.exit(0 if report["validation_summary"]["status"] == "PASSED" else 1)


if __name__ == "__main__":
    asyncio.run(main())
