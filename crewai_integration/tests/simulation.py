"""
Simulation and Shadow Mode Runners
==================================

Provides testing modes where agents can:
- Operate on mock/historical data
- Make decisions without executing actions
- Run alongside the real system in shadow mode
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for simulation runs."""
    mode: str = "dry_run"  # "dry_run", "mock_data", "replay"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    speed_multiplier: float = 1.0
    record_decisions: bool = True
    compare_with_actual: bool = False


@dataclass
class SimulationResult:
    """Result of a simulation run."""
    config: SimulationConfig
    started_at: datetime
    completed_at: datetime
    decisions_made: int
    would_be_actions: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    metrics: Dict[str, Any]


class SimulationRunner:
    """
    Runs CrewAI agents in simulation mode.
    
    Simulation Modes:
    - dry_run: Agents make decisions but don't execute
    - mock_data: Agents operate on synthetic data
    - replay: Agents process historical data
    """
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        self.decisions: List[Dict[str, Any]] = []
        self.recommendations: List[Dict[str, Any]] = []
        self._running = False
    
    async def run(
        self,
        duration_minutes: int = 5,
        agents: Optional[List[str]] = None
    ) -> SimulationResult:
        """
        Run a simulation.
        
        Args:
            duration_minutes: How long to run
            agents: Which agents to include (None for all)
            
        Returns:
            SimulationResult with outcomes
        """
        logger.info(f"Starting simulation in {self.config.mode} mode for {duration_minutes} minutes")
        
        started_at = datetime.utcnow()
        self._running = True
        self.decisions.clear()
        self.recommendations.clear()
        
        try:
            if self.config.mode == "dry_run":
                await self._run_dry_mode(duration_minutes, agents)
            elif self.config.mode == "mock_data":
                await self._run_mock_mode(duration_minutes, agents)
            elif self.config.mode == "replay":
                await self._run_replay_mode(agents)
        except Exception as e:
            logger.error(f"Simulation error: {e}")
        finally:
            self._running = False
        
        completed_at = datetime.utcnow()
        
        return SimulationResult(
            config=self.config,
            started_at=started_at,
            completed_at=completed_at,
            decisions_made=len(self.decisions),
            would_be_actions=[d for d in self.decisions if d.get("is_action")],
            recommendations=self.recommendations,
            metrics=self._calculate_metrics()
        )
    
    async def _run_dry_mode(self, duration_minutes: int, agents: Optional[List[str]]):
        """Run in dry mode - real analysis, no execution."""
        end_time = datetime.utcnow() + timedelta(minutes=duration_minutes)
        
        while self._running and datetime.utcnow() < end_time:
            # Simulate agent decision cycle
            decision = {
                "timestamp": datetime.utcnow().isoformat(),
                "agent": "simulated_agent",
                "tool": "analyze_crypto_arbitrage",
                "would_execute": True,
                "is_action": False,
                "reasoning": "Simulation mode - no actual execution"
            }
            self.decisions.append(decision)
            
            await asyncio.sleep(10)  # Check every 10 seconds
    
    async def _run_mock_mode(self, duration_minutes: int, agents: Optional[List[str]]):
        """Run with mock data."""
        # Generate mock data for testing
        mock_prices = {
            "BTCUSDT": {"binance": 50000, "bybit": 50010, "okx": 49990},
            "ETHUSDT": {"binance": 3000, "bybit": 3005, "okx": 2995}
        }
        
        for i in range(duration_minutes):
            if not self._running:
                break
            
            # Simulate price changes
            for symbol in mock_prices:
                for exchange in mock_prices[symbol]:
                    mock_prices[symbol][exchange] *= (1 + (0.001 * (i % 5 - 2)))
            
            decision = {
                "timestamp": datetime.utcnow().isoformat(),
                "agent": "mock_agent",
                "input_data": dict(mock_prices),
                "analysis": "Mock analysis completed",
                "is_action": False
            }
            self.decisions.append(decision)
            
            await asyncio.sleep(60 / max(self.config.speed_multiplier, 0.1))
    
    async def _run_replay_mode(self, agents: Optional[List[str]]):
        """Replay historical data."""
        # Would load from historical data
        logger.info("Replay mode: Would process historical data")
        self.decisions.append({
            "timestamp": datetime.utcnow().isoformat(),
            "mode": "replay",
            "message": "Replay mode simulation placeholder"
        })
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """Calculate simulation metrics."""
        return {
            "total_decisions": len(self.decisions),
            "total_recommendations": len(self.recommendations),
            "decision_rate_per_minute": len(self.decisions) / max(1, 5),  # placeholder
        }
    
    def stop(self):
        """Stop the simulation."""
        self._running = False


class ShadowModeRunner:
    """
    Runs agents in shadow mode alongside the real system.
    
    In shadow mode:
    - Agents receive real data
    - Agents make real decisions
    - Actions are logged but NOT executed
    - Results are compared with what the system actually did
    """
    
    def __init__(self):
        self.shadow_decisions: List[Dict[str, Any]] = []
        self.actual_outcomes: List[Dict[str, Any]] = []
        self.comparisons: List[Dict[str, Any]] = []
        self._running = False
    
    async def start(self, duration_hours: int = 24):
        """
        Start shadow mode operation.
        
        Args:
            duration_hours: How long to run in shadow mode
        """
        logger.info(f"Starting shadow mode for {duration_hours} hours")
        self._running = True
        
        end_time = datetime.utcnow() + timedelta(hours=duration_hours)
        
        while self._running and datetime.utcnow() < end_time:
            try:
                # Process shadow decisions
                await self._process_shadow_cycle()
            except Exception as e:
                logger.error(f"Shadow mode error: {e}")
            
            await asyncio.sleep(60)  # Check every minute
        
        self._running = False
        logger.info("Shadow mode completed")
    
    async def _process_shadow_cycle(self):
        """Process one shadow mode cycle."""
        # Placeholder for shadow processing
        shadow_decision = {
            "timestamp": datetime.utcnow().isoformat(),
            "decision_type": "shadow",
            "would_recommend": True,
            "confidence": 0.75
        }
        self.shadow_decisions.append(shadow_decision)
    
    def stop(self):
        """Stop shadow mode."""
        self._running = False
    
    def get_comparison_report(self) -> Dict[str, Any]:
        """Generate comparison report between shadow and actual."""
        return {
            "shadow_decisions": len(self.shadow_decisions),
            "actual_outcomes": len(self.actual_outcomes),
            "comparisons": len(self.comparisons),
            "accuracy": self._calculate_accuracy(),
            "summary": "Shadow mode comparison report"
        }
    
    def _calculate_accuracy(self) -> float:
        """Calculate shadow mode accuracy."""
        if not self.comparisons:
            return 0.0
        
        correct = sum(1 for c in self.comparisons if c.get("correct", False))
        return correct / len(self.comparisons)


async def run_48_hour_shadow_test() -> Dict[str, Any]:
    """
    Run the Phase 1 48-hour shadow mode test.
    
    This is the validation test before proceeding to Phase 2.
    """
    logger.info("Starting 48-hour shadow mode validation test")
    
    runner = ShadowModeRunner()
    
    # Run for 48 hours (or shorter for testing)
    await runner.start(duration_hours=48)
    
    report = runner.get_comparison_report()
    
    # Phase 1 success criteria
    success = (
        len(runner.shadow_decisions) > 0 and
        runner._calculate_accuracy() >= 0.7  # 70% accuracy threshold
    )
    
    return {
        "test": "48_hour_shadow_mode",
        "success": success,
        "report": report,
        "timestamp": datetime.utcnow().isoformat()
    }
