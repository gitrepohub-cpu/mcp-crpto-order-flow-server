"""
Configuration Loader for CrewAI Integration
==========================================

Loads and validates YAML configuration files for agents, tasks, and system settings.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import json

logger = logging.getLogger(__name__)

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.warning("PyYAML not installed. Using JSON fallback for configs.")


class ConfigLoader:
    """
    Loads configuration files for CrewAI components.
    
    Supports YAML (preferred) and JSON formats.
    Provides validation and default value handling.
    """
    
    def __init__(self, config_dir: str = "crewai_integration/config"):
        """
        Initialize the config loader.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self._cache: Dict[str, Any] = {}
        self._loaded = False
    
    def load_all(self) -> Dict[str, Any]:
        """
        Load all configuration files.
        
        Returns:
            Dict containing all configurations
        """
        configs = {
            "system": self.load_system_config(),
            "agents": self.load_agent_configs(),
            "tasks": self.load_task_configs(),
            "crews": self.load_crew_configs(),
        }
        self._loaded = True
        return configs
    
    def load_system_config(self) -> Dict[str, Any]:
        """Load system configuration."""
        if "system" in self._cache:
            return self._cache["system"]
        
        config = self._load_file("system.yaml") or self._load_file("system.json")
        if not config:
            config = self._get_default_system_config()
        
        self._cache["system"] = config
        return config
    
    def load_agent_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load all agent configurations."""
        if "agents" in self._cache:
            return self._cache["agents"]
        
        agents = {}
        
        # Try loading from agents.yaml
        config = self._load_file("agents.yaml") or self._load_file("agents.json")
        if config and "agents" in config:
            # Handle both formats: list of dicts with 'id' or dict keyed by agent id
            agents_data = config["agents"]
            if isinstance(agents_data, dict):
                # Format: agents: {agent_id: {...}}
                for agent_id, agent_config in agents_data.items():
                    agent_config["id"] = agent_id
                    agents[agent_id] = agent_config
            elif isinstance(agents_data, list):
                # Format: agents: [{id: ..., ...}]
                for agent in agents_data:
                    agents[agent["id"]] = agent
        
        # Also check agents/ directory for individual files
        agents_dir = self.config_dir / "agents"
        if agents_dir.exists():
            for file in agents_dir.glob("*.yaml"):
                agent_config = self._load_file(f"agents/{file.name}")
                if agent_config:
                    agents[agent_config.get("id", file.stem)] = agent_config
            for file in agents_dir.glob("*.json"):
                agent_config = self._load_file(f"agents/{file.name}")
                if agent_config:
                    agents[agent_config.get("id", file.stem)] = agent_config
        
        # Use defaults if no configs found
        if not agents:
            agents = self._get_default_agent_configs()
        
        self._cache["agents"] = agents
        return agents
    
    def load_task_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load all task configurations."""
        if "tasks" in self._cache:
            return self._cache["tasks"]
        
        tasks = {}
        
        config = self._load_file("tasks.yaml") or self._load_file("tasks.json")
        if config and "tasks" in config:
            tasks_data = config["tasks"]
            if isinstance(tasks_data, dict):
                for task_id, task_config in tasks_data.items():
                    task_config["id"] = task_id
                    tasks[task_id] = task_config
            elif isinstance(tasks_data, list):
                for task in tasks_data:
                    tasks[task["id"]] = task
        
        if not tasks:
            tasks = self._get_default_task_configs()
        
        self._cache["tasks"] = tasks
        return tasks
    
    def load_crew_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load all crew configurations."""
        if "crews" in self._cache:
            return self._cache["crews"]
        
        crews = {}
        
        config = self._load_file("crews.yaml") or self._load_file("crews.json")
        if config and "crews" in config:
            crews_data = config["crews"]
            if isinstance(crews_data, dict):
                for crew_id, crew_config in crews_data.items():
                    crew_config["id"] = crew_id
                    crews[crew_id] = crew_config
            elif isinstance(crews_data, list):
                for crew in crews_data:
                    crews[crew["id"]] = crew
        
        if not crews:
            crews = self._get_default_crew_configs()
        
        self._cache["crews"] = crews
        return crews
    
    def _load_file(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load a configuration file."""
        filepath = self.config_dir / filename
        
        if not filepath.exists():
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                if filepath.suffix in ['.yaml', '.yml']:
                    if YAML_AVAILABLE:
                        return yaml.safe_load(f)
                    else:
                        logger.warning(f"Cannot load YAML file {filename}: PyYAML not installed")
                        return None
                elif filepath.suffix == '.json':
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config file {filename}: {e}")
            return None
        
        return None
    
    def _get_default_system_config(self) -> Dict[str, Any]:
        """Get default system configuration."""
        return {
            "version": "1.0.0",
            "phase": "foundation",
            "llm": {
                "provider": "openai",
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 4096
            },
            "rate_limits": {
                "binance": {"calls_per_second": 0.5, "burst": 3},
                "bybit": {"calls_per_second": 0.33, "burst": 2},
                "okx": {"calls_per_second": 0.5, "burst": 3},
                "hyperliquid": {"calls_per_second": 1.0, "burst": 5},
                "gateio": {"calls_per_second": 0.5, "burst": 3},
                "default": {"calls_per_second": 1.0, "burst": 5}
            },
            "logging": {
                "level": "INFO",
                "file": "logs/crewai.log",
                "max_size_mb": 100,
                "backup_count": 5
            },
            "alerts": {
                "enabled": True,
                "error_threshold": 5,
                "latency_threshold_ms": 5000
            },
            "human_approval": {
                "required_for": [
                    "streaming_control",
                    "system_config",
                    "database_write"
                ],
                "timeout_seconds": 300
            },
            "shadow_mode": {
                "enabled": True,
                "log_recommendations": True,
                "compare_with_actual": True
            }
        }
    
    def _get_default_agent_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get default agent configurations."""
        return {
            # Data Crew Agents
            "data_acquisition_agent": {
                "id": "data_acquisition_agent",
                "crew": "data_crew",
                "role": "Senior Data Acquisition Specialist",
                "goal": "Ensure continuous, high-quality data collection from all configured exchanges",
                "backstory": """You are an expert in cryptocurrency market data infrastructure with 
                deep knowledge of exchange APIs, rate limits, and data quality assurance. Your 
                primary responsibility is to monitor and maintain the data collection pipelines, 
                ensuring no gaps in coverage and alerting on anomalies.""",
                "tools": [
                    "binance_get_ticker", "bybit_futures_ticker", "okx_ticker",
                    "hyperliquid_ticker_tool", "gateio_futures_ticker_tool",
                    "get_streaming_status", "get_streaming_health"
                ],
                "guardrails": {
                    "max_api_calls_per_minute": 60,
                    "require_approval_for": ["start_streaming", "stop_streaming"]
                },
                "verbose": True,
                "allow_delegation": False
            },
            
            "data_quality_agent": {
                "id": "data_quality_agent",
                "crew": "data_crew",
                "role": "Data Quality Analyst",
                "goal": "Monitor data quality metrics and detect anomalies in collected data",
                "backstory": """You specialize in data quality assessment for financial time series.
                You understand common data issues like gaps, outliers, and inconsistencies.
                You proactively identify potential problems before they affect downstream analysis.""",
                "tools": [
                    "get_streaming_health", "detect_anomalies", "get_streaming_alerts"
                ],
                "guardrails": {
                    "alert_on_gaps": True,
                    "gap_threshold_seconds": 60
                },
                "verbose": True,
                "allow_delegation": True
            },
            
            # Analytics Crew Agents
            "forecasting_agent": {
                "id": "forecasting_agent",
                "crew": "analytics_crew",
                "role": "Quantitative Forecasting Specialist",
                "goal": "Generate accurate price and volatility forecasts using the best available models",
                "backstory": """You are an expert quantitative analyst specializing in time series
                forecasting. You understand the strengths and weaknesses of different forecasting
                approaches and can select the most appropriate model for each situation.""",
                "tools": [
                    "forecast_quick", "forecast_zero_shot", "route_forecast_request",
                    "ensemble_auto_select", "compare_all_models", "registry_recommend_model"
                ],
                "guardrails": {
                    "min_confidence_threshold": 0.6,
                    "max_horizon_hours": 168
                },
                "verbose": True,
                "allow_delegation": True
            },
            
            "regime_detection_agent": {
                "id": "regime_detection_agent",
                "crew": "analytics_crew",
                "role": "Market Regime Analyst",
                "goal": "Identify current market regimes and detect regime changes early",
                "backstory": """You specialize in identifying market states and regime transitions.
                You understand how different market conditions (trending, ranging, volatile)
                affect trading strategies and model performance.""",
                "tools": [
                    "detect_market_regime", "detect_anomalies", "detect_change_points",
                    "detect_trend", "compute_alpha_signals"
                ],
                "guardrails": {
                    "regime_change_threshold": 0.7
                },
                "verbose": True,
                "allow_delegation": True
            },
            
            # Intelligence Crew Agents
            "institutional_flow_agent": {
                "id": "institutional_flow_agent",
                "crew": "intelligence_crew",
                "role": "Institutional Flow Analyst",
                "goal": "Detect and analyze institutional trading activity and smart money flows",
                "backstory": """You are an expert in detecting institutional trading patterns.
                You understand how large players operate, including their accumulation and
                distribution tactics, and can identify their footprints in market data.""",
                "tools": [
                    "get_smart_money_flow", "get_aggregated_intelligence",
                    "get_institutional_pressure", "get_whale_detection",
                    "get_cvd_analysis", "get_orderbook_features"
                ],
                "guardrails": {
                    "min_signal_strength": 0.5
                },
                "verbose": True,
                "allow_delegation": True
            },
            
            "risk_assessment_agent": {
                "id": "risk_assessment_agent",
                "crew": "intelligence_crew",
                "role": "Risk Assessment Specialist",
                "goal": "Monitor and assess market risks including liquidation cascades and squeezes",
                "backstory": """You specialize in identifying potential market risks before they
                materialize. You understand leverage dynamics, funding rate pressures, and
                conditions that can trigger cascading liquidations.""",
                "tools": [
                    "get_short_squeeze_probability", "get_liquidation_cascade_risk",
                    "get_leverage_risk", "analyze_leverage_positioning",
                    "compute_funding_stress", "get_funding_features"
                ],
                "guardrails": {
                    "alert_threshold": 0.7
                },
                "verbose": True,
                "allow_delegation": True
            },
            
            # Operations Crew Agents
            "system_health_agent": {
                "id": "system_health_agent",
                "crew": "operations_crew",
                "role": "System Health Monitor",
                "goal": "Monitor system health and ensure all components are operating correctly",
                "backstory": """You are responsible for monitoring the health of the entire system.
                You track performance metrics, detect anomalies, and ensure all components
                are functioning within acceptable parameters.""",
                "tools": [
                    "get_streaming_status", "get_streaming_health", "get_streaming_alerts",
                    "arbitrage_scanner_health"
                ],
                "guardrails": {
                    "health_check_interval_seconds": 60,
                    "alert_on_degradation": True
                },
                "verbose": True,
                "allow_delegation": False
            },
            
            # Research Crew Agents
            "market_researcher_agent": {
                "id": "market_researcher_agent",
                "crew": "research_crew",
                "role": "Market Research Analyst",
                "goal": "Conduct comprehensive market research and generate actionable insights",
                "backstory": """You are a senior market researcher with expertise in cryptocurrency
                markets. You synthesize data from multiple sources to identify patterns,
                correlations, and opportunities that others might miss.""",
                "tools": [
                    "get_signal_dashboard", "get_correlation_matrix", "get_regime_visualization",
                    "query_historical_features", "get_feature_statistics",
                    "binance_full_analysis", "okx_full_analysis"
                ],
                "guardrails": {
                    "max_report_length": 5000
                },
                "verbose": True,
                "allow_delegation": True
            }
        }
    
    def _get_default_task_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get default task configurations."""
        return {
            "data_health_check": {
                "id": "data_health_check",
                "name": "Data Pipeline Health Check",
                "description": "Check the health of all data collection pipelines",
                "agent": "data_acquisition_agent",
                "expected_output": "Health status report with any issues flagged",
                "async_execution": True,
                "timeout_seconds": 60,
                "retry_count": 3
            },
            
            "data_quality_assessment": {
                "id": "data_quality_assessment",
                "name": "Data Quality Assessment",
                "description": "Assess the quality of recently collected data",
                "agent": "data_quality_agent",
                "expected_output": "Quality metrics and anomaly report",
                "async_execution": True,
                "timeout_seconds": 120,
                "dependencies": ["data_health_check"]
            },
            
            "price_forecast": {
                "id": "price_forecast",
                "name": "Price Forecast Generation",
                "description": "Generate price forecasts for configured symbols",
                "agent": "forecasting_agent",
                "expected_output": "Forecast with confidence intervals",
                "parameters": {
                    "symbols": ["BTCUSDT", "ETHUSDT"],
                    "horizon": 24
                },
                "async_execution": True,
                "timeout_seconds": 300
            },
            
            "regime_detection": {
                "id": "regime_detection",
                "name": "Market Regime Detection",
                "description": "Detect current market regime and any recent changes",
                "agent": "regime_detection_agent",
                "expected_output": "Current regime classification and change probability",
                "async_execution": True,
                "timeout_seconds": 120
            },
            
            "institutional_flow_analysis": {
                "id": "institutional_flow_analysis",
                "name": "Institutional Flow Analysis",
                "description": "Analyze recent institutional trading activity",
                "agent": "institutional_flow_agent",
                "expected_output": "Smart money flow summary and signals",
                "async_execution": True,
                "timeout_seconds": 180
            },
            
            "risk_assessment": {
                "id": "risk_assessment",
                "name": "Market Risk Assessment",
                "description": "Assess current market risk levels",
                "agent": "risk_assessment_agent",
                "expected_output": "Risk metrics and alerts",
                "async_execution": True,
                "timeout_seconds": 120
            },
            
            "system_health_check": {
                "id": "system_health_check",
                "name": "System Health Check",
                "description": "Check overall system health",
                "agent": "system_health_agent",
                "expected_output": "System health report",
                "async_execution": False,
                "timeout_seconds": 60,
                "priority": 1
            },
            
            "daily_market_report": {
                "id": "daily_market_report",
                "name": "Daily Market Report",
                "description": "Generate comprehensive daily market report",
                "agent": "market_researcher_agent",
                "expected_output": "Markdown formatted market report",
                "async_execution": True,
                "timeout_seconds": 600,
                "dependencies": [
                    "price_forecast",
                    "regime_detection",
                    "institutional_flow_analysis",
                    "risk_assessment"
                ]
            }
        }
    
    def _get_default_crew_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get default crew configurations."""
        return {
            "data_crew": {
                "id": "data_crew",
                "name": "Data Operations Crew",
                "description": "Manages data collection, quality, and pipeline health",
                "agents": ["data_acquisition_agent", "data_quality_agent"],
                "tasks": ["data_health_check", "data_quality_assessment"],
                "process": "sequential",
                "verbose": True,
                "memory": True
            },
            
            "analytics_crew": {
                "id": "analytics_crew",
                "name": "Analytics & Forecasting Crew",
                "description": "Generates forecasts and detects market regimes",
                "agents": ["forecasting_agent", "regime_detection_agent"],
                "tasks": ["price_forecast", "regime_detection"],
                "process": "parallel",
                "verbose": True,
                "memory": True
            },
            
            "intelligence_crew": {
                "id": "intelligence_crew",
                "name": "Institutional Intelligence Crew",
                "description": "Analyzes institutional flows and assesses risks",
                "agents": ["institutional_flow_agent", "risk_assessment_agent"],
                "tasks": ["institutional_flow_analysis", "risk_assessment"],
                "process": "parallel",
                "verbose": True,
                "memory": True
            },
            
            "operations_crew": {
                "id": "operations_crew",
                "name": "Operations & Monitoring Crew",
                "description": "Monitors system health and manages operations",
                "agents": ["system_health_agent"],
                "tasks": ["system_health_check"],
                "process": "sequential",
                "verbose": True,
                "memory": False
            },
            
            "research_crew": {
                "id": "research_crew",
                "name": "Research & Reporting Crew",
                "description": "Conducts research and generates reports",
                "agents": ["market_researcher_agent"],
                "tasks": ["daily_market_report"],
                "process": "sequential",
                "verbose": True,
                "memory": True
            }
        }
    
    def save_default_configs(self):
        """Save default configurations to files for customization."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        configs = {
            "system.yaml": self._get_default_system_config(),
            "agents.yaml": {"agents": list(self._get_default_agent_configs().values())},
            "tasks.yaml": {"tasks": list(self._get_default_task_configs().values())},
            "crews.yaml": {"crews": list(self._get_default_crew_configs().values())}
        }
        
        for filename, config in configs.items():
            filepath = self.config_dir / filename
            if not filepath.exists():
                with open(filepath, 'w', encoding='utf-8') as f:
                    if YAML_AVAILABLE and filename.endswith('.yaml'):
                        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                    else:
                        json.dump(config, f, indent=2)
                logger.info(f"Saved default config: {filepath}")
    
    def reload(self):
        """Clear cache and reload all configurations."""
        self._cache.clear()
        self._loaded = False
        return self.load_all()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key path (e.g., 'system.llm.model')."""
        if not self._loaded:
            self.load_all()
        
        parts = key.split('.')
        value = self._cache
        
        try:
            for part in parts:
                value = value[part]
            return value
        except (KeyError, TypeError):
            return default
