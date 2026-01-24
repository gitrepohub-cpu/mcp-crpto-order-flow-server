"""
Phase 2: Data Operations Crew - Database Schema
================================================

This module defines the database tables required for the Data Operations Crew.
These tables are separate from the market data tables and store:
- Agent audit logs
- Data quality issues
- Interpolation records
- Schema health metrics
"""

import duckdb
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


# SQL statements for creating Phase 2 tables
CREATE_TABLES_SQL = """
-- =========================================================
-- AGENT AUDIT LOG TABLE
-- =========================================================
-- Stores all actions taken by Data Operations Crew agents
-- Used for audit trail, debugging, and performance analysis

CREATE SEQUENCE IF NOT EXISTS agent_audit_log_seq;
CREATE TABLE IF NOT EXISTS agent_audit_log (
    id INTEGER PRIMARY KEY DEFAULT nextval('agent_audit_log_seq'),
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    agent_id VARCHAR NOT NULL,
    agent_role VARCHAR NOT NULL,
    action_type VARCHAR NOT NULL,
    action_category VARCHAR NOT NULL,
    target_entity VARCHAR,
    target_id VARCHAR,
    action_details JSON,
    decision_rationale TEXT,
    outcome VARCHAR NOT NULL,
    error_message TEXT,
    duration_ms DOUBLE,
    context JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for common queries
CREATE INDEX IF NOT EXISTS idx_audit_agent ON agent_audit_log(agent_id);
CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON agent_audit_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_action ON agent_audit_log(action_type);
CREATE INDEX IF NOT EXISTS idx_audit_outcome ON agent_audit_log(outcome);


-- =========================================================
-- DATA QUALITY ISSUES TABLE
-- =========================================================
-- Stores detected data anomalies and quality issues
-- Written by Data Validator Agent

CREATE SEQUENCE IF NOT EXISTS data_quality_issues_seq;
CREATE TABLE IF NOT EXISTS data_quality_issues (
    id INTEGER PRIMARY KEY DEFAULT nextval('data_quality_issues_seq'),
    detected_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    exchange VARCHAR NOT NULL,
    symbol VARCHAR NOT NULL,
    data_type VARCHAR NOT NULL,
    issue_type VARCHAR NOT NULL,
    severity VARCHAR NOT NULL,
    problematic_value VARCHAR,
    expected_range VARCHAR,
    actual_value DOUBLE,
    deviation_percent DOUBLE,
    cross_validation_result JSON,
    validation_rule VARCHAR,
    action_taken VARCHAR NOT NULL,
    resolution_status VARCHAR NOT NULL DEFAULT 'pending',
    resolved_at TIMESTAMP,
    resolution_notes TEXT,
    related_record_ids JSON,
    context JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_quality_exchange ON data_quality_issues(exchange);
CREATE INDEX IF NOT EXISTS idx_quality_symbol ON data_quality_issues(symbol);
CREATE INDEX IF NOT EXISTS idx_quality_type ON data_quality_issues(issue_type);
CREATE INDEX IF NOT EXISTS idx_quality_severity ON data_quality_issues(severity);
CREATE INDEX IF NOT EXISTS idx_quality_status ON data_quality_issues(resolution_status);
CREATE INDEX IF NOT EXISTS idx_quality_detected ON data_quality_issues(detected_at);


-- =========================================================
-- INTERPOLATION LOG TABLE
-- =========================================================
-- Records all data interpolation actions
-- Written by Data Cleaner Agent

CREATE SEQUENCE IF NOT EXISTS interpolation_log_seq;
CREATE TABLE IF NOT EXISTS interpolation_log (
    id INTEGER PRIMARY KEY DEFAULT nextval('interpolation_log_seq'),
    interpolated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    exchange VARCHAR NOT NULL,
    symbol VARCHAR NOT NULL,
    data_type VARCHAR NOT NULL,
    gap_start TIMESTAMP NOT NULL,
    gap_end TIMESTAMP NOT NULL,
    gap_duration_seconds DOUBLE NOT NULL,
    strategy_used VARCHAR NOT NULL,
    records_created INTEGER NOT NULL,
    source_records_used INTEGER,
    confidence_score DOUBLE,
    interpolation_parameters JSON,
    validation_passed BOOLEAN DEFAULT FALSE,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_interp_exchange ON interpolation_log(exchange);
CREATE INDEX IF NOT EXISTS idx_interp_symbol ON interpolation_log(symbol);
CREATE INDEX IF NOT EXISTS idx_interp_timestamp ON interpolation_log(interpolated_at);
CREATE INDEX IF NOT EXISTS idx_interp_strategy ON interpolation_log(strategy_used);


-- =========================================================
-- SCHEMA HEALTH METRICS TABLE
-- =========================================================
-- Stores periodic health metrics for all monitored tables
-- Written by Schema Manager Agent

CREATE SEQUENCE IF NOT EXISTS schema_health_metrics_seq;
CREATE TABLE IF NOT EXISTS schema_health_metrics (
    id INTEGER PRIMARY KEY DEFAULT nextval('schema_health_metrics_seq'),
    measured_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    table_name VARCHAR NOT NULL,
    database_name VARCHAR DEFAULT 'main',
    row_count BIGINT,
    size_bytes BIGINT,
    size_mb DOUBLE,
    growth_rate_rows_per_day DOUBLE,
    growth_rate_mb_per_day DOUBLE,
    last_write_timestamp TIMESTAMP,
    avg_query_latency_ms DOUBLE,
    fragmentation_percent DOUBLE,
    health_status VARCHAR NOT NULL,
    recommendations JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_health_table ON schema_health_metrics(table_name);
CREATE INDEX IF NOT EXISTS idx_health_timestamp ON schema_health_metrics(measured_at);
CREATE INDEX IF NOT EXISTS idx_health_status ON schema_health_metrics(health_status);


-- =========================================================
-- CREW PERFORMANCE METRICS TABLE
-- =========================================================
-- Aggregated performance metrics for the Data Operations Crew

CREATE SEQUENCE IF NOT EXISTS crew_performance_metrics_seq;
CREATE TABLE IF NOT EXISTS crew_performance_metrics (
    id INTEGER PRIMARY KEY DEFAULT nextval('crew_performance_metrics_seq'),
    period_start TIMESTAMP NOT NULL,
    period_end TIMESTAMP NOT NULL,
    period_type VARCHAR NOT NULL,
    
    -- Collection metrics
    records_collected BIGINT,
    records_per_minute DOUBLE,
    collection_success_rate DOUBLE,
    
    -- Validation metrics
    records_validated BIGINT,
    validation_pass_rate DOUBLE,
    quarantine_rate DOUBLE,
    rejection_rate DOUBLE,
    
    -- Anomaly metrics
    anomalies_detected INTEGER,
    anomaly_rate DOUBLE,
    anomalies_resolved INTEGER,
    
    -- Interpolation metrics
    gaps_detected INTEGER,
    gaps_filled INTEGER,
    interpolated_records INTEGER,
    
    -- Connection metrics
    connection_uptime_percent DOUBLE,
    avg_latency_ms DOUBLE,
    disconnection_count INTEGER,
    
    -- Schema metrics
    tables_monitored INTEGER,
    tables_healthy INTEGER,
    optimizations_performed INTEGER,
    
    -- Agent metrics
    decisions_made INTEGER,
    tools_invoked INTEGER,
    avg_decision_time_ms DOUBLE,
    escalations INTEGER,
    
    -- Overall
    overall_quality_score DOUBLE,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_perf_period ON crew_performance_metrics(period_start, period_end);
CREATE INDEX IF NOT EXISTS idx_perf_type ON crew_performance_metrics(period_type);


-- =========================================================
-- ESCALATION LOG TABLE
-- =========================================================
-- Records all escalations to human operators

CREATE SEQUENCE IF NOT EXISTS escalation_log_seq;
CREATE TABLE IF NOT EXISTS escalation_log (
    id INTEGER PRIMARY KEY DEFAULT nextval('escalation_log_seq'),
    escalated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    escalation_level VARCHAR NOT NULL,
    trigger_condition VARCHAR NOT NULL,
    trigger_value VARCHAR,
    threshold_value VARCHAR,
    affected_exchanges JSON,
    affected_symbols JSON,
    description TEXT NOT NULL,
    recommended_actions JSON,
    notification_sent BOOLEAN DEFAULT FALSE,
    notification_channels JSON,
    acknowledged_at TIMESTAMP,
    acknowledged_by VARCHAR,
    resolved_at TIMESTAMP,
    resolution_notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_esc_level ON escalation_log(escalation_level);
CREATE INDEX IF NOT EXISTS idx_esc_timestamp ON escalation_log(escalated_at);
CREATE INDEX IF NOT EXISTS idx_esc_resolved ON escalation_log(resolved_at);


-- =========================================================
-- EXCHANGE CONNECTION STATUS TABLE
-- =========================================================
-- Real-time status of exchange connections

CREATE SEQUENCE IF NOT EXISTS exchange_connection_status_seq;
CREATE TABLE IF NOT EXISTS exchange_connection_status (
    id INTEGER PRIMARY KEY DEFAULT nextval('exchange_connection_status_seq'),
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    exchange VARCHAR NOT NULL,
    connection_type VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    latency_ms DOUBLE,
    last_data_received TIMESTAMP,
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    in_fallback_mode BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_conn_exchange ON exchange_connection_status(exchange);
CREATE INDEX IF NOT EXISTS idx_conn_status ON exchange_connection_status(status);
CREATE INDEX IF NOT EXISTS idx_conn_timestamp ON exchange_connection_status(timestamp);
"""


class DataOpsSchemaManager:
    """
    Manages the database schema for Phase 2 Data Operations Crew.
    
    Creates and maintains tables for:
    - Agent audit logging
    - Data quality issue tracking
    - Interpolation records
    - Schema health metrics
    - Crew performance metrics
    - Escalation tracking
    - Connection status
    """
    
    def __init__(self, db_path: str = "data/crewai_state.duckdb"):
        """
        Initialize the schema manager.
        
        Args:
            db_path: Path to the DuckDB database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[duckdb.DuckDBPyConnection] = None
    
    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = duckdb.connect(str(self.db_path))
        return self._conn
    
    def initialize_schema(self) -> bool:
        """
        Initialize all Phase 2 tables.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = self._get_connection()
            conn.execute(CREATE_TABLES_SQL)
            conn.commit()
            logger.info("Phase 2 Data Operations schema initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize schema: {e}")
            return False
    
    def verify_schema(self) -> Dict[str, bool]:
        """
        Verify that all required tables exist.
        
        Returns:
            Dict mapping table name to existence status
        """
        required_tables = [
            "agent_audit_log",
            "data_quality_issues",
            "interpolation_log",
            "schema_health_metrics",
            "crew_performance_metrics",
            "escalation_log",
            "exchange_connection_status"
        ]
        
        conn = self._get_connection()
        results = {}
        
        for table in required_tables:
            try:
                conn.execute(f"SELECT 1 FROM {table} LIMIT 1")
                results[table] = True
            except Exception:
                results[table] = False
        
        return results
    
    def get_table_stats(self) -> List[Dict[str, Any]]:
        """
        Get statistics for all Phase 2 tables.
        
        Returns:
            List of table statistics
        """
        conn = self._get_connection()
        stats = []
        
        tables = [
            "agent_audit_log",
            "data_quality_issues",
            "interpolation_log",
            "schema_health_metrics",
            "crew_performance_metrics",
            "escalation_log",
            "exchange_connection_status"
        ]
        
        for table in tables:
            try:
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                stats.append({
                    "table": table,
                    "row_count": count,
                    "status": "ok"
                })
            except Exception as e:
                stats.append({
                    "table": table,
                    "row_count": 0,
                    "status": f"error: {e}"
                })
        
        return stats
    
    def close(self):
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None


# Convenience functions for writing to Phase 2 tables

def log_agent_action(
    conn: duckdb.DuckDBPyConnection,
    agent_id: str,
    agent_role: str,
    action_type: str,
    action_category: str,
    outcome: str,
    target_entity: Optional[str] = None,
    target_id: Optional[str] = None,
    action_details: Optional[Dict] = None,
    decision_rationale: Optional[str] = None,
    error_message: Optional[str] = None,
    duration_ms: Optional[float] = None,
    context: Optional[Dict] = None
) -> int:
    """
    Log an agent action to the audit log.
    
    Returns:
        The ID of the inserted record
    """
    import json
    
    sql = """
    INSERT INTO agent_audit_log (
        timestamp, agent_id, agent_role, action_type, action_category,
        target_entity, target_id, action_details, decision_rationale,
        outcome, error_message, duration_ms, context
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    conn.execute(sql, [
        datetime.utcnow(),
        agent_id,
        agent_role,
        action_type,
        action_category,
        target_entity,
        target_id,
        json.dumps(action_details) if action_details else None,
        decision_rationale,
        outcome,
        error_message,
        duration_ms,
        json.dumps(context) if context else None
    ])
    
    # Get the last inserted ID
    result = conn.execute("SELECT currval('agent_audit_log_seq')").fetchone()
    return result[0]


def log_quality_issue(
    conn: duckdb.DuckDBPyConnection,
    exchange: str,
    symbol: str,
    data_type: str,
    issue_type: str,
    severity: str,
    action_taken: str,
    problematic_value: Optional[str] = None,
    expected_range: Optional[str] = None,
    actual_value: Optional[float] = None,
    deviation_percent: Optional[float] = None,
    cross_validation_result: Optional[Dict] = None,
    validation_rule: Optional[str] = None,
    context: Optional[Dict] = None
) -> int:
    """
    Log a data quality issue.
    
    Returns:
        The ID of the inserted record
    """
    import json
    
    sql = """
    INSERT INTO data_quality_issues (
        detected_at, exchange, symbol, data_type, issue_type, severity,
        problematic_value, expected_range, actual_value, deviation_percent,
        cross_validation_result, validation_rule, action_taken, context
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    conn.execute(sql, [
        datetime.utcnow(),
        exchange,
        symbol,
        data_type,
        issue_type,
        severity,
        problematic_value,
        expected_range,
        actual_value,
        deviation_percent,
        json.dumps(cross_validation_result) if cross_validation_result else None,
        validation_rule,
        action_taken,
        json.dumps(context) if context else None
    ])
    
    # Get the last inserted ID
    result = conn.execute("SELECT currval('data_quality_issues_seq')").fetchone()
    return result[0]


def log_interpolation(
    conn: duckdb.DuckDBPyConnection,
    exchange: str,
    symbol: str,
    data_type: str,
    gap_start: datetime,
    gap_end: datetime,
    strategy_used: str,
    records_created: int,
    source_records_used: Optional[int] = None,
    confidence_score: Optional[float] = None,
    interpolation_parameters: Optional[Dict] = None,
    notes: Optional[str] = None
) -> int:
    """
    Log an interpolation action.
    
    Returns:
        The ID of the inserted record
    """
    import json
    
    gap_duration = (gap_end - gap_start).total_seconds()
    
    sql = """
    INSERT INTO interpolation_log (
        interpolated_at, exchange, symbol, data_type, gap_start, gap_end,
        gap_duration_seconds, strategy_used, records_created, source_records_used,
        confidence_score, interpolation_parameters, notes
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    conn.execute(sql, [
        datetime.utcnow(),
        exchange,
        symbol,
        data_type,
        gap_start,
        gap_end,
        gap_duration,
        strategy_used,
        records_created,
        source_records_used,
        confidence_score,
        json.dumps(interpolation_parameters) if interpolation_parameters else None,
        notes
    ])
    
    # Get the last inserted ID
    result = conn.execute("SELECT currval('interpolation_log_seq')").fetchone()
    return result[0]


def log_escalation(
    conn: duckdb.DuckDBPyConnection,
    escalation_level: str,
    trigger_condition: str,
    description: str,
    trigger_value: Optional[str] = None,
    threshold_value: Optional[str] = None,
    affected_exchanges: Optional[List[str]] = None,
    affected_symbols: Optional[List[str]] = None,
    recommended_actions: Optional[List[str]] = None,
    notification_channels: Optional[List[str]] = None
) -> int:
    """
    Log an escalation event.
    
    Returns:
        The ID of the inserted record
    """
    import json
    
    sql = """
    INSERT INTO escalation_log (
        escalated_at, escalation_level, trigger_condition, trigger_value,
        threshold_value, affected_exchanges, affected_symbols, description,
        recommended_actions, notification_channels
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    conn.execute(sql, [
        datetime.utcnow(),
        escalation_level,
        trigger_condition,
        trigger_value,
        threshold_value,
        json.dumps(affected_exchanges) if affected_exchanges else None,
        json.dumps(affected_symbols) if affected_symbols else None,
        description,
        json.dumps(recommended_actions) if recommended_actions else None,
        json.dumps(notification_channels) if notification_channels else None
    ])
    
    # Get the last inserted ID
    result = conn.execute("SELECT currval('escalation_log_seq')").fetchone()
    return result[0]


if __name__ == "__main__":
    # Initialize schema when run directly
    logging.basicConfig(level=logging.INFO)
    
    manager = DataOpsSchemaManager()
    
    print("Initializing Phase 2 Data Operations schema...")
    if manager.initialize_schema():
        print("Schema initialized successfully!")
        
        print("\nVerifying tables...")
        verification = manager.verify_schema()
        for table, exists in verification.items():
            status = "[OK]" if exists else "[MISSING]"
            print(f"  {status} {table}")
        
        print("\nTable statistics:")
        for stat in manager.get_table_stats():
            print(f"  {stat['table']}: {stat['row_count']} rows ({stat['status']})")
    else:
        print("Failed to initialize schema!")
    
    manager.close()
