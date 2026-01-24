"""
State Manager for CrewAI Integration
====================================

Manages persistent state for CrewAI agents using a dedicated DuckDB database.
This database is SEPARATE from the main market data database to ensure:
- No interference with data collection
- Independent state management
- Clean separation of concerns
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import json

logger = logging.getLogger(__name__)

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False
    logger.warning("DuckDB not available, state management will be in-memory only")


class StateManager:
    """
    Manages all persistent state for CrewAI agents.
    
    Uses a dedicated DuckDB database (separate from market data) to store:
    - Agent decision history
    - Flow state and checkpoints
    - Shared knowledge base
    - Inter-agent communication logs
    
    Database Tables:
    ----------------
    - agent_decisions: Records of agent tool invocations and decisions
    - agent_memory: Short-term and long-term agent memory
    - flow_state: Checkpoints for multi-step flows
    - knowledge_base: Shared insights discovered by agents
    - communication_log: Inter-agent messages
    - metrics: Performance metrics for agents
    """
    
    # Schema definitions
    SCHEMAS = {
        "agent_decisions": """
            CREATE TABLE IF NOT EXISTS agent_decisions (
                id INTEGER PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                agent_id VARCHAR(100) NOT NULL,
                crew VARCHAR(50) NOT NULL,
                tool_name VARCHAR(200) NOT NULL,
                parameters JSON,
                result JSON,
                success BOOLEAN,
                latency_ms DOUBLE,
                reasoning TEXT,
                context JSON
            )
        """,
        
        "agent_memory": """
            CREATE TABLE IF NOT EXISTS agent_memory (
                id INTEGER PRIMARY KEY,
                agent_id VARCHAR(100) NOT NULL,
                memory_type VARCHAR(50) NOT NULL,  -- 'short_term', 'long_term', 'working'
                key VARCHAR(200) NOT NULL,
                value JSON NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                importance DOUBLE DEFAULT 0.5,
                access_count INTEGER DEFAULT 0,
                last_accessed TIMESTAMP
            )
        """,
        
        "flow_state": """
            CREATE TABLE IF NOT EXISTS flow_state (
                id INTEGER PRIMARY KEY,
                flow_id VARCHAR(100) NOT NULL,
                flow_name VARCHAR(200) NOT NULL,
                current_step VARCHAR(100),
                state JSON NOT NULL,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                status VARCHAR(50) DEFAULT 'running',
                checkpoint_data JSON
            )
        """,
        
        "knowledge_base": """
            CREATE TABLE IF NOT EXISTS knowledge_base (
                id INTEGER PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                discovered_by VARCHAR(100) NOT NULL,
                category VARCHAR(100) NOT NULL,
                title VARCHAR(500) NOT NULL,
                content TEXT NOT NULL,
                metadata JSON,
                confidence DOUBLE DEFAULT 0.5,
                verified BOOLEAN DEFAULT FALSE,
                verified_by VARCHAR(100),
                access_count INTEGER DEFAULT 0,
                usefulness_score DOUBLE DEFAULT 0.5
            )
        """,
        
        "communication_log": """
            CREATE TABLE IF NOT EXISTS communication_log (
                id INTEGER PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                from_agent VARCHAR(100) NOT NULL,
                to_agent VARCHAR(100),  -- NULL for broadcasts
                message_type VARCHAR(50) NOT NULL,
                content JSON NOT NULL,
                priority INTEGER DEFAULT 5,
                acknowledged BOOLEAN DEFAULT FALSE,
                acknowledged_at TIMESTAMP
            )
        """,
        
        "agent_metrics": """
            CREATE TABLE IF NOT EXISTS agent_metrics (
                id INTEGER PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                agent_id VARCHAR(100) NOT NULL,
                metric_name VARCHAR(100) NOT NULL,
                metric_value DOUBLE NOT NULL,
                dimensions JSON
            )
        """
    }
    
    def __init__(self, db_path: str = "data/crewai_state.duckdb"):
        """
        Initialize the state manager.
        
        Args:
            db_path: Path to the DuckDB database file
        """
        self.db_path = db_path
        self._conn = None
        self._initialized = False
        self._in_memory_state: Dict[str, Any] = {
            "decisions": [],
            "memory": {},
            "flows": {},
            "knowledge": [],
            "messages": []
        }
        self._lock = asyncio.Lock()
    
    async def initialize(self) -> bool:
        """
        Initialize the state database.
        
        Creates the database and all required tables if they don't exist.
        
        Returns:
            True if initialization successful
        """
        if self._initialized:
            return True
        
        try:
            if DUCKDB_AVAILABLE:
                # Ensure directory exists
                Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
                
                # Connect to database
                self._conn = duckdb.connect(self.db_path)
                
                # Create tables
                for table_name, schema in self.SCHEMAS.items():
                    self._conn.execute(schema)
                    logger.debug(f"Ensured table '{table_name}' exists")
                
                # Create indexes for performance
                self._create_indexes()
                
                logger.info(f"State database initialized at {self.db_path}")
            else:
                logger.warning("Using in-memory state management (DuckDB unavailable)")
            
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize state database: {e}")
            return False
    
    def _create_indexes(self):
        """Create indexes for better query performance."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_decisions_agent ON agent_decisions(agent_id, timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_decisions_tool ON agent_decisions(tool_name)",
            "CREATE INDEX IF NOT EXISTS idx_memory_agent ON agent_memory(agent_id, memory_type)",
            "CREATE INDEX IF NOT EXISTS idx_flow_id ON flow_state(flow_id)",
            "CREATE INDEX IF NOT EXISTS idx_knowledge_category ON knowledge_base(category)",
            "CREATE INDEX IF NOT EXISTS idx_comms_to ON communication_log(to_agent, acknowledged)",
            "CREATE INDEX IF NOT EXISTS idx_metrics_agent ON agent_metrics(agent_id, metric_name)",
        ]
        for idx in indexes:
            try:
                self._conn.execute(idx)
            except Exception as e:
                logger.debug(f"Index creation note: {e}")
    
    async def record_decision(
        self,
        agent_id: str,
        crew: str,
        tool_name: str,
        parameters: Dict[str, Any],
        result: Any,
        success: bool,
        latency_ms: float,
        reasoning: str = "",
        context: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Record an agent decision/tool invocation.
        
        Args:
            agent_id: ID of the agent making the decision
            crew: Crew the agent belongs to
            tool_name: Tool that was invoked
            parameters: Parameters passed to the tool
            result: Result from the tool
            success: Whether the invocation succeeded
            latency_ms: Time taken in milliseconds
            reasoning: Agent's reasoning for the decision
            context: Additional context
            
        Returns:
            ID of the recorded decision
        """
        async with self._lock:
            decision = {
                "timestamp": datetime.utcnow().isoformat(),
                "agent_id": agent_id,
                "crew": crew,
                "tool_name": tool_name,
                "parameters": parameters,
                "result": result,
                "success": success,
                "latency_ms": latency_ms,
                "reasoning": reasoning,
                "context": context or {}
            }
            
            if self._conn:
                # Get next ID manually for DuckDB
                max_id_result = self._conn.execute("SELECT COALESCE(MAX(id), 0) FROM agent_decisions").fetchone()
                next_id = max_id_result[0] + 1
                
                self._conn.execute("""
                    INSERT INTO agent_decisions 
                    (id, agent_id, crew, tool_name, parameters, result, success, latency_ms, reasoning, context)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [
                    next_id, agent_id, crew, tool_name,
                    json.dumps(parameters), json.dumps(result),
                    success, latency_ms, reasoning, json.dumps(context or {})
                ])
                result_id = next_id
            else:
                self._in_memory_state["decisions"].append(decision)
                result_id = len(self._in_memory_state["decisions"])
            
            return result_id
    
    async def get_agent_decisions(
        self,
        agent_id: str,
        limit: int = 100,
        tool_name: Optional[str] = None,
        success_only: bool = False
    ) -> List[Dict[str, Any]]:
        """Get decision history for an agent."""
        async with self._lock:
            if self._conn:
                query = "SELECT * FROM agent_decisions WHERE agent_id = ?"
                params = [agent_id]
                
                if tool_name:
                    query += " AND tool_name = ?"
                    params.append(tool_name)
                if success_only:
                    query += " AND success = TRUE"
                
                query += f" ORDER BY timestamp DESC LIMIT {limit}"
                
                result = self._conn.execute(query, params).fetchall()
                columns = [desc[0] for desc in self._conn.description]
                return [dict(zip(columns, row)) for row in result]
            else:
                decisions = self._in_memory_state["decisions"]
                filtered = [d for d in decisions if d["agent_id"] == agent_id]
                if tool_name:
                    filtered = [d for d in filtered if d["tool_name"] == tool_name]
                if success_only:
                    filtered = [d for d in filtered if d["success"]]
                return filtered[-limit:]
    
    async def set_memory(
        self,
        agent_id: str,
        key: str,
        value: Any,
        memory_type: str = "working",
        importance: float = 0.5,
        expires_in_hours: Optional[int] = None
    ):
        """
        Store a value in agent memory.
        
        Args:
            agent_id: Agent ID
            key: Memory key
            value: Value to store
            memory_type: 'short_term', 'long_term', or 'working'
            importance: Importance score (0-1)
            expires_in_hours: Optional expiration time
        """
        async with self._lock:
            expires_at = None
            if expires_in_hours:
                expires_at = datetime.utcnow() + timedelta(hours=expires_in_hours)
            
            if self._conn:
                # Check if entry exists
                existing = self._conn.execute("""
                    SELECT id FROM agent_memory 
                    WHERE agent_id = ? AND key = ?
                """, [agent_id, key]).fetchone()
                
                if existing:
                    # Update existing
                    self._conn.execute("""
                        UPDATE agent_memory 
                        SET memory_type = ?, value = ?, importance = ?, 
                            expires_at = ?, last_accessed = CURRENT_TIMESTAMP
                        WHERE agent_id = ? AND key = ?
                    """, [memory_type, json.dumps(value), importance, expires_at, agent_id, key])
                else:
                    # Get next ID and insert new
                    max_id_result = self._conn.execute("SELECT COALESCE(MAX(id), 0) FROM agent_memory").fetchone()
                    next_id = max_id_result[0] + 1
                    self._conn.execute("""
                        INSERT INTO agent_memory 
                        (id, agent_id, memory_type, key, value, importance, expires_at, last_accessed)
                        VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """, [next_id, agent_id, memory_type, key, json.dumps(value), importance, expires_at])
            else:
                if agent_id not in self._in_memory_state["memory"]:
                    self._in_memory_state["memory"][agent_id] = {}
                self._in_memory_state["memory"][agent_id][key] = {
                    "value": value,
                    "type": memory_type,
                    "importance": importance,
                    "expires_at": expires_at.isoformat() if expires_at else None
                }
    
    async def get_memory(
        self,
        agent_id: str,
        key: str,
        default: Any = None
    ) -> Any:
        """Retrieve a value from agent memory."""
        async with self._lock:
            if self._conn:
                result = self._conn.execute("""
                    SELECT value FROM agent_memory 
                    WHERE agent_id = ? AND key = ? 
                    AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
                """, [agent_id, key]).fetchone()
                
                if result:
                    # Update access tracking
                    self._conn.execute("""
                        UPDATE agent_memory 
                        SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP
                        WHERE agent_id = ? AND key = ?
                    """, [agent_id, key])
                    return json.loads(result[0])
            else:
                if agent_id in self._in_memory_state["memory"]:
                    entry = self._in_memory_state["memory"][agent_id].get(key)
                    if entry:
                        return entry["value"]
            
            return default
    
    async def save_flow_state(
        self,
        flow_id: str,
        flow_name: str,
        state: Dict[str, Any],
        current_step: str = "",
        checkpoint_data: Optional[Dict[str, Any]] = None
    ):
        """Save flow state for checkpointing."""
        async with self._lock:
            if self._conn:
                existing = self._conn.execute(
                    "SELECT id FROM flow_state WHERE flow_id = ?", [flow_id]
                ).fetchone()
                
                if existing:
                    self._conn.execute("""
                        UPDATE flow_state 
                        SET state = ?, current_step = ?, updated_at = CURRENT_TIMESTAMP, 
                            checkpoint_data = ?
                        WHERE flow_id = ?
                    """, [json.dumps(state), current_step, json.dumps(checkpoint_data or {}), flow_id])
                else:
                    self._conn.execute("""
                        INSERT INTO flow_state (flow_id, flow_name, state, current_step, checkpoint_data)
                        VALUES (?, ?, ?, ?, ?)
                    """, [flow_id, flow_name, json.dumps(state), current_step, json.dumps(checkpoint_data or {})])
            else:
                self._in_memory_state["flows"][flow_id] = {
                    "name": flow_name,
                    "state": state,
                    "current_step": current_step,
                    "checkpoint": checkpoint_data
                }
    
    async def get_flow_state(self, flow_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve flow state."""
        async with self._lock:
            if self._conn:
                result = self._conn.execute(
                    "SELECT * FROM flow_state WHERE flow_id = ?", [flow_id]
                ).fetchone()
                if result:
                    columns = [desc[0] for desc in self._conn.description]
                    return dict(zip(columns, result))
            else:
                return self._in_memory_state["flows"].get(flow_id)
            return None
    
    async def add_knowledge(
        self,
        discovered_by: str,
        category: str,
        title: str,
        content: str,
        confidence: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add an entry to the shared knowledge base.
        
        Args:
            discovered_by: Agent that discovered this knowledge
            category: Knowledge category (e.g., 'market_pattern', 'correlation')
            title: Short title
            content: Full content/description
            confidence: Confidence score (0-1)
            metadata: Additional metadata
            
        Returns:
            ID of the knowledge entry
        """
        async with self._lock:
            if self._conn:
                # Get next ID manually for DuckDB
                max_id_result = self._conn.execute("SELECT COALESCE(MAX(id), 0) FROM knowledge_base").fetchone()
                next_id = max_id_result[0] + 1
                
                self._conn.execute("""
                    INSERT INTO knowledge_base 
                    (id, discovered_by, category, title, content, confidence, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, [next_id, discovered_by, category, title, content, confidence, json.dumps(metadata or {})])
                return next_id
            else:
                entry = {
                    "discovered_by": discovered_by,
                    "category": category,
                    "title": title,
                    "content": content,
                    "confidence": confidence,
                    "metadata": metadata or {}
                }
                self._in_memory_state["knowledge"].append(entry)
                return len(self._in_memory_state["knowledge"])
    
    async def search_knowledge(
        self,
        category: Optional[str] = None,
        query: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Search the knowledge base."""
        async with self._lock:
            if self._conn:
                sql = "SELECT * FROM knowledge_base WHERE confidence >= ?"
                params = [min_confidence]
                
                if category:
                    sql += " AND category = ?"
                    params.append(category)
                if query:
                    sql += " AND (title LIKE ? OR content LIKE ?)"
                    params.extend([f"%{query}%", f"%{query}%"])
                
                sql += f" ORDER BY usefulness_score DESC, confidence DESC LIMIT {limit}"
                
                result = self._conn.execute(sql, params).fetchall()
                columns = [desc[0] for desc in self._conn.description]
                return [dict(zip(columns, row)) for row in result]
            else:
                entries = self._in_memory_state["knowledge"]
                filtered = [e for e in entries if e["confidence"] >= min_confidence]
                if category:
                    filtered = [e for e in filtered if e["category"] == category]
                if query:
                    filtered = [e for e in filtered if query.lower() in e["title"].lower() or query.lower() in e["content"].lower()]
                return filtered[:limit]
    
    async def send_message(
        self,
        from_agent: str,
        message_type: str,
        content: Dict[str, Any],
        to_agent: Optional[str] = None,
        priority: int = 5
    ) -> int:
        """
        Send an inter-agent message.
        
        Args:
            from_agent: Sending agent
            message_type: Type of message
            content: Message content
            to_agent: Target agent (None for broadcast)
            priority: Priority (1-10, higher = more urgent)
            
        Returns:
            Message ID
        """
        async with self._lock:
            if self._conn:
                self._conn.execute("""
                    INSERT INTO communication_log 
                    (from_agent, to_agent, message_type, content, priority)
                    VALUES (?, ?, ?, ?, ?)
                """, [from_agent, to_agent, message_type, json.dumps(content), priority])
                return self._conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            else:
                msg = {
                    "from": from_agent,
                    "to": to_agent,
                    "type": message_type,
                    "content": content,
                    "priority": priority,
                    "timestamp": datetime.utcnow().isoformat()
                }
                self._in_memory_state["messages"].append(msg)
                return len(self._in_memory_state["messages"])
    
    async def get_messages(
        self,
        agent_id: str,
        unacknowledged_only: bool = True,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get messages for an agent."""
        async with self._lock:
            if self._conn:
                sql = """
                    SELECT * FROM communication_log 
                    WHERE (to_agent = ? OR to_agent IS NULL)
                """
                params = [agent_id]
                
                if unacknowledged_only:
                    sql += " AND acknowledged = FALSE"
                
                sql += f" ORDER BY priority DESC, timestamp DESC LIMIT {limit}"
                
                result = self._conn.execute(sql, params).fetchall()
                columns = [desc[0] for desc in self._conn.description]
                return [dict(zip(columns, row)) for row in result]
            else:
                msgs = self._in_memory_state["messages"]
                filtered = [m for m in msgs if m["to"] == agent_id or m["to"] is None]
                return filtered[-limit:]
    
    async def record_metric(
        self,
        agent_id: str,
        metric_name: str,
        value: float,
        dimensions: Optional[Dict[str, Any]] = None
    ):
        """Record a performance metric."""
        async with self._lock:
            if self._conn:
                self._conn.execute("""
                    INSERT INTO agent_metrics (agent_id, metric_name, metric_value, dimensions)
                    VALUES (?, ?, ?, ?)
                """, [agent_id, metric_name, value, json.dumps(dimensions or {})])
    
    async def get_metrics(
        self,
        agent_id: str,
        metric_name: Optional[str] = None,
        hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Get metrics for an agent."""
        async with self._lock:
            if self._conn:
                sql = """
                    SELECT * FROM agent_metrics 
                    WHERE agent_id = ? AND timestamp > ?
                """
                params = [agent_id, datetime.utcnow() - timedelta(hours=hours)]
                
                if metric_name:
                    sql += " AND metric_name = ?"
                    params.append(metric_name)
                
                sql += " ORDER BY timestamp DESC"
                
                result = self._conn.execute(sql, params).fetchall()
                columns = [desc[0] for desc in self._conn.description]
                return [dict(zip(columns, row)) for row in result]
            return []
    
    async def flush(self):
        """Flush any pending writes."""
        # DuckDB auto-commits, but this is here for interface consistency
        pass
    
    async def close(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
        self._initialized = False
    
    async def log_decision(
        self,
        agent_id: str,
        decision_type: str,
        decision_data: Dict[str, Any],
        reasoning: str = "",
        context: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Log a decision (convenience method wrapping record_decision).
        
        Args:
            agent_id: ID of the agent making the decision
            decision_type: Type of decision (e.g., 'shadow_trade', 'analysis')
            decision_data: Decision details
            reasoning: Agent's reasoning
            context: Additional context
            
        Returns:
            ID of the recorded decision
        """
        return await self.record_decision(
            agent_id=agent_id,
            crew="shadow_mode",
            tool_name=decision_type,
            parameters=decision_data,
            result={"logged": True},
            success=True,
            latency_ms=0.0,
            reasoning=reasoning,
            context=context
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get state manager statistics."""
        if self._conn:
            stats = {}
            for table in self.SCHEMAS.keys():
                count = self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                stats[table] = count
            return {
                "database": self.db_path,
                "tables": stats,
                "initialized": self._initialized
            }
        else:
            return {
                "database": "in_memory",
                "decisions": len(self._in_memory_state["decisions"]),
                "memory_keys": sum(len(v) for v in self._in_memory_state["memory"].values()),
                "flows": len(self._in_memory_state["flows"]),
                "knowledge": len(self._in_memory_state["knowledge"]),
                "messages": len(self._in_memory_state["messages"])
            }
