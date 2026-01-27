"""
Crypto Data Dashboard - ULTRA FAST VERSION
============================================
Uses dedicated MCP tools with connection pooling and caching.
Loads in under 2 seconds.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
import duckdb
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

RAY_DATA_DIR = Path(__file__).parent / "data" / "ray_partitions"

EXCHANGES = [
    'binance_futures', 'binance_spot', 'bybit_linear', 'bybit_spot',
    'okx', 'gateio', 'hyperliquid', 'kucoin_spot', 'kucoin_futures', 'poller'
]

st.set_page_config(
    page_title="Crypto Dashboard",
    page_icon="📊",
    layout="wide"
)

# ============================================================================
# FAST MCP TOOLS - Embedded for zero import overhead
# ============================================================================

class ConnectionPool:
    """Singleton connection pool."""
    _conns = {}
    
    @classmethod
    def get(cls, exchange: str):
        if exchange not in cls._conns:
            db_path = RAY_DATA_DIR / f"{exchange}.duckdb"
            if db_path.exists():
                cls._conns[exchange] = duckdb.connect(str(db_path), read_only=True)
        return cls._conns.get(exchange)


@st.cache_data(ttl=120)
def mcp_get_stats_fast():
    """MCP Tool: Get FAST stats WITHOUT counting rows (instant load)."""
    stats = {
        'total_rows': 0, 'total_tables': 0,
        'exchanges': {}, 'symbols': set(),
        'streams': set(), 'markets': set()
    }
    
    for exc in EXCHANGES:
        db_path = RAY_DATA_DIR / f"{exc}.duckdb"
        if not db_path.exists():
            continue
            
        try:
            conn = duckdb.connect(str(db_path), read_only=True)
            tables_raw = conn.execute("SHOW TABLES").fetchall()
            
            tables = []
            symbols = set()
            streams = set()
            markets = set()
            
            for (tbl,) in tables_raw:
                parts = tbl.split('_')
                sym = parts[0].upper() if parts else 'UNK'
                mkt = parts[2] if len(parts) > 2 else 'unknown'
                strm = '_'.join(parts[3:]) if len(parts) > 3 else tbl
                
                symbols.add(sym)
                streams.add(strm)
                markets.add(mkt)
                
                tables.append({'table': tbl, 'symbol': sym, 'market': mkt, 'stream_type': strm})
            
            conn.close()
            
            stats['exchanges'][exc] = {
                'tables': len(tables),
                'symbols': sorted(symbols),
                'stream_types': sorted(streams),
                'markets': sorted(markets),
                'table_list': tables
            }
            stats['total_tables'] += len(tables)
            stats['symbols'].update(symbols)
            stats['streams'].update(streams)
            stats['markets'].update(markets)
            
        except:
            pass
    
    stats['symbols'] = sorted(stats['symbols'])
    stats['streams'] = sorted(stats['streams'])
    stats['markets'] = sorted(stats['markets'])
    
    # Get approximate size from file system (instant)
    for exc in stats['exchanges']:
        db_path = RAY_DATA_DIR / f"{exc}.duckdb"
        if db_path.exists():
            stats['exchanges'][exc]['size_mb'] = db_path.stat().st_size / 1e6
    
    return stats


@st.cache_data(ttl=30)
def mcp_get_tables_fast(exchange: str):
    """MCP Tool: Get tables for exchange WITHOUT row counts (fast)."""
    db_path = RAY_DATA_DIR / f"{exchange}.duckdb"
    if not db_path.exists():
        return []
    
    try:
        conn = duckdb.connect(str(db_path), read_only=True)
        tables = conn.execute("SHOW TABLES").fetchall()
        result = []
        for (tbl,) in tables:
            parts = tbl.split('_')
            result.append({
                'table': tbl,
                'symbol': parts[0].upper() if parts else 'UNK',
                'market': parts[2] if len(parts) > 2 else 'unk',
                'stream_type': '_'.join(parts[3:]) if len(parts) > 3 else tbl,
            })
        conn.close()
        return result
    except:
        return []


@st.cache_data(ttl=10)
def mcp_count_table_rows(exchange: str, table: str):
    """MCP Tool: Count rows in a specific table (only called when needed)."""
    db_path = RAY_DATA_DIR / f"{exchange}.duckdb"
    if not db_path.exists():
        return 0
    try:
        conn = duckdb.connect(str(db_path), read_only=True)
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        conn.close()
        return count
    except:
        return 0


@st.cache_data(ttl=10)
def mcp_get_data(exchange: str, table: str, limit: int = 100):
    """MCP Tool: Get table data."""
    db_path = RAY_DATA_DIR / f"{exchange}.duckdb"
    if not db_path.exists():
        return []
    
    try:
        conn = duckdb.connect(str(db_path), read_only=True)
        cols = [c[1] for c in conn.execute(f"PRAGMA table_info({table})").fetchall()]
        rows = conn.execute(f"SELECT * FROM {table} ORDER BY ts DESC LIMIT {limit}").fetchall()
        conn.close()
        return [dict(zip(cols, r)) for r in rows]
    except:
        return []


@st.cache_data(ttl=60)
def mcp_get_schema(exchange: str, table: str):
    """MCP Tool: Get table schema."""
    db_path = RAY_DATA_DIR / f"{exchange}.duckdb"
    if not db_path.exists():
        return []
    try:
        conn = duckdb.connect(str(db_path), read_only=True)
        cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
        conn.close()
        return [(c[1], c[2]) for c in cols]
    except:
        return []


def mcp_query(exchange: str, sql: str):
    """MCP Tool: Run SQL query."""
    db_path = RAY_DATA_DIR / f"{exchange}.duckdb"
    if not db_path.exists():
        return pd.DataFrame()
    try:
        conn = duckdb.connect(str(db_path), read_only=True)
        df = conn.execute(sql).df()
        conn.close()
        return df
    except Exception as e:
        st.error(f"Error: {e}")
        return pd.DataFrame()


def fmt(n):
    """Format number."""
    if n >= 1e6: return f"{n/1e6:.1f}M"
    if n >= 1e3: return f"{n/1e3:.1f}K"
    return str(n)


# ============================================================================
# MAIN DASHBOARD
# ============================================================================

def main():
    st.title("📊 Crypto Data Dashboard")
    st.caption("⚡ Ultra-Fast Loading • Using MCP Tools")
    
    # Load stats using FAST MCP tool (no row counting - instant!)
    with st.spinner("Loading dashboard..."):
        stats = mcp_get_stats_fast()
    
    if not stats['exchanges']:
        st.error("❌ No data! Run: `python ray_collector.py 5`")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.metric("Exchanges", len(stats['exchanges']))
        st.metric("Tables", stats['total_tables'])
        st.metric("Symbols", len(stats['symbols']))
        
        total_size = sum(
            stats['exchanges'][e].get('size_mb', 0)
            for e in stats['exchanges']
        )
        st.metric("Storage", f"{total_size:.1f} MB")
        
        st.info("💡 Fast mode: Row counts on-demand")
        
        if st.button("🔄 Refresh"):
            st.cache_data.clear()
            st.rerun()
    
    # Tabs
    t1, t2, t3, t4 = st.tabs(["📊 Overview", "🔍 Explorer", "📋 Tables", "🔬 Query"])
    
    # TAB 1: Overview
    with t1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🏦 Exchanges", len(stats['exchanges']))
        c2.metric("📊 Tables", stats['total_tables'])
        c3.metric("🪙 Symbols", len(stats['symbols']))
        c4.metric("📡 Streams", len(stats['streams']))
        
        st.divider()
        
        # Exchange table (no charts to avoid infinite extent warnings)
        st.subheader("🏦 Exchange Summary")
        exc_data = []
        for exc, info in sorted(stats['exchanges'].items()):
            exc_data.append({
                'Exchange': exc,
                'Tables': info['tables'],
                'Symbols': len(info['symbols']),
                'Streams': len(info['stream_types']),
                'Size (MB)': f"{info.get('size_mb', 0):.1f}"
            })
        
        if exc_data:
            st.dataframe(exc_data, use_container_width=True, hide_index=True)
        
        # Symbol & Stream lists
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🪙 Symbols")
            st.write(", ".join(stats['symbols']))
        
        with col2:
            st.subheader("📡 Streams")
            st.write(", ".join(stats['streams']))
        
        # Exchange cards
        st.subheader("🏦 Exchange Details")
        cols = st.columns(3)
        for i, (exc, info) in enumerate(sorted(stats['exchanges'].items())):
            with cols[i % 3]:
                st.markdown(f"""
                **{exc}**  
                Tables: {info['tables']}  
                Symbols: {', '.join(info['symbols'][:4])}  
                Streams: {', '.join(info['stream_types'][:3])}
                """)
    
    # TAB 2: Explorer
    with t2:
        c1, c2, c3 = st.columns(3)
        
        with c1:
            exc = st.selectbox("🏦 Exchange", sorted(stats['exchanges'].keys()))
        
        exc_info = stats['exchanges'].get(exc, {})
        
        with c2:
            sym = st.selectbox("🪙 Symbol", ['All'] + exc_info.get('symbols', []))
        
        with c3:
            strm = st.selectbox("📡 Stream", ['All'] + exc_info.get('stream_types', []))
        
        # Get tables (fast - no row counts)
        tables = mcp_get_tables_fast(exc)
        
        if sym != 'All':
            tables = [t for t in tables if t['symbol'] == sym]
        if strm != 'All':
            tables = [t for t in tables if strm in t['stream_type']]
        
        st.info(f"**{len(tables)}** tables found")
        
        if tables:
            opts = [f"{t['table']}" for t in tables]
            idx = st.selectbox("📋 Table", range(len(opts)), format_func=lambda x: opts[x])
            
            tbl = tables[idx]['table']
            
            # Show row count on-demand
            with st.spinner("Counting rows..."):
                row_count = mcp_count_table_rows(exc, tbl)
            st.metric("Rows", f"{row_count:,}")
            
            c1, c2 = st.columns([1, 3])
            
            with c1:
                st.markdown("**Schema**")
                for col, typ in mcp_get_schema(exc, tbl):
                    st.text(f"• {col}: {typ}")
            
            with c2:
                limit = st.slider("Rows", 10, 500, 100)
                data = mcp_get_data(exc, tbl, limit)
                if data:
                    df = pd.DataFrame(data)
                    st.dataframe(df, use_container_width=True, height=400)
                    st.download_button("📥 CSV", df.to_csv(index=False), f"{tbl}.csv")
    
    # TAB 3: All Tables
    with t3:
        all_tables = []
        for exc_name in stats['exchanges']:
            for t in stats['exchanges'][exc_name].get('table_list', []):
                all_tables.append({
                    'Exchange': exc_name,
                    'Symbol': t['symbol'],
                    'Stream': t['stream_type'],
                    'Table': t['table']
                })
        
        if all_tables:
            df = pd.DataFrame(all_tables)
            
            c1, c2 = st.columns(2)
            with c1:
                f_exc = st.multiselect("Exchange", df['Exchange'].unique())
            with c2:
                f_sym = st.multiselect("Symbol", df['Symbol'].unique())
            
            if f_exc:
                df = df[df['Exchange'].isin(f_exc)]
            if f_sym:
                df = df[df['Symbol'].isin(f_sym)]
            
            search = st.text_input("🔍 Search")
            if search:
                df = df[df['Table'].str.contains(search, case=False)]
            
            st.metric("Tables", len(df))
            st.dataframe(df, use_container_width=True, height=500, hide_index=True)
    
    # TAB 4: Query
    with t4:
        c1, c2 = st.columns([1, 2])
        
        with c1:
            qexc = st.selectbox("Database", sorted(stats['exchanges'].keys()), key="qexc")
            st.markdown("**Tables:**")
            for t in mcp_get_tables_fast(qexc)[:8]:
                st.text(f"• {t['table']}")
        
        with c2:
            tables = mcp_get_tables_fast(qexc)
            default = f"SELECT * FROM {tables[0]['table'] if tables else 'table'}\nORDER BY ts DESC LIMIT 100"
            
            sql = st.text_area("SQL", default, height=120)
            
            if st.button("▶️ Run", type="primary"):
                result = mcp_query(qexc, sql)
                if not result.empty:
                    st.success(f"✅ {len(result)} rows")
                    st.dataframe(result, use_container_width=True, height=400)


if __name__ == "__main__":
    main()
