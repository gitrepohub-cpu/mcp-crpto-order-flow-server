"""
📊 Streamlit Dashboard for Crypto Order Flow Data
==================================================

This dashboard reads from DuckDB databases.
NOTE: On Windows, stop the collector before running this dashboard
      due to DuckDB file locking limitations.

Usage: 
  1. Stop the collector: Stop-Process -Name python -Force
  2. Run: streamlit run streamlit_viewer.py
  3. When done, restart collector: python all_in_one_collector.py
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json

# Check for duckdb
try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

# Check for plotly
try:
    import plotly.express as px
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Paths
RAW_DB_PATH = Path("data/isolated_exchange_data.duckdb")
FEATURE_DB_PATH = Path("data/features_data.duckdb")

# Page config
st.set_page_config(
    page_title="Crypto Order Flow Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 10px;
    }
    .big-font {
        font-size: 24px !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_raw_connection():
    """Get read-only connection to raw database."""
    if not RAW_DB_PATH.exists():
        return None
    try:
        return duckdb.connect(str(RAW_DB_PATH), read_only=True)
    except Exception as e:
        st.error(f"Cannot connect to raw database: {e}")
        return None


@st.cache_resource  
def get_feature_connection():
    """Get read-only connection to feature database."""
    if not FEATURE_DB_PATH.exists():
        return None
    try:
        return duckdb.connect(str(FEATURE_DB_PATH), read_only=True)
    except Exception as e:
        st.error(f"Cannot connect to feature database: {e}")
        return None


@st.cache_data(ttl=30)
def get_tables(conn_type: str) -> list:
    """Get list of tables."""
    conn = get_raw_connection() if conn_type == "raw" else get_feature_connection()
    if not conn:
        return []
    try:
        tables = conn.execute("SHOW TABLES").fetchall()
        return [t[0] for t in tables if not t[0].startswith('_')]
    except:
        return []


@st.cache_data(ttl=10)
def get_table_data(conn_type: str, table_name: str, limit: int = 1000) -> pd.DataFrame:
    """Get data from a table."""
    conn = get_raw_connection() if conn_type == "raw" else get_feature_connection()
    if not conn:
        return pd.DataFrame()
    try:
        df = conn.execute(f"""
            SELECT * FROM {table_name} 
            ORDER BY timestamp DESC 
            LIMIT {limit}
        """).fetchdf()
        return df
    except Exception as e:
        st.error(f"Error reading {table_name}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=10)
def get_database_stats(conn_type: str) -> dict:
    """Get database statistics."""
    conn = get_raw_connection() if conn_type == "raw" else get_feature_connection()
    if not conn:
        return {"tables": 0, "tables_with_data": 0, "total_rows": 0}
    
    try:
        tables = conn.execute("SHOW TABLES").fetchall()
        total_tables = len(tables)
        tables_with_data = 0
        total_rows = 0
        
        for (table_name,) in tables:
            if table_name.startswith('_'):
                continue
            try:
                result = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
                if result[0] > 0:
                    tables_with_data += 1
                    total_rows += result[0]
            except:
                pass
                
        return {
            "tables": total_tables,
            "tables_with_data": tables_with_data,
            "total_rows": total_rows
        }
    except:
        return {"tables": 0, "tables_with_data": 0, "total_rows": 0}


def main():
    st.title("📊 Crypto Order Flow Dashboard")
    
    # Check dependencies
    if not HAS_DUCKDB:
        st.error("❌ DuckDB not installed. Run: pip install duckdb")
        return
        
    # Check if databases exist
    if not RAW_DB_PATH.exists() and not FEATURE_DB_PATH.exists():
        st.error("❌ No databases found. Run the collector first.")
        return
        
    # Test connections
    raw_conn = get_raw_connection()
    feature_conn = get_feature_connection()
    
    if not raw_conn and not feature_conn:
        st.error("""
        ❌ Cannot connect to databases!
        
        On Windows, DuckDB doesn't allow multiple processes to access the same file.
        
        **To view data:**
        1. Stop the collector: `Stop-Process -Name python -Force`
        2. Refresh this page
        3. When done, restart: `python all_in_one_collector.py`
        """)
        return
    
    # Sidebar
    st.sidebar.header("⚙️ Settings")
    
    # Database selector
    db_options = []
    if raw_conn:
        db_options.append("Raw Data")
    if feature_conn:
        db_options.append("Features")
    
    selected_db = st.sidebar.radio("Database", db_options) if db_options else None
    
    if not selected_db:
        st.warning("No databases accessible")
        return
        
    conn_type = "raw" if selected_db == "Raw Data" else "feature"
    
    # Get stats
    stats = get_database_stats(conn_type)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Data Explorer", "🔍 Features"])
    
    # Tab 1: Overview
    with tab1:
        st.header("Database Overview")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Tables", stats["tables"])
        with col2:
            st.metric("Tables with Data", stats["tables_with_data"])
        with col3:
            st.metric("Total Rows", f"{stats['total_rows']:,}")
            
        # List tables with data
        st.subheader("📋 Tables with Data")
        
        tables = get_tables(conn_type)
        conn = get_raw_connection() if conn_type == "raw" else get_feature_connection()
        
        if conn and tables:
            table_info = []
            for table_name in tables:
                try:
                    result = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
                    if result[0] > 0:
                        # Get latest timestamp
                        latest = conn.execute(f"""
                            SELECT MAX(timestamp) FROM {table_name}
                        """).fetchone()
                        table_info.append({
                            "Table": table_name,
                            "Rows": result[0],
                            "Latest": latest[0] if latest and latest[0] else "N/A"
                        })
                except:
                    pass
                    
            if table_info:
                df_tables = pd.DataFrame(table_info)
                df_tables = df_tables.sort_values("Rows", ascending=False)
                st.dataframe(df_tables, use_container_width=True, height=400)
    
    # Tab 2: Data Explorer
    with tab2:
        st.header("Data Explorer")
        
        tables = get_tables(conn_type)
        if not tables:
            st.warning("No tables available")
        else:
            # Symbol filter
            symbols = list(set([t.split('_')[0].upper() for t in tables if '_' in t]))
            symbols.sort()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                selected_symbol = st.selectbox("Symbol", ["All"] + symbols)
                
            # Filter tables by symbol
            if selected_symbol != "All":
                filtered_tables = [t for t in tables if t.startswith(selected_symbol.lower())]
            else:
                filtered_tables = tables
                
            with col2:
                selected_table = st.selectbox("Table", filtered_tables) if filtered_tables else None
                
            with col3:
                row_limit = st.slider("Row Limit", 100, 5000, 1000)
                
            if selected_table:
                # Get data
                df = get_table_data(conn_type, selected_table, row_limit)
                
                if not df.empty:
                    st.subheader(f"📊 {selected_table}")
                    
                    # Show data
                    st.dataframe(df, use_container_width=True, height=400)
                    
                    # Time series chart
                    if 'timestamp' in df.columns and HAS_PLOTLY:
                        st.subheader("📈 Time Series")
                        
                        # Get numeric columns
                        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
                        if 'id' in numeric_cols:
                            numeric_cols.remove('id')
                            
                        if numeric_cols:
                            selected_col = st.selectbox("Select column to plot", numeric_cols)
                            
                            fig = px.line(
                                df.sort_values('timestamp'),
                                x='timestamp',
                                y=selected_col,
                                title=f"{selected_col} over time"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"No data in {selected_table}")
    
    # Tab 3: Features
    with tab3:
        st.header("🔍 Feature Analysis")
        
        if conn_type != "feature":
            st.info("Switch to 'Features' database to see feature data")
        else:
            # Get feature tables
            tables = get_tables("feature")
            feature_types = ["price_features", "trade_features", "flow_features"]
            
            col1, col2 = st.columns(2)
            
            with col1:
                symbols = list(set([t.split('_')[0].upper() for t in tables if '_' in t]))
                symbols.sort()
                selected_symbol = st.selectbox("Symbol", symbols, key="feat_symbol")
                
            with col2:
                selected_type = st.selectbox("Feature Type", feature_types)
                
            # Find matching table
            matching_tables = [t for t in tables 
                             if t.startswith(selected_symbol.lower()) 
                             and selected_type in t]
            
            if matching_tables:
                for table in matching_tables[:3]:  # Show up to 3 matching tables
                    df = get_table_data("feature", table, 100)
                    
                    if not df.empty:
                        st.subheader(f"📊 {table}")
                        
                        # Show latest values
                        if 'timestamp' in df.columns:
                            latest = df.iloc[0]
                            
                            # Display metrics based on feature type
                            if 'price_features' in table:
                                cols = st.columns(4)
                                if 'mid_price' in latest:
                                    cols[0].metric("Mid Price", f"${latest['mid_price']:,.2f}")
                                if 'spread_bps' in latest:
                                    cols[1].metric("Spread (bps)", f"{latest['spread_bps']:.2f}")
                                if 'depth_imbalance_5' in latest:
                                    cols[2].metric("Depth Imbalance", f"{latest['depth_imbalance_5']:.4f}")
                                if 'bid_depth_5' in latest:
                                    cols[3].metric("Bid Depth", f"{latest['bid_depth_5']:,.4f}")
                                    
                            elif 'trade_features' in table:
                                cols = st.columns(4)
                                if 'trade_count_1m' in latest:
                                    cols[0].metric("Trade Count (1m)", f"{latest['trade_count_1m']:,}")
                                if 'volume_1m' in latest:
                                    cols[1].metric("Volume (1m)", f"{latest['volume_1m']:,.4f}")
                                if 'cvd_1m' in latest:
                                    cols[2].metric("CVD (1m)", f"{latest['cvd_1m']:,.4f}")
                                if 'vwap_1m' in latest:
                                    cols[3].metric("VWAP (1m)", f"${latest['vwap_1m']:,.2f}")
                                    
                            elif 'flow_features' in table:
                                cols = st.columns(4)
                                if 'buy_sell_ratio' in latest:
                                    cols[0].metric("Buy/Sell Ratio", f"{latest['buy_sell_ratio']:.4f}")
                                if 'flow_imbalance' in latest:
                                    cols[1].metric("Flow Imbalance", f"{latest['flow_imbalance']:.4f}")
                                if 'taker_buy_ratio' in latest:
                                    cols[2].metric("Taker Buy %", f"{latest['taker_buy_ratio']*100:.1f}%")
                                if 'absorption_ratio' in latest:
                                    cols[3].metric("Absorption", f"{latest['absorption_ratio']:.4f}")
                                    
                        # Show data table
                        st.dataframe(df, use_container_width=True, height=200)
            else:
                st.warning(f"No {selected_type} tables found for {selected_symbol}")
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **📝 Notes:**
    - Stop collector before viewing
    - Data updates when collector runs
    - Refresh to see latest data
    """)


if __name__ == "__main__":
    main()
