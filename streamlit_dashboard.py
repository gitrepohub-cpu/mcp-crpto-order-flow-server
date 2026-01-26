"""
📊 Crypto Order Flow Dashboard
==============================

Streamlit dashboard to visualize:
- Raw streaming data (503 tables)
- Computed features (493 tables)
- Real-time updates

Run with: streamlit run streamlit_dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time

try:
    import duckdb
except ImportError:
    st.error("DuckDB not installed. Run: pip install duckdb")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Crypto Order Flow Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Database paths
RAW_DB_PATH = Path("data/isolated_exchange_data.duckdb")
FEATURE_DB_PATH = Path("data/features_data.duckdb")

# Configuration
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ARUSDT", 
           "BRETTUSDT", "POPCATUSDT", "WIFUSDT", "PNUTUSDT"]
FUTURES_EXCHANGES = ["binance", "bybit", "okx", "kraken", "gateio", "hyperliquid"]
SPOT_EXCHANGES = ["binance", "bybit"]
RAW_STREAMS = ["prices", "orderbooks", "trades", "funding_rates", "open_interest", 
               "mark_prices", "liquidations", "candles_1m", "ticker_24h"]
FEATURE_TYPES = ["price_features", "trade_features", "flow_features", 
                 "funding_features", "oi_features", "volatility_features", 
                 "momentum_features", "signals"]


@st.cache_resource
def get_db_connection(db_path: Path, read_only: bool = True):
    """Get database connection (cached)."""
    try:
        return duckdb.connect(str(db_path), read_only=read_only)
    except Exception as e:
        st.error(f"Failed to connect to {db_path}: {e}")
        return None


def get_table_stats(conn, table_name: str) -> dict:
    """Get stats for a table."""
    try:
        # Row count
        count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        
        # Latest timestamp
        try:
            latest = conn.execute(f"""
                SELECT MAX(timestamp) FROM {table_name}
            """).fetchone()[0]
        except:
            latest = None
            
        # Oldest timestamp
        try:
            oldest = conn.execute(f"""
                SELECT MIN(timestamp) FROM {table_name}
            """).fetchone()[0]
        except:
            oldest = None
            
        return {
            'count': count,
            'latest': latest,
            'oldest': oldest
        }
    except Exception as e:
        return {'count': 0, 'latest': None, 'oldest': None, 'error': str(e)}


def get_table_list(conn) -> list:
    """Get list of all tables."""
    try:
        tables = conn.execute("SHOW TABLES").fetchall()
        return [t[0] for t in tables]
    except:
        return []


def get_recent_data(conn, table_name: str, limit: int = 100) -> pd.DataFrame:
    """Get recent data from a table."""
    try:
        df = conn.execute(f"""
            SELECT * FROM {table_name}
            ORDER BY timestamp DESC
            LIMIT {limit}
        """).fetchdf()
        return df
    except Exception as e:
        st.warning(f"Could not fetch data from {table_name}: {e}")
        return pd.DataFrame()


def get_price_timeseries(conn, symbol: str, exchange: str, market_type: str, 
                         minutes: int = 10) -> pd.DataFrame:
    """Get price time series data."""
    table_name = f"{symbol.lower()}_{exchange}_{market_type}_prices"
    try:
        df = conn.execute(f"""
            SELECT timestamp, price, bid, ask
            FROM {table_name}
            WHERE timestamp >= NOW() - INTERVAL '{minutes} minutes'
            ORDER BY timestamp ASC
        """).fetchdf()
        return df
    except:
        return pd.DataFrame()


def get_trade_timeseries(conn, symbol: str, exchange: str, market_type: str,
                         minutes: int = 10) -> pd.DataFrame:
    """Get trade time series data."""
    table_name = f"{symbol.lower()}_{exchange}_{market_type}_trades"
    try:
        df = conn.execute(f"""
            SELECT timestamp, price, quantity, side
            FROM {table_name}
            WHERE timestamp >= NOW() - INTERVAL '{minutes} minutes'
            ORDER BY timestamp ASC
        """).fetchdf()
        return df
    except:
        return pd.DataFrame()


def render_sidebar():
    """Render the sidebar with filters."""
    st.sidebar.title("📊 Dashboard Controls")
    
    # Database status
    st.sidebar.subheader("Database Status")
    
    raw_conn = get_db_connection(RAW_DB_PATH)
    feature_conn = get_db_connection(FEATURE_DB_PATH)
    
    if raw_conn:
        raw_tables = get_table_list(raw_conn)
        st.sidebar.success(f"✅ Raw DB: {len(raw_tables)} tables")
    else:
        st.sidebar.error("❌ Raw DB: Not connected")
        raw_tables = []
        
    if feature_conn:
        feature_tables = get_table_list(feature_conn)
        st.sidebar.success(f"✅ Feature DB: {len(feature_tables)} tables")
    else:
        st.sidebar.error("❌ Feature DB: Not connected")
        feature_tables = []
        
    st.sidebar.divider()
    
    # Filters
    st.sidebar.subheader("Filters")
    
    selected_symbol = st.sidebar.selectbox(
        "Symbol",
        options=SYMBOLS,
        index=0
    )
    
    market_type = st.sidebar.radio(
        "Market Type",
        options=["futures", "spot"],
        horizontal=True
    )
    
    if market_type == "futures":
        exchanges = FUTURES_EXCHANGES
    else:
        exchanges = SPOT_EXCHANGES
        
    selected_exchange = st.sidebar.selectbox(
        "Exchange",
        options=exchanges,
        index=0
    )
    
    time_range = st.sidebar.slider(
        "Time Range (minutes)",
        min_value=1,
        max_value=60,
        value=10
    )
    
    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh (5s)", value=False)
    
    return {
        'symbol': selected_symbol,
        'exchange': selected_exchange,
        'market_type': market_type,
        'time_range': time_range,
        'auto_refresh': auto_refresh,
        'raw_conn': raw_conn,
        'feature_conn': feature_conn,
        'raw_tables': raw_tables,
        'feature_tables': feature_tables
    }


def render_overview_tab(config: dict):
    """Render the overview tab."""
    st.header("📈 Data Collection Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Raw Tables", len(config['raw_tables']))
    with col2:
        st.metric("Feature Tables", len(config['feature_tables']))
    with col3:
        st.metric("Symbols", len(SYMBOLS))
    with col4:
        st.metric("Exchanges", len(FUTURES_EXCHANGES) + len(SPOT_EXCHANGES))
        
    st.divider()
    
    # Tables with data
    st.subheader("📊 Tables with Data (Recent Activity)")
    
    if config['raw_conn']:
        tables_with_data = []
        tables_empty = []
        
        # Check a sample of tables
        sample_tables = [
            f"{config['symbol'].lower()}_{config['exchange']}_{config['market_type']}_prices",
            f"{config['symbol'].lower()}_{config['exchange']}_{config['market_type']}_trades",
            f"{config['symbol'].lower()}_{config['exchange']}_{config['market_type']}_orderbooks",
        ]
        
        for table_name in sample_tables:
            if table_name in config['raw_tables']:
                stats = get_table_stats(config['raw_conn'], table_name)
                if stats['count'] > 0:
                    tables_with_data.append({
                        'Table': table_name,
                        'Rows': stats['count'],
                        'Latest': stats['latest'],
                        'Oldest': stats['oldest']
                    })
                else:
                    tables_empty.append(table_name)
                    
        if tables_with_data:
            st.dataframe(pd.DataFrame(tables_with_data), use_container_width=True)
        else:
            st.warning("No data in sampled tables yet. Make sure the collector is running.")
            
    # Feature tables status
    st.subheader("🧮 Feature Tables Status")
    
    if config['feature_conn']:
        feature_sample = [
            f"{config['symbol'].lower()}_{config['exchange']}_{config['market_type']}_price_features",
            f"{config['symbol'].lower()}_{config['exchange']}_{config['market_type']}_trade_features",
            f"{config['symbol'].lower()}_{config['exchange']}_{config['market_type']}_flow_features",
        ]
        
        feature_stats = []
        for table_name in feature_sample:
            if table_name in config['feature_tables']:
                stats = get_table_stats(config['feature_conn'], table_name)
                feature_stats.append({
                    'Table': table_name,
                    'Rows': stats['count'],
                    'Latest': stats['latest']
                })
                
        if feature_stats:
            st.dataframe(pd.DataFrame(feature_stats), use_container_width=True)


def render_price_tab(config: dict):
    """Render the price analysis tab."""
    st.header(f"💰 Price Analysis - {config['symbol']} ({config['exchange']})")
    
    if not config['raw_conn']:
        st.error("Database not connected")
        return
        
    # Get price data
    df = get_price_timeseries(
        config['raw_conn'],
        config['symbol'],
        config['exchange'],
        config['market_type'],
        config['time_range']
    )
    
    if df.empty:
        st.warning(f"No price data available for {config['symbol']}/{config['exchange']}/{config['market_type']}")
        st.info("Make sure the data collector is running and has been collecting for a few minutes.")
        return
        
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_price = df['price'].iloc[-1] if not df.empty else 0
        st.metric("Current Price", f"${current_price:,.2f}")
        
    with col2:
        if len(df) > 1:
            price_change = ((df['price'].iloc[-1] / df['price'].iloc[0]) - 1) * 100
            st.metric("Change", f"{price_change:+.2f}%")
        else:
            st.metric("Change", "N/A")
            
    with col3:
        bid = df['bid'].iloc[-1] if 'bid' in df.columns and not df['bid'].isna().all() else 0
        st.metric("Best Bid", f"${bid:,.2f}" if bid else "N/A")
        
    with col4:
        ask = df['ask'].iloc[-1] if 'ask' in df.columns and not df['ask'].isna().all() else 0
        st.metric("Best Ask", f"${ask:,.2f}" if ask else "N/A")
        
    # Price chart
    st.subheader("Price Chart")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['price'],
        mode='lines',
        name='Price',
        line=dict(color='#00d4aa', width=2)
    ))
    
    if 'bid' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['bid'],
            mode='lines',
            name='Bid',
            line=dict(color='#00ff00', width=1, dash='dot'),
            opacity=0.5
        ))
        
    if 'ask' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['ask'],
            mode='lines',
            name='Ask',
            line=dict(color='#ff0000', width=1, dash='dot'),
            opacity=0.5
        ))
        
    fig.update_layout(
        height=400,
        template='plotly_dark',
        xaxis_title='Time',
        yaxis_title='Price (USD)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Raw data table
    with st.expander("📋 Raw Price Data"):
        st.dataframe(df.tail(50), use_container_width=True)


def render_trades_tab(config: dict):
    """Render the trades analysis tab."""
    st.header(f"📊 Trade Analysis - {config['symbol']} ({config['exchange']})")
    
    if not config['raw_conn']:
        st.error("Database not connected")
        return
        
    # Get trade data
    df = get_trade_timeseries(
        config['raw_conn'],
        config['symbol'],
        config['exchange'],
        config['market_type'],
        config['time_range']
    )
    
    if df.empty:
        st.warning(f"No trade data available for {config['symbol']}/{config['exchange']}/{config['market_type']}")
        return
        
    # Convert side to lowercase for consistent comparison
    if 'side' in df.columns:
        df['side'] = df['side'].str.lower()
        
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Trades", len(df))
        
    with col2:
        total_volume = df['quantity'].sum() if 'quantity' in df.columns else 0
        st.metric("Total Volume", f"{total_volume:,.2f}")
        
    with col3:
        buy_volume = df[df['side'] == 'buy']['quantity'].sum() if 'side' in df.columns else 0
        st.metric("Buy Volume", f"{buy_volume:,.2f}")
        
    with col4:
        sell_volume = df[df['side'] == 'sell']['quantity'].sum() if 'side' in df.columns else 0
        st.metric("Sell Volume", f"{sell_volume:,.2f}")
        
    # Trade chart
    st.subheader("Trade Distribution")
    
    if 'side' in df.columns and 'quantity' in df.columns:
        # Create color based on side
        df['color'] = df['side'].map({'buy': 'green', 'sell': 'red'})
        
        fig = go.Figure()
        
        # Buy trades
        buy_df = df[df['side'] == 'buy']
        fig.add_trace(go.Scatter(
            x=buy_df['timestamp'],
            y=buy_df['price'],
            mode='markers',
            name='Buys',
            marker=dict(
                size=buy_df['quantity'] / buy_df['quantity'].max() * 20 + 3,
                color='green',
                opacity=0.6
            )
        ))
        
        # Sell trades
        sell_df = df[df['side'] == 'sell']
        fig.add_trace(go.Scatter(
            x=sell_df['timestamp'],
            y=sell_df['price'],
            mode='markers',
            name='Sells',
            marker=dict(
                size=sell_df['quantity'] / sell_df['quantity'].max() * 20 + 3,
                color='red',
                opacity=0.6
            )
        ))
        
        fig.update_layout(
            height=400,
            template='plotly_dark',
            xaxis_title='Time',
            yaxis_title='Price (USD)',
            legend=dict(orientation='h', yanchor='bottom', y=1.02)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    # Volume imbalance
    st.subheader("Volume Imbalance")
    
    if buy_volume + sell_volume > 0:
        imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume)
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=imbalance * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Buy/Sell Imbalance (%)"},
            gauge={
                'axis': {'range': [-100, 100]},
                'bar': {'color': "white"},
                'steps': [
                    {'range': [-100, -20], 'color': "red"},
                    {'range': [-20, 20], 'color': "gray"},
                    {'range': [20, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': imbalance * 100
                }
            }
        ))
        
        fig.update_layout(height=300, template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)
        
    # Raw data
    with st.expander("📋 Raw Trade Data"):
        st.dataframe(df.tail(100), use_container_width=True)


def render_features_tab(config: dict):
    """Render the computed features tab."""
    st.header(f"🧮 Computed Features - {config['symbol']} ({config['exchange']})")
    
    if not config['feature_conn']:
        st.error("Feature database not connected")
        return
        
    # Feature type selector
    feature_type = st.selectbox(
        "Feature Type",
        options=FEATURE_TYPES,
        index=0
    )
    
    table_name = f"{config['symbol'].lower()}_{config['exchange']}_{config['market_type']}_{feature_type}"
    
    if table_name not in config['feature_tables']:
        st.warning(f"Table {table_name} not found")
        return
        
    # Get feature data
    try:
        df = config['feature_conn'].execute(f"""
            SELECT * FROM {table_name}
            ORDER BY timestamp DESC
            LIMIT 500
        """).fetchdf()
    except Exception as e:
        st.error(f"Error fetching features: {e}")
        return
        
    if df.empty:
        st.warning(f"No feature data in {table_name}. Make sure the feature scheduler is running.")
        return
        
    # Display stats
    st.subheader("Feature Statistics")
    
    stats = get_table_stats(config['feature_conn'], table_name)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", stats['count'])
    with col2:
        st.metric("Latest Update", str(stats['latest'])[:19] if stats['latest'] else "N/A")
    with col3:
        st.metric("Oldest Record", str(stats['oldest'])[:19] if stats['oldest'] else "N/A")
        
    # Feature values
    st.subheader("Latest Feature Values")
    
    if not df.empty:
        latest = df.iloc[0]
        
        # Display as columns
        cols = st.columns(4)
        feature_cols = [c for c in df.columns if c not in ['id', 'timestamp']]
        
        for i, col_name in enumerate(feature_cols):
            with cols[i % 4]:
                value = latest[col_name]
                if isinstance(value, float):
                    st.metric(col_name, f"{value:,.4f}")
                else:
                    st.metric(col_name, str(value))
                    
    # Feature time series
    st.subheader("Feature Time Series")
    
    # Select features to plot
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != 'id']
    
    if numeric_cols:
        selected_features = st.multiselect(
            "Select features to plot",
            options=numeric_cols,
            default=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols
        )
        
        if selected_features and 'timestamp' in df.columns:
            df_sorted = df.sort_values('timestamp')
            
            fig = go.Figure()
            for feat in selected_features:
                fig.add_trace(go.Scatter(
                    x=df_sorted['timestamp'],
                    y=df_sorted[feat],
                    mode='lines',
                    name=feat
                ))
                
            fig.update_layout(
                height=400,
                template='plotly_dark',
                xaxis_title='Time',
                yaxis_title='Value',
                legend=dict(orientation='h', yanchor='bottom', y=1.02)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
    # Raw data
    with st.expander("📋 Raw Feature Data"):
        st.dataframe(df, use_container_width=True)


def render_debug_tab(config: dict):
    """Render the debug/diagnostic tab."""
    st.header("🔧 Debug & Diagnostics")
    
    # Database file info
    st.subheader("Database Files")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Raw Database:**")
        if RAW_DB_PATH.exists():
            size_mb = RAW_DB_PATH.stat().st_size / (1024 * 1024)
            st.success(f"✅ Exists: {RAW_DB_PATH}")
            st.info(f"Size: {size_mb:.2f} MB")
        else:
            st.error(f"❌ Not found: {RAW_DB_PATH}")
            
    with col2:
        st.write("**Feature Database:**")
        if FEATURE_DB_PATH.exists():
            size_mb = FEATURE_DB_PATH.stat().st_size / (1024 * 1024)
            st.success(f"✅ Exists: {FEATURE_DB_PATH}")
            st.info(f"Size: {size_mb:.2f} MB")
        else:
            st.error(f"❌ Not found: {FEATURE_DB_PATH}")
            
    st.divider()
    
    # Table inventory
    st.subheader("Table Inventory")
    
    if config['raw_conn']:
        with st.expander(f"📊 Raw Tables ({len(config['raw_tables'])})"):
            # Group by symbol
            for symbol in SYMBOLS:
                symbol_tables = [t for t in config['raw_tables'] if t.startswith(symbol.lower())]
                if symbol_tables:
                    st.write(f"**{symbol}:** {len(symbol_tables)} tables")
                    
    if config['feature_conn']:
        with st.expander(f"🧮 Feature Tables ({len(config['feature_tables'])})"):
            for symbol in SYMBOLS:
                symbol_tables = [t for t in config['feature_tables'] if t.startswith(symbol.lower())]
                if symbol_tables:
                    st.write(f"**{symbol}:** {len(symbol_tables)} tables")
                    
    st.divider()
    
    # Query executor
    st.subheader("SQL Query Executor")
    
    db_choice = st.radio("Database", ["Raw", "Feature"], horizontal=True)
    
    query = st.text_area(
        "SQL Query",
        value="SELECT * FROM btcusdt_binance_futures_prices ORDER BY timestamp DESC LIMIT 10",
        height=100
    )
    
    if st.button("Execute Query"):
        conn = config['raw_conn'] if db_choice == "Raw" else config['feature_conn']
        if conn:
            try:
                result = conn.execute(query).fetchdf()
                st.dataframe(result, use_container_width=True)
            except Exception as e:
                st.error(f"Query error: {e}")
        else:
            st.error("Database not connected")


def main():
    """Main dashboard application."""
    
    st.title("📊 Crypto Order Flow Dashboard")
    st.markdown("Real-time visualization of streaming data and computed features")
    
    # Render sidebar and get config
    config = render_sidebar()
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", 
        "💰 Prices", 
        "📊 Trades",
        "🧮 Features",
        "🔧 Debug"
    ])
    
    with tab1:
        render_overview_tab(config)
        
    with tab2:
        render_price_tab(config)
        
    with tab3:
        render_trades_tab(config)
        
    with tab4:
        render_features_tab(config)
        
    with tab5:
        render_debug_tab(config)
        
    # Auto-refresh
    if config['auto_refresh']:
        time.sleep(5)
        st.rerun()


if __name__ == "__main__":
    main()
