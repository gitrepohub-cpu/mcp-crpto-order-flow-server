"""
🎯 MCP-Powered Sibyl - Main Navigation Router
=============================================

This is the main entry point for the Sibyl Streamlit application.
It replaces the original Sibyl navigation with MCP-focused pages.

Run with:
    streamlit run sibyl_integration/frontend/index_router.py

Pages:
    - 📊 Dashboard: Main overview with metrics, signals, charts
    - 🏛️ Institutional: 139 institutional features (8 tabs)
    - 🔮 Forecasting: 38+ Darts models
    - 🌊 Streaming: Real-time data monitor
    - 🔍 Model Health: Drift detection, CV monitoring
    - 🌐 Cross-Exchange: Arbitrage, correlations
    - 📡 Signals: 15 composite signals
    - � Features: Feature explorer (40+ features)
    - 🎭 Regimes: Market regime analyzer
"""

import streamlit as st
from streamlit_option_menu import option_menu
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import pages
from sibyl_integration.frontend.tab_pages.mcp_dashboard import show_mcp_dashboard
from sibyl_integration.frontend.tab_pages.institutional_features import show_institutional_features
from sibyl_integration.frontend.tab_pages.forecasting_studio import show_forecasting_studio
from sibyl_integration.frontend.tab_pages.streaming_monitor import show_streaming_monitor
from sibyl_integration.frontend.tab_pages.model_health import show_model_health
from sibyl_integration.frontend.tab_pages.cross_exchange import show_cross_exchange
from sibyl_integration.frontend.tab_pages.signal_aggregator import show_signal_aggregator
from sibyl_integration.frontend.tab_pages.feature_explorer import show_feature_explorer
from sibyl_integration.frontend.tab_pages.regime_analyzer import show_regime_analyzer


def setup_page_config():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="MCP Crypto Intelligence Hub",
        page_icon="🔮",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://github.com/your-repo/mcp-crypto-orderflow',
            'Report a bug': 'https://github.com/your-repo/mcp-crypto-orderflow/issues',
            'About': """
            # MCP Crypto Intelligence Hub
            
            Powered by MCP Crypto Order Flow Server
            - 252 MCP Tools
            - 8 Exchanges
            - 504 DuckDB Tables
            - 139 Institutional Features
            - 38+ Darts Forecasting Models
            """
        }
    )


def apply_custom_css():
    """Apply custom CSS styling"""
    st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #667eea;
        --secondary-color: #764ba2;
        --bullish-color: #22c55e;
        --bearish-color: #ef4444;
        --neutral-color: #94a3b8;
        --bg-dark: #0e1117;
        --bg-card: #1a1a2e;
        --bg-card-hover: #16213e;
        --border-color: #334155;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-card-hover) 100%);
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid var(--border-color);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.15);
    }
    
    /* Signal colors */
    .signal-bullish { color: var(--bullish-color) !important; }
    .signal-bearish { color: var(--bearish-color) !important; }
    .signal-neutral { color: var(--neutral-color) !important; }
    
    /* Gauge container */
    .gauge-container {
        background: var(--bg-card);
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid var(--border-color);
        text-align: center;
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .status-healthy {
        background-color: rgba(34, 197, 94, 0.2);
        color: var(--bullish-color);
    }
    
    .status-warning {
        background-color: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
    }
    
    .status-critical {
        background-color: rgba(239, 68, 68, 0.2);
        color: var(--bearish-color);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: var(--bg-dark);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-dark);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-color);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: var(--bg-card);
        border-radius: 8px;
        padding: 8px 16px;
        border: 1px solid var(--border-color);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color) !important;
    }
    
    /* Feature grid */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
    }
    
    /* Chart container */
    .chart-container {
        background: var(--bg-card);
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid var(--border-color);
    }
    </style>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render sidebar with navigation and global controls"""
    from sibyl_integration.frontend.components.widget_components import (
        render_exchange_selector,
        render_symbol_search,
        render_connection_status
    )
    from sibyl_integration.mcp_client import get_sync_client
    
    with st.sidebar:
        # Logo/Title
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <div style="font-size: 2.5rem;">🔮</div>
            <div style="font-size: 1.2rem; font-weight: bold; 
                        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                        -webkit-background-clip: text;
                        -webkit-text-fill-color: transparent;">
                MCP Intelligence
            </div>
            <div style="font-size: 0.75rem; color: #64748b;">
                Crypto Order Flow Analytics
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Global controls
        st.markdown("### ⚙️ Global Settings")
        
        # Exchange selector
        exchange = render_exchange_selector(
            key="global_exchange",
            default=st.session_state.get("exchange", "binance")
        )
        st.session_state["exchange"] = exchange
        
        # Symbol selector
        symbol = render_symbol_search(
            exchange=exchange,
            key="global_symbol",
            default=st.session_state.get("symbol", "BTCUSDT")
        )
        st.session_state["symbol"] = symbol
        
        st.markdown("---")
        
        # Navigation menu
        selected = option_menu(
            menu_title=None,
            options=[
                "📊 Dashboard",
                "🏛️ Institutional",
                "🔮 Forecasting",
                "🌊 Streaming",
                "🔍 Model Health",
                "🌐 Cross-Exchange",
                "📡 Signals",
                "🔬 Features",
                "🎭 Regimes",
            ],
            icons=[
                "speedometer2",
                "bank",
                "magic",
                "broadcast",
                "heart-pulse",
                "globe",
                "broadcast-pin",
                "search",
                "mask",
            ],
            default_index=0,
            styles={
                "container": {
                    "padding": "0!important",
                    "background-color": "transparent",
                },
                "icon": {
                    "color": "#667eea",
                    "font-size": "16px",
                },
                "nav-link": {
                    "font-size": "14px",
                    "text-align": "left",
                    "margin": "2px 0",
                    "padding": "10px 15px",
                    "border-radius": "8px",
                    "--hover-color": "#1e3a5f",
                },
                "nav-link-selected": {
                    "background-color": "#1e3a5f",
                    "font-weight": "600",
                },
            }
        )
        
        st.markdown("---")
        
        # System status with live connection check
        st.markdown("### 🔧 System Status")
        
        try:
            client = get_sync_client()
            render_connection_status(client, key="sidebar_connection")
        except Exception as e:
            st.error(f"🔴 Client error: {str(e)[:30]}...")
        
        st.markdown("---")
        
        # Quick stats
        st.markdown("### 📈 Quick Stats")
        st.markdown("""
        <div style="font-size: 0.8rem; color: #94a3b8;">
            <div>🛠️ <b>253</b> MCP Tools</div>
            <div>🏦 <b>8</b> Exchanges</div>
            <div>📊 <b>504</b> Tables</div>
            <div>🎯 <b>149</b> Features</div>
            <div>🔮 <b>38+</b> Models</div>
        </div>
        """, unsafe_allow_html=True)
        
        return selected


def main():
    """Main application entry point"""
    # Setup
    setup_page_config()
    apply_custom_css()
    
    # Render sidebar and get selected page
    selected = render_sidebar()
    
    # Route to selected page
    page_mapping = {
        "📊 Dashboard": show_mcp_dashboard,
        "🏛️ Institutional": show_institutional_features,
        "🔮 Forecasting": show_forecasting_studio,
        "🌊 Streaming": show_streaming_monitor,
        "🔍 Model Health": show_model_health,
        "🌐 Cross-Exchange": show_cross_exchange,
        "📡 Signals": show_signal_aggregator,
        "🔬 Features": show_feature_explorer,
        "🎭 Regimes": show_regime_analyzer,
    }
    
    # Execute selected page
    if selected in page_mapping:
        page_mapping[selected]()
    else:
        st.error(f"Page not found: {selected}")


if __name__ == "__main__":
    main()
