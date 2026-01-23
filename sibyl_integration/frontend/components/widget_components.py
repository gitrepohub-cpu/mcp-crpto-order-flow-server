"""
🎨 Widget Components
===================

Reusable Streamlit widget components for consistent UI across all pages.

Components:
1. Symbol Selector
2. Exchange Selector
3. Timeframe Selector
4. Data Range Picker
5. Feature Category Tabs
6. Alert Banner
7. Status Indicator
8. Progress Card
9. Info Tooltip
10. Action Button Group
"""

import streamlit as st
from typing import List, Optional, Callable, Any, Dict, Union
from datetime import datetime, timedelta

# Import supported symbols from central config
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from sibyl_integration.config import (
        ALL_SUPPORTED_SYMBOLS,
        SUPPORTED_EXCHANGES,
        DERIBIT_SYMBOLS,
        get_okx_symbols,
        get_kraken_symbols
    )
except ImportError:
    # Fallback if config not available
    ALL_SUPPORTED_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ARUSDT",
                            "BRETTUSDT", "POPCATUSDT", "WIFUSDT", "PNUTUSDT"]
    SUPPORTED_EXCHANGES = ["binance", "bybit", "okx", "kraken", "gateio", "hyperliquid", "deribit"]
    DERIBIT_SYMBOLS = ["BTC-PERPETUAL", "ETH-PERPETUAL"]
    get_okx_symbols = lambda: [s.replace("USDT", "-USDT-SWAP") for s in ALL_SUPPORTED_SYMBOLS]
    get_kraken_symbols = lambda: ["XXBTZUSD", "XETHZUSD", "SOLUSD", "XXRPZUSD", "ARUSD"]


def symbol_selector(
    key: str = "symbol",
    symbols: List[str] = None,
    default: str = "BTCUSDT",
    label: str = "Symbol"
) -> str:
    """
    Standardized symbol selector dropdown.
    
    Args:
        key: Streamlit widget key
        symbols: List of available symbols
        default: Default selected symbol
        label: Widget label
        
    Returns:
        Selected symbol string
    """
    if symbols is None:
        symbols = ALL_SUPPORTED_SYMBOLS
    
    return st.selectbox(label, symbols, index=symbols.index(default) if default in symbols else 0, key=key)


def exchange_selector(
    key: str = "exchange",
    exchanges: List[str] = None,
    default: str = "binance",
    label: str = "Exchange"
) -> str:
    """
    Standardized exchange selector dropdown.
    
    Args:
        key: Streamlit widget key
        exchanges: List of available exchanges
        default: Default selected exchange
        label: Widget label
        
    Returns:
        Selected exchange string
    """
    if exchanges is None:
        exchanges = SUPPORTED_EXCHANGES
    
    return st.selectbox(label, exchanges, index=exchanges.index(default) if default in exchanges else 0, key=key)


def timeframe_selector(
    key: str = "timeframe",
    timeframes: List[str] = None,
    default: str = "1h",
    label: str = "Timeframe"
) -> str:
    """
    Standardized timeframe selector.
    
    Args:
        key: Streamlit widget key
        timeframes: List of available timeframes
        default: Default selected timeframe
        label: Widget label
        
    Returns:
        Selected timeframe string
    """
    if timeframes is None:
        timeframes = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
    
    return st.selectbox(label, timeframes, index=timeframes.index(default) if default in timeframes else 0, key=key)


def date_range_picker(
    key: str = "date_range",
    default_days: int = 7,
    label: str = "Date Range"
) -> tuple:
    """
    Date range picker widget.
    
    Args:
        key: Streamlit widget key
        default_days: Default range in days
        label: Widget label
        
    Returns:
        Tuple of (start_date, end_date)
    """
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=default_days),
            key=f"{key}_start"
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            key=f"{key}_end"
        )
    
    return start_date, end_date


def status_indicator(
    status: str,
    label: str = "",
    size: str = "medium"
) -> None:
    """
    Display a status indicator with optional label.
    
    Args:
        status: 'healthy', 'warning', 'error', 'inactive'
        label: Optional status label
        size: 'small', 'medium', 'large'
    """
    status_config = {
        "healthy": {"color": "#22c55e", "icon": "🟢", "text": "Healthy"},
        "active": {"color": "#22c55e", "icon": "🟢", "text": "Active"},
        "warning": {"color": "#f59e0b", "icon": "🟡", "text": "Warning"},
        "error": {"color": "#ef4444", "icon": "🔴", "text": "Error"},
        "inactive": {"color": "#64748b", "icon": "⚫", "text": "Inactive"},
    }
    
    config = status_config.get(status.lower(), status_config["inactive"])
    
    size_config = {
        "small": {"icon_size": "0.8rem", "text_size": "0.75rem"},
        "medium": {"icon_size": "1.2rem", "text_size": "0.9rem"},
        "large": {"icon_size": "1.5rem", "text_size": "1.1rem"},
    }
    
    sizes = size_config.get(size, size_config["medium"])
    display_label = label or config["text"]
    
    st.markdown(f"""
    <div style="display: inline-flex; align-items: center; gap: 0.5rem;">
        <span style="font-size: {sizes['icon_size']};">{config['icon']}</span>
        <span style="font-size: {sizes['text_size']}; color: {config['color']};">{display_label}</span>
    </div>
    """, unsafe_allow_html=True)


def alert_banner(
    message: str,
    level: str = "info",
    dismissible: bool = False,
    key: str = None
) -> None:
    """
    Display a styled alert banner.
    
    Args:
        message: Alert message
        level: 'info', 'success', 'warning', 'error'
        dismissible: Whether the alert can be dismissed
        key: Streamlit key for dismissible alerts
    """
    level_config = {
        "info": {"color": "#3b82f6", "bg": "59, 130, 246", "icon": "ℹ️"},
        "success": {"color": "#22c55e", "bg": "34, 197, 94", "icon": "✅"},
        "warning": {"color": "#f59e0b", "bg": "245, 158, 11", "icon": "⚠️"},
        "error": {"color": "#ef4444", "bg": "239, 68, 68", "icon": "❌"},
    }
    
    config = level_config.get(level, level_config["info"])
    
    if dismissible and key:
        if key not in st.session_state:
            st.session_state[key] = True
        
        if not st.session_state[key]:
            return
    
    st.markdown(f"""
    <div style="padding: 0.75rem 1rem;
                background: rgba({config['bg']}, 0.1);
                border-left: 4px solid {config['color']};
                border-radius: 0 8px 8px 0;
                margin-bottom: 1rem;">
        <span style="margin-right: 0.5rem;">{config['icon']}</span>
        {message}
    </div>
    """, unsafe_allow_html=True)


def progress_card(
    title: str,
    current: float,
    total: float,
    unit: str = "",
    color: str = "#667eea"
) -> None:
    """
    Display a progress card with visual bar.
    
    Args:
        title: Card title
        current: Current value
        total: Total/target value
        unit: Unit suffix
        color: Progress bar color
    """
    progress = (current / total * 100) if total > 0 else 0
    
    st.markdown(f"""
    <div style="padding: 1rem;
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                border-radius: 12px;
                border: 1px solid #334155;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
            <span style="font-size: 0.85rem; color: #94a3b8;">{title}</span>
            <span style="font-size: 0.85rem;">{current:,.0f} / {total:,.0f} {unit}</span>
        </div>
        <div style="background: #334155; border-radius: 5px; height: 8px;">
            <div style="background: {color}; width: {min(progress, 100)}%; height: 100%; border-radius: 5px;"></div>
        </div>
        <div style="text-align: right; font-size: 0.75rem; color: #64748b; margin-top: 0.25rem;">
            {progress:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)


def info_tooltip(text: str, tooltip: str) -> None:
    """
    Display text with an info tooltip.
    
    Args:
        text: Main text
        tooltip: Tooltip content
    """
    st.markdown(f"""
    <span title="{tooltip}" style="cursor: help; border-bottom: 1px dotted #64748b;">
        {text} <span style="color: #64748b;">ⓘ</span>
    </span>
    """, unsafe_allow_html=True)


def action_button_group(
    buttons: List[Dict[str, Any]],
    key_prefix: str = "action"
) -> Optional[str]:
    """
    Display a group of action buttons.
    
    Args:
        buttons: List of button configs [{"label": str, "icon": str, "key": str, "type": str}]
        key_prefix: Prefix for button keys
        
    Returns:
        Key of clicked button or None
    """
    cols = st.columns(len(buttons))
    clicked = None
    
    for i, btn in enumerate(buttons):
        with cols[i]:
            btn_type = btn.get("type", "secondary")
            label = f"{btn.get('icon', '')} {btn['label']}"
            
            if st.button(label, key=f"{key_prefix}_{btn.get('key', i)}", use_container_width=True):
                clicked = btn.get("key", str(i))
    
    return clicked


def feature_category_tabs(categories: List[str]) -> str:
    """
    Create styled feature category tabs.
    
    Args:
        categories: List of category names
        
    Returns:
        Selected category name
    """
    return st.radio(
        "Feature Category",
        categories,
        horizontal=True,
        label_visibility="collapsed"
    )


def loading_spinner(message: str = "Loading..."):
    """
    Context manager for loading state.
    
    Args:
        message: Loading message
    """
    return st.spinner(message)


def empty_state(
    icon: str = "📭",
    title: str = "No data available",
    message: str = "Start streaming to see data here",
    action_label: Optional[str] = None,
    action_callback: Optional[Callable] = None
) -> None:
    """
    Display an empty state placeholder.
    
    Args:
        icon: Emoji icon
        title: Empty state title
        message: Description message
        action_label: Optional action button label
        action_callback: Optional action button callback
    """
    st.markdown(f"""
    <div style="text-align: center; padding: 3rem 1rem;
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                border-radius: 12px; border: 1px dashed #334155;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">{icon}</div>
        <div style="font-size: 1.25rem; font-weight: 600; margin-bottom: 0.5rem;">{title}</div>
        <div style="font-size: 0.9rem; color: #64748b;">{message}</div>
    </div>
    """, unsafe_allow_html=True)
    
    if action_label and action_callback:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button(action_label):
                action_callback()


def data_freshness_indicator(
    last_update: datetime,
    threshold_seconds: int = 60
) -> None:
    """
    Display data freshness indicator.
    
    Args:
        last_update: Timestamp of last data update
        threshold_seconds: Seconds before data is considered stale
    """
    now = datetime.now()
    age = (now - last_update).total_seconds()
    
    if age < threshold_seconds:
        status = "fresh"
        color = "#22c55e"
        icon = "🟢"
    elif age < threshold_seconds * 3:
        status = "aging"
        color = "#f59e0b"
        icon = "🟡"
    else:
        status = "stale"
        color = "#ef4444"
        icon = "🔴"
    
    time_str = last_update.strftime("%H:%M:%S")
    
    st.markdown(f"""
    <div style="display: inline-flex; align-items: center; gap: 0.5rem; 
                padding: 0.25rem 0.5rem; background: rgba(26, 26, 46, 0.5);
                border-radius: 5px; font-size: 0.75rem;">
        <span>{icon}</span>
        <span style="color: #64748b;">Last update:</span>
        <span style="color: {color};">{time_str}</span>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# ENHANCED UI COMPONENTS - Exchange, Symbol, Connection, Error Handling
# =============================================================================

def render_exchange_selector(
    key: str = "exchange_selector",
    default: str = "binance",
    show_all: bool = False
) -> str:
    """
    Render enhanced exchange selector dropdown with icons.
    
    Args:
        key: Streamlit widget key
        default: Default exchange
        show_all: Whether to show "All Exchanges" option
        
    Returns:
        Selected exchange ID
    """
    exchanges = [
        ("binance", "🟡 Binance Futures"),
        ("binance_spot", "🟡 Binance Spot"),
        ("bybit", "🟠 Bybit"),
        ("okx", "🔵 OKX"),
        ("kraken", "🟣 Kraken"),
        ("gateio", "🟢 Gate.io"),
        ("hyperliquid", "⚪ Hyperliquid"),
        ("deribit", "🔴 Deribit"),
    ]
    
    if show_all:
        exchanges.insert(0, ("all", "🌐 All Exchanges"))
    
    exchange_ids = [e[0] for e in exchanges]
    exchange_labels = [e[1] for e in exchanges]
    
    default_idx = exchange_ids.index(default) if default in exchange_ids else 0
    
    selected_label = st.selectbox(
        "Exchange",
        options=exchange_labels,
        index=default_idx,
        key=key
    )
    
    selected_idx = exchange_labels.index(selected_label)
    return exchange_ids[selected_idx]


def render_symbol_search(
    exchange: str = "binance",
    key: str = "symbol_search",
    default: str = "BTCUSDT"
) -> str:
    """
    Render symbol search/autocomplete with exchange-specific symbols.
    
    Args:
        exchange: Exchange to get symbols for
        key: Streamlit widget key
        default: Default symbol
        
    Returns:
        Selected symbol
    """
    # Use centralized supported symbols
    
    # Exchange-specific adjustments
    if exchange == "deribit":
        symbols = DERIBIT_SYMBOLS
        default = "BTC-PERPETUAL"
    elif exchange == "okx":
        symbols = get_okx_symbols()
        default = "BTC-USDT-SWAP"
    elif exchange == "kraken":
        symbols = get_kraken_symbols()
        default = "XXBTZUSD"
    else:
        symbols = ALL_SUPPORTED_SYMBOLS
    
    # Use selectbox with search
    selected = st.selectbox(
        "Symbol",
        options=symbols,
        index=symbols.index(default) if default in symbols else 0,
        key=key
    )
    
    return selected


def render_connection_status(
    client: Any = None,
    key: str = "connection_status"
) -> bool:
    """
    Render real-time connection status indicator.
    
    Args:
        client: MCP client instance (optional)
        key: Streamlit widget key
        
    Returns:
        True if connected, False otherwise
    """
    try:
        # Try to get health from client
        if client and hasattr(client, 'get_health'):
            response = client.get_health()
            
            if response and response.success:
                status = response.data.get("status", "unknown")
                if status == "healthy":
                    st.success("🟢 Connected to MCP Server", icon="✅")
                    return True
                elif status == "degraded":
                    st.warning("🟡 MCP Server Degraded", icon="⚠️")
                    return True
                else:
                    st.error("🔴 MCP Server Unhealthy", icon="❌")
                    return False
            else:
                st.error("🔴 Cannot connect to MCP Server", icon="❌")
                if response and hasattr(response, 'error'):
                    st.caption(f"Error: {response.error}")
                return False
        else:
            # No client provided, show unknown status
            st.info("⚪ MCP Client not initialized", icon="ℹ️")
            return False
    except Exception as e:
        st.error(f"🔴 Connection Error: {str(e)}", icon="❌")
        return False


def render_api_error(
    response: Any,
    context: str = "API call"
) -> None:
    """
    Render API error message with details and suggestions.
    
    Args:
        response: MCPResponse object
        context: Context description for the error
    """
    if response is None:
        st.error(f"❌ {context} failed: No response received")
        return
    
    if hasattr(response, 'success') and not response.success:
        error_msg = getattr(response, 'error', 'Unknown error')
        
        with st.expander(f"❌ {context} failed", expanded=True):
            st.error(error_msg)
            
            if hasattr(response, 'data') and response.data:
                st.json(response.data)
            
            # Suggestions based on error type
            if "Connection" in str(error_msg) or "connect" in str(error_msg).lower():
                st.info("💡 **Suggestion**: Make sure the MCP server is running on http://localhost:8000")
                st.code("python -m uvicorn src.http_api:app --host 0.0.0.0 --port 8000", language="bash")
            elif "timeout" in str(error_msg).lower():
                st.info("💡 **Suggestion**: The request timed out. Try again or increase timeout.")
            elif "404" in str(error_msg):
                st.info("💡 **Suggestion**: The requested endpoint or tool was not found.")


def render_loading_skeleton(
    num_items: int = 3,
    height: int = 100
) -> None:
    """
    Render loading skeleton placeholder with shimmer animation.
    
    Args:
        num_items: Number of skeleton items
        height: Height of each skeleton item in pixels
    """
    for i in range(num_items):
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(90deg, #1e1e1e 25%, #2a2a2a 50%, #1e1e1e 75%);
                background-size: 200% 100%;
                animation: shimmer 1.5s infinite;
                border-radius: 8px;
                height: {height}px;
                margin-bottom: 10px;
            "></div>
            <style>
                @keyframes shimmer {{
                    0% {{ background-position: -200% 0; }}
                    100% {{ background-position: 200% 0; }}
                }}
            </style>
            """,
            unsafe_allow_html=True
        )


def render_refresh_button(
    key: str = "refresh",
    label: str = "🔄 Refresh",
    show_auto: bool = True
) -> bool:
    """
    Render refresh button with optional auto-refresh.
    
    Args:
        key: Streamlit widget key
        label: Button label
        show_auto: Whether to show auto-refresh option
        
    Returns:
        True if clicked
    """
    col1, col2 = st.columns([1, 3])
    
    with col1:
        clicked = st.button(label, key=key)
    
    if show_auto:
        with col2:
            auto_refresh = st.checkbox(
                "Auto-refresh",
                value=False,
                key=f"{key}_auto"
            )
            
            if auto_refresh:
                interval = st.select_slider(
                    "Interval",
                    options=[5, 10, 30, 60],
                    value=10,
                    key=f"{key}_interval",
                    format_func=lambda x: f"{x}s"
                )
                st.caption(f"Refreshing every {interval} seconds")
                
                # Note: Auto-refresh requires st_autorefresh or similar
                # st.rerun() would cause infinite loop without delay
    
    return clicked


def render_data_freshness(
    timestamp: str,
    max_age_seconds: int = 60
) -> None:
    """
    Render data freshness indicator from ISO timestamp.
    
    Args:
        timestamp: ISO format timestamp string
        max_age_seconds: Maximum acceptable age in seconds
    """
    try:
        if isinstance(timestamp, str):
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00").replace("+00:00", ""))
        else:
            dt = timestamp
        
        now = datetime.utcnow()
        age_seconds = (now - dt).total_seconds()
        
        if age_seconds < 0:
            age_seconds = 0
        
        if age_seconds < max_age_seconds:
            st.caption(f"🟢 Data is fresh ({int(age_seconds)}s ago)")
        elif age_seconds < max_age_seconds * 2:
            st.caption(f"🟡 Data is {int(age_seconds)}s old")
        else:
            st.caption(f"🔴 Data is stale ({int(age_seconds)}s ago)")
            
    except Exception as e:
        st.caption(f"⚪ Timestamp: {timestamp}")
