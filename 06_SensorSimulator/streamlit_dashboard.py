"""
Digital Twin Dashboard - Streamlit
===================================

Real-time monitoring dashboard untuk FLEXO Machine Digital Twin.
Features:
- OEE metrics visualization
- Component health monitoring
- Digital Twin 3D representation
- Historical trends (in-memory)
- Auto-refresh setiap 5 detik

Usage:
    streamlit run streamlit_dashboard.py --server.port 8501
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import sys
import os
from collections import deque
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from component_health_calculator import (
        ComponentHealthCalculator,
        calculate_oee_metrics,
        format_health_response
    )
except ImportError:
    st.error("❌ component_health_calculator.py tidak ditemukan!")
    st.stop()

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# In-memory data storage (max 1000 points)
if 'data_buffer' not in st.session_state:
    st.session_state.data_buffer = deque(maxlen=1000)

if 'latest_data' not in st.session_state:
    st.session_state.latest_data = None

if 'health_calculator' not in st.session_state:
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'component_thresholds.json')
    st.session_state.health_calculator = ComponentHealthCalculator(config_path)

# ==============================================================================
# PAGE CONFIG
# ==============================================================================

st.set_page_config(
    page_title="FLEXO Digital Twin Monitoring System",
    page_icon="⚙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Professional Theme
st.markdown("""
<style>
    /* Main container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .main-subtitle {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-top: 0.3rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border: 2px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: #3b82f6;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .status-normal {
        background: #d1fae5;
        color: #065f46;
    }
    
    .status-warning {
        background: #fed7aa;
        color: #92400e;
    }
    
    .status-critical {
        background: #fee2e2;
        color: #991b1b;
    }
    
    /* Health indicators */
    .health-excellent { 
        color: #10b981; 
        font-weight: 600;
    }
    
    .health-good { 
        color: #3b82f6; 
        font-weight: 600;
    }
    
    .health-warning { 
        color: #f59e0b; 
        font-weight: 600;
    }
    
    .health-critical { 
        color: #ef4444; 
        font-weight: 600;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1e3a8a;
        border-bottom: 3px solid #3b82f6;
        padding-bottom: 0.75rem;
        padding-top: 0.5rem;
        margin-bottom: 1.5rem;
        margin-top: 2rem;
        letter-spacing: -0.3px;
    }
    
    /* Info cards */
    .info-card {
        background: #f9fafb;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 4px;
    }
    
    /* Component details */
    .component-detail {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
    
    .component-title {
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    
    /* Data table styling */
    .dataframe {
        font-size: 0.9rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: #f9fafb;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def receive_sensor_data():
    """
    Simulate receiving data from sensor simulator.
    In production, this would be a WebSocket or HTTP endpoint.
    For now, we'll use a simple file-based approach.
    """
    data_file = os.path.join(os.path.dirname(__file__), 'latest_data.json')
    
    if os.path.exists(data_file):
        try:
            with open(data_file, 'r') as f:
                data = json.load(f)
                
            # Update latest data
            st.session_state.latest_data = data
            
            # Add to buffer
            st.session_state.data_buffer.append(data)
            
            return True
        except Exception as e:
            st.error(f"Error reading data: {e}")
            return False
    
    return False

def get_health_class(level):
    """Get CSS class for health level."""
    classes = {
        'EXCELLENT': 'health-excellent',
        'GOOD': 'health-good',
        'WARNING': 'health-warning',
        'CRITICAL': 'health-critical'
    }
    return classes.get(level, 'health-good')

def create_oee_gauge(value, title):
    """Create gauge chart for OEE metrics."""
    # Determine bar color based on value
    val_pct = value * 100
    if val_pct >= 85:
        bar_color = "#10b981"  # Green
    elif val_pct >= 60:
        bar_color = "#3b82f6"  # Blue
    elif val_pct >= 40:
        bar_color = "#f59e0b"  # Orange
    else:
        bar_color = "#ef4444"  # Red
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val_pct,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16, 'color': '#1f2937'}},
        number={'suffix': '%', 'font': {'size': 32, 'color': bar_color}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#6b7280"},
            'bar': {'color': bar_color, 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e5e7eb",
            'steps': [
                {'range': [0, 40], 'color': "#fee2e2"},
                {'range': [40, 60], 'color': "#fed7aa"},
                {'range': [60, 85], 'color': "#dbeafe"},
                {'range': [85, 100], 'color': "#d1fae5"}
            ],
            'threshold': {
                'line': {'color': "#ef4444", 'width': 3},
                'thickness': 0.8,
                'value': 85
            }
        }
    ))
    
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='white',
        font={'family': 'Arial'}
    )
    
    return fig

def create_component_health_chart(components):
    """Create bar chart for component health scores.
    
    Args:
        components: List of ComponentHealth objects
    """
    comp_names = [comp.component_name for comp in components]
    health_scores = [comp.health_score for comp in components]
    colors = []
    
    for comp in components:
        level = comp.health_level.value  # Get enum value
        if level == 'excellent':
            colors.append('#10b981')
        elif level == 'good':
            colors.append('#3b82f6')
        elif level == 'warning':
            colors.append('#f59e0b')
        else:
            colors.append('#ef4444')
    
    fig = go.Figure(data=[
        go.Bar(
            x=comp_names,
            y=health_scores,
            marker_color=colors,
            text=[f"{score:.1f}" for score in health_scores],
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title={
            'text': "Component Health Score Distribution",
            'font': {'size': 14, 'color': '#1f2937'}
        },
        xaxis_title="Component Name",
        yaxis_title="Health Score (0-100)",
        yaxis_range=[0, 105],
        height=320,
        margin=dict(l=50, r=20, t=50, b=80),
        showlegend=False,
        plot_bgcolor='#f9fafb',
        paper_bgcolor='white'
    )
    
    # Update x-axis for better readability
    fig.update_xaxes(tickangle=-45)
    
    fig.add_hline(y=80, line_dash="dot", line_color="#10b981", line_width=1.5, 
                  annotation_text="Excellent Threshold", annotation_position="right")
    fig.add_hline(y=60, line_dash="dot", line_color="#f59e0b", line_width=1.5,
                  annotation_text="Warning Threshold", annotation_position="right")
    
    return fig

def create_trend_chart(buffer_data):
    """Create time-series trend chart from buffer."""
    if len(buffer_data) < 2:
        return None
    
    timestamps = []
    oee_values = []
    avail_values = []
    perf_values = []
    qual_values = []
    
    for data in buffer_data:
        timestamps.append(data['timestamp'])
        overall = data.get('overall', {})
        oee_values.append(overall.get('oee', 0) * 100)
        avail_values.append(overall.get('availability', 0) * 100)
        perf_values.append(overall.get('performance', 0) * 100)
        qual_values.append(overall.get('quality', 0) * 100)
    
    fig = go.Figure()
    
    # Add traces with smooth lines
    fig.add_trace(go.Scatter(
        x=timestamps, y=oee_values, 
        name='OEE', 
        line=dict(color='#3b82f6', width=3),
        mode='lines',
        fill='tozeroy',
        fillcolor='rgba(59, 130, 246, 0.1)',
        hovertemplate='<b>OEE</b><br>%{y:.1f}%<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=timestamps, y=avail_values, 
        name='Availability', 
        line=dict(color='#10b981', width=2),
        mode='lines',
        hovertemplate='<b>Availability</b><br>%{y:.1f}%<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=timestamps, y=perf_values, 
        name='Performance', 
        line=dict(color='#f59e0b', width=2),
        mode='lines',
        hovertemplate='<b>Performance</b><br>%{y:.1f}%<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=timestamps, y=qual_values, 
        name='Quality', 
        line=dict(color='#8b5cf6', width=2),
        mode='lines',
        hovertemplate='<b>Quality</b><br>%{y:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': "OEE Performance Trends - Real-time Monitoring",
            'font': {'size': 16, 'color': '#1f2937', 'family': 'Arial'}
        },
        xaxis=dict(
            title="Time",
            gridcolor='#e5e7eb',
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            title="Performance (%)",
            gridcolor='#e5e7eb',
            showgrid=True,
            zeroline=False,
            range=[0, 105]
        ),
        height=450,
        hovermode='x unified',
        legend=dict(
            orientation="h", 
            yanchor="top", 
            y=-0.15, 
            xanchor="center", 
            x=0.5,
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#e5e7eb',
            borderwidth=1
        ),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=60, r=30, t=60, b=80)
    )
    
    # Add target line
    fig.add_hline(
        y=85, 
        line_dash="dash", 
        line_color="#ef4444", 
        line_width=2,
        annotation_text="Target: 85%", 
        annotation_position="right",
        annotation=dict(font=dict(size=11, color="#ef4444"))
    )
    
    # Add grid styling
    fig.update_xaxes(showline=True, linewidth=1, linecolor='#e5e7eb')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='#e5e7eb')
    
    return fig

def create_digital_twin_view(system_health):
    """Create simplified digital twin visualization."""
    components = system_health.components
    
    # Create component positions (simplified layout)
    positions = {
        'PRE_FEEDER': (1, 4),
        'FEEDER': (2, 4),
        'PRINTING': (3, 4),
        'SLOTTER': (4, 4),
        'DOWN_STACKER': (5, 4)
    }
    
    # Prepare data - components is a list of ComponentHealth objects
    comp_names = [comp.component_name for comp in components]
    health_scores = [comp.health_score for comp in components]
    
    # Get positions for each component
    x_pos = []
    y_pos = []
    for name in comp_names:
        if name in positions:
            x_pos.append(positions[name][0])
            y_pos.append(positions[name][1])
        else:
            # Fallback if component not in positions
            x_pos.append(0)
            y_pos.append(0)
    
    # Colors based on health
    colors = []
    for comp in components:
        if comp.health_level.value == 'excellent':
            colors.append('green')
        elif comp.health_level.value == 'good':
            colors.append('blue')
        elif comp.health_level.value == 'warning':
            colors.append('orange')
        else:
            colors.append('red')
    
    fig = go.Figure()
    
    # Add component boxes
    fig.add_trace(go.Scatter(
        x=x_pos,
        y=y_pos,
        mode='markers+text',
        marker=dict(
            size=80,
            color=colors,
            line=dict(color='black', width=2)
        ),
        text=comp_names,
        textposition="middle center",
        textfont=dict(size=10, color='white', family='Arial Black'),
        hovertemplate='<b>%{text}</b><br>Health: %{customdata:.1f}<extra></extra>',
        customdata=health_scores
    ))
    
    # Add flow lines
    for i in range(len(x_pos) - 1):
        fig.add_shape(
            type="line",
            x0=x_pos[i], y0=y_pos[i],
            x1=x_pos[i+1], y1=y_pos[i+1],
            line=dict(color="gray", width=3, dash="solid"),
            opacity=0.5
        )
    
    fig.update_layout(
        title="Digital Twin - Machine Flow",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 6]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[3, 5]),
        height=250,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor='rgba(240,240,240,0.5)'
    )
    
    return fig

# ==============================================================================
# MAIN DASHBOARD
# ==============================================================================

def main():
    # Professional Header
    st.markdown("""
    <div class="main-header">
        <div class="main-title">FLEXO Machine Digital Twin Monitoring System</div>
        <div class="main-subtitle">Real-time Production Performance & Component Health Tracking</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Control Bar
    col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns([3, 1, 1, 1])
    
    with col_ctrl1:
        data_file = os.path.join(os.path.dirname(__file__), 'latest_data.json')
        if os.path.exists(data_file):
            file_time = datetime.fromtimestamp(os.path.getmtime(data_file))
            st.markdown(f"""
            <div class="info-card">
                <strong>Connection Status:</strong> <span style="color: #10b981;">● LIVE</span> | 
                Last Update: {file_time.strftime('%H:%M:%S')} | 
                Data Points: {len(st.session_state.data_buffer)}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-card" style="border-left-color: #ef4444;">
                <strong>Connection Status:</strong> <span style="color: #ef4444;">● DISCONNECTED</span> | 
                Waiting for sensor data...
            </div>
            """, unsafe_allow_html=True)
    
    with col_ctrl2:
        auto_refresh = st.checkbox("Auto Refresh", value=True)
    
    with col_ctrl3:
        if st.button("↻ Refresh", use_container_width=True):
            receive_sensor_data()
            st.rerun()
    
    with col_ctrl4:
        if st.button("⊗ Clear", use_container_width=True):
            st.session_state.data_buffer.clear()
            st.rerun()
    
    # Try to receive latest data
    receive_sensor_data()
    
    # Check if we have data
    if st.session_state.latest_data is None:
        st.markdown('<p class="section-header">System Initialization</p>', unsafe_allow_html=True)
        
        col_wait1, col_wait2 = st.columns([1, 1])
        
        with col_wait1:
            st.warning("**Status:** Waiting for sensor data stream...")
            st.markdown("""
            <div class="component-detail">
                <div class="component-title">Required Actions:</div>
                <ol>
                    <li>Ensure this dashboard is running</li>
                    <li>Start sensor simulator in separate terminal</li>
                    <li>Verify data file generation</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
        
        with col_wait2:
            st.info("**Quick Start Command:**")
            st.code("""python sensor_simulator.py \\
  --file "../08_Data Produksi/data_xlsx/LAPORAN PRODUKSI 09 SEPTEMBER 2025.xlsx" \\
  --stream \\
  --stream-interval 5 \\
  --sim-interval 1.0 \\
  --api "file://latest_data.json"
""", language="bash")
        
        # Show example with dummy data
        if st.button("Show Example Data"):
            st.session_state.latest_data = {
                'timestamp': datetime.now().isoformat(),
                'machine_id': 'FLEXO_104',
                'shift': 'A',
                'overall': {
                    'produced_units': 18500,
                    'good_units': 17020,
                    'reject_units': 1480,
                    'oee': 0.6754,
                    'availability': 0.8684,
                    'performance': 0.84,
                    'quality': 0.92
                },
                'components': {
                    'PRE_FEEDER': {'tension_dev_pct': 2.15, 'feed_stops_hour': 1, 'uptime_ratio': 0.868},
                    'FEEDER': {'double_sheet_hour': 0, 'vacuum_dev_pct': 3.42, 'uptime_ratio': 0.868},
                    'PRINTING': {'registration_error_mm': 0.145, 'reject_rate_pct': 1.62, 'performance_ratio': 0.84},
                    'SLOTTER': {'miscut_pct': 0.95, 'burr_mm': 0.062, 'blade_life_used_pct': 45.8, 'uptime_ratio': 0.851},
                    'DOWN_STACKER': {'jam_hour': 0, 'misstack_pct': 0.48, 'sync_dev_pct': 1.85, 'uptime_ratio': 0.885}
                }
            }
            st.rerun()
        
        return
    
    # Process data with health calculator
    latest = st.session_state.latest_data
    overall = latest.get('overall', {})
    components_data = latest.get('components', {})
    
    # Prepare OEE metrics for calculator
    oee_metrics = {
        'oee': overall.get('oee', 0),
        'availability': overall.get('availability', 0),
        'performance': overall.get('performance', 0),
        'quality': overall.get('quality', 0)
    }
    
    # Calculate system health
    try:
        system_health = st.session_state.health_calculator.calculate_system_health(
            components_data=components_data,
            oee_metrics=oee_metrics
        )
    except Exception as e:
        st.error(f"Error calculating health: {e}")
        import traceback
        st.error(traceback.format_exc())
        return
    
    # ========== ROW 1: KEY PERFORMANCE INDICATORS ==========
    st.markdown('<p class="section-header">System Performance Overview</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Determine status badge
    status = system_health.system_status.value
    if status == "NORMAL":
        status_html = '<span class="status-badge status-normal">NORMAL</span>'
    elif status == "WARNING":
        status_html = '<span class="status-badge status-warning">WARNING</span>'
    else:
        status_html = '<span class="status-badge status-critical">CRITICAL</span>'
    
    with col1:
        # Determine health color
        if system_health.overall_health_score >= 80:
            health_color = "#10b981"
        elif system_health.overall_health_score >= 60:
            health_color = "#f59e0b"
        else:
            health_color = "#ef4444"
        
        st.markdown(f"""
        <div style="background: white; border: 2px solid #e5e7eb; border-radius: 10px; padding: 1.5rem; text-align: center; height: 100%;">
            <div style="font-size: 0.9rem; color: #6b7280; margin-bottom: 0.5rem;">System Health Score</div>
            <div style="font-size: 3rem; font-weight: 700; color: {health_color}; margin: 0.5rem 0;">{system_health.overall_health_score:.1f}</div>
            <div style="margin-top: 1rem;">{status_html}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        oee_val = overall.get('oee', 0) * 100
        oee_delta = oee_val - 85
        delta_color = "#10b981" if oee_delta >= 0 else "#ef4444"
        st.metric(
            label="Overall Equipment Effectiveness",
            value=f"{oee_val:.1f}%",
            delta=f"{oee_delta:+.1f}% vs target"
        )
    
    with col3:
        st.metric(
            label="Availability Rate",
            value=f"{overall.get('availability', 0)*100:.1f}%",
            help="Machine uptime ratio"
        )
    
    with col4:
        st.metric(
            label="Performance Rate",
            value=f"{overall.get('performance', 0)*100:.1f}%",
            help="Speed efficiency vs design speed"
        )
    
    with col5:
        st.metric(
            label="Quality Rate",
            value=f"{overall.get('quality', 0)*100:.1f}%",
            help="Good units vs total produced"
        )
    
    # ========== ROW 2: OEE BREAKDOWN & DIGITAL TWIN ==========
    st.markdown('<p class="section-header">OEE Detailed Analysis</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("**OEE Component Breakdown**")
        
        gauge_col1, gauge_col2, gauge_col3 = st.columns(3)
        
        with gauge_col1:
            st.plotly_chart(
                create_oee_gauge(overall.get('availability', 0), "Availability"),
                use_container_width=True
            )
        
        with gauge_col2:
            st.plotly_chart(
                create_oee_gauge(overall.get('performance', 0), "Performance"),
                use_container_width=True
            )
        
        with gauge_col3:
            st.plotly_chart(
                create_oee_gauge(overall.get('quality', 0), "Quality"),
                use_container_width=True
            )
    
    with col2:
        st.markdown("**Machine Flow Visualization**")
        st.plotly_chart(
            create_digital_twin_view(system_health),
            use_container_width=True
        )
    
    # ========== ROW 3: COMPONENT HEALTH MONITORING ==========
    st.markdown('<p class="section-header">Component Health Status</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        # Component health bars - pass list directly
        st.plotly_chart(
            create_component_health_chart(system_health.components),
            use_container_width=True
        )
    
    with col2:
        st.markdown("**Detailed Component Metrics**")
        # Component details - iterate list directly
        for comp in system_health.components:
            health_class = get_health_class(comp.health_level.value)
            status_display = comp.health_level.value.capitalize()
            
            with st.expander(f"**{comp.component_name}** | Score: {comp.health_score:.1f}/100"):
                st.markdown(f'<div class="component-detail">', unsafe_allow_html=True)
                st.markdown(f"**Health Status:** <span class='{health_class}'>{status_display}</span>", unsafe_allow_html=True)
                
                st.markdown("**Performance Metrics:**")
                metric_df = pd.DataFrame([
                    {"Metric": k.replace('_', ' ').title(), "Value": f"{v:.3f}" if isinstance(v, float) else str(v)}
                    for k, v in comp.metrics.items()
                ])
                st.dataframe(metric_df, hide_index=True, use_container_width=True)
                
                if comp.recommendations:
                    st.markdown("**Maintenance Recommendations:**")
                    for i, rec in enumerate(comp.recommendations, 1):
                        st.markdown(f"{i}. {rec}")
                
                st.markdown('</div>', unsafe_allow_html=True)
    
    # ========== ROW 4: HISTORICAL TRENDS ==========
    if len(st.session_state.data_buffer) > 1:
        st.markdown('<p class="section-header">Performance Trend Analysis</p>', unsafe_allow_html=True)
        
        # Add info about data collection
        col_info1, col_info2, col_info3 = st.columns([2, 1, 1])
        with col_info1:
            st.markdown(f"""
            <div class="info-card" style="border-left-color: #10b981;">
                <strong>Data Points:</strong> {len(st.session_state.data_buffer)} samples collected (max: 1000)
            </div>
            """, unsafe_allow_html=True)
        with col_info2:
            if st.session_state.data_buffer:
                first_time = st.session_state.data_buffer[0].get('timestamp', '')
                st.markdown(f"""
                <div class="info-card">
                    <strong>Start:</strong> {first_time[:19] if first_time else 'N/A'}
                </div>
                """, unsafe_allow_html=True)
        with col_info3:
            if st.session_state.data_buffer:
                last_time = st.session_state.data_buffer[-1].get('timestamp', '')
                st.markdown(f"""
                <div class="info-card">
                    <strong>Latest:</strong> {last_time[:19] if last_time else 'N/A'}
                </div>
                """, unsafe_allow_html=True)
        
        trend_fig = create_trend_chart(list(st.session_state.data_buffer))
        if trend_fig:
            st.plotly_chart(trend_fig, use_container_width=True, key="trend_chart")
    
    # ========== SIDEBAR: SYSTEM INFORMATION ==========
    with st.sidebar:
        st.markdown('<p class="section-header">System Alerts</p>', unsafe_allow_html=True)
        
        if system_health.recommendations:
            for idx, rec in enumerate(system_health.recommendations, 1):
                st.warning(f"{idx}. {rec}")
        else:
            st.success("All systems operating within normal parameters")
        
        st.markdown('<p class="section-header">Machine Information</p>', unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="info-card">
            <strong>Machine ID:</strong> {latest.get('machine_id', 'N/A')}<br>
            <strong>Current Shift:</strong> {latest.get('shift', 'N/A')}<br>
            <strong>Last Updated:</strong> {latest.get('timestamp', 'N/A')[:19]}
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<p class="section-header">Production Statistics</p>', unsafe_allow_html=True)
        
        # Production metrics in structured format
        prod_data = {
            "Metric": ["Units Produced", "Good Units", "Rejected Units", "Runtime", "Downtime"],
            "Value": [
                f"{overall.get('produced_units', 0):,}",
                f"{overall.get('good_units', 0):,}",
                f"{overall.get('reject_units', 0):,}",
                f"{overall.get('runtime_minutes', 0):.0f} min",
                f"{overall.get('downtime_minutes', 0):.0f} min"
            ]
        }
        
        prod_df = pd.DataFrame(prod_data)
        st.dataframe(prod_df, hide_index=True, use_container_width=True)
        
        # Quality rate calculation
        if overall.get('produced_units', 0) > 0:
            quality_pct = (overall.get('good_units', 0) / overall.get('produced_units', 0)) * 100
            st.metric("Quality Rate", f"{quality_pct:.2f}%")
    
    # Auto-refresh at the end (after all content is rendered)
    if auto_refresh:
        time.sleep(5)
        st.rerun()

# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()
