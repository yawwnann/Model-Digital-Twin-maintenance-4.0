
import streamlit as st
import pandas as pd
import numpy as np
from digital_twin_realtime import FlexoDigitalTwin
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import tempfile
import os

def main():
    # Page configuration
    st.set_page_config(
        page_title="FlexoTwin Digital Maintenance System",
        page_icon="▶",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Simple clean styling
    st.markdown("""
        <style>
        /* Main theme */
        .main {
            padding-top: 1rem;
        }
        
        /* Header styling */
        .main-header {
            background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 2rem;
            color: white;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* Card styling */
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-left: 5px solid #2a5298;
            margin: 1rem 0;
        }
        
        /* Upload area */
        .upload-section {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 15px;
            text-align: center;
            margin: 1rem 0;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        
        /* Feature badges */
        .feature-badge {
            display: inline-block;
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(10px);
            padding: 0.3rem 0.8rem;
            margin: 0.2rem;
            border-radius: 20px;
            font-size: 0.9rem;
            border: 1px solid rgba(255, 255, 255, 0.3);
        }
        
        /* File info card */
        .file-info-card {
            background: linear-gradient(135deg, #e3f2fd 0%, #f3e5f5 100%);
            padding: 1.5rem;
            border-radius: 12px;
            border-left: 4px solid #2196f3;
            margin: 1rem 0;
        }
        
        .file-info-card h4 {
            margin: 0 0 0.5rem 0;
            color: #1565c0;
        }
        
        /* Analysis preview */
        .analysis-preview {
            background: linear-gradient(135deg, #fff3e0 0%, #fce4ec 100%);
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #ff9800;
        }
        
        .analysis-preview p {
            margin: 0.3rem 0;
            font-size: 0.9rem;
        }
        
        /* Feature showcase */
        .feature-showcase {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 1.5rem;
            border-radius: 12px;
            height: 100%;
            border: 1px solid #dee2e6;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .feature-showcase:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        }
        
        .feature-showcase h4 {
            color: #495057;
            margin-bottom: 1rem;
            border-bottom: 2px solid #007bff;
            padding-bottom: 0.5rem;
        }
        
        /* Prediction section */
        .prediction-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 15px;
            color: white;
            margin: 2rem 0;
            box-shadow: 0 8px 20px rgba(0,0,0,0.2);
        }
        
        /* Status indicators */
        .status-good { color: #28a745; font-weight: bold; }
        .status-warning { color: #ffc107; font-weight: bold; }
        .status-danger { color: #dc3545; font-weight: bold; }
        
        /* Animation for loading */
        .loading-spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #3498db;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Main Header
    st.markdown("""
        <div class="main-header">
            <h1>🏭 FLEXO 104 Digital Twin System</h1>
            <h3>Smart Predictive Maintenance Dashboard</h3>
            <p>Industry 4.0 Manufacturing Intelligence Platform | Real-time Analytics & Predictions</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    setup_sidebar()

    # Show latest results ribbon if predictions already available
    if 'predictions' in st.session_state:
        preds = st.session_state.predictions
        st.markdown("### ✅ Latest Prediction Summary")
        rcol1, rcol2, rcol3, rcol4 = st.columns([1,1,1,1])
        with rcol1:
            st.metric("Health Score", f"{preds.get('health_score', 0):.1f}%")
        with rcol2:
            st.metric("Failure Risk", f"{preds.get('failure_risk', 0)*100:.1f}%")
        with rcol3:
            st.metric("OEE Forecast", f"{preds.get('oee_forecast', 0):.1f}%")
        with rcol4:
            if st.button("Open Detailed Results", key="open_results_banner"):
                st.session_state.active_view = "🔮 Predictions"
                st.rerun()
    
    # --- Stateful Navigation (replaces tabs) ---
    # Initialize active view (persist across reruns)
    if 'active_view' not in st.session_state:
        # Default to Predictions if predictions already exist, else Upload, else Dashboard
        if 'predictions' in st.session_state:
            st.session_state.active_view = "🔮 Predictions"
        else:
            st.session_state.active_view = "📁 Data Upload"

    nav_options = ["📊 Dashboard", "📁 Data Upload", "🔮 Predictions", "📈 Analytics"]

    # Top navigation
    nav_col1, nav_col2 = st.columns([3, 1])
    with nav_col1:
        active = st.radio(
            "Navigation",
            options=nav_options,
            index=nav_options.index(st.session_state.active_view),
            horizontal=True,
        )
        st.session_state.active_view = active
    with nav_col2:
        # If predictions exist and user not in Predictions, offer a quick jump
        if 'predictions' in st.session_state and st.session_state.active_view != "🔮 Predictions":
            if st.button("View Results ▶", type="primary"):
                st.session_state.active_view = "🔮 Predictions"
                st.rerun()

    st.markdown("---")

    # Render active view
    if st.session_state.active_view == "📊 Dashboard":
        show_dashboard()
    elif st.session_state.active_view == "📁 Data Upload":
        show_upload_interface()
    elif st.session_state.active_view == "🔮 Predictions":
        show_predictions()
    else:
        show_analytics()

def setup_sidebar():
    """Setup sidebar with system information"""
    with st.sidebar:
        st.markdown("### 🏭 System Overview")
        
        # System status
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Status", "🟢 Online")
        with col2:
            st.metric("Uptime", "99.9%")
        
        st.markdown("---")
        
        # Key metrics summary
        st.markdown("### 📊 Quick Stats")
        st.metric("Last OEE", "65.3%", "↑ 2.1%")
        st.metric("Health Score", "72.5%", "↑ 1.8%")
        st.metric("Risk Level", "Medium", "→")
        
        st.markdown("---")
        
        # Feature highlights
        st.markdown("### ✨ Features")
        st.markdown("""
        🔍 **Real-time Monitoring**
        - Live OEE tracking
        - Equipment status
        - Performance metrics
        
        🤖 **AI Predictions**
        - Maintenance forecasting
        - Failure risk assessment
        - Optimization recommendations
        
        📊 **Advanced Analytics**
        - Trend analysis
        - Pattern recognition
        - Historical comparisons
        """)
        
        st.markdown("---")
        st.markdown("### 📞 Support")
        st.info("For technical support:\n📧 support@flexotwin.com\n📱 +62-XXX-XXXX")

def show_dashboard():
    """Main dashboard view"""
    st.markdown("## 📊 Real-time FLEXO 104 Dashboard")
    
    # Live status indicator
    col_status1, col_status2, col_status3 = st.columns([1, 1, 2])
    with col_status1:
        st.markdown("**🟢 System Online**")
    with col_status2:
        st.markdown(f"**🕒 {datetime.now().strftime('%H:%M:%S')}**")
    with col_status3:
        if st.button("🔄 Refresh Data", help="Update real-time metrics"):
            st.rerun()
    
    st.markdown("---")
    
    # Key metrics row with enhanced visuals
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # OEE Score with gauge-like display
        oee_value = 65.3
        st.metric(
            label="🎯 OEE Score",
            value=f"{oee_value}%",
            delta="2.1%",
            help="Overall Equipment Effectiveness - Industry target: 85%"
        )
        # Mini progress bar
        progress_oee = oee_value / 100
        st.progress(progress_oee)
    
    with col2:
        # Health Score
        health_value = 72.5
        st.metric(
            label="💚 Health Score", 
            value=f"{health_value}%",
            delta="1.8%",
            help="Composite equipment health indicator"
        )
        progress_health = health_value / 100
        st.progress(progress_health)
    
    with col3:
        # Failure Risk (inverted - lower is better)
        risk_value = 38.2
        st.metric(
            label="⚠️ Failure Risk",
            value=f"{risk_value}%",
            delta="-5.3%",
            delta_color="inverse",
            help="Probability of equipment failure - lower is better"
        )
        progress_risk = (100 - risk_value) / 100  # Invert for progress bar
        st.progress(progress_risk)
    
    with col4:
        # Next Maintenance
        maintenance_days = 12
        st.metric(
            label="🔧 Next Maintenance",
            value=f"{maintenance_days} Days",
            delta="On Schedule",
            help="Predicted optimal maintenance timing"
        )
        progress_maintenance = max(0, (30 - maintenance_days) / 30)  # 30 days max
        st.progress(progress_maintenance)
    
    # Charts section
    st.markdown("---")
    
    # Interactive tabs for different views
    chart_tab1, chart_tab2, chart_tab3 = st.tabs(["📈 Performance Trends", "🔧 Component Status", "🏭 Machine Layout"])
    
    with chart_tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # OEE Trend Chart with enhanced styling
            st.markdown("#### 📈 OEE Performance Trend")
            months = ['Apr 24', 'May 24', 'Jun 24', 'Jul 24', 'Aug 24', 'Sep 24']
            oee_values = [45.2, 52.1, 58.7, 61.2, 63.8, 65.3]
            target_line = [85] * len(months)  # Industry target
            
            fig = go.Figure()
            
            # OEE actual line
            fig.add_trace(go.Scatter(
                x=months, y=oee_values,
                mode='lines+markers',
                line=dict(color='#2a5298', width=3),
                marker=dict(size=10, color='#ffffff', line=dict(width=3, color='#2a5298')),
                name='Actual OEE',
                hovertemplate='<b>%{x}</b><br>OEE: %{y}%<extra></extra>'
            ))
            
            # Target line
            fig.add_trace(go.Scatter(
                x=months, y=target_line,
                mode='lines',
                line=dict(color='#dc3545', width=2, dash='dash'),
                name='Industry Target (85%)',
                hovertemplate='<b>Target</b><br>OEE: %{y}%<extra></extra>'
            ))
            
            fig.update_layout(
                height=350,
                showlegend=True,
                legend=dict(x=0, y=1),
                margin=dict(l=0, r=0, t=0, b=0),
                hovermode='x unified'
            )
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            # Availability, Performance, Quality breakdown
            st.markdown("#### ⚙️ OEE Components Breakdown")
            
            categories = ['Availability', 'Performance', 'Quality']
            current_values = [85.2, 76.8, 89.5]
            target_values = [95, 95, 98]
            
            fig = go.Figure()
            
            # Current values
            fig.add_trace(go.Bar(
                name='Current',
                x=categories,
                y=current_values,
                marker_color='#2a5298',
                hovertemplate='<b>%{x}</b><br>Current: %{y}%<extra></extra>'
            ))
            
            # Target values
            fig.add_trace(go.Bar(
                name='Target',
                x=categories,
                y=target_values,
                marker_color='#ffc107',
                opacity=0.6,
                hovertemplate='<b>%{x}</b><br>Target: %{y}%<extra></extra>'
            ))
            
            fig.update_layout(
                height=350,
                showlegend=True,
                barmode='group',
                margin=dict(l=0, r=0, t=0, b=0)
            )
            st.plotly_chart(fig, width='stretch')
    
    with chart_tab2:
        # Component Health with interactive selection
        st.markdown("#### 🔧 Equipment Component Health Monitor")
        
        components_data = {
            'PRE FEEDER': {'health': 82, 'status': 'Good', 'last_maintenance': '12 days ago'},
            'FEEDER': {'health': 78, 'status': 'Warning', 'last_maintenance': '15 days ago'},
            'PRINTING 1': {'health': 65, 'status': 'Needs Attention', 'last_maintenance': '8 days ago'},
            'PRINTING 2': {'health': 71, 'status': 'Warning', 'last_maintenance': '10 days ago'},
            'PRINTING 3': {'health': 74, 'status': 'Warning', 'last_maintenance': '11 days ago'},
            'PRINTING 4': {'health': 77, 'status': 'Warning', 'last_maintenance': '13 days ago'},
            'SLOTTER': {'health': 68, 'status': 'Needs Attention', 'last_maintenance': '7 days ago'},
            'DOWN STACKER': {'health': 85, 'status': 'Good', 'last_maintenance': '20 days ago'}
        }
        
        # Create interactive component chart
        components = list(components_data.keys())
        health_scores = [components_data[c]['health'] for c in components]
        colors = ['#28a745' if h > 85 else '#ffc107' if h > 70 else '#dc3545' for h in health_scores]
        
        fig = go.Figure(go.Bar(
            y=components,
            x=health_scores,
            orientation='h',
            marker_color=colors,
            text=[f'{h}%' for h in health_scores],
            textposition='inside',
            hovertemplate='<b>%{y}</b><br>Health: %{x}%<br>Status: %{customdata}<extra></extra>',
            customdata=[components_data[c]['status'] for c in components]
        ))
        
        fig.update_layout(
            height=400,
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis_title="Health Score (%)",
            yaxis_title="Components"
        )
        st.plotly_chart(fig, width='stretch')
        
        # Component details table
        st.markdown("#### 📋 Component Details")
        df_components = pd.DataFrame(components_data).T
        df_components.index.name = 'Component'
        st.dataframe(df_components, width='stretch')
    
    with chart_tab3:
        # Machine layout visualization
        st.markdown("#### 🏭 FLEXO 104 Machine Layout")
        
        # Create machine layout diagram
        fig = go.Figure()
        
        # Machine components positions (mock layout)
        components_layout = {
            'Control Panel': (1, 4, '#28a745'),
            'Hydraulic System': (2, 3, '#ffc107'),
            'Print Cylinders': (3, 3, '#dc3545'),
            'Drive Motors': (4, 3, '#28a745'),
            'Ink System': (2, 2, '#ffc107'),
            'Web Handling': (4, 2, '#28a745')
        }
        
        for component, (x, y, color) in components_layout.items():
            health = components_data[component]['health']
            
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                marker=dict(size=80, color=color, opacity=0.7),
                text=component,
                textposition="middle center",
                textfont=dict(size=10, color='white'),
                name=component,
                hovertemplate=f'<b>{component}</b><br>Health: {health}%<br>Position: ({x}, {y})<extra></extra>'
            ))
        
        fig.update_layout(
            height=400,
            showlegend=False,
            xaxis=dict(range=[0, 5], showgrid=True, title="Machine Length"),
            yaxis=dict(range=[1, 5], showgrid=True, title="Machine Width"),
            title="Interactive Machine Layout - Click components for details"
        )
        st.plotly_chart(fig, width='stretch')
        
        # Status legend
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("🟢 **Excellent/Good:** >85%")
        with col2:
            st.markdown("🟡 **Warning:** 70-85%")
        with col3:
            st.markdown("🔴 **Needs Attention:** <70%")

def show_upload_interface():
    """Enhanced file upload interface with drag-and-drop styling"""
    st.markdown("## 📁 Production Data Upload Center")
    
    # Enhanced upload section with animations
    st.markdown("""
        <div class="upload-section">
            <h2>📊 Smart Production Analysis</h2>
            <p>Upload your Excel production report to generate AI-powered insights</p>
            <div style="margin-top: 1rem;">
                <span class="feature-badge">✨ AI Predictions</span>
                <span class="feature-badge">📈 Real-time Analytics</span>
                <span class="feature-badge">🔮 Future Forecasts</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # File upload with enhanced styling
    st.markdown("### 🎯 Upload Your Report")
    
    uploaded_file = st.file_uploader(
        "📂 Select Production Report File",
        type=['xls', 'xlsx'],
        help="Supported: Excel files (.xls, .xlsx) up to 50MB",
        accept_multiple_files=False
    )
    
    if uploaded_file is not None:
        # Enhanced file info display
        st.markdown("---")
        
        # File details card
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown("#### ✅ File Successfully Uploaded")
            
            file_size = len(uploaded_file.getvalue()) / 1024 / 1024
            
            st.markdown(f"""
                <div class="file-info-card">
                    <h4>📄 {uploaded_file.name}</h4>
                    <p><strong>Size:</strong> {file_size:.2f} MB</p>
                    <p><strong>Type:</strong> {uploaded_file.type}</p>
                    <p><strong>Status:</strong> <span style="color: #28a745;">Ready for Processing</span></p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # File validation status
            st.markdown("#### 🔍 Validation")
            
            # Basic file validation
            is_valid_size = file_size < 50
            is_valid_type = uploaded_file.type in ['application/vnd.ms-excel', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']
            
            if is_valid_size:
                st.success("✅ File size OK")
            else:
                st.error("❌ File too large")
            
            if is_valid_type:
                st.success("✅ Format valid")
            else:
                st.error("❌ Invalid format")
        
        with col3:
            # Analysis preview
            st.markdown("#### ⚡ Quick Analysis")
            
            # Extract period info
            current_month = extract_month_from_filename(uploaded_file.name)
            next_month = get_next_month(current_month)
            
            st.markdown(f"""
                <div class="analysis-preview">
                    <p><strong>📅 Current:</strong> {current_month}</p>
                    <p><strong>🔮 Forecast:</strong> {next_month}</p>
                    <p><strong>🎯 Ready:</strong> <span style="color: #007bff;">AI Analysis</span></p>
                </div>
            """, unsafe_allow_html=True)
        
        # Processing section
        st.markdown("---")
        st.markdown("### 🚀 AI Processing Center")
        
        process_col1, process_col2 = st.columns([1, 1])
        
        with process_col1:
            # Processing options
            st.markdown("#### ⚙️ Analysis Options")
            
            analysis_type = st.selectbox(
                "Select Analysis Type",
                ["Complete Analysis", "Quick Prediction", "Performance Focus", "Maintenance Focus"],
                help="Choose the type of analysis to perform"
            )
            
            include_recommendations = st.checkbox(
                "📋 Include Maintenance Recommendations", 
                value=True,
                help="Generate actionable maintenance insights"
            )
            
            include_forecasting = st.checkbox(
                "🔮 Include Future Forecasting", 
                value=True,
                help="Generate predictions for next month"
            )
        
        with process_col2:
            # Processing information
            st.markdown("#### ℹ️ What We'll Analyze")
            
            analysis_features = [
                "🎯 Overall Equipment Effectiveness (OEE)",
                "⚙️ Component performance trends",
                "📊 Quality metrics and patterns",
                "⚠️ Failure risk assessment",
                "🔧 Maintenance optimization opportunities",
                "📈 Performance improvement suggestions"
            ]
            
            for feature in analysis_features:
                st.markdown(f"- {feature}")
        
        # Main processing button
        st.markdown("---")
        
        if st.button("🎯 Start AI Analysis", type="primary", key="main_process"):
            with st.spinner("🤖 AI is analyzing your production data..."):
                # Add processing animation
                progress_container = st.container()
                
                with progress_container:
                    process_uploaded_file(uploaded_file, current_month, next_month)
    
    else:
        # Enhanced empty state with examples and guidance
        st.markdown("---")
        st.markdown("### 📚 Getting Started Guide")
        
        # Feature showcase
        showcase_col1, showcase_col2, showcase_col3 = st.columns(3)
        
        with showcase_col1:
            st.markdown("""
                <div class="feature-showcase">
                    <h4>🎯 Smart Analysis</h4>
                    <p>AI-powered analysis of production data with machine learning algorithms</p>
                    <ul>
                        <li>OEE calculation & trends</li>
                        <li>Component health scoring</li>
                        <li>Performance benchmarking</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        with showcase_col2:
            st.markdown("""
                <div class="feature-showcase">
                    <h4>🔮 Predictive Insights</h4>
                    <p>Future performance forecasting and maintenance planning</p>
                    <ul>
                        <li>Next month predictions</li>
                        <li>Failure risk assessment</li>
                        <li>Optimal maintenance timing</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        with showcase_col3:
            st.markdown("""
                <div class="feature-showcase">
                    <h4>📊 Visual Dashboard</h4>
                    <p>Interactive charts and real-time monitoring interface</p>
                    <ul>
                        <li>Real-time metrics display</li>
                        <li>Interactive trend charts</li>
                        <li>Component status indicators</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        # File format guidance
        st.markdown("---")
        st.markdown("### 📋 File Requirements & Format")
        
        format_col1, format_col2 = st.columns(2)
        
        with format_col1:
            st.markdown("""
                #### 📁 File Naming Convention
                ```
                ✅ LAPORAN PRODUKSI SEPTEMBER 2025.xls
                ✅ LAPORAN PRODUKSI OKTOBER 2024.xlsx
                ✅ LAPORAN PRODUKSI 01. JANUARI 2025.xls
                ```
                
                #### 📊 Required Data Sheets
                - **OEE DESEMBER** (or current month)
                - **OEE PER WEEK UPDATE**
                - Production metrics
                - Quality parameters
            """)
        
        with format_col2:
            st.markdown("""
                #### ⚙️ Technical Requirements
                
                | Requirement | Specification |
                |-------------|---------------|
                | **File Format** | .xls or .xlsx |
                | **Max File Size** | 50 MB |
                | **Encoding** | UTF-8 or Windows-1252 |
                | **Date Format** | DD/MM/YYYY |
                
                #### 🔍 Data Quality Tips
                - Ensure complete date ranges
                - Remove empty rows/columns
                - Use consistent number formats
                - Include all required sheets
            """)
        
        # Sample data preview
        st.markdown("---")
        st.markdown("### 🎯 Sample Data Structure")
        
        # Create sample data preview
        sample_data = {
            'Date': ['01/09/2024', '02/09/2024', '03/09/2024'],
            'OEE (%)': [85.2, 78.9, 92.1],
            'Availability (%)': [95.2, 89.1, 97.8],
            'Performance (%)': [89.5, 88.4, 94.2],
            'Quality (%)': [99.9, 99.8, 99.9],
            'Production (units)': [1250, 1180, 1340]
        }
        
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df, width='stretch')
        
        st.info("💡 **Pro Tip**: Your uploaded file should contain similar data structure with daily or weekly production metrics.")

def show_predictions():
    """Predictions display tab"""
    st.markdown("## 🔮 AI-Powered Predictive Analytics")
    
    # Debug info (remove in production)
    if st.sidebar.checkbox("Debug Info", False):
        st.sidebar.write("Session State Keys:", list(st.session_state.keys()))
        if 'predictions' in st.session_state:
            st.sidebar.write("Predictions Keys:", list(st.session_state.predictions.keys()))
            if 'component_health' in st.session_state.predictions:
                st.sidebar.write("Component Health Available:", len(st.session_state.predictions['component_health']))
            else:
                st.sidebar.warning("⚠️ component_health key missing!")
    
    # Check if predictions exist
    if 'predictions' not in st.session_state:
        # Enhanced empty state with call-to-action
        st.markdown("""
            <div class="prediction-box">
                <h2>🤖 AI Prediction Engine Ready</h2>
                <p>Upload your production data to unlock powerful predictions</p>
                <ul style="text-align: left; margin-top: 1rem;">
                    <li>🎯 Next month OEE forecast</li>
                    <li>⚠️ Equipment failure risk assessment</li>
                    <li>🔧 Optimal maintenance timing</li>
                    <li>📈 Performance improvement recommendations</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🧠 How Our AI Works")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
                **🔍 Data Analysis**
                - Historical OEE patterns
                - Component performance trends  
                - Maintenance history correlation
                - Seasonal variations
            """)
        
        with col2:
            st.markdown("""
                **🤖 Machine Learning**
                - Random Forest algorithms
                - Feature engineering (72 parameters)
                - Cross-validation training
                - Real-time model updates
            """)
        
        with col3:
            st.markdown("""
                **📊 Predictions**
                - Health score forecasting
                - Failure risk probability
                - Maintenance scheduling
                - Performance optimization
            """)
        return
    
    # Display predictions with enhanced visuals
    predictions = st.session_state.predictions
    
    # Main prediction metrics with gauges
    st.markdown("### 🎯 Prediction Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        health_score = predictions.get('health_score', 0)
        st.metric(
            label="💚 Predicted Health Score",
            value=f"{health_score:.1f}%",
            help="Overall equipment health prediction for next month"
        )
        
        # Health score gauge
        fig_health = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = health_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Health Score"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "#2a5298"},
                'steps': [
                    {'range': [0, 50], 'color': "#ffebee"},
                    {'range': [50, 80], 'color': "#fff3e0"},
                    {'range': [80, 100], 'color': "#e8f5e8"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90}}
        ))
        fig_health.update_layout(height=250)
        st.plotly_chart(fig_health, width='stretch')
    
    with col2:
        failure_risk = predictions.get('failure_risk', 0) * 100
        st.metric(
            label="⚠️ Failure Risk",
            value=f"{failure_risk:.1f}%",
            help="Probability of equipment failure in next month"
        )
        
        # Risk gauge (inverted - lower is better)
        fig_risk = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = failure_risk,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Failure Risk"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "#dc3545"},
                'steps': [
                    {'range': [0, 30], 'color': "#e8f5e8"},
                    {'range': [30, 60], 'color': "#fff3e0"},
                    {'range': [60, 100], 'color': "#ffebee"}],
                'threshold': {
                    'line': {'color': "orange", 'width': 4},
                    'thickness': 0.75,
                    'value': 70}}
        ))
        fig_risk.update_layout(height=250)
        st.plotly_chart(fig_risk, width='stretch')
    
    with col3:
        oee_forecast = predictions.get('oee_forecast', 0)
        st.metric(
            label="🎯 OEE Forecast",
            value=f"{oee_forecast:.1f}%",
            help="Expected OEE performance for next month"
        )
        
        # OEE forecast gauge
        fig_oee = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = oee_forecast,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "OEE Forecast"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "#28a745"},
                'steps': [
                    {'range': [0, 40], 'color': "#ffebee"},
                    {'range': [40, 70], 'color': "#fff3e0"},
                    {'range': [70, 100], 'color': "#e8f5e8"}],
                'threshold': {
                    'line': {'color': "blue", 'width': 4},
                    'thickness': 0.75,
                    'value': 85}}
        ))
        fig_oee.update_layout(height=250)
        st.plotly_chart(fig_oee, width='stretch')
    
    # Detailed predictions breakdown
    st.markdown("---")
    st.markdown("### 📋 Detailed Predictions & Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Maintenance recommendations
        st.markdown("#### 🔧 Maintenance Recommendations")
        
        recommendations = [
            "🔍 Inspect hydraulic system within 7 days",
            "🛠️ Schedule print cylinder maintenance in 2 weeks",
            "⚙️ Check drive motor alignment next week",
            "🧽 Clean ink system filters within 10 days"
        ]
        
        for rec in recommendations:
            st.markdown(f"- {rec}")
    
    with col2:
        # Performance insights
        st.markdown("#### 📈 Performance Insights")
        
        insights = [
            f"🎯 Target OEE achievable with focused maintenance",
            f"⚠️ Risk level: {'High' if failure_risk > 60 else 'Medium' if failure_risk > 30 else 'Low'}",
            f"📊 Health trend: {'Declining' if health_score < 60 else 'Stable' if health_score < 80 else 'Improving'}",
            f"🔮 Confidence level: {predictions.get('confidence', 0.85)*100:.1f}%"
        ]
        
        for insight in insights:
            st.markdown(f"- {insight}")
    
    # Action items
    st.markdown("---")
    st.markdown("#### ⚡ Immediate Action Items")
    
    action_col1, action_col2, action_col3 = st.columns(3)
    
    with action_col1:
        if failure_risk > 60:
            st.error("🚨 HIGH RISK: Schedule immediate inspection")
        elif failure_risk > 30:
            st.warning("⚠️ MEDIUM RISK: Plan preventive maintenance")
        else:
            st.success("✅ LOW RISK: Continue monitoring")
    
    with action_col2:
        if health_score < 60:
            st.error("🔧 Poor health: Immediate intervention needed")
        elif health_score < 80:
            st.warning("📋 Moderate health: Planned maintenance required")
        else:
            st.success("💚 Good health: Routine maintenance sufficient")
    
    with action_col3:
        if oee_forecast < 50:
            st.error("📉 Low OEE predicted: Performance improvement needed")
        elif oee_forecast < 75:
            st.warning("📈 Moderate OEE: Optimization opportunities exist")
        else:
            st.success("🎯 Good OEE forecast: On track for targets")
    
    # Component Health Section
    st.markdown("---")
    st.markdown("### 🔧 Component Health Monitor")
    
    if 'component_health' in predictions and predictions['component_health']:
        st.markdown("Detailed health status for each machine component")
        
        component_health = predictions['component_health']
        
        # Sort components by priority (lowest first - needs attention)
        sorted_components = sorted(
            component_health.items(),
            key=lambda x: (x[1]['priority'], -x[1]['health'])
        )
        
        # Simple component health display
        st.markdown("#### Component Health Status:")
        
        # Display in 3 columns for better layout
        col1, col2, col3 = st.columns(3)
        
        components_list = list(sorted_components)
        
        for idx, (component_name, component_data) in enumerate(components_list):
            # Choose column based on index
            if idx % 3 == 0:
                col = col1
            elif idx % 3 == 1:
                col = col2
            else:
                col = col3
                
            with col:
                health_pct = component_data['health']
                status = component_data['status']
                icon = component_data.get('icon', '⚙️')
                
                # Simple metric display
                st.metric(
                    label=f"{icon} {component_name}",
                    value=f"{health_pct:.1f}%",
                    help=f"Status: {status}"
                )
                
                # Simple color indicator
                if health_pct >= 80:
                    st.success("✅ Good condition")
                elif health_pct >= 60:
                    st.warning("⚠️ Needs monitoring")
                else:
                    st.error("🔧 Requires attention")
        
        # Simple summary table
        st.markdown("---")
        st.markdown("#### Quick Health Summary")
        
        # Create simple summary
        summary_cols = st.columns(4)
        
        with summary_cols[0]:
            excellent_count = sum(1 for _, data in component_health.items() if data['health'] >= 85)
            st.metric("Excellent", excellent_count, help="Health ≥ 85%")
        
        with summary_cols[1]:
            good_count = sum(1 for _, data in component_health.items() if 70 <= data['health'] < 85)
            st.metric("Good", good_count, help="Health 70-84%")
        
        with summary_cols[2]:
            warning_count = sum(1 for _, data in component_health.items() if 55 <= data['health'] < 70)
            st.metric("Warning", warning_count, help="Health 55-69%")
        
        with summary_cols[3]:
            attention_count = sum(1 for _, data in component_health.items() if data['health'] < 55)
            st.metric("Needs Attention", attention_count, help="Health < 55%")
    else:
        # Show message if component health not available
        st.info("🔄 Component health data will appear after processing production data. Please upload and analyze your production report first.")
        
        # Show example of what will be displayed
        st.markdown("#### 📋 Components That Will Be Monitored:")
        example_components = [
            "🔧 **Hydraulic System** - Monitors pressure, leaks, and fluid levels",
            "⚙️ **Print Cylinders** - Tracks alignment, wear, and impression quality",
            "⚡ **Drive Motors** - Checks torque, vibration, and electrical performance",
            "🖥️ **Control Panel** - Monitors electronic systems and sensors",
            "🎨 **Ink System** - Tracks ink flow, viscosity, and delivery",
            "📜 **Web Handling** - Monitors tension, alignment, and speed control"
        ]
        
        for comp in example_components:
            st.markdown(f"- {comp}")
    
    # ===== METODOLOGI ANALYSIS SECTIONS =====
    st.markdown("---")
    st.header("📊 Digital Twin Methodology Analysis")
    st.markdown("*Implementasi 8 langkah metodologi Digital Twin sesuai flowchart penelitian*")
    
    # Check if methodology data is available
    has_evaluation = 'evaluation_metrics' in predictions and predictions['evaluation_metrics']
    has_fishbone = 'fishbone_analysis' in predictions and predictions['fishbone_analysis'] 
    has_fmea = 'fmea_results' in predictions and predictions['fmea_results']
    
    if has_evaluation or has_fishbone or has_fmea:
        # Step 6: Model Evaluation Metrics (MAE, RMSE, MAPE)
        if has_evaluation:
            evaluation_metrics = predictions['evaluation_metrics']
            st.subheader("📈 Step 6: Model Evaluation Metrics")
            st.markdown("Validasi model menggunakan rumus standar MAE, RMSE, MAPE")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="MAE",
                    value=f"{evaluation_metrics.get('MAE', 0):.4f}",
                    help="Mean Absolute Error - Semakin rendah semakin baik"
                )
            
            with col2:
                st.metric(
                    label="RMSE", 
                    value=f"{evaluation_metrics.get('RMSE', 0):.4f}",
                    help="Root Mean Square Error - Semakin rendah semakin baik"
                )
            
            with col3:
                st.metric(
                    label="MAPE",
                    value=f"{evaluation_metrics.get('MAPE', 0):.2f}%",
                    help="Mean Absolute Percentage Error - Semakin rendah semakin baik"
                )
            
            with col4:
                st.metric(
                    label="Model Accuracy",
                    value=f"{evaluation_metrics.get('model_accuracy', 0):.1f}%",
                    help="Akurasi keseluruhan model prediksi"
                )
            
            # Evaluation status
            status = evaluation_metrics.get('evaluation_status', 'Unknown')
            if status == 'Excellent':
                st.success(f"🟢 **Model Evaluation Status: {status}** - Model berkinerja sangat baik")
            elif status == 'Good':
                st.info(f"🟡 **Model Evaluation Status: {status}** - Model berkinerja baik")
            else:
                st.warning(f"🔴 **Model Evaluation Status: {status}** - Model perlu perbaikan")
        
        # Step 3: Fishbone Analysis
        if has_fishbone:
            fishbone_analysis = predictions['fishbone_analysis']
            st.subheader("🐟 Step 3: Fishbone Analysis - Identifikasi Penyebab")
            st.markdown("Analisis faktor penyebab penurunan OEE berdasarkan 5M+E (Man, Machine, Method, Material, Environment)")
            
            categories = fishbone_analysis.get('categories', {})
            if categories:
                st.markdown("**Impact by Category:**")
                
                # Display category impacts
                cols = st.columns(5)
                category_names = ['Man', 'Machine', 'Method', 'Material', 'Environment']
                
                for i, category in enumerate(category_names):
                    if category in categories:
                        data = categories[category]
                        with cols[i]:
                            impact = data.get('total_impact', 0)
                            priority = data.get('priority', 'Low')
                            
                            if priority == 'High':
                                color = '🔴'
                                delta_color = 'inverse'
                            elif priority == 'Medium':
                                color = '🟡' 
                                delta_color = 'normal'
                            else:
                                color = '🟢'
                                delta_color = 'normal'
                                
                            st.metric(
                                label=f"{color} {category}",
                                value=f"{impact:.1f}%",
                                delta=f"Priority: {priority}",
                                delta_color=delta_color
                            )
                
                # Top contributing factors
                top_factors = fishbone_analysis.get('top_factors', [])
                if top_factors:
                    st.markdown("**🔍 Top Contributing Factors:**")
                    for i, factor in enumerate(top_factors[:3], 1):
                        category_emoji = {'Man': '👤', 'Machine': '⚙️', 'Method': '📋', 'Material': '📦', 'Environment': '🌡️'}
                        emoji = category_emoji.get(factor['category'], '📊')
                        st.write(f"{i}. {emoji} **{factor['factor'].replace('_', ' ').title()}** ({factor['category']}) - Impact: **{factor['impact_score']:.1f}%**")
            
            # Analysis summary
            summary = fishbone_analysis.get('analysis_summary', '')
            if summary:
                st.warning(f"📋 **Kesimpulan Analisis:** {summary}")
        
        # Step 4: FMEA Analysis
        if has_fmea:
            fmea_results = predictions['fmea_results']
            st.subheader("⚠️ Step 4: FMEA Analysis - Risk Priority Number")
            st.markdown("Evaluasi kritis dengan FMEA untuk menentukan prioritas kegagalan berdasarkan **RPN = Severity × Occurrence × Detection**")
            
            # FMEA Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                high_priority = fmea_results.get('high_priority_count', 0)
                st.metric(
                    label="High Priority Failures",
                    value=high_priority,
                    delta="RPN > 120",
                    delta_color="inverse" if high_priority > 0 else "normal"
                )
            
            with col2:
                highest_rpn = fmea_results.get('highest_rpn', 0)
                st.metric(
                    label="Highest RPN",
                    value=highest_rpn,
                    delta="Critical" if highest_rpn >= 200 else "High" if highest_rpn >= 120 else "Medium",
                    delta_color="inverse" if highest_rpn >= 200 else "normal"
                )
            
            with col3:
                avg_rpn = fmea_results.get('average_rpn', 0)
                st.metric(
                    label="Average RPN", 
                    value=f"{avg_rpn:.1f}",
                    help="Risk Priority Number rata-rata semua mode kegagalan"
                )
            
            with col4:
                critical_failure = fmea_results.get('critical_failure_mode', 'None')
                st.metric(
                    label="Critical Failure",
                    value=critical_failure[:12] + "..." if len(critical_failure) > 12 else critical_failure,
                    help=f"Mode kegagalan paling kritis: {critical_failure}"
                )
            
            # FMEA Summary
            fmea_summary = fmea_results.get('fmea_summary', '')
            if fmea_summary:
                if "critical" in fmea_summary.lower():
                    st.error(f"🚨 **FMEA Summary:** {fmea_summary}")
                else:
                    st.info(f"📊 **FMEA Summary:** {fmea_summary}")
            
            # Top failure modes table  
            failure_modes = fmea_results.get('failure_modes', [])
            if failure_modes:
                st.markdown("**🚨 Top Risk Failure Modes:**")
                
                # Create dataframe for better display
                df_data = []
                for fm in failure_modes[:5]:  # Top 5
                    risk_emoji = {'Critical': '🔴', 'High': '🟠', 'Medium': '🟡', 'Low': '🟢'}
                    emoji = risk_emoji.get(fm['risk_level'], '⚪')
                    
                    df_data.append({
                        'Failure Mode': fm['failure_mode'],
                        'S': fm['severity'],
                        'O': fm['occurrence'], 
                        'D': fm['detection'],
                        'RPN': fm['rpn'],
                        'Risk': f"{emoji} {fm['risk_level']}",
                        'Action Priority': fm['action_priority']
                    })
                
                if df_data:
                    import pandas as pd
                    fmea_df = pd.DataFrame(df_data)
                    st.dataframe(fmea_df, use_container_width=True, hide_index=True)
    else:
        st.info("🔄 **Metodologi analysis akan muncul setelah memproses data produksi.** Upload dan analisis laporan produksi untuk melihat:")
        
        st.markdown("""
        - **📈 Step 6: Model Evaluation** - Metrik MAE, RMSE, MAPE untuk validasi akurasi
        - **🐟 Step 3: Fishbone Analysis** - Identifikasi penyebab masalah berdasarkan 5M+E 
        - **⚠️ Step 4: FMEA Analysis** - Evaluasi risiko dengan Risk Priority Number (RPN)
        - **📊 Step 5-7: ML Model Results** - Prediksi OEE dan rekomendasi maintenance
        """)

def show_analytics():
    """Advanced analytics tab with comprehensive insights"""
    st.markdown("## 📊 Manufacturing Intelligence Analytics")
    
    # Generate sample analytics if data exists
    if 'data' in st.session_state:
        df = st.session_state.data
        
        # Analytics overview metrics
        st.markdown("### 📈 Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_oee = df['oee'].mean() if 'oee' in df.columns else 0
            st.metric(
                label="Average OEE", 
                value=f"{avg_oee:.1f}%",
                delta=f"{avg_oee - 75:.1f}% vs Target"
            )
        
        with col2:
            uptime = df['availability'].mean() if 'availability' in df.columns else 0
            st.metric(
                label="Average Uptime", 
                value=f"{uptime:.1f}%",
                delta=f"{uptime - 85:.1f}% vs Target"
            )
        
        with col3:
            performance = df['performance'].mean() if 'performance' in df.columns else 0
            st.metric(
                label="Average Performance", 
                value=f"{performance:.1f}%",
                delta=f"{performance - 80:.1f}% vs Target"
            )
        
        with col4:
            quality = df['quality'].mean() if 'quality' in df.columns else 0
            st.metric(
                label="Average Quality", 
                value=f"{quality:.1f}%",
                delta=f"{quality - 95:.1f}% vs Target"
            )
        
        # Multi-tab analytics
        st.markdown("---")
        analytics_tab1, analytics_tab2, analytics_tab3 = st.tabs([
            "🎯 Performance Trends", 
            "🔍 Root Cause Analysis", 
            "🚀 Optimization Insights"
        ])
        
        with analytics_tab1:
            # Performance trends over time
            st.markdown("#### 📈 Historical Performance Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # OEE trend chart with target line
                fig_oee_trend = go.Figure()
                
                oee_data = df['oee'].head(30) if 'oee' in df.columns else [70 + i*2 + np.random.randint(-5, 5) for i in range(30)]
                
                fig_oee_trend.add_trace(go.Scatter(
                    x=list(range(len(oee_data))),
                    y=oee_data,
                    mode='lines+markers',
                    name='OEE',
                    line=dict(color='#2a5298', width=3),
                    marker=dict(size=6)
                ))
                
                # Add target line
                fig_oee_trend.add_hline(y=75, line_dash="dash", 
                                       line_color="red", annotation_text="Target 75%")
                
                fig_oee_trend.update_layout(
                    title="OEE Performance Trend",
                    xaxis_title="Time Period",
                    yaxis_title="OEE (%)",
                    height=400,
                    showlegend=True
                )
                st.plotly_chart(fig_oee_trend, width='stretch')
            
            with col2:
                # Performance radar chart
                categories = ['Availability', 'Performance', 'Quality', 'Reliability']
                values = [
                    df['availability'].mean() if 'availability' in df.columns else 82,
                    df['performance'].mean() if 'performance' in df.columns else 78,
                    df['quality'].mean() if 'quality' in df.columns else 95,
                    85  # Reliability placeholder
                ]
                
                fig_radar = go.Figure()
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name='Current Performance',
                    fillcolor='rgba(42, 82, 152, 0.3)',
                    line=dict(color='#2a5298', width=2)
                ))
                
                # Add target performance
                target_values = [85, 80, 95, 90]
                fig_radar.add_trace(go.Scatterpolar(
                    r=target_values,
                    theta=categories,
                    fill=None,
                    name='Target Performance',
                    line=dict(color='red', dash='dash', width=2)
                ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )),
                    showlegend=True,
                    title="Performance vs Target",
                    height=400
                )
                st.plotly_chart(fig_radar, width='stretch')
        
        with analytics_tab2:
            # Root cause analysis
            st.markdown("#### 🔍 Root Cause Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Performance issues breakdown
                st.markdown("**Performance Loss Categories**")
                
                loss_categories = {
                    'Equipment Failures': 25,
                    'Setup/Changeover': 15,
                    'Minor Stoppages': 20,
                    'Reduced Speed': 10,
                    'Defects/Rework': 8,
                    'Startup Losses': 12,
                    'Other': 10
                }
                
                fig_losses = px.pie(
                    values=list(loss_categories.values()),
                    names=list(loss_categories.keys()),
                    title="OEE Loss Distribution"
                )
                fig_losses.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_losses, width='stretch')
            
            with col2:
                # Equipment issues frequency
                st.markdown("**Equipment Issue Frequency**")
                
                issues = ['Hydraulic System', 'Print Cylinder', 'Ink System', 'Drive Motor', 'Electrical']
                frequencies = [35, 25, 20, 15, 10]
                
                fig_issues = px.bar(
                    x=issues,
                    y=frequencies,
                    title="Equipment Issues (% of Total)",
                    color=frequencies,
                    color_continuous_scale='Reds'
                )
                fig_issues.update_layout(xaxis={'tickangle': 45})
                st.plotly_chart(fig_issues, width='stretch')
        
        with analytics_tab3:
            # Optimization insights
            st.markdown("#### 🚀 Optimization Recommendations")
            
            # Optimization priority matrix
            optimization_data = {
                'Improvement Area': [
                    'Hydraulic System Upgrade',
                    'Preventive Maintenance',
                    'Operator Training',
                    'Setup Optimization',
                    'Quality System Enhancement'
                ],
                'Impact (%)': [8, 6, 4, 5, 3],
                'Effort (1-10)': [9, 4, 3, 5, 6],
                'ROI Score': [8.9, 9.5, 8.7, 7.5, 6.2],
                'Timeline (months)': [6, 2, 1, 3, 4]
            }
            
            opt_df = pd.DataFrame(optimization_data)
            
            # Display recommendations
            col1, col2 = st.columns(2)
            
            with col1:
                # Priority matrix visualization
                fig_optimization = px.scatter(
                    opt_df, 
                    x='Effort (1-10)', 
                    y='Impact (%)',
                    size='ROI Score',
                    hover_data=['Timeline (months)'],
                    text='Improvement Area',
                    title='Optimization Impact vs Effort'
                )
                
                fig_optimization.update_traces(textposition='top center')
                fig_optimization.update_layout(height=400)
                st.plotly_chart(fig_optimization, width='stretch')
            
            with col2:
                # ROI ranking
                st.markdown("**ROI Priority Ranking**")
                
                # Sort by ROI score
                sorted_opt = opt_df.sort_values('ROI Score', ascending=False)
                
                for idx, row in sorted_opt.iterrows():
                    roi_color = "🟢" if row['ROI Score'] > 8.5 else "🟡" if row['ROI Score'] > 7.5 else "🔴"
                    st.markdown(f"""
                        {roi_color} **{row['Improvement Area']}**  
                        ROI: {row['ROI Score']:.1f} | Impact: {row['Impact (%)']}% | Timeline: {row['Timeline (months)']} months
                    """)
        
        # Statistical summary
        st.markdown("---")
        st.markdown("### 📋 Statistical Summary")
        
        # Enhanced statistical analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats_summary = df[numeric_cols].describe()
            st.dataframe(stats_summary, width='stretch')
        
        # Correlation analysis
        if len(numeric_cols) > 1:
            st.markdown("### 🔗 Component Correlation Matrix")
            
            corr_matrix = df[numeric_cols].corr()
            
            fig_corr = px.imshow(
                corr_matrix, 
                title="Component Performance Correlations",
                aspect="auto",
                color_continuous_scale="RdBu_r"
            )
            fig_corr.update_layout(height=500)
            st.plotly_chart(fig_corr, width='stretch')
    
    else:
        # Enhanced empty state with feature preview
        st.markdown("""
            <div class="prediction-box">
                <h2>🧠 Manufacturing Intelligence Ready</h2>
                <p>Upload production data to unlock powerful analytics capabilities</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🚀 Available Analytics Features")
        
        feature_col1, feature_col2, feature_col3 = st.columns(3)
        
        with feature_col1:
            st.markdown("""
                **📊 Performance Analytics**
                - OEE trend analysis
                - Component correlation
                - Statistical summaries
                - Performance patterns
            """)
        
        with feature_col2:
            st.markdown("""
                **🔍 Root Cause Analysis**
                - Loss category breakdown
                - Equipment issue tracking
                - Failure pattern detection
                - Impact assessment
            """)
        
        with feature_col3:
            st.markdown("""
                **🎯 Optimization Insights**
                - Improvement recommendations
                - ROI priority matrix
                - Efficiency opportunities
                - Action plan guidance
            """)
        
        st.info("💡 **Pro Tip**: Upload your production data to see these powerful analytics in action!")

def process_uploaded_file(uploaded_file, current_month, next_month):
    """Process the uploaded Excel file"""
    with st.spinner("🔄 Processing file and generating predictions..."):
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Process with Digital Twin
            dt = FlexoDigitalTwin()
            
            # Load data and train
            progress_bar = st.progress(0)
            st.text("Loading production data...")
            progress_bar.progress(25)
            
            data_dict = dt.load_comprehensive_data("../../08_Data Produksi")
            progress_bar.progress(50)
            
            st.text("Training predictive models...")
            dt.train_models(data_dict)
            progress_bar.progress(75)
            
            st.text("Generating predictions...")
            predictions = dt.predict_from_excel(tmp_file_path)
            progress_bar.progress(100)
            
            # Store predictions in session state
            st.session_state.predictions = predictions
            st.session_state.data_processed = True
            st.session_state.active_view = "🔮 Predictions"
            
            # Clean up temp file
            os.unlink(tmp_file_path)
            
            # Show success message with results preview
            st.success("✅ Analysis completed successfully!")
            st.balloons()
            
            # Show quick results preview
            st.markdown("### 🎯 Quick Results Preview")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                health_score = predictions.get('health_score', 0)
                st.metric("Health Score", f"{health_score:.1f}%")
            
            with col2:
                failure_risk = predictions.get('failure_risk', 0) * 100
                st.metric("Failure Risk", f"{failure_risk:.1f}%")
            
            with col3:
                oee_forecast = predictions.get('oee_forecast', 0)
                st.metric("OEE Forecast", f"{oee_forecast:.1f}%")
            
            st.success("Results are ready. Redirecting to Predictions view…")
            
            # Auto-switch to predictions view
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.error("Please check your file format and try again.")

def extract_month_from_filename(filename):
    """Extract month from filename"""
    months = {
        'JANUARI': 'January 2025', 'FEBRUARI': 'February 2025', 'MARET': 'March 2025',
        'APRIL': 'April 2025', 'MEI': 'May 2025', 'JUNI': 'June 2025',
        'JULI': 'July 2025', 'AGUSTUS': 'August 2025', 'SEPTEMBER': 'September',
        'OKTOBER': 'October', 'NOVEMBER': 'November', 'DESEMBER': 'December'
    }
    
    filename_upper = filename.upper()
    
    for month_name, display_name in months.items():
        if month_name in filename_upper:
            if '2024' in filename:
                return f"{display_name} 2024"
            elif '2025' in filename:
                return f"{display_name} 2025"
            else:
                return display_name
    
    return "Unknown Period"

def get_next_month(current_month):
    """Get next month for prediction"""
    if "September 2024" in current_month:
        return "October 2024"
    elif "October 2024" in current_month:
        return "November 2024"
    elif "September 2025" in current_month:
        return "October 2025"
    else:
        return "Next Month"

if __name__ == "__main__":
    main()