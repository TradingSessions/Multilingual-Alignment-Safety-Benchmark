# monitoring_dashboard.py - Real-time evaluation monitoring system

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sqlite3
from datetime import datetime, timedelta
import time
import json
from typing import Dict, List, Optional
import asyncio
import aiohttp
from pathlib import Path

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent))

from data_manager import DataManager
from config import get_config

# Page configuration
st.set_page_config(
    page_title="MASB-Alt Monitoring Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .alert-box {
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .alert-high {
        background-color: #ffcccc;
        border-left: 5px solid #ff0000;
    }
    .alert-medium {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
    }
    .status-indicator {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 5px;
    }
    .status-green { background-color: #28a745; }
    .status-yellow { background-color: #ffc107; }
    .status-red { background-color: #dc3545; }
</style>
""", unsafe_allow_html=True)

class MonitoringDashboard:
    """Real-time monitoring dashboard for MASB-Alt evaluations"""
    
    def __init__(self):
        self.config = get_config()
        self.data_manager = DataManager(self.config.data.base_path)
        self.refresh_interval = 5  # seconds
        
    def get_recent_evaluations(self, hours: int = 24) -> pd.DataFrame:
        """Get evaluations from the last N hours"""
        conn = sqlite3.connect(self.data_manager.db_path)
        
        cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        query = """
        SELECT 
            e.*,
            r.model,
            r.latency_ms,
            r.error as response_error,
            p.language,
            p.domain
        FROM evaluations e
        JOIN responses r ON e.response_id = r.response_id
        JOIN prompts p ON r.prompt_id = p.prompt_id
        WHERE e.timestamp > ?
        ORDER BY e.timestamp DESC
        """
        
        df = pd.read_sql_query(query, conn, params=(cutoff_time,))
        conn.close()
        
        # Parse timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def get_system_status(self) -> Dict:
        """Get current system status"""
        try:
            # Check database connection
            conn = sqlite3.connect(self.data_manager.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM evaluations")
            total_evaluations = cursor.fetchone()[0]
            conn.close()
            
            # Get recent activity
            recent_df = self.get_recent_evaluations(hours=1)
            evaluations_last_hour = len(recent_df)
            
            # Calculate error rate
            if evaluations_last_hour > 0:
                error_rate = (recent_df['response_error'].notna().sum() / evaluations_last_hour) * 100
            else:
                error_rate = 0
            
            # System health
            if evaluations_last_hour > 50 and error_rate < 5:
                health = "healthy"
            elif evaluations_last_hour > 10 or error_rate < 20:
                health = "degraded"
            else:
                health = "critical"
            
            return {
                "health": health,
                "total_evaluations": total_evaluations,
                "evaluations_last_hour": evaluations_last_hour,
                "error_rate": error_rate,
                "active_models": recent_df['model'].nunique() if not recent_df.empty else 0,
                "last_evaluation": recent_df['timestamp'].max() if not recent_df.empty else None
            }
        except Exception as e:
            return {
                "health": "error",
                "error": str(e)
            }
    
    def create_real_time_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create real-time evaluation activity chart"""
        # Group by minute
        df_resampled = df.set_index('timestamp').resample('5T').agg({
            'evaluation_id': 'count',
            'alignment_score': 'mean',
            'risk_level': lambda x: (x.isin(['high', 'critical'])).sum()
        }).reset_index()
        
        df_resampled.columns = ['timestamp', 'count', 'avg_score', 'high_risk_count']
        
        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"secondary_y": True}]]
        )
        
        # Add evaluation count
        fig.add_trace(
            go.Scatter(
                x=df_resampled['timestamp'],
                y=df_resampled['count'],
                name='Evaluations',
                mode='lines+markers',
                line=dict(color='blue', width=2),
                fill='tozeroy',
                fillcolor='rgba(0,100,255,0.2)'
            ),
            secondary_y=False
        )
        
        # Add average score
        fig.add_trace(
            go.Scatter(
                x=df_resampled['timestamp'],
                y=df_resampled['avg_score'],
                name='Avg Score',
                mode='lines+markers',
                line=dict(color='green', width=2, dash='dash'),
                yaxis='y2'
            ),
            secondary_y=True
        )
        
        # Add high risk count
        fig.add_trace(
            go.Bar(
                x=df_resampled['timestamp'],
                y=df_resampled['high_risk_count'],
                name='High Risk',
                marker_color='red',
                opacity=0.6,
                yaxis='y'
            ),
            secondary_y=False
        )
        
        fig.update_xaxis(title_text="Time")
        fig.update_yaxis(title_text="Count", secondary_y=False)
        fig.update_yaxis(title_text="Average Score", secondary_y=True, range=[0, 5])
        
        fig.update_layout(
            title="Real-Time Evaluation Activity",
            hovermode='x unified',
            height=400
        )
        
        return fig
    
    def create_model_performance_gauges(self, df: pd.DataFrame) -> Dict[str, go.Figure]:
        """Create performance gauges for each model"""
        model_stats = df.groupby('model').agg({
            'alignment_score': 'mean',
            'latency_ms': 'mean',
            'evaluation_id': 'count'
        }).round(2)
        
        gauges = {}
        
        for model in model_stats.index:
            stats = model_stats.loc[model]
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=stats['alignment_score'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"{model}<br>Score", 'font': {'size': 14}},
                delta={'reference': 3.5, 'increasing': {'color': "green"}},
                gauge={
                    'axis': {'range': [None, 5], 'tickwidth': 1},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 2], 'color': "lightgray"},
                        {'range': [2, 3.5], 'color': "gray"},
                        {'range': [3.5, 5], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 4
                    }
                }
            ))
            
            fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=20))
            gauges[model] = fig
        
        return gauges
    
    def create_language_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create language performance heatmap"""
        pivot_table = df.pivot_table(
            values='alignment_score',
            index='language',
            columns='domain',
            aggfunc='mean'
        )
        
        fig = px.imshow(
            pivot_table,
            labels=dict(x="Domain", y="Language", color="Avg Score"),
            title="Language-Domain Performance Heatmap",
            color_continuous_scale="RdYlGn",
            aspect="auto"
        )
        
        fig.update_layout(height=400)
        
        return fig
    
    def detect_anomalies(self, df: pd.DataFrame) -> List[Dict]:
        """Detect anomalies in recent evaluations"""
        anomalies = []
        
        # Check for sudden score drops
        if len(df) > 10:
            recent_avg = df.head(10)['alignment_score'].mean()
            overall_avg = df['alignment_score'].mean()
            
            if recent_avg < overall_avg - 0.5:
                anomalies.append({
                    "type": "score_drop",
                    "severity": "high",
                    "message": f"Recent average score ({recent_avg:.2f}) significantly below normal ({overall_avg:.2f})",
                    "timestamp": datetime.now()
                })
        
        # Check for high error rate
        error_rate = df['response_error'].notna().sum() / len(df) * 100 if len(df) > 0 else 0
        if error_rate > 10:
            anomalies.append({
                "type": "high_error_rate",
                "severity": "high" if error_rate > 20 else "medium",
                "message": f"High error rate detected: {error_rate:.1f}%",
                "timestamp": datetime.now()
            })
        
        # Check for language imbalance
        lang_distribution = df['language'].value_counts()
        if len(lang_distribution) > 0:
            min_lang_pct = (lang_distribution.min() / len(df)) * 100
            if min_lang_pct < 5:
                anomalies.append({
                    "type": "language_imbalance",
                    "severity": "medium",
                    "message": f"Language imbalance detected: {lang_distribution.idxmin()} only {min_lang_pct:.1f}% of evaluations",
                    "timestamp": datetime.now()
                })
        
        # Check for model latency
        high_latency_models = df.groupby('model')['latency_ms'].mean()
        for model, latency in high_latency_models.items():
            if latency > 5000:  # 5 seconds
                anomalies.append({
                    "type": "high_latency",
                    "severity": "medium",
                    "message": f"{model} showing high latency: {latency:.0f}ms average",
                    "timestamp": datetime.now()
                })
        
        return anomalies
    
    def create_risk_distribution_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create risk distribution chart"""
        risk_counts = df['risk_level'].value_counts()
        
        fig = go.Figure(data=[
            go.Pie(
                labels=risk_counts.index,
                values=risk_counts.values,
                hole=0.3,
                marker_colors=['#28a745', '#ffc107', '#fd7e14', '#dc3545']
            )
        ])
        
        fig.update_layout(
            title="Risk Level Distribution",
            height=300
        )
        
        return fig

def render_dashboard():
    """Main dashboard rendering function"""
    st.title("🚨 MASB-Alt Real-Time Monitoring Dashboard")
    
    # Initialize dashboard
    dashboard = MonitoringDashboard()
    
    # Sidebar controls
    st.sidebar.header("⚙️ Controls")
    
    # Time range selector
    time_range = st.sidebar.selectbox(
        "Time Range",
        options=[1, 6, 12, 24, 48, 168],
        format_func=lambda x: f"Last {x} hours" if x < 168 else "Last week",
        index=3
    )
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
    refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 5, 60, 10)
    
    # Filter options
    st.sidebar.header("🔍 Filters")
    
    # Get recent data
    df = dashboard.get_recent_evaluations(hours=time_range)
    
    if not df.empty:
        selected_languages = st.sidebar.multiselect(
            "Languages",
            options=df['language'].unique(),
            default=df['language'].unique()
        )
        
        selected_domains = st.sidebar.multiselect(
            "Domains",
            options=df['domain'].unique(),
            default=df['domain'].unique()
        )
        
        selected_models = st.sidebar.multiselect(
            "Models",
            options=df['model'].unique(),
            default=df['model'].unique()
        )
        
        # Apply filters
        df_filtered = df[
            (df['language'].isin(selected_languages)) &
            (df['domain'].isin(selected_domains)) &
            (df['model'].isin(selected_models))
        ]
    else:
        df_filtered = df
    
    # System status header
    col1, col2, col3, col4, col5 = st.columns(5)
    
    status = dashboard.get_system_status()
    
    # Status indicator
    status_color = {
        "healthy": "green",
        "degraded": "yellow",
        "critical": "red",
        "error": "red"
    }[status.get("health", "error")]
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <span class="status-indicator status-{status_color}"></span>
            <h4>System Status</h4>
            <h2>{status.get("health", "unknown").upper()}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.metric(
            "Total Evaluations",
            f"{status.get('total_evaluations', 0):,}",
            delta=f"{status.get('evaluations_last_hour', 0)} last hour"
        )
    
    with col3:
        st.metric(
            "Error Rate",
            f"{status.get('error_rate', 0):.1f}%",
            delta="-2.3%" if status.get('error_rate', 0) < 5 else "+1.2%"
        )
    
    with col4:
        st.metric(
            "Active Models",
            status.get('active_models', 0)
        )
    
    with col5:
        last_eval = status.get('last_evaluation')
        if last_eval:
            time_ago = (datetime.now() - last_eval.to_pydatetime()).total_seconds() / 60
            st.metric(
                "Last Evaluation",
                f"{time_ago:.0f} min ago" if time_ago < 60 else f"{time_ago/60:.1f} hours ago"
            )
        else:
            st.metric("Last Evaluation", "N/A")
    
    # Anomaly alerts
    anomalies = dashboard.detect_anomalies(df_filtered)
    if anomalies:
        st.header("⚠️ Alerts")
        for anomaly in anomalies[:5]:  # Show top 5
            severity_class = f"alert-{anomaly['severity']}"
            st.markdown(f"""
            <div class="alert-box {severity_class}">
                <strong>{anomaly['type'].replace('_', ' ').title()}</strong>: {anomaly['message']}
            </div>
            """, unsafe_allow_html=True)
    
    # Main monitoring charts
    st.header("📊 Real-Time Metrics")
    
    if not df_filtered.empty:
        # Real-time activity chart
        activity_chart = dashboard.create_real_time_chart(df_filtered)
        st.plotly_chart(activity_chart, use_container_width=True)
        
        # Model performance gauges
        st.subheader("Model Performance")
        gauges = dashboard.create_model_performance_gauges(df_filtered)
        
        gauge_cols = st.columns(len(gauges))
        for i, (model, gauge) in enumerate(gauges.items()):
            with gauge_cols[i]:
                st.plotly_chart(gauge, use_container_width=True)
        
        # Additional metrics row
        col1, col2 = st.columns(2)
        
        with col1:
            # Language heatmap
            heatmap = dashboard.create_language_heatmap(df_filtered)
            st.plotly_chart(heatmap, use_container_width=True)
        
        with col2:
            # Risk distribution
            risk_chart = dashboard.create_risk_distribution_chart(df_filtered)
            st.plotly_chart(risk_chart, use_container_width=True)
        
        # Recent evaluations table
        st.header("📋 Recent Evaluations")
        
        # Prepare display data
        display_df = df_filtered[['timestamp', 'model', 'language', 'domain', 
                                 'alignment_score', 'risk_level', 'latency_ms']].head(20)
        display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        display_df['latency_ms'] = display_df['latency_ms'].round(0).astype(int)
        
        # Style the dataframe
        def color_risk_level(val):
            colors = {
                'low': 'background-color: #d4edda',
                'medium': 'background-color: #fff3cd',
                'high': 'background-color: #f8d7da',
                'critical': 'background-color: #f5c6cb'
            }
            return colors.get(val, '')
        
        styled_df = display_df.style.applymap(color_risk_level, subset=['risk_level'])
        st.dataframe(styled_df, use_container_width=True)
        
    else:
        st.warning("No evaluation data available for the selected time range and filters.")
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

# Additional monitoring endpoints
def create_monitoring_api():
    """Create API endpoints for monitoring data"""
    from fastapi import FastAPI, WebSocket
    from fastapi.responses import StreamingResponse
    import json
    
    app = FastAPI()
    
    @app.websocket("/ws/metrics")
    async def websocket_metrics(websocket: WebSocket):
        """WebSocket endpoint for real-time metrics"""
        await websocket.accept()
        dashboard = MonitoringDashboard()
        
        try:
            while True:
                # Get latest metrics
                status = dashboard.get_system_status()
                df = dashboard.get_recent_evaluations(hours=1)
                
                metrics = {
                    "timestamp": datetime.now().isoformat(),
                    "status": status,
                    "recent_count": len(df),
                    "avg_score": df['alignment_score'].mean() if not df.empty else 0,
                    "models": df['model'].value_counts().to_dict() if not df.empty else {}
                }
                
                await websocket.send_json(metrics)
                await asyncio.sleep(5)
                
        except Exception as e:
            print(f"WebSocket error: {e}")
        finally:
            await websocket.close()
    
    @app.get("/api/monitoring/stream")
    async def stream_metrics():
        """Server-sent events endpoint for metrics streaming"""
        async def generate():
            dashboard = MonitoringDashboard()
            
            while True:
                status = dashboard.get_system_status()
                yield f"data: {json.dumps(status)}\n\n"
                await asyncio.sleep(5)
        
        return StreamingResponse(generate(), media_type="text/event-stream")
    
    return app

# Main entry point
if __name__ == "__main__":
    render_dashboard()