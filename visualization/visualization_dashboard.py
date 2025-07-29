# visualization_dashboard.py - Interactive visualization dashboard for MASB-Alt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import streamlit as st
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisualizationDashboard:
    """Interactive dashboard for MASB-Alt results visualization"""
    
    def __init__(self, data_path: str = "./data"):
        self.data_path = Path(data_path)
        self.db_path = self.data_path / "masb_alt.db"
        self.conn = None
        self._connect_db()
        
    def _connect_db(self):
        """Connect to the database"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            logger.info("Connected to database")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def load_evaluation_data(self) -> pd.DataFrame:
        """Load evaluation data with all relationships"""
        query = """
        SELECT 
            e.evaluation_id,
            e.alignment_score,
            e.risk_level,
            e.risk_flags,
            e.confidence_score,
            e.evaluator_id,
            e.timestamp as eval_timestamp,
            r.response_id,
            r.model,
            r.latency_ms,
            r.token_count,
            r.error as response_error,
            p.prompt_id,
            p.language,
            p.domain,
            p.risk_level as prompt_risk_level,
            p.tags
        FROM evaluations e
        JOIN responses r ON e.response_id = r.response_id
        JOIN prompts p ON r.prompt_id = p.prompt_id
        """
        
        df = pd.read_sql_query(query, self.conn)
        
        # Parse JSON fields
        df['risk_flags'] = df['risk_flags'].apply(lambda x: json.loads(x) if x else {})
        df['tags'] = df['tags'].apply(lambda x: json.loads(x) if x else [])
        
        # Convert timestamp
        df['eval_timestamp'] = pd.to_datetime(df['eval_timestamp'])
        
        return df
    
    def create_alignment_score_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Create alignment score distribution chart"""
        fig = px.histogram(
            df, 
            x='alignment_score',
            nbins=5,
            title='Distribution of Alignment Scores',
            labels={'alignment_score': 'Alignment Score', 'count': 'Number of Evaluations'},
            color_discrete_sequence=['#2E86AB']
        )
        
        fig.update_layout(
            xaxis_title="Alignment Score (1-5)",
            yaxis_title="Count",
            showlegend=False,
            bargap=0.1
        )
        
        # Add average line
        avg_score = df['alignment_score'].mean()
        fig.add_vline(
            x=avg_score, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Average: {avg_score:.2f}"
        )
        
        return fig
    
    def create_language_performance_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create language performance comparison chart"""
        lang_stats = df.groupby('language').agg({
            'alignment_score': ['mean', 'std', 'count'],
            'risk_level': lambda x: (x == 'high').sum() + (x == 'critical').sum()
        }).round(2)
        
        lang_stats.columns = ['avg_score', 'std_dev', 'count', 'high_risk_count']
        lang_stats = lang_stats.reset_index()
        
        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"secondary_y": True}]]
        )
        
        # Add bar chart for average scores
        fig.add_trace(
            go.Bar(
                x=lang_stats['language'],
                y=lang_stats['avg_score'],
                name='Average Score',
                error_y=dict(type='data', array=lang_stats['std_dev']),
                marker_color='lightblue',
                yaxis='y'
            ),
            secondary_y=False
        )
        
        # Add line chart for evaluation count
        fig.add_trace(
            go.Scatter(
                x=lang_stats['language'],
                y=lang_stats['count'],
                name='Evaluation Count',
                mode='lines+markers',
                line=dict(color='orange', width=2),
                yaxis='y2'
            ),
            secondary_y=True
        )
        
        fig.update_xaxis(title_text="Language")
        fig.update_yaxis(title_text="Average Alignment Score", secondary_y=False, range=[0, 5])
        fig.update_yaxis(title_text="Number of Evaluations", secondary_y=True)
        
        fig.update_layout(
            title="Alignment Performance by Language",
            hovermode='x unified'
        )
        
        return fig
    
    def create_domain_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create domain-language alignment heatmap"""
        pivot_table = df.pivot_table(
            values='alignment_score',
            index='domain',
            columns='language',
            aggfunc='mean'
        )
        
        fig = px.imshow(
            pivot_table,
            labels=dict(x="Language", y="Domain", color="Avg Score"),
            title="Alignment Scores: Domain vs Language Heatmap",
            color_continuous_scale="RdYlGn",
            aspect="auto"
        )
        
        # Add text annotations
        for i in range(len(pivot_table.index)):
            for j in range(len(pivot_table.columns)):
                value = pivot_table.iloc[i, j]
                if not np.isnan(value):
                    fig.add_annotation(
                        x=j, y=i,
                        text=f"{value:.2f}",
                        showarrow=False,
                        font=dict(size=10)
                    )
        
        return fig
    
    def create_model_comparison_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create model performance comparison"""
        model_stats = df.groupby('model').agg({
            'alignment_score': ['mean', 'std'],
            'latency_ms': 'mean',
            'token_count': 'mean'
        }).round(2)
        
        model_stats.columns = ['avg_score', 'std_dev', 'avg_latency', 'avg_tokens']
        model_stats = model_stats.reset_index()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Average Alignment Score', 'Response Latency', 
                          'Token Usage', 'Score Distribution'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "box"}]]
        )
        
        # Average scores
        fig.add_trace(
            go.Bar(
                x=model_stats['model'],
                y=model_stats['avg_score'],
                error_y=dict(type='data', array=model_stats['std_dev']),
                name='Alignment Score',
                marker_color='skyblue'
            ),
            row=1, col=1
        )
        
        # Latency
        fig.add_trace(
            go.Bar(
                x=model_stats['model'],
                y=model_stats['avg_latency'],
                name='Latency (ms)',
                marker_color='lightcoral'
            ),
            row=1, col=2
        )
        
        # Token usage
        fig.add_trace(
            go.Bar(
                x=model_stats['model'],
                y=model_stats['avg_tokens'],
                name='Tokens',
                marker_color='lightgreen'
            ),
            row=2, col=1
        )
        
        # Score distribution box plot
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            fig.add_trace(
                go.Box(
                    y=model_data['alignment_score'],
                    name=model,
                    boxpoints='outliers'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Model Performance Comparison",
            showlegend=False,
            height=800
        )
        
        # Update y-axis ranges
        fig.update_yaxes(range=[0, 5], row=1, col=1)
        
        return fig
    
    def create_risk_analysis_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create risk analysis visualization"""
        # Parse risk flags and count occurrences
        risk_counts = {}
        for flags in df['risk_flags']:
            if isinstance(flags, dict):
                for risk, value in flags.items():
                    if value:
                        risk_counts[risk] = risk_counts.get(risk, 0) + 1
        
        # Create DataFrame for plotting
        risk_df = pd.DataFrame(
            list(risk_counts.items()),
            columns=['Risk Type', 'Count']
        ).sort_values('Count', ascending=True)
        
        # Create horizontal bar chart
        fig = px.bar(
            risk_df,
            x='Count',
            y='Risk Type',
            orientation='h',
            title='Frequency of Risk Flags',
            color='Count',
            color_continuous_scale='Reds'
        )
        
        fig.update_layout(
            xaxis_title="Number of Occurrences",
            yaxis_title="Risk Type",
            showlegend=False,
            height=max(400, len(risk_df) * 40)
        )
        
        return fig
    
    def create_temporal_analysis(self, df: pd.DataFrame) -> go.Figure:
        """Create temporal analysis of evaluations"""
        # Group by date
        df['eval_date'] = df['eval_timestamp'].dt.date
        daily_stats = df.groupby('eval_date').agg({
            'alignment_score': 'mean',
            'evaluation_id': 'count'
        }).reset_index()
        
        daily_stats.columns = ['date', 'avg_score', 'count']
        
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add score trend
        fig.add_trace(
            go.Scatter(
                x=daily_stats['date'],
                y=daily_stats['avg_score'],
                name='Average Score',
                mode='lines+markers',
                line=dict(color='blue', width=2)
            ),
            secondary_y=False
        )
        
        # Add evaluation count
        fig.add_trace(
            go.Bar(
                x=daily_stats['date'],
                y=daily_stats['count'],
                name='Evaluations',
                marker_color='lightgray',
                opacity=0.6
            ),
            secondary_y=True
        )
        
        fig.update_xaxis(title_text="Date")
        fig.update_yaxis(title_text="Average Alignment Score", secondary_y=False, range=[0, 5])
        fig.update_yaxis(title_text="Number of Evaluations", secondary_y=True)
        
        fig.update_layout(
            title="Alignment Scores Over Time",
            hovermode='x unified'
        )
        
        return fig
    
    def create_evaluator_performance(self, df: pd.DataFrame) -> go.Figure:
        """Analyze evaluator performance and consistency"""
        evaluator_stats = df.groupby('evaluator_id').agg({
            'alignment_score': ['mean', 'std', 'count'],
            'confidence_score': 'mean'
        }).round(2)
        
        evaluator_stats.columns = ['avg_score', 'std_dev', 'count', 'avg_confidence']
        evaluator_stats = evaluator_stats.reset_index()
        
        # Filter out evaluators with too few evaluations
        evaluator_stats = evaluator_stats[evaluator_stats['count'] >= 5]
        
        # Create scatter plot
        fig = px.scatter(
            evaluator_stats,
            x='avg_score',
            y='std_dev',
            size='count',
            color='avg_confidence',
            hover_data=['evaluator_id', 'count'],
            title='Evaluator Consistency Analysis',
            labels={
                'avg_score': 'Average Score Given',
                'std_dev': 'Standard Deviation (Consistency)',
                'avg_confidence': 'Average Confidence'
            },
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            xaxis_range=[0, 5],
            yaxis_title='Standard Deviation (Lower = More Consistent)'
        )
        
        return fig
    
    def generate_summary_metrics(self, df: pd.DataFrame) -> Dict:
        """Generate summary metrics for display"""
        total_evaluations = len(df)
        unique_prompts = df['prompt_id'].nunique()
        unique_models = df['model'].nunique()
        unique_languages = df['language'].nunique()
        unique_domains = df['domain'].nunique()
        
        avg_score = df['alignment_score'].mean()
        high_risk_pct = ((df['risk_level'].isin(['high', 'critical'])).sum() / total_evaluations) * 100
        
        # Best and worst performing language
        lang_scores = df.groupby('language')['alignment_score'].mean()
        best_lang = lang_scores.idxmax()
        worst_lang = lang_scores.idxmin()
        
        # Best and worst performing model
        model_scores = df.groupby('model')['alignment_score'].mean()
        best_model = model_scores.idxmax()
        worst_model = model_scores.idxmin()
        
        return {
            'total_evaluations': total_evaluations,
            'unique_prompts': unique_prompts,
            'unique_models': unique_models,
            'unique_languages': unique_languages,
            'unique_domains': unique_domains,
            'avg_score': round(avg_score, 2),
            'high_risk_pct': round(high_risk_pct, 1),
            'best_language': f"{best_lang} ({lang_scores[best_lang]:.2f})",
            'worst_language': f"{worst_lang} ({lang_scores[worst_lang]:.2f})",
            'best_model': f"{best_model} ({model_scores[best_model]:.2f})",
            'worst_model': f"{worst_model} ({model_scores[worst_model]:.2f})"
        }
    
    def export_visualizations(self, df: pd.DataFrame, output_dir: str = "./reports/visualizations"):
        """Export all visualizations as static images"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create all visualizations
        visualizations = {
            'alignment_distribution': self.create_alignment_score_distribution(df),
            'language_performance': self.create_language_performance_chart(df),
            'domain_heatmap': self.create_domain_heatmap(df),
            'model_comparison': self.create_model_comparison_chart(df),
            'risk_analysis': self.create_risk_analysis_chart(df),
            'temporal_analysis': self.create_temporal_analysis(df),
            'evaluator_performance': self.create_evaluator_performance(df)
        }
        
        # Export as HTML and PNG
        for name, fig in visualizations.items():
            # HTML (interactive)
            html_file = output_path / f"{name}_{timestamp}.html"
            fig.write_html(str(html_file))
            
            # PNG (static)
            png_file = output_path / f"{name}_{timestamp}.png"
            fig.write_image(str(png_file), width=1200, height=800)
        
        # Generate summary report
        summary_file = output_path / f"summary_metrics_{timestamp}.json"
        summary_metrics = self.generate_summary_metrics(df)
        with open(summary_file, 'w') as f:
            json.dump(summary_metrics, f, indent=2)
        
        logger.info(f"Exported visualizations to {output_path}")
        return str(output_path)

def create_streamlit_dashboard():
    """Create interactive Streamlit dashboard"""
    st.set_page_config(
        page_title="MASB-Alt Dashboard",
        page_icon="🌍",
        layout="wide"
    )
    
    st.title("🌍 Multilingual Alignment Safety Benchmark Dashboard")
    st.markdown("---")
    
    # Initialize dashboard
    dashboard = VisualizationDashboard()
    
    # Load data
    with st.spinner("Loading evaluation data..."):
        df = dashboard.load_evaluation_data()
    
    if df.empty:
        st.error("No evaluation data found. Please run evaluations first.")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    metrics = dashboard.generate_summary_metrics(df)
    
    with col1:
        st.metric("Total Evaluations", metrics['total_evaluations'])
        st.metric("Unique Prompts", metrics['unique_prompts'])
    
    with col2:
        st.metric("Average Score", f"{metrics['avg_score']}/5.0")
        st.metric("High Risk %", f"{metrics['high_risk_pct']}%")
    
    with col3:
        st.metric("Best Language", metrics['best_language'])
        st.metric("Worst Language", metrics['worst_language'])
    
    with col4:
        st.metric("Best Model", metrics['best_model'])
        st.metric("Worst Model", metrics['worst_model'])
    
    st.markdown("---")
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🌐 Language Analysis", 
        "🤖 Model Comparison",
        "⚠️ Risk Analysis",
        "📈 Temporal Trends"
    ])
    
    with tab1:
        st.subheader("Alignment Score Distribution")
        fig1 = dashboard.create_alignment_score_distribution(df)
        st.plotly_chart(fig1, use_container_width=True)
        
        st.subheader("Domain-Language Heatmap")
        fig2 = dashboard.create_domain_heatmap(df)
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab2:
        st.subheader("Language Performance Analysis")
        fig3 = dashboard.create_language_performance_chart(df)
        st.plotly_chart(fig3, use_container_width=True)
        
        # Language details table
        st.subheader("Detailed Language Statistics")
        lang_stats = df.groupby('language').agg({
            'alignment_score': ['mean', 'std', 'min', 'max', 'count']
        }).round(2)
        st.dataframe(lang_stats)
    
    with tab3:
        st.subheader("Model Performance Comparison")
        fig4 = dashboard.create_model_comparison_chart(df)
        st.plotly_chart(fig4, use_container_width=True)
    
    with tab4:
        st.subheader("Risk Flag Analysis")
        fig5 = dashboard.create_risk_analysis_chart(df)
        st.plotly_chart(fig5, use_container_width=True)
        
        # Risk details
        st.subheader("Risk Distribution by Language")
        risk_by_lang = df.groupby(['language', 'risk_level']).size().unstack(fill_value=0)
        st.dataframe(risk_by_lang)
    
    with tab5:
        st.subheader("Temporal Analysis")
        fig6 = dashboard.create_temporal_analysis(df)
        st.plotly_chart(fig6, use_container_width=True)
        
        st.subheader("Evaluator Performance")
        fig7 = dashboard.create_evaluator_performance(df)
        st.plotly_chart(fig7, use_container_width=True)
    
    # Export functionality
    st.markdown("---")
    if st.button("📥 Export All Visualizations"):
        with st.spinner("Exporting visualizations..."):
            export_path = dashboard.export_visualizations(df)
            st.success(f"Visualizations exported to: {export_path}")
    
    # Data explorer
    with st.expander("🔍 Explore Raw Data"):
        st.dataframe(df)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name=f"masb_alt_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# Command-line interface
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--export":
        # Export mode
        dashboard = VisualizationDashboard()
        df = dashboard.load_evaluation_data()
        
        if not df.empty:
            export_path = dashboard.export_visualizations(df)
            print(f"Visualizations exported to: {export_path}")
        else:
            print("No data to visualize")
    else:
        # Interactive mode
        print("Starting Streamlit dashboard...")
        print("Run: streamlit run visualization_dashboard.py")
        create_streamlit_dashboard()