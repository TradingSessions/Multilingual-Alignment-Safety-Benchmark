# comparative_analyzer.py - Evaluation results comparative analysis tool

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from pathlib import Path
import sqlite3
import logging
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ComparisonResult:
    """Results of comparative analysis"""
    comparison_id: str
    timestamp: datetime
    datasets: List[str]
    models: List[str]
    languages: List[str]
    domains: List[str]
    metrics: Dict[str, Any]
    statistical_tests: Dict[str, Any]
    insights: List[str]
    visualizations: List[str]

class ComparativeAnalyzer:
    """Comprehensive tool for comparing evaluation results across models, languages, and domains"""
    
    def __init__(self, data_path: str = "./data", output_dir: str = "./comparisons"):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Statistical test configurations
        self.stat_tests = {
            "normality": stats.normaltest,
            "anova": stats.f_oneway,
            "kruskal": stats.kruskal,
            "ttest": stats.ttest_ind,
            "mannwhitney": stats.mannwhitneyu,
            "correlation": stats.pearsonr,
            "chi2": stats.chi2_contingency
        }
        
        # Visualization settings
        self.color_palette = sns.color_palette("husl", 12)
        try:
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            # Fallback to default style if seaborn style not available
            plt.style.use('default')
    
    def load_evaluation_data(self, 
                           dataset_ids: Optional[List[str]] = None,
                           date_range: Optional[Tuple[datetime, datetime]] = None) -> pd.DataFrame:
        """Load evaluation data from database"""
        db_path = self.data_path / "masb_alt.db"
        conn = sqlite3.connect(db_path)
        
        query = """
        SELECT 
            e.*,
            r.model,
            r.latency_ms,
            r.token_count,
            p.language,
            p.domain,
            p.risk_level as prompt_risk_level,
            d.name as dataset_name
        FROM evaluations e
        JOIN responses r ON e.response_id = r.response_id
        JOIN prompts p ON r.prompt_id = p.prompt_id
        LEFT JOIN datasets d ON p.dataset_id = d.dataset_id
        WHERE 1=1
        """
        
        params = []
        
        if dataset_ids:
            placeholders = ','.join(['?' for _ in dataset_ids])
            query += f" AND p.dataset_id IN ({placeholders})"
            params.extend(dataset_ids)
        
        if date_range:
            query += " AND e.timestamp BETWEEN ? AND ?"
            params.extend([date_range[0].isoformat(), date_range[1].isoformat()])
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        # Parse JSON fields
        df['risk_flags'] = df['risk_flags'].apply(lambda x: json.loads(x) if x else {})
        
        # Convert timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    
    def compare_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compare performance across different models"""
        results = {
            "summary": {},
            "statistical_tests": {},
            "pairwise_comparisons": {}
        }
        
        models = df['model'].unique()
        
        # Summary statistics per model
        for model in models:
            model_data = df[df['model'] == model]
            results["summary"][model] = {
                "n_evaluations": len(model_data),
                "avg_alignment_score": model_data['alignment_score'].mean(),
                "std_alignment_score": model_data['alignment_score'].std(),
                "median_alignment_score": model_data['alignment_score'].median(),
                "avg_latency_ms": model_data['latency_ms'].mean(),
                "error_rate": (model_data['alignment_score'] < 3).sum() / len(model_data),
                "high_risk_rate": (model_data['risk_level'].isin(['high', 'critical'])).sum() / len(model_data),
                "confidence_score": model_data['confidence_score'].mean()
            }
        
        # Statistical tests
        if len(models) > 2:
            # ANOVA for multiple models
            model_scores = [df[df['model'] == m]['alignment_score'].values for m in models]
            f_stat, p_value = stats.f_oneway(*model_scores)
            results["statistical_tests"]["anova"] = {
                "f_statistic": f_stat,
                "p_value": p_value,
                "significant": p_value < 0.05
            }
            
            # Kruskal-Wallis (non-parametric alternative)
            h_stat, p_value = stats.kruskal(*model_scores)
            results["statistical_tests"]["kruskal_wallis"] = {
                "h_statistic": h_stat,
                "p_value": p_value,
                "significant": p_value < 0.05
            }
        
        # Pairwise comparisons
        for i, model1 in enumerate(models):
            for model2 in models[i+1:]:
                scores1 = df[df['model'] == model1]['alignment_score'].values
                scores2 = df[df['model'] == model2]['alignment_score'].values
                
                # T-test
                t_stat, p_value = stats.ttest_ind(scores1, scores2)
                
                # Mann-Whitney U test
                u_stat, p_value_mw = stats.mannwhitneyu(scores1, scores2, alternative='two-sided')
                
                # Effect size (Cohen's d)
                cohens_d = (scores1.mean() - scores2.mean()) / np.sqrt((scores1.std()**2 + scores2.std()**2) / 2)
                
                results["pairwise_comparisons"][f"{model1}_vs_{model2}"] = {
                    "t_statistic": t_stat,
                    "t_test_p_value": p_value,
                    "mann_whitney_u": u_stat,
                    "mann_whitney_p_value": p_value_mw,
                    "cohens_d": cohens_d,
                    "effect_size": self._interpret_effect_size(cohens_d)
                }
        
        return results
    
    def compare_languages(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compare performance across different languages"""
        results = {
            "summary": {},
            "by_domain": {},
            "statistical_tests": {}
        }
        
        languages = df['language'].unique()
        
        # Overall summary
        for lang in languages:
            lang_data = df[df['language'] == lang]
            results["summary"][lang] = {
                "n_evaluations": len(lang_data),
                "avg_alignment_score": lang_data['alignment_score'].mean(),
                "std_alignment_score": lang_data['alignment_score'].std(),
                "domains_covered": lang_data['domain'].nunique(),
                "models_tested": lang_data['model'].nunique()
            }
        
        # By domain analysis
        for domain in df['domain'].unique():
            domain_data = df[df['domain'] == domain]
            results["by_domain"][domain] = {}
            
            for lang in languages:
                lang_domain_data = domain_data[domain_data['language'] == lang]
                if len(lang_domain_data) > 0:
                    results["by_domain"][domain][lang] = {
                        "avg_score": lang_domain_data['alignment_score'].mean(),
                        "n_samples": len(lang_domain_data)
                    }
        
        # Statistical tests for language differences
        if len(languages) > 1:
            lang_scores = [df[df['language'] == lang]['alignment_score'].values for lang in languages]
            
            # Check if scores are significantly different across languages
            h_stat, p_value = stats.kruskal(*lang_scores)
            results["statistical_tests"]["language_differences"] = {
                "kruskal_wallis_h": h_stat,
                "p_value": p_value,
                "significant": p_value < 0.05
            }
        
        return results
    
    def compare_domains(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compare performance across different domains"""
        results = {
            "summary": {},
            "risk_analysis": {},
            "model_performance": {}
        }
        
        domains = df['domain'].unique()
        
        # Domain summaries
        for domain in domains:
            domain_data = df[df['domain'] == domain]
            
            # Risk flag analysis
            risk_flags_counts = {}
            for _, row in domain_data.iterrows():
                for flag, value in row['risk_flags'].items():
                    if value:
                        risk_flags_counts[flag] = risk_flags_counts.get(flag, 0) + 1
            
            results["summary"][domain] = {
                "n_evaluations": len(domain_data),
                "avg_alignment_score": domain_data['alignment_score'].mean(),
                "high_risk_percentage": (domain_data['risk_level'].isin(['high', 'critical'])).sum() / len(domain_data) * 100,
                "most_common_risks": sorted(risk_flags_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            }
        
        # Model performance by domain
        for model in df['model'].unique():
            results["model_performance"][model] = {}
            for domain in domains:
                model_domain_data = df[(df['model'] == model) & (df['domain'] == domain)]
                if len(model_domain_data) > 0:
                    results["model_performance"][model][domain] = {
                        "avg_score": model_domain_data['alignment_score'].mean(),
                        "n_samples": len(model_domain_data)
                    }
        
        return results
    
    def temporal_analysis(self, df: pd.DataFrame, time_window: str = 'D') -> Dict[str, Any]:
        """Analyze trends over time"""
        results = {
            "trends": {},
            "change_points": {},
            "seasonality": {}
        }
        
        # Resample by time window
        df_time = df.set_index('timestamp')
        
        # Overall trend
        overall_trend = df_time['alignment_score'].resample(time_window).agg(['mean', 'std', 'count'])
        results["trends"]["overall"] = overall_trend.to_dict()
        
        # Trends by model
        for model in df['model'].unique():
            model_data = df_time[df_time['model'] == model]
            model_trend = model_data['alignment_score'].resample(time_window).mean()
            results["trends"][model] = model_trend.to_dict()
        
        # Detect significant changes
        scores = overall_trend['mean'].values
        if len(scores) > 10:
            # Simple change point detection using rolling statistics
            window_size = min(5, len(scores) // 3)
            rolling_mean = pd.Series(scores).rolling(window=window_size).mean()
            rolling_std = pd.Series(scores).rolling(window=window_size).std()
            
            # Identify points where score deviates significantly
            z_scores = np.abs((scores - rolling_mean) / rolling_std)
            change_points = np.where(z_scores > 2)[0]
            
            results["change_points"]["indices"] = change_points.tolist()
            results["change_points"]["timestamps"] = overall_trend.index[change_points].tolist()
        
        return results
    
    def cross_factor_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze interactions between factors (model, language, domain)"""
        results = {
            "interactions": {},
            "best_combinations": [],
            "worst_combinations": []
        }
        
        # Create combination groups
        df['combination'] = df['model'] + '_' + df['language'] + '_' + df['domain']
        
        # Analyze each combination
        combination_stats = df.groupby('combination').agg({
            'alignment_score': ['mean', 'std', 'count'],
            'latency_ms': 'mean',
            'confidence_score': 'mean'
        }).round(3)
        
        # Flatten column names
        combination_stats.columns = ['_'.join(col).strip() for col in combination_stats.columns]
        combination_stats = combination_stats.reset_index()
        
        # Extract components
        combination_stats[['model', 'language', 'domain']] = combination_stats['combination'].str.split('_', n=2, expand=True)
        
        # Sort by performance
        combination_stats = combination_stats.sort_values('alignment_score_mean', ascending=False)
        
        # Best and worst combinations
        min_samples = 5
        valid_combinations = combination_stats[combination_stats['alignment_score_count'] >= min_samples]
        
        if len(valid_combinations) > 0:
            results["best_combinations"] = valid_combinations.head(10).to_dict('records')
            results["worst_combinations"] = valid_combinations.tail(10).to_dict('records')
        
        # Interaction effects using factorial ANOVA
        if len(df) > 100:  # Need sufficient data
            try:
                import statsmodels.api as sm
                from statsmodels.formula.api import ols
                
                # Fit factorial model
                model = ols('alignment_score ~ model * language * domain', data=df).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                
                results["interactions"]["anova"] = anova_table.to_dict()
            except Exception as e:
                logger.warning(f"Could not perform factorial ANOVA: {e}")
        
        return results
    
    def generate_insights(self, comparison_results: Dict[str, Any]) -> List[str]:
        """Generate actionable insights from comparison results"""
        insights = []
        
        # Model insights
        if "model_comparison" in comparison_results:
            model_data = comparison_results["model_comparison"]["summary"]
            
            # Find best performing model
            best_model = max(model_data.items(), key=lambda x: x[1]['avg_alignment_score'])[0]
            insights.append(f"Best performing model: {best_model} with average alignment score of {model_data[best_model]['avg_alignment_score']:.3f}")
            
            # Check for significant differences
            if comparison_results["model_comparison"]["statistical_tests"].get("anova", {}).get("significant", False):
                insights.append("Statistically significant differences found between models (p < 0.05)")
            
            # Latency vs performance trade-off
            for model, stats in model_data.items():
                if stats['avg_latency_ms'] < 500 and stats['avg_alignment_score'] > 4.0:
                    insights.append(f"{model} offers excellent balance of speed (<500ms) and quality (>4.0 score)")
        
        # Language insights
        if "language_comparison" in comparison_results:
            lang_data = comparison_results["language_comparison"]["summary"]
            
            # Language performance gaps
            scores = {lang: data['avg_alignment_score'] for lang, data in lang_data.items()}
            best_lang = max(scores.items(), key=lambda x: x[1])
            worst_lang = min(scores.items(), key=lambda x: x[1])
            
            gap = best_lang[1] - worst_lang[1]
            if gap > 0.5:
                insights.append(f"Significant language gap detected: {best_lang[0]} ({best_lang[1]:.3f}) vs {worst_lang[0]} ({worst_lang[1]:.3f})")
        
        # Domain insights
        if "domain_comparison" in comparison_results:
            domain_data = comparison_results["domain_comparison"]["summary"]
            
            # High-risk domains
            for domain, stats in domain_data.items():
                if stats['high_risk_percentage'] > 20:
                    insights.append(f"{domain} domain shows high risk rate ({stats['high_risk_percentage']:.1f}%)")
                
                # Common risks
                if stats['most_common_risks']:
                    top_risk = stats['most_common_risks'][0]
                    insights.append(f"Most common risk in {domain}: {top_risk[0]} ({top_risk[1]} occurrences)")
        
        # Cross-factor insights
        if "cross_factor" in comparison_results:
            best = comparison_results["cross_factor"].get("best_combinations", [])
            worst = comparison_results["cross_factor"].get("worst_combinations", [])
            
            if best:
                top_combo = best[0]
                insights.append(f"Best combination: {top_combo['model']} + {top_combo['language']} + {top_combo['domain']} (score: {top_combo['alignment_score_mean']:.3f})")
            
            if worst:
                bottom_combo = worst[0]
                insights.append(f"Needs improvement: {bottom_combo['model']} + {bottom_combo['language']} + {bottom_combo['domain']} (score: {bottom_combo['alignment_score_mean']:.3f})")
        
        return insights
    
    def create_comparison_visualizations(self, df: pd.DataFrame, comparison_results: Dict[str, Any]) -> List[str]:
        """Create comprehensive comparison visualizations"""
        viz_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Model comparison radar chart
        if "model_comparison" in comparison_results:
            fig = self._create_model_radar_chart(comparison_results["model_comparison"]["summary"])
            filename = self.output_dir / f"model_radar_{timestamp}.png"
            fig.write_image(str(filename))
            viz_files.append(str(filename))
        
        # 2. Language-Domain heatmap
        fig = self._create_language_domain_heatmap(df)
        filename = self.output_dir / f"language_domain_heatmap_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        viz_files.append(str(filename))
        
        # 3. Distribution comparison
        fig = self._create_distribution_comparison(df)
        filename = self.output_dir / f"distribution_comparison_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        viz_files.append(str(filename))
        
        # 4. Temporal trends
        if "temporal" in comparison_results:
            fig = self._create_temporal_trends(comparison_results["temporal"])
            filename = self.output_dir / f"temporal_trends_{timestamp}.png"
            fig.write_image(str(filename))
            viz_files.append(str(filename))
        
        return viz_files
    
    def _create_model_radar_chart(self, model_summary: Dict) -> go.Figure:
        """Create radar chart comparing models across metrics"""
        models = list(model_summary.keys())
        
        # Define metrics for radar chart
        metrics = ['avg_alignment_score', 'confidence_score', 'latency_efficiency', 'reliability', 'consistency']
        
        fig = go.Figure()
        
        for model in models:
            stats = model_summary[model]
            
            # Calculate derived metrics
            latency_efficiency = 1 - min(stats['avg_latency_ms'] / 5000, 1)  # Normalize latency
            reliability = 1 - stats['error_rate']
            consistency = 1 - (stats['std_alignment_score'] / 5)  # Normalize std dev
            
            values = [
                stats['avg_alignment_score'] / 5,  # Normalize to 0-1
                stats['confidence_score'],
                latency_efficiency,
                reliability,
                consistency
            ]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name=model
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Model Performance Comparison"
        )
        
        return fig
    
    def _create_language_domain_heatmap(self, df: pd.DataFrame) -> plt.Figure:
        """Create heatmap of performance by language and domain"""
        pivot_table = df.pivot_table(
            values='alignment_score',
            index='language',
            columns='domain',
            aggfunc='mean'
        )
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            pivot_table,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            center=3,
            vmin=1,
            vmax=5,
            cbar_kws={'label': 'Average Alignment Score'}
        )
        plt.title('Performance Heatmap: Language vs Domain')
        plt.tight_layout()
        
        return plt.gcf()
    
    def _create_distribution_comparison(self, df: pd.DataFrame) -> plt.Figure:
        """Create distribution comparison plots"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Score distribution by model
        ax = axes[0, 0]
        for model in df['model'].unique():
            model_scores = df[df['model'] == model]['alignment_score']
            ax.hist(model_scores, alpha=0.6, label=model, bins=20)
        ax.set_xlabel('Alignment Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Score Distribution by Model')
        ax.legend()
        
        # Box plot by language
        ax = axes[0, 1]
        df.boxplot(column='alignment_score', by='language', ax=ax)
        ax.set_title('Score Distribution by Language')
        ax.set_xlabel('Language')
        ax.set_ylabel('Alignment Score')
        
        # Violin plot by domain
        ax = axes[1, 0]
        domains = df['domain'].unique()
        positions = range(len(domains))
        
        violin_data = [df[df['domain'] == d]['alignment_score'].values for d in domains]
        ax.violinplot(violin_data, positions=positions, showmeans=True)
        ax.set_xticks(positions)
        ax.set_xticklabels(domains, rotation=45)
        ax.set_xlabel('Domain')
        ax.set_ylabel('Alignment Score')
        ax.set_title('Score Distribution by Domain')
        
        # Risk level distribution
        ax = axes[1, 1]
        risk_counts = df['risk_level'].value_counts()
        ax.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%',
               colors=['green', 'yellow', 'orange', 'red'])
        ax.set_title('Risk Level Distribution')
        
        plt.tight_layout()
        return fig
    
    def _create_temporal_trends(self, temporal_results: Dict) -> go.Figure:
        """Create temporal trend visualization"""
        fig = go.Figure()
        
        # Add traces for each model
        for key, values in temporal_results["trends"].items():
            if key != "overall" and isinstance(values, dict):
                timestamps = list(values.keys())
                scores = list(values.values())
                
                fig.add_trace(go.Scatter(
                    x=timestamps,
                    y=scores,
                    mode='lines+markers',
                    name=key
                ))
        
        # Add change points if available
        if "change_points" in temporal_results and temporal_results["change_points"]["timestamps"]:
            for cp_time in temporal_results["change_points"]["timestamps"]:
                fig.add_vline(x=cp_time, line_dash="dash", line_color="red",
                             annotation_text="Change Point")
        
        fig.update_layout(
            title="Alignment Score Trends Over Time",
            xaxis_title="Time",
            yaxis_title="Average Alignment Score",
            hovermode='x unified'
        )
        
        return fig
    
    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        if abs(d) < 0.2:
            return "negligible"
        elif abs(d) < 0.5:
            return "small"
        elif abs(d) < 0.8:
            return "medium"
        else:
            return "large"
    
    def run_full_comparison(self, 
                           dataset_ids: Optional[List[str]] = None,
                           save_report: bool = True) -> ComparisonResult:
        """Run complete comparative analysis"""
        comparison_id = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Starting comparative analysis: {comparison_id}")
        
        # Load data
        df = self.load_evaluation_data(dataset_ids)
        
        if df.empty:
            logger.warning("No data available for comparison")
            return None
        
        # Run comparisons
        results = {
            "model_comparison": self.compare_models(df),
            "language_comparison": self.compare_languages(df),
            "domain_comparison": self.compare_domains(df),
            "temporal": self.temporal_analysis(df),
            "cross_factor": self.cross_factor_analysis(df)
        }
        
        # Generate insights
        insights = self.generate_insights(results)
        
        # Create visualizations
        viz_files = self.create_comparison_visualizations(df, results)
        
        # Create result object
        comparison_result = ComparisonResult(
            comparison_id=comparison_id,
            timestamp=datetime.now(),
            datasets=dataset_ids or ["all"],
            models=df['model'].unique().tolist(),
            languages=df['language'].unique().tolist(),
            domains=df['domain'].unique().tolist(),
            metrics=results,
            statistical_tests={
                "n_samples": len(df),
                "date_range": [df['timestamp'].min(), df['timestamp'].max()]
            },
            insights=insights,
            visualizations=viz_files
        )
        
        if save_report:
            self._save_comparison_report(comparison_result)
        
        logger.info(f"Comparative analysis completed: {comparison_id}")
        
        return comparison_result
    
    def _save_comparison_report(self, result: ComparisonResult):
        """Save comparison report to file"""
        report_file = self.output_dir / f"{result.comparison_id}.json"
        
        # Convert to dict for serialization
        report_data = {
            "comparison_id": result.comparison_id,
            "timestamp": result.timestamp.isoformat(),
            "datasets": result.datasets,
            "models": result.models,
            "languages": result.languages,
            "domains": result.domains,
            "metrics": result.metrics,
            "statistical_tests": result.statistical_tests,
            "insights": result.insights,
            "visualizations": result.visualizations
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        logger.info(f"Comparison report saved to {report_file}")

# Command-line interface
def main():
    """CLI for comparative analyzer"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MASB-Alt Comparative Analysis Tool")
    parser.add_argument("--datasets", nargs="+", help="Dataset IDs to compare")
    parser.add_argument("--output-dir", default="./comparisons", help="Output directory")
    parser.add_argument("--no-viz", action="store_true", help="Skip visualizations")
    
    args = parser.parse_args()
    
    analyzer = ComparativeAnalyzer(output_dir=args.output_dir)
    
    print("Running comparative analysis...")
    result = analyzer.run_full_comparison(dataset_ids=args.datasets)
    
    if result:
        print(f"\nAnalysis completed: {result.comparison_id}")
        print(f"Models compared: {', '.join(result.models)}")
        print(f"Languages: {', '.join(result.languages)}")
        print(f"Domains: {', '.join(result.domains)}")
        
        print("\nKey Insights:")
        for i, insight in enumerate(result.insights, 1):
            print(f"{i}. {insight}")
        
        print(f"\nFull report saved to: {args.output_dir}/{result.comparison_id}.json")
        print(f"Visualizations: {len(result.visualizations)} files generated")

if __name__ == "__main__":
    main()