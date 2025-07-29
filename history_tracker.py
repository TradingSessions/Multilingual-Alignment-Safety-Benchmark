# history_tracker.py - Evaluation history tracking and analysis system

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import logging
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HistorySnapshot:
    """Snapshot of evaluation metrics at a point in time"""
    snapshot_id: str
    timestamp: datetime
    period_start: datetime
    period_end: datetime
    total_evaluations: int
    unique_prompts: int
    unique_models: int
    avg_alignment_score: float
    risk_distribution: Dict[str, int]
    model_metrics: Dict[str, Dict[str, float]]
    language_metrics: Dict[str, Dict[str, float]]
    domain_metrics: Dict[str, Dict[str, float]]
    metadata: Dict[str, Any]

@dataclass
class TrendAnalysis:
    """Trend analysis results"""
    metric: str
    trend_direction: str  # 'improving', 'declining', 'stable'
    change_rate: float
    confidence: float
    forecast: List[float]
    change_points: List[datetime]
    seasonality: Optional[Dict[str, Any]]

@dataclass
class AnomalyRecord:
    """Record of detected anomalies"""
    anomaly_id: str
    timestamp: datetime
    anomaly_type: str  # 'score_drop', 'high_risk', 'error_spike', 'latency_spike'
    severity: str  # 'low', 'medium', 'high', 'critical'
    affected_model: Optional[str]
    affected_language: Optional[str]
    description: str
    metrics: Dict[str, Any]
    resolved: bool

class HistoryTracker:
    """Comprehensive evaluation history tracking system"""
    
    def __init__(self, data_path: str = "./data", history_path: str = "./history"):
        self.data_path = Path(data_path)
        self.history_path = Path(history_path)
        self.history_path.mkdir(exist_ok=True)
        
        # Initialize history database
        self.history_db = self.history_path / "evaluation_history.db"
        self._init_history_database()
        
        # Configuration
        self.snapshot_intervals = {
            "hourly": timedelta(hours=1),
            "daily": timedelta(days=1),
            "weekly": timedelta(days=7),
            "monthly": timedelta(days=30)
        }
        
        # Anomaly detection thresholds
        self.anomaly_thresholds = {
            "score_drop": 0.5,  # 0.5 point drop
            "error_rate": 0.2,  # 20% error rate
            "latency_spike": 2.0,  # 2x normal latency
            "risk_increase": 0.3  # 30% increase in high-risk
        }
    
    def _init_history_database(self):
        """Initialize history tracking database"""
        conn = sqlite3.connect(self.history_db)
        cursor = conn.cursor()
        
        # History snapshots table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS history_snapshots (
                snapshot_id TEXT PRIMARY KEY,
                timestamp TIMESTAMP,
                period_start TIMESTAMP,
                period_end TIMESTAMP,
                interval_type TEXT,
                total_evaluations INTEGER,
                unique_prompts INTEGER,
                unique_models INTEGER,
                avg_alignment_score REAL,
                risk_distribution TEXT,
                model_metrics TEXT,
                language_metrics TEXT,
                domain_metrics TEXT,
                metadata TEXT
            )
        ''')
        
        # Trend analysis table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trend_analyses (
                analysis_id TEXT PRIMARY KEY,
                timestamp TIMESTAMP,
                metric TEXT,
                entity_type TEXT,
                entity_name TEXT,
                trend_direction TEXT,
                change_rate REAL,
                confidence REAL,
                forecast TEXT,
                change_points TEXT,
                seasonality TEXT
            )
        ''')
        
        # Anomaly records table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS anomaly_records (
                anomaly_id TEXT PRIMARY KEY,
                timestamp TIMESTAMP,
                anomaly_type TEXT,
                severity TEXT,
                affected_model TEXT,
                affected_language TEXT,
                description TEXT,
                metrics TEXT,
                resolved INTEGER DEFAULT 0,
                resolution_time TIMESTAMP,
                resolution_notes TEXT
            )
        ''')
        
        # Evaluation checkpoints table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS evaluation_checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                timestamp TIMESTAMP,
                evaluation_count INTEGER,
                last_evaluation_id TEXT,
                metrics_summary TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info("History database initialized")
    
    def create_snapshot(self, interval_type: str = "daily") -> Optional[HistorySnapshot]:
        """Create a snapshot of current evaluation metrics"""
        # Connect to main database
        main_db = self.data_path / "masb_alt.db"
        if not main_db.exists():
            logger.error("Main database not found")
            return None
        
        conn = sqlite3.connect(main_db)
        
        # Determine time period
        period_end = datetime.now()
        period_start = period_end - self.snapshot_intervals[interval_type]
        
        # Get evaluations in period
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
        WHERE e.timestamp BETWEEN ? AND ?
        """
        
        df = pd.read_sql_query(query, conn, params=(
            period_start.isoformat(),
            period_end.isoformat()
        ))
        
        if df.empty:
            logger.info(f"No evaluations found for {interval_type} snapshot")
            conn.close()
            return None
        
        # Parse risk flags
        df['risk_flags'] = df['risk_flags'].apply(lambda x: json.loads(x) if x else {})
        
        # Calculate metrics
        snapshot = HistorySnapshot(
            snapshot_id=f"{interval_type}_{period_end.strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            period_start=period_start,
            period_end=period_end,
            total_evaluations=len(df),
            unique_prompts=df.groupby(['language', 'domain']).size().sum(),
            unique_models=df['model'].nunique(),
            avg_alignment_score=df['alignment_score'].mean(),
            risk_distribution=df['risk_level'].value_counts().to_dict(),
            model_metrics=self._calculate_model_metrics(df),
            language_metrics=self._calculate_language_metrics(df),
            domain_metrics=self._calculate_domain_metrics(df),
            metadata={
                "interval_type": interval_type,
                "data_quality": self._assess_data_quality(df)
            }
        )
        
        conn.close()
        
        # Save snapshot
        self._save_snapshot(snapshot)
        
        return snapshot
    
    def _calculate_model_metrics(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate metrics by model"""
        metrics = {}
        
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            
            metrics[model] = {
                "count": len(model_data),
                "avg_score": model_data['alignment_score'].mean(),
                "std_score": model_data['alignment_score'].std(),
                "error_rate": (model_data['response_error'].notna()).mean(),
                "avg_latency": model_data['latency_ms'].mean(),
                "high_risk_rate": (model_data['risk_level'].isin(['high', 'critical'])).mean()
            }
        
        return metrics
    
    def _calculate_language_metrics(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate metrics by language"""
        metrics = {}
        
        for language in df['language'].unique():
            lang_data = df[df['language'] == language]
            
            metrics[language] = {
                "count": len(lang_data),
                "avg_score": lang_data['alignment_score'].mean(),
                "domains_covered": lang_data['domain'].nunique(),
                "models_tested": lang_data['model'].nunique()
            }
        
        return metrics
    
    def _calculate_domain_metrics(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate metrics by domain"""
        metrics = {}
        
        for domain in df['domain'].unique():
            domain_data = df[df['domain'] == domain]
            
            # Count risk flags
            risk_counts = defaultdict(int)
            for _, row in domain_data.iterrows():
                for flag, value in row['risk_flags'].items():
                    if value:
                        risk_counts[flag] += 1
            
            metrics[domain] = {
                "count": len(domain_data),
                "avg_score": domain_data['alignment_score'].mean(),
                "languages_covered": domain_data['language'].nunique(),
                "top_risks": dict(sorted(risk_counts.items(), 
                                       key=lambda x: x[1], 
                                       reverse=True)[:3])
            }
        
        return metrics
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess quality of evaluation data"""
        return {
            "completeness": 1 - df.isnull().sum().sum() / (len(df) * len(df.columns)),
            "balance_score": self._calculate_balance_score(df),
            "coverage": {
                "languages": df['language'].nunique(),
                "domains": df['domain'].nunique(),
                "models": df['model'].nunique()
            }
        }
    
    def _calculate_balance_score(self, df: pd.DataFrame) -> float:
        """Calculate how balanced the evaluation distribution is"""
        # Check balance across models
        model_counts = df['model'].value_counts()
        model_balance = 1 - model_counts.std() / model_counts.mean() if len(model_counts) > 1 else 1
        
        # Check balance across languages
        lang_counts = df['language'].value_counts()
        lang_balance = 1 - lang_counts.std() / lang_counts.mean() if len(lang_counts) > 1 else 1
        
        return (model_balance + lang_balance) / 2
    
    def _save_snapshot(self, snapshot: HistorySnapshot):
        """Save snapshot to database"""
        conn = sqlite3.connect(self.history_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO history_snapshots
            (snapshot_id, timestamp, period_start, period_end, interval_type,
             total_evaluations, unique_prompts, unique_models, avg_alignment_score,
             risk_distribution, model_metrics, language_metrics, domain_metrics, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            snapshot.snapshot_id,
            snapshot.timestamp,
            snapshot.period_start,
            snapshot.period_end,
            snapshot.metadata.get("interval_type", "unknown"),
            snapshot.total_evaluations,
            snapshot.unique_prompts,
            snapshot.unique_models,
            snapshot.avg_alignment_score,
            json.dumps(snapshot.risk_distribution),
            json.dumps(snapshot.model_metrics),
            json.dumps(snapshot.language_metrics),
            json.dumps(snapshot.domain_metrics),
            json.dumps(snapshot.metadata)
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Saved snapshot: {snapshot.snapshot_id}")
    
    def analyze_trends(self, 
                      metric: str = "alignment_score",
                      entity_type: str = "overall",
                      entity_name: Optional[str] = None,
                      lookback_days: int = 30) -> Optional[TrendAnalysis]:
        """Analyze trends in evaluation metrics"""
        conn = sqlite3.connect(self.history_db)
        
        # Get historical snapshots
        query = """
        SELECT * FROM history_snapshots
        WHERE timestamp > datetime('now', '-' || ? || ' days')
        ORDER BY timestamp
        """
        
        df = pd.read_sql_query(query, conn, params=(lookback_days,))
        conn.close()
        
        if len(df) < 3:
            logger.warning("Not enough data points for trend analysis")
            return None
        
        # Extract metric time series
        if entity_type == "overall":
            if metric == "alignment_score":
                values = df['avg_alignment_score'].values
            else:
                values = df['total_evaluations'].values
        else:
            # Extract from JSON fields
            values = []
            for _, row in df.iterrows():
                if entity_type == "model":
                    metrics = json.loads(row['model_metrics'])
                elif entity_type == "language":
                    metrics = json.loads(row['language_metrics'])
                else:
                    metrics = json.loads(row['domain_metrics'])
                
                if entity_name in metrics and metric in metrics[entity_name]:
                    values.append(metrics[entity_name][metric])
                else:
                    values.append(np.nan)
            
            values = np.array(values)
            values = values[~np.isnan(values)]
        
        if len(values) < 3:
            return None
        
        # Calculate trend
        x = np.arange(len(values))
        z = np.polyfit(x, values, 1)
        slope = z[0]
        
        # Determine trend direction
        if abs(slope) < 0.01:
            trend_direction = "stable"
        elif slope > 0:
            trend_direction = "improving"
        else:
            trend_direction = "declining"
        
        # Calculate change rate
        change_rate = (values[-1] - values[0]) / values[0] if values[0] != 0 else 0
        
        # Simple forecast (linear extrapolation)
        future_x = np.arange(len(values), len(values) + 7)
        forecast = np.polyval(z, future_x).tolist()
        
        # Detect change points (simple method)
        change_points = self._detect_change_points(values, df['timestamp'].values)
        
        # Create trend analysis
        analysis = TrendAnalysis(
            metric=metric,
            trend_direction=trend_direction,
            change_rate=change_rate,
            confidence=self._calculate_trend_confidence(values),
            forecast=forecast,
            change_points=change_points,
            seasonality=None  # Could implement more sophisticated seasonality detection
        )
        
        # Save analysis
        self._save_trend_analysis(analysis, entity_type, entity_name)
        
        return analysis
    
    def _detect_change_points(self, values: np.ndarray, timestamps: np.ndarray) -> List[datetime]:
        """Detect significant changes in time series"""
        change_points = []
        
        if len(values) < 10:
            return change_points
        
        # Simple method: detect when value changes by more than 1 std dev
        mean = np.mean(values)
        std = np.std(values)
        
        for i in range(1, len(values)):
            if abs(values[i] - values[i-1]) > std:
                change_points.append(pd.to_datetime(timestamps[i]))
        
        return change_points
    
    def _calculate_trend_confidence(self, values: np.ndarray) -> float:
        """Calculate confidence in trend based on consistency"""
        if len(values) < 2:
            return 0.0
        
        # Calculate R-squared for linear fit
        x = np.arange(len(values))
        z = np.polyfit(x, values, 1)
        p = np.poly1d(z)
        
        yhat = p(x)
        ybar = np.mean(values)
        ssreg = np.sum((yhat - ybar)**2)
        sstot = np.sum((values - ybar)**2)
        
        r_squared = ssreg / sstot if sstot > 0 else 0
        
        return min(1.0, max(0.0, r_squared))
    
    def _save_trend_analysis(self, 
                           analysis: TrendAnalysis,
                           entity_type: str,
                           entity_name: Optional[str]):
        """Save trend analysis to database"""
        conn = sqlite3.connect(self.history_db)
        cursor = conn.cursor()
        
        analysis_id = hashlib.md5(
            f"{analysis.metric}_{entity_type}_{entity_name}_{datetime.now()}".encode()
        ).hexdigest()[:12]
        
        cursor.execute('''
            INSERT INTO trend_analyses
            (analysis_id, timestamp, metric, entity_type, entity_name,
             trend_direction, change_rate, confidence, forecast, change_points, seasonality)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            analysis_id,
            datetime.now(),
            analysis.metric,
            entity_type,
            entity_name or "overall",
            analysis.trend_direction,
            analysis.change_rate,
            analysis.confidence,
            json.dumps(analysis.forecast),
            json.dumps([cp.isoformat() for cp in analysis.change_points]),
            json.dumps(analysis.seasonality) if analysis.seasonality else None
        ))
        
        conn.commit()
        conn.close()
    
    def detect_anomalies(self, lookback_hours: int = 24) -> List[AnomalyRecord]:
        """Detect anomalies in recent evaluations"""
        anomalies = []
        
        # Get recent data
        main_db = self.data_path / "masb_alt.db"
        conn = sqlite3.connect(main_db)
        
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
        WHERE e.timestamp > datetime('now', '-' || ? || ' hours')
        """
        
        df = pd.read_sql_query(query, conn, params=(lookback_hours,))
        
        if df.empty:
            conn.close()
            return anomalies
        
        # Get historical baselines
        baseline_query = """
        SELECT 
            AVG(e.alignment_score) as baseline_score,
            AVG(r.latency_ms) as baseline_latency,
            AVG(CASE WHEN r.error IS NOT NULL THEN 1 ELSE 0 END) as baseline_error_rate
        FROM evaluations e
        JOIN responses r ON e.response_id = r.response_id
        WHERE e.timestamp BETWEEN datetime('now', '-30 days') AND datetime('now', '-' || ? || ' hours')
        """
        
        baseline = pd.read_sql_query(baseline_query, conn, params=(lookback_hours,))
        conn.close()
        
        if baseline.empty:
            return anomalies
        
        # Check for anomalies by model
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            
            # Score drop
            avg_score = model_data['alignment_score'].mean()
            if avg_score < baseline['baseline_score'].iloc[0] - self.anomaly_thresholds['score_drop']:
                anomalies.append(self._create_anomaly_record(
                    anomaly_type="score_drop",
                    severity="high",
                    affected_model=model,
                    description=f"Alignment score dropped to {avg_score:.2f}",
                    metrics={"current_score": avg_score, 
                            "baseline_score": baseline['baseline_score'].iloc[0]}
                ))
            
            # Error spike
            error_rate = (model_data['response_error'].notna()).mean()
            if error_rate > self.anomaly_thresholds['error_rate']:
                anomalies.append(self._create_anomaly_record(
                    anomaly_type="error_spike",
                    severity="critical" if error_rate > 0.5 else "high",
                    affected_model=model,
                    description=f"Error rate spiked to {error_rate:.1%}",
                    metrics={"error_rate": error_rate, "error_count": model_data['response_error'].notna().sum()}
                ))
            
            # Latency spike
            avg_latency = model_data['latency_ms'].mean()
            if avg_latency > baseline['baseline_latency'].iloc[0] * self.anomaly_thresholds['latency_spike']:
                anomalies.append(self._create_anomaly_record(
                    anomaly_type="latency_spike",
                    severity="medium",
                    affected_model=model,
                    description=f"Latency increased to {avg_latency:.0f}ms",
                    metrics={"current_latency": avg_latency,
                            "baseline_latency": baseline['baseline_latency'].iloc[0]}
                ))
        
        # Check for high-risk content spikes
        risk_rate = (df['risk_level'].isin(['high', 'critical'])).mean()
        if risk_rate > self.anomaly_thresholds['risk_increase']:
            anomalies.append(self._create_anomaly_record(
                anomaly_type="high_risk",
                severity="high",
                description=f"High-risk content rate at {risk_rate:.1%}",
                metrics={"risk_rate": risk_rate, 
                        "high_risk_count": (df['risk_level'].isin(['high', 'critical'])).sum()}
            ))
        
        # Save anomalies
        for anomaly in anomalies:
            self._save_anomaly(anomaly)
        
        return anomalies
    
    def _create_anomaly_record(self,
                              anomaly_type: str,
                              severity: str,
                              description: str,
                              metrics: Dict[str, Any],
                              affected_model: Optional[str] = None,
                              affected_language: Optional[str] = None) -> AnomalyRecord:
        """Create an anomaly record"""
        return AnomalyRecord(
            anomaly_id=hashlib.md5(
                f"{anomaly_type}_{affected_model}_{datetime.now()}".encode()
            ).hexdigest()[:12],
            timestamp=datetime.now(),
            anomaly_type=anomaly_type,
            severity=severity,
            affected_model=affected_model,
            affected_language=affected_language,
            description=description,
            metrics=metrics,
            resolved=False
        )
    
    def _save_anomaly(self, anomaly: AnomalyRecord):
        """Save anomaly to database"""
        conn = sqlite3.connect(self.history_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR IGNORE INTO anomaly_records
            (anomaly_id, timestamp, anomaly_type, severity, affected_model,
             affected_language, description, metrics, resolved)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            anomaly.anomaly_id,
            anomaly.timestamp,
            anomaly.anomaly_type,
            anomaly.severity,
            anomaly.affected_model,
            anomaly.affected_language,
            anomaly.description,
            json.dumps(anomaly.metrics),
            int(anomaly.resolved)
        ))
        
        conn.commit()
        conn.close()
    
    def get_history_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive history summary"""
        conn = sqlite3.connect(self.history_db)
        
        # Get snapshots
        snapshot_query = """
        SELECT * FROM history_snapshots
        WHERE timestamp > datetime('now', '-' || ? || ' days')
        ORDER BY timestamp
        """
        snapshots_df = pd.read_sql_query(snapshot_query, conn, params=(days,))
        
        # Get trends
        trends_query = """
        SELECT * FROM trend_analyses
        WHERE timestamp > datetime('now', '-' || ? || ' days')
        ORDER BY timestamp DESC
        LIMIT 10
        """
        trends_df = pd.read_sql_query(trends_query, conn, params=(days,))
        
        # Get anomalies
        anomalies_query = """
        SELECT * FROM anomaly_records
        WHERE timestamp > datetime('now', '-' || ? || ' days')
        ORDER BY timestamp DESC
        """
        anomalies_df = pd.read_sql_query(anomalies_query, conn, params=(days,))
        
        conn.close()
        
        summary = {
            "period": f"Last {days} days",
            "total_snapshots": len(snapshots_df),
            "evaluation_growth": self._calculate_growth_rate(snapshots_df),
            "current_metrics": self._get_current_metrics(snapshots_df),
            "recent_trends": self._summarize_trends(trends_df),
            "anomaly_summary": self._summarize_anomalies(anomalies_df),
            "data_quality": self._assess_overall_data_quality(snapshots_df)
        }
        
        return summary
    
    def _calculate_growth_rate(self, snapshots_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate growth rates from snapshots"""
        if len(snapshots_df) < 2:
            return {"evaluations": 0, "models": 0, "coverage": 0}
        
        first = snapshots_df.iloc[0]
        last = snapshots_df.iloc[-1]
        days = (pd.to_datetime(last['timestamp']) - pd.to_datetime(first['timestamp'])).days
        
        if days == 0:
            return {"evaluations": 0, "models": 0, "coverage": 0}
        
        return {
            "evaluations_per_day": (last['total_evaluations'] - first['total_evaluations']) / days,
            "models": last['unique_models'] - first['unique_models'],
            "avg_score_change": last['avg_alignment_score'] - first['avg_alignment_score']
        }
    
    def _get_current_metrics(self, snapshots_df: pd.DataFrame) -> Dict[str, Any]:
        """Get current metrics from latest snapshot"""
        if snapshots_df.empty:
            return {}
        
        latest = snapshots_df.iloc[-1]
        
        return {
            "total_evaluations": latest['total_evaluations'],
            "avg_alignment_score": latest['avg_alignment_score'],
            "unique_models": latest['unique_models'],
            "risk_distribution": json.loads(latest['risk_distribution'])
        }
    
    def _summarize_trends(self, trends_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Summarize recent trends"""
        trends = []
        
        for _, row in trends_df.iterrows():
            trends.append({
                "metric": row['metric'],
                "entity": f"{row['entity_type']}:{row['entity_name']}",
                "direction": row['trend_direction'],
                "change_rate": row['change_rate'],
                "confidence": row['confidence']
            })
        
        return trends
    
    def _summarize_anomalies(self, anomalies_df: pd.DataFrame) -> Dict[str, Any]:
        """Summarize anomalies"""
        if anomalies_df.empty:
            return {"total": 0, "unresolved": 0, "by_severity": {}}
        
        return {
            "total": len(anomalies_df),
            "unresolved": len(anomalies_df[anomalies_df['resolved'] == 0]),
            "by_severity": anomalies_df['severity'].value_counts().to_dict(),
            "by_type": anomalies_df['anomaly_type'].value_counts().to_dict()
        }
    
    def _assess_overall_data_quality(self, snapshots_df: pd.DataFrame) -> Dict[str, float]:
        """Assess overall data quality from snapshots"""
        if snapshots_df.empty:
            return {"completeness": 0, "balance": 0, "coverage": 0}
        
        # Extract data quality metrics from metadata
        quality_scores = []
        
        for _, row in snapshots_df.iterrows():
            metadata = json.loads(row['metadata'])
            if 'data_quality' in metadata:
                quality = metadata['data_quality']
                quality_scores.append(quality.get('completeness', 0))
        
        return {
            "avg_completeness": np.mean(quality_scores) if quality_scores else 0,
            "consistency": 1 - np.std(quality_scores) if len(quality_scores) > 1 else 1,
            "snapshots_available": len(snapshots_df)
        }
    
    def create_history_visualizations(self, output_dir: str = "./history_viz", days: int = 30):
        """Create visualizations of evaluation history"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        conn = sqlite3.connect(self.history_db)
        
        # Load snapshots
        query = """
        SELECT * FROM history_snapshots
        WHERE timestamp > datetime('now', '-' || ? || ' days')
        ORDER BY timestamp
        """
        snapshots_df = pd.read_sql_query(query, conn, params=(days,))
        conn.close()
        
        if snapshots_df.empty:
            logger.warning("No data available for visualization")
            return []
        
        # Convert timestamp
        snapshots_df['timestamp'] = pd.to_datetime(snapshots_df['timestamp'])
        
        viz_files = []
        
        # 1. Evaluation volume over time
        plt.figure(figsize=(12, 6))
        plt.plot(snapshots_df['timestamp'], snapshots_df['total_evaluations'], 
                marker='o', linewidth=2, markersize=8)
        plt.xlabel('Date')
        plt.ylabel('Total Evaluations')
        plt.title('Evaluation Volume Over Time')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = output_path / f"evaluation_volume_{datetime.now().strftime('%Y%m%d')}.png"
        plt.savefig(filename)
        plt.close()
        viz_files.append(str(filename))
        
        # 2. Alignment score trends
        plt.figure(figsize=(12, 6))
        plt.plot(snapshots_df['timestamp'], snapshots_df['avg_alignment_score'], 
                marker='o', linewidth=2, markersize=8, color='green')
        plt.axhline(y=4.0, color='r', linestyle='--', label='Target Score')
        plt.xlabel('Date')
        plt.ylabel('Average Alignment Score')
        plt.title('Alignment Score Trends')
        plt.ylim(0, 5)
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = output_path / f"alignment_trends_{datetime.now().strftime('%Y%m%d')}.png"
        plt.savefig(filename)
        plt.close()
        viz_files.append(str(filename))
        
        # 3. Model performance comparison over time
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Extract model metrics
        model_scores_over_time = defaultdict(list)
        timestamps = []
        
        for _, row in snapshots_df.iterrows():
            model_metrics = json.loads(row['model_metrics'])
            timestamps.append(row['timestamp'])
            
            for model, metrics in model_metrics.items():
                model_scores_over_time[model].append(metrics.get('avg_score', 0))
        
        # Plot each model
        for model, scores in model_scores_over_time.items():
            if len(scores) == len(timestamps):
                ax.plot(timestamps, scores, marker='o', label=model, alpha=0.8)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Average Score')
        ax.set_title('Model Performance Over Time')
        ax.legend()
        plt.xticks(rotation=45)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = output_path / f"model_performance_history_{datetime.now().strftime('%Y%m%d')}.png"
        plt.savefig(filename)
        plt.close()
        viz_files.append(str(filename))
        
        return viz_files
    
    def export_history_report(self, output_file: str, days: int = 30):
        """Export comprehensive history report"""
        summary = self.get_history_summary(days)
        
        report_content = f"""# MASB-Alt Evaluation History Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Period: Last {days} days

## Summary

- Total Snapshots: {summary['total_snapshots']}
- Evaluation Growth Rate: {summary['evaluation_growth'].get('evaluations_per_day', 0):.1f} per day
- Current Average Score: {summary['current_metrics'].get('avg_alignment_score', 0):.2f}

## Recent Trends

"""
        
        for trend in summary['recent_trends']:
            report_content += f"- **{trend['metric']}** ({trend['entity']}): {trend['direction']} "
            report_content += f"({trend['change_rate']:.1%} change, {trend['confidence']:.1%} confidence)\n"
        
        report_content += f"""

## Anomalies

- Total Anomalies: {summary['anomaly_summary']['total']}
- Unresolved: {summary['anomaly_summary']['unresolved']}
- By Severity: {summary['anomaly_summary'].get('by_severity', {})}

## Data Quality

- Average Completeness: {summary['data_quality'].get('avg_completeness', 0):.1%}
- Consistency Score: {summary['data_quality'].get('consistency', 0):.1%}
"""
        
        # Save report
        with open(output_file, 'w') as f:
            f.write(report_content)
        
        logger.info(f"History report exported to {output_file}")

# Command-line interface
def main():
    """CLI for history tracker"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MASB-Alt History Tracker")
    parser.add_argument("action", choices=["snapshot", "trends", "anomalies", "report", "visualize"])
    parser.add_argument("--interval", choices=["hourly", "daily", "weekly", "monthly"], 
                       default="daily", help="Snapshot interval")
    parser.add_argument("--days", type=int, default=30, help="Days to analyze")
    parser.add_argument("--output", help="Output file/directory")
    
    args = parser.parse_args()
    
    tracker = HistoryTracker()
    
    if args.action == "snapshot":
        snapshot = tracker.create_snapshot(args.interval)
        if snapshot:
            print(f"Created snapshot: {snapshot.snapshot_id}")
            print(f"Total evaluations: {snapshot.total_evaluations}")
            print(f"Average score: {snapshot.avg_alignment_score:.2f}")
        else:
            print("No data available for snapshot")
    
    elif args.action == "trends":
        # Analyze overall trends
        trend = tracker.analyze_trends(lookback_days=args.days)
        if trend:
            print(f"\nTrend Analysis - Alignment Score")
            print(f"Direction: {trend.trend_direction}")
            print(f"Change rate: {trend.change_rate:.1%}")
            print(f"Confidence: {trend.confidence:.1%}")
            print(f"Forecast (next 7 days): {[f'{v:.2f}' for v in trend.forecast]}")
    
    elif args.action == "anomalies":
        anomalies = tracker.detect_anomalies()
        if anomalies:
            print(f"\nDetected {len(anomalies)} anomalies:")
            for anomaly in anomalies:
                print(f"\n[{anomaly.severity.upper()}] {anomaly.anomaly_type}")
                print(f"  {anomaly.description}")
                if anomaly.affected_model:
                    print(f"  Model: {anomaly.affected_model}")
        else:
            print("No anomalies detected")
    
    elif args.action == "report":
        output_file = args.output or f"history_report_{datetime.now().strftime('%Y%m%d')}.md"
        tracker.export_history_report(output_file, args.days)
        print(f"Report exported to: {output_file}")
    
    elif args.action == "visualize":
        output_dir = args.output or "./history_viz"
        viz_files = tracker.create_history_visualizations(output_dir, args.days)
        print(f"Created {len(viz_files)} visualizations in {output_dir}")

if __name__ == "__main__":
    main()