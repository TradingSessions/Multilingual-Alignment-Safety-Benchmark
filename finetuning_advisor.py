# finetuning_advisor.py - Model fine-tuning recommendation system

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path
import json
import sqlite3
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FineTuningRecommendation:
    """Fine-tuning recommendation for a model"""
    model: str
    priority: str  # 'critical', 'high', 'medium', 'low'
    recommendation_type: str  # 'language', 'domain', 'safety', 'general'
    issue_description: str
    suggested_actions: List[str]
    training_data_requirements: Dict[str, Any]
    expected_improvement: Dict[str, float]
    effort_estimate: str  # 'low', 'medium', 'high'
    confidence_score: float

@dataclass
class TrainingDataset:
    """Recommended training dataset specification"""
    dataset_id: str
    name: str
    focus_areas: List[str]
    languages: List[str]
    domains: List[str]
    min_examples: int
    data_sources: List[str]
    quality_requirements: Dict[str, Any]

class FineTuningAdvisor:
    """System for analyzing model performance and recommending fine-tuning strategies"""
    
    def __init__(self, data_path: str = "./data", output_dir: str = "./finetuning_recommendations"):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Performance thresholds
        self.thresholds = {
            'min_acceptable_score': 3.5,
            'target_score': 4.0,
            'max_acceptable_error_rate': 0.1,
            'min_language_coverage': 0.8,
            'significant_gap': 0.5
        }
        
        # Fine-tuning strategies
        self.strategies = {
            'language_specific': {
                'description': 'Fine-tune on specific language data',
                'effort': 'medium',
                'expected_improvement': 0.3
            },
            'domain_specific': {
                'description': 'Fine-tune on domain-specific examples',
                'effort': 'medium',
                'expected_improvement': 0.4
            },
            'safety_alignment': {
                'description': 'Fine-tune with safety-focused examples',
                'effort': 'high',
                'expected_improvement': 0.5
            },
            'general_improvement': {
                'description': 'Fine-tune on diverse high-quality examples',
                'effort': 'low',
                'expected_improvement': 0.2
            }
        }
    
    def analyze_model_performance(self, model: str, days: int = 30) -> Dict[str, Any]:
        """Comprehensive analysis of model performance"""
        db_path = self.data_path / "masb_alt.db"
        conn = sqlite3.connect(db_path)
        
        # Load evaluation data
        query = """
        SELECT 
            e.*,
            r.model,
            r.latency_ms,
            r.error as response_error,
            p.language,
            p.domain,
            p.text as prompt_text
        FROM evaluations e
        JOIN responses r ON e.response_id = r.response_id
        JOIN prompts p ON r.prompt_id = p.prompt_id
        WHERE r.model = ? 
        AND e.timestamp > datetime('now', '-' || ? || ' days')
        """
        
        df = pd.read_sql_query(query, conn, params=(model, days))
        conn.close()
        
        if df.empty:
            return {'error': 'No data available for analysis'}
        
        # Parse risk flags
        df['risk_flags'] = df['risk_flags'].apply(lambda x: json.loads(x) if x else {})
        
        # Overall performance metrics
        analysis = {
            'model': model,
            'evaluation_period': f"Last {days} days",
            'total_evaluations': len(df),
            'overall_metrics': {
                'avg_alignment_score': df['alignment_score'].mean(),
                'std_alignment_score': df['alignment_score'].std(),
                'error_rate': (df['response_error'].notna()).mean(),
                'high_risk_rate': (df['risk_level'].isin(['high', 'critical'])).mean()
            }
        }
        
        # Language-specific analysis
        language_performance = {}
        for lang in df['language'].unique():
            lang_data = df[df['language'] == lang]
            language_performance[lang] = {
                'n_samples': len(lang_data),
                'avg_score': lang_data['alignment_score'].mean(),
                'std_score': lang_data['alignment_score'].std(),
                'error_rate': (lang_data['response_error'].notna()).mean(),
                'common_risks': self._get_common_risks(lang_data)
            }
        analysis['language_performance'] = language_performance
        
        # Domain-specific analysis
        domain_performance = {}
        for domain in df['domain'].unique():
            domain_data = df[df['domain'] == domain]
            domain_performance[domain] = {
                'n_samples': len(domain_data),
                'avg_score': domain_data['alignment_score'].mean(),
                'high_risk_rate': (domain_data['risk_level'].isin(['high', 'critical'])).mean(),
                'common_risks': self._get_common_risks(domain_data)
            }
        analysis['domain_performance'] = domain_performance
        
        # Identify weak areas
        analysis['weak_areas'] = self._identify_weak_areas(df)
        
        # Error pattern analysis
        analysis['error_patterns'] = self._analyze_error_patterns(df)
        
        return analysis
    
    def _get_common_risks(self, df: pd.DataFrame) -> List[Tuple[str, int]]:
        """Get most common risk flags"""
        risk_counts = defaultdict(int)
        for _, row in df.iterrows():
            for flag, value in row['risk_flags'].items():
                if value:
                    risk_counts[flag] += 1
        
        return sorted(risk_counts.items(), key=lambda x: x[1], reverse=True)[:3]
    
    def _identify_weak_areas(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Identify areas where model performs poorly"""
        weak_areas = {
            'languages': [],
            'domains': [],
            'risk_types': []
        }
        
        # Weak languages
        lang_scores = df.groupby('language')['alignment_score'].mean()
        weak_areas['languages'] = lang_scores[lang_scores < self.thresholds['min_acceptable_score']].index.tolist()
        
        # Weak domains
        domain_scores = df.groupby('domain')['alignment_score'].mean()
        weak_areas['domains'] = domain_scores[domain_scores < self.thresholds['min_acceptable_score']].index.tolist()
        
        # Common risk types
        risk_counts = defaultdict(int)
        for _, row in df.iterrows():
            for flag, value in row['risk_flags'].items():
                if value:
                    risk_counts[flag] += 1
        
        total_evals = len(df)
        for risk, count in risk_counts.items():
            if count / total_evals > 0.1:  # More than 10% occurrence
                weak_areas['risk_types'].append(risk)
        
        return weak_areas
    
    def _analyze_error_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns in errors and low scores"""
        error_patterns = {
            'low_score_examples': [],
            'error_examples': [],
            'risk_correlations': {}
        }
        
        # Get examples of low scores
        low_scores = df[df['alignment_score'] <= 2].head(10)
        for _, row in low_scores.iterrows():
            error_patterns['low_score_examples'].append({
                'prompt': row['prompt_text'][:100] + '...',
                'language': row['language'],
                'domain': row['domain'],
                'score': row['alignment_score'],
                'risks': [k for k, v in row['risk_flags'].items() if v]
            })
        
        # Get error examples
        errors = df[df['response_error'].notna()].head(5)
        for _, row in errors.iterrows():
            error_patterns['error_examples'].append({
                'prompt': row['prompt_text'][:100] + '...',
                'language': row['language'],
                'domain': row['domain'],
                'error': row['response_error']
            })
        
        # Risk correlations
        for risk_type in ['hallucination', 'unsafe_medical_advice', 'culturally_insensitive']:
            risk_df = df[df['risk_flags'].apply(lambda x: x.get(risk_type, False))]
            if len(risk_df) > 0:
                error_patterns['risk_correlations'][risk_type] = {
                    'languages': risk_df['language'].value_counts().head(3).to_dict(),
                    'domains': risk_df['domain'].value_counts().head(3).to_dict()
                }
        
        return error_patterns
    
    def generate_recommendations(self, analysis: Dict[str, Any]) -> List[FineTuningRecommendation]:
        """Generate fine-tuning recommendations based on analysis"""
        recommendations = []
        model = analysis['model']
        
        # Language-specific recommendations
        for lang, perf in analysis['language_performance'].items():
            if perf['avg_score'] < self.thresholds['min_acceptable_score']:
                priority = 'critical' if perf['avg_score'] < 3.0 else 'high'
                
                rec = FineTuningRecommendation(
                    model=model,
                    priority=priority,
                    recommendation_type='language',
                    issue_description=f"{lang} performance below acceptable threshold ({perf['avg_score']:.2f})",
                    suggested_actions=[
                        f"Collect high-quality {lang} training examples",
                        f"Focus on {lang}-specific cultural context",
                        "Include diverse domains in training data",
                        "Add safety-aligned examples in target language"
                    ],
                    training_data_requirements={
                        'language': lang,
                        'min_examples': 1000,
                        'quality_score_threshold': 4.0,
                        'diversity_requirements': ['multiple_domains', 'varied_complexity']
                    },
                    expected_improvement={
                        'alignment_score': 0.5,
                        'error_reduction': 0.3
                    },
                    effort_estimate='medium',
                    confidence_score=0.8
                )
                recommendations.append(rec)
        
        # Domain-specific recommendations
        for domain, perf in analysis['domain_performance'].items():
            if perf['avg_score'] < self.thresholds['min_acceptable_score'] or perf['high_risk_rate'] > 0.2:
                priority = 'high' if domain in ['healthcare', 'financial_literacy'] else 'medium'
                
                rec = FineTuningRecommendation(
                    model=model,
                    priority=priority,
                    recommendation_type='domain',
                    issue_description=f"{domain} domain shows poor performance or high risk",
                    suggested_actions=[
                        f"Curate domain-specific training data for {domain}",
                        "Include expert-validated examples",
                        "Add safety constraints for sensitive topics",
                        "Balance with general examples to prevent overfitting"
                    ],
                    training_data_requirements={
                        'domain': domain,
                        'min_examples': 500,
                        'expert_validation': True,
                        'risk_examples': True
                    },
                    expected_improvement={
                        'alignment_score': 0.4,
                        'risk_reduction': 0.5
                    },
                    effort_estimate='medium',
                    confidence_score=0.75
                )
                recommendations.append(rec)
        
        # Safety-specific recommendations
        if analysis['overall_metrics']['high_risk_rate'] > 0.15:
            rec = FineTuningRecommendation(
                model=model,
                priority='critical',
                recommendation_type='safety',
                issue_description=f"High risk rate detected ({analysis['overall_metrics']['high_risk_rate']:.1%})",
                suggested_actions=[
                    "Implement stronger safety constraints",
                    "Fine-tune on adversarial safety examples",
                    "Add explicit refusal training for harmful requests",
                    "Include diverse safety scenarios across languages"
                ],
                training_data_requirements={
                    'focus': 'safety_alignment',
                    'min_examples': 2000,
                    'adversarial_examples': True,
                    'refusal_examples': True
                },
                expected_improvement={
                    'risk_reduction': 0.7,
                    'alignment_score': 0.3
                },
                effort_estimate='high',
                confidence_score=0.85
            )
            recommendations.append(rec)
        
        # General performance recommendations
        if analysis['overall_metrics']['avg_alignment_score'] < self.thresholds['target_score']:
            rec = FineTuningRecommendation(
                model=model,
                priority='medium',
                recommendation_type='general',
                issue_description="Overall performance below target",
                suggested_actions=[
                    "Fine-tune on high-quality diverse examples",
                    "Include examples from all supported languages",
                    "Balance across all domains",
                    "Focus on clarity and helpfulness"
                ],
                training_data_requirements={
                    'diversity': 'high',
                    'min_examples': 5000,
                    'quality_threshold': 4.5,
                    'balanced_distribution': True
                },
                expected_improvement={
                    'alignment_score': 0.3,
                    'consistency': 0.4
                },
                effort_estimate='low',
                confidence_score=0.7
            )
            recommendations.append(rec)
        
        # Sort by priority
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        recommendations.sort(key=lambda x: priority_order[x.priority])
        
        return recommendations
    
    def create_training_dataset_specs(self, 
                                    recommendations: List[FineTuningRecommendation]) -> List[TrainingDataset]:
        """Create specifications for training datasets based on recommendations"""
        dataset_specs = []
        
        # Group recommendations by type
        by_type = defaultdict(list)
        for rec in recommendations:
            by_type[rec.recommendation_type].append(rec)
        
        # Language-specific datasets
        if 'language' in by_type:
            languages = [rec.training_data_requirements['language'] for rec in by_type['language']]
            
            spec = TrainingDataset(
                dataset_id=f"lang_specific_{datetime.now().strftime('%Y%m%d')}",
                name="Language-Specific Fine-Tuning Dataset",
                focus_areas=['language_alignment', 'cultural_sensitivity'],
                languages=languages,
                domains=['healthcare', 'education', 'financial_literacy', 'civic_participation'],
                min_examples=1000 * len(languages),
                data_sources=[
                    'high_quality_translations',
                    'native_speaker_validations',
                    'cultural_expert_reviews'
                ],
                quality_requirements={
                    'min_alignment_score': 4.0,
                    'native_speaker_validated': True,
                    'culturally_appropriate': True
                }
            )
            dataset_specs.append(spec)
        
        # Domain-specific datasets
        if 'domain' in by_type:
            domains = [rec.training_data_requirements['domain'] for rec in by_type['domain']]
            
            spec = TrainingDataset(
                dataset_id=f"domain_specific_{datetime.now().strftime('%Y%m%d')}",
                name="Domain-Specific Fine-Tuning Dataset",
                focus_areas=['domain_expertise', 'terminology_accuracy'],
                languages=['en', 'sw', 'ar', 'hi', 'vi'],  # Core languages
                domains=domains,
                min_examples=500 * len(domains),
                data_sources=[
                    'expert_curated_content',
                    'professional_guidelines',
                    'validated_qa_pairs'
                ],
                quality_requirements={
                    'expert_validated': True,
                    'factually_accurate': True,
                    'up_to_date': True
                }
            )
            dataset_specs.append(spec)
        
        # Safety dataset
        if 'safety' in by_type:
            spec = TrainingDataset(
                dataset_id=f"safety_alignment_{datetime.now().strftime('%Y%m%d')}",
                name="Safety Alignment Dataset",
                focus_areas=['harm_prevention', 'appropriate_refusals', 'bias_mitigation'],
                languages=['en', 'sw', 'ar', 'hi', 'vi', 'ug'],  # All supported languages
                domains=['all'],
                min_examples=3000,
                data_sources=[
                    'adversarial_examples',
                    'edge_case_scenarios',
                    'refusal_templates',
                    'debiasing_examples'
                ],
                quality_requirements={
                    'safety_validated': True,
                    'includes_edge_cases': True,
                    'culturally_diverse': True
                }
            )
            dataset_specs.append(spec)
        
        return dataset_specs
    
    def generate_finetuning_plan(self, 
                                 model: str,
                                 recommendations: List[FineTuningRecommendation],
                                 dataset_specs: List[TrainingDataset]) -> Dict[str, Any]:
        """Generate comprehensive fine-tuning plan"""
        plan = {
            'model': model,
            'generated_at': datetime.now().isoformat(),
            'total_recommendations': len(recommendations),
            'priority_breakdown': {},
            'phases': [],
            'resource_requirements': {},
            'expected_outcomes': {},
            'timeline': {}
        }
        
        # Priority breakdown
        for rec in recommendations:
            plan['priority_breakdown'][rec.priority] = plan['priority_breakdown'].get(rec.priority, 0) + 1
        
        # Create phases based on priority
        phases = []
        
        # Phase 1: Critical issues
        critical_recs = [r for r in recommendations if r.priority == 'critical']
        if critical_recs:
            phases.append({
                'phase': 1,
                'name': 'Critical Improvements',
                'duration': '2-3 weeks',
                'focus': [r.issue_description for r in critical_recs],
                'datasets': [ds.dataset_id for ds in dataset_specs if any(
                    r.recommendation_type in ds.focus_areas for r in critical_recs
                )],
                'expected_improvement': {
                    'risk_reduction': 0.5,
                    'alignment_improvement': 0.3
                }
            })
        
        # Phase 2: High priority
        high_recs = [r for r in recommendations if r.priority == 'high']
        if high_recs:
            phases.append({
                'phase': 2,
                'name': 'High Priority Enhancements',
                'duration': '3-4 weeks',
                'focus': [r.issue_description for r in high_recs],
                'datasets': [ds.dataset_id for ds in dataset_specs if any(
                    r.recommendation_type in ['language', 'domain'] for r in high_recs
                )],
                'expected_improvement': {
                    'language_gaps': 0.4,
                    'domain_performance': 0.4
                }
            })
        
        # Phase 3: General improvements
        other_recs = [r for r in recommendations if r.priority in ['medium', 'low']]
        if other_recs:
            phases.append({
                'phase': 3,
                'name': 'General Performance Optimization',
                'duration': '2 weeks',
                'focus': [r.issue_description for r in other_recs],
                'datasets': [ds.dataset_id for ds in dataset_specs],
                'expected_improvement': {
                    'overall_score': 0.2,
                    'consistency': 0.3
                }
            })
        
        plan['phases'] = phases
        
        # Resource requirements
        total_examples = sum(ds.min_examples for ds in dataset_specs)
        plan['resource_requirements'] = {
            'total_training_examples': total_examples,
            'estimated_compute_hours': total_examples / 100,  # Rough estimate
            'human_validation_hours': total_examples / 50,
            'expert_reviewers_needed': len(set(r.recommendation_type for r in recommendations))
        }
        
        # Expected outcomes
        plan['expected_outcomes'] = {
            'target_alignment_score': 4.2,
            'target_risk_reduction': 0.7,
            'improved_languages': list(set(r.training_data_requirements.get('language', '') 
                                         for r in recommendations if r.recommendation_type == 'language')),
            'improved_domains': list(set(r.training_data_requirements.get('domain', '') 
                                       for r in recommendations if r.recommendation_type == 'domain'))
        }
        
        # Timeline
        total_weeks = sum(int(p['duration'].split('-')[1].split()[0]) for p in phases)
        plan['timeline'] = {
            'total_duration': f"{total_weeks} weeks",
            'start_date': datetime.now().isoformat(),
            'phases': [
                {
                    'phase': p['phase'],
                    'start_week': sum(int(phases[i]['duration'].split('-')[1].split()[0]) 
                                    for i in range(p['phase']-1)),
                    'duration': p['duration']
                }
                for p in phases
            ]
        }
        
        return plan
    
    def visualize_recommendations(self, 
                                 analysis: Dict[str, Any],
                                 recommendations: List[FineTuningRecommendation]) -> List[str]:
        """Create visualizations for fine-tuning recommendations"""
        viz_files = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Performance gaps visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Language performance
        lang_perf = analysis['language_performance']
        languages = list(lang_perf.keys())
        scores = [lang_perf[lang]['avg_score'] for lang in languages]
        
        ax1.bar(languages, scores, color=['red' if s < 3.5 else 'yellow' if s < 4.0 else 'green' for s in scores])
        ax1.axhline(y=3.5, color='r', linestyle='--', label='Min Acceptable')
        ax1.axhline(y=4.0, color='g', linestyle='--', label='Target')
        ax1.set_ylabel('Average Alignment Score')
        ax1.set_title('Performance by Language')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        
        # Domain performance
        domain_perf = analysis['domain_performance']
        domains = list(domain_perf.keys())
        scores = [domain_perf[d]['avg_score'] for d in domains]
        
        ax2.bar(domains, scores, color=['red' if s < 3.5 else 'yellow' if s < 4.0 else 'green' for s in scores])
        ax2.axhline(y=3.5, color='r', linestyle='--', label='Min Acceptable')
        ax2.axhline(y=4.0, color='g', linestyle='--', label='Target')
        ax2.set_ylabel('Average Alignment Score')
        ax2.set_title('Performance by Domain')
        ax2.legend()
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        filename = self.output_dir / f"performance_gaps_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        viz_files.append(str(filename))
        
        # 2. Recommendation priorities
        fig, ax = plt.subplots(figsize=(10, 6))
        
        priority_counts = defaultdict(int)
        for rec in recommendations:
            priority_counts[rec.priority] += 1
        
        priorities = ['critical', 'high', 'medium', 'low']
        counts = [priority_counts[p] for p in priorities]
        colors = ['red', 'orange', 'yellow', 'green']
        
        ax.bar(priorities, counts, color=colors)
        ax.set_ylabel('Number of Recommendations')
        ax.set_title('Fine-Tuning Recommendations by Priority')
        
        # Add recommendation types
        for i, (priority, count) in enumerate(zip(priorities, counts)):
            if count > 0:
                rec_types = [r.recommendation_type for r in recommendations if r.priority == priority]
                ax.text(i, count + 0.1, f"{', '.join(set(rec_types))}", ha='center', fontsize=8)
        
        plt.tight_layout()
        filename = self.output_dir / f"recommendation_priorities_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        viz_files.append(str(filename))
        
        # 3. Expected improvements
        fig, ax = plt.subplots(figsize=(12, 6))
        
        improvements = []
        labels = []
        
        for rec in recommendations[:10]:  # Top 10 recommendations
            if 'alignment_score' in rec.expected_improvement:
                improvements.append(rec.expected_improvement['alignment_score'])
                labels.append(f"{rec.recommendation_type[:4]}_{rec.model[:10]}")
        
        if improvements:
            ax.barh(labels, improvements, color='lightblue')
            ax.set_xlabel('Expected Score Improvement')
            ax.set_title('Expected Improvements from Fine-Tuning')
            
            # Add effort indicators
            for i, rec in enumerate(recommendations[:len(improvements)]):
                effort_color = {'low': 'green', 'medium': 'yellow', 'high': 'red'}[rec.effort_estimate]
                ax.text(improvements[i] + 0.01, i, rec.effort_estimate, 
                       va='center', color=effort_color, fontweight='bold')
        
        plt.tight_layout()
        filename = self.output_dir / f"expected_improvements_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        viz_files.append(str(filename))
        
        return viz_files
    
    def export_recommendations(self, 
                             model: str,
                             analysis: Dict[str, Any],
                             recommendations: List[FineTuningRecommendation],
                             dataset_specs: List[TrainingDataset],
                             plan: Dict[str, Any]) -> str:
        """Export complete fine-tuning recommendations package"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_data = {
            'metadata': {
                'model': model,
                'generated_at': timestamp,
                'analyzer_version': '1.0'
            },
            'performance_analysis': analysis,
            'recommendations': [asdict(r) for r in recommendations],
            'dataset_specifications': [asdict(ds) for ds in dataset_specs],
            'implementation_plan': plan
        }
        
        # Save as JSON
        output_file = self.output_dir / f"finetuning_recommendations_{model}_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        # Create summary markdown
        summary_file = self.output_dir / f"finetuning_summary_{model}_{timestamp}.md"
        with open(summary_file, 'w') as f:
            f.write(f"# Fine-Tuning Recommendations for {model}\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Executive Summary\n\n")
            f.write(f"- Total Recommendations: {len(recommendations)}\n")
            f.write(f"- Critical Issues: {sum(1 for r in recommendations if r.priority == 'critical')}\n")
            f.write(f"- Estimated Timeline: {plan['timeline']['total_duration']}\n")
            f.write(f"- Training Examples Needed: {plan['resource_requirements']['total_training_examples']:,}\n\n")
            
            f.write("## Key Recommendations\n\n")
            for i, rec in enumerate(recommendations[:5], 1):
                f.write(f"### {i}. [{rec.priority.upper()}] {rec.issue_description}\n")
                f.write(f"**Type:** {rec.recommendation_type}\n")
                f.write(f"**Actions:**\n")
                for action in rec.suggested_actions:
                    f.write(f"- {action}\n")
                f.write(f"**Expected Improvement:** {rec.expected_improvement}\n\n")
            
            f.write("## Implementation Phases\n\n")
            for phase in plan['phases']:
                f.write(f"### Phase {phase['phase']}: {phase['name']}\n")
                f.write(f"**Duration:** {phase['duration']}\n")
                f.write(f"**Focus Areas:**\n")
                for focus in phase['focus']:
                    f.write(f"- {focus}\n")
                f.write("\n")
        
        logger.info(f"Recommendations exported to {output_file}")
        return str(output_file)

# Main execution
def main():
    """CLI for fine-tuning advisor"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MASB-Alt Fine-Tuning Advisor")
    parser.add_argument("model", help="Model name to analyze")
    parser.add_argument("--days", type=int, default=30, help="Days of data to analyze")
    parser.add_argument("--output-dir", default="./finetuning_recommendations", help="Output directory")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    
    args = parser.parse_args()
    
    advisor = FineTuningAdvisor(output_dir=args.output_dir)
    
    print(f"Analyzing {args.model} performance...")
    
    # Analyze performance
    analysis = advisor.analyze_model_performance(args.model, args.days)
    
    if 'error' in analysis:
        print(f"Error: {analysis['error']}")
        return
    
    print(f"Analysis complete. Total evaluations: {analysis['total_evaluations']}")
    
    # Generate recommendations
    recommendations = advisor.generate_recommendations(analysis)
    print(f"\nGenerated {len(recommendations)} recommendations")
    
    # Create dataset specifications
    dataset_specs = advisor.create_training_dataset_specs(recommendations)
    print(f"Created {len(dataset_specs)} dataset specifications")
    
    # Generate plan
    plan = advisor.generate_finetuning_plan(args.model, recommendations, dataset_specs)
    print(f"\nImplementation plan created with {len(plan['phases'])} phases")
    
    # Visualize if requested
    if args.visualize:
        viz_files = advisor.visualize_recommendations(analysis, recommendations)
        print(f"Generated {len(viz_files)} visualizations")
    
    # Export everything
    export_file = advisor.export_recommendations(
        args.model, analysis, recommendations, dataset_specs, plan
    )
    
    print(f"\nRecommendations exported to: {export_file}")
    
    # Print summary
    print("\n=== Summary ===")
    for i, rec in enumerate(recommendations[:3], 1):
        print(f"\n{i}. [{rec.priority.upper()}] {rec.issue_description}")
        print(f"   Type: {rec.recommendation_type}")
        print(f"   Effort: {rec.effort_estimate}")
        print(f"   Expected Improvement: {list(rec.expected_improvement.values())[0]:.1%}")

if __name__ == "__main__":
    main()