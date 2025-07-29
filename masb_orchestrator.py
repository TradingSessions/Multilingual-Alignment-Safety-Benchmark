# masb_orchestrator.py - Main orchestrator for MASB-Alt system

import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json

from dataclasses import asdict
from prompt_generator.multilingual_prompt_gen import MultilingualPromptGenerator
from llm_api_client import LLMClientFactory
from evaluation.evaluator import MultilingualAlignmentEvaluator
from data_manager import DataManager
from api_server import APIServer
from monitoring_dashboard import MonitoringDashboard
from benchmark_runner import ModelBenchmark
from corpus_manager import CorpusManager
from comparative_analyzer import ComparativeAnalyzer
from report_generator import ReportGenerator, ReportConfig
from finetuning_advisor import FineTuningAdvisor
from history_tracker import HistoryTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MASBOrchestrator:
    """Main orchestrator for the Multilingual Alignment Safety Benchmark system"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        
        # Initialize all components
        self.prompt_generator = MultilingualPromptGenerator()
        self.evaluator = MultilingualAlignmentEvaluator()
        self.data_manager = DataManager()
        self.benchmark_runner = ModelBenchmark()
        self.corpus_manager = CorpusManager()
        self.comparative_analyzer = ComparativeAnalyzer()
        self.report_generator = ReportGenerator()
        self.finetuning_advisor = FineTuningAdvisor()
        self.history_tracker = HistoryTracker()
        
        # System state
        self.running_services = {}
        
        logger.info("MASB-Alt Orchestrator initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load system configuration"""
        default_config = {
            "models": ["openai", "anthropic"],
            "languages": ["en", "sw", "ar", "hi", "vi"],
            "domains": ["healthcare", "education", "financial_literacy", "civic_participation"],
            "evaluation_batch_size": 10,
            "auto_monitoring": True,
            "snapshot_interval": "daily",
            "report_schedule": "weekly"
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    async def run_evaluation_suite(self, 
                                  models: Optional[List[str]] = None,
                                  languages: Optional[List[str]] = None,
                                  domains: Optional[List[str]] = None,
                                  prompts_per_combination: int = 5) -> str:
        """Run comprehensive evaluation suite"""
        models = models or self.config["models"]
        languages = languages or self.config["languages"]
        domains = domains or self.config["domains"]
        
        logger.info(f"Starting evaluation suite with {len(models)} models, {len(languages)} languages, {len(domains)} domains")
        
        # Create dataset
        dataset_id = f"eval_suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.data_manager.create_dataset(
            dataset_id=dataset_id,
            name=f"Evaluation Suite {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            description="Comprehensive evaluation across models, languages, and domains"
        )
        
        # Generate prompts
        total_prompts = 0
        for language in languages:
            for domain in domains:
                for _ in range(prompts_per_combination):
                    prompt = self.prompt_generator.get_random_prompt(domain, language)
                    if prompt:
                        self.data_manager.add_prompt({
                            "text": prompt["text"],
                            "language": language,
                            "domain": domain,
                            "risk_level": prompt.get("risk_level", "medium"),
                            "tags": prompt.get("tags", []),
                            "metadata": prompt.get("metadata", {})
                        }, dataset_id)
                        total_prompts += 1
        
        logger.info(f"Generated {total_prompts} prompts")
        
        # Run evaluations
        evaluation_count = 0
        for model_name in models:
            try:
                client = LLMClientFactory.create_client(model_name)
                prompts = self.data_manager.get_prompts(dataset_id)
                
                for prompt in prompts:
                    # Generate response
                    response = await client.generate_async(prompt["text"])
                    
                    if not response.error:
                        # Store response
                        response_id = self.data_manager.add_response({
                            "model": model_name,
                            "response": response.response,
                            "latency_ms": response.latency_ms,
                            "token_count": response.token_count,
                            "timestamp": datetime.now().isoformat()
                        }, prompt["prompt_id"])
                        
                        # Evaluate response
                        eval_result = self.evaluator.evaluate_response(
                            prompt=prompt,
                            llm_output=response.response,
                            llm_model=model_name
                        )
                        
                        # Store evaluation
                        self.data_manager.add_evaluation({
                            "alignment_score": eval_result.alignment_score.value,
                            "confidence_score": eval_result.confidence_score,
                            "risk_level": eval_result.risk_level.value,
                            "risk_flags": asdict(eval_result.risk_flags),
                            "comments": eval_result.comments,
                            "evaluator_id": "auto_evaluator",
                            "timestamp": datetime.now().isoformat()
                        }, response_id)
                        
                        evaluation_count += 1
                        
                        if evaluation_count % 10 == 0:
                            logger.info(f"Completed {evaluation_count} evaluations")
            
            except Exception as e:
                logger.error(f"Error evaluating with {model_name}: {e}")
        
        logger.info(f"Evaluation suite completed: {evaluation_count} total evaluations")
        return dataset_id
    
    async def run_benchmark_analysis(self, 
                                   models: Optional[List[str]] = None,
                                   suite_type: str = "standard") -> str:
        """Run comprehensive benchmark analysis"""
        models = models or self.config["models"]
        
        logger.info(f"Starting benchmark analysis for models: {models}")
        
        # Run benchmarks
        suite = await self.benchmark_runner.run_benchmark_suite(models, suite_type)
        
        # Generate report
        report_path = self.benchmark_runner.generate_benchmark_report(suite, "html")
        
        logger.info(f"Benchmark analysis completed: {report_path}")
        return str(report_path)
    
    def run_comparative_analysis(self, dataset_ids: Optional[List[str]] = None) -> str:
        """Run comparative analysis across models and languages"""
        logger.info("Starting comparative analysis")
        
        result = self.comparative_analyzer.run_full_comparison(dataset_ids)
        
        if result:
            logger.info(f"Comparative analysis completed: {result.comparison_id}")
            return result.comparison_id
        else:
            logger.error("Comparative analysis failed")
            return ""
    
    def generate_comprehensive_report(self, 
                                    report_type: str = "executive",
                                    format: str = "html",
                                    days: int = 7) -> str:
        """Generate comprehensive system report"""
        logger.info(f"Generating {report_type} report for last {days} days")
        
        config = ReportConfig(
            title=f"MASB-Alt {report_type.title()} Report",
            subtitle=f"Analysis for {datetime.now().strftime('%Y-%m-%d')}",
            report_type=report_type,
            format=format
        )
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        report_path = self.report_generator.generate_report(
            date_range=(start_date, end_date),
            config=config
        )
        
        logger.info(f"Report generated: {report_path}")
        return report_path
    
    def analyze_finetuning_recommendations(self, model: str, days: int = 30) -> str:
        """Generate fine-tuning recommendations for a model"""
        logger.info(f"Analyzing fine-tuning opportunities for {model}")
        
        # Analyze performance
        analysis = self.finetuning_advisor.analyze_model_performance(model, days)
        
        if 'error' in analysis:
            logger.error(f"Analysis failed: {analysis['error']}")
            return ""
        
        # Generate recommendations
        recommendations = self.finetuning_advisor.generate_recommendations(analysis)
        
        # Create dataset specifications
        dataset_specs = self.finetuning_advisor.create_training_dataset_specs(recommendations)
        
        # Generate implementation plan
        plan = self.finetuning_advisor.generate_finetuning_plan(model, recommendations, dataset_specs)
        
        # Export everything
        export_file = self.finetuning_advisor.export_recommendations(
            model, analysis, recommendations, dataset_specs, plan
        )
        
        logger.info(f"Fine-tuning analysis completed: {export_file}")
        return export_file
    
    def create_system_snapshot(self) -> str:
        """Create comprehensive system snapshot"""
        logger.info("Creating system snapshot")
        
        # Create history snapshot
        snapshot = self.history_tracker.create_snapshot("daily")
        
        # Detect anomalies
        anomalies = self.history_tracker.detect_anomalies()
        
        # Get system summary
        summary = self.history_tracker.get_history_summary()
        
        snapshot_id = snapshot.snapshot_id if snapshot else "no_data"
        
        logger.info(f"System snapshot created: {snapshot_id}")
        logger.info(f"Detected {len(anomalies)} anomalies")
        
        return snapshot_id
    
    async def start_monitoring_services(self):
        """Start monitoring and API services"""
        logger.info("Starting monitoring services")
        
        # Start API server
        try:
            api_server = APIServer()
            # In a real deployment, you'd run this in a separate process
            logger.info("API server would start on port 8000")
            self.running_services["api"] = "started"
        except Exception as e:
            logger.error(f"Failed to start API server: {e}")
        
        # Start monitoring dashboard
        try:
            dashboard = MonitoringDashboard()
            # In a real deployment, you'd run this in a separate process
            logger.info("Monitoring dashboard would start on port 8501")
            self.running_services["dashboard"] = "started"
        except Exception as e:
            logger.error(f"Failed to start monitoring dashboard: {e}")
    
    def import_corpus_data(self, 
                          language: str, 
                          domain: str, 
                          file_path: str, 
                          file_format: str = "jsonl") -> int:
        """Import corpus data"""
        logger.info(f"Importing {language} corpus for {domain} from {file_path}")
        
        count = self.corpus_manager.import_corpus_file(
            file_path=file_path,
            language=language,
            domain=domain,
            source="import",
            file_format=file_format
        )
        
        logger.info(f"Imported {count} corpus entries")
        return count
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        status = {
            "timestamp": datetime.now().isoformat(),
            "components": {
                "prompt_generator": "active",
                "evaluator": "active",
                "data_manager": "active",
                "benchmark_runner": "active",
                "corpus_manager": "active",
                "comparative_analyzer": "active",
                "report_generator": "active",
                "finetuning_advisor": "active",
                "history_tracker": "active"
            },
            "services": self.running_services,
            "configuration": self.config
        }
        
        # Get data statistics
        try:
            stats = self.data_manager.get_database_stats()
            status["data_stats"] = stats
        except Exception as e:
            status["data_stats"] = {"error": str(e)}
        
        # Get corpus statistics
        try:
            corpus_stats = self.corpus_manager.get_statistics()
            status["corpus_stats"] = corpus_stats.to_dict('records') if not corpus_stats.empty else []
        except Exception as e:
            status["corpus_stats"] = {"error": str(e)}
        
        # Get recent anomalies
        try:
            anomalies = self.history_tracker.detect_anomalies(lookback_hours=24)
            status["recent_anomalies"] = len(anomalies)
        except Exception as e:
            status["recent_anomalies"] = {"error": str(e)}
        
        return status
    
    async def run_automated_maintenance(self):
        """Run automated system maintenance tasks"""
        logger.info("Starting automated maintenance")
        
        # Create daily snapshot
        self.create_system_snapshot()
        
        # Detect anomalies
        anomalies = self.history_tracker.detect_anomalies()
        if anomalies:
            logger.warning(f"Detected {len(anomalies)} anomalies during maintenance")
        
        # Generate weekly report if it's the right day
        if datetime.now().weekday() == 0:  # Monday
            self.generate_comprehensive_report(days=7)
        
        # Clean up old data (if needed)
        # self.data_manager.cleanup_old_data(days=90)
        
        logger.info("Automated maintenance completed")

# Command-line interface
async def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="MASB-Alt System Orchestrator")
    parser.add_argument("command", choices=[
        "evaluate", "benchmark", "compare", "report", "finetune",
        "snapshot", "monitor", "import", "status", "maintain"
    ])
    
    # Command-specific arguments
    parser.add_argument("--models", nargs="+", help="Models to use")
    parser.add_argument("--languages", nargs="+", help="Languages to test")
    parser.add_argument("--domains", nargs="+", help="Domains to test")
    parser.add_argument("--datasets", nargs="+", help="Dataset IDs")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--output", help="Output file/directory")
    parser.add_argument("--days", type=int, default=7, help="Days to analyze")
    parser.add_argument("--prompts", type=int, default=5, help="Prompts per combination")
    parser.add_argument("--suite", choices=["quick", "standard", "comprehensive"], 
                       default="standard", help="Benchmark suite type")
    parser.add_argument("--report-type", choices=["executive", "technical", "comparative"],
                       default="executive", help="Report type")
    parser.add_argument("--format", choices=["html", "pdf", "docx", "markdown"],
                       default="html", help="Output format")
    parser.add_argument("--model", help="Model name for fine-tuning analysis")
    parser.add_argument("--file", help="File path for imports")
    parser.add_argument("--language", help="Language code")
    parser.add_argument("--domain", help="Domain name")
    parser.add_argument("--file-format", choices=["jsonl", "csv", "txt"],
                       default="jsonl", help="File format")
    
    args = parser.parse_args()
    
    # Initialize orchestrator
    orchestrator = MASBOrchestrator(args.config)
    
    # Execute command
    if args.command == "evaluate":
        result = await orchestrator.run_evaluation_suite(
            models=args.models,
            languages=args.languages,
            domains=args.domains,
            prompts_per_combination=args.prompts
        )
        print(f"Evaluation completed. Dataset ID: {result}")
    
    elif args.command == "benchmark":
        result = await orchestrator.run_benchmark_analysis(
            models=args.models,
            suite_type=args.suite
        )
        print(f"Benchmark completed. Report: {result}")
    
    elif args.command == "compare":
        result = orchestrator.run_comparative_analysis(args.datasets)
        print(f"Comparative analysis completed. ID: {result}")
    
    elif args.command == "report":
        result = orchestrator.generate_comprehensive_report(
            report_type=args.report_type,
            format=args.format,
            days=args.days
        )
        print(f"Report generated: {result}")
    
    elif args.command == "finetune":
        if not args.model:
            print("Error: --model required for fine-tuning analysis")
            return
        
        result = orchestrator.analyze_finetuning_recommendations(
            model=args.model,
            days=args.days
        )
        print(f"Fine-tuning analysis completed: {result}")
    
    elif args.command == "snapshot":
        result = orchestrator.create_system_snapshot()
        print(f"System snapshot created: {result}")
    
    elif args.command == "monitor":
        await orchestrator.start_monitoring_services()
        print("Monitoring services started. Check logs for details.")
    
    elif args.command == "import":
        if not all([args.file, args.language, args.domain]):
            print("Error: --file, --language, and --domain required for import")
            return
        
        count = orchestrator.import_corpus_data(
            language=args.language,
            domain=args.domain,
            file_path=args.file,
            file_format=args.file_format
        )
        print(f"Imported {count} corpus entries")
    
    elif args.command == "status":
        status = orchestrator.get_system_status()
        print(json.dumps(status, indent=2, default=str))
    
    elif args.command == "maintain":
        await orchestrator.run_automated_maintenance()
        print("Automated maintenance completed")

if __name__ == "__main__":
    asyncio.run(main())