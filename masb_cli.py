#!/usr/bin/env python3
# masb_cli.py - Comprehensive CLI tool for MASB-Alt system

import sys
import os
import asyncio
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from masb_orchestrator import MASBOrchestrator
from data_manager import DataManager
from corpus_manager import CorpusManager
from history_tracker import HistoryTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MASBCLITool:
    """Comprehensive CLI tool for MASB-Alt system management"""
    
    def __init__(self):
        self.orchestrator = None
        self.data_manager = DataManager()
        self.corpus_manager = CorpusManager()
        self.history_tracker = HistoryTracker()
    
    def check_prerequisites(self) -> bool:
        """Check if system prerequisites are met"""
        print("🔍 Checking system prerequisites...")
        
        # Check Python version
        if sys.version_info < (3, 8):
            print("❌ Python 3.8+ required")
            return False
        print("✅ Python version OK")
        
        # Check environment variables
        required_vars = ['OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'COHERE_API_KEY']
        missing_vars = []
        
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            print(f"⚠️  Missing environment variables: {', '.join(missing_vars)}")
            print("   Add them to your .env file or environment")
        else:
            print("✅ Environment variables OK")
        
        # Check data directory
        data_dir = Path("./data")
        if not data_dir.exists():
            data_dir.mkdir(parents=True)
            print("✅ Created data directory")
        else:
            print("✅ Data directory exists")
        
        return True
    
    def initialize_system(self):
        """Initialize MASB-Alt system"""
        print("🚀 Initializing MASB-Alt system...")
        
        try:
            # Initialize orchestrator
            self.orchestrator = MASBOrchestrator()
            print("✅ Orchestrator initialized")
            
            # Initialize database
            self.data_manager = DataManager()
            print("✅ Database initialized")
            
            # Initialize corpus manager
            self.corpus_manager = CorpusManager()
            print("✅ Corpus manager initialized")
            
            # Initialize history tracker
            self.history_tracker = HistoryTracker()
            print("✅ History tracker initialized")
            
            print("🎉 System initialization complete!")
            return True
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            return False
    
    async def run_quick_demo(self):
        """Run a quick demonstration of the system"""
        print("🎯 Running MASB-Alt Quick Demo...")
        
        if not self.orchestrator:
            if not self.initialize_system():
                return
        
        try:
            # Run small evaluation
            print("\n📝 Running sample evaluation...")
            dataset_id = await self.orchestrator.run_evaluation_suite(
                models=["openai"],
                languages=["en", "sw"],
                domains=["healthcare"],
                prompts_per_combination=2
            )
            
            if dataset_id:
                print(f"✅ Evaluation completed! Dataset ID: {dataset_id}")
                
                # Generate quick report
                print("\n📊 Generating report...")
                report_path = self.orchestrator.generate_comprehensive_report(
                    report_type="executive",
                    format="html",
                    days=1
                )
                
                if report_path:
                    print(f"✅ Report generated: {report_path}")
                
                # Show system status
                print("\n📈 System Status:")
                status = self.orchestrator.get_system_status()
                print(f"   Total evaluations in DB: {status.get('data_stats', {}).get('total_evaluations', 'N/A')}")
                print(f"   Models available: {len(status.get('configuration', {}).get('models', []))}")
                print(f"   Languages supported: {len(status.get('configuration', {}).get('languages', []))}")
            
        except Exception as e:
            print(f"❌ Demo failed: {e}")
    
    def show_system_status(self):
        """Show comprehensive system status"""
        print("📊 MASB-Alt System Status")
        print("=" * 50)
        
        if not self.orchestrator:
            if not self.initialize_system():
                return
        
        try:
            status = self.orchestrator.get_system_status()
            
            print(f"🕐 Timestamp: {status['timestamp']}")
            print(f"🔧 Components: {len(status['components'])} active")
            print(f"🚀 Services: {len(status['services'])} running")
            
            if 'data_stats' in status and not isinstance(status['data_stats'], dict) or 'error' not in status['data_stats']:
                data_stats = status['data_stats']
                print(f"📊 Total Evaluations: {data_stats.get('total_evaluations', 'N/A')}")
                print(f"📝 Total Prompts: {data_stats.get('total_prompts', 'N/A')}")
                print(f"🤖 Models: {data_stats.get('unique_models', 'N/A')}")
            
            print(f"⚠️  Recent Anomalies: {status.get('recent_anomalies', 'N/A')}")
            
            # Show configuration
            config = status.get('configuration', {})
            print(f"\n🛠️  Configuration:")
            print(f"   Models: {', '.join(config.get('models', []))}")
            print(f"   Languages: {', '.join(config.get('languages', []))}")
            print(f"   Domains: {', '.join(config.get('domains', []))}")
            
        except Exception as e:
            print(f"❌ Failed to get status: {e}")
    
    def list_datasets(self):
        """List available datasets"""
        print("📚 Available Datasets")
        print("=" * 50)
        
        try:
            datasets = self.data_manager.list_datasets()
            
            if not datasets:
                print("No datasets found. Run an evaluation to create datasets.")
                return
            
            for dataset in datasets:
                print(f"📊 {dataset['dataset_id']}")
                print(f"   Name: {dataset['name']}")
                print(f"   Created: {dataset['created_at']}")
                print(f"   Prompts: {dataset.get('prompt_count', 'N/A')}")
                print()
                
        except Exception as e:
            print(f"❌ Failed to list datasets: {e}")
    
    def show_evaluation_results(self, dataset_id: str):
        """Show evaluation results for a dataset"""
        print(f"📈 Evaluation Results for {dataset_id}")
        print("=" * 50)
        
        try:
            # Get basic stats
            results = self.data_manager.get_evaluation_results(dataset_id)
            
            if not results:
                print("No results found for this dataset.")
                return
            
            # Calculate summary statistics
            total_evals = len(results)
            avg_score = sum(r['alignment_score'] for r in results) / total_evals
            
            # Count by model
            model_counts = {}
            model_scores = {}
            
            for result in results:
                model = result['model']
                model_counts[model] = model_counts.get(model, 0) + 1
                
                if model not in model_scores:
                    model_scores[model] = []
                model_scores[model].append(result['alignment_score'])
            
            print(f"📊 Summary:")
            print(f"   Total Evaluations: {total_evals}")
            print(f"   Average Score: {avg_score:.2f}/5")
            
            print(f"\n🤖 By Model:")
            for model, count in model_counts.items():
                avg_model_score = sum(model_scores[model]) / len(model_scores[model])
                print(f"   {model}: {count} evaluations, avg {avg_model_score:.2f}")
            
            # Show risk distribution
            risk_counts = {}
            for result in results:
                risk = result['risk_level']
                risk_counts[risk] = risk_counts.get(risk, 0) + 1
            
            print(f"\n⚠️  Risk Distribution:")
            for risk, count in risk_counts.items():
                percentage = (count / total_evals) * 100
                print(f"   {risk}: {count} ({percentage:.1f}%)")
                
        except Exception as e:
            print(f"❌ Failed to show results: {e}")
    
    def export_data(self, dataset_id: str, format: str = "csv", output_file: Optional[str] = None):
        """Export evaluation data"""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"masb_export_{dataset_id}_{timestamp}.{format}"
        
        print(f"📤 Exporting dataset {dataset_id} to {output_file}...")
        
        try:
            success = self.data_manager.export_evaluation_data(
                dataset_id=dataset_id,
                output_file=output_file,
                format=format
            )
            
            if success:
                print(f"✅ Export completed: {output_file}")
            else:
                print("❌ Export failed")
                
        except Exception as e:
            print(f"❌ Export failed: {e}")
    
    def cleanup_old_data(self, days: int = 30):
        """Clean up old evaluation data"""
        print(f"🧹 Cleaning up data older than {days} days...")
        
        try:
            count = self.data_manager.cleanup_old_data(days)
            print(f"✅ Cleaned up {count} old records")
            
        except Exception as e:
            print(f"❌ Cleanup failed: {e}")
    
    def import_corpus(self, file_path: str, language: str, domain: str, format: str = "jsonl"):
        """Import corpus data"""
        print(f"📥 Importing corpus from {file_path}...")
        
        try:
            count = self.corpus_manager.import_corpus_file(
                file_path=file_path,
                language=language,
                domain=domain,
                source="cli_import",
                file_format=format
            )
            
            print(f"✅ Imported {count} corpus entries")
            
        except Exception as e:
            print(f"❌ Import failed: {e}")
    
    def show_help(self):
        """Show comprehensive help"""
        help_text = """
🌍 MASB-Alt CLI Tool - Comprehensive Help

USAGE:
  python masb_cli.py <command> [options]

COMMANDS:
  
  System Management:
    check           Check system prerequisites
    init            Initialize MASB-Alt system
    status          Show system status
    demo            Run quick demonstration
    
  Data Management:
    datasets        List available datasets
    results <id>    Show evaluation results for dataset
    export <id>     Export dataset to file
    cleanup         Clean up old data
    
  Corpus Management:
    import <file>   Import corpus data
    corpus-stats    Show corpus statistics
    
  Evaluation:
    evaluate        Run evaluation (use orchestrator)
    benchmark       Run benchmarks (use orchestrator)
    
  Monitoring:
    monitor         Start monitoring services
    history         Show evaluation history
    anomalies       Check for anomalies
    
  Reporting:
    report          Generate reports (use orchestrator)

EXAMPLES:
  
  # System setup
  python masb_cli.py check
  python masb_cli.py init
  python masb_cli.py demo
  
  # View data
  python masb_cli.py status
  python masb_cli.py datasets
  python masb_cli.py results eval_20240101_120000
  
  # Export data
  python masb_cli.py export eval_20240101_120000 --format csv
  
  # Import corpus
  python masb_cli.py import corpus_sw.jsonl --language sw --domain healthcare
  
  # Cleanup
  python masb_cli.py cleanup --days 30

For more detailed operations, use the main orchestrator:
  python masb_orchestrator.py evaluate --models openai --languages en sw
  python masb_orchestrator.py benchmark --suite standard
  python masb_orchestrator.py report --report-type executive

Visit https://github.com/masb-alt/masb-alt for documentation.
        """
        print(help_text)

async def main():
    """Main CLI entry point"""
    if len(sys.argv) < 2:
        print("❌ No command specified. Use 'help' for usage information.")
        return
    
    command = sys.argv[1]
    cli = MASBCLITool()
    
    if command == "check":
        cli.check_prerequisites()
    
    elif command == "init":
        cli.initialize_system()
    
    elif command == "demo":
        await cli.run_quick_demo()
    
    elif command == "status":
        cli.show_system_status()
    
    elif command == "datasets":
        cli.list_datasets()
    
    elif command == "results":
        if len(sys.argv) < 3:
            print("❌ Dataset ID required. Usage: results <dataset_id>")
            return
        cli.show_evaluation_results(sys.argv[2])
    
    elif command == "export":
        if len(sys.argv) < 3:
            print("❌ Dataset ID required. Usage: export <dataset_id> [--format csv|json]")
            return
        
        dataset_id = sys.argv[2]
        format = "csv"
        output_file = None
        
        # Parse additional arguments
        for i, arg in enumerate(sys.argv[3:], 3):
            if arg == "--format" and i + 1 < len(sys.argv):
                format = sys.argv[i + 1]
            elif arg == "--output" and i + 1 < len(sys.argv):
                output_file = sys.argv[i + 1]
        
        cli.export_data(dataset_id, format, output_file)
    
    elif command == "cleanup":
        days = 30
        if "--days" in sys.argv:
            idx = sys.argv.index("--days")
            if idx + 1 < len(sys.argv):
                days = int(sys.argv[idx + 1])
        
        cli.cleanup_old_data(days)
    
    elif command == "import":
        if len(sys.argv) < 3:
            print("❌ File path required. Usage: import <file> --language <lang> --domain <domain>")
            return
        
        file_path = sys.argv[2]
        language = None
        domain = None
        format = "jsonl"
        
        # Parse arguments
        for i, arg in enumerate(sys.argv[3:], 3):
            if arg == "--language" and i + 1 < len(sys.argv):
                language = sys.argv[i + 1]
            elif arg == "--domain" and i + 1 < len(sys.argv):
                domain = sys.argv[i + 1]
            elif arg == "--format" and i + 1 < len(sys.argv):
                format = sys.argv[i + 1]
        
        if not language or not domain:
            print("❌ Both --language and --domain required")
            return
        
        cli.import_corpus(file_path, language, domain, format)
    
    elif command == "corpus-stats":
        stats = cli.corpus_manager.get_statistics()
        if not stats.empty:
            print("📊 Corpus Statistics:")
            print(stats.to_string())
        else:
            print("No corpus data available")
    
    elif command == "history":
        summary = cli.history_tracker.get_history_summary()
        print("📈 Evaluation History Summary:")
        print(json.dumps(summary, indent=2, default=str))
    
    elif command == "anomalies":
        anomalies = cli.history_tracker.detect_anomalies()
        if anomalies:
            print(f"⚠️  Detected {len(anomalies)} anomalies:")
            for anomaly in anomalies:
                print(f"   [{anomaly.severity.upper()}] {anomaly.description}")
        else:
            print("✅ No anomalies detected")
    
    elif command == "help":
        cli.show_help()
    
    else:
        print(f"❌ Unknown command: {command}")
        print("Use 'help' for available commands")

if __name__ == "__main__":
    asyncio.run(main())