# Advanced MASB-Alt usage examples

import asyncio
import sys
from pathlib import Path
import json
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from masb_orchestrator import MASBOrchestrator
from data_manager import DataManager
from comparative_analyzer import ComparativeAnalyzer
from report_generator import ReportGenerator, ReportConfig
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def custom_evaluation_pipeline():
    """
    Advanced example: Custom evaluation pipeline with specific configurations
    """
    print("🔧 Advanced Custom Evaluation Pipeline")
    print("=" * 50)
    
    try:
        # Initialize with custom configuration
        config_path = "examples/custom_config.json"
        orchestrator = MASBOrchestrator(config_path)
        
        # Create custom evaluation parameters
        evaluation_config = {
            "models": ["openai", "anthropic", "cohere"],
            "languages": ["en", "sw", "ar", "hi", "vi"],  # 5 languages
            "domains": ["healthcare", "education", "financial_literacy"],
            "prompts_per_combination": 10,
            "focus_areas": ["cultural_sensitivity", "medical_accuracy", "educational_appropriateness"]
        }
        
        print(f"📋 Configuration: {len(evaluation_config['models'])} models, {len(evaluation_config['languages'])} languages")
        print(f"🎯 Focus areas: {', '.join(evaluation_config['focus_areas'])}")
        
        # Run comprehensive evaluation
        dataset_id = await orchestrator.run_evaluation_suite(
            models=evaluation_config["models"],
            languages=evaluation_config["languages"],
            domains=evaluation_config["domains"],
            prompts_per_combination=evaluation_config["prompts_per_combination"]
        )
        
        print(f"✅ Custom evaluation completed: {dataset_id}")
        
        # Analyze results with specific focus
        print("🔍 Running focused analysis...")
        
        # Generate multiple report types
        report_types = ["executive", "technical", "comparative"]
        for report_type in report_types:
            report_path = orchestrator.generate_comprehensive_report(
                report_type=report_type,
                format="html",
                days=1
            )
            print(f"📊 {report_type.title()} report: {report_path}")
        
        return dataset_id
        
    except Exception as e:
        print(f"❌ Custom evaluation failed: {e}")
        logger.error(f"Custom evaluation failed: {e}", exc_info=True)
        return None

async def batch_model_comparison():
    """
    Advanced example: Comprehensive model comparison across multiple dimensions
    """
    print("\n📊 Advanced Model Comparison Analysis")
    print("=" * 50)
    
    try:
        orchestrator = MASBOrchestrator()
        
        # Run evaluations for different model configurations
        model_configs = [
            {"name": "openai_focused", "models": ["openai"], "temperature": 0.3},
            {"name": "anthropic_focused", "models": ["anthropic"], "temperature": 0.3},
            {"name": "mixed_models", "models": ["openai", "anthropic"], "temperature": 0.7}
        ]
        
        dataset_ids = []
        
        for config in model_configs:
            print(f"🎯 Running evaluation for: {config['name']}")
            
            dataset_id = await orchestrator.run_evaluation_suite(
                models=config["models"],
                languages=["en", "sw", "ar"],
                domains=["healthcare", "education"],
                prompts_per_combination=5
            )
            
            if dataset_id:
                dataset_ids.append(dataset_id)
                print(f"✅ Completed: {config['name']} -> {dataset_id}")
        
        # Run comparative analysis across all datasets
        if len(dataset_ids) > 1:
            print("🔍 Running cross-model comparative analysis...")
            comparison_id = orchestrator.run_comparative_analysis(dataset_ids)
            
            if comparison_id:
                print(f"📈 Comparative analysis completed: {comparison_id}")
            else:
                print("⚠️  Comparative analysis had insufficient data")
        
        return dataset_ids
        
    except Exception as e:
        print(f"❌ Model comparison failed: {e}")
        logger.error(f"Model comparison failed: {e}", exc_info=True)
        return []

async def longitudinal_analysis():
    """
    Advanced example: Longitudinal analysis tracking performance over time
    """
    print("\n📅 Longitudinal Performance Analysis")
    print("=" * 50)
    
    try:
        orchestrator = MASBOrchestrator()
        
        # Simulate multiple evaluation runs over time
        print("🕐 Simulating historical evaluations...")
        
        time_periods = [
            {"name": "week_1", "days_ago": 7},
            {"name": "week_2", "days_ago": 14},
            {"name": "current", "days_ago": 0}
        ]
        
        dataset_ids = []
        
        for period in time_periods:
            print(f"📊 Running evaluation for: {period['name']}")
            
            # Run smaller evaluation for time series
            dataset_id = await orchestrator.run_evaluation_suite(
                models=["openai", "anthropic"],
                languages=["en", "sw"],
                domains=["healthcare"],
                prompts_per_combination=3
            )
            
            if dataset_id:
                dataset_ids.append(dataset_id)
                print(f"✅ Period {period['name']}: {dataset_id}")
        
        # Generate historical analysis
        if dataset_ids:
            print("📈 Generating longitudinal report...")
            
            # Create comprehensive historical report
            report_path = orchestrator.generate_comprehensive_report(
                report_type="comparative",
                format="html",
                days=30  # Look back 30 days
            )
            
            print(f"📋 Historical analysis report: {report_path}")
            
            # Create system snapshot for tracking
            snapshot_id = orchestrator.create_system_snapshot()
            print(f"📸 System snapshot created: {snapshot_id}")
        
        return dataset_ids
        
    except Exception as e:
        print(f"❌ Longitudinal analysis failed: {e}")
        logger.error(f"Longitudinal analysis failed: {e}", exc_info=True)
        return []

async def specialized_domain_analysis():
    """
    Advanced example: Deep dive analysis for specific domains
    """
    print("\n🏥 Specialized Domain Analysis - Healthcare Focus")
    print("=" * 50)
    
    try:
        orchestrator = MASBOrchestrator()
        
        # Healthcare-focused evaluation
        healthcare_config = {
            "models": ["openai", "anthropic"],
            "languages": ["en", "sw", "ar", "hi"],  # Languages with different medical traditions
            "domains": ["healthcare"],  # Healthcare only
            "prompts_per_combination": 15,  # More prompts for deeper analysis
        }
        
        print("🏥 Running healthcare-focused evaluation...")
        print(f"   Languages: {', '.join(healthcare_config['languages'])}")
        print(f"   Prompts per language: {healthcare_config['prompts_per_combination']}")
        
        dataset_id = await orchestrator.run_evaluation_suite(**healthcare_config)
        
        if dataset_id:
            print(f"✅ Healthcare evaluation completed: {dataset_id}")
            
            # Generate specialized healthcare report
            print("📋 Generating specialized healthcare report...")
            
            report_config = ReportConfig(
                title="Healthcare AI Safety Analysis",
                subtitle="Cross-Cultural Medical Response Evaluation",
                report_type="technical",
                format="html"
            )
            
            # This would require extending the report generator for domain-specific reports
            print("📊 Healthcare-specific analysis would include:")
            print("   - Medical accuracy assessment")
            print("   - Cultural sensitivity in health advice")
            print("   - Disclaimer and safety warnings evaluation")
            print("   - Cross-cultural medical practice respect")
            
            # Run fine-tuning analysis specifically for healthcare
            print("🎯 Analyzing healthcare fine-tuning opportunities...")
            
            for model in healthcare_config["models"]:
                finetuning_path = orchestrator.analyze_finetuning_recommendations(
                    model=model,
                    days=30
                )
                
                if finetuning_path:
                    print(f"📋 {model} healthcare fine-tuning analysis: {finetuning_path}")
        
        return dataset_id
        
    except Exception as e:
        print(f"❌ Healthcare analysis failed: {e}")
        logger.error(f"Healthcare analysis failed: {e}", exc_info=True)
        return None

async def automated_monitoring_setup():
    """
    Advanced example: Setting up automated monitoring and alerting
    """
    print("\n📡 Automated Monitoring Setup")
    print("=" * 50)
    
    try:
        orchestrator = MASBOrchestrator()
        
        print("🔧 Setting up monitoring services...")
        
        # Start monitoring services
        await orchestrator.start_monitoring_services()
        
        print("📊 Monitoring services started:")
        print("   - API server (port 8000)")
        print("   - Dashboard (port 8501)")
        print("   - Real-time metrics collection")
        
        # Run automated maintenance
        print("🔧 Running automated maintenance...")
        await orchestrator.run_automated_maintenance()
        
        print("✅ Automated monitoring setup completed!")
        print("\nMonitoring features:")
        print("   - Real-time performance metrics")
        print("   - Anomaly detection")
        print("   - Historical trend analysis")
        print("   - Automated reporting")
        
        # Show system status
        status = orchestrator.get_system_status()
        print(f"\n📈 Current system status: {len(status['components'])} components active")
        
    except Exception as e:
        print(f"❌ Monitoring setup failed: {e}")
        logger.error(f"Monitoring setup failed: {e}", exc_info=True)

async def data_export_and_integration():
    """
    Advanced example: Data export and integration with external systems
    """
    print("\n💾 Data Export and Integration")
    print("=" * 50)
    
    try:
        # Initialize data manager directly for advanced operations
        data_manager = DataManager()
        
        print("📊 Available data export options:")
        print("   - CSV format for statistical analysis")
        print("   - JSON format for API integration")
        print("   - Database backup for archival")
        
        # Get database statistics
        try:
            stats = data_manager.get_database_stats()
            print(f"\n📈 Database statistics:")
            for key, value in stats.items():
                print(f"   {key}: {value}")
        except Exception as e:
            print(f"⚠️  Could not retrieve database stats: {e}")
        
        # Example export operations (would need actual data)
        print("\n💾 Export operations:")
        print("   - Full evaluation dataset export")
        print("   - Filtered results by language/domain")
        print("   - Performance metrics for external analysis")
        print("   - Integration ready JSON format")
        
        print("✅ Data export capabilities demonstrated!")
        
    except Exception as e:
        print(f"❌ Data export demo failed: {e}")
        logger.error(f"Data export demo failed: {e}", exc_info=True)

async def main():
    """
    Main function running all advanced examples
    """
    print("🌟 MASB-Alt Advanced Usage Examples")
    print("=" * 60)
    print("This script demonstrates advanced MASB-Alt capabilities")
    print()
    
    # Run advanced examples
    examples = [
        ("Custom Evaluation Pipeline", custom_evaluation_pipeline),
        ("Batch Model Comparison", batch_model_comparison),
        ("Longitudinal Analysis", longitudinal_analysis),
        ("Specialized Domain Analysis", specialized_domain_analysis),
        ("Automated Monitoring Setup", automated_monitoring_setup),
        ("Data Export and Integration", data_export_and_integration)
    ]
    
    results = {}
    
    for name, example_func in examples:
        try:
            print(f"\n🚀 Running: {name}")
            if asyncio.iscoroutinefunction(example_func):
                result = await example_func()
            else:
                result = example_func()
            results[name] = result
        except Exception as e:
            print(f"❌ {name} failed: {e}")
            results[name] = None
    
    print("\n" + "=" * 60)
    print("🎯 Advanced Examples Summary:")
    
    for name, result in results.items():
        status = "✅ Success" if result is not None else "❌ Failed"
        print(f"   {name}: {status}")
    
    print("\n📚 Next Steps:")
    print("   - Explore the generated reports and data")
    print("   - Customize configurations for your use case")
    print("   - Set up production monitoring")
    print("   - Integrate with your ML pipeline")

if __name__ == "__main__":
    # Run the advanced examples
    asyncio.run(main())