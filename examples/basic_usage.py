# Example usage of MASB-Alt system - Basic Evaluation

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from masb_orchestrator import MASBOrchestrator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def basic_evaluation_example():
    """
    Basic example showing how to run a simple evaluation with MASB-Alt
    """
    print("🚀 MASB-Alt Basic Evaluation Example")
    print("=" * 50)
    
    try:
        # Initialize the orchestrator
        print("📋 Initializing MASB-Alt system...")
        orchestrator = MASBOrchestrator()
        
        # Check system status
        print("🔍 Checking system status...")
        status = orchestrator.get_system_status()
        print(f"✅ System components active: {len(status['components'])}")
        
        # Run a small evaluation suite
        print("🎯 Running evaluation suite...")
        print("   Models: OpenAI, Anthropic")
        print("   Languages: English, Swahili")
        print("   Domains: Healthcare, Education")
        print("   Prompts per combination: 3")
        
        dataset_id = await orchestrator.run_evaluation_suite(
            models=["openai", "anthropic"],
            languages=["en", "sw"],
            domains=["healthcare", "education"],
            prompts_per_combination=3
        )
        
        print(f"✅ Evaluation completed! Dataset ID: {dataset_id}")
        
        # Generate a report
        print("📊 Generating evaluation report...")
        report_path = orchestrator.generate_comprehensive_report(
            report_type="technical",
            format="html",
            days=1
        )
        
        print(f"📄 Report generated: {report_path}")
        
        # Run comparative analysis
        print("🔍 Running comparative analysis...")
        comparison_id = orchestrator.run_comparative_analysis([dataset_id])
        
        if comparison_id:
            print(f"📈 Comparative analysis completed: {comparison_id}")
        else:
            print("⚠️  Comparative analysis skipped (insufficient data)")
        
        print("\n🎉 Basic evaluation example completed successfully!")
        print("\nNext steps:")
        print("- Check the generated report for detailed results")
        print("- View the monitoring dashboard with: python masb_orchestrator.py monitor")
        print("- Run benchmark tests with: python masb_orchestrator.py benchmark")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        logger.error(f"Evaluation failed: {e}", exc_info=True)

async def benchmark_example():
    """
    Example showing how to run performance benchmarks
    """
    print("\n🏃 MASB-Alt Benchmark Example")
    print("=" * 50)
    
    try:
        orchestrator = MASBOrchestrator()
        
        print("⚡ Running performance benchmarks...")
        print("   Suite type: Quick")
        print("   Models: OpenAI, Anthropic")
        
        report_path = await orchestrator.run_benchmark_analysis(
            models=["openai", "anthropic"],
            suite_type="quick"
        )
        
        print(f"📊 Benchmark report generated: {report_path}")
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        logger.error(f"Benchmark failed: {e}", exc_info=True)

def status_check_example():
    """
    Example showing system status checking
    """
    print("\n🔍 MASB-Alt System Status Example")
    print("=" * 50)
    
    try:
        orchestrator = MASBOrchestrator()
        
        # Get comprehensive system status
        status = orchestrator.get_system_status()
        
        print("📊 System Status:")
        print(f"   Timestamp: {status['timestamp']}")
        
        print("\n🔧 Components:")
        for component, state in status['components'].items():
            print(f"   {component}: {state}")
        
        print("\n🌐 Services:")
        for service, state in status['services'].items():
            print(f"   {service}: {state}")
        
        if 'data_stats' in status and 'error' not in status['data_stats']:
            print("\n📈 Data Statistics:")
            for key, value in status['data_stats'].items():
                print(f"   {key}: {value}")
        
        print("\n✅ Status check completed!")
        
    except Exception as e:
        print(f"❌ Status check failed: {e}")
        logger.error(f"Status check failed: {e}", exc_info=True)

async def finetuning_analysis_example():
    """
    Example showing fine-tuning analysis
    """
    print("\n🎯 MASB-Alt Fine-tuning Analysis Example")
    print("=" * 50)
    
    try:
        orchestrator = MASBOrchestrator()
        
        print("🔍 Analyzing fine-tuning opportunities...")
        print("   Model: OpenAI")
        print("   Analysis period: 30 days")
        
        result_path = orchestrator.analyze_finetuning_recommendations(
            model="openai",
            days=30
        )
        
        if result_path:
            print(f"📋 Fine-tuning analysis completed: {result_path}")
        else:
            print("⚠️  Fine-tuning analysis skipped (insufficient data)")
        
    except Exception as e:
        print(f"❌ Fine-tuning analysis failed: {e}")
        logger.error(f"Fine-tuning analysis failed: {e}", exc_info=True)

async def main():
    """
    Main function running all examples
    """
    print("🌟 MASB-Alt System Examples")
    print("=" * 60)
    print("This script demonstrates various MASB-Alt capabilities")
    print()
    
    # Check system status first
    status_check_example()
    
    # Run basic evaluation
    await basic_evaluation_example()
    
    # Run benchmark example
    await benchmark_example()
    
    # Run fine-tuning analysis
    await finetuning_analysis_example()
    
    print("\n" + "=" * 60)
    print("🎯 All examples completed!")
    print("\nFor more advanced usage:")
    print("- Check the USER_GUIDE.md for detailed instructions")
    print("- Run 'python masb_cli.py help' for CLI options")
    print("- Visit the documentation for API reference")

if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())