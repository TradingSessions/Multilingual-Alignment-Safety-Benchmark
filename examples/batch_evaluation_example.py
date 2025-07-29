#!/usr/bin/env python3
"""
batch_evaluation_example.py - Example of batch evaluation workflow

This example shows how to:
1. Load prompts from a file
2. Collect responses from multiple LLMs
3. Run batch evaluation
4. Generate comprehensive reports
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from prompt_generator.multilingual_prompt_gen import MultilingualPromptGenerator
from llm_api_client import LLMClientFactory, MultiLLMClient
from evaluation.evaluator import MultilingualAlignmentEvaluator
from data_manager import DataManager

async def run_batch_evaluation():
    """Run a complete batch evaluation workflow"""
    
    print("=== Batch Evaluation Example ===\n")
    
    # Step 1: Initialize components
    print("Step 1: Initializing components...")
    generator = MultilingualPromptGenerator()
    evaluator = MultilingualAlignmentEvaluator()
    data_manager = DataManager("./example_data")
    
    # Step 2: Generate evaluation prompts
    print("\nStep 2: Generating evaluation prompts...")
    prompts = []
    
    # Generate prompts for each domain and language combination
    domains = ["healthcare", "education", "financial_literacy"]
    languages = ["en", "sw", "ar"]
    
    for domain in domains:
        for language in languages:
            # Get 2 prompts per combination
            domain_prompts = generator.get_all_prompts(domain, language)[:2]
            prompts.extend(domain_prompts)
    
    print(f"Generated {len(prompts)} prompts")
    
    # Create dataset
    dataset_id = data_manager.create_dataset(
        name=f"Batch Evaluation {datetime.now().strftime('%Y%m%d')}",
        description="Example batch evaluation across multiple domains and languages",
        prompts=prompts
    )
    
    # Step 3: Mock LLM responses (in real usage, use actual API clients)
    print("\nStep 3: Collecting LLM responses...")
    
    # Mock response generator
    def generate_mock_response(prompt_text, model_name):
        """Generate mock responses for demonstration"""
        responses = {
            "gpt-4": {
                "healthcare": "It's important to consult with a healthcare professional for personalized medical advice. General wellness includes regular check-ups, balanced nutrition, and appropriate exercise.",
                "education": "Effective learning strategies include active engagement, spaced repetition, and connecting new information to existing knowledge. Each learner may benefit from different approaches.",
                "financial_literacy": "Financial planning involves budgeting, saving, and making informed decisions about spending and investments. Consider seeking advice from qualified financial advisors."
            },
            "claude-3": {
                "healthcare": "For health concerns, professional medical guidance is essential. Preventive care, healthy lifestyle choices, and regular monitoring contribute to overall well-being.",
                "education": "Learning is enhanced through diverse methods including visual aids, hands-on practice, and collaborative activities. Understanding your learning style can improve outcomes.",
                "financial_literacy": "Sound financial management includes tracking expenses, building emergency funds, and understanding basic investment principles. Education about financial concepts is valuable."
            }
        }
        
        # Determine domain from prompt
        domain = "healthcare"  # Default
        if "learn" in prompt_text.lower() or "education" in prompt_text.lower():
            domain = "education"
        elif "money" in prompt_text.lower() or "financial" in prompt_text.lower():
            domain = "financial_literacy"
        
        return responses.get(model_name, {}).get(domain, "This is a mock response for demonstration purposes.")
    
    # Collect responses
    response_data = []
    models = ["gpt-4", "claude-3"]
    
    for prompt in prompts:
        prompt_id = data_manager.add_prompt(prompt, dataset_id)
        
        for model in models:
            # Generate mock response
            response_text = generate_mock_response(prompt["text"], model)
            
            # Save response
            response_info = {
                "model": model,
                "response": response_text,
                "timestamp": datetime.now().isoformat(),
                "latency_ms": 150,  # Mock latency
                "token_count": len(response_text.split())
            }
            
            response_id = data_manager.add_response(response_info, prompt_id)
            
            # Prepare for evaluation
            response_data.append({
                "prompt": prompt,
                "response_id": response_id,
                "llm_output": response_text,
                "llm_model": model
            })
    
    print(f"Collected {len(response_data)} responses from {len(models)} models")
    
    # Step 4: Run evaluations
    print("\nStep 4: Running evaluations...")
    evaluation_results = []
    
    for data in response_data:
        # Evaluate response
        result = evaluator.evaluate_response(
            prompt=data["prompt"],
            llm_output=data["llm_output"],
            llm_model=data["llm_model"],
            evaluator_id="batch_example"
        )
        
        # Save evaluation
        eval_data = result.to_dict()
        data_manager.add_evaluation(eval_data, data["response_id"])
        
        evaluation_results.append(result)
    
    print(f"Completed {len(evaluation_results)} evaluations")
    
    # Step 5: Generate reports
    print("\nStep 5: Generating reports...")
    
    # Summary statistics
    summary = evaluator.generate_summary_report(evaluation_results)
    
    print("\n=== Evaluation Summary ===")
    print(f"Total Evaluations: {summary['total_evaluations']}")
    print(f"Average Alignment Score: {summary['average_alignment_score']:.2f}/5")
    
    print("\nScores by Language:")
    for lang, data in summary['by_language'].items():
        print(f"  {lang}: {data['avg_score']:.2f} (n={data['count']})")
    
    print("\nScores by Domain:")
    for domain, data in summary['by_domain'].items():
        print(f"  {domain}: {data['avg_score']:.2f} (n={data['count']})")
    
    print("\nScores by Model:")
    for model, data in summary['model_comparison'].items():
        print(f"  {model}: {data['avg_score']:.2f} (n={data['count']})")
    
    print("\nRisk Level Distribution:")
    for level, count in summary['by_risk_level'].items():
        percentage = (count / summary['total_evaluations']) * 100
        print(f"  {level}: {count} ({percentage:.1f}%)")
    
    # Save detailed report
    report_path = Path("./example_reports")
    report_path.mkdir(exist_ok=True)
    
    report_file = report_path / f"batch_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": summary,
            "dataset_id": dataset_id,
            "evaluation_details": [r.to_dict() for r in evaluation_results]
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nDetailed report saved to: {report_file}")
    
    # Step 6: Export for visualization
    print("\nStep 6: Exporting data for visualization...")
    export_file = data_manager.export_dataset_for_evaluation(
        dataset_id,
        f"batch_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    print(f"Data exported to: {export_file}")
    
    # Clean up (optional)
    # import shutil
    # shutil.rmtree("./example_data", ignore_errors=True)
    # shutil.rmtree("./example_reports", ignore_errors=True)
    
    return summary

def main():
    """Main entry point"""
    print("=" * 60)
    print("MASB-Alt: Batch Evaluation Example")
    print("=" * 60)
    
    # Run async evaluation
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        summary = loop.run_until_complete(run_batch_evaluation())
        print("\n✅ Batch evaluation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {str(e)}")
        raise
    
    finally:
        loop.close()

if __name__ == "__main__":
    main()