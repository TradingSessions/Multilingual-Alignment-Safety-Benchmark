#!/usr/bin/env python3
# run_evaluation.py - Automated evaluation pipeline

import os
import sys
import json
import argparse
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import asyncio
from tqdm import tqdm
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prompt_generator.multilingual_prompt_gen import MultilingualPromptGenerator
from llm_api_client import LLMClientFactory, MultiLLMClient
from evaluation.evaluator import MultilingualAlignmentEvaluator
from data_manager import DataManager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class EvaluationPipeline:
    """Automated evaluation pipeline for multilingual alignment"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.prompt_generator = MultilingualPromptGenerator()
        self.evaluator = MultilingualAlignmentEvaluator()
        self.data_manager = DataManager(config.get("data_path", "./data"))
        
        # Initialize LLM clients
        self._init_llm_clients()
        
    def _init_llm_clients(self):
        """Initialize LLM clients based on configuration"""
        providers = self.config.get("llm_providers", ["openai", "anthropic", "cohere"])
        self.llm_client = LLMClientFactory.create_multi_client(providers)
        logger.info(f"Initialized LLM clients for: {providers}")
    
    def generate_prompts(self, 
                        domains: Optional[List[str]] = None,
                        languages: Optional[List[str]] = None,
                        num_prompts_per_combination: int = 5) -> List[Dict]:
        """Generate prompts for evaluation"""
        if not domains:
            domains = self.prompt_generator.get_domains()
        
        if not languages:
            languages = list(self.prompt_generator.get_supported_languages().keys())
        
        prompts = []
        prompt_id_counter = 1
        
        for domain in domains:
            domain_prompts = self.prompt_generator.get_all_prompts(domain)
            
            for language in languages:
                # Select prompts for this language
                selected_count = 0
                for template in domain_prompts:
                    if selected_count >= num_prompts_per_combination:
                        break
                    
                    # Extract prompt for specific language
                    if "texts" in template and language in template["texts"]:
                        prompt_text = template["texts"][language]
                        if prompt_text:  # Only add if text exists for this language
                            prompt = {
                                "id": f"auto_{prompt_id_counter:04d}",
                                "language": language,
                                "domain": domain,
                                "text": prompt_text,
                                "risk_level": template.get("risk_level", "unknown"),
                                "tags": template.get("tags", [])
                            }
                            prompts.append(prompt)
                            prompt_id_counter += 1
                            selected_count += 1
        
        logger.info(f"Generated {len(prompts)} prompts across {len(domains)} domains and {len(languages)} languages")
        return prompts
    
    async def collect_responses(self, prompts: List[Dict], save_intermediate: bool = True) -> List[Dict]:
        """Collect responses from all LLM providers"""
        results = []
        
        # Create dataset for this evaluation run
        dataset_id = self.data_manager.create_dataset(
            name=f"Evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            description=f"Automated evaluation with {len(prompts)} prompts",
            prompts=prompts
        )
        
        logger.info(f"Created dataset: {dataset_id}")
        
        # Process prompts with progress bar
        with tqdm(total=len(prompts), desc="Collecting responses") as pbar:
            for prompt in prompts:
                prompt_id = self.data_manager.add_prompt(prompt, dataset_id)
                
                # Get responses from all models
                try:
                    responses = await self.llm_client.generate_all_async(
                        prompt["text"],
                        max_tokens=self.config.get("max_tokens", 1000),
                        temperature=self.config.get("temperature", 0.7)
                    )
                    
                    for response in responses:
                        if not response.error:
                            # Save response
                            response_data = {
                                "model": response.model,
                                "response": response.response,
                                "timestamp": response.timestamp.isoformat(),
                                "latency_ms": response.latency_ms,
                                "token_count": response.token_count,
                                "error": response.error
                            }
                            
                            response_id = self.data_manager.add_response(response_data, prompt_id)
                            
                            # Prepare for evaluation
                            result = {
                                "prompt": prompt,
                                "response_id": response_id,
                                "llm_output": response.response,
                                "llm_model": response.model
                            }
                            results.append(result)
                        else:
                            logger.error(f"Error from {response.model}: {response.error}")
                
                except Exception as e:
                    logger.error(f"Error processing prompt {prompt_id}: {str(e)}")
                
                pbar.update(1)
                
                # Save intermediate results
                if save_intermediate and len(results) % 10 == 0:
                    self._save_intermediate_results(results, dataset_id)
        
        logger.info(f"Collected {len(results)} responses")
        return results
    
    def evaluate_responses(self, response_data: List[Dict]) -> List[Dict]:
        """Evaluate all collected responses"""
        evaluation_results = []
        
        with tqdm(total=len(response_data), desc="Evaluating responses") as pbar:
            for data in response_data:
                try:
                    # Run evaluation
                    eval_result = self.evaluator.evaluate_response(
                        prompt=data["prompt"],
                        llm_output=data["llm_output"],
                        llm_model=data["llm_model"],
                        evaluator_id="automated"
                    )
                    
                    # Save evaluation to database
                    eval_data = eval_result.to_dict()
                    self.data_manager.add_evaluation(eval_data, data["response_id"])
                    
                    evaluation_results.append(eval_result)
                    
                except Exception as e:
                    logger.error(f"Error evaluating response: {str(e)}")
                
                pbar.update(1)
        
        logger.info(f"Completed {len(evaluation_results)} evaluations")
        return evaluation_results
    
    def generate_report(self, evaluation_results: List[Dict], output_dir: str = "./reports"):
        """Generate comprehensive evaluation report"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate summary report
        summary = self.evaluator.generate_summary_report(evaluation_results)
        
        # Add pipeline metadata
        summary["pipeline_metadata"] = {
            "timestamp": timestamp,
            "total_prompts": len(set(r.prompt_id for r in evaluation_results)),
            "total_evaluations": len(evaluation_results),
            "llm_providers": list(set(r.llm_model for r in evaluation_results)),
            "languages": list(set(r.language for r in evaluation_results)),
            "domains": list(set(r.domain for r in evaluation_results))
        }
        
        # Save summary report
        summary_file = output_path / f"summary_report_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        # Save detailed results
        detailed_file = output_path / f"detailed_results_{timestamp}.json"
        self.evaluator.save_results(evaluation_results, str(detailed_file))
        
        # Generate markdown report
        self._generate_markdown_report(summary, evaluation_results, output_path / f"report_{timestamp}.md")
        
        # Generate CSV for analysis
        self._generate_csv_report(evaluation_results, output_path / f"results_{timestamp}.csv")
        
        logger.info(f"Reports generated in {output_path}")
        return str(summary_file)
    
    def _generate_markdown_report(self, summary: Dict, results: List, output_file: Path):
        """Generate human-readable markdown report"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Multilingual Alignment Safety Benchmark - Evaluation Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write(f"- **Total Evaluations:** {summary['total_evaluations']}\n")
            f.write(f"- **Average Alignment Score:** {summary['average_alignment_score']:.2f}/5\n")
            f.write(f"- **Languages Tested:** {', '.join(summary['by_language'].keys())}\n")
            f.write(f"- **Domains Tested:** {', '.join(summary['by_domain'].keys())}\n\n")
            
            # Risk Summary
            f.write("## Risk Analysis\n\n")
            f.write("### Risk Level Distribution\n")
            for level, count in summary['by_risk_level'].items():
                percentage = (count / summary['total_evaluations']) * 100
                f.write(f"- **{level.upper()}:** {count} ({percentage:.1f}%)\n")
            
            f.write("\n### Most Common Risk Flags\n")
            for flag, count in sorted(summary['risk_flag_frequency'].items(), 
                                     key=lambda x: x[1], reverse=True)[:5]:
                f.write(f"- {flag.replace('_', ' ').title()}: {count} occurrences\n")
            
            # Performance by Language
            f.write("\n## Performance by Language\n\n")
            f.write("| Language | Evaluations | Avg Score |\n")
            f.write("|----------|-------------|----------|\n")
            for lang, data in summary['by_language'].items():
                f.write(f"| {lang} | {data['count']} | {data['avg_score']:.2f} |\n")
            
            # Performance by Domain
            f.write("\n## Performance by Domain\n\n")
            f.write("| Domain | Evaluations | Avg Score |\n")
            f.write("|--------|-------------|----------|\n")
            for domain, data in summary['by_domain'].items():
                f.write(f"| {domain} | {data['count']} | {data['avg_score']:.2f} |\n")
            
            # Model Comparison
            f.write("\n## Model Comparison\n\n")
            f.write("| Model | Evaluations | Avg Score |\n")
            f.write("|-------|-------------|----------|\n")
            for model, data in summary['model_comparison'].items():
                f.write(f"| {model} | {data['count']} | {data['avg_score']:.2f} |\n")
            
            # Sample Issues
            f.write("\n## Sample Issues Identified\n\n")
            
            # Find some problematic examples
            problematic = [r for r in results if r.alignment_score.value <= 2][:5]
            for i, result in enumerate(problematic, 1):
                f.write(f"### Issue {i}\n")
                f.write(f"- **Language:** {result.language}\n")
                f.write(f"- **Domain:** {result.domain}\n")
                f.write(f"- **Model:** {result.llm_model}\n")
                f.write(f"- **Score:** {result.alignment_score.value}/5\n")
                f.write(f"- **Risk Flags:** {', '.join(result.risk_flags.get_flagged_risks())}\n")
                f.write(f"- **Comments:** {result.comments}\n\n")
    
    def _generate_csv_report(self, results: List, output_file: Path):
        """Generate CSV report for further analysis"""
        import csv
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = [
                'prompt_id', 'language', 'domain', 'model', 'alignment_score',
                'risk_level', 'risk_flags', 'confidence_score', 'timestamp'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                writer.writerow({
                    'prompt_id': result.prompt_id,
                    'language': result.language,
                    'domain': result.domain,
                    'model': result.llm_model,
                    'alignment_score': result.alignment_score.value,
                    'risk_level': result.risk_level.value,
                    'risk_flags': ';'.join(result.risk_flags.get_flagged_risks()),
                    'confidence_score': result.confidence_score,
                    'timestamp': result.timestamp.isoformat()
                })
    
    def _save_intermediate_results(self, results: List[Dict], dataset_id: str):
        """Save intermediate results during processing"""
        intermediate_file = f"intermediate_{dataset_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(intermediate_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved intermediate results to {intermediate_file}")
    
    async def run_full_evaluation(self,
                                 domains: Optional[List[str]] = None,
                                 languages: Optional[List[str]] = None,
                                 num_prompts: int = 5):
        """Run complete evaluation pipeline"""
        logger.info("Starting full evaluation pipeline")
        
        # Step 1: Generate prompts
        logger.info("Step 1: Generating prompts")
        prompts = self.generate_prompts(domains, languages, num_prompts)
        
        # Step 2: Collect responses
        logger.info("Step 2: Collecting LLM responses")
        response_data = await self.collect_responses(prompts)
        
        # Step 3: Evaluate responses
        logger.info("Step 3: Evaluating responses")
        evaluation_results = self.evaluate_responses(response_data)
        
        # Step 4: Generate reports
        logger.info("Step 4: Generating reports")
        report_path = self.generate_report(evaluation_results)
        
        # Step 5: Generate database summary
        logger.info("Step 5: Updating database statistics")
        db_stats = self.data_manager.generate_summary_statistics()
        
        logger.info("Evaluation pipeline completed successfully")
        
        return {
            "prompts_generated": len(prompts),
            "responses_collected": len(response_data),
            "evaluations_completed": len(evaluation_results),
            "report_path": report_path,
            "database_stats": db_stats
        }

def main():
    """Main entry point for command-line usage"""
    parser = argparse.ArgumentParser(description="Run automated multilingual alignment evaluation")
    
    parser.add_argument("--domains", nargs="+", 
                       help="Domains to evaluate (default: all)")
    parser.add_argument("--languages", nargs="+",
                       help="Languages to evaluate (default: all)")
    parser.add_argument("--num-prompts", type=int, default=5,
                       help="Number of prompts per domain/language combination")
    parser.add_argument("--config", type=str, default=".env",
                       help="Configuration file path")
    parser.add_argument("--data-path", type=str, default="./data",
                       help="Data storage path")
    parser.add_argument("--dry-run", action="store_true",
                       help="Generate prompts only, don't call LLMs")
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and os.path.exists(args.config):
        load_dotenv(args.config)
    
    config = {
        "llm_providers": os.getenv("LLM_PROVIDERS", "openai,anthropic,cohere").split(","),
        "max_tokens": int(os.getenv("DEFAULT_MAX_TOKENS", 1000)),
        "temperature": float(os.getenv("DEFAULT_TEMPERATURE", 0.7)),
        "data_path": args.data_path
    }
    
    # Initialize pipeline
    pipeline = EvaluationPipeline(config)
    
    if args.dry_run:
        # Just generate and display prompts
        prompts = pipeline.generate_prompts(args.domains, args.languages, args.num_prompts)
        print(f"\nGenerated {len(prompts)} prompts:")
        for i, prompt in enumerate(prompts[:10]):  # Show first 10
            print(f"\n{i+1}. [{prompt['language']}] {prompt['domain']}")
            print(f"   {prompt['text'][:100]}...")
        if len(prompts) > 10:
            print(f"\n... and {len(prompts) - 10} more prompts")
    else:
        # Run full evaluation
        try:
            loop = asyncio.get_event_loop()
            results = loop.run_until_complete(
                pipeline.run_full_evaluation(args.domains, args.languages, args.num_prompts)
            )
            
            print("\n=== Evaluation Complete ===")
            print(f"Prompts Generated: {results['prompts_generated']}")
            print(f"Responses Collected: {results['responses_collected']}")
            print(f"Evaluations Completed: {results['evaluations_completed']}")
            print(f"Report saved to: {results['report_path']}")
            
        except KeyboardInterrupt:
            logger.info("Evaluation interrupted by user")
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise

if __name__ == "__main__":
    main()