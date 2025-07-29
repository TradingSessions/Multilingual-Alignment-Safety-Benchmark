# benchmark_runner.py - Model performance benchmarking system

import asyncio
import time
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent))

from prompt_generator.multilingual_prompt_gen import MultilingualPromptGenerator
from llm_api_client import LLMClientFactory, LLMClient, LLMResponse
from evaluation.evaluator import MultilingualAlignmentEvaluator
from data_manager import DataManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Single benchmark test result"""
    model: str
    test_name: str
    language: str
    domain: str
    prompt_count: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_per_second: float
    avg_tokens_per_response: float
    avg_alignment_score: float
    error_rate: float
    memory_usage_mb: Optional[float] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results"""
    suite_name: str
    start_time: datetime
    end_time: datetime
    total_duration_seconds: float
    models_tested: List[str]
    languages_tested: List[str]
    domains_tested: List[str]
    total_prompts: int
    results: List[BenchmarkResult]
    system_info: Dict[str, Any]

class ModelBenchmark:
    """Comprehensive model benchmarking system"""
    
    def __init__(self, output_dir: str = "./benchmarks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.prompt_generator = MultilingualPromptGenerator()
        self.evaluator = MultilingualAlignmentEvaluator()
        self.data_manager = DataManager()
        
        # Benchmark configurations
        self.benchmark_configs = {
            "quick": {
                "prompts_per_test": 10,
                "languages": ["en"],
                "domains": ["healthcare", "education"],
                "iterations": 1
            },
            "standard": {
                "prompts_per_test": 50,
                "languages": ["en", "sw", "ar"],
                "domains": ["healthcare", "education", "financial_literacy"],
                "iterations": 3
            },
            "comprehensive": {
                "prompts_per_test": 100,
                "languages": ["en", "sw", "ar", "hi", "vi"],
                "domains": list(self.prompt_generator.get_domains()),
                "iterations": 5
            }
        }
    
    async def run_latency_test(self, 
                              client: LLMClient,
                              prompts: List[str],
                              iterations: int = 3) -> Dict[str, float]:
        """Test response latency for a model"""
        latencies = []
        
        for _ in range(iterations):
            for prompt in prompts:
                start_time = time.time()
                try:
                    response = await client.generate_async(prompt)
                    if not response.error:
                        latency = (time.time() - start_time) * 1000
                        latencies.append(latency)
                except Exception as e:
                    logger.error(f"Error in latency test: {e}")
        
        if not latencies:
            return {"error": "No successful responses"}
        
        return {
            "avg_latency_ms": np.mean(latencies),
            "p50_latency_ms": np.percentile(latencies, 50),
            "p95_latency_ms": np.percentile(latencies, 95),
            "p99_latency_ms": np.percentile(latencies, 99),
            "min_latency_ms": np.min(latencies),
            "max_latency_ms": np.max(latencies),
            "std_latency_ms": np.std(latencies)
        }
    
    async def run_throughput_test(self,
                                 client: LLMClient,
                                 prompts: List[str],
                                 concurrent_requests: int = 5) -> Dict[str, float]:
        """Test throughput with concurrent requests"""
        start_time = time.time()
        successful_responses = 0
        total_tokens = 0
        
        # Create batches for concurrent processing
        batches = [prompts[i:i+concurrent_requests] 
                  for i in range(0, len(prompts), concurrent_requests)]
        
        for batch in batches:
            tasks = [client.generate_async(prompt) for prompt in batch]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            for response in responses:
                if isinstance(response, LLMResponse) and not response.error:
                    successful_responses += 1
                    if response.token_count:
                        total_tokens += response.token_count
        
        duration = time.time() - start_time
        
        return {
            "total_requests": len(prompts),
            "successful_responses": successful_responses,
            "duration_seconds": duration,
            "throughput_per_second": successful_responses / duration if duration > 0 else 0,
            "avg_tokens_per_response": total_tokens / successful_responses if successful_responses > 0 else 0,
            "success_rate": successful_responses / len(prompts) if prompts else 0
        }
    
    async def run_quality_test(self,
                              client: LLMClient,
                              test_cases: List[Dict]) -> Dict[str, float]:
        """Test response quality and alignment"""
        alignment_scores = []
        risk_levels = []
        risk_flag_counts = {}
        
        for test_case in test_cases:
            prompt = test_case["prompt"]
            
            try:
                response = await client.generate_async(prompt["text"])
                if not response.error:
                    # Evaluate response
                    eval_result = self.evaluator.evaluate_response(
                        prompt=prompt,
                        llm_output=response.response,
                        llm_model=client.model_name
                    )
                    
                    alignment_scores.append(eval_result.alignment_score.value)
                    risk_levels.append(eval_result.risk_level.value)
                    
                    # Count risk flags
                    for flag in eval_result.risk_flags.get_flagged_risks():
                        risk_flag_counts[flag] = risk_flag_counts.get(flag, 0) + 1
            
            except Exception as e:
                logger.error(f"Error in quality test: {e}")
        
        if not alignment_scores:
            return {"error": "No successful evaluations"}
        
        return {
            "avg_alignment_score": np.mean(alignment_scores),
            "min_alignment_score": np.min(alignment_scores),
            "max_alignment_score": np.max(alignment_scores),
            "std_alignment_score": np.std(alignment_scores),
            "high_risk_percentage": (risk_levels.count("high") + risk_levels.count("critical")) / len(risk_levels) * 100,
            "most_common_risks": sorted(risk_flag_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        }
    
    async def run_consistency_test(self,
                                  client: LLMClient,
                                  prompt: str,
                                  runs: int = 5) -> Dict[str, Any]:
        """Test response consistency for the same prompt"""
        responses = []
        
        for _ in range(runs):
            try:
                response = await client.generate_async(prompt)
                if not response.error:
                    responses.append(response.response)
            except Exception as e:
                logger.error(f"Error in consistency test: {e}")
        
        if len(responses) < 2:
            return {"error": "Not enough responses for consistency test"}
        
        # Calculate similarity between responses
        # Simple approach: character overlap ratio
        similarities = []
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                similarity = self._calculate_similarity(responses[i], responses[j])
                similarities.append(similarity)
        
        return {
            "response_count": len(responses),
            "avg_similarity": np.mean(similarities),
            "min_similarity": np.min(similarities),
            "max_similarity": np.max(similarities),
            "response_lengths": [len(r) for r in responses]
        }
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts (simple character overlap)"""
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    async def run_language_test(self,
                               client: LLMClient,
                               languages: List[str],
                               domain: str,
                               prompts_per_language: int = 10) -> Dict[str, Dict]:
        """Test performance across different languages"""
        results = {}
        
        for language in languages:
            logger.info(f"Testing {client.model_name} on {language}")
            
            # Get prompts for this language
            prompts = []
            test_cases = []
            
            for _ in range(prompts_per_language):
                prompt = self.prompt_generator.get_random_prompt(domain, language)
                if prompt:
                    prompts.append(prompt["text"])
                    test_cases.append({"prompt": prompt})
            
            if not prompts:
                results[language] = {"error": "No prompts available"}
                continue
            
            # Run tests
            latency_results = await self.run_latency_test(client, prompts, iterations=1)
            quality_results = await self.run_quality_test(client, test_cases)
            
            results[language] = {
                "prompt_count": len(prompts),
                "avg_latency_ms": latency_results.get("avg_latency_ms", 0),
                "avg_alignment_score": quality_results.get("avg_alignment_score", 0),
                "high_risk_percentage": quality_results.get("high_risk_percentage", 0)
            }
        
        return results
    
    async def run_benchmark_suite(self,
                                 models: List[str],
                                 suite_type: str = "standard") -> BenchmarkSuite:
        """Run complete benchmark suite"""
        config = self.benchmark_configs[suite_type]
        start_time = datetime.now()
        
        logger.info(f"Starting {suite_type} benchmark suite with {len(models)} models")
        
        results = []
        
        # Create progress bar
        total_tests = len(models) * len(config["languages"]) * len(config["domains"])
        pbar = tqdm(total=total_tests, desc="Running benchmarks")
        
        for model_name in models:
            try:
                client = LLMClientFactory.create_client(model_name)
                
                for domain in config["domains"]:
                    for language in config["languages"]:
                        # Prepare test data
                        prompts = []
                        test_cases = []
                        
                        for _ in range(config["prompts_per_test"]):
                            prompt = self.prompt_generator.get_random_prompt(domain, language)
                            if prompt:
                                prompts.append(prompt["text"])
                                test_cases.append({"prompt": prompt})
                        
                        if not prompts:
                            continue
                        
                        # Run tests
                        latency_results = await self.run_latency_test(
                            client, prompts, config["iterations"]
                        )
                        
                        throughput_results = await self.run_throughput_test(
                            client, prompts
                        )
                        
                        quality_results = await self.run_quality_test(
                            client, test_cases
                        )
                        
                        # Create benchmark result
                        result = BenchmarkResult(
                            model=model_name,
                            test_name=f"{suite_type}_{domain}_{language}",
                            language=language,
                            domain=domain,
                            prompt_count=len(prompts),
                            avg_latency_ms=latency_results.get("avg_latency_ms", 0),
                            p50_latency_ms=latency_results.get("p50_latency_ms", 0),
                            p95_latency_ms=latency_results.get("p95_latency_ms", 0),
                            p99_latency_ms=latency_results.get("p99_latency_ms", 0),
                            throughput_per_second=throughput_results.get("throughput_per_second", 0),
                            avg_tokens_per_response=throughput_results.get("avg_tokens_per_response", 0),
                            avg_alignment_score=quality_results.get("avg_alignment_score", 0),
                            error_rate=1 - throughput_results.get("success_rate", 0)
                        )
                        
                        results.append(result)
                        pbar.update(1)
                        
            except Exception as e:
                logger.error(f"Error benchmarking {model_name}: {e}")
                pbar.update(len(config["languages"]) * len(config["domains"]))
        
        pbar.close()
        
        end_time = datetime.now()
        
        # Create suite results
        suite = BenchmarkSuite(
            suite_name=f"{suite_type}_benchmark_{start_time.strftime('%Y%m%d_%H%M%S')}",
            start_time=start_time,
            end_time=end_time,
            total_duration_seconds=(end_time - start_time).total_seconds(),
            models_tested=models,
            languages_tested=config["languages"],
            domains_tested=config["domains"],
            total_prompts=sum(r.prompt_count for r in results),
            results=results,
            system_info={
                "platform": sys.platform,
                "python_version": sys.version,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Save results
        self._save_benchmark_results(suite)
        
        return suite
    
    def _save_benchmark_results(self, suite: BenchmarkSuite):
        """Save benchmark results to file"""
        output_file = self.output_dir / f"{suite.suite_name}.json"
        
        # Convert to dict
        suite_dict = asdict(suite)
        
        # Convert datetime objects
        suite_dict["start_time"] = suite.start_time.isoformat()
        suite_dict["end_time"] = suite.end_time.isoformat()
        for result in suite_dict["results"]:
            result["timestamp"] = result["timestamp"].isoformat()
        
        with open(output_file, 'w') as f:
            json.dump(suite_dict, f, indent=2)
        
        logger.info(f"Benchmark results saved to {output_file}")
    
    def generate_benchmark_report(self, suite: BenchmarkSuite, output_format: str = "html"):
        """Generate comprehensive benchmark report"""
        # Convert results to DataFrame
        df = pd.DataFrame([asdict(r) for r in suite.results])
        
        # Create visualizations
        self._create_latency_charts(df, suite.suite_name)
        self._create_quality_charts(df, suite.suite_name)
        self._create_comparison_charts(df, suite.suite_name)
        
        # Generate report
        if output_format == "html":
            self._generate_html_report(suite, df)
        elif output_format == "markdown":
            self._generate_markdown_report(suite, df)
        
        return self.output_dir / f"{suite.suite_name}_report.{output_format}"
    
    def _create_latency_charts(self, df: pd.DataFrame, suite_name: str):
        """Create latency comparison charts"""
        plt.figure(figsize=(12, 6))
        
        # Latency by model
        plt.subplot(1, 2, 1)
        model_latency = df.groupby('model')['avg_latency_ms'].mean().sort_values()
        model_latency.plot(kind='barh')
        plt.xlabel('Average Latency (ms)')
        plt.title('Average Latency by Model')
        plt.tight_layout()
        
        # Latency distribution
        plt.subplot(1, 2, 2)
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            plt.plot(model_data.index, model_data['avg_latency_ms'], 
                    label=model, marker='o', alpha=0.7)
        plt.xlabel('Test Index')
        plt.ylabel('Latency (ms)')
        plt.title('Latency Distribution Across Tests')
        plt.legend()
        
        plt.savefig(self.output_dir / f"{suite_name}_latency_charts.png", dpi=300)
        plt.close()
    
    def _create_quality_charts(self, df: pd.DataFrame, suite_name: str):
        """Create quality comparison charts"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Alignment scores by model
        ax = axes[0, 0]
        model_scores = df.groupby('model')['avg_alignment_score'].mean().sort_values()
        model_scores.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_ylabel('Average Alignment Score')
        ax.set_title('Alignment Scores by Model')
        ax.set_ylim(0, 5)
        
        # Alignment scores by language
        ax = axes[0, 1]
        lang_scores = df.groupby('language')['avg_alignment_score'].mean()
        lang_scores.plot(kind='bar', ax=ax, color='lightgreen')
        ax.set_ylabel('Average Alignment Score')
        ax.set_title('Alignment Scores by Language')
        ax.set_ylim(0, 5)
        
        # Error rates
        ax = axes[1, 0]
        error_rates = df.groupby('model')['error_rate'].mean() * 100
        error_rates.plot(kind='bar', ax=ax, color='salmon')
        ax.set_ylabel('Error Rate (%)')
        ax.set_title('Error Rates by Model')
        
        # Throughput comparison
        ax = axes[1, 1]
        throughput = df.groupby('model')['throughput_per_second'].mean()
        throughput.plot(kind='bar', ax=ax, color='gold')
        ax.set_ylabel('Requests per Second')
        ax.set_title('Throughput by Model')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{suite_name}_quality_charts.png", dpi=300)
        plt.close()
    
    def _create_comparison_charts(self, df: pd.DataFrame, suite_name: str):
        """Create comprehensive comparison charts"""
        # Create heatmap
        plt.figure(figsize=(10, 8))
        
        # Prepare data for heatmap
        metrics = ['avg_latency_ms', 'avg_alignment_score', 'throughput_per_second', 'error_rate']
        models = df['model'].unique()
        
        # Normalize metrics for comparison
        normalized_data = []
        for model in models:
            model_data = df[df['model'] == model]
            row = []
            for metric in metrics:
                values = model_data[metric].values
                if len(values) > 0:
                    if metric in ['avg_latency_ms', 'error_rate']:  # Lower is better
                        normalized = 1 - (np.mean(values) - df[metric].min()) / (df[metric].max() - df[metric].min())
                    else:  # Higher is better
                        normalized = (np.mean(values) - df[metric].min()) / (df[metric].max() - df[metric].min())
                    row.append(normalized)
                else:
                    row.append(0)
            normalized_data.append(row)
        
        # Create heatmap
        sns.heatmap(normalized_data, 
                   xticklabels=['Latency\n(lower better)', 'Alignment\nScore', 'Throughput', 'Error Rate\n(lower better)'],
                   yticklabels=models,
                   annot=True, 
                   fmt='.2f',
                   cmap='RdYlGn',
                   center=0.5)
        
        plt.title('Normalized Performance Metrics Comparison')
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{suite_name}_comparison_heatmap.png", dpi=300)
        plt.close()
    
    def _generate_html_report(self, suite: BenchmarkSuite, df: pd.DataFrame):
        """Generate HTML report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>MASB-Alt Benchmark Report - {suite.suite_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .metric {{ background-color: #e8f4f8; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                img {{ max-width: 100%; height: auto; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>MASB-Alt Benchmark Report</h1>
            
            <div class="metric">
                <h2>Suite Information</h2>
                <p><strong>Suite Name:</strong> {suite.suite_name}</p>
                <p><strong>Duration:</strong> {suite.total_duration_seconds:.2f} seconds</p>
                <p><strong>Models Tested:</strong> {', '.join(suite.models_tested)}</p>
                <p><strong>Languages:</strong> {', '.join(suite.languages_tested)}</p>
                <p><strong>Domains:</strong> {', '.join(suite.domains_tested)}</p>
                <p><strong>Total Prompts:</strong> {suite.total_prompts}</p>
            </div>
            
            <h2>Performance Summary</h2>
            <table>
                <tr>
                    <th>Model</th>
                    <th>Avg Latency (ms)</th>
                    <th>P95 Latency (ms)</th>
                    <th>Throughput (req/s)</th>
                    <th>Avg Alignment Score</th>
                    <th>Error Rate</th>
                </tr>
        """
        
        # Add model summaries
        for model in suite.models_tested:
            model_data = df[df['model'] == model]
            html_content += f"""
                <tr>
                    <td>{model}</td>
                    <td>{model_data['avg_latency_ms'].mean():.2f}</td>
                    <td>{model_data['p95_latency_ms'].mean():.2f}</td>
                    <td>{model_data['throughput_per_second'].mean():.2f}</td>
                    <td>{model_data['avg_alignment_score'].mean():.2f}</td>
                    <td>{model_data['error_rate'].mean() * 100:.1f}%</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <h2>Visualizations</h2>
            <img src="{}_latency_charts.png" alt="Latency Charts">
            <img src="{}_quality_charts.png" alt="Quality Charts">
            <img src="{}_comparison_heatmap.png" alt="Comparison Heatmap">
            
            <h2>Detailed Results</h2>
            <p>Full results are available in the JSON file: {}.json</p>
        </body>
        </html>
        """.format(suite.suite_name, suite.suite_name, suite.suite_name, suite.suite_name)
        
        output_file = self.output_dir / f"{suite.suite_name}_report.html"
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to {output_file}")

# Command-line interface
async def main():
    """Main entry point for benchmarking"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run MASB-Alt model benchmarks")
    parser.add_argument("--models", nargs="+", required=True,
                       help="Models to benchmark (e.g., openai anthropic)")
    parser.add_argument("--suite", choices=["quick", "standard", "comprehensive"],
                       default="standard", help="Benchmark suite type")
    parser.add_argument("--output-dir", default="./benchmarks",
                       help="Output directory for results")
    parser.add_argument("--report-format", choices=["html", "markdown"],
                       default="html", help="Report format")
    
    args = parser.parse_args()
    
    # Run benchmark
    benchmark = ModelBenchmark(output_dir=args.output_dir)
    
    print(f"Starting {args.suite} benchmark for models: {', '.join(args.models)}")
    
    suite = await benchmark.run_benchmark_suite(args.models, args.suite)
    
    print(f"\nBenchmark completed in {suite.total_duration_seconds:.2f} seconds")
    print(f"Total evaluations: {len(suite.results)}")
    
    # Generate report
    report_path = benchmark.generate_benchmark_report(suite, args.report_format)
    print(f"\nReport generated: {report_path}")
    
    # Print summary
    df = pd.DataFrame([asdict(r) for r in suite.results])
    print("\n=== Summary ===")
    for model in args.models:
        model_data = df[df['model'] == model]
        if not model_data.empty:
            print(f"\n{model}:")
            print(f"  Avg Latency: {model_data['avg_latency_ms'].mean():.2f}ms")
            print(f"  Avg Alignment Score: {model_data['avg_alignment_score'].mean():.2f}/5")
            print(f"  Throughput: {model_data['throughput_per_second'].mean():.2f} req/s")
            print(f"  Error Rate: {model_data['error_rate'].mean() * 100:.1f}%")

if __name__ == "__main__":
    asyncio.run(main())