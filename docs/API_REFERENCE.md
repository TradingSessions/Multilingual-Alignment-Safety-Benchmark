# API Reference Documentation

## Overview

The MASB-Alt API provides programmatic access to all evaluation, analysis, and reporting capabilities. This reference covers the core APIs and their usage.

## Table of Contents

1. [Core Classes](#core-classes)
2. [Orchestrator API](#orchestrator-api)
3. [Data Manager API](#data-manager-api)
4. [Evaluation API](#evaluation-api)
5. [REST API Endpoints](#rest-api-endpoints)
6. [Examples](#examples)

## Core Classes

### MASBOrchestrator

The main orchestration class that coordinates all system components.

```python
from masb_orchestrator import MASBOrchestrator

orchestrator = MASBOrchestrator(config_path: Optional[str] = None)
```

**Parameters:**
- `config_path` (str, optional): Path to custom configuration file

### MultilingualPromptGenerator

Generates evaluation prompts in multiple languages.

```python
from prompt_generator.multilingual_prompt_gen import MultilingualPromptGenerator

generator = MultilingualPromptGenerator()
```

### MultilingualAlignmentEvaluator

Evaluates LLM responses for alignment and safety.

```python
from evaluation.evaluator import MultilingualAlignmentEvaluator

evaluator = MultilingualAlignmentEvaluator()
```

## Orchestrator API

### run_evaluation_suite

Run a comprehensive evaluation across models, languages, and domains.

```python
async def run_evaluation_suite(
    models: Optional[List[str]] = None,
    languages: Optional[List[str]] = None,
    domains: Optional[List[str]] = None,
    prompts_per_combination: int = 5
) -> str
```

**Parameters:**
- `models`: List of model names (e.g., ["openai", "anthropic"])
- `languages`: List of language codes (e.g., ["en", "sw", "ar"])
- `domains`: List of domains (e.g., ["healthcare", "education"])
- `prompts_per_combination`: Number of prompts per language-domain combination

**Returns:**
- `dataset_id`: Unique identifier for the evaluation dataset

**Example:**
```python
dataset_id = await orchestrator.run_evaluation_suite(
    models=["openai", "anthropic"],
    languages=["en", "sw"],
    domains=["healthcare"],
    prompts_per_combination=10
)
```

### run_benchmark_analysis

Run performance benchmarks on models.

```python
async def run_benchmark_analysis(
    models: Optional[List[str]] = None,
    suite_type: str = "standard"
) -> str
```

**Parameters:**
- `models`: List of models to benchmark
- `suite_type`: One of "quick", "standard", or "comprehensive"

**Returns:**
- `report_path`: Path to generated benchmark report

### generate_comprehensive_report

Generate detailed evaluation reports.

```python
def generate_comprehensive_report(
    report_type: str = "executive",
    format: str = "html",
    days: int = 7
) -> str
```

**Parameters:**
- `report_type`: One of "executive", "technical", or "comparative"
- `format`: One of "html", "pdf", "docx", or "markdown"
- `days`: Number of days to include in analysis

**Returns:**
- `report_path`: Path to generated report

### analyze_finetuning_recommendations

Analyze model performance and generate fine-tuning recommendations.

```python
def analyze_finetuning_recommendations(
    model: str,
    days: int = 30
) -> str
```

**Parameters:**
- `model`: Model name to analyze
- `days`: Analysis period in days

**Returns:**
- `export_file`: Path to recommendations file

## Data Manager API

### add_prompt

Add a new prompt to the database.

```python
def add_prompt(
    prompt_data: Dict,
    dataset_id: Optional[str] = None
) -> str
```

**Parameters:**
- `prompt_data`: Dictionary containing prompt information
  - `text` (str): Prompt text
  - `language` (str): Language code
  - `domain` (str): Domain name
  - `risk_level` (str): Risk level (low/medium/high/critical)
  - `tags` (List[str]): Associated tags
  - `metadata` (Dict): Additional metadata
- `dataset_id`: Optional dataset identifier

**Returns:**
- `prompt_id`: Unique prompt identifier

### add_response

Add an LLM response to the database.

```python
def add_response(
    response_data: Dict,
    prompt_id: str
) -> str
```

**Parameters:**
- `response_data`: Dictionary containing response information
  - `model` (str): Model name
  - `response` (str): Generated response text
  - `latency_ms` (float): Response latency in milliseconds
  - `token_count` (int): Number of tokens
  - `timestamp` (str): ISO format timestamp
- `prompt_id`: Associated prompt identifier

**Returns:**
- `response_id`: Unique response identifier

### add_evaluation

Add an evaluation result to the database.

```python
def add_evaluation(
    evaluation_data: Dict,
    response_id: str
) -> str
```

**Parameters:**
- `evaluation_data`: Dictionary containing evaluation information
  - `alignment_score` (int): Score from 1-5
  - `confidence_score` (float): Confidence from 0.0-1.0
  - `risk_level` (str): Risk level
  - `risk_flags` (Dict): Risk indicators
  - `comments` (str): Evaluation comments
  - `evaluator_id` (str): Evaluator identifier
  - `timestamp` (str): ISO format timestamp
- `response_id`: Associated response identifier

**Returns:**
- `evaluation_id`: Unique evaluation identifier

## Evaluation API

### evaluate_response

Evaluate an LLM response for alignment and safety.

```python
def evaluate_response(
    prompt: Dict,
    llm_output: str,
    llm_model: str
) -> EvaluationResult
```

**Parameters:**
- `prompt`: Prompt dictionary with text, language, domain
- `llm_output`: Generated response text
- `llm_model`: Model name

**Returns:**
- `EvaluationResult`: Object containing:
  - `alignment_score`: AlignmentScore enum (1-5)
  - `confidence_score`: Float (0.0-1.0)
  - `risk_level`: RiskLevel enum
  - `risk_flags`: RiskFlags object
  - `comments`: String

## REST API Endpoints

### Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-12-01T10:00:00Z",
  "version": "1.0.0"
}
```

### Run Evaluation

```http
POST /evaluate
Content-Type: application/json

{
  "prompt": "What is diabetes?",
  "language": "en",
  "domain": "healthcare",
  "models": ["openai", "anthropic"]
}
```

**Response:**
```json
{
  "evaluation_id": "eval_123",
  "results": [
    {
      "model": "openai",
      "response": "Diabetes is...",
      "alignment_score": 4,
      "risk_level": "low"
    }
  ]
}
```

### Get Evaluation Results

```http
GET /evaluations/{dataset_id}
```

**Response:**
```json
{
  "dataset_id": "dataset_123",
  "prompts": 50,
  "responses": 150,
  "evaluations": 150,
  "summary": {
    "avg_alignment_score": 4.2,
    "risk_distribution": {
      "low": 120,
      "medium": 25,
      "high": 5
    }
  }
}
```

### Generate Report

```http
POST /reports/generate
Content-Type: application/json

{
  "dataset_ids": ["dataset_123"],
  "report_type": "technical",
  "format": "html"
}
```

**Response:**
```json
{
  "report_id": "report_456",
  "status": "generating",
  "eta_seconds": 30
}
```

### System Status

```http
GET /status
```

**Response:**
```json
{
  "timestamp": "2024-12-01T10:00:00Z",
  "components": {
    "evaluator": "active",
    "data_manager": "active",
    "api_server": "active"
  },
  "stats": {
    "total_evaluations": 1500,
    "active_datasets": 10
  }
}
```

## Examples

### Complete Evaluation Workflow

```python
import asyncio
from masb_orchestrator import MASBOrchestrator

async def complete_evaluation():
    # Initialize
    orchestrator = MASBOrchestrator()
    
    # Run evaluation
    dataset_id = await orchestrator.run_evaluation_suite(
        models=["openai", "anthropic"],
        languages=["en", "sw", "ar"],
        domains=["healthcare", "education"],
        prompts_per_combination=10
    )
    
    # Wait for completion
    await asyncio.sleep(5)
    
    # Generate reports
    html_report = orchestrator.generate_comprehensive_report(
        report_type="technical",
        format="html"
    )
    
    pdf_report = orchestrator.generate_comprehensive_report(
        report_type="executive",
        format="pdf"
    )
    
    # Analyze for fine-tuning
    recommendations = orchestrator.analyze_finetuning_recommendations(
        model="openai",
        days=30
    )
    
    return {
        "dataset_id": dataset_id,
        "html_report": html_report,
        "pdf_report": pdf_report,
        "recommendations": recommendations
    }

# Run
results = asyncio.run(complete_evaluation())
print(results)
```

### Custom Evaluation Pipeline

```python
from prompt_generator.multilingual_prompt_gen import MultilingualPromptGenerator
from evaluation.evaluator import MultilingualAlignmentEvaluator
from data_manager import DataManager

# Initialize components
generator = MultilingualPromptGenerator()
evaluator = MultilingualAlignmentEvaluator()
data_manager = DataManager()

# Generate custom prompts
prompts = []
for _ in range(10):
    prompt = generator.get_random_prompt("healthcare", "sw")
    if prompt:
        prompts.append(prompt)

# Evaluate with custom model
for prompt in prompts:
    # Get LLM response (implement your own)
    llm_response = get_llm_response(prompt["text"])
    
    # Evaluate
    result = evaluator.evaluate_response(
        prompt=prompt,
        llm_output=llm_response,
        llm_model="custom_model"
    )
    
    # Store results
    prompt_id = data_manager.add_prompt(prompt)
    response_id = data_manager.add_response({
        "model": "custom_model",
        "response": llm_response,
        "timestamp": datetime.now().isoformat()
    }, prompt_id)
    
    data_manager.add_evaluation({
        "alignment_score": result.alignment_score.value,
        "confidence_score": result.confidence_score,
        "risk_level": result.risk_level.value,
        "timestamp": datetime.now().isoformat()
    }, response_id)
```

### Batch Processing

```python
import asyncio
from typing import List

async def batch_evaluate(prompts: List[Dict], models: List[str]):
    orchestrator = MASBOrchestrator()
    results = []
    
    # Process in batches
    batch_size = 10
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        
        # Evaluate batch
        tasks = []
        for prompt in batch:
            for model in models:
                task = evaluate_single(prompt, model)
                tasks.append(task)
        
        # Wait for batch completion
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)
    
    return results
```

## Error Handling

All API methods include comprehensive error handling:

```python
try:
    dataset_id = await orchestrator.run_evaluation_suite(
        models=["invalid_model"]
    )
except ValueError as e:
    print(f"Invalid parameter: {e}")
except ConnectionError as e:
    print(f"API connection failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Rate Limiting

The API includes built-in rate limiting:

- Default: 60 requests per minute
- Configurable in `config.json`
- Returns 429 status code when exceeded

## Authentication

For production deployments, implement authentication:

```python
# Example with API key
headers = {
    "Authorization": "Bearer YOUR_API_KEY",
    "Content-Type": "application/json"
}
```

---

For more examples and advanced usage, see the `examples/` directory.