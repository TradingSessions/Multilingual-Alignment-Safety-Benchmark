# api_server.py - RESTful API server for MASB-Alt

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
import uvicorn
import logging
from pathlib import Path
import json
import uuid

# Add parent directory to path
import sys
sys.path.append(str(Path(__file__).parent))

from prompt_generator.multilingual_prompt_gen import MultilingualPromptGenerator
from llm_api_client import LLMClientFactory
from evaluation.evaluator import MultilingualAlignmentEvaluator
from data_manager import DataManager
from config import get_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MASB-Alt API",
    description="Multilingual Alignment Safety Benchmark API Service",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
config = get_config()
prompt_generator = MultilingualPromptGenerator()
evaluator = MultilingualAlignmentEvaluator()
data_manager = DataManager(config.data.base_path)
llm_client = LLMClientFactory.create_multi_client(config.get_enabled_providers())

# Request/Response models
class PromptRequest(BaseModel):
    domain: str
    language: Optional[str] = None
    risk_level: Optional[str] = None
    count: int = Field(default=1, ge=1, le=100)

class EvaluationRequest(BaseModel):
    prompt: Dict[str, Any]
    llm_output: str
    llm_model: str
    evaluator_id: Optional[str] = "api_user"

class BatchEvaluationRequest(BaseModel):
    domains: List[str]
    languages: List[str]
    models: Optional[List[str]] = None
    prompts_per_combination: int = Field(default=5, ge=1, le=50)

class DatasetRequest(BaseModel):
    name: str
    description: Optional[str] = ""
    prompts: Optional[List[Dict]] = None

class AnalysisRequest(BaseModel):
    dataset_id: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    languages: Optional[List[str]] = None
    domains: Optional[List[str]] = None

# Background task tracking
background_tasks = {}

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "MASB-Alt API",
        "version": "1.0.0",
        "status": "operational",
        "documentation": "/api/docs"
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "database": "connected",
            "llm_providers": config.get_enabled_providers()
        }
    }

@app.get("/api/languages")
async def get_languages():
    """Get list of supported languages"""
    return {
        "languages": prompt_generator.get_supported_languages(),
        "total": len(prompt_generator.get_supported_languages())
    }

@app.get("/api/domains")
async def get_domains():
    """Get list of evaluation domains"""
    return {
        "domains": prompt_generator.get_domains(),
        "total": len(prompt_generator.get_domains())
    }

@app.post("/api/prompts/generate")
async def generate_prompts(request: PromptRequest):
    """Generate prompts based on criteria"""
    try:
        prompts = []
        for _ in range(request.count):
            prompt = prompt_generator.get_random_prompt(
                domain=request.domain,
                language=request.language,
                risk_level=request.risk_level
            )
            if prompt:
                prompts.append(prompt)
        
        return {
            "prompts": prompts,
            "count": len(prompts),
            "domain": request.domain,
            "language": request.language
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/prompts/{domain}")
async def get_domain_prompts(
    domain: str,
    language: Optional[str] = None,
    limit: int = Query(default=10, ge=1, le=100)
):
    """Get all prompts for a specific domain"""
    try:
        all_prompts = prompt_generator.get_all_prompts(domain, language)
        return {
            "domain": domain,
            "language": language,
            "prompts": all_prompts[:limit],
            "total": len(all_prompts)
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/api/evaluate")
async def evaluate_response(request: EvaluationRequest):
    """Evaluate a single LLM response"""
    try:
        result = evaluator.evaluate_response(
            prompt=request.prompt,
            llm_output=request.llm_output,
            llm_model=request.llm_model,
            evaluator_id=request.evaluator_id
        )
        
        # Save to database
        prompt_id = data_manager.add_prompt(request.prompt)
        response_id = data_manager.add_response({
            "model": request.llm_model,
            "response": request.llm_output,
            "timestamp": datetime.now().isoformat()
        }, prompt_id)
        data_manager.add_evaluation(result.to_dict(), response_id)
        
        return {
            "evaluation": result.to_dict(),
            "prompt_id": prompt_id,
            "response_id": response_id
        }
    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/evaluate/batch")
async def batch_evaluate(
    request: BatchEvaluationRequest,
    background_tasks: BackgroundTasks
):
    """Start batch evaluation in background"""
    task_id = str(uuid.uuid4())
    
    # Start background task
    background_tasks.add_task(
        run_batch_evaluation,
        task_id,
        request.domains,
        request.languages,
        request.models,
        request.prompts_per_combination
    )
    
    background_tasks[task_id] = {
        "status": "started",
        "started_at": datetime.now().isoformat(),
        "request": request.dict()
    }
    
    return {
        "task_id": task_id,
        "status": "started",
        "message": "Batch evaluation started in background"
    }

@app.get("/api/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get status of background task"""
    if task_id not in background_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return background_tasks[task_id]

@app.post("/api/datasets")
async def create_dataset(request: DatasetRequest):
    """Create a new dataset"""
    try:
        dataset_id = data_manager.create_dataset(
            name=request.name,
            description=request.description,
            prompts=request.prompts
        )
        
        return {
            "dataset_id": dataset_id,
            "name": request.name,
            "prompt_count": len(request.prompts) if request.prompts else 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/datasets")
async def list_datasets():
    """List all datasets"""
    datasets = data_manager.list_datasets()
    return {
        "datasets": datasets,
        "total": len(datasets)
    }

@app.get("/api/datasets/{dataset_id}")
async def get_dataset(dataset_id: str):
    """Get specific dataset"""
    dataset = data_manager.get_dataset(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset

@app.post("/api/analysis")
async def analyze_evaluations(request: AnalysisRequest):
    """Analyze evaluation results"""
    try:
        # Get summary statistics
        stats = data_manager.generate_summary_statistics()
        
        # Apply filters if provided
        # This is a simplified version - you'd implement proper filtering
        filtered_stats = stats.copy()
        
        if request.languages:
            filtered_stats["languages"] = {
                lang: count for lang, count in stats["languages"].items()
                if lang in request.languages
            }
        
        if request.domains:
            filtered_stats["domains"] = {
                domain: count for domain, count in stats["domains"].items()
                if domain in request.domains
            }
        
        return {
            "statistics": filtered_stats,
            "filters_applied": {
                "languages": request.languages,
                "domains": request.domains,
                "date_range": {
                    "start": request.start_date.isoformat() if request.start_date else None,
                    "end": request.end_date.isoformat() if request.end_date else None
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models")
async def get_models():
    """Get list of available LLM models"""
    return {
        "models": config.get_enabled_providers(),
        "configurations": {
            provider: {
                "model": config.llm_configs[provider].model,
                "max_tokens": config.llm_configs[provider].max_tokens,
                "temperature": config.llm_configs[provider].temperature
            }
            for provider in config.get_enabled_providers()
        }
    }

@app.post("/api/llm/generate")
async def generate_llm_response(
    prompt: str,
    model: Optional[str] = None,
    temperature: float = Query(default=0.7, ge=0, le=2),
    max_tokens: int = Query(default=1000, ge=1, le=4000)
):
    """Generate response from specified LLM"""
    try:
        if model:
            # Use specific model
            client = LLMClientFactory.create_client(model)
            response = await client.generate_async(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return {
                "model": response.model,
                "response": response.response,
                "latency_ms": response.latency_ms,
                "token_count": response.token_count
            }
        else:
            # Use all available models
            responses = await llm_client.generate_all_async(
                prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return {
                "responses": [
                    {
                        "model": r.model,
                        "response": r.response,
                        "latency_ms": r.latency_ms,
                        "token_count": r.token_count,
                        "error": r.error
                    }
                    for r in responses
                ]
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/export/{dataset_id}")
async def export_dataset(
    dataset_id: str,
    format: str = Query(default="json", regex="^(json|csv)$")
):
    """Export dataset for evaluation"""
    try:
        export_file = f"export_{dataset_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"
        export_path = data_manager.export_dataset_for_evaluation(dataset_id, export_file)
        
        return FileResponse(
            path=export_path,
            filename=export_file,
            media_type="application/json" if format == "json" else "text/csv"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Background task functions
async def run_batch_evaluation(
    task_id: str,
    domains: List[str],
    languages: List[str],
    models: Optional[List[str]],
    prompts_per_combination: int
):
    """Run batch evaluation in background"""
    try:
        background_tasks[task_id]["status"] = "running"
        background_tasks[task_id]["progress"] = 0
        
        # Generate prompts
        prompts = []
        for domain in domains:
            for language in languages:
                domain_prompts = prompt_generator.get_all_prompts(domain, language)
                prompts.extend(domain_prompts[:prompts_per_combination])
        
        total_evaluations = len(prompts) * len(models or config.get_enabled_providers())
        completed = 0
        
        # Create dataset
        dataset_id = data_manager.create_dataset(
            name=f"Batch API Evaluation {datetime.now().strftime('%Y%m%d_%H%M%S')}",
            description=f"Batch evaluation via API: {len(domains)} domains, {len(languages)} languages",
            prompts=prompts
        )
        
        # Evaluate with each model
        results = []
        for prompt in prompts:
            prompt_id = data_manager.add_prompt(prompt, dataset_id)
            
            # Get responses from models
            if models:
                # Use specified models
                responses = []
                for model in models:
                    client = LLMClientFactory.create_client(model)
                    response = await client.generate_async(prompt["text"])
                    responses.append(response)
            else:
                # Use all available models
                responses = await llm_client.generate_all_async(prompt["text"])
            
            # Evaluate each response
            for response in responses:
                if not response.error:
                    # Save response
                    response_id = data_manager.add_response({
                        "model": response.model,
                        "response": response.response,
                        "timestamp": response.timestamp.isoformat(),
                        "latency_ms": response.latency_ms
                    }, prompt_id)
                    
                    # Evaluate
                    result = evaluator.evaluate_response(
                        prompt=prompt,
                        llm_output=response.response,
                        llm_model=response.model,
                        evaluator_id="batch_api"
                    )
                    
                    # Save evaluation
                    data_manager.add_evaluation(result.to_dict(), response_id)
                    results.append(result)
                
                completed += 1
                background_tasks[task_id]["progress"] = (completed / total_evaluations) * 100
        
        # Generate summary
        summary = evaluator.generate_summary_report(results)
        
        # Update task status
        background_tasks[task_id].update({
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "dataset_id": dataset_id,
            "summary": summary,
            "total_evaluations": len(results)
        })
        
    except Exception as e:
        background_tasks[task_id].update({
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.now().isoformat()
        })

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Resource not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("MASB-Alt API server starting...")
    
    # Validate configuration
    issues = config.validate()
    if issues:
        logger.warning(f"Configuration issues: {issues}")
    
    logger.info(f"Enabled LLM providers: {config.get_enabled_providers()}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("MASB-Alt API server shutting down...")

# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )