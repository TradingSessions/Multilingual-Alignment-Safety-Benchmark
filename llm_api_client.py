# llm_api_client.py - Unified API client for multiple LLMs

import os
import json
import time
import logging
from typing import Dict, List, Optional, Any, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    """Standardized LLM response format"""
    model: str
    prompt: str
    response: str
    timestamp: datetime
    latency_ms: float
    token_count: Optional[int] = None
    finish_reason: Optional[str] = None
    error: Optional[str] = None

class LLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    @abstractmethod
    async def generate_async(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response asynchronously"""
        pass
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response synchronously"""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name"""
        pass

class OpenAIClient(LLMClient):
    """OpenAI API client (GPT-3.5, GPT-4)"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        self.model = model
        self.base_url = "https://api.openai.com/v1/chat/completions"
        
    @property
    def model_name(self) -> str:
        return self.model
    
    async def generate_async(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using OpenAI API asynchronously"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1000)
        }
        
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        latency_ms = (time.time() - start_time) * 1000
                        
                        return LLMResponse(
                            model=self.model,
                            prompt=prompt,
                            response=result["choices"][0]["message"]["content"],
                            timestamp=datetime.now(),
                            latency_ms=latency_ms,
                            token_count=result.get("usage", {}).get("total_tokens"),
                            finish_reason=result["choices"][0].get("finish_reason")
                        )
                    else:
                        error_text = await response.text()
                        return LLMResponse(
                            model=self.model,
                            prompt=prompt,
                            response="",
                            timestamp=datetime.now(),
                            latency_ms=(time.time() - start_time) * 1000,
                            error=f"API Error {response.status}: {error_text}"
                        )
        except Exception as e:
            return LLMResponse(
                model=self.model,
                prompt=prompt,
                response="",
                timestamp=datetime.now(),
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response synchronously"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.generate_async(prompt, **kwargs))
        finally:
            loop.close()

class AnthropicClient(LLMClient):
    """Anthropic API client (Claude)"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-opus-20240229"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key not provided")
        self.model = model
        self.base_url = "https://api.anthropic.com/v1/messages"
        
    @property
    def model_name(self) -> str:
        return self.model
    
    async def generate_async(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using Anthropic API asynchronously"""
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 0.7)
        }
        
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        latency_ms = (time.time() - start_time) * 1000
                        
                        return LLMResponse(
                            model=self.model,
                            prompt=prompt,
                            response=result["content"][0]["text"],
                            timestamp=datetime.now(),
                            latency_ms=latency_ms,
                            token_count=result.get("usage", {}).get("output_tokens"),
                            finish_reason=result.get("stop_reason")
                        )
                    else:
                        error_text = await response.text()
                        return LLMResponse(
                            model=self.model,
                            prompt=prompt,
                            response="",
                            timestamp=datetime.now(),
                            latency_ms=(time.time() - start_time) * 1000,
                            error=f"API Error {response.status}: {error_text}"
                        )
        except Exception as e:
            return LLMResponse(
                model=self.model,
                prompt=prompt,
                response="",
                timestamp=datetime.now(),
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response synchronously"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.generate_async(prompt, **kwargs))
        finally:
            loop.close()

class CohereClient(LLMClient):
    """Cohere API client"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "command"):
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("Cohere API key not provided")
        self.model = model
        self.base_url = "https://api.cohere.ai/v1/generate"
        
    @property
    def model_name(self) -> str:
        return self.model
    
    async def generate_async(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using Cohere API asynchronously"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 0.7),
            "return_likelihoods": "NONE"
        }
        
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, headers=headers, json=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        latency_ms = (time.time() - start_time) * 1000
                        
                        return LLMResponse(
                            model=self.model,
                            prompt=prompt,
                            response=result["generations"][0]["text"].strip(),
                            timestamp=datetime.now(),
                            latency_ms=latency_ms,
                            token_count=None,  # Cohere doesn't return token count
                            finish_reason=result["generations"][0].get("finish_reason")
                        )
                    else:
                        error_text = await response.text()
                        return LLMResponse(
                            model=self.model,
                            prompt=prompt,
                            response="",
                            timestamp=datetime.now(),
                            latency_ms=(time.time() - start_time) * 1000,
                            error=f"API Error {response.status}: {error_text}"
                        )
        except Exception as e:
            return LLMResponse(
                model=self.model,
                prompt=prompt,
                response="",
                timestamp=datetime.now(),
                latency_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response synchronously"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.generate_async(prompt, **kwargs))
        finally:
            loop.close()

class MultiLLMClient:
    """Client for querying multiple LLMs simultaneously"""
    
    def __init__(self, clients: Optional[List[LLMClient]] = None):
        self.clients = clients or []
        self.results = []
        
    def add_client(self, client: LLMClient):
        """Add a new LLM client"""
        self.clients.append(client)
        
    def remove_client(self, model_name: str):
        """Remove a client by model name"""
        self.clients = [c for c in self.clients if c.model_name != model_name]
    
    async def generate_all_async(self, prompt: str, **kwargs) -> List[LLMResponse]:
        """Generate responses from all clients asynchronously"""
        tasks = [client.generate_async(prompt, **kwargs) for client in self.clients]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        responses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                responses.append(LLMResponse(
                    model=self.clients[i].model_name,
                    prompt=prompt,
                    response="",
                    timestamp=datetime.now(),
                    latency_ms=0,
                    error=str(result)
                ))
            else:
                responses.append(result)
        
        return responses
    
    def generate_all(self, prompt: str, **kwargs) -> List[LLMResponse]:
        """Generate responses from all clients synchronously"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.generate_all_async(prompt, **kwargs))
        finally:
            loop.close()
    
    def generate_batch(self, prompts: List[str], **kwargs) -> Dict[str, List[LLMResponse]]:
        """Generate responses for multiple prompts from all clients"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=len(self.clients)) as executor:
            # Submit all tasks
            future_to_prompt = {}
            for prompt in prompts:
                for client in self.clients:
                    future = executor.submit(client.generate, prompt, **kwargs)
                    future_to_prompt[future] = (prompt, client.model_name)
            
            # Collect results
            for future in as_completed(future_to_prompt):
                prompt, model_name = future_to_prompt[future]
                try:
                    response = future.result()
                    if prompt not in results:
                        results[prompt] = []
                    results[prompt].append(response)
                except Exception as e:
                    logger.error(f"Error generating response for {model_name}: {str(e)}")
                    if prompt not in results:
                        results[prompt] = []
                    results[prompt].append(LLMResponse(
                        model=model_name,
                        prompt=prompt,
                        response="",
                        timestamp=datetime.now(),
                        latency_ms=0,
                        error=str(e)
                    ))
        
        return results
    
    def save_results(self, filename: str, results: Optional[List[LLMResponse]] = None):
        """Save results to JSON file"""
        if results is None:
            results = self.results
        
        data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_responses": len(results),
                "models": list(set(r.model for r in results))
            },
            "responses": [
                {
                    "model": r.model,
                    "prompt": r.prompt,
                    "response": r.response,
                    "timestamp": r.timestamp.isoformat(),
                    "latency_ms": r.latency_ms,
                    "token_count": r.token_count,
                    "finish_reason": r.finish_reason,
                    "error": r.error
                }
                for r in results
            ]
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(results)} responses to {filename}")

class LLMClientFactory:
    """Factory for creating LLM clients"""
    
    @staticmethod
    def create_client(provider: str, **kwargs) -> LLMClient:
        """Create an LLM client based on provider name"""
        provider_lower = provider.lower()
        
        if provider_lower in ["openai", "gpt", "gpt-3.5", "gpt-4"]:
            model = kwargs.get("model", "gpt-4")
            return OpenAIClient(api_key=kwargs.get("api_key"), model=model)
        
        elif provider_lower in ["anthropic", "claude", "claude-3"]:
            model = kwargs.get("model", "claude-3-opus-20240229")
            return AnthropicClient(api_key=kwargs.get("api_key"), model=model)
        
        elif provider_lower in ["cohere", "command"]:
            model = kwargs.get("model", "command")
            return CohereClient(api_key=kwargs.get("api_key"), model=model)
        
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    @staticmethod
    def create_multi_client(providers: List[str], **kwargs) -> MultiLLMClient:
        """Create a multi-LLM client with specified providers"""
        clients = []
        for provider in providers:
            try:
                client = LLMClientFactory.create_client(provider, **kwargs)
                clients.append(client)
                logger.info(f"Added {provider} client")
            except Exception as e:
                logger.error(f"Failed to create {provider} client: {str(e)}")
        
        return MultiLLMClient(clients)

# Example usage
if __name__ == "__main__":
    # Example 1: Single client usage
    try:
        # Create OpenAI client
        openai_client = LLMClientFactory.create_client("openai", model="gpt-3.5-turbo")
        response = openai_client.generate("What is the capital of France?")
        print(f"OpenAI Response: {response.response}")
    except Exception as e:
        print(f"OpenAI Error: {str(e)}")
    
    # Example 2: Multi-client usage
    multi_client = LLMClientFactory.create_multi_client(
        ["openai", "anthropic", "cohere"],
        api_key=None  # Will use environment variables
    )
    
    # Generate responses from all models
    prompt = "Explain the importance of clean water in 50 words."
    responses = multi_client.generate_all(prompt)
    
    for response in responses:
        if response.error:
            print(f"\n{response.model} - Error: {response.error}")
        else:
            print(f"\n{response.model} - Response ({response.latency_ms:.0f}ms):")
            print(response.response)
    
    # Example 3: Batch processing
    prompts = [
        "What is machine learning?",
        "Explain quantum computing",
        "What are the benefits of exercise?"
    ]
    
    batch_results = multi_client.generate_batch(prompts, max_tokens=100)
    
    for prompt, responses in batch_results.items():
        print(f"\nPrompt: {prompt}")
        for response in responses:
            if not response.error:
                print(f"{response.model}: {response.response[:100]}...")
    
    # Save results
    all_responses = []
    for responses in batch_results.values():
        all_responses.extend(responses)
    multi_client.save_results("llm_responses.json", all_responses)