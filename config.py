# config.py - Configuration management for MASB-Alt

import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import json
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    """LLM provider configuration"""
    provider: str
    api_key: Optional[str] = None
    model: Optional[str] = None
    max_tokens: int = 1000
    temperature: float = 0.7
    timeout: int = 60
    
    def __post_init__(self):
        # Set defaults based on provider
        if not self.model:
            defaults = {
                "openai": "gpt-4",
                "anthropic": "claude-3-opus-20240229",
                "cohere": "command"
            }
            self.model = defaults.get(self.provider.lower(), "unknown")
        
        # Get API key from environment if not provided
        if not self.api_key:
            env_keys = {
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "cohere": "COHERE_API_KEY"
            }
            env_key = env_keys.get(self.provider.lower())
            if env_key:
                self.api_key = os.getenv(env_key)

@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    min_response_length: int = 20
    max_response_length: int = 2000
    confidence_threshold: float = 0.7
    batch_size: int = 10
    max_concurrent_requests: int = 5
    auto_save_interval: int = 5  # minutes
    
@dataclass
class DataConfig:
    """Data management configuration"""
    base_path: Path = Path("./data")
    prompts_dir: Path = field(init=False)
    responses_dir: Path = field(init=False)
    evaluations_dir: Path = field(init=False)
    reports_dir: Path = field(init=False)
    backup_dir: Path = field(init=False)
    
    def __post_init__(self):
        self.prompts_dir = self.base_path / "prompts"
        self.responses_dir = self.base_path / "responses"
        self.evaluations_dir = self.base_path / "evaluations"
        self.reports_dir = self.base_path / "reports"
        self.backup_dir = self.base_path / "backups"
        
        # Create directories
        for dir_path in [self.prompts_dir, self.responses_dir, 
                        self.evaluations_dir, self.reports_dir, self.backup_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

@dataclass
class VisualizationConfig:
    """Visualization configuration"""
    theme: str = "plotly"
    color_scheme: str = "viridis"
    figure_width: int = 1200
    figure_height: int = 800
    export_formats: List[str] = field(default_factory=lambda: ["html", "png", "pdf"])
    
class Config:
    """Main configuration class"""
    
    def __init__(self, config_file: Optional[str] = None):
        # Load from file if provided
        if config_file and Path(config_file).exists():
            self._load_from_file(config_file)
        else:
            self._load_from_env()
        
        # Initialize sub-configurations
        self.llm_configs = self._init_llm_configs()
        self.evaluation = self._init_evaluation_config()
        self.data = self._init_data_config()
        self.visualization = self._init_visualization_config()
        
        # General settings
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.log_file = os.getenv("LOG_FILE", "./logs/masb_alt.log")
        self.debug_mode = os.getenv("DEBUG_MODE", "false").lower() == "true"
        
    def _load_from_file(self, config_file: str):
        """Load configuration from JSON file"""
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        # Set environment variables from config
        for key, value in config_data.items():
            os.environ[key] = str(value)
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        # This is handled by python-dotenv
        pass
    
    def _init_llm_configs(self) -> Dict[str, LLMConfig]:
        """Initialize LLM configurations"""
        configs = {}
        
        # Parse LLM providers from environment
        providers = os.getenv("LLM_PROVIDERS", "openai,anthropic,cohere").split(",")
        
        for provider in providers:
            provider = provider.strip().lower()
            config = LLMConfig(
                provider=provider,
                max_tokens=int(os.getenv(f"{provider.upper()}_MAX_TOKENS", 
                                        os.getenv("DEFAULT_MAX_TOKENS", "1000"))),
                temperature=float(os.getenv(f"{provider.upper()}_TEMPERATURE", 
                                           os.getenv("DEFAULT_TEMPERATURE", "0.7")))
            )
            configs[provider] = config
        
        return configs
    
    def _init_evaluation_config(self) -> EvaluationConfig:
        """Initialize evaluation configuration"""
        return EvaluationConfig(
            min_response_length=int(os.getenv("MIN_RESPONSE_LENGTH", "20")),
            max_response_length=int(os.getenv("MAX_RESPONSE_LENGTH", "2000")),
            confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.7")),
            batch_size=int(os.getenv("BATCH_SIZE", "10")),
            max_concurrent_requests=int(os.getenv("MAX_CONCURRENT_REQUESTS", "5"))
        )
    
    def _init_data_config(self) -> DataConfig:
        """Initialize data configuration"""
        base_path = Path(os.getenv("DATA_PATH", "./data"))
        return DataConfig(base_path=base_path)
    
    def _init_visualization_config(self) -> VisualizationConfig:
        """Initialize visualization configuration"""
        return VisualizationConfig(
            theme=os.getenv("VIZ_THEME", "plotly"),
            color_scheme=os.getenv("VIZ_COLOR_SCHEME", "viridis")
        )
    
    def get_llm_client_config(self, provider: str) -> Optional[LLMConfig]:
        """Get configuration for a specific LLM provider"""
        return self.llm_configs.get(provider.lower())
    
    def get_enabled_providers(self) -> List[str]:
        """Get list of enabled LLM providers"""
        return list(self.llm_configs.keys())
    
    def save_to_file(self, output_file: str):
        """Save current configuration to file"""
        config_data = {
            "llm_providers": {
                provider: {
                    "model": config.model,
                    "max_tokens": config.max_tokens,
                    "temperature": config.temperature
                }
                for provider, config in self.llm_configs.items()
            },
            "evaluation": {
                "min_response_length": self.evaluation.min_response_length,
                "max_response_length": self.evaluation.max_response_length,
                "confidence_threshold": self.evaluation.confidence_threshold,
                "batch_size": self.evaluation.batch_size
            },
            "data": {
                "base_path": str(self.data.base_path)
            },
            "visualization": {
                "theme": self.visualization.theme,
                "color_scheme": self.visualization.color_scheme
            },
            "general": {
                "log_level": self.log_level,
                "log_file": self.log_file,
                "debug_mode": self.debug_mode
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Configuration saved to {output_file}")
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Check API keys
        for provider, config in self.llm_configs.items():
            if not config.api_key:
                issues.append(f"Missing API key for {provider}")
        
        # Check paths
        if not self.data.base_path.exists():
            issues.append(f"Data path does not exist: {self.data.base_path}")
        
        # Check numeric ranges
        if self.evaluation.min_response_length < 0:
            issues.append("min_response_length must be positive")
        
        if self.evaluation.confidence_threshold < 0 or self.evaluation.confidence_threshold > 1:
            issues.append("confidence_threshold must be between 0 and 1")
        
        return issues
    
    def __repr__(self) -> str:
        """String representation of configuration"""
        return f"""MASB-Alt Configuration:
- LLM Providers: {', '.join(self.get_enabled_providers())}
- Data Path: {self.data.base_path}
- Batch Size: {self.evaluation.batch_size}
- Debug Mode: {self.debug_mode}
"""

# Singleton instance
_config_instance: Optional[Config] = None

def get_config(config_file: Optional[str] = None) -> Config:
    """Get or create configuration instance"""
    global _config_instance
    
    if _config_instance is None:
        _config_instance = Config(config_file)
    
    return _config_instance

def reset_config():
    """Reset configuration (mainly for testing)"""
    global _config_instance
    _config_instance = None

# Example usage
if __name__ == "__main__":
    # Load configuration
    config = get_config()
    
    # Validate
    issues = config.validate()
    if issues:
        print("Configuration issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Configuration is valid")
    
    # Display configuration
    print("\n" + str(config))
    
    # Save example configuration
    config.save_to_file("config_example.json")