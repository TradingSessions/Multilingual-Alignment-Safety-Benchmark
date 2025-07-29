
# Multilingual Alignment Safety Benchmark (MASB-Alt)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-available-green)](./USER_GUIDE.md)

A comprehensive framework for evaluating and improving multilingual alignment and safety in large language models (LLMs), with a focus on underrepresented languages and sensitive domains.

## 🌍 Overview

MASB-Alt addresses the critical need for equitable AI systems by:
- **Evaluating LLM performance** across diverse languages, especially underrepresented ones
- **Assessing safety and alignment** in culturally sensitive contexts
- **Identifying gaps** in multilingual capabilities
- **Providing actionable insights** for model improvement

### Key Features

- 🌐 **10+ Language Support**: Including Swahili, Uyghur, Vietnamese, Arabic, Hindi, and more
- 🏥 **Multi-Domain Coverage**: Healthcare, Education, Civic Participation, Financial Literacy, Technology
- 🤖 **Multi-Model Integration**: OpenAI, Anthropic, Cohere, and custom models
- 📊 **Comprehensive Evaluation**: Automated and human-in-the-loop assessment
- 📈 **Rich Visualizations**: Interactive dashboards and detailed reports
- 🔧 **Extensible Framework**: Easy to add new languages, domains, and models

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/your-org/masb-alt.git
cd masb-alt

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Run a quick demo
python examples/quick_demo.py

# Run evaluation
python run_evaluation.py --languages en sw --domains healthcare --num-prompts 5
```

## 📁 Project Structure

```
Multilingual-Alignment-Safety-Benchmark2/
├── prompt_generator/           # Multilingual prompt generation
│   └── multilingual_prompt_gen.py
├── evaluation/                 # Evaluation framework
│   ├── evaluator.py           # Automated evaluation
│   └── human_evaluation_tool.py # GUI for human evaluation
├── visualization/              # Data visualization tools
│   └── visualization_dashboard.py
├── examples/                   # Usage examples
│   ├── quick_demo.py
│   ├── batch_evaluation_example.py
│   └── custom_integration_example.py
├── llm_api_client.py          # LLM API integration
├── data_manager.py            # Data management system
├── run_evaluation.py          # Main evaluation pipeline
├── config.py                  # Configuration management
├── requirements.txt           # Python dependencies
├── .env.example              # Environment variables template
├── USER_GUIDE.md             # Detailed documentation
└── README.md                 # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager
- API keys for LLM providers (OpenAI, Anthropic, Cohere)

### Detailed Setup

1. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API keys**
   ```bash
   cp .env.example .env
   # Add your API keys to .env file
   ```

4. **Initialize database**
   ```bash
   python -c "from data_manager import DataManager; DataManager()"
   ```

## 📊 Core Components

### 1. Prompt Generator
Generate culturally-aware prompts across multiple languages and domains:

```python
from prompt_generator.multilingual_prompt_gen import MultilingualPromptGenerator

generator = MultilingualPromptGenerator()
prompt = generator.get_random_prompt("healthcare", language="sw")
```

### 2. LLM Integration
Seamlessly work with multiple LLM providers:

```python
from llm_api_client import LLMClientFactory

client = LLMClientFactory.create_client("openai", model="gpt-4")
response = client.generate("Your prompt here")
```

### 3. Evaluation Framework
Comprehensive evaluation with risk detection:

```python
from evaluation.evaluator import MultilingualAlignmentEvaluator

evaluator = MultilingualAlignmentEvaluator()
result = evaluator.evaluate_response(prompt, llm_output, llm_model)
```

### 4. Visualization Dashboard
Interactive visualizations and reports:

```bash
# Launch interactive dashboard
streamlit run visualization/visualization_dashboard.py
```

## 📈 Evaluation Metrics

- **Alignment Score** (1-5): Overall quality and appropriateness
- **Risk Flags**: Hallucination, unsafe advice, cultural insensitivity
- **Risk Levels**: Low, Medium, High, Critical
- **Performance Metrics**: Latency, token usage, error rates

## 🌍 Supported Languages

| Language | Code | Status |
|----------|------|--------|
| English | en | ✅ Full Support |
| Swahili | sw | ✅ Full Support |
| Vietnamese | vi | ✅ Full Support |
| Uyghur | ug | ✅ Full Support |
| Arabic | ar | ✅ Full Support |
| Hindi | hi | ✅ Full Support |
| Bengali | bn | 🔄 In Progress |
| Yoruba | yo | 🔄 In Progress |
| Amharic | am | 🔄 In Progress |
| Burmese | my | 🔄 In Progress |

## 🏥 Evaluation Domains

- **Healthcare**: Medical advice, symptoms, wellness
- **Education**: Learning strategies, academic guidance
- **Civic Participation**: Democratic processes, rights
- **Financial Literacy**: Budgeting, financial planning
- **Technology Literacy**: Digital safety, AI understanding

## 🔧 Extending the Framework

### Add a New Language

```python
# In multilingual_prompt_gen.py
SUPPORTED_LANGUAGES["new_lang"] = "Language Name"

# Add translations to prompt templates
"new_lang": "Translated prompt text"
```

### Add a Custom Domain

```python
generator.add_custom_prompt("new_domain", {
    "en": "English prompt",
    "sw": "Swahili prompt",
    "risk_level": "medium",
    "tags": ["tag1", "tag2"]
})
```

### Integrate Custom LLM

```python
class CustomLLMClient(LLMClient):
    async def generate_async(self, prompt: str, **kwargs) -> LLMResponse:
        # Your implementation
        pass
```

## 📊 Example Results

```
=== Evaluation Summary ===
Total Evaluations: 150
Average Alignment Score: 3.82/5

Scores by Language:
  en: 4.15 (n=50)
  sw: 3.68 (n=50)
  ar: 3.62 (n=50)

Risk Level Distribution:
  low: 105 (70.0%)
  medium: 35 (23.3%)
  high: 10 (6.7%)
```

## 🤝 Contributing

We welcome contributions! Areas of particular interest:

- Adding support for more languages
- Expanding domain coverage
- Improving evaluation metrics
- Enhancing visualization capabilities

Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📚 Documentation

- [User Guide](USER_GUIDE.md) - Detailed usage instructions
- [API Reference](docs) - Complete API documentation
- [Examples](examples/) - Code examples and tutorials

## 🔐 Ethical Considerations

This project is designed to:
- Promote AI fairness across languages and cultures
- Identify and mitigate potential harms
- Support responsible AI development
- Protect user privacy and data security

## 📜 Citation

If you use MASB-Alt in your research, please cite:

```bibtex
@software{masb_alt_2024,
  title = {Multilingual Alignment Safety Benchmark (MASB-Alt)},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/your-org/masb-alt}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

- Project Lead: Trading Session
- Email: tradingsession568@gmail.com
  
---

**Note**: This project is under active development. Features and APIs may change. We recommend using tagged releases for production use.
