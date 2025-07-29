# Quick Start Guide

Welcome to MASB-Alt! This guide will help you get started with the Multilingual Alignment Safety Benchmark system in minutes.

## 🚀 Prerequisites

- Python 3.8 or higher
- API keys for at least one LLM provider (OpenAI, Anthropic, or Cohere)
- 4GB+ RAM recommended
- Internet connection for API calls

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/masb-alt/masb-alt.git
cd masb-alt
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env file and add your API keys
# OPENAI_API_KEY=your-openai-key-here
# ANTHROPIC_API_KEY=your-anthropic-key-here
# COHERE_API_KEY=your-cohere-key-here
```

## 🎯 Your First Evaluation

### Option 1: Using the CLI (Easiest)

```bash
# Check system status
python masb_cli.py check

# Initialize the system
python masb_cli.py init

# Run a quick demo
python masb_cli.py demo
```

### Option 2: Using the Orchestrator

```bash
# Run a basic evaluation
python masb_orchestrator.py evaluate \
  --models openai anthropic \
  --languages en sw \
  --domains healthcare education \
  --prompts 5
```

### Option 3: Using Python Script

Create a file `my_first_evaluation.py`:

```python
import asyncio
from masb_orchestrator import MASBOrchestrator

async def main():
    # Initialize the system
    orchestrator = MASBOrchestrator()
    
    # Run evaluation
    dataset_id = await orchestrator.run_evaluation_suite(
        models=["openai"],
        languages=["en", "sw"],
        domains=["healthcare"],
        prompts_per_combination=3
    )
    
    print(f"Evaluation completed! Dataset ID: {dataset_id}")
    
    # Generate report
    report_path = orchestrator.generate_comprehensive_report()
    print(f"Report generated: {report_path}")

if __name__ == "__main__":
    asyncio.run(main())
```

Run it:
```bash
python my_first_evaluation.py
```

## 📊 View Results

### 1. Start the Monitoring Dashboard

```bash
python masb_orchestrator.py monitor
```

Then open your browser to: http://localhost:8501

### 2. Generate Reports

```bash
# Generate HTML report
python masb_orchestrator.py report --format html

# Generate PDF report
python masb_orchestrator.py report --format pdf
```

### 3. Access the API

```bash
# Start API server
python api_server.py
```

API documentation available at: http://localhost:8000/docs

## 🎨 Common Use Cases

### Evaluate Multiple Models

```bash
python masb_orchestrator.py evaluate \
  --models openai anthropic cohere \
  --languages en sw ar \
  --domains healthcare education
```

### Run Performance Benchmarks

```bash
python masb_orchestrator.py benchmark \
  --models openai anthropic \
  --suite standard
```

### Analyze Fine-tuning Opportunities

```bash
python masb_orchestrator.py finetune \
  --model openai \
  --days 30
```

### Compare Models

```bash
python masb_orchestrator.py compare \
  --datasets dataset_001 dataset_002
```

## 🛠️ Configuration

### Basic Configuration

Edit `config.json` to customize:
- Languages to evaluate
- Domains to test
- Evaluation parameters
- Risk thresholds

### Advanced Configuration

See `examples/custom_config.json` for advanced options:
- Model-specific settings
- Performance tuning
- Monitoring configuration
- Cultural sensitivity settings

## 🐛 Troubleshooting

### Common Issues

1. **API Key Errors**
   ```
   Error: Invalid API key
   Solution: Check your .env file and ensure API keys are correct
   ```

2. **Import Errors**
   ```
   Error: Module not found
   Solution: Ensure you're in the virtual environment and dependencies are installed
   ```

3. **Database Errors**
   ```
   Error: Database locked
   Solution: Ensure no other process is using the database
   ```

### Getting Help

1. Run system validation:
   ```bash
   python system_validation.py
   ```

2. Run automatic fixes:
   ```bash
   python fix_issues.py
   ```

3. Check logs:
   ```bash
   tail -f logs/masb_alt.log
   ```

## 📚 Next Steps

1. **Explore Examples**
   - `examples/basic_usage.py` - Simple evaluation examples
   - `examples/advanced_usage.py` - Advanced features

2. **Read Documentation**
   - `USER_GUIDE.md` - Detailed user guide
   - `CONTRIBUTING.md` - Contribution guidelines
   - API documentation at `/docs`

3. **Join the Community**
   - Report issues on GitHub
   - Contribute improvements
   - Share your research

## 🎉 Quick Wins

### 5-Minute Evaluation

```bash
# Run this for a quick taste of MASB-Alt
python masb_cli.py demo --quick
```

This will:
- ✅ Test 2 models (OpenAI, Anthropic)
- ✅ Evaluate 2 languages (English, Swahili)
- ✅ Cover 2 domains (Healthcare, Education)
- ✅ Generate an HTML report
- ✅ Show performance metrics

### 15-Minute Deep Dive

```bash
# More comprehensive evaluation
python masb_orchestrator.py evaluate \
  --models openai anthropic \
  --languages en sw ar hi \
  --domains healthcare education financial_literacy \
  --prompts 10

# Start monitoring
python masb_orchestrator.py monitor

# Generate detailed report
python masb_orchestrator.py report --format html --report-type technical
```

## 🌟 Tips for Success

1. **Start Small**: Begin with 1-2 models and languages
2. **Monitor Progress**: Use the dashboard to track evaluations
3. **Review Reports**: Check HTML reports for detailed insights
4. **Iterate**: Adjust parameters based on results
5. **Document**: Keep notes on your findings

---

**Ready to evaluate AI safety across languages?** 🌍

Start with the demo and gradually explore more features. The system is designed to grow with your needs!

For detailed information, see the [User Guide](USER_GUIDE.md).