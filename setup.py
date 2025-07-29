# setup.py - Installation script for MASB-Alt

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="masb-alt",
    version="1.0.0",
    author="MASB-Alt Development Team",
    author_email="masb-alt@example.com",
    description="Multilingual Alignment Safety Benchmark - Alternative Implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/masb-alt/masb-alt",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Testing",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Core dependencies
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "requests>=2.28.0",
        "aiohttp>=3.8.0",
        "asyncio",
        
        # Data processing
        "sqlite3",
        "json5>=0.9.0",
        "python-dotenv>=0.19.0",
        
        # Machine learning and evaluation
        "scikit-learn>=1.1.0",
        "scipy>=1.9.0",
        "transformers>=4.20.0",
        "torch>=1.12.0",
        
        # Visualization
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "plotly>=5.10.0",
        
        # Web frameworks
        "fastapi>=0.85.0",
        "uvicorn>=0.18.0",
        "streamlit>=1.15.0",
        
        # Document generation
        "jinja2>=3.1.0",
        "pdfkit>=1.0.0",
        "python-docx>=0.8.11",
        "markdown>=3.4.0",
        
        # Email support
        "smtplib",
        "email",
        
        # Progress bars and utilities
        "tqdm>=4.64.0",
        "python-dateutil>=2.8.0",
        "pathlib",
        "hashlib",
        "logging",
        
        # API clients
        "openai>=0.27.0",
        "anthropic>=0.3.0",
        "cohere>=4.0.0",
        
        # Statistical analysis
        "statsmodels>=0.13.0",
        "pingouin>=0.5.0",
        
        # Additional utilities
        "colorama>=0.4.5",
        "rich>=12.5.0",
        "click>=8.1.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.1.0",
            "pytest-asyncio>=0.19.0",
            "pytest-cov>=3.0.0",
            "black>=22.6.0",
            "flake8>=5.0.0",
            "mypy>=0.971",
            "pre-commit>=2.20.0"
        ],
        "docs": [
            "sphinx>=5.1.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0"
        ],
        "gpu": [
            "torch>=1.12.0+cu116",
            "transformers[torch]>=4.20.0"
        ]
    },
    entry_points={
        "console_scripts": [
            "masb-alt=masb_orchestrator:main",
            "masb-evaluate=masb_orchestrator:main",
            "masb-benchmark=benchmark_runner:main",
            "masb-monitor=monitoring_dashboard:main",
            "masb-report=report_generator:main",
            "masb-corpus=corpus_manager:main",
            "masb-history=history_tracker:main"
        ],
    },
    include_package_data=True,
    package_data={
        "masb_alt": [
            "templates/*.html",
            "templates/*.md",
            "config/*.json",
            "data/*.sql"
        ]
    },
    zip_safe=False,
)