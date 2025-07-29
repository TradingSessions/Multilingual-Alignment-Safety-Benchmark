# Contributing to MASB-Alt

Thank you for your interest in contributing to MASB-Alt (Multilingual Alignment Safety Benchmark)! This document provides guidelines for contributing to this AI safety research project.

## 🎯 Project Mission

MASB-Alt aims to improve AI safety and alignment across underrepresented languages. We welcome contributions that advance this mission while maintaining high standards of research integrity and cultural sensitivity.

## 🤝 Code of Conduct

This project adheres to a code of conduct that promotes a welcoming, inclusive environment for all contributors. By participating, you agree to:

- Use welcoming and inclusive language
- Respect differing viewpoints and experiences
- Accept constructive criticism gracefully
- Focus on what's best for the community
- Show empathy towards other community members

## 🚀 How to Contribute

### 1. Reporting Issues

Before creating an issue, please:
- Search existing issues to avoid duplicates
- Use the issue templates when available
- Provide clear, detailed descriptions
- Include system information and steps to reproduce

### 2. Suggesting Features

We welcome feature suggestions that align with our mission:
- Open a GitHub issue with the "enhancement" label
- Describe the feature and its benefits for AI safety
- Consider cultural sensitivity implications
- Provide implementation ideas if possible

### 3. Code Contributions

#### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/masb-alt/masb-alt.git
cd masb-alt

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run system validation
python system_validation.py

# Run tests
pytest tests/
```

#### Development Workflow

1. **Fork the repository** on GitHub
2. **Create a feature branch** from `main`
3. **Make your changes** following our coding standards
4. **Write or update tests** for your changes
5. **Run the test suite** to ensure nothing breaks
6. **Update documentation** as needed
7. **Submit a pull request** with a clear description

#### Coding Standards

- **Python Style**: Follow PEP 8 guidelines
- **Type Hints**: Use type annotations for all functions
- **Documentation**: Include docstrings for all public methods
- **Error Handling**: Implement comprehensive error handling
- **Logging**: Use the project's logging configuration
- **Testing**: Write unit tests for new functionality

#### Code Review Process

1. All submissions require review before merging
2. Reviewers will check for:
   - Code quality and standards compliance
   - Test coverage and functionality
   - Documentation completeness
   - Cultural sensitivity (for language-related changes)
   - Performance implications
   - Security considerations

## 🌍 Language and Cultural Contributions

Given our focus on underrepresented languages, we especially value:

### Language Expertise
- Native speakers who can validate cultural appropriateness
- Linguists who understand language-specific nuances
- Cultural experts who can advise on sensitive topics

### Translation and Localization
- Prompt translations that maintain semantic meaning
- Cultural adaptation of evaluation criteria
- Documentation translation for accessibility

### Evaluation Improvements
- Domain-specific evaluation criteria
- Cultural sensitivity assessments
- Risk detection patterns for specific languages

## 📊 Research Contributions

We welcome research-oriented contributions:

### Data and Datasets
- High-quality multilingual evaluation datasets
- Cultural sensitivity annotations
- Benchmark comparisons with other systems

### Methodology Improvements
- Enhanced evaluation metrics
- Statistical analysis methods
- Bias detection algorithms

### Publications and Documentation
- Research papers using MASB-Alt
- Case studies and best practices
- Tutorial content and examples

## 🔧 Technical Priorities

Current areas where contributions are most needed:

### High Priority
- Support for additional languages (especially African and Asian languages)
- Enhanced cultural sensitivity detection
- Performance optimizations for large-scale evaluation
- Integration with more LLM providers

### Medium Priority
- Advanced visualization features
- Multi-tenant deployment capabilities
- Enhanced reporting formats
- API rate limiting and caching

### Low Priority
- UI/UX improvements for the dashboard
- Additional export formats
- Integration with external tools
- Performance profiling tools

## 📋 Contribution Guidelines

### Pull Request Requirements

- [ ] Code follows project style guidelines
- [ ] Tests pass (run `pytest tests/`)
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated (for significant changes)
- [ ] Cultural sensitivity is considered (for language-related changes)
- [ ] No breaking changes without discussion

### Commit Message Format

Use conventional commit format:
```
type(scope): description

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:
- `feat(evaluation): add cultural sensitivity scoring`
- `fix(api): resolve async timeout issue`
- `docs(readme): update installation instructions`

## 🏷️ Issue Labels

We use the following labels to categorize issues:

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements or additions to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is desired
- `language-specific`: Related to specific language support
- `cultural-sensitivity`: Related to cultural appropriateness
- `performance`: Performance-related improvements
- `security`: Security-related issues

## 🎓 Getting Help

If you need help:
- Check the [README](README.md) and [User Guide](USER_GUIDE.md)
- Search existing issues and discussions
- Join our community discussions
- Reach out to maintainers via GitHub issues

## 🙏 Recognition

Contributors will be recognized in:
- The project's contributors list
- Release notes for significant contributions
- Academic publications when appropriate
- Project documentation

## 📜 License

By contributing to MASB-Alt, you agree that your contributions will be licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

Thank you for helping make AI safer and more inclusive across all languages! 🌍✨