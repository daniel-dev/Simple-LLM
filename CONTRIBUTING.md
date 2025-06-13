# Contributing to LLM Factory

First off, thanks for taking the time to contribute! 🎉

The following is a set of guidelines for contributing to LLM Factory. These are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the existing issues as you might find out that you don't need to create one. When you are creating a bug report, please include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples to demonstrate the steps**
- **Describe the behavior you observed and what behavior you expected**
- **Include screenshots if possible**
- **Include your environment details** (OS, Python version, etc.)

### Suggesting Enhancements

Enhancement suggestions are welcome! Please provide the following information:

- **Use a clear and descriptive title**
- **Provide a step-by-step description of the suggested enhancement**
- **Provide specific examples to demonstrate the steps**
- **Describe the current behavior and explain which behavior you expected**
- **Explain why this enhancement would be useful**

### Pull Requests

- Fill in the required template
- Follow the Python style guidelines
- Include appropriate test coverage
- Update documentation as needed

## Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/llm-factory.git
   cd llm-factory
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If available
   ```

4. **Install pre-commit hooks**
   ```bash
   pip install pre-commit
   pre-commit install
   ```

5. **Run tests to verify setup**
   ```bash
   python -m pytest tests/  # If test directory exists
   python run_pipeline.py --test  # Basic functionality test
   ```

## Pull Request Process

1. **Create a new branch** for your feature or bug fix
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b bugfix/issue-description
   ```

2. **Make your changes** following the style guidelines

3. **Add or update tests** as necessary

4. **Run the test suite** and ensure all tests pass

5. **Update documentation** if you've added new features

6. **Commit your changes** with clear, descriptive commit messages
   ```bash
   git add .
   git commit -m "Add feature: brief description of changes"
   ```

7. **Push to your fork** and create a pull request
   ```bash
   git push origin feature/your-feature-name
   ```

8. **Create a Pull Request** from your fork to the main repository

## Style Guidelines

### Python Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use meaningful variable and function names
- Write docstrings for all functions and classes
- Maximum line length: 88 characters (Black default)
- Use type hints where appropriate

### Git Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line

### Documentation

- Use clear, concise language
- Include code examples where helpful
- Update relevant documentation when making changes
- Follow the existing documentation style

## Project Structure

```
llm-factory/
├── src/                    # Source code
├── data/                   # Data files
├── models/                 # Trained models
├── notebooks/              # Jupyter notebooks
├── static/                 # Web assets
├── templates/              # HTML templates
├── docs/                   # Documentation
├── tests/                  # Test files (if any)
├── requirements.txt        # Dependencies
└── README.md              # Project overview
```

## Questions?

Don't hesitate to ask questions by opening an issue with the "question" label or reaching out to the maintainers.

Thank you for contributing! 🚀
