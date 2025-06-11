# Contributing to TOKLABEL

Thank you for your interest in contributing to TOKLABEL! This document provides guidelines and instructions for contributing to this project.

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/toklabel.git
   cd toklabel
   ```
3. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
4. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If available
   ```
5. Copy `.env.example` to `.env` and fill in your configuration:
   ```bash
   cp .env.example .env
   ```

## Development Workflow

1. Create a new branch for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Run tests:
   ```bash
   pytest
   ```
4. Commit your changes:
   ```bash
   git commit -m "Description of your changes"
   ```
5. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
6. Create a Pull Request

## Code Style

- Follow PEP 8 style guide
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Write unit tests for new features
- Update documentation as needed

## Pull Request Process

1. Update the README.md with details of changes if needed
2. Update the CHANGELOG.md with a summary of changes
3. Ensure all tests pass
4. The PR will be merged once you have the sign-off of at least one other developer

## Reporting Bugs

- Use the GitHub issue tracker
- Include steps to reproduce the bug
- Include expected and actual behavior
- Include system information if relevant

## Feature Requests

- Use the GitHub issue tracker
- Describe the feature in detail
- Explain why this feature would be useful
- Include any relevant examples

## Questions and Discussion

- Use GitHub Discussions for general questions
- Use GitHub Issues for specific problems or feature requests

## License

By contributing to TOKLABEL, you agree that your contributions will be licensed under the project's license. 