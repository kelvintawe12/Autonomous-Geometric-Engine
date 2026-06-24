# Contributing to AGE

Thank you for your interest in contributing to the Autonomous Geometric Engine (AGE)! This document provides guidelines and instructions for contributing to the project.

## 🤝 How to Contribute

### Reporting Bugs

Before creating bug reports, please check the existing issues to avoid duplicates. When creating a bug report, include:

- **Clear description** of the problem
- **Minimal reproducible example** showing the bug
- **Expected behavior** vs actual behavior
- **Environment details**: Python version, OS, package versions
- **Screenshots** or logs if applicable

### Suggesting Enhancements

Enhancement suggestions are welcome! Please:

- **Describe the use case** for the enhancement
- **Provide examples** of how it would be used
- **Discuss implementation ideas** if you have them
- **Consider trade-offs** and limitations

### Pull Request Process

1. **Fork the repository** and create a feature branch
2. **Make your changes** following the coding standards
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Run tests** to ensure nothing breaks
6. **Submit a pull request** with clear description

## 📋 Development Setup

### Clone and Install

```bash
git clone https://github.com/yourusername/autonomous-geometric-engine.git
cd autonomous-geometric-engine
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest test_suite.py

# Run with coverage
pytest test_suite.py --cov=age --cov-report=html

# Run specific test file
pytest test_parameter_tuning.py
```

### Code Quality

```bash
# Format code with black
black age.py

# Lint with flake8
flake8 age.py

# Type check with mypy
mypy age.py
```

## 🎯 Coding Standards

### Code Style

- Follow **PEP 8** guidelines
- Use **Black** for code formatting
- Keep functions **focused and single-purpose**
- Add **docstrings** to all public functions and classes
- Use **type hints** where appropriate

### Documentation

- Update **README.md** for user-facing changes
- Add **docstrings** to new functions
- Update **API reference** if changing public API
- Add **examples** for new features

### Testing

- Write **unit tests** for new functionality
- Ensure **test coverage** doesn't decrease
- Add **integration tests** for complex features
- Test on **Python 3.8+** and multiple platforms

## 🏗️ Project Structure

```
autonomous-geometric-engine/
├── age.py                  # Main AGE implementation
├── benchmark_age.py       # Benchmark suite
├── setup.py               # Package configuration
├── requirements.txt       # Core dependencies
├── environment.yml        # Conda environment
├── README.md              # Main documentation
├── CONTRIBUTING.md        # Contribution guidelines
├── LICENSE                # MIT License
├── examples/              # Usage examples
│   ├── basic_usage.py
│   ├── ensemble_clustering.py
│   ├── real_world_applications.py
│   └── integration_guide.py
├── tests/                 # Unit tests
│   ├── test_suite.py
│   ├── test_academic_features.py
│   └── test_parameter_tuning.py
└── .github/               # GitHub workflows
    └── workflows/
        └── ci.yml
```

## 🔄 Development Workflow

### Feature Development

1. Create a new branch: `git checkout -b feature/my-feature`
2. Make changes and commit: `git commit -m "Add my feature"`
3. Push to fork: `git push origin feature/my-feature`
4. Create pull request

### Bug Fixes

1. Create branch: `git checkout -b fix/bug-description`
2. Fix the bug and add tests
3. Commit: `git commit -m "Fix: describe the fix"`
4. Push and create pull request

### Documentation Updates

1. Create branch: `git checkout -b docs/update-docs`
2. Update documentation
3. Commit: `git commit -m "Docs: update documentation"`
4. Push and create pull request

## 📝 Commit Message Guidelines

Follow conventional commits format:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test changes
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `chore:` Maintenance tasks

Examples:
```
feat: add ensemble clustering method
fix: resolve memory leak in envelope building
docs: update API reference
test: add tests for geometry detection
```

## 🧪 Testing Guidelines

### Unit Tests

- Test **individual functions** in isolation
- Use **fixtures** for common test data
- Mock **external dependencies**
- Test **edge cases** and error conditions

### Integration Tests

- Test **component interactions**
- Use **realistic data** when possible
- Test **end-to-end workflows**
- Validate **performance characteristics**

### Benchmark Tests

- Run **benchmark_age.py** before changes
- Ensure **performance doesn't degrade**
- Document **any performance changes**
- Update **benchmarks** for improvements

## 🚀 Release Process

Releases are managed through GitHub releases:

1. Update **version** in setup.py
2. Update **CHANGELOG.md**
3. Create **git tag**: `git tag v1.0.0`
4. Push tag: `git push origin v1.0.0`
5. Create **GitHub release**
6. **CI/CD** publishes to PyPI automatically

## 💬 Communication

- **GitHub Issues** for bugs and features
- **GitHub Discussions** for questions and ideas
- **Pull Requests** for code contributions
- **Email** for security issues

## 📜 Code of Conduct

- Be **respectful** and inclusive
- Focus on **constructive feedback**
- Welcome **new contributors**
- Follow **open source principles**

## 🎓 Getting Help

- Read the **documentation** first
- Check **existing issues**
- Ask questions in **GitHub Discussions**
- Join our **community channels**

## 🙏 Recognition

Contributors will be recognized in:
- **CONTRIBUTORS.md** file
- **Release notes**
- **Documentation acknowledgments**

Thank you for contributing to AGE! 🎉