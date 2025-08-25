# Contributing to Python & 3D Programming Book 🤝

Thank you for your interest in contributing to the Python & 3D Programming Book! This document provides guidelines and information for contributors.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Style Guidelines](#style-guidelines)
- [Pull Request Process](#pull-request-process)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)
- [Code Review Process](#code-review-process)
- [Release Process](#release-process)

## 📜 Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

### Our Standards

- **Be respectful and inclusive**: Use welcoming and inclusive language
- **Be collaborative**: Work together to achieve common goals
- **Be constructive**: Provide constructive feedback and suggestions
- **Be professional**: Maintain professional behavior in all interactions

## 🎯 How Can I Contribute?

### 🐛 Reporting Bugs

- Use the GitHub issue tracker
- Include detailed steps to reproduce the bug
- Provide system information (OS, Python version, etc.)
- Include error messages and stack traces

### 💡 Suggesting Enhancements

- Use the GitHub issue tracker with the "enhancement" label
- Describe the feature and its benefits
- Provide examples of how it would be used
- Consider implementation complexity

### 📝 Improving Documentation

- Fix typos and grammatical errors
- Clarify unclear explanations
- Add missing examples
- Update outdated information

### 🔧 Adding Code Examples

- Ensure examples are functional and well-documented
- Follow the existing code style
- Include appropriate error handling
- Add comments explaining complex logic

### 🧪 Writing Tests

- Add unit tests for new functionality
- Ensure existing tests pass
- Maintain good test coverage
- Use descriptive test names

## 🛠️ Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- A code editor (VS Code, PyCharm, etc.)

### Setup Steps

1. **Fork the repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/yourusername/python-3d-programming-book.git
   cd python-3d-programming-book
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install development dependencies**
   ```bash
   pip install -r requirements-dev.txt  # If available
   ```

5. **Set up pre-commit hooks** (optional)
   ```bash
   pre-commit install
   ```

## 📏 Style Guidelines

### Python Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- Use 4 spaces for indentation (no tabs)
- Maximum line length of 88 characters (Black formatter)
- Use descriptive variable and function names
- Add type hints where appropriate

### Documentation Style

- Use clear, concise language
- Include code examples where helpful
- Follow the existing documentation structure
- Use proper Markdown formatting

### Commit Messages

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests after the first line

Example:
```
Add real-time ray tracing example

- Implement hardware-accelerated ray tracing
- Add denoising and temporal accumulation
- Include performance benchmarks

Closes #123
```

## 🔄 Pull Request Process

### Before Submitting

1. **Ensure your code works**
   - Test all examples
   - Run linting tools
   - Check for any new warnings

2. **Update documentation**
   - Update README files if needed
   - Add docstrings to new functions
   - Update any relevant documentation

3. **Follow the checklist**
   - [ ] Code follows style guidelines
   - [ ] Self-review completed
   - [ ] Comments added for complex sections
   - [ ] Documentation updated
   - [ ] Tests pass
   - [ ] No new warnings generated

### Submitting a Pull Request

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write your code
   - Add tests if applicable
   - Update documentation

3. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add your descriptive commit message"
   ```

4. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create a Pull Request**
   - Go to your fork on GitHub
   - Click "New Pull Request"
   - Select your feature branch
   - Fill out the PR template

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex sections
- [ ] Documentation updated
- [ ] No new warnings generated

## Additional Notes
Any additional information or context
```

## 🐛 Reporting Bugs

### Bug Report Template

```markdown
## Bug Description
Clear and concise description of the bug

## Steps to Reproduce
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

## Expected Behavior
What you expected to happen

## Actual Behavior
What actually happened

## Environment
- OS: [e.g. Windows 10, macOS 12.0, Ubuntu 20.04]
- Python Version: [e.g. 3.8.10]
- OpenGL Version: [e.g. 4.6]
- Graphics Card: [e.g. NVIDIA GTX 1080]

## Additional Context
Any other context about the problem
```

## 💡 Suggesting Enhancements

### Enhancement Request Template

```markdown
## Enhancement Description
Clear and concise description of the enhancement

## Problem Statement
What problem does this enhancement solve?

## Proposed Solution
Description of the proposed solution

## Alternative Solutions
Any alternative solutions you've considered

## Additional Context
Any other context, screenshots, or examples
```

## 👀 Code Review Process

### For Contributors

1. **Address review comments**
   - Respond to all review comments
   - Make requested changes
   - Push updates to your branch

2. **Be patient**
   - Reviews may take time
   - Maintainers are volunteers
   - Be respectful of their time

### For Reviewers

1. **Be constructive**
   - Provide helpful feedback
   - Suggest improvements
   - Be respectful and encouraging

2. **Check thoroughly**
   - Review code quality
   - Test functionality
   - Verify documentation

## 🚀 Release Process

### For Maintainers

1. **Version bump**
   - Update version numbers
   - Update changelog
   - Tag the release

2. **Quality assurance**
   - Run all tests
   - Check documentation
   - Verify examples work

3. **Release**
   - Create GitHub release
   - Update documentation
   - Announce to community

## 📞 Getting Help

If you need help with contributing:

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: For private or sensitive matters

## 🙏 Recognition

Contributors will be recognized in:

- The project README
- Release notes
- Contributor statistics
- Special acknowledgments for significant contributions

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project (MIT License).

---

Thank you for contributing to the Python & 3D Programming Book! 🎉
