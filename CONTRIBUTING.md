# Contributing to MoodMentor

Thank you for your interest in contributing to the **AI-Based Employee Wellness Management Platform (MoodMentor)**!

This project is part of the **Infosys Springboard Virtual Internship**. Follow the guidelines below to submit improvements, bug fixes, or enhancements.

---

## 🛠️ Development Setup

1. **Fork and Clone:**
   ```bash
   git clone https://github.com/DeekshaG96/AI_Based_Employee_Wellness_Management_Platform.git
   cd AI_Based_Employee_Wellness_Management_Platform
   ```

2. **Create a Virtual Environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install Editable Package & Dependencies:**
   ```bash
   pip install -e ".[dev,ui]"
   python -m wellness.cli --download-nltk-data
   ```

4. **Verify Tests & Code Style:**
   ```bash
   pytest
   ruff check .
   ```

---

## 📐 Coding Standards

- **Code Style:** Formatted and linted using [Ruff](https://github.com/astral-sh/ruff) (`line-length = 100`, `py310+`).
- **Typing:** Use type annotations for function signatures.
- **Testing:** Every new feature, parser, or database helper must have corresponding tests under `tests/`.
- **Git Commits:** Use clear, descriptive commit messages following the Conventional Commits style (`feat:`, `fix:`, `docs:`, `test:`).

---

## 🚀 Submitting Pull Requests

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes and verify that all tests pass (`pytest`).
3. Commit and push your branch to GitHub.
4. Open a Pull Request against the `main` branch with a clear description of the modifications.
