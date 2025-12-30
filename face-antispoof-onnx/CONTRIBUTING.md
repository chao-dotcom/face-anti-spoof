# Contributing to Face Liveness ONNX

Thanks for helping out! This is the short version of how we work.

## Quick setup
```bash
git clone https://github.com/yourusername/face-liveness-onnx.git
cd face-liveness-onnx
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pip install -e .
pre-commit install
```

## Workflow
1) Create a branch: `git checkout -b feature/your-change`
2) Make focused commits with clear messages
3) Keep docs and examples in sync when behavior changes
4) Open a PR referencing any related issue

## Tests and checks
Before pushing:
```bash
pytest
black --check src tests
flake8 src tests
mypy src
```

## Code style
- PEP8 via Black (88 cols), isort ordering, type hints required
- Prefer clear errors using the custom exceptions in `src/core/exceptions.py`
- Add comments only where intent is non-obvious

## PR checklist
- [ ] Tests added/updated and passing
- [ ] Lint/type checks passing
- [ ] Docs/examples updated (if behavior changes)
- [ ] No unrelated changes bundled

## Getting help
Open a GitHub issue or discussion for questions or proposals. Thanks for contributing!
