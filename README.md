# VirtuJudge AI-ML

Python worker for presentation analysis, answer analysis, report generation, and AI-owned deletion.

The worker scaffold is the next step. Follow the [AI-ML architecture](https://github.com/VirtuJudge/Docs/blob/main/Architecture/AI-ML-Architecture.md) and the [backend/AI contract](https://github.com/VirtuJudge/Docs/blob/main/Contracts/Backend-AI-Contract.md).

## Development setup

```bash
pip install -e ".[dev]"
```

## Check the repository

```bash
bash scripts/validate-repository.sh
```

## Run the gates

```bash
ruff check app/ tests/
mypy app/
pytest tests/ -q
```
