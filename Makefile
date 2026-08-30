.PHONY: install dev test lint format index ask eval clean

install: ## Install runtime dependencies
	python3 -m pip install -r requirements.txt

dev: ## Install runtime + development dependencies
	python3 -m pip install -r requirements-dev.txt

test: ## Run the test suite
	pytest -q

lint: ## Lint and check formatting
	ruff check docpilot/ main.py tests/
	ruff format --check docpilot/ main.py tests/

format: ## Auto-format the code
	ruff format docpilot/ main.py tests/

index: ## Build the index from sample_docs
	python main.py index sample_docs

ask: ## Ask a question, e.g. make ask Q="what is RAG?"
	python main.py ask "$(Q)"

eval: ## Run the question benchmark
	python main.py eval --questions eval_questions.json --generate-answers

clean: ## Remove build artifacts and the local index
	rm -rf .pytest_cache data/index
	find docpilot tests -type d -name __pycache__ -exec rm -rf {} +
	find . -maxdepth 2 -name '*.egg-info' -exec rm -rf {} +