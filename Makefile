.PHONY: *
.DEFAULT_GOAL := help

SHELL := /bin/bash

##@ Setup
deps: ## Install dependencies
	@uv sync

##@ Testing
test: ## Run tests
	@uv run python -m pytest -v

can-release: lint test ## Run all checks

##@ Running
run: ## Run the agent in interactive mode
	@uv run agent run

run-headless: ## Run the agent in headless mode (PROMPT="...")
	@if [ -z "$(PROMPT)" ]; then \
		echo "PROMPT is required. Example: make run-headless PROMPT=\"List all Python files\""; \
		exit 1; \
	fi
	@uv run agent run --headless "$(PROMPT)"

##@ Linting
lint: ## Run linters and type checks
	@uv run ruff check src/agent tests examples
	@uv run ruff format --check src/agent tests examples
	@uv run mypy src/agent

format: ## Format code
	@uv run ruff format src/agent tests examples

##@ Maintenance
clean: ## Remove cache artifacts
	@rm -rf .pytest_cache .mypy_cache .ruff_cache **/__pycache__

##@ Helpers
fmt: format
t: test

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_\/-]+:.*?##/ { printf "  \033[36m%-25s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
