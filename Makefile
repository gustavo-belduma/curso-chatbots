# Makefile for managing the tools project

# Variables
VENV_NAME = .venv
PYTHON_FILES = $(shell python -c "import toml; print(' '.join(toml.load('pyproject.toml')['tool']['setuptools']['packages']['find']['include']))")

# Detect the operating system
ifeq ($(OS),Windows_NT)
    OS_NAME := Windows
else
    OS_NAME := $(shell uname -s)
endif

# install uv
install_uv:
ifeq ($(OS_NAME),Linux)
	@echo "Detected Linux OS"
	curl -LsSf https://astral.sh/uv/install.sh | sh
else ifeq ($(OS_NAME),Darwin)
	@echo "Detected macOS"
	curl -LsSf https://astral.sh/uv/install.sh | sh
else ifeq ($(OS_NAME),Windows)
	@echo "Detected Windows OS"
	powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
	# "If you are on Windows and this doesn't work, check your permissions or run the command manually."
endif

# Create a virtual environment
venv:
	uv venv $(VENV_NAME)
	@echo "Virtual $(VENV_NAME) environment created."
	@echo "To activate the virtual environment, please run: source $(VENV_NAME)/bin/activate"


# Install dependencies using uv
install_old:
	uv pip install pip --upgrade
	uv pip install -r requirements.txt
	uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
	@echo "Dependencies installed."
	@echo "To activate the virtual environment, please run: source $(VENV_NAME)/bin/activate or the corresponding command for your operating system."

# Install dependencies using uv
install:
	uv pip install pip --upgrade
	uv pip install -r requirements-app.txt
	@echo "Dependencies installed."
	@echo "To activate the virtual environment, please run: source $(VENV_NAME)/bin/activate or the corresponding command for your operating system."


# Pre-commit hooks
pre_commit:
	pre-commit install
	@echo "Pre-commit hooks installed."

# Clean up
clean:
ifeq ($(OS_NAME),Windows)
	del /s /q $(VENV_NAME)
	rmdir /s /q $(VENV_NAME)
else
	rm -rf $(VENV_NAME)
endif
	@echo "Cleaned up the environment."


# Tests
test:
	pytest
	@echo "Tests executed."

# Build the Docker image
# build:
# 	docker build -t tools .
# 	@echo "Docker image built."

# Run the Docker container : docker run -it --rm mle_test ls -la venv
# run:
# 	docker run -it --rm --entrypoint /bin/bash tools
# 	@echo "Docker container running with interactive shell."

# Linting and formatting
.PHONY: format
format:
	black --line-length=99 $(PYTHON_FILES)
	isort --profile black --line-length 99 $(PYTHON_FILES)
	@echo "Formatting completed."

.PHONY: check-format
check-format:
	black --check --line-length=99 $(PYTHON_FILES)
	isort --check --diff --profile black --line-length 99 $(PYTHON_FILES)
	@echo "Format check completed."

.PHONY: flake8
flake8:
	flake8 --max-line-length=99 --ignore=E501,W503,E203 $(PYTHON_FILES)
	@echo "Flake8 check completed."

.PHONY: bandit
bandit:
	bandit --skip=B101 -r $(PYTHON_FILES) --exclude tests/
	@echo "Bandit security check completed."

.PHONY: mypy
mypy:  ## Run mypy type checker
	mypy --config-file pyproject.toml $(PYTHON_FILES)
	@echo "MyPy type checking completed."

.PHONY: pyupgrade
pyupgrade:
	pyupgrade --py312-plus $(PYTHON_FILES)
	@echo "Python code upgraded to Python 3.12+ syntax."


.PHONY: lint
# bandit mypy
lint: format flake8
	@echo "All linting checks completed."

.PHONY: check-all
check-all: check-format lint test
	@echo "All checks and tests completed."

# Help
help:
	@echo "Makefile for tools"
	@echo "Usage:"
	@echo "  make install_uv   - Install uv tool"
	@echo "  make venv         - Create a virtual environment"
	@echo "  make activate     - Activate the virtual environment"
	@echo "  make install      - Install dependencies using uv"
	@echo "  make pre_commit   - Install pre-commit hooks"
	@echo "  make pyupgrade    - Upgrade Python syntax to 3.12+"
	@echo "  make pre-commit-hooks - Run basic pre-commit hooks"
	@echo "  make format       - Format code with black and isort"
	@echo "  make check-format - Check formatting without changing files"
	@echo "  make flake8       - Run flake8 linting"
	@echo "  make bandit       - Run bandit security checks"
	@echo "  make mypy         - Run mypy type checking"
	@echo "  make lint         - Run all linting tools"
	@echo "  make test         - Run pytest"
	@echo "  make check-all    - Run all checks and tests"
	@echo "  make clean        - Clean up the environment"
	@echo "  make build        - Build the Docker image"
	@echo "  make run          - Run the Docker container"
	@echo "  make help         - Display this help message"
