# Структура пакета и зависимости NormaML

## Структура проекта

```
normaml/
├── README.md
├── LICENSE
├── pyproject.toml
├── setup.py
├── requirements/
│   ├── base.txt
│   ├── dev.txt
│   ├── docker.txt
│   └── optional.txt
├── docker/
│   ├── Dockerfile.base
│   ├── Dockerfile.sklearn
│   ├── Dockerfile.xgboost
│   └── docker-compose.yml
├── .github/
│   └── workflows/
│       ├── ci.yml
│       ├── publish.yml
│       └── docker-build.yml
├── docs/
│   ├── conf.py
│   ├── index.rst
│   ├── user_guide/
│   ├── api_reference/
│   └── examples/
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_core/
│   │   ├── test_plugins/
│   │   ├── test_data/
│   │   └── test_execution/
│   ├── integration/
│   │   ├── test_pipelines/
│   │   ├── test_docker/
│   │   └── test_mlflow/
│   └── performance/
│       ├── test_memory/
│       ├── test_speed/
│       └── benchmarks/
├── examples/
│   ├── basic_usage/
│   ├── custom_plugins/
│   ├── distributed/
│   └── notebooks/
├── normaml/
│   ├── __init__.py
│   ├── __version__.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── manager.py
│   │   ├── schemas.py
│   │   └── defaults.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── interfaces.py
│   │   ├── plugin_manager.py
│   │   ├── data_manager.py
│   │   ├── execution_engine.py
│   │   ├── experiment_tracker.py
│   │   └── event_bus.py
│   ├── plugins/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── data_loaders/
│   │   │   ├── __init__.py
│   │   │   ├── csv_loader.py
│   │   │   ├── parquet_loader.py
│   │   │   └── db_loader.py
│   │   ├── feature_engineers/
│   │   │   ├── __init__.py
│   │   │   ├── polars_engineer.py
│   │   │   └── sklearn_engineer.py
│   │   ├── model_trainers/
│   │   │   ├── __init__.py
│   │   │   ├── sklearn_trainer.py
│   │   │   └── xgboost_trainer.py
│   │   ├── evaluators/
│   │   │   ├── __init__.py
│   │   │   ├── classification_evaluator.py
│   │   │   └── regression_evaluator.py
│   │   └── backends/
│   │       ├── __init__.py
│   │       ├── sklearn_backend.py
│   │       └── xgboost_backend.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── zero_copy.py
│   │   ├── shared_memory.py
│   │   ├── streaming.py
│   │   └── optimization.py
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── scheduler.py
│   │   ├── engines/
│   │   │   ├── __init__.py
│   │   │   ├── async_engine.py
│   │   │   ├── process_engine.py
│   │   │   └── docker_engine.py
│   │   ├── resources.py
│   │   └── load_balancer.py
│   ├── tracking/
│   │   ├── __init__.py
│   │   ├── mlflow_integration.py
│   │   ├── plugin_logger.py
│   │   └── distributed_coordinator.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logging.py
│   │   ├── serialization.py
│   │   └── validation.py
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── commands/
│   │   │   ├── __init__.py
│   │   │   ├── run.py
│   │   │   ├── plugins.py
│   │   │   └── experiments.py
│   │   └── config.py
│   └── api/
│       ├── __init__.py
│       ├── public.py
│       ├── pipeline.py
│       └── factory.py
└── scripts/
    ├── install_dev.sh
    ├── run_tests.sh
    ├── build_docker.sh
    └── deploy.sh
```

## Основные зависимости

### pyproject.toml

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "normaml"
version = "1.0.0"
description = "Modular Machine Learning Library with Zero-Copy Data Processing"
readme = "README.md"
requires-python = ">=3.9"
license = {text = "GPL-3.0"}
authors = [
    {name = "NormaML Team", email = "team@normaml.org"}
]
keywords = ["machine-learning", "data-science", "zero-copy", "modular", "mlflow"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries :: Python Modules",
]

dependencies = [
    # Core data processing
    "polars>=0.19.0",
    "pyarrow>=10.0.0",
    "numpy>=1.21.0",
    
    # MLflow integration
    "mlflow>=2.0.0",
    
    # Async and concurrency
    "asyncio-mqtt>=0.11.0",
    
    # Docker integration
    "docker>=6.0.0",
    
    # Configuration management
    "pydantic>=2.0.0",
    "pyyaml>=6.0",
    
    # Resource monitoring
    "psutil>=5.9.0",
    
    # Plugin discovery
    "packaging>=21.0",
    "importlib-metadata>=4.0.0",
    
    # Utilities
    "click>=8.0.0",
    "rich>=12.0.0",
    "tqdm>=4.64.0",
]

[project.optional-dependencies]
# Machine Learning frameworks
sklearn = [
    "scikit-learn>=1.3.0",
]
xgboost = [
    "xgboost>=1.7.0",
]
lightgbm = [
    "lightgbm>=3.3.0",
]
catboost = [
    "catboost>=1.2.0",
]

# Deep Learning
pytorch = [
    "torch>=1.13.0",
    "torchvision>=0.14.0",
]
tensorflow = [
    "tensorflow>=2.10.0",
]

# Additional data formats
database = [
    "sqlalchemy>=1.4.0",
    "psycopg2-binary>=2.9.0",
    "pymongo>=4.0.0",
]
cloud = [
    "boto3>=1.26.0",
    "google-cloud-storage>=2.7.0",
    "azure-storage-blob>=12.14.0",
]

# Development dependencies
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
    "pytest-mock>=3.10.0",
    "pytest-benchmark>=4.0.0",
    "black>=22.0.0",
    "isort>=5.12.0",
    "flake8>=5.0.0",
    "mypy>=1.0.0",
    "pre-commit>=3.0.0",
]

# Documentation
docs = [
    "sphinx>=5.0.0",
    "sphinx-rtd-theme>=1.2.0",
    "sphinx-autodoc-typehints>=1.19.0",
    "myst-parser>=0.18.0",
]

# All optional dependencies
all = [
    "normaml[sklearn,xgboost,lightgbm,catboost,pytorch,tensorflow,database,cloud]"
]

[project.urls]
Homepage = "https://github.com/normaml/normaml"
Documentation = "https://normaml.readthedocs.io"
Repository = "https://github.com/normaml/normaml"
Issues = "https://github.com/normaml/normaml/issues"

[project.scripts]
normaml = "normaml.cli.main:main"

[project.entry-points."normaml.data_loaders"]
csv = "normaml.plugins.data_loaders.csv_loader:CSVLoader"
parquet = "normaml.plugins.data_loaders.parquet_loader:ParquetLoader"
database = "normaml.plugins.data_loaders.db_loader:DatabaseLoader"

[project.entry-points."normaml.feature_engineers"]
polars = "normaml.plugins.feature_engineers.polars_engineer:PolarsFeatureEngineer"
sklearn = "normaml.plugins.feature_engineers.sklearn_engineer:SklearnFeatureEngineer"

[project.entry-points."normaml.model_trainers"]
sklearn_classifier = "normaml.plugins.model_trainers.sklearn_trainer:SklearnClassifier"
sklearn_regressor = "normaml.plugins.model_trainers.sklearn_trainer:SklearnRegressor"
xgboost_classifier = "normaml.plugins.model_trainers.xgboost_trainer:XGBoostClassifier"
xgboost_regressor = "normaml.plugins.model_trainers.xgboost_trainer:XGBoostRegressor"

[project.entry-points."normaml.evaluators"]
classification = "normaml.plugins.evaluators.classification_evaluator:ClassificationEvaluator"
regression = "normaml.plugins.evaluators.regression_evaluator:RegressionEvaluator"

[project.entry-points."normaml.backends"]
sklearn = "normaml.plugins.backends.sklearn_backend:SklearnBackend"
xgboost = "normaml.plugins.backends.xgboost_backend:XGBoostBackend"

[tool.setuptools.packages.find]
where = ["."]
include = ["normaml*"]

[tool.setuptools.package-data]
normaml = ["config/*.yaml", "config/*.json"]

# Black configuration
[tool.black]
line-length = 88
target-version = ['py39']
include = '\.pyi?$'

# isort configuration
[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88

# mypy configuration
[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
ignore_missing_imports = true

# pytest configuration
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --cov=normaml --cov-report=html --cov-report=term-missing"
asyncio_mode = "auto"
```

## Docker Configuration

### Dockerfile.base

```dockerfile
FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python environment
WORKDIR /app
COPY requirements/base.txt requirements/docker.txt ./
RUN pip install --no-cache-dir -r base.txt -r docker.txt

# NormaML installation
COPY ../.. .
RUN pip install -e .

# Configure shared memory
RUN echo "tmpfs /dev/shm tmpfs defaults,size=4g 0 0" >> /etc/fstab

# User setup
RUN useradd -m -s /bin/bash normaml
USER normaml

CMD ["python", "-m", "normaml.cli"]
```

### Dockerfile.sklearn

```dockerfile
FROM normaml/base:latest

USER root
COPY requirements/optional.txt ./
RUN pip install --no-cache-dir scikit-learn xgboost lightgbm

USER normaml

# Plugin-specific environment
ENV NORMAML_PLUGIN_PATH="/app/normaml/plugins"
ENV PYTHONPATH="/app:${PYTHONPATH}"

CMD ["python", "-m", "normaml.plugins.model_trainers.sklearn_trainer"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  normaml-base:
    build:
      context: .
      dockerfile: docker/Dockerfile.base
    image: normaml/base:latest
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
      - /dev/shm:/dev/shm
    environment:
      - NORMAML_CONFIG_PATH=/app/config
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    depends_on:
      - mlflow
      - redis
    networks:
      - normaml-network

  mlflow:
    image: python:3.11-slim
    command: >
      sh -c "pip install mlflow[extras] psycopg2-binary &&
             mlflow server --host 0.0.0.0 --port 5000 
             --backend-store-uri postgresql://mlflow:password@postgres:5432/mlflow
             --default-artifact-root s3://mlflow-artifacts"
    ports:
      - "5000:5000"
    environment:
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
    depends_on:
      - postgres
    networks:
      - normaml-network

  postgres:
    image: postgres:13
    environment:
      - POSTGRES_DB=mlflow
      - POSTGRES_USER=mlflow
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - normaml-network

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - normaml-network

  worker-sklearn:
    build:
      context: .
      dockerfile: docker/Dockerfile.sklearn
    image: normaml/sklearn:latest
    deploy:
      replicas: 2
    volumes:
      - ./data:/app/data:ro
      - ./models:/app/models
      - /dev/shm:/dev/shm
    environment:
      - NORMAML_WORKER_TYPE=sklearn
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    depends_on:
      - normaml-base
    networks:
      - normaml-network

volumes:
  postgres_data:
  redis_data:

networks:
  normaml-network:
    driver: bridge
```

## CI/CD Configuration

### .github/workflows/ci.yml

```yaml
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11, 3.12]

    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Cache pip dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('requirements/*.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements/base.txt
        pip install -r requirements/dev.txt
        pip install -e .
    
    - name: Lint with flake8
      run: |
        flake8 normaml tests --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 normaml tests --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics
    
    - name: Type check with mypy
      run: mypy normaml
    
    - name: Test with pytest
      run: |
        pytest tests/unit --cov=normaml --cov-report=xml
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

  integration-test:
    runs-on: ubuntu-latest
    needs: test
    
    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_mlflow
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432

    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python 3.11
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements/base.txt
        pip install -r requirements/dev.txt
        pip install -e .[all]
    
    - name: Run integration tests
      run: pytest tests/integration
      env:
        POSTGRES_HOST: localhost
        POSTGRES_PORT: 5432
        POSTGRES_DB: test_mlflow
        POSTGRES_USER: postgres
        POSTGRES_PASSWORD: postgres

  performance-test:
    runs-on: ubuntu-latest
    needs: test
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python 3.11
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements/base.txt
        pip install -r requirements/dev.txt
        pip install -e .[sklearn]
    
    - name: Run performance tests
      run: pytest tests/performance --benchmark-only --benchmark-json=benchmark.json
    
    - name: Store benchmark result
      uses: benchmark-action/github-action-benchmark@v1
      with:
        tool: 'pytest'
        output-file-path: benchmark.json
        github-token: ${{ secrets.GITHUB_TOKEN }}
        auto-push: true
```

### .github/workflows/docker-build.yml

```yaml
name: Docker Build

on:
  push:
    branches: [ main ]
    tags: [ 'v*' ]
  pull_request:
    branches: [ main ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
    - name: Checkout repository
      uses: actions/checkout@v4

    - name: Log in to Container Registry
      if: github.event_name != 'pull_request'
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}

    - name: Build and push base image
      uses: docker/build-push-action@v5
      with:
        context: .
        file: ./docker/Dockerfile.base
        push: ${{ github.event_name != 'pull_request' }}
        tags: ${{ steps.meta.outputs.tags }}-base
        labels: ${{ steps.meta.outputs.labels }}

    - name: Build and push sklearn image
      uses: docker/build-push-action@v5
      with:
        context: .
        file: ./docker/Dockerfile.sklearn
        push: ${{ github.event_name != 'pull_request' }}
        tags: ${{ steps.meta.outputs.tags }}-sklearn
        labels: ${{ steps.meta.outputs.labels }}
```

## Требования к развертыванию

### requirements/base.txt

```text
# Core data processing
polars>=0.19.0,<1.0.0
pyarrow>=10.0.0,<15.0.0
numpy>=1.21.0,<2.0.0

# MLflow integration
mlflow>=2.0.0,<3.0.0

# Async and concurrency
asyncio-mqtt>=0.11.0

# Docker integration
docker>=6.0.0,<8.0.0

# Configuration management
pydantic>=2.0.0,<3.0.0
pyyaml>=6.0,<7.0

# Resource monitoring
psutil>=5.9.0

# Plugin discovery
packaging>=21.0
importlib-metadata>=4.0.0

# Utilities
click>=8.0.0,<9.0.0
rich>=12.0.0,<14.0.0
tqdm>=4.64.0,<5.0.0
```

### requirements/dev.txt

```text
# Testing
pytest>=7.0.0,<8.0.0
pytest-asyncio>=0.21.0
pytest-cov>=4.0.0
pytest-mock>=3.10.0
pytest-benchmark>=4.0.0
pytest-xdist>=3.0.0

# Code quality
black>=22.0.0,<24.0.0
isort>=5.12.0,<6.0.0
flake8>=5.0.0,<7.0.0
mypy>=1.0.0,<2.0.0
pre-commit>=3.0.0,<4.0.0

# Documentation
sphinx>=5.0.0,<8.0.0
sphinx-rtd-theme>=1.2.0
sphinx-autodoc-typehints>=1.19.0
myst-parser>=0.18.0

# Jupyter support
jupyter>=1.0.0
ipykernel>=6.0.0
nbconvert>=7.0.0
```

### requirements/optional.txt

```text
# Machine Learning frameworks
scikit-learn>=1.3.0,<2.0.0
xgboost>=1.7.0,<3.0.0
lightgbm>=3.3.0,<5.0.0
catboost>=1.2.0,<2.0.0

# Deep Learning
torch>=1.13.0,<3.0.0
torchvision>=0.14.0,<1.0.0
tensorflow>=2.10.0,<3.0.0

# Database support
sqlalchemy>=1.4.0,<3.0.0
psycopg2-binary>=2.9.0
pymongo>=4.0.0,<5.0.0

# Cloud storage
boto3>=1.26.0,<2.0.0
google-cloud-storage>=2.7.0,<3.0.0
azure-storage-blob>=12.14.0,<13.0.0
```

## Управление версиями

### normaml/__version__.py

```python
"""Version information for NormaML."""

__version__ = "1.0.0"
__version_info__ = tuple(int(i) for i in __version__.split('.'))

# Compatibility information
MIN_PYTHON_VERSION = (3, 9)
MAX_PYTHON_VERSION = (3, 12)

# Dependencies versions
MIN_POLARS_VERSION = "0.19.0"
MIN_PYARROW_VERSION = "10.0.0"
MIN_MLFLOW_VERSION = "2.0.0"

def check_python_version():
    """Check if current Python version is supported."""
    import sys
    current_version = sys.version_info[:2]
    
    if current_version < MIN_PYTHON_VERSION:
        raise RuntimeError(
            f"NormaML requires Python {MIN_PYTHON_VERSION[0]}.{MIN_PYTHON_VERSION[1]} "
            f"or later, but you are using Python {current_version[0]}.{current_version[1]}"
        )
    
    if current_version > MAX_PYTHON_VERSION:
        import warnings
        warnings.warn(
            f"NormaML has not been tested with Python {current_version[0]}.{current_version[1]}. "
            f"Maximum tested version is {MAX_PYTHON_VERSION[0]}.{MAX_PYTHON_VERSION[1]}",
            UserWarning
        )
```

## Scripts для развертывания

### scripts/install_dev.sh

```bash
#!/bin/bash
set -e

echo "Installing NormaML in development mode..."

# Check Python version
python -c "import sys; assert sys.version_info >= (3, 9), 'Python 3.9+ required'"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install development dependencies
pip install -r requirements/base.txt
pip install -r requirements/dev.txt

# Install in editable mode
pip install -e .

# Install pre-commit hooks
pre-commit install

echo "Development installation complete!"
echo "Activate the environment with: source venv/bin/activate"
```

### scripts/build_docker.sh

```bash
#!/bin/bash
set -e

# Build arguments
VERSION=${1:-latest}
REGISTRY=${2:-normaml}

echo "Building Docker images with version: $VERSION"

# Build base image
docker build -f docker/Dockerfile.base -t $REGISTRY/base:$VERSION .

# Build plugin-specific images
docker build -f docker/Dockerfile.sklearn -t $REGISTRY/sklearn:$VERSION .

# Tag latest if version provided
if [ "$VERSION" != "latest" ]; then
    docker tag $REGISTRY/base:$VERSION $REGISTRY/base:latest
    docker tag $REGISTRY/sklearn:$VERSION $REGISTRY/sklearn:latest
fi

echo "Docker images built successfully!"
echo "Images:"
echo "  $REGISTRY/base:$VERSION"
echo "  $REGISTRY/sklearn:$VERSION"
```

## Plugin Template Generator

### scripts/create_plugin.py

```python
#!/usr/bin/env python3
"""Script to generate plugin template."""

import os
import sys
from pathlib import Path
import click

PLUGIN_TEMPLATE = '''
"""
{plugin_name} plugin for NormaML.
"""

from typing import Dict, Any, List
from normaml.core.interfaces import {plugin_type}Plugin
from normaml.plugins.base import BasePlugin


class {class_name}({plugin_type}Plugin, BasePlugin):
    """
    {description}
    """
    
    name = "{plugin_name}"
    version = "1.0.0"
    capabilities = {capabilities}
    dependencies = {dependencies}
    
    def __init__(self):
        super().__init__()
        # Initialize your plugin here
        pass
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin with configuration."""
        # Implementation here
        pass
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate plugin configuration."""
        # Implementation here
        return True
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return plugin capabilities."""
        return {{
            "supports_streaming": False,
            "supports_parallel": True,
            "memory_efficient": True
        }}
    
    # Add your plugin-specific methods here
'''

@click.command()
@click.option('--name', required=True, help='Plugin name')
@click.option('--type', 'plugin_type', required=True, 
              type=click.Choice(['DataLoader', 'FeatureEngineer', 'ModelTrainer', 'Evaluator']))
@click.option('--description', default='Custom plugin', help='Plugin description')
def create_plugin(name, plugin_type, description):
    """Create a new plugin template."""
    
    # Generate class name
    class_name = ''.join(word.capitalize() for word in name.split('_'))
    
    # Default capabilities and dependencies
    capabilities = ['custom']
    dependencies = ['polars>=0.19.0']
    
    # Generate plugin code
    plugin_code = PLUGIN_TEMPLATE.format(
        plugin_name=name,
        plugin_type=plugin_type,
        class_name=class_name,
        description=description,
        capabilities=capabilities,
        dependencies=dependencies
    )
    
    # Determine output directory
    type_dir = f"{plugin_type.lower()}s"
    plugin_dir = Path(f"normaml/plugins/{type_dir}")
    plugin_dir.mkdir(parents=True, exist_ok=True)
    
    # Write plugin file
    plugin_file = plugin_dir / f"{name}.py"
    
    if plugin_file.exists():
        click.echo(f"Plugin file {plugin_file} already exists!")
        return
    
    with open(plugin_file, 'w') as f:
        f.write(plugin_code)
    
    # Update __init__.py
    init_file = plugin_dir / "__init__.py"
    with open(init_file, 'a') as f:
        f.write(f"\nfrom .{name} import {class_name}")
    
    click.echo(f"Plugin {name} created successfully!")
    click.echo(f"File: {plugin_file}")
    click.echo(f"Class: {class_name}")

if __name__ == '__main__':
    create_plugin()
```

Эта структура пакета обеспечивает:

1. **Модульную организацию** - четкое разделение компонентов
2. **Plugin систему** - entry points для автоматического обнаружения
3. **Гибкие зависимости** - опциональные пакеты для разных use cases
4. **Docker поддержку** - контейнеризация для изоляции и развертывания
5. **CI/CD** - автоматизированное тестирование и сборка
6. **Документацию** - sphinx для API reference
7. **Качество кода** - линтеры, форматеры, pre-commit hooks
8. **Производительность** - benchmarking и профилирование
9. **Расширяемость** - templates и scripts для создания новых plugins
10. **Production-ready** - конфигурация для промышленного развертывания