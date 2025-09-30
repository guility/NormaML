# NormaML

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![uv](https://img.shields.io/badge/Package%20Manager-uv-orange)](https://github.com/astral-sh/uv)

**NormaML** - это модульная библиотека машинного обучения с фокусом на zero-copy обработку данных и эффективное использование ресурсов.

## ✨ Ключевые возможности

- 🔌 **Модульная архитектура** - Система плагинов для легкого расширения функциональности
- ⚡ **Zero-Copy обработка данных** - Минимизация копирования данных в памяти
- 🚀 **Параллельное выполнение** - Поддержка многопроцессорности и Docker-контейнеров
- 📊 **Интеграция с MLflow** - Встроенное отслеживание экспериментов
- 🔄 **Асинхронная обработка** - Поддержка async/await для неблокирующих операций
- 🛠 **Готовые плагины** - Поддержка scikit-learn, XGBoost, LightGBM, CatBoost
- 📈 **Мониторинг ресурсов** - Отслеживание использования памяти и CPU
- 🎯 **Production-ready** - Готовность к промышленному развертыванию

## 🚀 Быстрый старт

### Установка

Убедитесь, что у вас установлен Python 3.9+ и [uv](https://github.com/astral-sh/uv):

```bash
# Установка основной версии
uv add normaml

# Установка с поддержкой scikit-learn
uv add normaml[sklearn]

# Установка с поддержкой XGBoost
uv add normaml[xgboost]

# Установка всех опциональных зависимостей
uv add normaml[all]
```

### Базовое использование

```python
import normaml
from normaml.api import PipelineBuilder

# Создание пайплайна
pipeline = (PipelineBuilder()
    .load_data("csv", path="data.csv")
    .engineer_features("polars", operations=["normalize", "encode"])
    .train_model("sklearn_classifier", algorithm="RandomForest")
    .evaluate("classification")
    .build())

# Запуск пайплайна
results = pipeline.run()
```

## 📦 Архитектура

NormaML построен на модульной архитектуре с основными компонентами:

### Основные модули

- **`normaml.core`** - Базовая функциональность и интерфейсы
- **`normaml.plugins`** - Система плагинов и готовые реализации
- **`normaml.data`** - Zero-copy управление данными
- **`normaml.execution`** - Движки выполнения и планировщики
- **`normaml.tracking`** - Интеграция с MLflow
- **`normaml.api`** - Публичное API

### Плагины

#### Загрузчики данных
- `csv` - Загрузка CSV файлов
- `parquet` - Загрузка Parquet файлов  
- `database` - Загрузка из баз данных

#### Инженеры признаков
- `polars` - Обработка с использованием Polars
- `sklearn` - Обработка с использованием scikit-learn

#### Тренеры моделей
- `sklearn_classifier` / `sklearn_regressor` - Модели scikit-learn
- `xgboost_classifier` / `xgboost_regressor` - Модели XGBoost

#### Оценщики
- `classification` - Метрики для классификации
- `regression` - Метрики для регрессии

## 🛠 Разработка

### Настройка окружения разработки

```bash
# Клонирование репозитория
git clone https://github.com/normaml/normaml.git
cd normaml

# Установка зависимостей для разработки
uv sync --dev

# Установка pre-commit хуков
pre-commit install
```

### Запуск тестов

```bash
# Юнит-тесты
pytest tests/unit

# Интеграционные тесты
pytest tests/integration

# Тесты производительности
pytest tests/performance --benchmark-only
```

### Линтинг и форматирование

```bash
# Форматирование кода
black normaml tests
isort normaml tests

# Проверка типов
mypy normaml

# Линтинг
flake8 normaml tests
```

## 🐳 Docker

### Сборка образов

```bash
# Сборка базового образа
docker build -f docker/Dockerfile.base -t normaml/base .

# Сборка образа с scikit-learn
docker build -f docker/Dockerfile.sklearn -t normaml/sklearn .
```

### Использование docker-compose

```bash
# Запуск всей инфраструктуры
docker-compose up -d

# Запуск только MLflow
docker-compose up mlflow
```

## 📚 Документация

- **Пользовательское руководство**: [docs/user_guide/](docs/user_guide/)
- **API Reference**: [docs/api_reference/](docs/api_reference/)
- **Примеры**: [examples/](examples/)
- **Jupyter ноутбуки**: [examples/notebooks/](examples/notebooks/)

## 🤝 Участие в разработке

Мы приветствуем вклад в развитие проекта! Пожалуйста:

1. Форкните репозиторий
2. Создайте ветку для вашей функции (`git checkout -b feature/amazing-feature`)
3. Сделайте коммит изменений (`git commit -m 'Add amazing feature'`)
4. Отправьте изменения в ветку (`git push origin feature/amazing-feature`)
5. Откройте Pull Request

### Создание собственных плагинов

```python
from normaml.plugins.base import BasePlugin
from normaml.core.interfaces import DataLoaderPlugin

class CustomDataLoader(DataLoaderPlugin, BasePlugin):
    name = "custom_loader"
    version = "1.0.0"
    
    def load_data(self, **kwargs):
        # Ваша реализация
        pass
```

## 📄 Лицензия

Проект распространяется под лицензией GPL-3.0. См. файл [LICENSE](LICENSE) для подробностей.

## 🙏 Благодарности

- [Polars](https://www.pola.rs/) - за высокопроизводительную обработку данных
- [MLflow](https://mlflow.org/) - за отслеживание экспериментов
- [scikit-learn](https://scikit-learn.org/) - за алгоритмы машинного обучения

## 🔗 Ссылки

- **Homepage**: https://github.com/normaml/normaml
- **Documentation**: https://normaml.readthedocs.io
- **Issues**: https://github.com/normaml/normaml/issues
- **Discussions**: https://github.com/normaml/normaml/discussions

---

**NormaML** - делаем машинное обучение эффективным и масштабируемым! 🚀