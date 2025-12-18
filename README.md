# Городская идентичность — Jupyter Book

Проект собирает учебные материалы и ноутбуки по анализу городской идентичности. Репозиторий настроен на сборку **Jupyter Book 2.x** (через `mystmd`) с интерактивными ячейками (Thebe) и кнопками запуска на MyBinder.

## Облака для запуска
- **Google Colab** — быстрый способ открыть ноутбуки без сборки окружения. Ссылки:
  - [Collect texts — Open in Colab](https://colab.research.google.com/github/Sandrro/idu_identity/blob/main/data/collect_texts.ipynb)
  - [Web parser — Open in Colab](https://colab.research.google.com/github/Sandrro/idu_identity/blob/main/data/web_parser.ipynb)
- **MyBinder** — остаётся как запасной вариант, но может временно не запускаться из-за нестабильности сборки.

## Структура
- `index.md` — главная страница курса.
- `setup.md` — подготовка окружения и вводные сведения.
- `data/collect_texts.ipynb` — сбор и первичная обработка текстов.
- `data/web_parser.ipynb` — пример парсинга данных с сайтов.
- `myst.yml` — конфигурация MyST / Jupyter Book 2.x и оглавление книги.

## Установка окружения
1. Создайте виртуальное окружение Python 3.10+.
2. Установите зависимости для запуска ноутбуков и сборки книги:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-build.txt
   ```
