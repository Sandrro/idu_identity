# Установка / запуск

## Запуск в облаке
- **Google Colab** (рекомендуется, если Binder недоступен):
  - Откройте нужный ноутбук по ссылке:
    - [Collect texts — Open in Colab](https://colab.research.google.com/github/Sandrro/idu_identity/blob/main/data/collect_texts.ipynb)
    - [Web parser — Open in Colab](https://colab.research.google.com/github/Sandrro/idu_identity/blob/main/data/web_parser.ipynb)
  - Сразу после открытия выполните «Подключить к среде выполнения» → «Сохранить копию на Диске», чтобы изменения сохранялись.
- **MyBinder** — остаётся как резерв, но может дольше стартовать или прерывать сборку: https://mybinder.org/v2/gh/Sandrro/idu_identity/main?urlpath=lab

## Локальная установка
1. Создайте виртуальное окружение Python 3.10+.
2. Установите зависимости для запуска ноутбуков и сборки книги:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-build.txt
   ```
3. Запустите Jupyter Lab или Notebook в каталоге проекта:
   ```bash
   jupyter lab
   ```
