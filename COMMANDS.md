# Шпаргалка команд

Усі команди виконуються з кореня проєкту (`E:\github\doc-scanner`).
Приклади для Windows (PowerShell / cmd); на Linux/macOS відрізняється лише активація venv.

## 1. Середовище (один раз)

```bash
python --version                      # має бути 3.10+ (рекомендовано 3.12 / 3.13)
python -m venv .venv                  # створити віртуальне середовище
.venv\Scripts\activate                # активувати (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt       # встановити залежності
python src/check_env.py               # перевірити версії та фото в samples/
```

Якщо PowerShell не дає активувати venv:
```bash
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## 2. Git

```bash
git pull                              # підтягнути зміни з GitHub (робити ПЕРЕД роботою)
git status                            # що змінено
git add -A                            # додати всі зміни
git commit -m "Опис змін"             # зробити коміт
git push                              # відправити на GitHub
```

> Датасети (`datasets/`, `docs/datasets/`) виключені в `.gitignore` — їх не комітимо.

## 3. Веб-демонстрація

```bash
streamlit run app.py          # відкриється в браузері: http://localhost:8501
```

Вкладки: **Сканування** (фото / завантаження / камера, усі етапи, параметри в бічній панелі,
ручне коригування кутів, завантаження скану), **Порівняння** simple vs multi,
**Датасет** (випадкові приклади з еталоном і IoU). Зупинити — `Ctrl+C` у терміналі.

## 3б. Сканер з командного рядка

```bash
# одне фото -> output/<назва>_scan.png
python src/main.py samples/examples/02_card_angle.jpg

# кольоровий режим (CLAHE) замість ч/б
python src/main.py samples/examples/04_color_flyer.jpg --mode color

# без підсилення, лише вирівнювання
python src/main.py samples/examples/04_color_flyer.jpg --mode none

# показати всі етапи у вікні
python src/main.py samples/examples/04_color_flyer.jpg --show

# ручний вибір 4 кутів мишкою (Enter — ок, R — скинути, Esc — скасувати)
python src/main.py samples/examples/10_white_on_white.jpg --manual

# пакетна обробка папки + картинки етапів (output/debug) + звіт output/report.csv
python src/main.py samples/ --debug

# зберегти результати в іншу папку
python src/main.py samples/ -o output/run2

# довідка по всіх параметрах
python src/main.py -h
```

### Параметри алгоритму (для експериментів і захисту)

| Прапорець | За замовч. | Що змінює |
|-----------|-----------|-----------|
| `--strategy` | multi | `multi` — кілька карт країв і вибір кращого кандидата, `simple` — одна карта (Canny на grayscale) |
| `--canny LOW HIGH` | авто | ручні пороги Canny (стратегія simple) |
| `--sigma` | 0.33 | ширина автопорогу Canny |
| `--blur` | 5 | ядро розмиття Гауса |
| `--morph` | 5 | ядро морфологічного закриття (0 — вимкнути) |
| `--eps` | 0.02 | точність апроксимації контуру (частка периметра) |
| `--min-area` | 0.05 | мінімальна частка площі кадру для документа |
| `--block` | 21 | розмір вікна adaptiveThreshold |
| `--C` | 10 | константа adaptiveThreshold |

```bash
python src/main.py samples/examples/05_shadow_synthetic.jpg --canny 30 90 --show
python src/main.py samples/examples/05_shadow_synthetic.jpg --block 31 --C 15
python src/main.py samples/examples/01_low_contrast_table.jpg --morph 0 --show
```

## 4. Синтетичні тести

```bash
python src/make_synthetic.py          # згенерувати samples/synthetic/ + ground_truth.json
python src/evaluate.py                # IoU / похибка кутів на синтетиці
python src/main.py samples/synthetic --debug
```

## 5. Датасети

### DocCornerDataset (parquet)

```bash
# 300 випадкових фото з документами -> datasets/doccorner_sample/
python src/extract_parquet.py docs/datasets/DocCornerDataset/test --limit 300

# + фото без документа (перевірка хибних спрацьовувань)
python src/extract_parquet.py docs/datasets/DocCornerDataset/test --limit 300 --negatives -o datasets/doccorner_neg

# оцінка точності
python src/evaluate.py datasets/doccorner_sample

# обробити вибірку з картинками етапів
python src/main.py datasets/doccorner_sample --debug -o output/doccorner
```

### Оцінка точності (evaluate.py)

Формат датасету визначається автоматично: `ground_truth.json` → gt, `*.gt.xml` → SmartDoc, `*.csv` → Mendeley.

```bash
# Mendeley ("All My Dataset"): 200 випадкових фото
python src/evaluate.py "docs/datasets/All My Dataset" --limit 200

# SmartDoc: кожен 50-й кадр усіх відео, випадкові 300
python src/evaluate.py docs/datasets/SmartDoc/testDataset --frame-step 50 --limit 300

# показати 15 найгірших і зберегти їх картинки в output/eval_worst
#   (зелений — еталон, червоний — знайдене сканером)
python src/evaluate.py "docs/datasets/All My Dataset" --limit 200 --worst 15 --save-worst

# порівняти стратегії: simple (одна карта країв) vs multi (за замовч.)
python src/evaluate.py docs/datasets/SmartDoc/testDataset --limit 300 --strategy simple
python src/evaluate.py docs/datasets/SmartDoc/testDataset --limit 300 --strategy multi

# порівняти параметри алгоритму
python src/evaluate.py docs/datasets/SmartDoc/testDataset --limit 300 --sigma 0.5
python src/evaluate.py docs/datasets/SmartDoc/testDataset --limit 300 --canny 20 60 --morph 7

# інший поріг успіху (за замовч. IoU ≥ 0.9)
python src/evaluate.py datasets/doccorner_sample --iou 0.8
```

Звіт по кожному зображенню: `output/eval_<назва_папки>.csv` (IoU, похибка кутів, метод, час).

### Поділ tune / test і підбір параметрів

```bash
# оцінка лише на test (чесні цифри) або на tune
python src/evaluate.py "docs/datasets/All My Dataset" --split test --limit 300
python src/evaluate.py docs/datasets/SmartDoc/testDataset --split test --frame-step 50

# змінити будь-який параметр алгоритму (і в evaluate.py, і в main.py)
python src/evaluate.py docs/datasets/SmartDoc/testDataset --split test --set use_lines=false
python src/main.py samples/examples/04_color_flyer.jpg --show --set refine=true min_angle=40

# підбір параметрів на tune (результат -> docs/tuning_results.json)
python src/tune.py "docs/datasets/All My Dataset" docs/datasets/SmartDoc/testDataset --frame-step 50

# те саме частинами (наприклад, на слабкому ПК): підготовка шматками, потім перебір
python src/tune.py "docs/datasets/All My Dataset" docs/datasets/SmartDoc/testDataset --parts 8 --part 0
python src/tune.py "docs/datasets/All My Dataset" docs/datasets/SmartDoc/testDataset --parts 8 --search

# довга обробка частинами (кожна частина — окремий CSV)
python src/evaluate.py "docs/datasets/All My Dataset" --split test --limit 300 --parts 4 --part 0 -o output/m_0.csv
```

Основні параметри стратегії multi (`--set ключ=значення`):

| Параметр | За замовч. | Зміст |
|----------|-----------|-------|
| `use_lines` | true | кандидати з прямих (LSD) |
| `line_weight` | 0.5 | множник оцінки кандидатів з прямих |
| `line_side_min` | 0.7 | мін. опора на краї для кожної сторони 4-кутника з прямих |
| `line_inside_thr` | 0.9 | прямі всередині кращого контура -> обрати контур |
| `filter_quads` | true | фільтр правдоподібності |
| `min_angle` | 50 | мін. внутрішній кут, градуси |
| `score_area_pow` | 0.5 | вага площі в оцінці |
| `refine` | false | уточнення сторін через fitLine |
