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

## 3. Сканер

```bash
# одне фото -> output/<назва>_scan.png
python src/main.py samples/01_a4_dark_top.jpg

# кольоровий режим (CLAHE) замість ч/б
python src/main.py samples/02_a4_dark_angle.jpg --mode color

# без підсилення, лише вирівнювання
python src/main.py samples/02_a4_dark_angle.jpg --mode none

# показати всі етапи у вікні
python src/main.py samples/02_a4_dark_angle.jpg --show

# ручний вибір 4 кутів мишкою (Enter — ок, R — скинути, Esc — скасувати)
python src/main.py samples/09_white_on_white.jpg --manual

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
| `--canny LOW HIGH` | авто | ручні пороги Canny |
| `--sigma` | 0.33 | ширина автопорогу Canny |
| `--blur` | 5 | ядро розмиття Гауса |
| `--morph` | 5 | ядро морфологічного закриття (0 — вимкнути) |
| `--eps` | 0.02 | точність апроксимації контуру (частка периметра) |
| `--min-area` | 0.10 | мінімальна частка площі кадру для документа |
| `--block` | 21 | розмір вікна adaptiveThreshold |
| `--C` | 10 | константа adaptiveThreshold |

```bash
python src/main.py samples/07_shadow.jpg --canny 30 90 --show
python src/main.py samples/07_shadow.jpg --block 31 --C 15
python src/main.py samples/08_low_light.jpg --morph 0 --show
```

## 4. Синтетичні тести

```bash
python src/make_synthetic.py          # згенерувати samples/synthetic/ + ground_truth.json
python src/evaluate.py                # похибка кутів на синтетиці
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
