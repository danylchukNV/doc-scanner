# Інтелектуальний сканер документів

Індивідуальний проєкт з дисципліни «Комп'ютерний зір», варіант 13.
Автор: Назарій Данильчук.

## Задача

Програма знаходить аркуш документа на фотографії, виправляє перспективу,
обрізає зайві ділянки, підсилює контрастність і зберігає результат.

## Архітектура

```
Фото ─► Попередня обробка ─► Пошук країв ─► Пошук 4 кутів ─► Корекція перспективи ─► Підсилення ─► Збереження
        (resize, gray, blur)   (Canny)       (контури,         (гомографія)            (adaptive
                                              approxPolyDP)                              threshold)
```

| Модуль | Відповідальність |
|--------|------------------|
| `src/scanner.py` | алгоритм: обробка, пошук контуру, гомографія, підсилення |
| `src/main.py` | CLI, режим debug, збереження |
| `src/check_env.py` | перевірка залежностей і тестових даних |
| `src/make_synthetic.py` | генерація синтетичних тестів з відомими кутами |
| `src/evaluate.py` | вимірювання похибки кутів на синтетичних тестах |

Детальний аналіз і обґрунтування вибору методів — [docs/01_analysis.md](docs/01_analysis.md).

## Використані методи

_Заповнюється на етапі 4._

## Встановлення та запуск

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python src/check_env.py
```

### Запуск сканера

```bash
# одне фото -> output/<назва>_scan.png
python src/main.py samples/01_a4_dark_top.jpg

# кольоровий режим (CLAHE) і показ усіх етапів у вікні
python src/main.py samples/02_a4_dark_angle.jpg --mode color --show

# ручний вибір 4 кутів мишкою (коли автоматичний пошук не впорався)
python src/main.py samples/09_white_on_white.jpg --manual

# пакетна обробка папки + зображення етапів у output/debug + output/report.csv
python src/main.py samples/ --debug

# зміна параметрів алгоритму
python src/main.py samples/07_shadow.jpg --canny 30 90 --block 31 --C 15
```

Усі параметри: `python src/main.py -h`.

### Синтетичні тести

```bash
python src/make_synthetic.py   # створює samples/synthetic/ + ground_truth.json
python src/evaluate.py         # похибка знайдених кутів у пікселях
```

## Тестування

| № | Файл | Умови | Результат | Примітка |
|---|------|-------|-----------|----------|
| 1 | 01_a4_dark_top.jpg | A4, темний стіл, зверху | | |
| 2 | 02_a4_dark_angle.jpg | під кутом | | |
| 3 | 03_a4_rotated.jpg | поворот 45° | | |
| 4 | 04_receipt.jpg | чек | | |
| 5 | 05_card.jpg | візитка | | |
| 6 | 06_book_page.jpg | кольоровий | | |
| 7 | 07_shadow.jpg | тінь | | |
| 8 | 08_low_light.jpg | слабке світло | | |
| 9 | 09_white_on_white.jpg | білий на білому | | |
| 10 | 10_folded_corner.jpg | загнутий кут | | |
| 11 | 11_cut_off.jpg | кут поза кадром | | |
| 12 | 12_cluttered.jpg | захаращений фон | | |

## Відомі обмеження

_Заповнюється за результатами тестування._

## Структура репозиторію

```
doc-scanner/
├── src/            код
├── samples/        тестові фото
├── output/         результати (не комітяться)
├── docs/           аналіз, чек-лист фото
├── requirements.txt
└── README.md
```
