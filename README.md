# Інтелектуальний сканер документів

Індивідуальний проєкт з дисципліни «Комп'ютерний зір», варіант 13.
Автор: Назарій Данильчук.

## Задача

Програма знаходить аркуш документа на фотографії, виправляє перспективу,
обрізає зайві ділянки, підсилює контрастність і зберігає результат.

## Архітектура

```
                           ┌─ 8 карт країв (Canny: gray, B/G/R, низькі пороги; Otsu) ─► контури ─► approxPolyDP ─┐
Фото ─► зменшення, gray, ──┤                                                                                    ├─► фільтр ─► оцінка ─► кращий ─► гомографія ─► підсилення ─► скан
        розмиття Гауса     └─ відрізки LSD ─► 2 групи напрямків (k-means) ─► перетини пар прямих ───────────────┘   (кути,    (контраст ×  4-кутник   (warpPerspective) (adaptive
                                                                                                                      площа)    опора × площа)                              threshold / CLAHE)
```

| Модуль | Відповідальність |
|--------|------------------|
| `app.py` | веб-демонстрація (Streamlit) |
| `src/scanner.py` | алгоритм: обробка, пошук контуру, гомографія, підсилення |
| `src/main.py` | CLI, режим debug, збереження |
| `src/check_env.py` | перевірка залежностей і тестових даних |
| `src/make_synthetic.py` | генерація синтетичних тестів з відомими кутами |
| `src/evaluate.py` | оцінка точності: IoU, похибка кутів, час; звіт CSV і найгірші випадки |
| `src/datasets.py` | завантажувачі розмітки: ground_truth.json, Mendeley (csv), SmartDoc (avi + gt.xml); поділ tune/test |
| `src/tune.py` | підбір параметрів перебором сітки на частині tune |
| `src/extract_parquet.py` | витягування jpg + розмітки з parquet (DocCornerDataset) |

Детальний аналіз і обґрунтування вибору методів — [docs/01_analysis.md](docs/01_analysis.md).

## Методика

Рішення не повторює одну конкретну публікацію: воно поєднує класичний конвеєр сканера
документів, схему «генерація → перевірка гіпотез» і стандартну експериментальну методику.

### 1. Базовий підхід — класичний конвеєр сканера документів

Знайти межі → знайти чотирикутник → вирівняти перспективу → підсилити контраст
(Canny → `findContours` → `approxPolyDP` → перетворення за 4 точками → бінаризація) [1].
Це стратегія `simple`; її залишено як базову лінію для порівняння.

### 2. Розширення — генерація та перевірка гіпотез (generate-and-verify)

Базовий підхід шукає один контур на одній карті країв і часто помиляється
(на тестових даних — 13–20 % успіху). Тому пошук документа перебудовано за схемою,
близькою до описаної Dropbox для мобільного сканування [2]:

1. **Генерація гіпотез** — кандидати-чотирикутники з кількох незалежних джерел:
   - 8 карт країв: Canny на яскравості (автопоріг і низький поріг), на кожному каналі
     B, G, R окремо (межа «білий аркуш / кольоровий стіл» помітна за кольором, але не за
     яскравістю), межі областей після бінаризації Otsu яскравості й насиченості;
   - прямі лінії: відрізки LSD → дві групи за напрямком (k-means на подвоєному куті) →
     чотирикутники з перетинів двох пар прямих. Знаходить документ, навіть коли частину
     сторони закриває рука.
2. **Перевірка гіпотез:**
   - геометричний фільтр: внутрішні кути 50–130°, площа 5–95 % кадру, вершини не далеко за кадром;
   - оцінка = **контраст з фоном** (різниця середнього кольору Lab всередині й у смузі навколо)
     × **опора на краї** (частка периметра на знайдених краях) × **площа**^0.5;
   - правило: якщо переміг чотирикутник з прямих, але він лежить усередині найкращого
     контура (межі колонок, рамки на сторінці), обирається контур.
3. **Корекція перспективи** обраного чотирикутника гомографією і **підсилення** результату.

У Dropbox краї знаходить навчена модель; тут — лише класичні методи, тому кожен крок
можна пояснити й перевірити.

### 3. Методи комп'ютерного зору

| Етап | Метод | Джерело |
|------|-------|---------|
| Пошук країв | детектор Canny | Canny, 1986 [3] |
| Контури | трасування меж (`cv2.findContours`) | Suzuki & Abe, 1985 [4] |
| Спрощення контуру | алгоритм Дугласа–Пекера (`cv2.approxPolyDP`) | Douglas & Peucker, 1973 [5] |
| Пошук прямих | Line Segment Detector (`cv2.createLineSegmentDetector`) | Grompone von Gioi et al., 2010 [6] |
| Бінаризація областей | метод Otsu | Otsu, 1979 [7] |
| Корекція перспективи | гомографія за 4 парами точок | Hartley & Zisserman [8] |
| Підсилення | адаптивна бінаризація (Гаусове вікно); CLAHE | Zuiderveld, 1994 [9] |

### 4. Методика розробки та оцінки

- **Протокол SmartDoc 2015 (ICDAR)** [10]: метрика — IoU (індекс Жаккара) між знайденим і
  еталонним чотирикутником; успіх — IoU ≥ 0.9. Відмінність від оригіналу: IoU рахується
  в координатах кадру, а не в системі координат документа.
- **Датасети:** SmartDoc 2015 Challenge 1 (150 відео, 5 фонів), Mendeley «Document Corner
  Localization» (~1100 фото з телефона), синтетичні зображення з точно відомими кутами.
- **Розробка, керована аналізом помилок:** прогін → перегляд найгірших випадків →
  гіпотеза про причину → зміна → повторний прогін. Так з'явились кольорові канали,
  кандидати з прямих і правило «прямі всередині контура».
- **Поділ tune / test:** параметри підбирались лише на ~30 % даних (tune), підсумкові
  цифри — на решті (test). Поділ детермінований: SmartDoc — за документом, Mendeley — за хешем назви.
- **Абляційний аналіз:** внесок кожної зміни виміряно окремо; зміни без приросту
  (уточнення сторін `fitLine`, рамка кадру на картах країв) не увійшли до фінальної версії.
- **Підбір параметрів** перебором сітки (1296 комбінацій, `src/tune.py`). Найкраща на tune
  комбінація виявилась перенавченою (на test −14 % на SmartDoc), тому обрано стійкіший варіант.

Деталі й усі цифри — [docs/03_testing.md](docs/03_testing.md).

### Джерела

1. A. Rosebrock. *How to Build a Kick-Ass Mobile Document Scanner in Just 5 Minutes.* PyImageSearch, 2014. https://pyimagesearch.com/2014/09/01/build-kick-ass-mobile-document-scanner-just-5-minutes/
2. Dropbox Tech. *Fast and Accurate Document Detection for Scanning.* 2016. https://dropbox.tech/machine-learning/fast-and-accurate-document-detection-for-scanning
3. J. Canny. *A Computational Approach to Edge Detection.* IEEE TPAMI, 8(6), 1986.
4. S. Suzuki, K. Abe. *Topological Structural Analysis of Digitized Binary Images by Border Following.* CVGIP, 30(1), 1985.
5. D. Douglas, T. Peucker. *Algorithms for the Reduction of the Number of Points Required to Represent a Digitized Line or its Caricature.* Cartographica, 10(2), 1973.
6. R. Grompone von Gioi, J. Jakubowicz, J.-M. Morel, G. Randall. *LSD: A Fast Line Segment Detector with a False Detection Control.* IEEE TPAMI, 32(4), 2010.
7. N. Otsu. *A Threshold Selection Method from Gray-Level Histograms.* IEEE Trans. SMC, 9(1), 1979.
8. R. Hartley, A. Zisserman. *Multiple View Geometry in Computer Vision.* Cambridge University Press, 2004.
9. K. Zuiderveld. *Contrast Limited Adaptive Histogram Equalization.* Graphics Gems IV, 1994.
10. J.-C. Burie et al. *ICDAR 2015 Competition on Smartphone Document Capture and OCR (SmartDoc).* ICDAR, 2015. Дані: https://doi.org/10.5281/zenodo.1230217

## Використані методи

_Заповнюється на етапі 4._

## Встановлення та запуск

> Повний список команд — [COMMANDS.md](COMMANDS.md).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python src/check_env.py
```

### Веб-демонстрація

```bash
streamlit run app.py
```

У браузері: завантаження фото або знімок з камери, усі етапи обробки, зміна параметрів
повзунками, порівняння стратегій simple / multi, приклади з датасетів з IoU.

### Запуск сканера з командного рядка

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

### Зовнішній датасет (DocCornerDataset, parquet)

```bash
pip install pyarrow
# 300 випадкових зображень з документами -> datasets/doccorner_sample/ + ground_truth.json
python src/extract_parquet.py docs/datasets/DocCornerDataset/test --limit 300
python src/evaluate.py datasets/doccorner_sample
```

Датасети не зберігаються в репозиторії (див. `.gitignore`).

## Тестування

Результати на датасетах SmartDoc 2015 і Mendeley, аналіз помилок і обмеження — [docs/03_testing.md](docs/03_testing.md).

Підсумок на частині **test** (не використовувалась для налаштування), успіх — IoU ≥ 0.9:

| Датасет | `simple` | `multi` v1 | **`multi` v2** |
|---------|----------|------------|----------------|
| SmartDoc 2015, 341 кадр | 20.2 % | 78.9 % | **86.8 %** |
| Mendeley, 300 фото | 13.0 % | 35.0 % | **42.7 %** |

Власні тестові фото:

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
├── COMMANDS.md     шпаргалка команд
├── requirements.txt
└── README.md
```
