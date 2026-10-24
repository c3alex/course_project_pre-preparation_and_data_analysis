"""
Очистка датасета продаж автомобилей в Казахстане за 9 месяцев 2019 года.

Курсовой проект «Предподготовка и анализ данных».
Скрипт превращает сырой CSV (autokz2019.csv) в чистый датасет, готовый
для анализа или загрузки в BI-систему.

Запуск:
    python clean_autokz.py [путь_к_сырому_csv] [путь_к_чистому_csv]

По умолчанию читает autokz2019.csv и пишет autokz2019_clean.csv
в ту же папку, где лежит скрипт. Журнал очистки сохраняется в cleaning_log.txt.
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_PATH = BASE_DIR / "autokz2019.csv"
CLEAN_PATH = BASE_DIR / "autokz2019_clean.csv"
LOG_PATH = BASE_DIR / "cleaning_log.txt"

# ---------------------------------------------------------------------------
# Справочники
# ---------------------------------------------------------------------------

# Столбцы без аналитической ценности или с критичным числом пропусков.
# Названия в задании и в файле немного отличаются: «Дилер» = «Наименование
# дилерского центра», «Тип платежа» = «Форма расчета», «Сегмент ЕПР» =
# «Сегментация Eng», «Локализация» = «Локализация производства».
COLUMNS_TO_DROP = [
    "Наименование дилерского центра",
    "Форма расчета",
    "Сегмент",
    "Сегментация Eng",
    "Локализация производства",
    "Тип клиента",
    "Модификация",
]

COLUMN_NAMES = {
    "Год": "year",
    "Месяц": "month",
    "Компания": "company",
    "Бренд": "brand",
    "Модель": "model",
    "Год выпуска": "release_year",
    "Страна-производитель": "country",
    "Вид топлива": "fuel_type",
    "Объём двиг, л,": "engine_volume",
    "Коробка передач": "transmission",
    "Тип привода": "drive_type",
    "Регион": "region",
    "Количество": "quantity",
    "Цена, USD": "price_usd",
    "Продажа, USD": "sale_usd",
    "Область": "area",
    "Сегментация 2013": "segment_2013",
    "Класс 2013": "class_2013",
}

MONTHS = {
    "Январь": 1, "Февраль": 2, "Март": 3, "Апрель": 4, "Май": 5, "Июнь": 6,
    "Июль": 7, "Август": 8, "Сентябрь": 9, "Октябрь": 10, "Ноябрь": 11,
    "Декабрь": 12,
}

# ISO 3166-1 alpha-3
COUNTRY_CODES = {
    "Республика Казахстан": "KAZ",
    "Российская Федерация": "RUS",
    "Япония": "JPN",
    "Турция": "TUR",
    "Таиланд": "THA",
    "США": "USA",
    "Германия": "DEU",
    "UK": "GBR",
    "Китай": "CHN",
    "Узбекистан": "UZB",
    "Австрия": "AUT",
    "Швеция": "SWE",
    "Корея": "KOR",
    "Нидерланды": "NLD",
    "Испания": "ESP",
    "Польша": "POL",
    "Бельгия": "BEL",
    "Венгрия": "HUN",
    "Белоруссия": "BLR",
}

FUEL_CODES = {
    "бензин": "F",
    "дизель": "D",
    "электричество": "E",
    "электро": "E",
    "гибрид": "HYB",
}

# Тип привода -> единый формат: 4WD (полный), FWD (передний), RWD (задний),
# 2WD (монопривод без уточнения оси — в т.ч. колёсная формула 4x2 у грузовиков).
DRIVE_CODES = {
    "4wd": "4WD", "4 wd": "4WD", "полный": "4WD", "awd": "4WD",
    "quattro": "4WD", "4motion": "4WD", "4x4": "4WD", "4х4": "4WD",
    "ff": "FWD", "fwd": "FWD", "передний": "FWD", "передний (ff)": "FWD",
    "rwd": "RWD", "задний": "RWD",
    "2wd": "2WD", "2 wd": "2WD", "4x2": "2WD", "4х2": "2WD", "4х2.2": "2WD",
}

# Неявные дубликаты в названиях компаний -> единое написание
COMPANY_NAMES = {
    "Mercur Autos": "Mercur Auto",
    "Меркур Авто": "Mercur Auto",
    "Ravon Motors Kazakstan": "Равон Моторс Казахстан",
    "MMC RUS": "ММС Рус",
    "Hino Motors": "Хино Моторс Казахстан",
    "Autokapital": "Автокапитал",
    "Каспиан Моторс": "Caspian Motors",
}

# Значения-заглушки, которые означают отсутствие данных
MISSING_MARKERS = ["", "0", "#Н/Д", "#N/A", "nan"]

CATEGORY_COLUMNS = ["fuel_type", "transmission", "drive_type",
                    "segment_2013", "class_2013"]

BULK_THRESHOLD = 10  # строка с 10+ машинами считается оптовой сделкой


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------

class Logger:
    """Печатает шаги очистки и сохраняет их в файл-журнал."""

    def __init__(self):
        self.lines = []

    def __call__(self, message=""):
        print(message)
        self.lines.append(str(message))

    def save(self, path):
        Path(path).write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def strip_strings(df):
    """Убирает лишние пробелы по краям и двойные пробелы внутри строк."""
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip().str.replace(r"\s{2,}", " ", regex=True)
    return df


def fill_by_model_mode(df, column):
    """Заполняет пропуски самым частым значением для той же пары бренд+модель."""
    def mode_or_nan(s):
        m = s.dropna().mode()
        return m.iloc[0] if not m.empty else np.nan

    modes = df.groupby(["brand", "model"])[column].transform(mode_or_nan)
    before = df[column].isna().sum()
    df[column] = df[column].fillna(modes)
    return before, df[column].isna().sum()


def map_transmission(value):
    """Сводит 80+ вариантов записи КПП к двум категориям: Автомат / Механика."""
    if pd.isna(value):
        return np.nan
    original = str(value).upper().replace(" ", "")
    if original in {"", "0", "#Н/Д", "4WD", "FWD", "ПЕРЕДНИЙ"}:
        return np.nan  # мусор / значение из соседнего столбца
    if original.startswith("МЕХ") or original.startswith("МКП"):
        return "Механика"  # «Мех.», МКП, МКПП
    # Кириллические М/Т/А (их часто путают с латинскими) -> латиница
    v = original.translate(str.maketrans("МТА", "MTA"))
    if re.fullmatch(r"\d?M(/?T)?", v):
        return "Механика"  # MT, 5MT, 6 MT, 6M/T, 5M
    if v == "TDI":  # у JAC T6 в поле КПП попал тип двигателя; модель продаётся с МКПП
        return "Механика"
    # Всё остальное — автоматические коробки: AT, CVT, AMT, DSG, DCT, PDK,
    # Tiptronic, Steptronic, G-TRONIC, S-tronic, АКПП, Powershift,
    # «8» (8-ступенчатый автомат Porsche), «РЕДУКТОР» (электромобиль).
    return "Автомат"


def parse_engine_volume(value):
    """Достаёт объём двигателя в литрах из строк вида '1,6 MPI', '4,98 L,', '2.4G'."""
    if pd.isna(value):
        return np.nan
    v = str(value).upper()
    if "KWH" in v or "Л.С" in v:
        return np.nan  # ёмкость батареи / мощность вместо объёма (электромобили)
    v = v.replace(",", ".").replace("L", "")
    match = re.search(r"\d+(?:\.\d+)?", v)
    if not match:
        return np.nan  # 'AT', 'MT', '#Н/Д' и т.п.
    number = float(match.group())
    return number if number > 0 else np.nan


def title_case(series):
    """Каждое слово с заглавной буквы; 'г.Алматы' -> 'Г. Алматы'."""
    s = series.str.replace(r"^г\.\s*", "г. ", regex=True)
    return s.str.lower().str.title()


# ---------------------------------------------------------------------------
# Этапы очистки
# ---------------------------------------------------------------------------

def load(path, log):
    # sep=';' — разделитель столбцов, decimal=',' — десятичная запятая в ценах
    df = pd.read_csv(path, sep=";", decimal=",", dtype={"Год выпуска": str,
                                                        "Вид топлива": str,
                                                        "Объём двиг, л,": str})
    log(f"Загружено: {df.shape[0]} строк, {df.shape[1]} столбцов")
    log("Пропуски в сыром файле (доля):")
    log((df.isna().mean().sort_values(ascending=False).round(3)).to_string())
    return df


def basic_cleaning(df, log):
    n = len(df)
    df = df.dropna(how="all")
    log(f"\nПолностью пустых строк удалено: {n - len(df)}")

    # Явные дубликаты ищем по ВСЕМ исходным столбцам (включая дилера, тип
    # клиента, модификацию) до удаления столбцов: иначе разные сделки,
    # отличающиеся только удаляемыми полями, ошибочно склеились бы.
    n = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    log(f"Явных дубликатов (полностью совпадающих строк) удалено: {n - len(df)} "
        f"(осталось {len(df)} строк)")

    df = df.drop(columns=COLUMNS_TO_DROP)
    df = df.rename(columns=COLUMN_NAMES)
    df = strip_strings(df)
    log(f"Удалено столбцов: {len(COLUMNS_TO_DROP)}; осталось {df.shape[1]}")
    return df


def fix_shifted_rows(df, log):
    """
    У части строк Renault данные «съехали» на один столбец влево: в «Виде
    топлива» стоит объём (2 / 1,6), в «Объёме» — КПП (AT/MT), в «КПП» — привод.
    Возвращаем значения на свои места, топливо потом восстановим по модели.
    """
    df.loc[df["fuel_type"] == "0", "fuel_type"] = np.nan  # UAZ 3741: «0» = нет данных
    shifted = df["fuel_type"].str.fullmatch(r"\d+(?:[.,]\d+)?", na=False)
    df.loc[shifted, "drive_type"] = df.loc[shifted, "transmission"]
    df.loc[shifted, "transmission"] = df.loc[shifted, "engine_volume"]
    df.loc[shifted, "engine_volume"] = df.loc[shifted, "fuel_type"]
    df.loc[shifted, "fuel_type"] = np.nan
    log(f"\nСтрок со сдвигом столбцов исправлено: {shifted.sum()}")
    return df


def clean_categories(df, log):
    # Страна производства -> ALPHA-3
    unknown = set(df["country"].dropna()) - set(COUNTRY_CODES)
    if unknown:
        log(f"ВНИМАНИЕ: нет кода для стран {unknown}")
    df["country"] = df["country"].map(COUNTRY_CODES)

    # Вид топлива -> F / D / E / HYB
    df["fuel_type"] = df["fuel_type"].str.lower().map(FUEL_CODES)

    # Тип привода -> 4WD / FWD / RWD / 2WD (мусор вроде '0', '#Н/Д', 'Астана' -> NaN)
    df["drive_type"] = df["drive_type"].str.lower().map(DRIVE_CODES)

    # Коробка передач -> Автомат / Механика
    df["transmission"] = df["transmission"].replace(MISSING_MARKERS, np.nan)
    df["transmission"] = df["transmission"].map(map_transmission)

    # Пропуски в категориях восстанавливаем по самому частому значению модели:
    # у одной модели в датасете, как правило, одна комплектация по этим признакам.
    log("\nЗаполнение пропусков модой по бренду+модели (было -> осталось):")
    for col in ["fuel_type", "transmission", "drive_type"]:
        before, after = fill_by_model_mode(df, col)
        log(f"  {col}: {before} -> {after}")

    # У оставшихся моделей топливо не указано ни в одной строке. На рынке РК
    # грузовики и автобусы (Shacman, Foton, Урал, Dong Feng …) — дизельные,
    # легковые модели без данных (Renault Logan/Kaptur, Mazda CX-5/6) — бензиновые.
    heavy = df["class_2013"].str.contains("грузовик|автобус", case=False, na=False)
    missing = df["fuel_type"].isna()
    df.loc[missing & heavy, "fuel_type"] = "D"
    df.loc[missing & ~heavy, "fuel_type"] = "F"
    log(f"  fuel_type: по типу ТС дозаполнено {missing.sum()} "
        f"(D — {(missing & heavy).sum()}, F — {(missing & ~heavy).sum()})")
    return df


def clean_engine_volume(df, log):
    raw = df["engine_volume"].copy()
    df["engine_volume"] = raw.map(parse_engine_volume)

    # Chevrolet Niva выпускается только с мотором 1,7 л; значения 3,7 … 26,7 —
    # ошибка протягивания ячеек в Excel (+1 на каждой строке).
    niva = (df["brand"] == "Chevrolet") & (df["model"] == "Niva") & (df["engine_volume"] > 2)
    df.loc[niva, "engine_volume"] = 1.7
    log(f"\nОбъём двигателя: исправлено аномалий Chevrolet Niva: {niva.sum()}")

    # Jaguar I-Pace: в поле объёма записана мощность «400 л.с.»; у ANKAI EV —
    # ёмкость батареи «88 kWh». Это электромобили, объёма ДВС у них нет.
    electric = df["fuel_type"] == "E"
    log(f"Объём двигателя: обнулено (NaN) у электромобилей: "
        f"{(electric & raw.notna()).sum()} (Jaguar I-Pace 400 л.с., ANKAI 88 kWh)")
    df.loc[electric, "engine_volume"] = np.nan

    # Легковой автомобиль/внедорожник с объёмом > 7 л — ошибка ввода
    # (например, VW Polo 8,7 л). Заменяем медианой этой модели.
    passenger = df["segment_2013"] != "Коммерческие автомобили"
    outlier = passenger & (df["engine_volume"] > 7)
    model_median = df.loc[~outlier].groupby(["brand", "model"])["engine_volume"].median()
    df.loc[outlier, "engine_volume"] = [
        model_median.get((b, m), np.nan)
        for b, m in zip(df.loc[outlier, "brand"], df.loc[outlier, "model"])
    ]
    log(f"Объём двигателя: выбросов у легковых (> 7 л) заменено медианой модели: "
        f"{outlier.sum()}")

    # Объёмы вида 1,596 / 2,693 / 4,98 приводим к общепринятой точности 0,1 л
    df["engine_volume"] = df["engine_volume"].round(1)
    log(f"Объём двигателя: пропусков {df['engine_volume'].isna().sum()}, "
        f"диапазон {df['engine_volume'].min()}–{df['engine_volume'].max()} л")
    return df


def clean_numbers(df, log):
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["price_usd"] = pd.to_numeric(df["price_usd"], errors="coerce")
    df["sale_usd"] = pd.to_numeric(df["sale_usd"], errors="coerce")

    # Строки без количества и с нулевой суммой продажи — это не сделки
    no_sale = df["quantity"].isna() & (df["sale_usd"] == 0)
    df = df[~no_sale].copy()
    log(f"\nУдалено строк без продажи (количество пусто, сумма 0): {no_sale.sum()}")
    df["quantity"] = df["quantity"].astype("int64")

    check = (df["price_usd"] * df["quantity"] - df["sale_usd"]).abs() > 1
    log(f"Строк, где Цена × Количество ≠ Продажа: {check.sum()}")

    # Аномалии оставляем (это реальные операции), но помечаем флагами
    df["is_return"] = df["quantity"] < 0
    df["is_bulk"] = df["quantity"] >= BULK_THRESHOLD
    log(f"Возвраты (количество < 0): {df['is_return'].sum()}")
    log(f"Оптовые сделки (≥ {BULK_THRESHOLD} шт. в строке): {df['is_bulk'].sum()}, "
        f"максимум {df['quantity'].max()} шт.")
    top = df.nlargest(5, "quantity")[["company", "brand", "model", "region",
                                      "quantity", "sale_usd"]]
    log("Крупнейшие сделки:\n" + top.to_string(index=False))

    df["release_year"] = pd.to_numeric(
        df["release_year"].str.replace(r"\s", "", regex=True), errors="coerce"
    ).astype("Int64")
    log(f"Год выпуска: пропусков {df['release_year'].isna().sum()}, "
        f"диапазон {df['release_year'].min()}–{df['release_year'].max()}")
    return df


def build_sale_date(df, log):
    # Последний день месяца: 2019-01-31, 2019-02-28, 2019-04-30 ...
    month_num = df["month"].map(MONTHS)
    first_day = pd.to_datetime(dict(year=df["year"], month=month_num, day=1))
    df["sale_date"] = first_day + pd.offsets.MonthEnd(0)
    df = df.drop(columns=["year", "month"])
    log(f"\nДата продажи: {df['sale_date'].min().date()} – {df['sale_date'].max().date()}")
    return df


def clean_company_and_geo(df, log):
    df["company"] = (df["company"]
                     .str.replace(r'^"?ТОО\s*', "", regex=True)
                     .str.replace('"', "", regex=False)
                     .str.strip()
                     .replace(COMPANY_NAMES))
    log(f"\nКомпаний после объединения неявных дубликатов: {df['company'].nunique()}")

    df["region"] = title_case(df["region"])
    df["area"] = title_case(df["area"])
    for col in ["segment_2013", "class_2013"]:
        df[col] = df[col].str.strip()
    return df


def finalize(df, log):
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    for col in CATEGORY_COLUMNS:
        df[col] = df[col].astype("category")

    ordered = ["sale_date", "company", "brand", "model", "release_year", "country",
               "fuel_type", "engine_volume", "transmission", "drive_type",
               "segment_2013", "class_2013", "region", "area",
               "quantity", "price_usd", "sale_usd", "is_return", "is_bulk"]
    df = df[ordered].sort_values(["sale_date", "company", "brand", "model"])
    df = df.reset_index(drop=True)

    log(f"\nИтог: {df.shape[0]} строк, {df.shape[1]} столбцов")
    log("Типы данных:\n" + df.dtypes.to_string())
    log("Пропуски:\n" + df.isna().sum().to_string())
    return df


def main(raw_path=RAW_PATH, clean_path=CLEAN_PATH):
    log = Logger()
    df = load(raw_path, log)
    df = basic_cleaning(df, log)
    df = fix_shifted_rows(df, log)
    df = clean_categories(df, log)
    df = clean_engine_volume(df, log)
    df = clean_numbers(df, log)
    df = build_sale_date(df, log)
    df = clean_company_and_geo(df, log)
    df = finalize(df, log)

    # utf-8-sig — чтобы кириллица корректно открывалась и в Excel, и в BI
    df.to_csv(clean_path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    log(f"\nСохранено: {clean_path}")
    log.save(LOG_PATH)
    return df


if __name__ == "__main__":
    args = sys.argv[1:]
    main(*(Path(a) for a in args[:2]))
