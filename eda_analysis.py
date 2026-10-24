"""
Разведочный анализ (EDA) рынка автомобилей Казахстана (январь–сентябрь 2019)
и анализ позиции автоцентра «Меркур Авто» (Mercur Auto).

Запуск (после clean_autokz.py):
    python eda_analysis.py

Читает autokz2019_clean.csv, сохраняет графики в папку figures/
и сводные таблицы в analysis_results.md.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "autokz2019_clean.csv"
FIG_DIR = BASE_DIR / "figures"
RESULTS_PATH = BASE_DIR / "analysis_results.md"

COMPANY = "Mercur Auto"

# Палитра: основной цвет рынка, акцент для «Меркур Авто», нейтральный серый
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#b5b3ad"
TEXT, MUTED = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": GRAY, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
    "axes.axisbelow": True,
})

MONTH_RU = {1: "янв", 2: "фев", 3: "мар", 4: "апр", 5: "май", 6: "июн",
            7: "июл", 8: "авг", 9: "сен"}

report = []  # строки markdown-отчёта с цифрами


def md(text=""):
    print(text)
    report.append(text)


def table(df, floatfmt=1):
    """DataFrame -> markdown-таблица."""
    df = df.copy()
    for c in df.select_dtypes("number").columns:
        df[c] = df[c].round(floatfmt)
    md(df.to_markdown())
    md()


def save(fig, name):
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / name)
    plt.close(fig)


def barh(series, title, xlabel, name, highlight=None, fmt="{:,.0f}"):
    """Горизонтальный столбчатый график с подписями значений."""
    s = series.sort_values()
    colors = [ORANGE if highlight and idx == highlight else BLUE for idx in s.index]
    fig, ax = plt.subplots(figsize=(8, 0.38 * len(s) + 1))
    ax.barh(s.index.astype(str), s.values, color=colors, height=0.7)
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}".replace(",", " ")))
    for y, v in enumerate(s.values):
        ax.text(v, y, " " + fmt.format(v).replace(",", " "), va="center", fontsize=8, color=MUTED)
    ax.margins(x=0.12)
    save(fig, name)


# ---------------------------------------------------------------------------

def load():
    df = pd.read_csv(DATA_PATH, parse_dates=["sale_date"])
    for col in ["fuel_type", "transmission", "drive_type", "segment_2013", "class_2013"]:
        df[col] = df[col].astype("category")
    df["month"] = df["sale_date"].dt.month
    return df


def eda(df):
    md("# Результаты анализа\n")
    md("## 1. Общая картина рынка\n")
    md(f"- Строк (сделок): {len(df):,}".replace(",", " "))
    md(f"- Продано автомобилей: {df['quantity'].sum():,} шт.".replace(",", " "))
    md(f"- Выручка рынка: {df['sale_usd'].sum() / 1e6:,.1f} млн USD".replace(",", " "))
    md(f"- Средняя цена автомобиля (выручка / шт.): "
       f"{df['sale_usd'].sum() / df['quantity'].sum():,.0f} USD".replace(",", " "))
    md(f"- Медианная цена в строке: {df['price_usd'].median():,.0f} USD".replace(",", " "))
    md(f"- Брендов: {df['brand'].nunique()}, моделей: {df['model'].nunique()}, "
       f"компаний: {df['company'].nunique()}, регионов: {df['region'].nunique()}\n")

    # --- Распределения основных переменных --------------------------------
    md("### Описательная статистика числовых признаков\n")
    table(df[["price_usd", "engine_volume", "release_year", "quantity"]].describe().T)

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    axes[0].hist(df["price_usd"], bins=60, color=BLUE)
    axes[0].set_title("Цена автомобиля, USD", loc="left")
    axes[0].axvline(df["price_usd"].median(), color=ORANGE, lw=2)
    axes[0].text(df["price_usd"].median(), axes[0].get_ylim()[1] * 0.92,
                 f"  медиана {df['price_usd'].median():,.0f}".replace(",", " "), color=ORANGE, fontsize=9)
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
    axes[1].hist(df["engine_volume"].dropna(), bins=50, color=BLUE)
    axes[1].set_title("Объём двигателя, л", loc="left")
    ry = df["release_year"].value_counts().sort_index()
    axes[2].bar(ry.index.astype(int).astype(str), ry.values, color=BLUE)
    axes[2].set_title("Год выпуска (число сделок)", loc="left")
    for ax in axes:
        ax.set_ylabel("сделок")
    save(fig, "01_distributions.png")

    md("Доля автомобилей по году выпуска (по штукам):\n")
    table((df.groupby("release_year")["quantity"].sum() / df["quantity"].sum() * 100)
          .rename("доля, %").to_frame())

    # --- Динамика по месяцам ----------------------------------------------
    monthly = df.groupby("month").agg(units=("quantity", "sum"), revenue=("sale_usd", "sum"))
    monthly["revenue_mln"] = monthly["revenue"] / 1e6
    md("## 2. Динамика продаж по месяцам\n")
    table(monthly[["units", "revenue_mln"]].rename(index=MONTH_RU))

    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    labels = [MONTH_RU[m] for m in monthly.index]
    axes[0].plot(labels, monthly["units"], color=BLUE, lw=2, marker="o", ms=6)
    axes[0].set_title("Продажи, шт.", loc="left")
    axes[0].set_ylim(0, monthly["units"].max() * 1.15)
    axes[1].plot(labels, monthly["revenue_mln"], color=BLUE, lw=2, marker="o", ms=6)
    axes[1].set_title("Выручка, млн USD", loc="left")
    axes[1].set_ylim(0, monthly["revenue_mln"].max() * 1.15)
    save(fig, "02_monthly_trend.png")

    # --- Бренды, модели, компании -----------------------------------------
    brands = df.groupby("brand").agg(units=("quantity", "sum"), revenue=("sale_usd", "sum"))
    brands["share_units_%"] = brands["units"] / brands["units"].sum() * 100
    brands["share_rev_%"] = brands["revenue"] / brands["revenue"].sum() * 100
    brands["avg_price"] = brands["revenue"] / brands["units"]
    md("## 3. Бренды и модели\n")
    md("Топ-10 брендов по количеству:\n")
    table(brands.sort_values("units", ascending=False).head(10)
          .assign(revenue=lambda d: d["revenue"] / 1e6).rename(columns={"revenue": "revenue_mln"}))
    md("Топ-10 брендов по выручке:\n")
    table(brands.sort_values("revenue", ascending=False).head(10)
          .assign(revenue=lambda d: d["revenue"] / 1e6).rename(columns={"revenue": "revenue_mln"}))
    barh(brands["units"].nlargest(15), "Топ-15 брендов по продажам, шт.", "шт.", "03_brands_units.png")
    barh(brands["revenue"].nlargest(15) / 1e6, "Топ-15 брендов по выручке, млн USD",
         "млн USD", "04_brands_revenue.png", fmt="{:,.1f}")

    models = df.groupby(["brand", "model"])["quantity"].sum().nlargest(10)
    models.index = [f"{b} {m}" for b, m in models.index]
    md("Топ-10 моделей по количеству:\n")
    table(models.rename("units").to_frame())
    barh(models, "Топ-10 моделей, шт.", "шт.", "05_models.png")

    companies = df.groupby("company").agg(units=("quantity", "sum"), revenue=("sale_usd", "sum"))
    companies["share_units_%"] = companies["units"] / companies["units"].sum() * 100
    companies["share_rev_%"] = companies["revenue"] / companies["revenue"].sum() * 100
    companies = companies.sort_values("revenue", ascending=False)
    companies["rank_rev"] = range(1, len(companies) + 1)
    companies["rank_units"] = companies["units"].rank(ascending=False, method="min").astype(int)
    md("Компании (дистрибьюторы/дилеры), топ-12 по выручке:\n")
    table(companies.head(12).assign(revenue=lambda d: d["revenue"] / 1e6)
          .rename(columns={"revenue": "revenue_mln"}))
    barh(companies["revenue"].head(15) / 1e6, "Топ-15 компаний по выручке, млн USD",
         "млн USD", "06_companies_revenue.png", highlight=COMPANY, fmt="{:,.1f}")

    # --- Регионы ----------------------------------------------------------
    regions = df.groupby("region").agg(units=("quantity", "sum"), revenue=("sale_usd", "sum"))
    regions["share_units_%"] = regions["units"] / regions["units"].sum() * 100
    regions["avg_price"] = regions["revenue"] / regions["units"]
    md("## 4. Регионы\n")
    table(regions.sort_values("units", ascending=False).head(12).drop(columns="revenue"))
    barh(regions["units"].nlargest(15), "Продажи по регионам (топ-15), шт.", "шт.", "07_regions.png")

    # --- Топливо, КПП, привод, сегменты -----------------------------------
    md("## 5. Структура рынка по техническим признакам и сегментам\n")
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    for ax, col, title in zip(axes, ["fuel_type", "transmission", "drive_type"],
                              ["Тип топлива", "Коробка передач", "Тип привода"]):
        share = df.groupby(col, observed=True)["quantity"].sum()
        share = (share / share.sum() * 100).sort_values()
        md(f"{title}, доля продаж (шт.), %:\n")
        table(share.rename("доля, %").to_frame())
        ax.barh(share.index.astype(str), share.values, color=BLUE, height=0.6)
        ax.set_title(title + ", % продаж", loc="left")
        ax.grid(axis="y", visible=False)
        for y, v in enumerate(share.values):
            ax.text(v, y, f" {v:.1f}%", va="center", fontsize=8, color=MUTED)
        ax.margins(x=0.18)
    save(fig, "08_fuel_transmission_drive.png")

    seg = df.groupby("segment_2013", observed=True).agg(units=("quantity", "sum"),
                                                        revenue=("sale_usd", "sum"))
    seg["share_units_%"] = seg["units"] / seg["units"].sum() * 100
    seg["share_rev_%"] = seg["revenue"] / seg["revenue"].sum() * 100
    seg["avg_price"] = seg["revenue"] / seg["units"]
    md("Сегменты:\n")
    table(seg.sort_values("units", ascending=False).drop(columns="revenue"))

    # --- Взаимосвязи ------------------------------------------------------
    md("## 6. Взаимосвязи\n")
    corr = df[["price_usd", "engine_volume", "release_year"]].astype(float).corr()
    md("Корреляция Пирсона:\n")
    table(corr, 2)
    trans_price = df.groupby("transmission", observed=True)["price_usd"].median()
    md("Медианная цена по типу КПП:\n")
    table(trans_price.rename("median_price").to_frame(), 0)

    cars = df[df["segment_2013"] != "Коммерческие автомобили"].dropna(subset=["engine_volume"])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.scatter(cars["engine_volume"], cars["price_usd"], s=10, alpha=0.25, color=BLUE,
               edgecolors="none")
    ax.set_title("Цена и объём двигателя (легковые и SUV)", loc="left")
    ax.set_xlabel("объём двигателя, л")
    ax.set_ylabel("цена, USD")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
    save(fig, "09_price_vs_engine.png")

    # --- Аномалии ---------------------------------------------------------
    md("## 7. Аномалии\n")
    md(f"- Возвратов: {df['is_return'].sum()} "
       f"({df.loc[df['is_return'], ['brand', 'model', 'quantity', 'sale_usd']].to_dict('records')})")
    bulk = df[df["is_bulk"]]
    md(f"- Оптовых сделок (≥10 шт.): {len(bulk)}; в них "
       f"{bulk['quantity'].sum():,}".replace(",", " ")
       + f" шт. ({bulk['quantity'].sum() / df['quantity'].sum() * 100:.1f}% рынка)")
    md("- Крупнейшие оптовые сделки:\n")
    table(df.nlargest(8, "quantity")[["sale_date", "company", "brand", "model", "region",
                                      "quantity", "sale_usd"]].set_index("sale_date"))
    return brands, companies, regions, seg


def mercur(df, brands, companies, regions):
    m = df[df["company"] == COMPANY]
    md("## 8. Позиция «Меркур Авто»\n")
    units, rev = m["quantity"].sum(), m["sale_usd"].sum()
    md(f"- Продано: {units} шт. на {rev / 1e6:.2f} млн USD")
    md(f"- Доля рынка: {units / df['quantity'].sum() * 100:.2f}% по штукам, "
       f"{rev / df['sale_usd'].sum() * 100:.2f}% по выручке")
    md(f"- Место среди {len(companies)} компаний: "
       f"{companies.loc[COMPANY, 'rank_rev']} по выручке, "
       f"{companies.loc[COMPANY, 'rank_units']} по штукам")
    avg_m = rev / units
    avg_all = df["sale_usd"].sum() / df["quantity"].sum()
    md(f"- Средний чек «Меркур Авто»: {avg_m:,.0f} USD против {avg_all:,.0f} USD по рынку "
       f"(×{avg_m / avg_all:.2f})".replace(",", " "))
    md(f"- Регионы присутствия: {', '.join(m['region'].unique())}")
    md(f"- Бренды: {', '.join(sorted(m['brand'].unique()))}\n")

    # Доля в брендах
    mb = m.groupby("brand").agg(units=("quantity", "sum"), revenue=("sale_usd", "sum"))
    mb["brand_market_units"] = brands.loc[mb.index, "units"]
    mb["share_in_brand_%"] = mb["units"] / mb["brand_market_units"] * 100
    mb["share_of_mercur_rev_%"] = mb["revenue"] / rev * 100
    mb["avg_price"] = mb["revenue"] / mb["units"]
    md("Доля «Меркур Авто» в продажах брендов:\n")
    table(mb.assign(revenue=lambda d: d["revenue"] / 1e6).rename(columns={"revenue": "revenue_mln"}))

    vw_sellers = (df[df["brand"] == "Volkswagen"].groupby("company")["quantity"].sum()
                  .sort_values(ascending=False))
    md("Кто продаёт Volkswagen (шт.):\n")
    table(vw_sellers.to_frame())

    md("Топ моделей «Меркур Авто»:\n")
    mm = m.groupby(["brand", "model"]).agg(units=("quantity", "sum"), revenue=("sale_usd", "sum"))
    mm["share_rev_%"] = mm["revenue"] / rev * 100
    table(mm.sort_values("revenue", ascending=False).head(10)
          .assign(revenue=lambda d: d["revenue"] / 1e6).rename(columns={"revenue": "revenue_mln"}))

    # Доля в регионах
    mr = m.groupby("region").agg(units=("quantity", "sum"), revenue=("sale_usd", "sum"))
    mr["region_units"] = regions.loc[mr.index, "units"]
    mr["share_in_region_%"] = mr["units"] / mr["region_units"] * 100
    mr["share_of_mercur_units_%"] = mr["units"] / units * 100
    md("Доля «Меркур Авто» в регионах:\n")
    table(mr.sort_values("units", ascending=False).drop(columns="revenue"))

    # Доля в сегментах / классах
    for col, title in [("segment_2013", "сегментах"), ("class_2013", "классах")]:
        market = df.groupby(col, observed=True)["quantity"].sum()
        mc = m.groupby(col, observed=True)["quantity"].sum()
        mc = mc[mc > 0]
        out = pd.DataFrame({"mercur_units": mc, "market_units": market[mc.index]})
        out["share_in_market_%"] = out["mercur_units"] / out["market_units"] * 100
        out["mercur_structure_%"] = out["mercur_units"] / units * 100
        out["market_structure_%"] = out["market_units"] / df["quantity"].sum() * 100
        md(f"Доля «Меркур Авто» в {title}:\n")
        table(out.sort_values("mercur_units", ascending=False))

    # Премиальный сегмент: прямые конкуренты
    premium = ["Audi", "Porsche", "BMW", "Mercedes-Benz", "Lexus", "Land Rover", "Jaguar",
               "Infiniti", "Cadillac", "Volvo", "Mini"]
    p = df[df["brand"].isin(premium)]
    pb = p.groupby("brand").agg(units=("quantity", "sum"), revenue=("sale_usd", "sum"))
    pb["share_units_%"] = pb["units"] / pb["units"].sum() * 100
    pb["share_rev_%"] = pb["revenue"] / pb["revenue"].sum() * 100
    pb["avg_price"] = pb["revenue"] / pb["units"]
    pb["company"] = p.groupby("brand")["company"].agg(lambda s: ", ".join(sorted(s.unique())))
    pb = pb.sort_values("revenue", ascending=False)
    md("Премиальный сегмент (бренды-конкуренты Audi и Porsche):\n")
    table(pb.assign(revenue=lambda d: d["revenue"] / 1e6).rename(columns={"revenue": "revenue_mln"}))
    m_prem = m[m["brand"].isin(premium)]
    md(f"Доля «Меркур Авто» в премиум-сегменте: "
       f"{m_prem['quantity'].sum() / pb['units'].sum() * 100:.1f}% по штукам, "
       f"{m_prem['sale_usd'].sum() / pb['revenue'].sum() * 100:.1f}% по выручке\n")

    # Динамика
    mmon = m.groupby("month")["quantity"].sum()
    allmon = df.groupby("month")["quantity"].sum()
    dyn = pd.DataFrame({"mercur": mmon, "market": allmon}).fillna(0)
    dyn["share_%"] = dyn["mercur"] / dyn["market"] * 100
    md("Динамика «Меркур Авто» по месяцам:\n")
    table(dyn.rename(index=MONTH_RU))
    md(f"Рост 3 кв. к 1 кв.: Меркур {dyn.loc[7:9, 'mercur'].sum() / dyn.loc[1:3, 'mercur'].sum() - 1:+.0%}, "
       f"рынок {dyn.loc[7:9, 'market'].sum() / dyn.loc[1:3, 'market'].sum() - 1:+.0%}\n")

    # Структура по КПП/топливу/приводу vs рынок
    md("Структура продаж «Меркур Авто» vs рынок (% шт.):\n")
    for col in ["transmission", "drive_type", "fuel_type"]:
        a = m.groupby(col, observed=True)["quantity"].sum()
        b = df.groupby(col, observed=True)["quantity"].sum()
        table(pd.DataFrame({"mercur_%": a / a.sum() * 100, "market_%": b / b.sum() * 100}))

    # Средний чек по сегментам vs рынок
    ac = pd.DataFrame({
        "mercur_avg": m.groupby("segment_2013", observed=True)["sale_usd"].sum()
                      / m.groupby("segment_2013", observed=True)["quantity"].sum(),
        "market_avg": df.groupby("segment_2013", observed=True)["sale_usd"].sum()
                      / df.groupby("segment_2013", observed=True)["quantity"].sum(),
    }).dropna()
    md("Средний чек по сегментам, USD:\n")
    table(ac, 0)

    # ---- Графики Меркур ----
    fig, ax = plt.subplots(figsize=(8, 3.6))
    s = mb["share_in_brand_%"].sort_values()
    ax.barh(s.index, s.values, color=ORANGE, height=0.5)
    ax.set_xlim(0, 110)
    ax.set_title("Доля «Меркур Авто» в продажах бренда в РК, %", loc="left")
    ax.grid(axis="y", visible=False)
    for y, v in enumerate(s.values):
        ax.text(v, y, f" {v:.1f}%", va="center", fontsize=9, color=MUTED)
    save(fig, "10_mercur_brand_share.png")

    fig, ax = plt.subplots(figsize=(8, 3.6))
    s = mr["share_in_region_%"].sort_values()
    ax.barh(s.index, s.values, color=ORANGE, height=0.5)
    ax.set_title("Доля «Меркур Авто» в продажах региона, % шт.", loc="left")
    ax.grid(axis="y", visible=False)
    for y, v in enumerate(s.values):
        ax.text(v, y, f" {v:.1f}%", va="center", fontsize=9, color=MUTED)
    ax.margins(x=0.15)
    save(fig, "11_mercur_region_share.png")

    # Структура по классам: Меркур vs рынок (сгруппированные столбцы)
    market_cls = df.groupby("class_2013", observed=True)["quantity"].sum()
    merc_cls = m.groupby("class_2013", observed=True)["quantity"].sum()
    top_cls = (market_cls / market_cls.sum()).add(merc_cls / units, fill_value=0).nlargest(9).index
    comp = pd.DataFrame({"Меркур Авто": merc_cls / units * 100,
                         "Рынок": market_cls / market_cls.sum() * 100}).loc[top_cls].fillna(0)
    comp = comp.sort_values("Рынок")
    fig, ax = plt.subplots(figsize=(8, 5))
    y = range(len(comp))
    ax.barh([i + 0.2 for i in y], comp["Рынок"], height=0.38, color=BLUE, label="Рынок")
    ax.barh([i - 0.2 for i in y], comp["Меркур Авто"], height=0.38, color=ORANGE, label="Меркур Авто")
    ax.set_yticks(list(y))
    ax.set_yticklabels(comp.index)
    ax.grid(axis="y", visible=False)
    ax.set_title("Структура продаж по классам, % шт.", loc="left")
    ax.legend(frameon=False, loc="lower right")
    save(fig, "12_mercur_class_structure.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    s = pb["avg_price"].sort_values()
    colors = [ORANGE if b in ("Audi", "Porsche") else BLUE for b in s.index]
    ax.barh(s.index, s.values, color=colors, height=0.6)
    ax.set_title("Средний чек премиальных брендов, USD (оранжевый — бренды Меркур Авто)", loc="left")
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
    for yy, v in enumerate(s.values):
        ax.text(v, yy, f" {v / 1000:.0f}k", va="center", fontsize=8, color=MUTED)
    ax.margins(x=0.1)
    save(fig, "13_premium_avg_check.png")


def mercur_dealer_check(df):
    """
    Проверка гипотезы о падении продаж «Меркур Авто» с июня.
    В чистом датасете столбца «Дилер» нет (удалён по заданию), поэтому для
    проверки читаем сырой файл: с июня Volkswagen, проданные в дилерских
    центрах «Mercur Auto …», записаны на импортёра Volkswagen Group Rus.
    """
    raw_path = BASE_DIR / "autokz2019.csv"
    if not raw_path.exists():
        return
    raw = pd.read_csv(raw_path, sep=";", decimal=",").drop_duplicates()
    raw["Компания"] = raw["Компания"].str.strip()
    via = raw[(raw["Компания"] == "Volkswagen Group Rus")
              & raw["Наименование дилерского центра"].str.contains("Mercur", na=False)].copy()
    via["month"] = via["Месяц"].map({"Январь": 1, "Февраль": 2, "Март": 3, "Апрель": 4, "Май": 5,
                                     "Июнь": 6, "Июль": 7, "Август": 8, "Сентябрь": 9})
    m = df[df["company"] == COMPANY]
    md("## 9. Скорректированная оценка «Меркур Авто» (с учётом продаж VW через импортёра)\n")
    md(f"- Volkswagen, проданные в салонах Mercur Auto, но записанные на Volkswagen Group Rus: "
       f"{via['Количество'].sum():.0f} шт. на {via['Продажа, USD'].sum() / 1e6:.2f} млн USD")
    units = m["quantity"].sum() + via["Количество"].sum()
    rev = m["sale_usd"].sum() + via["Продажа, USD"].sum()
    md(f"- Итого через сеть Mercur Auto: {units:.0f} шт., {rev / 1e6:.2f} млн USD → доля рынка "
       f"{units / df['quantity'].sum() * 100:.2f}% по штукам, "
       f"{rev / df['sale_usd'].sum() * 100:.2f}% по выручке")
    vw_total = df.loc[df["brand"] == "Volkswagen", "quantity"].sum()
    vw_merc = m.loc[m["brand"] == "Volkswagen", "quantity"].sum() + via["Количество"].sum()
    md(f"- Доля сети Mercur в продажах Volkswagen в РК: {vw_merc / vw_total * 100:.1f}%")
    md(f"- Средний чек сети: {rev / units:,.0f} USD\n".replace(",", " "))

    own = m.groupby("month")["quantity"].sum()
    imp = via.groupby("month")["Количество"].sum()
    dyn = pd.DataFrame({"Mercur Auto (компания)": own, "VW через импортёра": imp},
                       index=range(1, 10)).fillna(0)
    dyn["Итого"] = dyn.sum(axis=1)
    md("По месяцам, шт.:\n")
    table(dyn.rename(index=MONTH_RU), 0)
    md(f"Рост 3 кв. к 1 кв. с учётом импортёра: "
       f"{dyn.loc[7:9, 'Итого'].sum() / dyn.loc[1:3, 'Итого'].sum() - 1:+.0%}\n")

    fig, ax = plt.subplots(figsize=(8, 3.8))
    labels = [MONTH_RU[i] for i in dyn.index]
    ax.bar(labels, dyn["Mercur Auto (компания)"], color=ORANGE, width=0.6,
           label="записано на Mercur Auto", edgecolor="white", linewidth=1.5)
    ax.bar(labels, dyn["VW через импортёра"], bottom=dyn["Mercur Auto (компания)"],
           color=BLUE, width=0.6, label="VW в салонах Mercur, записано на VW Group Rus",
           edgecolor="white", linewidth=1.5)
    ax.set_title("Продажи сети Mercur Auto по месяцам, шт.", loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.set_ylim(0, dyn["Итого"].max() * 1.3)
    save(fig, "14_mercur_network_monthly.png")


def main():
    df = load()
    brands, companies, regions, _ = eda(df)
    mercur(df, brands, companies, regions)
    mercur_dealer_check(df)
    RESULTS_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"\nСохранено: {RESULTS_PATH} и графики в {FIG_DIR}")


if __name__ == "__main__":
    main()
