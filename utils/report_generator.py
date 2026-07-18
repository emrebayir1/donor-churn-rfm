import os
import re
from datetime import datetime
import tempfile
from contextlib import contextmanager
from fpdf import FPDF, XPos, YPos
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
from textwrap import dedent

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

os.environ['CPL_LOG'] = '/dev/null'
os.environ['GDAL_DATA'] = '/dev/null'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
LOGO_PATH = os.path.join(ASSETS_DIR, "logo.png")
TR_CITIES = os.path.join(ASSETS_DIR, "tr_cities.json")
TRANSLATIONS = {
    "Kampanya Adi": {
        "Eğitim Bursu ve Öğrenci Destekleri": "Education Scholarships & Student Support",
        "Doğa ve Çevre Koruma Projeleri": "Nature & Environmental Protection Projects",
        "Afet ve Acil Durum Yönetimi": "Disaster & Emergency Response",
        "Toplumsal Gelişim ve Eğitim Seminerleri": "Community Development & Training Seminars",
        "Bilim ve Teknoloji Fonu": "Science & Technology Fund",
        "Kültür ve Sanat Faaliyetleri": "Culture & Arts Activities",
        "Dezavantajlı Gruplara Sosyal Destek": "Social Support for Disadvantaged Groups",
        "Sürdürülebilir Yaşam Alanları": "Sustainable Living Areas",
        "Yerel Kalkınma Girişimleri": "Local Development Initiatives",
        "Gönüllülük Yaygınlaştırma Programı": "Volunteer Promotion Program",
        "DİĞER": "OTHER",
    },

    "Tür": {
        "Genel Bağış": "General Donation",
        "Proje Bazlı Fon": "Project-Based Fund",
        "Düzenli Destek": "Recurring Support",
        "Sponsorluk": "Sponsorship",
    },

    "Bagis Kanali": {
        "Web": "Web",
        "Mobil Uygulama": "Mobile App",
        "Banka Entegrasyonu": "Bank Integration",
        "Şube/Ofis": "Branch / Office",
        "Çağrı Merkezi": "Call Center",
        "Sosyal Medya": "Social Media",
    },

    "Odeme Sekli": {
        "Kredi Kartı": "Credit Card",
        "EFT/Havale": "Bank Transfer",
        "Nakit": "Cash",
        "Mobil Ödeme": "Mobile Payment",
    },

    "Para Birimi": {
        "TRY": "TRY",
        "USD": "USD",
        "EUR": "EUR",
    },

    "Terk Riski": {
        "Düşük": "Low",
        "Orta": "Medium",
        "Yüksek": "High",
    },

    "Deger_Segmenti": {
        "VIP": "VIP",
        "Yüksek": "High",
        "Orta": "Medium",
        "Standart": "Standard",
    },

    "Davranis_Segmenti": {
        "Şampiyonlar": "Champions",
        "Sadık Bağışçılar": "Loyal Donors",
        "Potansiyel Sadıklar": "Potential Loyal Donors",
        "Yeni Bağışçılar": "New Donors",
        "İlgi Gerektiren": "Need Attention",
        "Uykuda": "Dormant",
        "Kayıp Riskli": "At Risk",
        "Pasif": "Inactive",
        "Tanımsız": "Undefined",
    },

    "Final_Segment": {
        "VIP Şampiyonlar": "VIP Champions",
        "VIP Sadık Bağışçılar": "VIP Loyal Donors",
        "VIP Potansiyel Sadıklar": "VIP Potential Loyal Donors",
        "VIP Yeni Bağışçılar": "VIP New Donors",
        "VIP İlgi Gerektiren": "VIP Need Attention",
        "VIP Uykuda": "VIP Dormant",
        "VIP Kayıp Riskli": "VIP At Risk",
        "Yüksek Şampiyonlar": "High Champions",
        "Yüksek Sadık Bağışçılar": "High Loyal Donors",
        "Yüksek Potansiyel Sadıklar": "High Potential Loyal Donors",
        "Yüksek Yeni Bağışçılar": "High New Donors",
        "Yüksek İlgi Gerektiren": "High Need Attention",
        "Yüksek Uykuda": "High Dormant",
        "Yüksek Kayıp Riskli": "High At Risk",
        "Orta Şampiyonlar": "Medium Champions",
        "Orta Sadık Bağışçılar": "Medium Loyal Donors",
        "Orta Potansiyel Sadıklar": "Medium Potential Loyal Donors",
        "Orta Yeni Bağışçılar": "Medium New Donors",
        "Orta İlgi Gerektiren": "Medium Need Attention",
        "Orta Uykuda": "Medium Dormant",
        "Orta Kayıp Riskli": "Medium At Risk",
        "Standart Şampiyonlar": "Standard Champions",
        "Standart Sadık Bağışçılar": "Standard Loyal Donors",
        "Standart Potansiyel Sadıklar": "Standard Potential Loyal Donors",
        "Standart Yeni Bağışçılar": "Standard New Donors",
        "Standart İlgi Gerektiren": "Standard Need Attention",
        "Standart Uykuda": "Standard Dormant",
        "Standart Kayıp Riskli": "Standard At Risk",
        "Pasif / Aday": "Inactive / Prospect",
    },
}
#=======================================
#CLASS
#=======================================
class DonorReport(FPDF):

    def __init__(self, language='tr'):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=15)
        self.language = language

        base_dir = os.path.dirname(os.path.abspath(__file__))
        font_dir = os.path.join(base_dir, "fonts")

        self.add_font("DejaVu", "", os.path.join(font_dir, "DejaVuSans.ttf"))
        self.add_font("DejaVu", "B", os.path.join(font_dir, "DejaVuSans-Bold.ttf"))
        self.add_font("DejaVu", "I", os.path.join(font_dir, "DejaVuSans-Oblique.ttf"))
        self.set_font("DejaVu", size=10)

        self.color_title = (0, 0, 51)
        self.color_section = (0, 0, 51)
        self.color_header_bg = (102, 126, 234)
        self.is_cover_page = False

    # -------------------------
    # Helpers
    # -------------------------
    def page_width_usable(self):
        return self.w - self.l_margin - self.r_margin

    def ensure_space(self, needed_h: float):
        if self.get_y() + needed_h > (self.h - self.b_margin):
            self.add_page()

    # -------------------------
    # Header / Footer
    # -------------------------
    def header(self):
        if self.is_cover_page:
            if os.path.exists(LOGO_PATH):
                self.image(LOGO_PATH, x=55, y=25, w=100)
            return

        # TOP LINE
        self.set_draw_color(0, 0, 0)
        self.set_line_width(0.4)
        self.line(
            self.l_margin,
            20,
            self.w - self.r_margin,
            20
        )

        if self.language == 'en':
            title_text = "Donor Segmentation Analysis Report"
        else:
            title_text = "Bağışçı Segmentasyon Analizi Raporu"

        self.set_xy(self.l_margin, 10)
        self.set_font("DejaVu", "B", 12)
        self.set_text_color(*self.color_title)
        self.cell(
            self.epw,
            8,
            title_text,
            align="C"
        )

        self.ln(15)

        def footer(self):
            if self.is_cover_page:
                return

            # BOTTOM LINE
            y = self.h - 20
            self.set_draw_color(0, 0, 0)
            self.set_line_width(0.4)
            self.line(
                self.l_margin,
                y,
                self.w - self.r_margin,
                y
            )

            self.set_y(-15)
            self.set_font("DejaVu", "I", 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 10, f"Page {self.page_no()}", align="C")

    # -------------------------
    # Blocks
    # -------------------------
    def chapter_title(self, title):
        self.ensure_space(20)
        self.set_font("DejaVu", "B", 16)
        self.set_text_color(*self.color_section)
        self.cell(0, 10, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(3)

    def sub_title(self, title):
        self.ensure_space(15)
        self.set_font("DejaVu", "B", 12)
        self.set_text_color(*self.color_section)
        self.cell(0, 10, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.ln(3)

    def info_box(self, label, value):
        self.set_font("DejaVu", "B", 10)

        label_width = self.get_string_width(label + ":") + 2

        self.cell(
            label_width,
            8,
            label + ":",
            new_x=XPos.RIGHT,
            new_y=YPos.TOP,
        )

        self.set_font("DejaVu", "", 10)

        self.cell(
            0,
            8,
            str(value),
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )

    def paragraph(self, text, size=10, style="", indent=10):
        self.ensure_space(10)
        self.set_font("DejaVu", style, size)
        self.set_text_color(0, 0, 0)

        original_margin = self.l_margin
        self.set_left_margin(original_margin + indent)

        self.multi_cell(0, 6, text)
        self.ln(2)

        self.set_left_margin(original_margin)

    # -------------------------
    # Chart: direct image
    # -------------------------
    def add_chart(
        self,
        img_path: str,
        title: str = None,
        w: float = 0,
        h: float = 0,
        align: str = "C",
        caption: str = None,
        pad_top: float = 2,
        pad_bottom: float = 3,
        approx_ratio: float = 0.60,
    ):
        if not img_path or not os.path.exists(img_path):
            self.set_font("DejaVu", "I", 9)
            self.set_text_color(150, 0, 0)
            self.multi_cell(0, 5, f"[Chart not found] {img_path}")
            self.set_text_color(0, 0, 0)
            return

        if title:
            self.ensure_space(10)
            self.set_font("DejaVu", "B", 11)
            self.set_text_color(0, 0, 0)
            self.cell(0, 7, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        self.ln(pad_top)

        usable_w = self.page_width_usable()
        if w == 0:
            w = usable_w

        approx_h = h if h else (w * approx_ratio)
        self.ensure_space(approx_h + 10)

        if align.upper() == "L":
            x = self.l_margin
        elif align.upper() == "R":
            x = self.w - self.r_margin - w
        else:
            x = self.l_margin + (usable_w - w) / 2

        y = self.get_y()
        self.image(img_path, x=x, y=y, w=w, h=h)

        self.set_y(y + (approx_h if h == 0 else h))
        self.ln(pad_bottom)

        if caption:
            self.set_font("DejaVu", "I", 8)
            self.set_text_color(90, 90, 90)
            self.multi_cell(0, 4, caption)
            self.set_text_color(0, 0, 0)

    # -------------------------
    # Chart: from function -> temp png -> embed -> delete
    # -------------------------
    @contextmanager
    def _temp_png(self, prefix="chart_", suffix=".png"):
        fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix)
        os.close(fd)
        try:
            yield path
        finally:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass

    def add_chart_from_func(
        self,
        chart_func,
        *func_args,
        title: str = None,
        caption: str = None,
        w: float = 0,
        h: float = 0,
        align: str = "C",
        approx_ratio: float = 0.60,
        **func_kwargs
    ):
        with self._temp_png() as tmp_path:
            chart_func(tmp_path, *func_args, **func_kwargs)

            self.add_chart(
                img_path=tmp_path,
                title=title,
                caption=caption,
                w=w,
                h=h,
                align=align,
                approx_ratio=approx_ratio,
            )

#=======================================
# CHART FUNCTIONS
#=======================================

# Donut Chart Function
def chart_donut(
    out_path,
    data,
    column_name,
    language="tr",
    translations=TRANSLATIONS,
):
    if column_name not in data.columns:
        return

    s = data[column_name]
    filtered = s[(s.notna()) & (s != "Bilinmiyor") & (s != "BİLİNMİYOR")]

    counts = filtered.value_counts()
    percentages = counts / counts.sum() * 100

    base_colors = plt.get_cmap("tab10").colors
    if len(counts) > 10:
        extra_colors = plt.get_cmap("Set3").colors
        colors = list(base_colors) + list(extra_colors)
    else:
        colors = base_colors[: len(counts)]

    fig, ax = plt.subplots(figsize=(6.2, 3.8))

    wedges, _ = ax.pie(
        counts,
        startangle=90,
        colors=colors[: len(counts)],
        wedgeprops=dict(width=0.45),
    )

    ax.set(aspect="equal")

    # --------------------------
    # TRANSLATIONS
    # --------------------------

    translation_map = {}

    if language == "en" and translations is not None:

        translation_key = column_name

        for suffix in ("_mode", "_prev", "_next"):
            if translation_key.endswith(suffix):
                translation_key = translation_key[:-len(suffix)]
                break

        translation_map = translations.get(translation_key, {})

    legend_labels = []

    for cat, count, perc in zip(counts.index, counts.values, percentages):

        display_cat = translation_map.get(cat, cat)

        if language == "en":
            label = f"{display_cat}   {count:,.0f}  ({perc:.1f}%)"
        else:
            label = f"{display_cat}   {count:,.0f}  (%{perc:.1f})"

        legend_labels.append(label)

    legend = ax.legend(
        wedges,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=9,
    )

    if legend.get_texts():
        legend.get_texts()[0].set_fontweight("bold")

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    plt.tight_layout()
    plt.savefig(
        out_path,
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close()

# Bar Chart
def chart_bar(
    out_path,
    data,
    col,
    try_col="Toplam Tutar",
    top_n=10,
    agg_type="sum",
    language="tr",
    translations=TRANSLATIONS,
):

    if agg_type not in ["sum", "mean"]:
        raise ValueError("agg_type must be 'sum' or 'mean'.")

    df = data.copy()

    df = df[df[col].notna()]
    df = df[df[col] != "BİLİNMİYOR"]

    def make_numeric(series):
        if pd.api.types.is_numeric_dtype(series):
            return series

        s = series.astype(str).str.strip()
        s = s.str.replace(r"[^\d,.\-]", "", regex=True)

        mask_tr = s.str.contains(r"\.") & s.str.contains(r",")

        s.loc[mask_tr] = (
            s.loc[mask_tr]
            .str.replace(".", "", regex=False)
            .str.replace(",", ".", regex=False)
        )

        mask_only_comma = s.str.contains(",") & ~s.str.contains(r"\.")

        s.loc[mask_only_comma] = (
            s.loc[mask_only_comma]
            .str.replace(",", ".", regex=False)
        )

        return pd.to_numeric(s, errors="coerce")

    df[try_col] = make_numeric(df[try_col])
    df = df.dropna(subset=[try_col])

    grouped = (
        df.groupby(col)[try_col]
        .agg(agg_type)
        .sort_values(ascending=False)
    )

    top = grouped.head(top_n)
    other = grouped.iloc[top_n:]

    if not other.empty:
        other_value = other.sum() if agg_type == "sum" else other.mean()
        top.loc["OTHER" if language == "en" else "DİĞER"] = other_value

    final_df = top[::-1]

    # --------------------------
    # TRANSLATIONS
    # --------------------------

    labels = final_df.index.tolist()

    if language == "en" and translations is not None:

        translation_key = col

        for suffix in ("_mode", "_prev", "_next"):
            if translation_key.endswith(suffix):
                translation_key = translation_key[:-len(suffix)]
                break

        translation_map = translations.get(translation_key, {})

        labels = [
            translation_map.get(label, label)
            for label in labels
        ]

    values = final_df.values

    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(8.8, 5.8))

    color = "#0B3C5D"

    bars = ax.barh(
        y,
        values,
        color=color,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)

    for bar in bars:

        if language == "en":
            value_text = f" ₺ {bar.get_width():,.0f}"
        else:
            value_text = f" ₺ {bar.get_width():,.0f}"

        ax.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            value_text,
            va="center",
            ha="left",
            fontsize=8,
        )

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(left=False, bottom=False)
    ax.set_xticks([])
    ax.grid(False)

    if language == "en":
        legend_label = "Total" if agg_type == "sum" else "Average"
        legend_text = f"{legend_label} Donation Amount"
    else:
        legend_label = "Toplam" if agg_type == "sum" else "Ortalama"
        legend_text = f"{legend_label} Bağış Tutarı"

    ax.legend(
        [legend_text],
        frameon=False,
        fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(
        out_path,
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close()

# Province Map - Total Donations
def chart_province_total(
    out_path,
    df,
    il_col="Il",
    amount_col="Tutar",
    language="tr",
):
    df = df.copy()
    df[amount_col] = pd.to_numeric(df[amount_col], errors="coerce").fillna(0)

    df_plot = df.groupby(il_col)[amount_col].sum().reset_index()
    df_plot[il_col] = (
        df_plot[il_col]
        .str.upper()
        .str.replace("İ", "I")
        .str.strip()
    )

    gdf = gpd.read_file(TR_CITIES)
    gdf["name"] = (
        gdf["name"]
        .str.upper()
        .str.replace("İ", "I")
        .str.strip()
    )

    merged = gdf.merge(df_plot, left_on="name", right_on=il_col, how="left")
    merged[amount_col] = merged[amount_col].fillna(0)

    fig, ax = plt.subplots(1, 1, figsize=(15, 9))

    merged.plot(
        column=amount_col,
        cmap="YlGnBu",
        linewidth=0.5,
        ax=ax,
        edgecolor="0.4",
        scheme="Quantiles",
        k=5,
        legend=True,
        legend_kwds={"loc": "lower right", "frameon": True},
    )

    if language == "en":
        labels = [
            "Low – Provinces with lowest total donations",
            "Below Average – Provinces with low/medium total donations",
            "Average – Provinces with moderate total donations",
            "High – Provinces with high total donations",
            "Very High – Provinces with highest total donations",
        ]
        legend_title = "Donation Performance"
        stats_title = "Top 5 Provinces by Total Donations"

    else:
        labels = [
            "Düşük – En düşük toplam bağış alan iller",
            "Orta Altı – Düşük/orta toplam bağış alan iller",
            "Orta – Orta düzey toplam bağış alan iller",
            "Yüksek – Yüksek toplam bağış alan iller",
            "Çok Yüksek – En yüksek toplam bağış alan iller",
        ]
        legend_title = "Bağış Performansı"
        stats_title = "Toplam Bağış Tutarına Göre İlk 5 İl"

    legend = ax.get_legend()

    if legend:
        for i, text in enumerate(legend.get_texts()):
            if i < len(labels):
                text.set_text(labels[i])

        legend.set_title(legend_title)
        legend.set_bbox_to_anchor((1.02, -0.25))

    top_5 = df_plot.nlargest(5, amount_col)

    top_list = "\n".join(
        f"• {row[il_col]}: {row[amount_col]:,.0f} TL"
        for _, row in top_5.iterrows()
    )

    stats_text = f"{stats_title}:\n{top_list}"

    ax.annotate(
        stats_text,
        xy=(0.05, -0.001),
        xycoords="axes fraction",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9),
        ha="left",
        va="top",
    )

    ax.axis("off")

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# Province Map - Average Donations
def chart_province_average(
    out_path,
    df,
    il_col="Il",
    amount_col="Tutar",
    language="tr",
):
    df = df.copy()
    df[amount_col] = pd.to_numeric(df[amount_col], errors="coerce").fillna(0)

    df_plot = df.groupby(il_col)[amount_col].mean().reset_index()
    df_plot[il_col] = (
        df_plot[il_col]
        .str.upper()
        .str.replace("İ", "I")
        .str.strip()
    )

    gdf = gpd.read_file(TR_CITIES)
    gdf["name"] = (
        gdf["name"]
        .str.upper()
        .str.replace("İ", "I")
        .str.strip()
    )

    merged = gdf.merge(df_plot, left_on="name", right_on=il_col, how="left")
    merged[amount_col] = merged[amount_col].fillna(0)

    fig, ax = plt.subplots(1, 1, figsize=(15, 9))

    merged.plot(
        column=amount_col,
        cmap="YlGnBu",
        linewidth=0.5,
        ax=ax,
        edgecolor="0.4",
        scheme="Quantiles",
        k=5,
        legend=True,
        legend_kwds={"loc": "lower right", "frameon": True},
    )

    if language == "en":
        labels = [
            "Low – Provinces with lowest average donations",
            "Below Average – Provinces with low/medium average donations",
            "Average – Provinces with moderate average donations",
            "High – Provinces with high average donations",
            "Very High – Provinces with highest average donations",
        ]
        legend_title = "Donation Performance"
        stats_title = "Top 5 Provinces by Average Donation"

    else:
        labels = [
            "Düşük – En düşük ortalama bağış alan iller",
            "Orta Altı – Düşük/orta ortalama bağış alan iller",
            "Orta – Orta düzey ortalama bağış alan iller",
            "Yüksek – Yüksek ortalama bağış alan iller",
            "Çok Yüksek – En yüksek ortalama bağış alan iller",
        ]
        legend_title = "Bağış Performansı"
        stats_title = "Ortalama Bağış Tutarına Göre İlk 5 İl"

    legend = ax.get_legend()

    if legend:
        for i, text in enumerate(legend.get_texts()):
            if i < len(labels):
                text.set_text(labels[i])

        legend.set_title(legend_title)
        legend.set_bbox_to_anchor((1.02, -0.25))

    top_5 = df_plot.nlargest(5, amount_col)

    top_list = "\n".join(
        f"• {row[il_col]}: {row[amount_col]:,.0f} TL"
        for _, row in top_5.iterrows()
    )

    stats_text = f"{stats_title}:\n{top_list}"

    ax.annotate(
        stats_text,
        xy=(0.05, -0.001),
        xycoords="axes fraction",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9),
        ha="left",
        va="top",
    )

    ax.axis("off")

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

# Churn eligibility bar
def chart_churn_uygunluk_bari(
    out_path,
    data,
    risk_column="Terk Riski",
    language="tr",
):

    df = data.copy()

    uygun = (df[risk_column] != "Terk Riski Yok").sum()
    uygun_degil = (df[risk_column] == "Terk Riski Yok").sum()

    total = uygun + uygun_degil

    if language == "en":
        category = ["Donors"]
        labels = {
            "eligible": "Eligible",
            "not": "Not Eligible",
        }
    else:
        category = ["Bağışçılar"]
        labels = {
            "eligible": "Analize Uygun",
            "not": "Analize Uygun Değil",
        }

    colors = {
        "eligible": "#0B3C5D",
        "not": "#CCCCCC",
    }

    fig, ax = plt.subplots(figsize=(8, 2))

    ax.barh(
        category,
        uygun,
        color=colors["eligible"],
        label=labels["eligible"],
    )

    ax.barh(
        category,
        uygun_degil,
        left=uygun,
        color=colors["not"],
        label=labels["not"],
    )

    def add_label(start, value, color):

        if value == 0:
            return

        pct = value / total * 100 if total else 0

        ax.text(
            start + value / 2,
            0,
            f"{value:,}\n({pct:.1f}%)",
            ha="center",
            va="center",
            fontsize=9,
            color=color,
        )

    add_label(0, uygun, "white")
    add_label(uygun, uygun_degil, "black")

    ax.set_xlim(0, total * 1.05 if total else 1)
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.legend(
        frameon=False,
        fontsize=9,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.5),
        ncol=2,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close()

# Top Churn Risk
def chart_top_churn_risk(
    out_path,
    data,
    risk_column="Terk Riski",
    donor_column="Bagisci No",
    top_n=10,
    language="tr",
):

    df = data.copy()

    df = df[df[risk_column] != "Terk Riski Yok"].copy()

    if df.empty:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(
            0.5,
            0.5,
            "No eligible donors for churn analysis."
            if language == "en"
            else "Terk analizi için uygun bağışçı bulunmamaktadır.",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.axis("off")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        return

    df["risk_numeric"] = (
        df[risk_column]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

    top = (
        df.sort_values("risk_numeric", ascending=False)
        .head(top_n)
        .sort_values("risk_numeric", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    if language == "en":
        ax.set_title(
            f"Top {top_n} Donors with Highest Churn Risk",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )
    else:
        ax.set_title(
            f"En Yüksek Terk Riskine Sahip İlk {top_n} Bağışçı",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )

    color = "#0B3C5D"

    bars = ax.barh(
        top[donor_column].astype(str),
        top["risk_numeric"],
        color=color,
        height=0.65,
    )

    max_val = top["risk_numeric"].max()

    for bar, value in zip(bars, top["risk_numeric"]):
        ax.text(
            value + max_val * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"%{value:.2f}",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])

    ax.tick_params(axis="y", length=0, labelsize=10)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.grid(False)
    ax.margins(y=0.08)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

# Pareto Analysis
def chart_rfm_revenue_pareto(
    out_path,
    data,
    segment_column="Final_Segment_next",
    revenue_column="Tutar_sum",
    dpi=600,
    language="tr",
    translations=TRANSLATIONS,
):
    data = data.copy()

    revenue = (
        data.groupby(segment_column)[revenue_column]
        .sum()
        .sort_values(ascending=False)
    )

    total_revenue = revenue.sum()
    percentage_share = (revenue / total_revenue) * 100

    revenue = revenue.iloc[::-1]
    percentage_share = percentage_share.iloc[::-1]

    # --------------------------
    # TRANSLATIONS
    # --------------------------

    labels = revenue.index.tolist()

    if language == "en" and translations is not None:

        translation_key = segment_column

        for suffix in ("_mode", "_prev", "_next"):
            if translation_key.endswith(suffix):
                translation_key = translation_key[:-len(suffix)]
                break

        translation_map = translations.get(translation_key, {})

        labels = [
            translation_map.get(label, label)
            for label in labels
        ]

    num_segments = len(revenue)

    h = max(num_segments * 0.35, 1.5)
    fig, ax = plt.subplots(figsize=(10, h))

    bars = ax.barh(
        labels,
        revenue.values,
        color="#0B3C5D",
        alpha=0.9,
        height=0.8,
    )

    for i, bar in enumerate(bars):

        width = bar.get_width()
        pct = percentage_share.iloc[i]

        if language == "en":
            if width >= 1_000_000:
                value_text = f"{width/1_000_000:.1f}M TRY"
            elif width >= 1_000:
                value_text = f"{width/1_000:.0f}K TRY"
            else:
                value_text = f"{width:.0f} TRY"

            pct_text = f"{pct:.1f}%"
        else:
            if width >= 1_000_000:
                value_text = f"{width/1_000_000:.1f}M TL"
            elif width >= 1_000:
                value_text = f"{width/1_000:.0f} Bin TL"
            else:
                value_text = f"{width:.0f} TL"

            pct_text = f"%{pct:.1f}"

        ax.text(
            width + (total_revenue * 0.005),
            bar.get_y() + bar.get_height() / 2,
            value_text,
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#0B3C5D",
        )

        if pct > 3:
            ax.text(
                width / 2,
                bar.get_y() + bar.get_height() / 2,
                pct_text,
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                fontweight="bold",
            )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    ax.get_xaxis().set_visible(False)
    ax.tick_params(axis="y", labelsize=9, length=0)

    plt.subplots_adjust(
        left=0.2,
        right=0.9,
        top=0.95,
        bottom=0.05,
    )

    plt.savefig(
        out_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
    )

    plt.close()

# Segment Mobility Bar
def chart_segment_mobility_bar(
    out_path,
    data,
    prev_col="Davranis_Segmenti_prev",
    next_col="Davranis_Segmenti_next",
    language="tr",
):
    df = data.dropna(subset=[prev_col, next_col]).copy()

    segment_rank = {
        "Şampiyonlar": 1,
        "Sadık Bağışçılar": 2,
        "Yeni Bağışçılar": 3,
        "Potansiyel Sadıklar": 4,
        "İlgi Gerektiren": 5,
        "Uykuda": 6,
        "Kayıp Riskli": 7,
    }

    df["Rank_prev"] = df[prev_col].map(segment_rank)
    df["Rank_next"] = df[next_col].map(segment_rank)
    df = df.dropna(subset=["Rank_prev", "Rank_next"])

    def determine_mobility(row):
        if row["Rank_next"] < row["Rank_prev"]:
            return "Improving"
        elif row["Rank_next"] > row["Rank_prev"]:
            return "Declining"
        else:
            return "Stable"

    df["Mobility"] = df.apply(determine_mobility, axis=1)

    ser = df["Mobility"].value_counts()

    improving = ser.get("Improving", 0)
    stable = ser.get("Stable", 0)
    declining = ser.get("Declining", 0)

    total = improving + stable + declining

    if language == "en":
        categories = ["Donors"]
        improving_label = "Improving"
        stable_label = "Stable"
        declining_label = "Declining"

        if total > 0:
            summary_text = (
                f"Improving: {improving:,} ({(improving/total*100):.1f}%)\n"
                f"Stable: {stable:,} ({(stable/total*100):.1f}%)\n"
                f"Declining: {declining:,} ({(declining/total*100):.1f}%)"
            )
        else:
            summary_text = (
                f"Improving: {improving:,} (0.0%)\n"
                f"Stable: {stable:,} (0.0%)\n"
                f"Declining: {declining:,} (0.0%)"
            )
    else:
        categories = ["Bağışçılar"]
        improving_label = "İyileşen"
        stable_label = "Sabit"
        declining_label = "Gerileyen"

        if total > 0:
            summary_text = (
                f"İyileşen: {improving:,} ({(improving/total*100):.1f}%)\n"
                f"Sabit: {stable:,} ({(stable/total*100):.1f}%)\n"
                f"Gerileyen: {declining:,} ({(declining/total*100):.1f}%)"
            )
        else:
            summary_text = (
                f"İyileşen: {improving:,} (0.0%)\n"
                f"Sabit: {stable:,} (0.0%)\n"
                f"Gerileyen: {declining:,} (0.0%)"
            )

    colors = {
        "Improving": "#0B3C5D",
        "Stable": "#F28E2B",
        "Declining": "#CCCCCC",
    }

    fig, ax = plt.subplots(figsize=(8, 2))

    ax.barh(categories, improving, color=colors["Improving"], label=improving_label)
    ax.barh(categories, stable, left=improving, color=colors["Stable"], label=stable_label)
    ax.barh(
        categories,
        declining,
        left=improving + stable,
        color=colors["Declining"],
        label=declining_label,
    )

    ax.text(
        total * 1.02,
        0,
        summary_text,
        va="center",
        ha="left",
        fontsize=9,
    )

    ax.set_xlim(0, total * 1.35)
    ax.set_yticks([])
    ax.set_xticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.legend(
        frameon=False,
        fontsize=9,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.5),
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close()

# Specific Segment Transition
def chart_specific_segment_transition(
    out_path,
    data,
    target_prev_segment="Loyal Donors",
    prev_col="Davranis_Segmenti_prev",
    next_col="Davranis_Segmenti_next",
    dpi=200,
    language="tr",
    translations=TRANSLATIONS,
):

    df = data[data[prev_col] == target_prev_segment].copy()

    # --------------------------
    # TRANSLATIONS
    # --------------------------

    display_target_segment = target_prev_segment
    translation_map = {}

    if language == "en" and translations is not None:

        translation_key = next_col

        for suffix in ("_mode", "_prev", "_next"):
            if translation_key.endswith(suffix):
                translation_key = translation_key[:-len(suffix)]
                break

        translation_map = translations.get(translation_key, {})
        display_target_segment = translation_map.get(
            target_prev_segment,
            target_prev_segment,
        )

    if df.empty:

        fig, ax = plt.subplots(figsize=(6, 2))

        if language == "en":
            no_data_text = f"No data for {display_target_segment}."
        else:
            no_data_text = f"{target_prev_segment} için veri bulunamadı."

        ax.text(0.5, 0.5, no_data_text, ha="center")

        plt.savefig(out_path, dpi=dpi)
        plt.close()
        return

    transition_counts = df[next_col].value_counts().sort_values(ascending=True)
    total = transition_counts.sum()

    labels = transition_counts.index.tolist()

    if language == "en":
        labels = [
            translation_map.get(label, label)
            for label in labels
        ]

    fig, ax = plt.subplots(figsize=(8, max(len(transition_counts) * 0.5, 3)))

    colors = [
        "#5DA9E9" if seg != target_prev_segment else "#0B3C5D"
        for seg in transition_counts.index
    ]

    bars = ax.barh(
        labels,
        transition_counts.values,
        color=colors,
        height=0.6,
    )

    for bar in bars:

        width = bar.get_width()
        pct = (width / total) * 100

        if language == "en":
            label = f"{int(width):,} donors ({pct:.1f}%)"
        else:
            label = f"{int(width):,} bağışçı (%{pct:.1f})"

        ax.text(
            width + (total * 0.01),
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            fontsize=9,
            color="#333333",
        )

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(left=False, bottom=False)
    ax.set_xticks([])

    plt.tight_layout()
    plt.savefig(
        out_path,
        dpi=dpi,
        bbox_inches="tight",
        transparent=True,
    )
    plt.close()

# Second donation performance
# def chart_second_donation_performance(
#     out_path,
#     raw_data,
#     dpi=600,
#     language="tr",
# ):
#     df = raw_data[["Bagisci No", "Bagis Tarihi"]].dropna().copy()

#     df["Bagis Tarihi"] = pd.to_datetime(df["Bagis Tarihi"])
#     df = df.sort_values(by=["Bagisci No", "Bagis Tarihi"])
#     df["Donation_Rank"] = df.groupby("Bagisci No").cumcount() + 1

#     first = df[df["Donation_Rank"] == 1].set_index("Bagisci No")["Bagis Tarihi"]
#     second = df[df["Donation_Rank"] == 2].set_index("Bagisci No")["Bagis Tarihi"]

#     donor_journey = pd.DataFrame({
#         "First": first,
#         "Second": second
#     })

#     donor_journey["Days_Between"] = (
#         donor_journey["Second"] - donor_journey["First"]
#     ).dt.days

#     total = len(donor_journey)
#     second_total = donor_journey["Second"].notna().sum()
#     second_60 = (donor_journey["Days_Between"] <= 60).sum()

#     if language == "en":
#         total_first_label = "Total First-Time\nDonors"
#         repeat_label = "Repeat\nDonors"
#         repeat_60_label = "Repeat Within\n60 Days"
#     else:
#         total_first_label = "Toplam İlk Kez\nBağış Yapanlar"
#         repeat_label = "Tekrar\nBağış Yapanlar"
#         repeat_60_label = "60 Gün İçinde\nTekrar Bağış Yapanlar"

#     fig = plt.figure(figsize=(9, 3.5))

#     ax = fig.add_axes([0.05, 0.05, 0.9, 0.9])
#     ax.axis("off")

#     ax.text(
#         0.15,
#         0.6,
#         f"{total:,}",
#         fontsize=22,
#         fontweight="bold",
#         ha="center",
#     )
#     ax.text(
#         0.15,
#         0.3,
#         total_first_label,
#         ha="center",
#     )

#     ax.text(
#         0.5,
#         0.6,
#         f"{second_total:,}",
#         fontsize=22,
#         fontweight="bold",
#         ha="center",
#     )
#     ax.text(
#         0.5,
#         0.3,
#         repeat_label,
#         ha="center",
#     )

#     ax.text(
#         0.85,
#         0.6,
#         f"{second_60:,}",
#         fontsize=22,
#         fontweight="bold",
#         ha="center",
#     )
#     ax.text(
#         0.85,
#         0.3,
#         repeat_60_label,
#         ha="center",
#     )

#     plt.savefig(out_path, dpi=dpi, bbox_inches="tight", transparent=True)
#     plt.close()

def chart_second_donation_performance(
    out_path,
    raw_data,
    dpi=600,
    language="tr",
):
    df = raw_data[["Bagisci No", "Bagis Tarihi"]].dropna().copy()

    df["Bagis Tarihi"] = pd.to_datetime(df["Bagis Tarihi"])
    df = df.sort_values(by=["Bagisci No", "Bagis Tarihi"])
    df["Donation_Rank"] = df.groupby("Bagisci No").cumcount() + 1

    first = df[df["Donation_Rank"] == 1].set_index("Bagisci No")["Bagis Tarihi"]
    second = df[df["Donation_Rank"] == 2].set_index("Bagisci No")["Bagis Tarihi"]

    donor_journey = pd.DataFrame({
        "First": first,
        "Second": second,
    })

    donor_journey["Days_Between"] = (
        donor_journey["Second"] - donor_journey["First"]
    ).dt.days

    total = len(donor_journey)
    second_total = donor_journey["Second"].notna().sum()
    second_60 = (donor_journey["Days_Between"] <= 60).sum()

    if language == "en":
        total_first_label = "Total First-Time\nDonors"
        repeat_label = "Repeat\nDonors"
        repeat_60_label = "Repeat Within\n60 Days"
    else:
        total_first_label = "Toplam İlk Kez\nBağış Yapanlar"
        repeat_label = "Tekrar\nBağış Yapanlar"
        repeat_60_label = "60 Gün İçinde\nTekrar Bağış Yapanlar"

    fig = plt.figure(figsize=(9, 2.0))

    # Sol
    fig.text(
        0.17,
        0.62,
        f"{total:,}",
        fontsize=22,
        fontweight="bold",
        ha="center",
        va="center",
    )
    fig.text(
        0.17,
        0.22,
        total_first_label,
        fontsize=11,
        ha="center",
        va="center",
    )

    # Orta
    fig.text(
        0.50,
        0.62,
        f"{second_total:,}",
        fontsize=22,
        fontweight="bold",
        ha="center",
        va="center",
    )
    fig.text(
        0.50,
        0.22,
        repeat_label,
        fontsize=11,
        ha="center",
        va="center",
    )

    # Sağ
    fig.text(
        0.83,
        0.62,
        f"{second_60:,}",
        fontsize=22,
        fontweight="bold",
        ha="center",
        va="center",
    )
    fig.text(
        0.83,
        0.22,
        repeat_60_label,
        fontsize=11,
        ha="center",
        va="center",
    )

    plt.savefig(
        out_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.02,
        transparent=True,
    )
    plt.close(fig)

# Days to Second Donation Violin
def chart_days_to_second_donation_violin(
    out_path,
    raw_data,
    dpi=200,
    max_days=720,
    language="tr",
):
    df = raw_data[
        ["Bagisci No", "Bagis Tarihi"]
    ].dropna(subset=["Bagisci No", "Bagis Tarihi"]).copy()

    df["Bagis Tarihi"] = pd.to_datetime(df["Bagis Tarihi"])
    df = df.drop_duplicates(subset=["Bagisci No", "Bagis Tarihi"], keep="first")
    df = df.sort_values(by=["Bagisci No", "Bagis Tarihi"])
    df["Donation_Rank"] = df.groupby("Bagisci No").cumcount() + 1

    first_donations = (
        df[df["Donation_Rank"] == 1]
        .set_index("Bagisci No")["Bagis Tarihi"]
        .rename("First_Date")
    )

    second_donations = (
        df[df["Donation_Rank"] == 2]
        .set_index("Bagisci No")["Bagis Tarihi"]
        .rename("Second_Date")
    )

    donor_journey = pd.concat(
        [first_donations, second_donations],
        axis=1,
        join="inner",
    )

    donor_journey["Days_Between"] = (
        donor_journey["Second_Date"] - donor_journey["First_Date"]
    ).dt.days

    if donor_journey.empty:
        fig, ax = plt.subplots(figsize=(6, 3))

        if language == "en":
            msg = "Insufficient second donation data."
        else:
            msg = "İkinci bağış için yeterli veri bulunamadı."

        ax.text(0.5, 0.5, msg, ha="center", fontsize=12)
        plt.savefig(out_path, dpi=dpi)
        plt.close()
        return

    if max_days is not None:
        donor_journey = donor_journey[
            donor_journey["Days_Between"] <= max_days
        ]

        if donor_journey.empty:
            fig, ax = plt.subplots(figsize=(6, 3))

            if language == "en":
                msg = f"No data within {max_days} days."
            else:
                msg = f"{max_days} gün içinde veri bulunamadı."

            ax.text(0.5, 0.5, msg, ha="center", fontsize=12)
            plt.savefig(out_path, dpi=dpi)
            plt.close()
            return

    if language == "en":
        title = "Days from 1st to 2nd Donation"
        ylabel = "Days Elapsed"
        avg_prefix = "Avg"
        median_prefix = "Med"
        category_label = "All Donors"
    else:
        title = "İlk Bağıştan İkinci Bağışa Geçen Gün Sayısı"
        ylabel = "Geçen Gün"
        avg_prefix = "Ort"
        median_prefix = "Medyan"
        category_label = "Tüm Bağışçılar"

    fig, ax = plt.subplots(figsize=(8, 6))

    plot_df = donor_journey.copy()
    plot_df["Category"] = category_label

    sns.violinplot(
        data=plot_df,
        x="Category",
        y="Days_Between",
        color="#0B3C5D",
        inner="quartile",
        cut=0,
        ax=ax,
    )

    mean_val = plot_df["Days_Between"].mean()
    median_val = plot_df["Days_Between"].median()

    ax.text(
        0,
        mean_val,
        f" {avg_prefix}: {mean_val:.0f} ",
        color="white",
        bbox=dict(
            facecolor="#333333",
            alpha=0.8,
            boxstyle="round,pad=0.3",
        ),
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )

    ax.text(
        0,
        median_val,
        f" {median_prefix}: {median_val:.0f} ",
        color="black",
        bbox=dict(
            facecolor="white",
            alpha=0.9,
            edgecolor="#cccccc",
            boxstyle="round,pad=0.3",
        ),
        ha="center",
        va="top",
        fontsize=9,
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, fontsize=11, fontweight="bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#DDDDDD")
    ax.spines["bottom"].set_color("#DDDDDD")

    ax.yaxis.grid(True, linestyle="--", alpha=0.6, color="#DDDDDD")
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(
        out_path,
        dpi=dpi,
        bbox_inches="tight",
        transparent=False,
        facecolor="white",
    )
    plt.close()


#========================================
# Column Check Helpers
#========================================

def _has_cols(df: pd.DataFrame, *cols: str) -> bool:
    if df is None:
        return False
    return all((c in df.columns) for c in cols)


def _any_cols(df: pd.DataFrame, *cols: str) -> bool:
    if df is None:
        return False
    return any((c in df.columns) for c in cols)


#========================================
# Main Function
#========================================

def generate_report(
    final_data: pd.DataFrame,
    duzenlenmis_data: pd.DataFrame,
    output_path: str,
    language: str = 'tr'
):
    """
    Generate PDF report from analysis results
    language: 'tr' for Turkish, 'en' for English
    """

    def para_donustur(x: float) -> str:
        return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

    pdf = DonorReport(language=language)

    # ==========================
    # COVER PAGE
    # ==========================
    pdf.is_cover_page = True
    pdf.add_page()

    pdf.ln(150)  # Başlığı aşağıya almak için isteğe göre ayarlanabilir.

    pdf.set_font("DejaVu", "B", 20)
    pdf.set_text_color(0, 0, 51)

    if language == 'en':
        pdf.cell(
            0, 20,
            "DONOR SEGMENTATION ANALYSIS REPORT",
            align="C",
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT
        )
    else:
        pdf.cell(
            0, 20,
            "BAĞIŞÇI SEGMENTASYON ANALİZİ RAPORU",
            align="C",
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT
        )

    pdf.set_font("DejaVu", "B", 16)

    # Attribution (cover page bottom)
    pdf.set_y(pdf.h - pdf.b_margin - 6)
    pdf.set_font("DejaVu", "", 6)
    pdf.set_text_color(140, 140, 140)
    pdf.cell(
        0,
        3,
        "Illustration: PNGTree - https://pngtree.com/freepng/business-data-analysis-through-annual-report-magnifying-glass_24380556.html",
        align="C",
    )

    pdf.is_cover_page = False
    pdf.add_page()


    # ========================================
    # GENERAL INFORMATION
    # ========================================

    if language == 'en':
        pdf.chapter_title("1 - General Information")
    else:
        pdf.chapter_title("1 - Genel Bilgiler")

    pdf.set_font("DejaVu", "", 11)
    pdf.set_text_color(0, 0, 0)

    # Report Date
    if language == 'en':
        pdf.info_box("Report Date", datetime.now().strftime("%d.%m.%Y"))
    else:
        pdf.info_box("Rapor Hazırlama Tarihi", datetime.now().strftime("%d.%m.%Y"))

    # Analysis Period
    if _has_cols(final_data, "Bagis Tarihi_min", "Bagis Tarihi_max"):
        baslangic = pd.to_datetime(final_data["Bagis Tarihi_min"].min()).strftime("%d.%m.%Y")
        bitis = pd.to_datetime(final_data["Bagis Tarihi_max"].max()).strftime("%d.%m.%Y")

        if language == "en":
            pdf.info_box("Analysis Period", f"{baslangic} / {bitis}")
        else:
            pdf.info_box("Analiz Dönemi", f"{baslangic} / {bitis}")

    # Total Donors
    if language == "en":
        pdf.info_box("Total Donors", len(final_data))
    else:
        pdf.info_box("Toplam Bağışçı", len(final_data))

    # Total Donations
    if _has_cols(final_data, "Bağış Sayısı"):
        toplam = int(final_data["Bağış Sayısı"].sum())

        if language == "en":
            pdf.info_box("Total Donations", f"{toplam:,}")
        else:
            pdf.info_box("Toplam Bağış Sayısı", f"{toplam:,}")

    # Total Donation Amount
    if _has_cols(final_data, "Tutar_sum"):
        toplam = final_data["Tutar_sum"].sum()

        if language == "en":
            pdf.info_box("Total Donation Amount", f"{para_donustur(toplam)} ₺")
        else:
            pdf.info_box("Toplam Bağış Tutarı", f"{para_donustur(toplam)} ₺")

    # Average Donation Count per Donor
    if _has_cols(final_data, "Bağış Sayısı"):
        ort = final_data["Bağış Sayısı"].mean()

        if language == "en":
            pdf.info_box("Average Donations per Donor", f"{ort:.2f}")
        else:
            pdf.info_box("Bağışçı Başına Ortalama Bağış Sayısı", f"{ort:.2f}")

    # Average Total Donation per Donor
    if _has_cols(final_data, "Tutar_sum"):
        ort = final_data["Tutar_sum"].mean()

        if language == "en":
            pdf.info_box("Average Total Donation per Donor", f"{para_donustur(ort)} ₺")
        else:
            pdf.info_box("Bağışçı Başına Ortalama Toplam Bağış Tutarı", f"{para_donustur(ort)} ₺")

    # Median Total Donation per Donor
    if _has_cols(final_data, "Tutar_sum"):
        medyan = final_data["Tutar_sum"].median()

        if language == "en":
            pdf.info_box("Median Total Donation per Donor", f"{para_donustur(medyan)} ₺")
        else:
            pdf.info_box("Bağışçı Başına Medyan Toplam Bağış Tutarı", f"{para_donustur(medyan)} ₺")

    # Highest Total Donation by a Donor
    if _has_cols(final_data, "Tutar_sum"):
        maksimum = final_data["Tutar_sum"].max()

        if language == "en":
            pdf.info_box("Highest Total Donation by a Donor", f"{para_donustur(maksimum)} ₺")
        else:
            pdf.info_box("Bir Bağışçının En Yüksek Toplam Bağış Tutarı", f"{para_donustur(maksimum)} ₺")

    # Repeat Donor Rate
    if _has_cols(final_data, "Bağış Sayısı"):
        tekrar = (final_data["Bağış Sayısı"] > 1).mean() * 100

        if language == "en":
            pdf.info_box("Repeat Donor Rate", f"%{tekrar:.1f}")
        else:
            pdf.info_box("Tekrar Bağışçı Oranı", f"%{tekrar:.1f}")

    # One-Time Donors
    if _has_cols(final_data, "Bağış Sayısı"):
        tek = (final_data["Bağış Sayısı"] == 1).sum()

        if language == "en":
            pdf.info_box("One-Time Donors", tek)
        else:
            pdf.info_box("Tek Sefer Bağış Yapan Bağışçı Sayısı", tek)

    pdf.ln(10)


    # ========================================
    # GENERAL STATISTICS
    # ========================================
    have_tur = _has_cols(duzenlenmis_data, "Tür")
    have_kanal = _has_cols(duzenlenmis_data, "Bagis Kanali")
    have_odeme = _has_cols(duzenlenmis_data, "Odeme Sekli")
    have_para = _has_cols(duzenlenmis_data, "Para Birimi")
    have_kampanya = _has_cols(duzenlenmis_data, "Kampanya Adi", "Toplam Tutar")
    have_il = _has_cols(duzenlenmis_data, "Il", "Toplam Tutar")

    have_general_stats = any([
        have_tur,
        have_kanal,
        have_odeme,
        have_para,
        have_kampanya,
        have_il
    ])

    if have_general_stats:
        if language == "en":
            pdf.chapter_title("2 - General Statistics")
        else:
            pdf.chapter_title("2 - Genel İstatistikler")

    # ----------------------------------------
    # Donation Type Distribution
    # ----------------------------------------
    if have_tur:
        if language == "en":
            pdf.sub_title("Donation Type Distribution")
        else:
            pdf.sub_title("Bağışların Tür Bazında Dağılımı")

        pdf.add_chart_from_func(
            chart_donut,
            column_name="Tür",
            data=duzenlenmis_data,
            w=145,
            language=language
        )

    # ----------------------------------------
    # Donation Channel Distribution
    # ----------------------------------------
    if have_kanal:
        pdf.add_page()
        if language == "en":
            pdf.sub_title("Donation Channel Distribution")
        else:
            pdf.sub_title("Bağış Kanallarının Dağılımı")

        pdf.add_chart_from_func(
            chart_donut,
            column_name="Bagis Kanali",
            data=duzenlenmis_data,
            w=145,
            language=language
        )

    # ----------------------------------------
    # Payment Method Distribution
    # ----------------------------------------
    if have_odeme:
        if language == "en":
            pdf.sub_title("Payment Method Distribution")
        else:
            pdf.sub_title("Ödeme Şekillerinin Dağılımı")

        pdf.add_chart_from_func(
            chart_donut,
            column_name="Odeme Sekli",
            data=duzenlenmis_data,
            w=145,
            language=language
        )

    # ----------------------------------------
    # Currency Distribution
    # ----------------------------------------
    if have_para:
        pdf.add_page()
        if language == "en":
            pdf.sub_title("Currency Distribution")
        else:
            pdf.sub_title("Para Birimlerinin Dağılımı")

        pdf.add_chart_from_func(
            chart_donut,
            column_name="Para Birimi",
            data=duzenlenmis_data,
            w=145,
            language=language
        )

    # ----------------------------------------
    # Top Campaigns
    # ----------------------------------------
    if have_kampanya:
        if language == "en":
            pdf.sub_title("Top Campaigns by Donation Amount")
        else:
            pdf.sub_title("En Çok Bağış Alan Kampanyalar")

        pdf.add_chart_from_func(
            chart_bar,
            data=duzenlenmis_data,
            col="Kampanya Adi",
            agg_type="sum",
            language=language
        )

    # ----------------------------------------
    # Provinces
    # ----------------------------------------
    if have_il:
        pdf.add_page()

        if language == "en":
            pdf.sub_title("Provinces by Total Donations")
        else:
            pdf.sub_title("İllerin Toplam Bağışa Göre Sınıflandırılması")

        pdf.add_chart_from_func(
            chart_province_total,
            duzenlenmis_data,
            language=language,
            w=175
        )

        if language == "en":
            pdf.sub_title("Provinces by Average Donations")
        else:
            pdf.sub_title("İllerin Ortalama Bağışa Göre Sınıflandırılması")

        pdf.add_chart_from_func(
            chart_province_average,
            duzenlenmis_data,
            language=language,
            w=175
        )

    # ========================================
    # CHURN ANALYSIS
    # ========================================
    have_churn = _has_cols(final_data, "Terk Riski")

    if have_churn:
        pdf.add_page()

        if language == "en":
            pdf.chapter_title("3 - Churn Analysis")
        else:
            pdf.chapter_title("3 - Terk (Churn) Analizi")

        # ----------------------------------------
        # Eligible / Not Eligible Donors
        # ----------------------------------------

        if language == "en":
            pdf.sub_title("Eligible Donors for Churn Analysis")

            uygunluk_metni =dedent("""
    This chart shows the distribution of donors eligible for churn analysis.

    Donors with a predicted churn probability are considered eligible for analysis.
    Donors labeled as "No Churn Risk" are not included because there is insufficient donation history to calculate a reliable churn prediction.
            """).strip()

        else:
            pdf.sub_title("Analize Uygun Bağışçıların Dağılımı")

            uygunluk_metni = dedent("""
    Bu grafik, terk (churn) analizi kapsamında analize uygun olan ve olmayan bağışçıların dağılımını göstermektedir.

    Terk riski hesaplanabilen bağışçılar analize uygun kabul edilir. "Terk Riski Yok" olarak işaretlenen bağışçılar ise yeterli bağış geçmişi bulunmadığı için terk analizi kapsamına alınmamıştır.
            """).strip()

        pdf.paragraph(uygunluk_metni, indent=1)

        pdf.add_chart_from_func(
            chart_churn_uygunluk_bari,
            data=final_data,
            language=language,
            align="C",
            h=40,
        )

        # ----------------------------------------
        # Highest Churn Risk Donors
        # ----------------------------------------

        if language == "en":
            pdf.sub_title("Top Donors with the Highest Churn Risk")

            risk_metni = dedent("""
        The chart highlights the donors with the highest predicted churn risk.

        These donors represent the highest priority group for retention efforts. Early engagement through personalized communication, follow-up activities, or targeted campaigns may help reduce the likelihood of donor attrition.

        Only donors eligible for churn analysis are included.
            """).strip()

        else:
            pdf.sub_title("En Yüksek Terk Riskine Sahip Bağışçılar")

            risk_metni = dedent("""
        Grafik, tahmini terk riski en yüksek olan bağışçıları göstermektedir.

        Bu bağışçılar, elde tutma çalışmaları açısından öncelikli hedef grubu oluşturmaktadır. Kişiselleştirilmiş iletişim, takip çalışmaları ve uygun kampanyalar ile bağışçı kaybı riski azaltılabilir.

        "Terk Riski Yok" olarak işaretlenen bağışçılar bu analize dahil edilmemiştir.
            """).strip()

        pdf.paragraph(risk_metni, indent=1, size=10)

        pdf.add_chart_from_func(
            chart_top_churn_risk,
            data=final_data,
            language=language,
            donor_column="Ad Soyad",
            top_n=5,
            w=170,
            h=85
        )

    # ========================================
    # RFM ANALYSIS
    # ========================================
    have_davranis_next = _has_cols(final_data, "Davranis_Segmenti_next")
    have_deger_next = _has_cols(final_data, "Deger_Segmenti_next")
    have_final_next = _has_cols(final_data, "Final_Segment_next")
    have_tutar_sum = _has_cols(final_data, "Tutar_sum")
    have_prev_next_davranis = _has_cols(final_data, "Davranis_Segmenti_prev", "Davranis_Segmenti_next")


    have_davranis_avg = have_davranis_next and have_tutar_sum
    have_deger_avg = have_deger_next and have_tutar_sum
    have_pareto_davranis = have_davranis_next and have_tutar_sum
    have_pareto_deger = have_deger_next and have_tutar_sum
    have_pareto_final = have_final_next and have_tutar_sum

    have_rfm_section = any([
        have_davranis_next, have_deger_avg, have_deger_next, have_davranis_avg,
        have_pareto_davranis, have_pareto_deger, have_pareto_final, have_prev_next_davranis
    ])

    if have_rfm_section:
        pdf.add_page()

        if language == 'en':
            pdf.chapter_title("4 - RFM Segment Analysis")
        else:
            pdf.chapter_title("4 - RFM Segment Analizi")

    if have_davranis_next:
        if language == 'en':
            pdf.sub_title("Behavioral Segment Distribution")
        else:
            pdf.sub_title("Davranış Segmentlerinin Dağılımı")

        pdf.add_chart_from_func(
            chart_donut,
            data=final_data,
            column_name='Davranis_Segmenti_next',
            w=140,
            language=language
        )

    if have_davranis_avg:
        if language == 'en':
            pdf.sub_title("Behavioral Segments - Average Donation Amounts")
        else:
            pdf.sub_title("Davranış Segmentlerinin Ortalama Bağış Tutarları")

        pdf.add_chart_from_func(
            chart_bar,
            data=final_data,
            col="Davranis_Segmenti_next",
            agg_type="mean",
            try_col="Tutar_sum",
            language=language
        )

    if have_deger_next or have_deger_avg:
        pdf.add_page()

    if have_deger_next:
        if language == 'en':
            pdf.sub_title("Value Segment Distribution")
        else:
            pdf.sub_title("Değer Segmentlerinin Dağılımı")

        pdf.add_chart_from_func(
            chart_donut,
            data=final_data,
            column_name='Deger_Segmenti_next',
            w=140,
            language=language
        )

    if have_deger_avg:
        if language == 'en':
            pdf.sub_title("Value Segments - Average Donation Amounts")
        else:
            pdf.sub_title("Değer Segmentlerinin Ortalama Bağış Tutarları")

        pdf.add_chart_from_func(
            chart_bar,
            data=final_data,
            col="Deger_Segmenti_next",
            agg_type="mean",
            try_col="Tutar_sum",
            language=language
        )

    if have_pareto_davranis or have_pareto_deger:
        pdf.add_page()

        if language == 'en':
            pareto_text = dedent("""
Each horizontal bar represents the total revenue from a segment, while the percentages inside bars show each segment's share of total revenue.
        """).strip()

            pdf.sub_title("Segment Revenue Distribution and Revenue Shares")
        else:
            pareto_text = dedent("""
Her bir yatay bar bir segmentin getirdiği toplam tutarı temsil ederken, bar içindeki yüzdeler o segmentin tek başına cirodaki payını ifade eder.
        """).strip()

            pdf.sub_title("Segment Bazlı Gelir Dağılımı ve Ciro Payları")

        pdf.paragraph(pareto_text, indent=1, size=8)
        pdf.ln(3)

    if have_pareto_davranis:
        if language == 'en':
            pdf.sub_title("Behavioral Segments")
        else:
            pdf.sub_title("Davranış Segmentleri")

        pdf.ln(5)

        pdf.add_chart_from_func(
            chart_rfm_revenue_pareto,
            data=final_data,
            segment_column="Davranis_Segmenti_next",
            h=50,
            language=language
        )

    if have_pareto_deger:
        pdf.ln(3)

        if language == 'en':
            pdf.sub_title("Value Segments")
        else:
            pdf.sub_title("Değer Segmentleri")

        pdf.add_chart_from_func(
            chart_rfm_revenue_pareto,
            data=final_data,
            segment_column="Deger_Segmenti_next",
            language=language
        )

    if have_pareto_final:
        pdf.add_page()

        if language == 'en':
            pdf.sub_title("Final Segments")
        else:
            pdf.sub_title("Final Segmentler")

        pdf.add_chart_from_func(
            chart_rfm_revenue_pareto,
            data=final_data,
            segment_column="Final_Segment_next",
            language=language
        )

    if have_prev_next_davranis:

        if language == 'en':
            pdf.sub_title("Segment Mobility This Quarter (Improving / Declining)")
            pdf.paragraph(dedent("The chart below summarizes donor movements according to the strategy hierarchy.").strip(), indent=1)
        else:
            pdf.sub_title("Bu Çeyrekte Segment Hareketliliği (Yükselen / Düşen)")
            pdf.paragraph(dedent("Aşağıdaki grafik, strateji dokümanında belirtilen hiyerarşiye göre bağışçıların yukarı yönlü veya aşağı yönlü hareketlerini özetlemektedir.").strip(), indent=1)
        pdf.add_chart_from_func(
            chart_segment_mobility_bar,
            data=final_data,
            h=40,
            language=language
        )

    if have_prev_next_davranis:
        pdf.add_page()

        if language == "en":
            pdf.sub_title("Segment Transition Analysis: Previous Quarter vs Current Quarter")

            pdf.paragraph(
                dedent("""
                    The chart below shows how donors who were classified as Champions in the previous
                    quarter transitioned across behavioral segments in the current quarter, including
                    how many remained in the Champions segment.
                """).strip(),
                indent=1,
            )

        else:
            pdf.sub_title("Segment Geçiş Analizi: Bir Önceki Çeyrekten Bu Çeyreğe Davranış Değişimi")

            pdf.paragraph(
                dedent("""
                    Aşağıdaki grafik, geçen çeyrekte Şampiyon segmentinde yer alan bağışçıların bu
                    çeyrekte hangi davranış segmentlerine geçtiğini ve ne kadarının Şampiyon
                    segmentinde kaldığını göstermektedir.
                """).strip(),
                indent=1,
            )

        pdf.add_chart_from_func(
            chart_specific_segment_transition,
            data=final_data,
            target_prev_segment="Şampiyonlar",
            h=55,
            language=language
        )

        if language == "en":
            pdf.paragraph(
                dedent("""
                    The chart below shows how donors who were classified as Loyal Donors in the
                    previous quarter transitioned across behavioral segments in the current
                    quarter, including those who were promoted to Champions, remained Loyal
                    Donors, or moved to lower-value segments.
                """).strip(),
                indent=1,
            )

        else:
            pdf.paragraph(
                dedent("""
                    Aşağıdaki grafik, geçen çeyrekte Sadık Bağışçı segmentinde yer alan
                    bağışçıların bu çeyrekte hangi davranış segmentlerine geçtiğini; Şampiyon
                    segmentine yükselenleri, Sadık Bağışçı olarak kalanları ve alt segmentlere
                    geçenleri göstermektedir.
                """).strip(),
                indent=1,
            )

        pdf.add_chart_from_func(
            chart_specific_segment_transition,
            data=final_data,
            target_prev_segment="Sadık Bağışçılar",
            h=55,
            language=language
        )

        pdf.add_page()

        if language == "en":
            pdf.paragraph(
                dedent("""
                    The chart below shows how donors who were classified as Potential Loyalists
                    in the previous quarter transitioned across behavioral segments in the
                    current quarter.
                """).strip(),
                indent=1,
            )
        else:
            pdf.paragraph(
                dedent("""
                    Aşağıdaki grafik, geçen çeyrekte Potansiyel Sadıklar segmentinde yer alan
                    bağışçıların bu çeyrekte hangi davranış segmentlerine geçtiğini
                    göstermektedir.
                """).strip(),
                indent=1,
            )

        pdf.add_chart_from_func(
            chart_specific_segment_transition,
            data=final_data,
            target_prev_segment="Potansiyel Sadıklar",
            h=55,
            language=language,
        )



        if language == "en":
            pdf.paragraph(
                dedent("""
                    The chart below shows how donors who were classified as Dormant in the
                    previous quarter transitioned across behavioral segments in the current
                    quarter.
                """).strip(),
                indent=1,
            )
        else:
            pdf.paragraph(
                dedent("""
                    Aşağıdaki grafik, geçen çeyrekte Uykuda segmentinde yer alan bağışçıların
                    bu çeyrekte hangi davranış segmentlerine geçtiğini göstermektedir.
                """).strip(),
                indent=1,
            )

        pdf.add_chart_from_func(
            chart_specific_segment_transition,
            data=final_data,
            target_prev_segment="Uykuda",
            h=55,
            language=language,
        )

        if language == "en":
            pdf.paragraph(
                dedent("""
                    The chart below shows how donors who were classified as At Risk in the
                    previous quarter transitioned across behavioral segments in the current
                    quarter.
                """).strip(),
                indent=1,
            )
        else:
            pdf.paragraph(
                dedent("""
                    Aşağıdaki grafik, geçen çeyrekte Kayıp Riskli segmentinde yer alan
                    bağışçıların bu çeyrekte hangi davranış segmentlerine geçtiğini
                    göstermektedir.
                """).strip(),
                indent=1,
            )

        pdf.add_chart_from_func(
            chart_specific_segment_transition,
            data=final_data,
            target_prev_segment="Kayıp Riskli",
            h=55,
            language=language,
        )


    have_second_donation = _has_cols(duzenlenmis_data, "Bagisci No", "Bagis Tarihi")
    have_violin = _has_cols(duzenlenmis_data, "Bagisci No", "Bagis Tarihi", "Tür")

    if have_second_donation:
        pdf.add_page()

        if language == 'en':
            pdf.sub_title("First to Repeat Donation Performance")
        else:
            pdf.sub_title("İlk Bağıştan Tekrar Bağışa Geçiş Performansı")

        pdf.add_chart_from_func(
            chart_second_donation_performance,
            raw_data=duzenlenmis_data,
            language=language,
            approx_ratio=0.22,
        )

    if have_violin:

        if language == 'en':
            violin_text = dedent("""
The chart summarizes the number of days until donors make their second donation.
- Width represents donor density: wider sections = more donors.
- Central lines show median and quartile values.
- Average and median values are highlighted in boxes: dark for average, light for median.
        """).strip()

            pdf.sub_title("Time to Second Donation")
            pdf.paragraph(violin_text, indent=1)
        else:
            violin_text = dedent("""
Grafik, bağışçıların ikinci bağışlarını yapana kadar geçen gün sayısını özetlemektedir.
- Grafiğin genişliği bağışçı yoğunluğunu gösterir: geniş kısımlar daha çok bağışçıyı temsil eder.
- Ortadaki çizgiler medyan ve çeyreklik değerleri verir.
- Ortalama ve medyan değerler kutucuklarla vurgulanmıştır: koyu renk ortalama, açık renk medyan.
        """).strip()

            pdf.sub_title("İkinci Bağışa Geçiş Süresi")
            pdf.paragraph(violin_text, indent=1)

        pdf.add_chart_from_func(
            chart_days_to_second_donation_violin,
            raw_data=duzenlenmis_data,
            language=language,
            align="C",
        )

    # ========================================
    # INFORMATION & DISCLAIMER
    # ========================================
    pdf.add_page()

    if language == "en":
        pdf.chapter_title("5 - Information and Disclaimer")

        info_text = dedent("""
This report is a sample output prepared to demonstrate the donor analytics and segmentation framework developed for this project. It represents a simplified demonstration of an analytical system originally designed and implemented for a real non-profit organization.

The original project includes not only analytical reports such as this one, but also comprehensive strategic documentation covering donor segmentation strategy, action plans, performance indicators, management recommendations, and implementation guidelines. This demonstration intentionally presents only a limited subset of the original deliverables in order to showcase the analytical capabilities of the system.

All data presented in this report is synthetic (artificially generated). Names, organizations, donation records, financial figures, segment distributions, charts, and all other contents are fictional and do not represent real individuals or institutions. Any resemblance to actual persons, organizations, or events is entirely coincidental.

As part of this project, all source code—including the synthetic data generation pipeline—is openly available. This enables the entire workflow, from data generation to analysis and reporting, to be examined, reproduced, and extended without the use of any real personal or organizational data.
        """).strip()

    else:
        pdf.chapter_title("5 - Bilgilendirme ve Sorumluluk Reddi")

        info_text = dedent("""
Bu rapor, geliştirilen bağışçı analizi ve segmentasyon altyapısının tanıtımı amacıyla hazırlanmış örnek bir çıktıdır. Gerçek bir sivil toplum kuruluşu için geliştirilen ve kullanılan analiz sisteminin sadeleştirilmiş bir demonstrasyonunu temsil etmektedir.

Orijinal proje kapsamında bu rapora ek olarak; bağışçı segmentasyonu stratejisi, aksiyon planları, performans göstergeleri, yönetsel değerlendirmeler ve uygulama önerilerini içeren kapsamlı strateji dokümanları da sunulmaktadır. Bu örnek raporda ise yalnızca analiz altyapısının temel yeteneklerini göstermek amacıyla sınırlı bir içerik paylaşılmıştır.

Raporda kullanılan tüm veriler sentetik (yapay olarak üretilmiş) verilerdir. Kişi adları, kurum adları, bağış kayıtları, finansal büyüklükler, segment dağılımları, grafikler ve diğer tüm içerikler gerçek verileri temsil etmemektedir. Herhangi bir gerçek kişi, kurum veya organizasyon ile benzerlik bulunması tamamen tesadüfidir.

Bu proje kapsamında, sentetik veri üretimi dahil olmak üzere analiz sürecine ait tüm kaynak kodlar açık olarak paylaşılmıştır. Böylece raporda sunulan veri üretim süreci, analiz adımları ve raporlama altyapısı gerçek kişisel veriler kullanılmaksızın uçtan uca incelenebilir, yeniden üretilebilir ve geliştirilebilir.
        """).strip()

    pdf.paragraph(info_text, indent=1)

    # ========================================
    # FOOTER
    # ========================================
    pdf.ln(10)
    pdf.set_font("DejaVu", "I", 8)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(
        0,
        10,
        f'Report generated automatically - {datetime.now().strftime("%d.%m.%Y %H:%M")}',
        align="C",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )


    # Save PDF
    pdf.output(output_path)

    return output_path

if __name__ == "__main__":
    from utils.demo_data_generator import generate_demo_data
    from utils.consolidator import consolidate
    from utils.churn_risk_predictor import predict_churn_risk
    from utils.rfm_analyzer import analyze_rfm

    n_bagisci = 60
    n_bagis = 1000
    random_seed = 1907
    cutoff_date = None
    language = "en"

    output_path = "bagisci_segmentasyonu_raporu.pdf"

    donation_data, donor_data = generate_demo_data(
        n_bagisci=n_bagisci, 
        n_bagis=n_bagis, 
        random_seed=random_seed
    )

    consolidated_data = consolidate(
        df=donation_data,
        df_bagiscilar=donor_data
    )

    churns = predict_churn_risk(
        duzenlenmis_data=donation_data,
        toplulasmis_data=consolidated_data,
        cutoff_date=cutoff_date
    )

    rfms = analyze_rfm(donation_data)

    name_surname = (
        donation_data[["Bagisci No", "Ad Soyad"]]
        .drop_duplicates(subset="Bagisci No")
    )

    final_data = (
        consolidated_data
        .merge(name_surname, on="Bagisci No", how="left")
        .merge(churns, on="Bagisci No", how="left")
        .merge(rfms, on="Bagisci No", how="left")
    )

    if "Terk Riski" in final_data.columns:
        final_data["Terk Riski"] = final_data["Terk Riski"].fillna("Terk Riski Yok")

    path = generate_report(
        final_data=final_data,
        duzenlenmis_data=donation_data,
        output_path=output_path,
        language=language
    )

