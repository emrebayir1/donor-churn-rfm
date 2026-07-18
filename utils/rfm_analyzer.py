import numpy as np
import pandas as pd
from typing import Optional, Callable

def safe_qcut(series: pd.Series, q: int = 5, reverse: bool = False) -> pd.Series:
    series_clean = series.dropna()

    if series_clean.nunique() < 2:
        mid_score = (q + 1) // 2
        return pd.Series([mid_score] * len(series), index=series.index)

    try:
        _, bins = pd.qcut(series_clean, q, retbins=True, duplicates="drop")
        n_bins = len(bins) - 1

        if n_bins < q:
            bins = np.linspace(series_clean.min(), series_clean.max(), q + 1)
            n_bins = q

        labels = list(range(1, n_bins + 1))
        if reverse:
            labels = labels[::-1]

        return pd.cut(series, bins=bins, labels=labels, include_lowest=True)

    except Exception:
        mid_score = (q + 1) // 2
        return pd.Series([mid_score] * len(series), index=series.index)

# Default bin counts
R_BINS = 5
M_BINS = 5

# Behavior segments (based on R and F score)
BEHAVIOR_MAP = {
    "Şampiyonlar":         {"R": [5],       "F": [2]},
    "Sadık Bağışçılar":    {"R": [4],       "F": [2]},
    "Potansiyel Sadıklar": {"R": [3],       "F": [2]},
    "Yeni Bağışçılar":     {"R": [4, 5],    "F": [1]},
    "İlgi Gerektiren":     {"R": [3],       "F": [1]},
    "Uykuda":              {"R": [2],       "F": [1, 2]},
    "Kayıp Riskli":        {"R": [1],       "F": [1, 2]},
}

def new_m_segment_rules(m_score: int, m_bins: int) -> Optional[str]:
    if m_score == 0:
        return None
    if m_score == 5:
        return "VIP"
    if m_score == 4:
        return "Yüksek"
    if m_score == 3:
        return "Orta"
    return "Standart"

def new_behavioral_rules(r_score: int, f_score: int) -> str:
    if r_score == 0:
        return "Pasif"

    for segment_name, rules in BEHAVIOR_MAP.items():
        if r_score in rules["R"] and f_score in rules["F"]:
            return segment_name

    return "Tanımsız"

def new_final_segment_rules(deger: Optional[str], davranis: str) -> str:
    if davranis == "Pasif":
        return "Pasif / Aday"
    if deger is None:
        return davranis
    return f"{deger} {davranis}"

def _aggregate_window(
    df_txn: pd.DataFrame,
    id_col: str,
    tarih_col: str,
    tutar_col: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Summarizes transaction data within the [start, end] range on a per-donor basis:
    - last_date
    - freq
    - monetary_sum
    """
    m = (df_txn[tarih_col] >= start) & (df_txn[tarih_col] <= end)
    d = df_txn.loc[m, [id_col, tarih_col, tutar_col]].copy()

    if len(d) == 0:
        return pd.DataFrame(columns=[id_col, "last_date", "freq", "monetary_sum"])

    out = (
        d.groupby(id_col, as_index=False)
        .agg(
            last_date=(tarih_col, "max"),
            freq=(tarih_col, "size"),
            monetary_sum=(tutar_col, "sum"),
        )
    )
    return out

def _rfm_from_agg(
    df_agg: pd.DataFrame,
    id_col: str,
    referans_tarihi: pd.Timestamp,
    r_bins: int,
    m_bins: int,
    m_segment_func: Callable,
    behavioral_func: Callable,
    final_segment_func: Callable,
) -> pd.DataFrame:
    """
    Produces RFM score + segment from the donor-level summary.
    """
    bagiscilar = df_agg.copy()

    # --- RECENCY CALCULATION (NaT safety) ---
    bagiscilar["last_date"] = pd.to_datetime(bagiscilar["last_date"], errors='coerce')
    
    # Recency (days): handle rows with NaT
    bagiscilar["Recency"] = 999  # Default: inactive
    valid_mask = bagiscilar["last_date"].notna()
    bagiscilar.loc[valid_mask, "Recency"] = (
        (referans_tarihi - bagiscilar.loc[valid_mask, "last_date"]).dt.days
    )
    
    # Frequency / Monetary
    bagiscilar["Frequency"] = bagiscilar["freq"].fillna(0).astype(int)
    bagiscilar["Monetary"] = bagiscilar["monetary_sum"].fillna(0)

    # Only score donors with actual donations (Frequency > 0)
    active_mask = (bagiscilar["Frequency"] > 0) & (bagiscilar["last_date"].notna())

    bagiscilar["R_Score"] = 0
    bagiscilar["F_Score"] = 0
    bagiscilar["M_Score"] = 0

    if active_mask.sum() > 0:
        # R_Score: Recency (reverse order: recent = high score)
        bagiscilar.loc[active_mask, "R_Score"] = safe_qcut(
            bagiscilar.loc[active_mask, "Recency"],
            q=r_bins,
            reverse=True
        ).astype(int)

        # F_Score: Binary - just checks frequency > 1 (simple logic)
        # More sophisticated: a simple threshold instead of qcut
        bagiscilar.loc[active_mask, "F_Score"] = (
            (bagiscilar.loc[active_mask, "Frequency"] > 1).astype(int) + 1
        )

        # M_Score: Monetary value (ascending order)
        bagiscilar.loc[active_mask, "M_Score"] = safe_qcut(
            bagiscilar.loc[active_mask, "Monetary"],
            q=m_bins
        ).astype(int)

    # Convert scores to integer
    bagiscilar["R_Score"] = bagiscilar["R_Score"].astype(int)
    bagiscilar["F_Score"] = bagiscilar["F_Score"].astype(int)
    bagiscilar["M_Score"] = bagiscilar["M_Score"].astype(int)

    # RFM Combination
    bagiscilar["RFM_Score"] = (
        bagiscilar["R_Score"].astype(str)
        + bagiscilar["F_Score"].astype(str)
        + bagiscilar["M_Score"].astype(str)
    )

    # Segmentation Rules
    bagiscilar["Deger_Segmenti"] = bagiscilar["M_Score"].apply(lambda x: m_segment_func(x, m_bins))
    bagiscilar["Davranis_Segmenti"] = bagiscilar.apply(
        lambda row: behavioral_func(row["R_Score"], row["F_Score"]),
        axis=1
    )
    bagiscilar["Final_Segment"] = bagiscilar.apply(
        lambda row: final_segment_func(row["Deger_Segmenti"], row["Davranis_Segmenti"]),
        axis=1
    )

    return bagiscilar[[id_col, "R_Score", "F_Score", "M_Score", "RFM_Score",
                       "Deger_Segmenti", "Davranis_Segmenti", "Final_Segment"]]

def _coerce_ref_to_month_end(ref_raw) -> pd.Timestamp:
    """
    If ref_raw is not a month end, rounds it down to 'the last day of the last completed month'.
    E.g.: 2025-02-02 -> 2025-01-31
    E.g.: 2025-01-31 -> 2025-01-31
    """
    ref_raw = pd.Timestamp(ref_raw).normalize()
    if ref_raw.is_month_end:
        return ref_raw
    return (ref_raw - pd.offsets.MonthEnd(1)).normalize()

def _parse_txn_frame(
    df: pd.DataFrame,
    id_col: str,
    tarih_col: str,
    tutar_col: str,
    para_birimi_col: str = "Para Birimi",
) -> pd.DataFrame:
    """
    Cleans/parses the transaction data on an (id, date, amount) basis.
    Converts currencies to USD (for compatibility with demo_veri).
    
    Date format: "YYYY-MM-DD HH:MM:SS" (from demo_veri.py)
    """
    d = df[[id_col, tarih_col, tutar_col, para_birimi_col]].copy()

    # Date cleaning + parsing (demo_veri format: "YYYY-MM-DD HH:MM:SS")
    d[tarih_col] = (
        d[tarih_col]
        .astype(str)
        .replace(r"^\s*$", np.nan, regex=True)
        .replace("None", np.nan)
    )
    d[tarih_col] = pd.to_datetime(d[tarih_col], errors="coerce")

    # Amount to numeric
    d[tutar_col] = pd.to_numeric(d[tutar_col], errors="coerce").fillna(0)

    # Currency conversion (TRY, EUR -> USD)
    # Weights from demo_veri.py: TRY 0.88, USD 0.08, EUR 0.04
    exchange_rates = {"TRY": 0.032, "USD": 1.0, "EUR": 1.1}
    
    d["tutar_usd"] = d.apply(
        lambda row: (
            row[tutar_col] * exchange_rates.get(str(row[para_birimi_col]).upper().strip(), 1.0)
            if pd.notna(row[tutar_col]) and row[tutar_col] > 0
            else 0
        ),
        axis=1,
    )

    return d[[id_col, tarih_col, "tutar_usd"]]

def _build_windows(ref: pd.Timestamp, window_months: int, shift_months: int):
    """
    Computes the two period windows:
      prev: [ref-(window+shift), ref-shift]
      next: [ref-window, ref]
    """
    prev_end = ref - pd.DateOffset(months=int(shift_months))
    prev_start = ref - pd.DateOffset(months=int(window_months + shift_months))

    next_end = ref
    next_start = ref - pd.DateOffset(months=int(window_months))

    return prev_start, prev_end, next_start, next_end

def analyze_rfm(
    df: pd.DataFrame,
    id_col: str = "Bagisci No",
    tarih_col: str = "Bagis Tarihi",
    tutar_col: str = "Tutar",
    para_birimi_col: str = "Para Birimi",
    window_months: int = 24,
    shift_months: int = 3,
    r_bins: int = R_BINS,
    m_bins: int = M_BINS,
    referans_tarihi: Optional[pd.Timestamp] = None,
    filter: bool = True,
    m_segment_func: Optional[Callable] = None,
    behavioral_func: Optional[Callable] = None,
    final_segment_func: Optional[Callable] = None,
) -> pd.DataFrame:
    """
    Produces RFM/segment from transaction data (compatible with demo_veri.py).

    Expected columns:
    - id_col: "Bagisci No" (integer)
    - tarih_col: "Bagis Tarihi" (string "YYYY-MM-DD HH:MM:SS")
    - tutar_col: "Tutar" (float, in original currency)
    - para_birimi_col: "Para Birimi" (TRY/USD/EUR)

    - The reference (ref) is always pinned to the END OF THE MONTH.
      If the max date is not a month end, the last completed month end is used.

    - filter=True (default):
      Produces two shifted periods of 24+3.
        prev: [ref-(window+shift), ref-shift]
        next: [ref-window, ref]
      Output: Bagisci No + *_prev + *_next columns

    - filter=False:
      Produces a single-period RFM covering the entire history.
      Output: Bagisci No + single-period score/segment columns
    """
    # Default segment functions
    if m_segment_func is None:
        m_segment_func = new_m_segment_rules
    if behavioral_func is None:
        behavioral_func = new_behavioral_rules
    if final_segment_func is None:
        final_segment_func = new_final_segment_rules

    # 1) Parse/clean (including currency conversion)
    d = _parse_txn_frame(df, id_col, tarih_col, tutar_col, para_birimi_col)
    # Rename the column from tutar_col to tutar_usd
    tutar_col = "tutar_usd"

    # 2) Donor universe
    donors = pd.DataFrame({id_col: d[id_col].dropna().unique()})

    # 3) Reference date (pin to month end)
    ref_raw = d[tarih_col].max() if referans_tarihi is None else pd.to_datetime(referans_tarihi)
    if pd.isna(ref_raw):
        raise ValueError("Reference date could not be computed (all dates may be NaN).")

    ref = _coerce_ref_to_month_end(ref_raw)

    # =========================================================
    # filter=False -> ALL DATA single period
    # =========================================================
    if not filter:
        # all data: [min_date, ref]
        min_date = d[tarih_col].min()
        if pd.isna(min_date):
            # If there is no valid date at all: return an empty segment table
            out = donors.copy()
            for c in ["R_Score", "F_Score", "M_Score", "RFM_Score",
                      "Deger_Segmenti", "Davranis_Segmenti", "Final_Segment"]:
                out[c] = np.nan
            return out

        agg_all = _aggregate_window(d, id_col, tarih_col, tutar_col, start=min_date, end=ref)
        all_ = donors.merge(agg_all, on=id_col, how="left")

        all_["freq"] = all_["freq"].fillna(0)
        all_["monetary_sum"] = all_["monetary_sum"].fillna(0)

        rfm_all = _rfm_from_agg(
            all_,
            id_col=id_col,
            referans_tarihi=ref,
            r_bins=r_bins,
            m_bins=m_bins,
            m_segment_func=m_segment_func,
            behavioral_func=behavioral_func,
            final_segment_func=final_segment_func,
        )

        seg_cols = [
            id_col,
            "R_Score", "F_Score", "M_Score", "RFM_Score",
            "Deger_Segmenti", "Davranis_Segmenti", "Final_Segment",
        ]
        return rfm_all[seg_cols]

    # =========================================================
    # filter=True -> TWO PERIODS (prev/next)
    # =========================================================
    prev_start, prev_end, next_start, next_end = _build_windows(ref, window_months, shift_months)

    agg_prev = _aggregate_window(d, id_col, tarih_col, tutar_col, start=prev_start, end=prev_end)
    agg_next = _aggregate_window(d, id_col, tarih_col, tutar_col, start=next_start, end=next_end)

    prev_all = donors.merge(agg_prev, on=id_col, how="left")
    next_all = donors.merge(agg_next, on=id_col, how="left")

    prev_all["freq"] = prev_all["freq"].fillna(0)
    prev_all["monetary_sum"] = prev_all["monetary_sum"].fillna(0)
    next_all["freq"] = next_all["freq"].fillna(0)
    next_all["monetary_sum"] = next_all["monetary_sum"].fillna(0)

    rfm_prev = _rfm_from_agg(
        prev_all,
        id_col=id_col,
        referans_tarihi=prev_end,
        r_bins=r_bins,
        m_bins=m_bins,
        m_segment_func=m_segment_func,
        behavioral_func=behavioral_func,
        final_segment_func=final_segment_func,
    )
    rfm_next = _rfm_from_agg(
        next_all,
        id_col=id_col,
        referans_tarihi=next_end,
        r_bins=r_bins,
        m_bins=m_bins,
        m_segment_func=m_segment_func,
        behavioral_func=behavioral_func,
        final_segment_func=final_segment_func,
    )

    # Add suffixes
    rfm_prev = rfm_prev.rename(columns={c: f"{c}_prev" for c in rfm_prev.columns if c != id_col})
    rfm_next = rfm_next.rename(columns={c: f"{c}_next" for c in rfm_next.columns if c != id_col})

    out = donors.merge(rfm_prev, on=id_col, how="left").merge(rfm_next, on=id_col, how="left")

    seg_cols = [
        id_col,
        "R_Score_prev", "F_Score_prev", "M_Score_prev", "RFM_Score_prev",
        "Deger_Segmenti_prev", "Davranis_Segmenti_prev", "Final_Segment_prev",
        "R_Score_next", "F_Score_next", "M_Score_next", "RFM_Score_next",
        "Deger_Segmenti_next", "Davranis_Segmenti_next", "Final_Segment_next",
    ]
    return out[seg_cols]


if __name__ == "__main__":
    df = pd.read_csv("bagislar.csv")
    rfm = analyze_rfm(df)
    print(rfm)
    print(rfm.columns)