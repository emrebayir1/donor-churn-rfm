import pandas as pd
import numpy as np


# =========================================================
# Helper Functions
# =========================================================
def _to_datetime_inplace(df: pd.DataFrame, cols: list[str]) -> None:
    """Converts the given columns to datetime (skips if not present)."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")


def _mode_or_first(values: np.ndarray):
    """
    Cleans NaN/None/empty strings, normalizes mixed types to string,
    and returns the most frequent value (mode) (ties resolved by first occurrence).
    """
    arr = values[pd.notna(values)]
    if len(arr) == 0:
        return None

    arr = arr.astype("object")
    arr = np.array([str(x).strip() for x in arr], dtype=object)
    arr = arr[arr != ""]
    if len(arr) == 0:
        return None

    u, c = np.unique(arr, return_counts=True)
    return u[c.argmax()]


def _nunique_nonnull(values: np.ndarray) -> int:
    arr = values[pd.notna(values)]
    if len(arr) == 0:
        return 0
    return len(np.unique(arr.astype("object")))


def _numeric_stats(values: np.ndarray):
    """
    Computes sum, mean, median, min, max, count statistics.
    Returns [0,0,0,0,0,0] if empty.
    """
    arr = np.asarray(values, dtype="float64")
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return [0, 0, 0, 0, 0, 0]
    return [arr.sum(), arr.mean(), np.median(arr), arr.min(), arr.max(), len(arr)]


def _coerce_ref_to_month_end(ref_raw) -> pd.Timestamp:
    """
    If ref_raw is not a month end, rounds it down to "the last day of the last completed month".
    E.g.: 2025-02-02 -> 2025-01-31
    """
    ref_raw = pd.Timestamp(ref_raw).normalize()
    if ref_raw.is_month_end:
        return ref_raw
    return (ref_raw - pd.offsets.MonthEnd(1)).normalize()


def _window_start_from_ref(ref: pd.Timestamp, months: int) -> pd.Timestamp:
    """
    Goes back `months` months from ref, then shifts to the 1st of the next month with MonthBegin(1).
    E.g.: ref=2025-12-31, months=24 -> start=2024-01-01
    """
    return (ref - pd.DateOffset(months=int(months))) + pd.offsets.MonthBegin(1)


# =========================================================
# Main Function
# =========================================================
def consolidate(
    df: pd.DataFrame,
    df_bagiscilar: pd.DataFrame = None,  # Optional second table parameter
    bagisci_no_kolon: str = "Bagisci",
    tarih_kolon: str = "Bagis Tarihi",
    filtrele: bool = True,
    ay: int = 24,
    referans_tarih=None,
    filtre_kolonu=None,
):
    """
    Summarizes donation data (and donor card data, if provided) by grouping on donor.
    """
    # 1) Automatic matching / protection steps for column names
    if bagisci_no_kolon not in df.columns:
        for alt in ["Bagisci", "Bagisci No", "Bağışçı No", "No"]:
            if alt in df.columns:
                bagisci_no_kolon = alt
                break

    # 2) If a donor card table was provided, automatically perform the merge
    if df_bagiscilar is not None:
        # Find the ID column in df_bagiscilar
        bagiscilar_id_kolon = None
        for alt in ["No", "Bagisci", "Bagisci No", "Bağışçı No"]:
            if alt in df_bagiscilar.columns:
                bagiscilar_id_kolon = alt
                break
        
        if bagiscilar_id_kolon is None:
            raise ValueError("No matching ID column ('No', 'Bagisci', etc.) found in the df_bagiscilar table.")
        
        # Merge with a suffix to preserve columns that would otherwise produce duplicate records
        df = df.merge(
            df_bagiscilar,
            left_on=bagisci_no_kolon,
            right_on=bagiscilar_id_kolon,
            suffixes=("", "_bagisci")
        )

    df = df.copy()

    if tarih_kolon not in df.columns:
        for alt in ["Bagis Tarihi", "Bağış Tarihi", "Tarih"]:
            if alt in df.columns:
                tarih_kolon = alt
                break

    # 3) Convert the date column
    _to_datetime_inplace(df, [tarih_kolon])

    # 4) Last `ay` months filter (optional)
    if filtrele:
        tarih_filtre_kolonu = filtre_kolonu or tarih_kolon
        if tarih_filtre_kolonu not in df.columns:
            raise ValueError(f"Filter column not found: {tarih_filtre_kolonu}")

        ref_raw = pd.to_datetime(referans_tarih) if referans_tarih is not None else df[tarih_filtre_kolonu].max()
        if pd.isna(ref_raw):
            raise ValueError("Reference date could not be computed (all dates may be NaN).")

        ref = _coerce_ref_to_month_end(ref_raw)
        baslangic = _window_start_from_ref(ref, ay)

        df = df[
            (df[tarih_filtre_kolonu].notna())
            & (df[tarih_filtre_kolonu] >= baslangic)
            & (df[tarih_filtre_kolonu] <= ref)
        ].copy()

    # 5) Group indices and array cache
    groups = df.groupby(bagisci_no_kolon, sort=False)
    group_indices = groups.indices
    bagisci_nolar = list(group_indices.keys())
    n_groups = len(bagisci_nolar)

    df_arrays = {col: df[col].values for col in df.columns}

    # 6) Dynamic Containers
    numeric_candidates = ["Tutar", "Toplam Tutar", "Tutar_TRY", "Tutar_USD"]
    numeric_cols = [c for c in numeric_candidates if c in df.columns]
    
    numeric_results = {
        f"{col}_{stat}": np.zeros(n_groups)
        for col in numeric_cols
        for stat in ["sum", "mean", "median", "min", "max", "count"]
    }

    mode_nunique_candidates = ["Tur", "Tür", "Bagis Kanali", "Bağış Kanalı", "Kampanya Adi", "Kampanya Adı", "Personel"]
    mode_candidates = ["Para Birimi", "Odeme Sekli", "Ödeme Şekli", "Il", "İl", "Ulke", "Ülke", "Kurban Ulke", "Kurban Ülke"]
    first_candidates = ["Cep Telefon", "T.C. Kimlik", "Uyelik Turu", "Üyelik Türü", "Temsilci", "Cinsiyet", "Eposta", "Meslek"]

    active_mode_nunique = [c for c in mode_nunique_candidates if c in df.columns]
    active_mode = [c for c in mode_candidates if c in df.columns]
    active_first = [c for c in first_candidates if c in df.columns]

    other_results = {
        "Bağış Sayısı": np.zeros(n_groups, dtype=int)
    }
    
    for c in active_mode_nunique:
        other_results[f"{c}_mode"] = [None] * n_groups
        other_results[f"{c}_nunique"] = np.zeros(n_groups, dtype=int)
        
    for c in active_mode:
        other_results[f"{c}_mode"] = [None] * n_groups
        
    for c in active_first:
        other_results[c] = [None] * n_groups

    active_dates = []
    if tarih_kolon in df.columns:
        active_dates.append(tarih_kolon)
        other_results[f"{tarih_kolon}_min"] = [None] * n_groups
        other_results[f"{tarih_kolon}_max"] = [None] * n_groups

    # 7) Main Loop (Numpy Aggregation)
    for i, no in enumerate(bagisci_nolar):
        idx = group_indices[no]
        other_results["Bağış Sayısı"][i] = len(idx)

        # Numeric
        for col in numeric_cols:
            s, m, med, mn, mx, cnt = _numeric_stats(df_arrays[col][idx])
            numeric_results[f"{col}_sum"][i] = s
            numeric_results[f"{col}_mean"][i] = m
            numeric_results[f"{col}_median"][i] = med
            numeric_results[f"{col}_min"][i] = mn
            numeric_results[f"{col}_max"][i] = mx
            numeric_results[f"{col}_count"][i] = cnt

        # Mode & Unique Count
        for col in active_mode_nunique:
            arr = df_arrays[col][idx]
            other_results[f"{col}_mode"][i] = _mode_or_first(arr)
            other_results[f"{col}_nunique"][i] = _nunique_nonnull(arr)

        # Mode Only
        for col in active_mode:
            arr = df_arrays[col][idx]
            other_results[f"{col}_mode"][i] = _mode_or_first(arr)

        # Constant Info (First Value)
        for col in active_first:
            other_results[col][i] = df_arrays[col][idx][0]

        # Dates
        for col in active_dates:
            vals = df_arrays[col][idx]
            vals = vals[pd.notna(vals)]
            if len(vals):
                other_results[f"{col}_min"][i] = vals.min()
                other_results[f"{col}_max"][i] = vals.max()

    # 8) Output DataFrame
    result = {bagisci_no_kolon: bagisci_nolar}
    result.update(other_results)
    result.update(numeric_results)

    ozet_df = pd.DataFrame(result)

    # Decimal rounding
    tutar_cols = [c for c in ozet_df.columns if c.startswith("Tutar") or c.startswith("Toplam Tutar")]
    ozet_df[tutar_cols] = ozet_df[tutar_cols].round(2)

    return ozet_df