import pandas as pd
import numpy as np
import contextlib
import io

from utils.consolidator import consolidate

def generate_training_data(
    df: pd.DataFrame,
    bagisci_no_kolon: str = "Bagisci No",
    tarih_kolon: str = "Bagis Tarihi",
    cutoff_freq: str = "MS",              # monthly cutoff: "MS"=month start, "M"=month end
    silent_toplulastir: bool = True,
    lookback_days: int = 180,             # Number of days to look back for feature extraction (Default: 180)
    lookforward_days: int = 180,          # Number of days to look forward for churn detection (Default: 180)
):
    """
    Generates a single training dataset for all donors.

    - Eligible (Analysis Eligibility): Active donors who made at least 2 donations within the
      `lookback_days` period before the cutoff.
    - Churn Label: 1 if the donor made no donations during the `lookforward_days` period after
      the cutoff, 0 if they did.

    Output:
      A single pandas DataFrame. Columns: toplulastir(...) features + ["Cutoff Tarihi", "churn"]
    """

    df = df.copy()

    # --- Required column check
    for c in [bagisci_no_kolon, tarih_kolon]:
        if c not in df.columns:
            raise ValueError(f"Column not found: {c}")

    # --- Date conversions
    df[tarih_kolon] = pd.to_datetime(df[tarih_kolon], errors="coerce")


    df = df[pd.notna(df[tarih_kolon])].copy()
    if df.empty:
        empty_features = consolidate(df=df, bagisci_no_kolon=bagisci_no_kolon, tarih_kolon=tarih_kolon)
        empty_features["Cutoff Tarihi"] = pd.to_datetime([])
        empty_features["churn"] = pd.Series([], dtype="int8")
        return empty_features.copy()

    # --- Normalization of categorical columns to prevent the np.unique (sort) error inside toplulastir()
    _CAT_COLS_FOR_TOPLULASTIR = [
        "Tür",
        "Para Birimi",
        "Bagis Kanali",
        "Personel",
        "Ulke",
        "Il",
        "Odeme Sekli",
        "Referans",
        "Kampanya Adi",
        "Hedef Bölge",
        "Bağış Kalem ID",
        "Üyelik Türü",
        "Temsilci",
        "Cinsiyet",
        "Eposta",
        "Meslek",
        "Bagis Tarihi Mevsim",
        "Gönderim Tarihi Mevsim",
        "Üyelik Tarihi Mevsim",
    ]

    def _sanitize_for_toplulastir(dfx: pd.DataFrame) -> pd.DataFrame:
        dfx = dfx.copy()
        for c in _CAT_COLS_FOR_TOPLULASTIR:
            if c in dfx.columns:
                dfx[c] = dfx[c].astype("string")
        return dfx

    def _run_toplulastir(dfx: pd.DataFrame):
        dfx = _sanitize_for_toplulastir(dfx)
        if silent_toplulastir:
            with contextlib.redirect_stdout(io.StringIO()):
                return consolidate(df=dfx, bagisci_no_kolon=bagisci_no_kolon, tarih_kolon=tarih_kolon)
        return consolidate(df=dfx, bagisci_no_kolon=bagisci_no_kolon, tarih_kolon=tarih_kolon)

    # --- toplulastir template columns (to fix the output structure)
    with contextlib.redirect_stdout(io.StringIO()) if silent_toplulastir else contextlib.nullcontext():
        template_cols = list(
            consolidate(df=df.head(1), bagisci_no_kolon=bagisci_no_kolon, tarih_kolon=tarih_kolon).columns
        )

    # --- Cutoff range calculation
    min_date = df[tarih_kolon].min()
    max_date = df[tarih_kolon].max()

    start_cutoff = (min_date + pd.Timedelta(days=lookback_days)).to_period("M").to_timestamp(how="start")
    end_cutoff   = (max_date - pd.Timedelta(days=lookforward_days)).to_period("M").to_timestamp(how="start")

    # If the data range is too narrow to produce a cutoff, return an empty df
    if start_cutoff > end_cutoff:
        empty_df = pd.DataFrame(columns=template_cols + ["Cutoff Tarihi", "churn"])
        empty_df["churn"] = empty_df["churn"].astype("int8")
        return empty_df

    cutoffs = pd.date_range(start=start_cutoff, end=end_cutoff, freq=cutoff_freq)

    dataset_parts = []

    for cutoff in cutoffs:
        lb = pd.Timedelta(days=lookback_days)
        fw = pd.Timedelta(days=lookforward_days)

        # Past and future data ranges relative to the cutoff
        past = df[(df[tarih_kolon] >= cutoff - lb) & (df[tarih_kolon] < cutoff)]
        future = df[(df[tarih_kolon] >= cutoff) & (df[tarih_kolon] < cutoff + fw)]

        # Identify donors who made at least 2 donations in the past 180 days (RFM Frequency active)
        past_counts = past.groupby(bagisci_no_kolon).size()
        eligible_donors = past_counts[past_counts >= 2].index

        if len(eligible_donors) > 0:
            # Churn based on whether a donation was made in the next 180 days (1: Churned, 0: Retained)
            future_counts = future.groupby(bagisci_no_kolon).size().reindex(eligible_donors, fill_value=0)
            churn_s = (future_counts == 0).astype("int8").rename("churn")

            # Aggregate the past data of eligible donors to produce RFM / Churn features
            past_eligible = past[past[bagisci_no_kolon].isin(eligible_donors)]
            feats = _run_toplulastir(past_eligible).reindex(columns=template_cols)

            # Add the cutoff date and the churn target variable to the dataset
            feats["Cutoff Tarihi"] = cutoff
            feats = feats.merge(churn_s, left_on=bagisci_no_kolon, right_index=True, how="left")
            feats["churn"] = feats["churn"].fillna(1).astype("int8")

            dataset_parts.append(feats)

    # --- Combine data from all time slices (cutoffs) into a single table
    if dataset_parts:
        df_egitim = pd.concat(dataset_parts, ignore_index=True)
    else:
        df_egitim = pd.DataFrame(columns=template_cols + ["Cutoff Tarihi", "churn"])
        df_egitim["churn"] = df_egitim["churn"].astype("int8")

    return df_egitim

if __name__ == "__main__":
    df = pd.read_csv("bagislar.csv")
    train_df = generate_training_data(df=df)
    train_df.to_csv("train_df.csv", index=False)