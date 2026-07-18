import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# Suppress sklearn/lightgbm feature name warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Suppress pandas datetime parsing warnings
warnings.filterwarnings("ignore", message="Parsing dates in")

# =========================================================
# Model Loading (ChurnEnsembleModel)
# =========================================================
MODEL_DIR = Path(__file__).resolve().parent / "models"

preprocessor = joblib.load(MODEL_DIR / "preprocessor.joblib")

models = {
    "LightGBM": joblib.load(MODEL_DIR / "lightgbm.joblib"),
    "XGBoost": joblib.load(MODEL_DIR / "xgboost.joblib"),
    "CatBoost": joblib.load(MODEL_DIR / "catboost.joblib"),
}

with open(MODEL_DIR / "weights.json", encoding="utf-8") as f:
    weights = json.load(f)

with open(MODEL_DIR / "feature_names.json", encoding="utf-8") as f:
    feature_names = json.load(f)

print("✅ Models loaded successfully.")

# =========================================================
# Helper Functions
# =========================================================

def _resolve_column_name(df, possible_names):
    """
    Resolve the actual column name from a list of possible alternatives.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to search for column names
    possible_names : list
        List of possible column name variations

    Returns
    -------
    str
        The first matching column name from df, or the first possibility as fallback
    """
    for col_name in possible_names:
        if col_name in df.columns:
            return col_name
    return possible_names[0]  # Return default if no match found


def _apply_feature_engineering(df, debug=False):
    """
    Apply the same feature engineering logic as train_model.py's engineer_features().

    FIXED VERSION:
    - Proper column name handling (Odeme Sekli_mode not Odeme_Sekli_mode)
    - Automatic feature alignment for missing categorical values
    - Debug option to track feature creation

    Creates 7 engineered features from aggregated donor metrics:
    - Ratio features: Tur_orani, Kanal_orani, Kampanya_orani
    - Distribution metrics: Tutar_max_mean_orani, Tutar_min_max_orani, Tutar_mean_median_farki
    - Behavioral features: Hesaplanan_Ortalama_Tutar
    - Categorical encoding: Frequency & One-Hot encoding

    Parameters
    ----------
    df : pd.DataFrame
        Aggregated donor data with raw metrics
    debug : bool, default False
        If True, print feature creation information

    Returns
    -------
    pd.DataFrame
        DataFrame with engineered features ready for model prediction
    """
    df_fe = df.copy()

    # ---------------------------------------------------------
    # 1. RATIO & STATISTICAL FEATURES (Numeric Features)
    # ---------------------------------------------------------

    # Diversity indicators per donation count (loyalty & engagement)
    df_fe['Tur_orani'] = df_fe['Tür_nunique'] / df_fe['Bağış Sayısı']
    df_fe['Kanal_orani'] = df_fe['Bagis Kanali_nunique'] / df_fe['Bağış Sayısı']
    df_fe['Kampanya_orani'] = df_fe['Kampanya Adi_nunique'] / df_fe['Bağış Sayısı']

    # Donation amount distribution metrics (skewness & spread)
    df_fe['Tutar_max_mean_orani'] = df_fe['Tutar_max'] / (df_fe['Tutar_mean'] + 1e-5)
    df_fe['Tutar_min_max_orani'] = df_fe['Tutar_min'] / (df_fe['Tutar_max'] + 1e-5)

    # Difference between Mean and Median (behavior stability indicator)
    df_fe['Tutar_mean_median_farki'] = df_fe['Tutar_mean'] - df_fe['Tutar_median']

    # Calculated average donation amount
    df_fe['Hesaplanan_Ortalama_Tutar'] = df_fe['Tutar_sum'] / (df_fe['Bağış Sayısı'] + 1e-5)

    # ---------------------------------------------------------
    # 2. CATEGORICAL VARIABLE ENCODING
    # ---------------------------------------------------------

    # High Cardinality Features (Frequency Encoding)
    for col in ['Il_mode', 'Kampanya Adi_mode']:
        if col in df_fe.columns:
            freq = df_fe[col].value_counts() / len(df_fe)
            df_fe[f'{col}_freq'] = df_fe[col].map(freq)

    # Low Cardinality Categorical Features (One-Hot Encoding)
    # ⚠️ FIXED: Correct column names
    categorical_cols = ['Tür_mode', 'Bagis Kanali_mode', 'Para Birimi_mode', 'Odeme Sekli_mode']

    # Use drop_first=True to avoid the dummy variable trap
    categorical_cols_present = [col for col in categorical_cols if col in df_fe.columns]

    if debug:
        print(f"📊 DEBUG: Kategorik sütunlar bulundu: {categorical_cols_present}")

    if categorical_cols_present:
        df_fe = pd.get_dummies(df_fe, columns=categorical_cols_present, drop_first=True, dtype=int)

        if debug:
            one_hot_cols = [col for col in df_fe.columns if any(
                cat in col for cat in ['Tür_mode_', 'Bagis Kanali_mode_', 'Para Birimi_mode_', 'Odeme Sekli_mode_']
            )]
            print(f"📊 DEBUG: Oluşturulan one-hot encoded sütunlar ({len(one_hot_cols)}):")
            for col in sorted(one_hot_cols)[:10]:
                print(f"   ✓ {col}")
            if len(one_hot_cols) > 10:
                print(f"   ... ve {len(one_hot_cols) - 10} tane daha")

    # Drop the original high cardinality categorical columns
    df_fe.drop(columns=['Il_mode', 'Kampanya Adi_mode'], inplace=True, errors='ignore')

    # Remove perfectly correlated column
    df_fe.drop(columns=['Tutar_count'], inplace=True, errors='ignore')

    return df_fe


def ensemble_predict_proba(X: pd.DataFrame, debug=False) -> np.ndarray:
    """
    Calculates weighted ensemble probabilities with automatic feature alignment.

    FIXED VERSION:
    - Automatically fills missing features with zero instead of raising error
    - Proper feature ordering and alignment
    - Debug option to track missing features

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix for prediction
    debug : bool, default False
        If True, print information about missing/aligned features

    Returns
    -------
    np.ndarray
        Ensemble prediction probabilities
    """

    missing = set(feature_names) - set(X.columns)

    if missing:
        if debug or len(missing) > 0:
            print(f"⚠️  Eksik feature'lar tespit edildi ({len(missing)} sütun)")
            print(f"Eksik feature'lar: {missing}")
            print(f"    Bunlar sıfır ile doldurulacak...")

        # Add missing features with zero values
        for col in missing:
            X[col] = 0

    # Select only required features in the correct order
    X = X[feature_names]

    if debug:
        print(f"✅ Feature alignment tamamlandı: {X.shape[1]} sütun")

    X_transformed = preprocessor.transform(X)

    preds = np.zeros(X_transformed.shape[0])

    for name, model in models.items():
        preds += (
            weights[name]
            * model.predict_proba(X_transformed)[:, 1]
        )

    return preds

# =========================================================
# Main Functions
# =========================================================

def get_eligible_donors(
    df: pd.DataFrame,
    cutoff_date,
    donor_col: str = "Bagisci No",
    date_col: str = "Bağış Tarihi",
    dayfirst: bool = True,
    pencere_gun: int = 365,
) -> pd.DataFrame:
    """
    Identify eligible donors for churn prediction based on a cutoff date.

    Following the same logic as the training data generator:
    Donors who made at least 2 donations within the lookback window
    (pencere_gun days before cutoff_date) are considered eligible.

    Parameters
    ----------
    df : pd.DataFrame
        Donation-level data containing transaction records
    cutoff_date : str or datetime
        Reference date for prediction window (e.g., "2024-06-30")
    donor_col : str, default "Bagisci No"
        Column name containing donor IDs
    date_col : str, default "Bağış Tarihi"
        Column name containing donation dates
    dayfirst : bool, default True
        If True, interpret dates as day-first format (e.g., 31/12/2024)
    pencere_gun : int, default 365
        Lookback window size in days

    Returns
    -------
    pd.DataFrame
        DataFrame containing eligible donor IDs
    """
    # Auto-resolve column names (robustness)
    active_donor_col = _resolve_column_name(df, [donor_col, "Bagisci No", "Bagisci", "Bağışçı No", "No"])
    active_date_col = _resolve_column_name(df, [date_col, "Bağış Tarihi", "Bagis Tarihi", "Tarih"])

    cutoff = pd.to_datetime(cutoff_date)
    start_date = cutoff - pd.Timedelta(days=pencere_gun)

    dates = pd.to_datetime(df[active_date_col], errors="coerce", dayfirst=dayfirst)

    # Apply the same "past" filter as the training data generator
    in_window = (dates >= start_date) & (dates < cutoff)
    past = df[in_window]

    past_counts = past.groupby(active_donor_col).size()

    # >= 2 donations rule
    eligible = past_counts[past_counts >= 2].reset_index()[[active_donor_col]]
    return eligible


def filter_eligible_donors(
    uygun_bagiscilar: pd.DataFrame,
    toplulasmis_data: pd.DataFrame,
    bagisci_kolonu: str = "Bagisci No",
) -> pd.DataFrame:
    """
    Filter aggregated donor data to include only eligible donors.

    Parameters
    ----------
    uygun_bagiscilar : pd.DataFrame
        DataFrame containing eligible donor IDs
    toplulasmis_data : pd.DataFrame
        Aggregated donor-level metrics
    bagisci_kolonu : str, default "Bagisci No"
        Donor ID column name

    Returns
    -------
    pd.DataFrame
        Filtered aggregated data containing only eligible donors
    """

    active_bagisci_kolonu = _resolve_column_name(
        uygun_bagiscilar,
        [bagisci_kolonu, "Bagisci No", "Bagisci", "Bağışçı No", "No"]
    )

    top_bagisci_kolonu = _resolve_column_name(
        toplulasmis_data,
        [bagisci_kolonu, "Bagisci No", "Bagisci", "Bağışçı No", "No"]
    )

    uygun_ids = uygun_bagiscilar[active_bagisci_kolonu]

    filtered = toplulasmis_data[toplulasmis_data[top_bagisci_kolonu].isin(uygun_ids)]
    return filtered


def preprocess_prediction_data(
    ham_data: pd.DataFrame,
    donor_col: str = "Bagisci No",
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess raw aggregated data for prediction.

    Parameters
    ----------
    ham_data : pd.DataFrame
        Raw aggregated donor data
    donor_col : str, default "Bagisci No"
        Donor ID column name

    Returns
    -------
    tuple
        (processed feature data, donor IDs)
    """

    # Auto-resolve column name
    active_donor_col = _resolve_column_name(ham_data, [donor_col, "Bagisci No", "Bagisci"])

    # Extract donor IDs
    ids = ham_data[active_donor_col].copy()

    # Create a copy for processing
    X = ham_data.drop(columns=[active_donor_col], errors='ignore')

    return X, ids


def generate_prediction_data(
    duzenlenmis_data: pd.DataFrame,
    toplulasmis_data: pd.DataFrame,
    cutoff_date: str,
    donor_col: str = "Bagisci No",
    pencere_gun: int = 365,
    debug: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Create prediction dataset with full feature engineering pipeline.

    Parameters
    ----------
    duzenlenmis_data : pd.DataFrame
        Donation-level transaction records
    toplulasmis_data : pd.DataFrame
        Aggregated donor-level metrics
    cutoff_date : str
        Reference date for prediction (YYYY-MM-DD format)
    donor_col : str, default "Bagisci No"
        Donor ID column name
    pencere_gun : int, default 365
        Lookback window in days
    debug : bool, default False
        Enable debug output

    Returns
    -------
    tuple
        (feature matrix, donor IDs)
    """

    # Auto-resolve column names
    active_donor_col = _resolve_column_name(
        duzenlenmis_data,
        [donor_col, "Bagisci No", "Bagisci", "Bağışçı No", "No"]
    )

    # Get eligible donors
    uygun_bagiscilar = get_eligible_donors(
        df=duzenlenmis_data,
        cutoff_date=cutoff_date,
        donor_col=active_donor_col,
        pencere_gun=pencere_gun
    )

    if debug:
        print(f"📊 Uygun bağışçı sayısı: {len(uygun_bagiscilar)}")

    # Filter aggregated data to eligible donors
    ham_data = filter_eligible_donors(
        uygun_bagiscilar=uygun_bagiscilar,
        toplulasmis_data=toplulasmis_data,
        bagisci_kolonu=active_donor_col
    )

    if debug:
        print(f"📊 Filtrelenen bağışçı sayısı: {len(ham_data)}")

    # Preprocess data
    X, ids = preprocess_prediction_data(ham_data, donor_col=active_donor_col)

    # Apply feature engineering
    X = _apply_feature_engineering(X, debug=debug)

    if debug:
        print(f"📊 Feature engineering sonrası feature sayısı: {X.shape[1]}")

    return X, ids


def score_churn_risk(X, ids, donor_col: str = "Bagisci No", debug: bool = False) -> pd.DataFrame:
    """
    Calculate churn risk probabilities using ChurnEnsembleModel.

    FIXED VERSION:
    - Handles missing features gracefully
    - Debug option for troubleshooting

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (output of tahmin_verisi_olustur)
    ids : pd.Series
        Donor IDs corresponding to rows in X
    donor_col : str, default "Bagisci No"
        Donor ID column name in output
    debug : bool, default False
        Enable debug output

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - Donor ID column (e.g., "Bagisci No")
        - "Terk Riski": Churn risk probabilities
    """
    if len(X) == 0:
        print("⚠️  No eligible donors found. Returning empty churn risk table.")
        return pd.DataFrame(columns=[donor_col, "Terk Riski"])

    try:
        risk_scores = ensemble_predict_proba(X, debug=debug)

        result_df = pd.DataFrame({
            donor_col: ids.values,
            "Terk Riski": risk_scores,
        })

        if debug:
            print(f"✅ Tahmin tamamlandı: {len(result_df)} bağışçı")

        return result_df

    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(columns=[donor_col, "Terk Riski"])


def predict_churn_risk(
   duzenlenmis_data: pd.DataFrame,
   toplulasmis_data: pd.DataFrame,
   cutoff_date=None,
   donor_col: str = "Bagisci No",
   pencere_gun: int = 365,
   debug: bool = False,
) -> pd.DataFrame:
   """
   Main end-to-end churn risk prediction function using ChurnEnsembleModel.

   Orchestrates the complete prediction pipeline:
   1. Determine reference date (auto-calculated if not provided)
   2. Identify eligible donors (≥2 donations in lookback window)
   3. Prepare features (aggregation, engineering, normalization)
   4. Generate predictions using ensemble model
   5. Format and return results

   Parameters
   ----------
   duzenlenmis_data : pd.DataFrame
       Donation-level transaction records with date information.
       Must contain a donor ID column and a date column.

   toplulasmis_data : pd.DataFrame
       Aggregated donor-level metrics generated by the training pipeline.
       Must contain pre-calculated features and statistics.

   cutoff_date : str, datetime, optional
       Reference date for eligibility determination (format: YYYY-MM-DD).
       If None, automatically set to the last day of the month preceding
       the latest donation in the dataset.

   donor_col : str, default "Bagisci No"
       Donor ID column name. Auto-detected from common variations if not found.

   pencere_gun : int, default 365
       Lookback window in days for determining eligibility.
       Must match the training pipeline's window size.

   debug : bool, default False
       Enable detailed debug output for troubleshooting

   Returns
   -------
   pd.DataFrame
       DataFrame with columns:
       - Donor ID (e.g., "Bagisci No" or column specified in donor_col)
       - "Terk Riski": Churn risk percentage (format: "%XX.XX")

   Examples
   --------
   >>> # Basic prediction with automatic cutoff date
   >>> result = terk_riski_tahmin_et(
   ...     duzenlenmis_data=raw_donations,
   ...     toplulasmis_data=aggregated_donors
   ... )

   >>> # Prediction with debug output
   >>> result = terk_riski_tahmin_et(
   ...     duzenlenmis_data=raw_donations,
   ...     toplulasmis_data=aggregated_donors,
   ...     debug=True
   ... )
   """

   if debug:
       print("🔄 Tahmin işlemi başlatılıyor...")

   # Auto-calculate cutoff_date if not provided
   if cutoff_date is None:
       active_date_col = _resolve_column_name(
           duzenlenmis_data,
           ["Bağış Tarihi", "Bagis Tarihi", "Tarih"]
       )

       parsed_dates = pd.to_datetime(
           duzenlenmis_data[active_date_col], errors="coerce", dayfirst=True
       )

       if parsed_dates.isna().all():
           raise ValueError(
               f"cutoff_date not specified and no valid dates found in '{active_date_col}' column."
           )

       max_date = parsed_dates.max()
       # Set cutoff to last day of previous month
       cutoff_date = max_date.replace(day=1) - pd.Timedelta(days=1)

   # Format cutoff date as ISO string
   cutoff_date = pd.to_datetime(cutoff_date).strftime("%Y-%m-%d")

   if debug:
       print(f"📅 Cutoff tarihi: {cutoff_date}")

   # Auto-resolve donor ID column name
   active_donor_col = _resolve_column_name(
       duzenlenmis_data,
       [donor_col, "Bagisci No", "Bagisci", "Bağışçı No", "No"]
   )

   # Create prediction dataset with feature engineering
   X, ids = generate_prediction_data(
       duzenlenmis_data=duzenlenmis_data,
       toplulasmis_data=toplulasmis_data,
       cutoff_date=cutoff_date,
       donor_col=active_donor_col,
       pencere_gun=pencere_gun,
       debug=debug
   )

   # Score churn risk
   terk_data = score_churn_risk(
       X=X,
       ids=ids,
       donor_col=active_donor_col,
       debug=debug
   )

   # Format as percentages (e.g., 0.75 -> "%75.00")
   if not terk_data.empty and "Terk Riski" in terk_data.columns:
       terk_data["Terk Riski"] = (terk_data["Terk Riski"] * 100).map(lambda x: f"%{x:.2f}")

   if debug:
       print("✅ Tahmin işlemi tamamlandı!")

   return terk_data


def predict_churn_risk_batch(
    duzenlenmis_data: pd.DataFrame,
    toplulasmis_data: pd.DataFrame,
    cutoff_dates,
    donor_col: str = "Bagisci No",
    pencere_gun: int = 365,
) -> dict:
    """
    Birden fazla cutoff tarihinde tahmin yapar ve sonuçları dict'te döner.

    Parametreler
    ----------
    cutoff_dates : list or iterable
        Tahmin yapılacak tarihler listesi

    Dönüş
    ------
    dict
        cutoff_date -> terk riski DataFrame eşlemesi
    """
    results = {}

    for cutoff in cutoff_dates:
        print(f"📊 Tahmin: {cutoff}")
        result = predict_churn_risk(
            duzenlenmis_data=duzenlenmis_data,
            toplulasmis_data=toplulasmis_data,
            cutoff_date=cutoff,
            donor_col=donor_col,
            pencere_gun=pencere_gun
        )
        results[str(cutoff)] = result

    return results


if __name__ == "__main__":
    bagislar = pd.read_csv("bagislar.csv")
    bagiscilar = pd.read_csv("bagiscilar.csv")

    from utils.consolidator import consolidate

    toplu_data = consolidate(df=bagislar, df_bagiscilar=bagiscilar)

    # Enable debug output
    terkler = predict_churn_risk(
        duzenlenmis_data=bagislar,
        toplulasmis_data=toplu_data,
        debug=True
    )

    print(terkler)
    print(terkler.columns)
