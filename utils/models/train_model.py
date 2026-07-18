# File Management and Exporting Models
import joblib
from pathlib import Path
from typing import Dict, Any

# Data Manipulation
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Data Visualisation
import matplotlib.pyplot as plt
import seaborn as sns

# Models
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# Parameter Optimization
import optuna

# Evaluation & CV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from sklearn.base import clone

# Hide warnings and display options
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="tqdm")
optuna.logging.set_verbosity(optuna.logging.WARNING)
plt.style.use("default")
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:,.2f}".format)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)

# Reading The Dataframe

# Uncomment these lines to generate a new file.
# from utils.demo veri import demo_data_olustur
# from utils.egitim_verisi_olusturucu import egitim_verisi_olustur

# demo_data = demo_data_olustur(n_bagisci=25000, n_bagis=2500000)
# train_data = egitim_verisi_olustur(df=demo_data)
# train_data.to_csv("train_df.csv", index=False)

DATA_PATH = "train_df.csv"
TARGET = "churn"

df = pd.read_csv(DATA_PATH)

# Remove unnecessary columns

df = df.drop(columns=[
    # Identifier columns
    "Bagisci No",
    "Cep Telefon",

    # Date columns
    "Bagis Tarihi_min",
    "Bagis Tarihi_max",

    # Constant columns
    "Personel_mode",
    "Personel_nunique",
    "Ulke_mode",

    # Duplicate aggregate columns
    "Toplam Tutar_sum",
    "Toplam Tutar_mean",
    "Toplam Tutar_median",
    "Toplam Tutar_min",
    "Toplam Tutar_max",
    "Toplam Tutar_count",
])

print(f"New Shape: {df.shape}")

def engineer_features(df):
    """
    Performs Feature Engineering on the donor dataset.
    Returns the processed DataFrame ready for modeling or exporting.
    """
    # Create a copy of the dataframe to avoid setting with copy warning
    df_fe = df.copy()
    
    # ---------------------------------------------------------
    # 1. RATIO & STATISTICAL FEATURES (Numeric Features)
    # ---------------------------------------------------------
    
    # Diversity ratios per donation count (Indicators of loyalty and engagement)
    df_fe['Tur_orani'] = df_fe['Tür_nunique'] / df_fe['Bağış Sayısı']
    df_fe['Kanal_orani'] = df_fe['Bagis Kanali_nunique'] / df_fe['Bağış Sayısı']
    df_fe['Kampanya_orani'] = df_fe['Kampanya Adi_nunique'] / df_fe['Bağış Sayısı']
    
    # Donation amount distribution metrics (Skewness / Spread)
    df_fe['Tutar_max_mean_orani'] = df_fe['Tutar_max'] / (df_fe['Tutar_mean'] + 1e-5)
    df_fe['Tutar_min_max_orani'] = df_fe['Tutar_min'] / (df_fe['Tutar_max'] + 1e-5)
    
    # Difference between Mean and Median (Indicates how stable the donor's behavior is)
    df_fe['Tutar_mean_median_farki'] = df_fe['Tutar_mean'] - df_fe['Tutar_median']
    
    # Calculated average amount
    df_fe['Hesaplanan_Ortalama_Tutar'] = df_fe['Tutar_sum'] / (df_fe['Bağış Sayısı'] + 1e-5)

    # ---------------------------------------------------------
    # 2. CATEGORICAL VARIABLE ENCODING
    # ---------------------------------------------------------
    
    # High Cardinality Features (Frequency Encoding)
    for col in ['Il_mode', 'Kampanya Adi_mode']:
        freq = df_fe[col].value_counts() / len(df_fe)
        df_fe[f'{col}_freq'] = df_fe[col].map(freq)
        
    # Low Cardinality Categorical Features (One-Hot Encoding)
    categorical_cols = ['Tür_mode', 'Bagis Kanali_mode', 'Para Birimi_mode', 'Odeme Sekli_mode']
    
    # Drop first=True to avoid the dummy variable trap
    df_fe = pd.get_dummies(df_fe, columns=categorical_cols, drop_first=True, dtype=int)
    
    # Drop the original high cardinality categorical columns
    df_fe.drop(columns=['Il_mode', 'Kampanya Adi_mode'], inplace=True, errors='ignore')
    
    # Drop perfectly correlated column
    df_fe.drop(columns=['Tutar_count'], inplace=True, errors='ignore')
    
    return df_fe

# --- Usage ---
df = engineer_features(df)

print(f"New Shape: {df.shape}")

import pandas as pd
import numpy as np

def time_based_train_test_split(
    df: pd.DataFrame,
    time_col: str = "Cutoff Tarihi",
    target: str = "churn",
    test_months: int = 6,
    embargo_days: int = 0,
    verbose: bool = True
):
    """
    Splits the dataset into Train and Test sets based strictly on a time boundary.
    
    Parameters:
    - df: Input DataFrame.
    - time_col: The date column used for splitting.
    - target: The churn (target) column name.
    - test_months: Number of months from the end of the timeline to use as test data.
    - embargo_days: Safe buffer zone (in days) between train and test sets.
    """
    df = df.copy()
    
    # Parse dates and drop rows missing the critical time identifier
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df[pd.notna(df[time_col])].copy()
    
    if df.empty:
        print("Warning: The dataset is empty after date cleaning!")
        return df.copy(), df.copy()

    # Find the latest available date in the dataset (Max Cutoff)
    max_cutoff = df[time_col].max()

    # 1. Calculate Test Start Date (Go back by test_months and start at the beginning of that month)
    test_start = (max_cutoff - pd.DateOffset(months=test_months - 1)).to_period("M").to_timestamp(how="start")

    # 2. Calculate Train End Date (Applying Embargo)
    train_end = test_start - pd.Timedelta(days=embargo_days)

    # 3. Perform Split
    test = df[df[time_col] >= test_start].copy()
    train = df[df[time_col] < train_end].copy()

    # 4. Output Split Summary (Verbose)
    if verbose:
        def get_set_info(data_subset, name):
            return {
                "Set Name": name,
                "Row Count": len(data_subset),
                "Min Date": data_subset[time_col].min().date() if len(data_subset) else None,
                "Max Date": data_subset[time_col].max().date() if len(data_subset) else None,
                "Churn Rate": f"{data_subset[target].mean() * 100:.2f}%" if len(data_subset) and target in data_subset.columns else "N/A"
            }

        print("=" * 80)
        print("TIME-BASED SPLIT SUMMARY")
        print("=" * 80)
        print(f"-> Train Cutoff Date < {train_end.date()} (Embargo: {embargo_days} days)")
        print(f"-> Test Start Date    >= {test_start.date()} (Max Date: {max_cutoff.date()})")
        print("-" * 80)
        
        tr_info = get_set_info(train, "Train")
        te_info = get_set_info(test, "Test")
        
        for k, v in tr_info.items():
            print(f"{k:<20}: Train -> {v:<15} | Test -> {te_info[k]}")
        print("=" * 80)

    # Clean up time column before returning
    train = train.drop(columns=[time_col], errors="ignore")
    test = test.drop(columns=[time_col], errors="ignore")

    return train, test

train, test = time_based_train_test_split(df)

print(f"Train Shape: {train.shape}")
print("*"*30)
print(f"Test Shape: {test.shape}")

def split_xy(df, target):
    X = df.drop(columns=[target])
    y = df[target]
    return X, y

X, y = split_xy(train, TARGET)
x, y_true = split_xy(test, TARGET)

def build_preprocessing_pipeline(X: pd.DataFrame, x: pd.DataFrame):
    """
    Automatically detects categorical and numerical columns, handles missing categorical values,
    and returns a fitted ColumnTransformer instance based on the training features (X).
    
    Parameters:
    - X: Training features (DataFrame)
    - x: Test features (DataFrame)
    
    Returns:
    - preprocessor: Fitted ColumnTransformer instance
    - X_clean: Cleaned training features (DataFrame with imputed missing categories)
    - x_clean: Cleaned test features (DataFrame with imputed missing categories)
    """
    # Copy the datasets to avoid setting with copy warnings
    X_clean = X.copy()
    x_clean = x.copy()
    
    # 1. Automatically separate Categorical and Numerical Columns
    cat_cols = X_clean.select_dtypes(exclude=["number"]).columns.tolist()
    num_cols = [c for c in X_clean.columns if c not in cat_cols]
    
    # 2. Safely Impute Missing Categorical Values
    CAT_NA_TOKEN = "__MISSING__"
    for df in [X_clean, x_clean]:
        df[cat_cols] = df[cat_cols].fillna(CAT_NA_TOKEN).astype(str)
        
    # 3. Configure the Preprocessing Pipeline (ColumnTransformer)
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop"
    )
    
    # Fit the preprocessor strictly on the cleaned training features (X_clean)
    preprocessor.fit(X_clean)
    
    return preprocessor, X_clean, x_clean

# Build the preprocessing pipeline and impute missing categories
preprocessor, X_clean, x_clean = build_preprocessing_pipeline(X, x)

# Transform the features into model-ready arrays (One-Hot Encoded & Numeric)
X_transformed = preprocessor.transform(X_clean)
x_transformed = preprocessor.transform(x_clean)

# Ensure data is in NumPy array format for foolproof indexing
X_arr = np.array(X_transformed)
y_arr = np.array(y)
x_test_arr = np.array(x_transformed)
y_test_arr = np.array(y_true)

# Global Stratified K-Fold setup
SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=1881)

# =====================================================================
# 1. OBJECTIVES
# =====================================================================
def get_lgbm_params(trial):
    return {
        "n_estimators": trial.suggest_categorical("n_estimators", [1000, 2000, 3000]),
        "learning_rate": trial.suggest_categorical("learning_rate", [0.01, 0.03, 0.05, 0.1]),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "objective": "binary",
        "random_state": 1881,
        "n_jobs": -1,
        "verbosity": -1,
    }

def get_xgb_params(trial):
    return {
        "n_estimators": trial.suggest_categorical("n_estimators", [1000, 2000, 3000]),
        "learning_rate": trial.suggest_categorical("learning_rate", [0.01, 0.03, 0.05, 0.1]),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "eval_metric": "logloss",
        "random_state": 1881,
        "n_jobs": -1,
    }

def get_cat_params(trial):
    return {
        "iterations": trial.suggest_categorical("iterations", [1000, 2000]),
        "learning_rate": trial.suggest_categorical("learning_rate", [0.01, 0.03, 0.05, 0.1]),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
        "eval_metric": "Logloss",
        "random_state": 1881,
        "verbose": 0,
        "thread_count": -1,
    }


# =====================================================================
# 2. TUNING EXECUTION & EVALUATION RUNNER
# =====================================================================
def objective_with_oof(trial, model_class, params_func, skf):
    params = params_func(trial)

    oof_preds = np.zeros(len(y_arr))
    scores = []

    print(f"\nTrial {trial.number + 1}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_arr, y_arr), start=1):
        model = model_class(**params)
        model.fit(X_arr[train_idx], y_arr[train_idx])

        preds = model.predict_proba(X_arr[val_idx])[:, 1]
        oof_preds[val_idx] = preds

        score = average_precision_score(y_arr[val_idx], preds)
        scores.append(score)

        print(f"  Fold {fold}: {score:.6f}")

    mean_score = np.mean(scores)
    std_score = np.std(scores)

    print(f"  Mean ± Std : {mean_score:.6f} ± {std_score:.6f}")

    return mean_score, oof_preds

# --- Tuning Execution ---
def run_hyperparameter_tuning(n_trials, skf):
    best_models = {}
    best_oof_preds = {}

    model_configs = {
        "LightGBM": (LGBMClassifier, get_lgbm_params),
        "XGBoost": (XGBClassifier, get_xgb_params),
        "CatBoost": (CatBoostClassifier, get_cat_params),
    }

    for name, (cls, param_func) in model_configs.items():
        print(f"\n🚀 Optimizing {name}...")

        study = optuna.create_study(direction="maximize")

        best_score = -np.inf
        best_oof = None

        def objective(trial):
            nonlocal best_score, best_oof

            score, oof = objective_with_oof(trial, cls, param_func, skf)

            if score > best_score:
                best_score = score
                best_oof = oof.copy()

            return score

        def print_callback(study, trial):
            print(f"  >>> Trial Score : {trial.value:.6f}")
            print(f"  >>> Best Score  : {study.best_value:.6f}")
            print("-" * 60)

        study.optimize(
            objective,
            n_trials=n_trials,
            callbacks=[print_callback]
        )

        # Save the best OOF predictions
        best_oof_preds[name] = best_oof

        # Train the final model on the entire dataset
        print(f"Training the final {name} model")
        
        # Recreate the full parameter dictionary (optimized + fixed parameters)
        final_params = param_func(optuna.trial.FixedTrial(study.best_params))
        
        final_model = cls(**final_params)
        final_model.fit(X_arr, y_arr)
        
        best_models[name] = final_model

    return best_models, best_oof_preds

# Usage
best_models, best_oof_preds = run_hyperparameter_tuning(n_trials=50, skf=SKF)
print(best_models)

def optimize_ensemble_weights(best_oof_preds, y_true_arr, n_trials):
    model_names = list(best_oof_preds.keys())
    
    def objective(trial):
        weights = [trial.suggest_float(f"w_{n}", 0.0, 1.0) for n in model_names]
        norm = [w / sum(weights) for w in weights]
        
        ensemble_oof = sum(norm[i] * best_oof_preds[name] for i, name in enumerate(model_names))
        return average_precision_score(y_true_arr, ensemble_oof)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    return {name: study.best_params[f"w_{name}"] / sum(study.best_params.values()) 
            for name in model_names}

# --- Execution ---
best_weights = best_weights = optimize_ensemble_weights(best_oof_preds, y_arr, n_trials=1000)
print("Optimal CV-based Weights:", best_weights)

# 1. Define a helper to calculate weighted predictions
def get_ensemble_preds(models, weights, X_input):
    ensemble_preds = np.zeros(len(X_input))
    for name, model in models.items():
        ensemble_preds += weights[name] * model.predict_proba(X_input)[:, 1]
    return ensemble_preds

# 2. Calculate Final Test Performance
# We use the 'best_weights' derived from the CV optimization process
final_test_preds = get_ensemble_preds(best_models, best_weights, x_transformed)
final_ap_score = average_precision_score(y_true, final_test_preds)

print("="*60)
print(f"Final Ensemble Test AP Score: {final_ap_score:.5f}")
print("="*60)

# 3. Optional: Compare with individual model performance on test set
print("\nIndividual Model Test Scores:")
for name, model in best_models.items():
    test_preds = model.predict_proba(x_transformed)[:, 1]
    ap = average_precision_score(y_true, test_preds)
    print(f"- {name}: {ap:.5f}")

class ChurnEnsembleModel:
    """
    A production-ready ensemble wrapper that encapsulates the preprocessing 
    pipeline and individual base models with optimized weights.
    """
    def __init__(self, preprocessor: Any, models: Dict[str, Any], weights: Dict[str, float]):
        self.preprocessor = preprocessor
        self.models = models
        self.weights = weights
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Applies transformation and calculates the weighted ensemble output."""
        # 1. Apply feature transformation
        X_transformed = self.preprocessor.transform(X)
        
        # 2. Compute weighted average of model predictions
        ensemble_preds = np.zeros(X_transformed.shape[0])
        for name, model in self.models.items():
            weight = self.weights[name]
            preds = model.predict_proba(X_transformed)[:, 1]
            ensemble_preds += weight * preds
            
        return ensemble_preds

# --- Configuration & Export ---
# Define directory and ensure it exists
OUTPUT_DIR = Path("../utils/models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = OUTPUT_DIR / "churn_prediction_model.joblib"

# Initialize and serialize the model
ensemble_model = ChurnEnsembleModel(preprocessor, best_models, best_weights)
joblib.dump(ensemble_model, MODEL_PATH)

print(f"✅ Ensemble model successfully saved to: {MODEL_PATH.resolve()}")
