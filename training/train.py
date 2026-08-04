"""Train an XGBoost model for IPO profitability prediction.

Uses 5-fold stratified cross-validation for reliable performance
estimation. The final model is retrained on the full dataset.
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_URL = (
    "https://raw.githubusercontent.com/AadiParkhi/IPO-Data-India-2010-2025"
    "/main/Initial%20Public%20Offering.csv"
)

SELECTED_FEATURES = [
    "Total_Sub",
    "QIB",
    "HNI",
    "HNI_pct",
    "RII_pct",
    "Issue_Size_crores",
]

MODEL_PATH = "models/model.pkl"
N_FOLDS = 5


def load_data(url: str) -> pd.DataFrame:
    """Download and load the IPO dataset."""
    df = pd.read_csv(url, encoding="latin1")
    logger.info(f"Loaded {len(df)} rows")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names and convert dtypes."""
    df = df.rename(
        columns={
            "Issue_Size(crores)": "Issue_Size_crores",
            "Offer Price": "Offer_Price",
            "List Price": "List_Price",
            "Listing Gain": "Listing_Gain",
        }
    )
    df["Issue_Size_crores"] = (
        df["Issue_Size_crores"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .astype(float)
    )
    df["Offer_Price"] = (
        df["Offer_Price"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .astype(float)
    )
    df = df.dropna(subset=["Listing_Gain", "QIB", "HNI", "RII"]).reset_index(drop=True)
    return df


def add_day_of_week(df: pd.DataFrame) -> pd.DataFrame:
    """Extract day of week (0=Monday, 6=Sunday) from Date column.

    Tries two date formats: dd-mm-yyyy (older rows) and mm-dd-yy (newer rows).
    """
    dates = pd.to_datetime(df["Date"], format="%d-%m-%Y", errors="coerce")
    still_null = dates.isna()
    dates[still_null] = pd.to_datetime(
        df.loc[still_null, "Date"], format="%m-%d-%y", errors="coerce"
    )
    df["Date_parsed"] = dates
    df["day_of_week"] = df["Date_parsed"].dt.dayofweek
    df = df.dropna(subset=["day_of_week"])
    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create total subscription and ratio features.

    Total_Sub captures overall demand magnitude; ratio features capture
    the composition of demand (which investor category is driving it).
    """
    df["Total_Sub"] = df["QIB"] + df["HNI"] + df["RII"]
    eps = 1e-6
    df["QIB_pct"] = df["QIB"] / (df["Total_Sub"] + eps)
    df["HNI_pct"] = df["HNI"] / (df["Total_Sub"] + eps)
    df["RII_pct"] = df["RII"] / (df["Total_Sub"] + eps)
    return df


def tune_hyperparameters(
    X: pd.DataFrame, y: np.ndarray, scale_pos_weight: float
) -> dict:
    """Search for optimal XGBoost hyperparameters using grid search.

    Uses 3-fold cross-validation (inner loop) with ROC AUC scoring.
    """
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    }

    model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbosity=0,
    )

    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    search = GridSearchCV(
        model,
        param_grid,
        cv=inner_cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X, y)
    logger.info(
        f"Best hyperparameters: {search.best_params_} "
        f"(CV AUC: {search.best_score_:.4f})"
    )
    return search.best_params_


def train_model(
    X_train: pd.DataFrame, y_train: pd.Series, scale_pos_weight: float
) -> XGBClassifier:
    """Train an XGBoost classifier with class imbalance handling."""
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


def find_best_threshold(
    y_true: np.ndarray, y_probs: np.ndarray
) -> tuple[float, float]:
    """Search for the decision threshold that maximizes F1 score.

    Args:
        y_true: Ground-truth labels.
        y_probs: Predicted probabilities for the positive class.

    Returns:
        (best_threshold, best_f1)
    """
    best_f1 = 0.0
    best_threshold = 0.5
    for threshold in np.arange(0.30, 0.71, 0.05):
        preds = (y_probs > threshold).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    logger.info(f"Best threshold {best_threshold:.2f} achieved F1 {best_f1:.4f}")
    return best_threshold, best_f1


def main() -> None:
    """Run the full training pipeline with 5-fold cross-validation."""
    logger.info("Loading data...")
    raw = load_data(DATA_URL)

    logger.info("Cleaning data...")
    df = clean_data(raw)

    logger.info("Engineering day_of_week...")
    df = add_day_of_week(df)

    logger.info("Adding engineered features...")
    df = add_engineered_features(df)

    df["target"] = (df["Listing_Gain"] > 0).astype(int)
    logger.info(f"Target distribution:\n{df['target'].value_counts(normalize=True)}")

    X = df[SELECTED_FEATURES]
    y = df["target"].values
    logger.info(f"Final feature matrix: {X.shape}")

    neg, pos = (y == 0).sum(), (y == 1).sum()
    scale_pos_weight = neg / pos

    logger.info("Tuning hyperparameters...")
    best_params = tune_hyperparameters(X, y, scale_pos_weight)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    oof_probs = np.zeros(len(X))
    oof_true = np.zeros(len(X))
    fold_aucs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_f, X_val_f = X.iloc[train_idx], X.iloc[val_idx]
        y_train_f, y_val_f = y[train_idx], y[val_idx]

        model = XGBClassifier(
            **best_params,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbosity=0,
        )
        model.fit(X_train_f, y_train_f)

        val_probs = model.predict_proba(X_val_f)[:, 1]
        oof_probs[val_idx] = val_probs
        oof_true[val_idx] = y_val_f

        fold_auc = roc_auc_score(y_val_f, val_probs)
        fold_aucs.append(fold_auc)
        logger.info(
            f"Fold {fold + 1} — ROC AUC: {fold_auc:.4f}, "
            f"scale_pos_weight: {scale_pos_weight:.2f}"
        )

    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    logger.info(f"Cross-validation ROC AUC: {mean_auc:.4f} ± {std_auc:.4f}")

    best_threshold, best_f1 = find_best_threshold(oof_true, oof_probs)

    oof_preds = (oof_probs > best_threshold).astype(int)
    logger.info(f"OOF F1 Score: {f1_score(oof_true, oof_preds):.4f}")
    logger.info(
        f"OOF Classification Report:\n"
        f"{classification_report(oof_true, oof_preds)}"
    )

    logger.info("Retraining on full dataset with best hyperparameters...")
    final_model = XGBClassifier(
        **best_params,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbosity=0,
    )
    final_model.fit(X, y)

    importance = dict(
        sorted(
            zip(SELECTED_FEATURES, final_model.feature_importances_),
            key=lambda x: x[1],
            reverse=True,
        )
    )
    logger.info("Feature importance:")
    for k, v in importance.items():
        logger.info(f"  {k}: {v:.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(
        {
            "model": final_model,
            "features": SELECTED_FEATURES,
            "threshold": best_threshold,
        },
        MODEL_PATH,
    )
    logger.success(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
