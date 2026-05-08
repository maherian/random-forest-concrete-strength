from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "concrete_strength_data.csv"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

RANDOM_STATE = 42
TARGET_COLUMN = "CS (MPa)"

COLUMN_ALIASES = {
    "cement": "Cement (kg/m3)",
    "cement (kg/m3)": "Cement (kg/m3)",
    "water": "Water (kg/m3)",
    "water (kg/m3)": "Water (kg/m3)",
    "fa": "FA (kg/m3)",
    "fly ash": "FA (kg/m3)",
    "fly_ash": "FA (kg/m3)",
    "fa (kg/m3)": "FA (kg/m3)",
    "ggbs": "GGBS (kg/m3)",
    "slag": "GGBS (kg/m3)",
    "ggbs (kg/m3)": "GGBS (kg/m3)",
    "sf": "SF (kg/m3)",
    "ms": "SF (kg/m3)",
    "silica fume": "SF (kg/m3)",
    "micro silica": "SF (kg/m3)",
    "micro_silica": "SF (kg/m3)",
    "sf (kg/m3)": "SF (kg/m3)",
    "ns": "NS (kg/m3)",
    "ns (kg/m3)": "NS (kg/m3)",
    "w/b": "W/B",
    "w/c": "W/B",
    "wtoc": "W/B",
    "water/binder": "W/B",
    "water_cement_ratio": "W/B",
    "f-agg": "F-Agg (kg/m3)",
    "fagg": "F-Agg (kg/m3)",
    "fine aggregate": "F-Agg (kg/m3)",
    "fine_aggregate": "F-Agg (kg/m3)",
    "f-agg (kg/m3)": "F-Agg (kg/m3)",
    "c-agg": "C-Agg (kg/m3)",
    "cagg": "C-Agg (kg/m3)",
    "coarse aggregate": "C-Agg (kg/m3)",
    "coarse_aggregate": "C-Agg (kg/m3)",
    "c-agg (kg/m3)": "C-Agg (kg/m3)",
    "sp": "SP (kg/m3)",
    "superplasticizer": "SP (kg/m3)",
    "sp (kg/m3)": "SP (kg/m3)",
    "age": "Age (day)",
    "age (day)": "Age (day)",
    "age (days)": "Age (day)",
    "ssa": "SSA (m2/g)",
    "size": "SSA (m2/g)",
    "ssa (m2/g)": "SSA (m2/g)",
    "cs": "CS (MPa)",
    "cs (mpa)": "CS (MPa)",
    "compressive strength": "CS (MPa)",
    "compressive_strength": "CS (MPa)",
}

FEATURE_COLUMNS = [
    "Cement (kg/m3)",
    "Water (kg/m3)",
    "FA (kg/m3)",
    "GGBS (kg/m3)",
    "SF (kg/m3)",
    "NS (kg/m3)",
    "W/B",
    "F-Agg (kg/m3)",
    "C-Agg (kg/m3)",
    "SP (kg/m3)",
    "Age (day)",
    "SSA (m2/g)",
]


def normalize_column_name(column: str) -> str:
    clean = str(column).strip()
    clean = clean.replace("\u00b3", "3").replace("\u00b2", "2")
    clean = clean.replace("\u00c2\u00b3", "3").replace("\u00c2\u00b2", "2")
    clean = " ".join(clean.split())
    return COLUMN_ALIASES.get(clean.lower(), clean)


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place concrete_strength_data.csv in the data directory before running."
        )

    data = pd.read_csv(path)
    data = data.rename(columns={column: normalize_column_name(column) for column in data.columns})

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    data = data[required_columns].copy()
    data = data.apply(pd.to_numeric, errors="coerce")
    before = len(data)
    data = data.dropna().reset_index(drop=True)
    dropped = before - len(data)
    if dropped:
        warnings.warn(f"Dropped {dropped} rows containing missing or non-numeric values.", stacklevel=2)
    return data


def split_dataset(data: pd.DataFrame):
    features = data[FEATURE_COLUMNS]
    target = data[TARGET_COLUMN]
    return train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )


def tune_random_forest(x_train: pd.DataFrame, y_train: pd.Series) -> RandomForestRegressor:
    model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    search_space = {
        "n_estimators": [300, 500, 700, 900],
        "max_depth": [None, 8, 12, 16, 20, 28],
        "min_samples_split": [2, 4, 6, 10],
        "min_samples_leaf": [1, 2, 3, 4],
        "max_features": ["sqrt", "log2", 0.6, 0.8, 1.0],
        "bootstrap": [True],
    }
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=search_space,
        n_iter=40,
        cv=5,
        scoring="neg_root_mean_squared_error",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(x_train, y_train)
    return search.best_estimator_


def evaluate_regression(y_true: pd.Series, y_pred: np.ndarray, baseline_mae: float) -> dict[str, float]:
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    nonzero_mask = y_true_array != 0
    mape = (
        np.mean(np.abs((y_true_array[nonzero_mask] - y_pred_array[nonzero_mask]) / y_true_array[nonzero_mask]))
        * 100
    )
    mse = mean_squared_error(y_true_array, y_pred_array)
    mae = mean_absolute_error(y_true_array, y_pred_array)

    return {
        "MAE": mae,
        "MASE": mae / baseline_mae,
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_true_array, y_pred_array),
        "MAPE (%)": mape,
    }


def evaluate_splits(
    model: RandomForestRegressor,
    splits: dict[str, tuple[pd.DataFrame, pd.Series]],
    train_target: pd.Series,
) -> pd.DataFrame:
    baseline_value = train_target.mean()
    baseline_mae = mean_absolute_error(train_target, np.full(len(train_target), baseline_value))
    rows = []

    for split_name, (features, target) in splits.items():
        predictions = model.predict(features)
        metrics = evaluate_regression(target, predictions, baseline_mae)
        rows.append({"split": split_name, **metrics})

    return pd.DataFrame(rows)


def save_prediction_examples(
    model: RandomForestRegressor,
    splits: dict[str, tuple[pd.DataFrame, pd.Series]],
) -> pd.DataFrame:
    examples_by_split = []

    for split_name, (features, target) in splits.items():
        split_examples = features.copy()
        split_examples.insert(0, "split", split_name)
        split_examples["actual_CS_MPa"] = target.to_numpy()
        split_examples["predicted_CS_MPa"] = model.predict(features)
        split_examples["residual_MPa"] = split_examples["actual_CS_MPa"] - split_examples["predicted_CS_MPa"]
        examples_by_split.append(split_examples)

    examples = pd.concat(examples_by_split, axis=0).reset_index(drop=True)
    examples.to_csv(RESULTS_DIR / "prediction_examples.csv", index=False)
    return examples


def save_permutation_importance(
    model: RandomForestRegressor,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    result = permutation_importance(
        model,
        x_test,
        y_test,
        n_repeats=20,
        random_state=RANDOM_STATE,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    importance = pd.DataFrame(
        {
            "feature": x_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    importance.to_csv(RESULTS_DIR / "permutation_importance.csv", index=False)
    return importance


def plot_actual_vs_predicted(
    y_true: pd.Series,
    y_pred: np.ndarray,
    output_path: Path,
    split_name: str,
) -> None:
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, s=42, color="#1f77b4", edgecolor="white", linewidth=0.4)
    limits = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(limits, limits, color="#d62728", linewidth=2)
    plt.xlabel("Measured CS (MPa)")
    plt.ylabel("Predicted CS (MPa)")
    plt.title(f"Random Forest {split_name.title()} Predictions")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_prediction_errors(y_true: pd.Series, y_pred: np.ndarray, output_path: Path) -> None:
    residuals = y_true.to_numpy() - y_pred
    plt.figure(figsize=(7, 5))
    sns.histplot(residuals, bins=30, kde=True, color="#1f77b4")
    plt.axvline(0, color="#d62728", linewidth=2)
    plt.xlabel("Residual (MPa)")
    plt.ylabel("Count")
    plt.title("Test Prediction Errors")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_feature_importance(importance: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 6))
    sns.barplot(data=importance, y="feature", x="importance_mean", color="#1f77b4")
    plt.xlabel("Permutation importance")
    plt.ylabel("")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def prepare_output_directories() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)


def main() -> None:
    prepare_output_directories()
    data = load_dataset()
    x_train, x_test, y_train, y_test = split_dataset(data)
    splits = {
        "train": (x_train, y_train),
        "test": (x_test, y_test),
    }

    model = tune_random_forest(x_train, y_train)
    metrics = evaluate_splits(model, splits, y_train)
    metrics.to_csv(RESULTS_DIR / "model_metrics.csv", index=False)

    y_test_pred = model.predict(x_test)
    prediction_examples = save_prediction_examples(model, splits)
    importance = save_permutation_importance(model, x_test, y_test)

    for split_name, (features, target) in splits.items():
        predictions = model.predict(features)
        plot_actual_vs_predicted(
            target,
            predictions,
            FIGURES_DIR / f"actual_vs_predicted_{split_name}.png",
            split_name,
        )

    plot_prediction_errors(y_test, y_test_pred, FIGURES_DIR / "prediction_errors_test.png")
    plot_feature_importance(importance, FIGURES_DIR / "feature_importance.png")

    joblib.dump(model, MODELS_DIR / "random_forest_concrete_strength.joblib")
    with open(RESULTS_DIR / "best_model_parameters.json", "w", encoding="utf-8") as file:
        json.dump(model.get_params(), file, indent=2, default=str)

    print("Random Forest workflow complete.")
    print(metrics.round(3).to_string(index=False))
    print(f"Saved {len(prediction_examples)} prediction rows across train and test splits.")


if __name__ == "__main__":
    main()
