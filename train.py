
"""
Train multiple models on the calories dataset with automatic feature detection,
save the best model + metrics for the Streamlit app.
Usage:
    python train.py --data data/calories.csv
"""
import argparse, json, os, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from joblib import dump
from utils import engineer_features, detect_features, build_preprocessor

warnings.filterwarnings("ignore")

def try_import_xgb():
    try:
        from xgboost import XGBRegressor
        return XGBRegressor
    except Exception:
        return None

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return {"model": name, "rmse": float(rmse), "mae": float(mae), "r2": float(r2)}

def main(data_path: str, artifacts_dir: str = "artifacts"):
    os.makedirs(artifacts_dir, exist_ok=True)
    df = pd.read_csv(data_path)
    df = engineer_features(df)
    if "Calories" not in df.columns:
        raise ValueError("Target column 'Calories' not found. Please ensure your CSV has a 'Calories' column.")
    num_cols, cat_cols = detect_features(df)

    features = num_cols + cat_cols
    X = df[features]
    y = df["Calories"]

    from utils import build_preprocessor
    preprocessor = build_preprocessor(num_cols, cat_cols)
    X_proc = preprocessor.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_proc, y, test_size=0.2, random_state=42)

    models = []
    models.append(("RandomForest", RandomForestRegressor(n_estimators=300, random_state=42)))
    XGBRegressor = try_import_xgb()
    if XGBRegressor is not None:
        models.append(("XGBoost", XGBRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=6,
            subsample=0.9, colsample_bytree=0.9, random_state=42, tree_method="hist"
        )))
    models.append(("LinearRegression", LinearRegression()))

    results = []
    fitted = {}
    for name, m in models:
        metrics = evaluate_model(name, m, X_train, X_test, y_train, y_test)
        results.append(metrics)
        fitted[name] = m

    # pick best by RMSE
    best = min(results, key=lambda d: d["rmse"])
    best_name = best["model"]
    best_model = fitted[best_name]

    # Save artifacts
    dump(preprocessor, os.path.join(artifacts_dir, "preprocessor.joblib"))
    for name, m in fitted.items():
        dump(m, os.path.join(artifacts_dir, f"{name}.joblib"))
    with open(os.path.join(artifacts_dir, "features.json"), "w") as f:
        json.dump({"num_cols": num_cols, "cat_cols": cat_cols, "feature_order": features}, f, indent=2)
    pd.DataFrame(results).to_csv(os.path.join(artifacts_dir, "metrics.csv"), index=False)
    with open(os.path.join(artifacts_dir, "best_model.json"), "w") as f:
        json.dump({"best_model": best_name}, f, indent=2)

    print("Training complete.")
    print("Models:", [r["model"] for r in results])
    print("Best:", best_name, "RMSE:", best["rmse"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data/calories.csv")
    parser.add_argument("--artifacts", type=str, default="artifacts")
    args = parser.parse_args()
    main(args.data, args.artifacts)
