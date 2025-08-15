
import json, os
import numpy as np
import pandas as pd
import streamlit as st
from joblib import load
from utils import engineer_features, detect_features
import plotly.express as px

st.set_page_config(page_title="Calories Burnt ML App", layout="wide")

DATA_PATH = "data/calories.csv"
ARTIFACTS = "artifacts"

st.title("🔥 Calories Burnt Prediction — Pro Version")
st.caption("Features: multiple models, SHAP explainability, visualizations, and cloud-ready")

# Load dataset (optional, for EDA)
df = None
if os.path.exists(DATA_PATH):
    try:
        df = pd.read_csv(DATA_PATH)
        df = engineer_features(df)
    except Exception as e:
        st.warning(f"Could not load data: {e}")

# Load artifacts
def load_artifacts():
    feats = {"num_cols": [], "cat_cols": [], "feature_order": []}
    try:
        with open(os.path.join(ARTIFACTS, "features.json")) as f:
            feats = json.load(f)
    except Exception as e:
        st.warning("Train the models first: run `python train.py --data data/calories.csv`")
    pre = None
    try:
        pre = load(os.path.join(ARTIFACTS, "preprocessor.joblib"))
    except Exception:
        pass

    models = {}
    for name in ["RandomForest","XGBoost","LinearRegression"]:
        path = os.path.join(ARTIFACTS, f"{name}.joblib")
        if os.path.exists(path):
            try:
                models[name] = load(path)
            except Exception:
                pass
    metrics = None
    metrics_path = os.path.join(ARTIFACTS, "metrics.csv")
    if os.path.exists(metrics_path):
        metrics = pd.read_csv(metrics_path)
    best = None
    best_path = os.path.join(ARTIFACTS, "best_model.json")
    if os.path.exists(best_path):
        with open(best_path) as f:
            best = json.load(f)
    return feats, pre, models, metrics, best

feats, preprocessor, models, metrics_df, best_info = load_artifacts()

# Sidebar navigation
page = st.sidebar.radio("Go to", ["Predict", "Compare Models", "Explore Data", "How to Deploy"])

def user_inputs():
    st.subheader("Enter Your Details")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=10, max_value=90, value=25)
        duration = st.number_input("Duration (minutes)", min_value=1, max_value=240, value=30)
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=220, value=120)
    with col2:
        weight = st.number_input("Weight (kg)", min_value=25.0, max_value=200.0, value=70.0, step=0.1)
        height = st.number_input("Height (cm)", min_value=100.0, max_value=220.0, value=170.0, step=0.1)
        body_temp = st.number_input("Body Temp (°C)", min_value=34.0, max_value=41.0, value=37.0, step=0.1)
    with col3:
        gender = st.selectbox("Gender", ["", "Male", "Female"])
        ex_type = st.selectbox("Exercise Type", ["", "Running", "Walking", "Cycling", "Gym", "Yoga"])
        steps = st.number_input("Steps (optional)", min_value=0, max_value=50000, value=0)
    bmi = weight / ((height/100.0)**2)
    st.caption(f"Calculated BMI: **{bmi:.2f}**")
    row = {
        "Age": age, "Duration": duration, "Heart_Rate": heart_rate,
        "Weight": weight, "Height": height, "Body_Temp": body_temp,
        "Gender": gender if gender else np.nan, "Exercise_Type": ex_type if ex_type else np.nan,
        "Steps": steps, "BMI": bmi
    }
    return pd.DataFrame([row])

def transform_for_model(df_row: pd.DataFrame):
    # features order from artifacts
    used = feats.get("feature_order", [])
    X = df_row.reindex(columns=used, fill_value=np.nan)
    if preprocessor is None:
        st.error("Preprocessor not found. Please train first: `python train.py --data data/calories.csv`")
        st.stop()
    Xp = preprocessor.transform(X)
    return Xp

if page == "Predict":
    st.write("Select a model and get a prediction. Models are trained from your dataset.")
    available_models = list(models.keys())
    if not available_models:
        st.warning("No trained models found. Run `python train.py --data data/calories.csv` first.")
    else:
        default_model = best_info.get("best_model") if best_info else available_models[0]
        model_name = st.selectbox("Choose model", available_models, index=max(0, available_models.index(default_model) if default_model in available_models else 0))
        form = st.form("pred_form")
        with form:
            row = user_inputs()
            submitted = st.form_submit_button("Predict Calories 🔮")
        if submitted:
            Xp = transform_for_model(row[feats.get("feature_order", [])])
            pred = models[model_name].predict(Xp)[0]
            st.success(f"Estimated Calories Burnt: **{pred:.2f}**")
            # SHAP explainability (tree models only)
            if model_name in ["RandomForest","XGBoost"]:
                try:
                    import shap
                    explainer = shap.TreeExplainer(models[model_name])
                    # Use a small random background for speed if df exists
                    if df is not None:
                        X_bg = df[feats["feature_order"]].head(200)
                        X_bg = preprocessor.transform(X_bg)
                        explainer = shap.TreeExplainer(models[model_name], data=X_bg)
                    sv = explainer.shap_values(Xp)
                    st.subheader("Feature Importance (SHAP)")
                    shap_df = pd.DataFrame({
                        "feature": feats["feature_order"],
                        "shap_value": np.abs(sv[0]).ravel() if isinstance(sv, np.ndarray) else np.abs(sv).ravel()
                    }).sort_values("shap_value", ascending=False)
                    st.dataframe(shap_df)
                    fig = px.bar(shap_df.head(15), x="shap_value", y="feature", orientation="h")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.info(f"SHAP plot unavailable: {e}")
            else:
                st.caption("Explanation available for tree-based models (RandomForest / XGBoost).")

elif page == "Compare Models":
    st.subheader("Model Comparison")
    if metrics_df is None:
        st.warning("Metrics not found. Train models first.")
    else:
        st.dataframe(metrics_df)
        fig = px.bar(metrics_df, x="model", y="rmse", title="RMSE (lower is better)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("We also save MAE and R²; pick the model with the lowest RMSE.")

elif page == "Explore Data":
    st.subheader("Quick Exploratory Analysis")
    if df is None:
        st.warning("Add your CSV at `data/calories.csv` to explore.")
    else:
        st.write("Sample rows")
        st.dataframe(df.head(20))
        num_cols, cat_cols = detect_features(df)
        if num_cols:
            c = st.selectbox("Numeric column to plot", num_cols)
            fig = px.histogram(df, x=c, nbins=30, title=f"Distribution of {c}")
            st.plotly_chart(fig, use_container_width=True)
        if "Calories" in df.columns:
            possible_x = [c for c in df.select_dtypes(include='number').columns if c != "Calories"]
            if possible_x:
                xcol = st.selectbox("X for scatter vs Calories", possible_x)
                fig2 = px.scatter(df, x=xcol, y="Calories", trendline="ols", title=f"{xcol} vs Calories")
                st.plotly_chart(fig2, use_container_width=True)

elif page == "How to Deploy":
    st.subheader("Cloud Deployment (Easy & Pro)")
    st.markdown("""
**Option A — Streamlit Community Cloud (Easy)**
1. Push this folder to GitHub (include `app.py`, `utils.py`, `requirements.txt`, and the `artifacts/` folder after training).
2. Go to share.streamlit.io (Streamlit Community Cloud), connect your repo, select `app.py`.
3. Add secrets if needed; click **Deploy**.

**Option B — Google Cloud Run (Docker, Pro)**
1. Build Docker image:
   ```bash
   docker build -t calories-app .
   docker run -p 8501:8501 calories-app
   ```
2. Push to Google Artifact Registry and deploy to Cloud Run.
""")

st.sidebar.markdown("**Tip:** First run: `python train.py --data data/calories.csv` to create artifacts. Then `streamlit run app.py`.")
