# Calories Burnt ML Project (Placement-Ready)

**Level:** Beginner → Early-Intermediate (with pro features)  
**You will build:** Data cleaning → Multiple ML models → Streamlit app → Graphs/Explainability → Cloud deploy

---

## 0) Software (Windows/Mac/Linux)
1. Install **Python 3.10+** from python.org (check: `python --version`)
2. Install **VS Code**
3. (Optional) Install **Git** for deployment

---

## 1) Project Setup
1. Extract this folder or clone from GitHub
2. Open in VS Code
3. Create virtual env (optional but recommended)
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Mac/Linux
   source .venv/bin/activate
   ```
4. Install packages
   ```bash
   pip install -r requirements.txt
   ```

---

## 2) Data
- Put your CSV inside `data/` and name it **calories.csv**  
- Required: a **Calories** column (target)  
- Optional features supported: Age, Weight, Height (cm or m), Duration (min), Heart_Rate (bpm), Body_Temp (°C), Gender, Exercise_Type, Steps, Distance

> Code will auto-detect columns and create BMI if Height+Weight exist.

---

## 3) Train Models
```bash
python train.py --data data/calories.csv
```
Outputs into `artifacts/`:
- `preprocessor.joblib` (scaler + encoder)
- `RandomForest.joblib`, `XGBoost.joblib` (if xgboost installed), `LinearRegression.joblib`
- `features.json`, `metrics.csv`, `best_model.json`

---

## 4) Run the App
```bash
streamlit run app.py
```
Pages:
- **Predict:** User inputs (incl. Gender, Exercise Type, BMI auto), choose model, get prediction + SHAP (tree models)
- **Compare Models:** RMSE/MAE/R² table + bar chart
- **Explore Data:** Histograms, scatter vs Calories
- **How to Deploy:** Cloud steps

---

## 5) Deploy (Two Options)
### A) Streamlit Community Cloud (Easy)
1. Push code to GitHub (include `artifacts/` from training)
2. On Streamlit Cloud, pick your repo and `app.py`
3. Deploy (free tier).

### B) Google Cloud Run (Pro)
1. Build docker image:
   ```bash
   docker build -t calories-app .
   docker run -p 8501:8501 calories-app
   ```
2. Push to Artifact Registry and deploy to Cloud Run.

---

## 6) Video Demo Script (Short)
1. **Intro (15s):** Problem + dataset
2. **EDA (30s):** 1–2 charts (Explore Data page)
3. **Modeling (30s):** Compare Models page → pick best (RMSE)
4. **App (45s):** Input values → Predict → SHAP bar
5. **Cloud (15s):** Show live app running
6. **Outro (10s):** Results + next steps

---

## Common Issues
- **xgboost install error:** continue without it; RandomForest + LinearRegression still work.
- **No 'Calories' column:** rename your target to `Calories`.
- **Height units:** code auto-detects cm vs meters.

Good luck! 🚀
