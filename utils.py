
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

COMMON_NUMERIC = [
    "Age","Weight","Height","Duration","Heart_Rate","Body_Temp","Steps","Distance"
]
COMMON_CATEG = ["Gender","Exercise_Type"]

def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # strip and unify names
    df = df.copy()
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    # title-case common columns variants
    rename_map = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "calories":
            rename_map[c] = "Calories"
        elif cl in {"age"}:
            rename_map[c] = "Age"
        elif cl in {"weight","weight_kg"}:
            rename_map[c] = "Weight"
        elif cl in {"height","height_cm","height_m"}:
            rename_map[c] = "Height"
        elif cl in {"duration","time","minutes"}:
            rename_map[c] = "Duration"
        elif cl in {"heart_rate","heartrate","hr"}:
            rename_map[c] = "Heart_Rate"
        elif cl in {"body_temp","temperature","temp"}:
            rename_map[c] = "Body_Temp"
        elif cl in {"steps"}:
            rename_map[c] = "Steps"
        elif cl in {"distance","km"}:
            rename_map[c] = "Distance"
        elif cl in {"gender","sex"}:
            rename_map[c] = "Gender"
        elif cl in {"exercise_type","activity_type","workout_type","type"}:
            rename_map[c] = "Exercise_Type"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = _canonicalize_columns(df)
    # BMI if possible
    if "Weight" in df.columns and "Height" in df.columns:
        # infer unit for height
        h = df["Height"].astype(float)
        # If median height > 3, assume centimeters
        if h.dropna().median() and h.dropna().median() > 3:
            h_m = h / 100.0
        else:
            h_m = h
        with np.errstate(divide='ignore', invalid='ignore'):
            bmi = df["Weight"].astype(float) / (h_m ** 2)
        df["BMI"] = bmi.replace([np.inf, -np.inf], np.nan)
    return df

def detect_features(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    df = _canonicalize_columns(df)
    num = [c for c in COMMON_NUMERIC + ["BMI"] if c in df.columns and c != "Calories"]
    cat = [c for c in COMMON_CATEG if c in df.columns]
    # ensure at least one numeric
    if not num and "Calories" in df.columns:
        # try any numeric columns except target
        num = [c for c in df.select_dtypes(include='number').columns if c != "Calories"]
    return num, cat

def build_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    num_pipe = make_num_pipeline()
    cat_pipe = make_cat_pipeline()
    transformers = []
    if num_cols:
        transformers.append(("num", num_pipe, num_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipe, cat_cols))
    return ColumnTransformer(transformers)

def make_num_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

def make_cat_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])

from sklearn.pipeline import Pipeline
