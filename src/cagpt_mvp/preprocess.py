# src/cagpt_mvp/preprocess.py
import json, numpy as np, pandas as pd
from pathlib import Path

META_DIR = Path(__file__).resolve().parent.parent.parent / "metadata"

def load_meta(name: str) -> dict:
    path = META_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"metadata file not found: {path}")
    return json.loads(path.read_text())

def apply_metadata(df: pd.DataFrame, meta: dict):
    """返回 (X_df, y_series) —— 未做 One-Hot / 标准化，交由 baseline 内 ColumnTransformer 完成"""
    df = df.replace(meta.get("missing_codes", []), np.nan)

    target = meta["target_primary"]
    if target not in df.columns:
        raise KeyError(f"target `{target}` not in DataFrame")

    feature_cols = []
    cat_cols, num_cols = [], []

    for col, cfg in meta["fields"].items():
        if cfg["role"] == "meta":
            continue
        if cfg["role"] == "target":
            continue      # 预留二级目标时可用
        if col not in df.columns:
            print(f"[warn] column `{col}` not found, skip")
            continue

        feature_cols.append(col)
        if cfg["dtype"] == "continuous":
            num_cols.append(col)
        else:
            cat_cols.append(col)

    X = df[feature_cols].copy()
    y = df[target].copy()

    return X, y, num_cols, cat_cols
