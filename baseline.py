"""
baseline.py
~~~~~~~~~~~
train_predict(X, y, model="mnl"|"ffnn")  →  (prob_pred, y_pred)

自动推断特征类型：
  - 数值列  →  SimpleImputer(median) + StandardScaler
  - 类别列  →  SimpleImputer("missing") + OneHotEncoder
"""

from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from cagpt_mvp.preprocess import load_meta, apply_metadata
from cagpt_mvp.mnl_estimator import fit_predict as mnl_fit_predict

def _build_preprocessor(num_cols, cat_cols):
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer

    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    cat_pipe = Pipeline(
        steps=[
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )
    return ColumnTransformer(
        [("num", num_pipe, num_cols),
         ("cat", cat_pipe, cat_cols)],
        remainder="drop",
        verbose_feature_names_out=False
    )


def train_predict(raw_df, meta_name, model="ffnn", verbose=0):
    meta = load_meta(meta_name)
    X, y, num_cols, cat_cols = apply_metadata(raw_df, meta)
    preprocess = _build_preprocessor(num_cols, cat_cols)

    if model == "mnl":
        prob_pred, y_pred = mnl_fit_predict(raw_df, meta_name, verbose=verbose)
        return prob_pred, y_pred
    elif model == "ffnn":
        clf = MLPClassifier(
            hidden_layer_sizes=(128,),
            max_iter=300,
            random_state=42,
            verbose=bool(verbose),  # MLP 用布尔
        )
    else:
        raise ValueError("model must be 'mnl' or 'ffnn'")

    pipe = Pipeline(steps=[("prep", preprocess), ("clf", clf)])
    pipe.fit(X, y)

    prob_pred = pipe.predict_proba(X)
    y_pred = pipe.predict(X)
    return prob_pred, y_pred
