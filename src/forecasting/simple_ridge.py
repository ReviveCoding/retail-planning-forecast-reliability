from __future__ import annotations

import numpy as np
import pandas as pd


def design_matrices(
    train: pd.DataFrame,
    predict_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Build leakage-safe design matrices for ridge regression.

    Numeric columns are used as numeric features.
    Categorical/string columns are one-hot encoded using categories observed in train.
    Prediction-time unseen categories are ignored by reindexing to train columns.
    """
    train_features = train[feature_cols].copy()
    pred_features = predict_df[feature_cols].copy()

    # Split numeric and categorical columns
    numeric_cols = [
        c for c in feature_cols
        if pd.api.types.is_numeric_dtype(train_features[c])
    ]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    if numeric_cols:
        train_num = train_features[numeric_cols].apply(pd.to_numeric, errors="coerce")
        pred_num = pred_features[numeric_cols].apply(pd.to_numeric, errors="coerce")

        medians = train_num.median(numeric_only=True).fillna(0.0)
        train_num = train_num.fillna(medians).fillna(0.0)
        pred_num = pred_num.fillna(medians).fillna(0.0)
    else:
        train_num = pd.DataFrame(index=train_features.index)
        pred_num = pd.DataFrame(index=pred_features.index)

    if categorical_cols:
        train_cat = train_features[categorical_cols].astype("string").fillna("__MISSING__")
        pred_cat = pred_features[categorical_cols].astype("string").fillna("__MISSING__")

        train_cat_encoded = pd.get_dummies(train_cat, columns=categorical_cols, dtype=float)
        pred_cat_encoded = pd.get_dummies(pred_cat, columns=categorical_cols, dtype=float)

        # Use train columns only. Unseen prediction categories become all-zero.
        pred_cat_encoded = pred_cat_encoded.reindex(
            columns=train_cat_encoded.columns,
            fill_value=0.0,
        )
    else:
        train_cat_encoded = pd.DataFrame(index=train_features.index)
        pred_cat_encoded = pd.DataFrame(index=pred_features.index)

    X_train_df = pd.concat(
        [
            train_num.reset_index(drop=True),
            train_cat_encoded.reset_index(drop=True),
        ],
        axis=1,
    )
    X_pred_df = pd.concat(
        [
            pred_num.reset_index(drop=True),
            pred_cat_encoded.reset_index(drop=True),
        ],
        axis=1,
    )

    feature_names = list(X_train_df.columns)

    X_train = X_train_df.to_numpy(dtype=float)
    X_pred = X_pred_df.to_numpy(dtype=float)

    return X_train, X_pred, feature_names


def ridge_fit_predict(
    train: pd.DataFrame,
    predict_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    alpha: float = 1.0,
) -> np.ndarray:
    X_train, X_pred, _ = design_matrices(train, predict_df, feature_cols)

    y = pd.to_numeric(train[target_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    # Add intercept
    X_train_aug = np.column_stack([np.ones(len(X_train)), X_train])
    X_pred_aug = np.column_stack([np.ones(len(X_pred)), X_pred])

    penalty = np.eye(X_train_aug.shape[1]) * alpha
    penalty[0, 0] = 0.0  # do not penalize intercept

    beta = np.linalg.pinv(X_train_aug.T @ X_train_aug + penalty) @ X_train_aug.T @ y
    pred = X_pred_aug @ beta

    return np.maximum(pred, 0.0)
