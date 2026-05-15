"""
Task 3: Survival Prediction (Survival / Death)
  --mode train    학습 + 모델 저장
  --mode predict  저장된 모델로 test 데이터 예측
"""
import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (MODEL_DIR, RESULT_DIR, TEST_DIR, TRAIN_DIR, align_genes,
                   filter_to_genes, find_file, load_expression, load_survival)

MODEL_PATH = os.path.join(MODEL_DIR, 'task3_survival.pkl')


def build_pipeline():
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('pca',     PCA(n_components=30, random_state=42)),
        ('clf',     LogisticRegression(
            class_weight='balanced', C=1.0, max_iter=1000, random_state=42
        )),
    ])


def load_train_data():
    print("Loading training data (tumor only) + survival labels...")
    lscc_expr = load_expression(find_file(TRAIN_DIR, 'LSCC', 'protein_expression_tumor'))
    luad_expr = load_expression(find_file(TRAIN_DIR, 'LUAD', 'protein_expression_tumor'))
    lscc_surv = load_survival(find_file(TRAIN_DIR, 'LSCC', 'overall_survival'))
    luad_surv = load_survival(find_file(TRAIN_DIR, 'LUAD', 'overall_survival'))

    lscc_expr, luad_expr = align_genes(lscc_expr, luad_expr)
    genes = lscc_expr.columns.tolist()

    # case_id 기준으로 expression ↔ survival 매칭
    lscc_ids = lscc_expr.index.intersection(lscc_surv.index)
    luad_ids = luad_expr.index.intersection(luad_surv.index)

    X = pd.concat([lscc_expr.loc[lscc_ids], luad_expr.loc[luad_ids]])
    y = np.concatenate([
        lscc_surv.loc[lscc_ids, 'OS_event'].values,
        luad_surv.loc[luad_ids, 'OS_event'].values,
    ])
    return X, y, genes


def train():
    print("=" * 55)
    print("Task 3: Survival Prediction  [train]")
    print("=" * 55)

    X, y, genes = load_train_data()
    print(f"\nSamples: {len(y)}  |  Features: {len(genes)}")
    print(f"Death (1): {y.sum()}  |  Survived/Censored (0): {(y == 0).sum()}")

    pipeline = build_pipeline()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(
        pipeline, X, y, cv=cv,
        scoring=['accuracy', 'roc_auc', 'f1_macro'],
        return_train_score=True,
    )
    print("\n--- 5-Fold CV Results ---")
    print(f"Train Accuracy : {scores['train_accuracy'].mean():.4f} ± {scores['train_accuracy'].std():.4f}")
    print(f"Test  Accuracy : {scores['test_accuracy'].mean():.4f} ± {scores['test_accuracy'].std():.4f}")
    print(f"Test  AUROC    : {scores['test_roc_auc'].mean():.4f} ± {scores['test_roc_auc'].std():.4f}")
    print(f"Test  F1 Macro : {scores['test_f1_macro'].mean():.4f} ± {scores['test_f1_macro'].std():.4f}")

    print("\nFitting final model on full training data...")
    pipeline.fit(X, y)
    joblib.dump({'pipeline': pipeline, 'genes': genes}, MODEL_PATH)
    print(f"Model saved → {MODEL_PATH}")


def predict():
    print("=" * 55)
    print("Task 3: Survival Prediction  [predict]")
    print("=" * 55)

    if not os.path.exists(MODEL_PATH):
        print("[Error] No trained model found. Run with --mode train first.")
        return

    saved    = joblib.load(MODEL_PATH)
    pipeline = saved['pipeline']
    genes    = saved['genes']

    print("\nLoading test data from Data/test/ ...")
    frames, ids = [], []

    for cancer in ['LSCC', 'LUAD']:
        try:
            path = find_file(TEST_DIR, cancer, 'protein_expression_tumor')
            df = load_expression(path)
            df = filter_to_genes(df, genes)
            frames.append(df)
            ids.extend([f"{cancer}_{sid}" for sid in df.index])
            print(f"  Loaded {cancer} tumor: {len(df)} samples")
        except FileNotFoundError:
            print(f"  Skipping {cancer}: file not found")

    if not frames:
        print("[Error] No test files found in Data/test/")
        return

    X_test = pd.concat(frames)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    results = pd.DataFrame({
        'sample_id':   ids,
        'prediction':  ['Death' if p == 1 else 'Survival' for p in y_pred],
        'prob_death':  y_prob.round(4),
    })

    out_path = os.path.join(RESULT_DIR, 'task3_predictions.csv')
    results.to_csv(out_path, index=False)
    print(f"\n--- Predictions ---")
    print(results.to_string(index=False))
    print(f"\nSaved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Task 3: Survival Prediction')
    parser.add_argument('--mode', choices=['train', 'predict'], default='train')
    args = parser.parse_args()

    if args.mode == 'train':
        train()
    else:
        predict()


if __name__ == '__main__':
    main()
