"""
Task 1: Tumor vs. Normal Classification
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
                   filter_to_genes, find_file, load_expression)

MODEL_PATH = os.path.join(MODEL_DIR, 'task1_tumor_vs_normal.pkl')


def build_pipeline():
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('pca',     PCA(n_components=50, random_state=42)),
        ('clf',     LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
    ])


def load_train_data():
    print("Loading training data...")
    lscc_tumor = load_expression(find_file(TRAIN_DIR, 'LSCC', 'protein_expression_tumor'))
    lscc_nat   = load_expression(find_file(TRAIN_DIR, 'LSCC', 'protein_expression_nat'))
    luad_tumor = load_expression(find_file(TRAIN_DIR, 'LUAD', 'protein_expression_tumor'))
    luad_nat   = load_expression(find_file(TRAIN_DIR, 'LUAD', 'protein_expression_nat'))

    lscc_tumor, lscc_nat, luad_tumor, luad_nat = align_genes(
        lscc_tumor, lscc_nat, luad_tumor, luad_nat
    )
    genes = lscc_tumor.columns.tolist()

    X = pd.concat([lscc_tumor, lscc_nat, luad_tumor, luad_nat])
    y = np.array(
        [1] * len(lscc_tumor) + [0] * len(lscc_nat) +
        [1] * len(luad_tumor) + [0] * len(luad_nat)
    )
    return X, y, genes


def train():
    print("=" * 55)
    print("Task 1: Tumor vs. Normal  [train]")
    print("=" * 55)

    X, y, genes = load_train_data()
    print(f"\nSamples: {len(y)}  |  Features: {len(genes)}")
    print(f"Tumor: {y.sum()}  |  Normal: {(y == 0).sum()}")

    pipeline = build_pipeline()

    # 5-Fold CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(
        pipeline, X, y, cv=cv,
        scoring=['accuracy', 'roc_auc', 'f1'],
        return_train_score=True,
    )
    print("\n--- 5-Fold CV Results ---")
    print(f"Train Accuracy : {scores['train_accuracy'].mean():.4f} ± {scores['train_accuracy'].std():.4f}")
    print(f"Test  Accuracy : {scores['test_accuracy'].mean():.4f} ± {scores['test_accuracy'].std():.4f}")
    print(f"Test  AUROC    : {scores['test_roc_auc'].mean():.4f} ± {scores['test_roc_auc'].std():.4f}")
    print(f"Test  F1       : {scores['test_f1'].mean():.4f} ± {scores['test_f1'].std():.4f}")

    # 전체 학습 데이터로 최종 모델 학습 후 저장
    print("\nFitting final model on full training data...")
    pipeline.fit(X, y)
    joblib.dump({'pipeline': pipeline, 'genes': genes}, MODEL_PATH)
    print(f"Model saved → {MODEL_PATH}")


def predict():
    print("=" * 55)
    print("Task 1: Tumor vs. Normal  [predict]")
    print("=" * 55)

    if not os.path.exists(MODEL_PATH):
        print(f"[Error] No trained model found. Run with --mode train first.")
        return

    saved   = joblib.load(MODEL_PATH)
    pipeline = saved['pipeline']
    genes    = saved['genes']

    print("\nLoading test data from Data/test/ ...")
    frames, labels, ids = [], [], []

    for cancer in ['LSCC', 'LUAD']:
        for tissue, label in [('tumor', 1), ('nat', 0)]:
            try:
                path = find_file(TEST_DIR, cancer, f'protein_expression_{tissue}')
                df = load_expression(path)
                df = filter_to_genes(df, genes)
                frames.append(df)
                labels.extend([label] * len(df))
                ids.extend([f"{cancer}_{tissue}_{sid}" for sid in df.index])
                print(f"  Loaded {cancer} {tissue}: {len(df)} samples")
            except FileNotFoundError:
                print(f"  Skipping {cancer} {tissue}: file not found")

    if not frames:
        print("[Error] No test files found in Data/test/")
        return

    X_test = pd.concat(frames)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    results = pd.DataFrame({
        'sample_id':  ids,
        'prediction': ['Tumor' if p == 1 else 'Normal' for p in y_pred],
        'prob_tumor': y_prob.round(4),
    })

    out_path = os.path.join(RESULT_DIR, 'task1_predictions.csv')
    results.to_csv(out_path, index=False)
    print(f"\n--- Predictions ---")
    print(results.to_string(index=False))
    print(f"\nSaved → {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Task 1: Tumor vs Normal')
    parser.add_argument('--mode', choices=['train', 'predict'], default='train')
    args = parser.parse_args()

    if args.mode == 'train':
        train()
    else:
        predict()


if __name__ == '__main__':
    main()
