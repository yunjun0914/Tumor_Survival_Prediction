# Tumor Survival Prediction

## 설치

```bash
pip install scikit-learn pandas numpy joblib
```

---

## 예측 실행 방법

### 1. 테스트 파일 준비

`Data/test/` 폴더에 테스트 파일을 넣습니다.

파일 이름에 아래 키워드가 포함되어 있으면 자동으로 인식됩니다.

| 키워드 | 설명 |
|--------|------|
| `LSCC` / `LUAD` | 암종 구분 |
| `protein_expression_tumor` | 종양 단백질 발현 |
| `protein_expression_nat` | 정상 인접 조직 단백질 발현 |
| `overall_survival` | 생존 라벨 |

### 2. 예측 실행

```bash
# Task 1: 종양 vs 정상 분류
python "Tumor vs Normal Classification/tumor_vs_normal.py" --mode predict

# Task 2: LUAD vs LSCC 분류
python "LUAD vs. LSCC Classification/luad_vs_lscc.py" --mode predict

# Task 3: 생존 예측
python "Survival Prediction/survival_prediction.py" --mode predict
```

### 3. 결과 확인

예측 결과는 `results/` 폴더에 저장됩니다.

| 파일 | 내용 |
|------|------|
| `task1_predictions.csv` | `sample_id`, `prediction` (Tumor/Normal), `prob_tumor` |
| `task2_predictions.csv` | `sample_id`, `prediction` (LUAD/LSCC), `prob_LUAD` |
| `task3_predictions.csv` | `sample_id`, `prediction` (Death/Survival), `prob_death` |

---

## 모델 재학습 (선택)

학습 데이터는 `Data/train/`에 있습니다. 모델을 직접 재학습하려면:

```bash
python "Tumor vs Normal Classification/tumor_vs_normal.py" --mode train
python "LUAD vs. LSCC Classification/luad_vs_lscc.py" --mode train
python "Survival Prediction/survival_prediction.py" --mode train
```
