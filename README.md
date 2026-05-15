# Tumor Survival Prediction

폐암 단백질 및 RNA 발현 데이터를 이용한 다중 분류 바이오인포매틱스 프로젝트입니다.

---

## 프로젝트 개요

| 태스크 | 설명 | 배점 |
|--------|------|------|
| Task 1 | 종양 vs 정상 조직 분류 | 20점 |
| Task 2 | LUAD vs LSCC 암종 분류 | 20점 |
| Task 3 | 생존 예측 | 10점 |

**암종 종류**
- **LUAD** (폐선암): 학습 88명 / 테스트 9명
- **LSCC** (폐편평세포암): 학습 74명 / 테스트 8명

---

## 디렉토리 구조

```
Tumor_Survival_Prediction/
├── Data/
│   ├── train/                          # 학습 데이터 (TSV 파일)
│   └── test/                           # 테스트 파일을 여기에 넣으면 자동 인식
├── models/                             # 학습 후 저장되는 모델 파일
├── results/                            # 예측 결과 CSV 저장 위치
├── Tumor vs Normal Classification/
│   └── tumor_vs_normal.py
├── LUAD vs. LSCC Classification/
│   └── luad_vs_lscc.py
├── Survival Prediction/
│   └── survival_prediction.py
└── utils.py
```

---

## 설치

```bash
pip install scikit-learn pandas numpy joblib
```

---

## 데이터 형식

모든 입력 파일은 탭 구분자(`.tsv`) 형식입니다.

**발현 파일** (`protein_expression_tumor`, `protein_expression_nat`):
- 행: 유전자 ID, 열: 환자 `case_id`
- 스크립트 내부에서 자동으로 전치(samples × genes)

**생존 파일** (`overall_survival`):
- 컬럼: `case_id`, `OS_days`, `OS_event`
- `OS_event`: 1 = 사망, 0 = 생존 또는 중도절단

**테스트 파일 명명 규칙** — 파일 이름에 아래 키워드가 포함되면 자동 인식됩니다.

| 키워드 | 의미 |
|--------|------|
| `LSCC` 또는 `LUAD` | 암종 구분 |
| `protein_expression_tumor` | 종양 단백질 발현 |
| `protein_expression_nat` | 정상 인접 조직 단백질 발현 |
| `overall_survival` | 생존 라벨 |

예시: `LSCC_testset_protein_expression_tumor.tsv` → 자동 인식

---

## 사용법

각 스크립트는 `--mode train`과 `--mode predict` 두 가지 모드를 지원합니다.

### Step 1 — 학습 (최초 1회 실행)

```bash
python "Tumor vs Normal Classification/tumor_vs_normal.py" --mode train
python "LUAD vs. LSCC Classification/luad_vs_lscc.py" --mode train
python "Survival Prediction/survival_prediction.py" --mode train
```

학습 시 동작:
1. `Data/train/` 에서 데이터 로드
2. 5-Fold 교차검증 실행 및 성능 출력
3. 전체 학습 데이터로 최종 모델 학습
4. `models/` 에 모델 저장

### Step 2 — 테스트 파일 추가

`Data/test/` 폴더에 테스트 파일을 넣습니다.

```
Data/test/
├── LSCC_testset_protein_expression_tumor.tsv
├── LSCC_testset_protein_expression_nat.tsv
├── LUAD_testset_protein_expression_tumor.tsv
├── LUAD_testset_protein_expression_nat.tsv
└── ...
```

### Step 3 — 예측

```bash
python "Tumor vs Normal Classification/tumor_vs_normal.py" --mode predict
python "LUAD vs. LSCC Classification/luad_vs_lscc.py" --mode predict
python "Survival Prediction/survival_prediction.py" --mode predict
```

예측 결과는 `results/` 폴더에 CSV 파일로 저장됩니다.

---

## 태스크 상세

### Task 1: 종양 vs 정상 조직 분류

**스크립트:** `Tumor vs Normal Classification/tumor_vs_normal.py`

**학습 입력:** LUAD + LSCC 단백질 발현 (종양 + 정상 인접 조직)

**예측 입력:** `Data/test/` 내 단백질 발현 파일 (종양 + 정상 인접 조직)

**모델:** Logistic Regression + PCA (50 components)

**파이프라인:**
```
결측값 처리(중앙값) → 표준화 → PCA(50) → 로지스틱 회귀
```

**출력 파일:** `results/task1_predictions.csv`

| 컬럼 | 설명 |
|------|------|
| `sample_id` | `{암종}_{조직}_{case_id}` |
| `prediction` | `Tumor` 또는 `Normal` |
| `prob_tumor` | 종양일 확률 |

---

### Task 2: LUAD vs LSCC 암종 분류

**스크립트:** `LUAD vs. LSCC Classification/luad_vs_lscc.py`

**학습 입력:** LUAD + LSCC 종양 단백질 발현

**예측 입력:** `Data/test/` 내 종양 단백질 발현 파일

**모델:** Linear SVM + PCA (50 components)

**파이프라인:**
```
결측값 처리(중앙값) → 표준화 → PCA(50) → SVM(선형 커널)
```

**출력 파일:** `results/task2_predictions.csv`

| 컬럼 | 설명 |
|------|------|
| `sample_id` | `{암종}_{case_id}` |
| `prediction` | `LUAD` 또는 `LSCC` |
| `prob_LUAD` | LUAD일 확률 |

---

### Task 3: 생존 예측

**스크립트:** `Survival Prediction/survival_prediction.py`

**학습 입력:** LUAD + LSCC 종양 단백질 발현 + 생존 라벨

**예측 입력:** `Data/test/` 내 종양 단백질 발현 파일

**모델:** Logistic Regression (class_weight='balanced') + PCA (30 components)

**파이프라인:**
```
결측값 처리(중앙값) → 표준화 → PCA(30) → 로지스틱 회귀(클래스 가중치 적용)
```

> 참고: 학습 데이터 내 사망(OS_event=1) 비율이 약 22%로 불균형합니다. `class_weight='balanced'` 옵션으로 보정합니다.

**출력 파일:** `results/task3_predictions.csv`

| 컬럼 | 설명 |
|------|------|
| `sample_id` | `{암종}_{case_id}` |
| `prediction` | `Death` 또는 `Survival` |
| `prob_death` | 사망 확률 |

---

## 학습 결과 (5-Fold 교차검증)

| 태스크 | Accuracy | AUROC | F1 |
|--------|----------|-------|----|
| Task 1: 종양 vs 정상 | 1.0000 | 1.0000 | 1.0000 |
| Task 2: LUAD vs LSCC | 1.0000 | 1.0000 | 1.0000 |
| Task 3: 생존 예측 | 0.7155 | 0.6430 | 0.5697 (macro) |
