# StudySmart — Project Documentation

## Project Metadata

| Field | Value |
|-------|-------|
| **Project title** | StudySmart: Student Performance Prediction and Personalised Study Coaching |
| **Student** | Katheesrupan |
| **GitHub repository URL** | https://github.com/katheden/studysmart |
| **Deployment URL** | https://huggingface.co/spaces/DKatheesrupan/studysmart |
| **Submission date** | 07 June 2026 |

---

## Mandatory Setup Checks

- [x] At least 2 AI blocks selected
- [x] Multiple and different data sources used
- [x] Deployment URL provided
- [x] Required GitHub users added to repository: `jasminh`, `bkuehnis`

---

## Selected AI Blocks

- [x] **ML Numeric Data** (Primary Block 1)
- [x] **NLP** (Primary Block 2)
- [ ] Computer Vision

---

## Project Overview

StudySmart is a Streamlit application that predicts student performance risk from structured academic and behavioural data. The prediction is then used by an NLP module to create a short explanation and practical study recommendations.

**User flow:**
1. Student fills in their academic profile (grades, study time, absences, etc.)
2. A trained Random Forest model classifies risk as Low / Medium / High
3. An NLP coaching module explains the prediction and generates three study recommendations
4. The app displays the predicted risk, probabilities, main influencing factors, and a suggested weekly study schedule

---

## 1. Project Foundation (Short)

### 1.1 Problem Definition

**Problem statement:** Students often receive performance feedback too late to adjust their study behaviour. StudySmart estimates academic performance risk early from structured academic data and turns the result into understandable coaching advice.

**Goal:** Predict whether a student is in Low, Medium, or High performance risk and provide a personalised, non-judgmental study recommendation.

**Success criteria:**
- At least two ML models are trained and compared quantitatively.
- The final model performs better than the baseline model on the held-out test set.
- The NLP output correctly mentions the predicted risk level and important contributing factors.
- The Streamlit deployment allows a user to enter data and receive a prediction plus study coaching.

### 1.2 Integration Logic

**How the selected blocks interact:** The ML block predicts the risk class and exports confidence plus top feature importances. The NLP block uses these ML outputs, together with the student's goal and the study-tips text source, to generate the final explanation and study advice.

**Data and output flow between blocks:** Student input → preprocessing → ML risk prediction → feature importance ranking → OpenAI/template NLP prompt → personalised study coaching output.

## 2. Block Documentation

## 2A. ML Numeric Data Block

### 2A.1 Data Sources

| # | Source | Type | Size | Role |
|---|--------|------|------|------|
| 1 | `student_performance.csv` | Structured CSV | 649 rows × 13 columns | Primary ML training data |
| 2 | `student_habits.csv` | Structured CSV | 1000 rows × 9 columns | EDA and behavioural context fields for coaching |
| 3 | User input (Streamlit form) | Structured form input | 1 student per prediction | Inference / real-time prediction |

**Dataset 1 — Student Performance** features: gender, age, family support, internet access, absences, study time (1–4 scale), past failures, higher education aspiration, extra-curricular activities, romantic relationship status, and grades G1, G2, G3 (0–20 scale).

**Dataset 2 — Student Habits** features: sleep hours, social media hours, daily study hours, motivation level, stress level, part-time job, learning style, exam anxiety, and mental load. It is used for EDA and to define additional behavioural fields in the app, such as sleep, social media use, and motivation. These fields personalise the coaching text but are not used by the final ML classifier.

### 2A.2 Preprocessing and Features

**Steps performed (`src/preprocessing.py`):**

1. **Missing value check** — no missing values present; median imputation applied defensively.
2. **Categorical encoding:**
   - `family_support` → ordinal integer (`none=0, low=1, medium=2, high=3`)
   - `gender` → binary integer (`F=1, M=0`)
3. **Feature engineering:**
   - `study_efficiency = average(G1, G2) / (study_time + 0.5)` — prior-grade performance per unit of study effort, using only information available before the final grade to avoid data leakage
   - `high_absence = 1 if absences > 10 else 0` — binary risk flag
4. **Target variable creation:**
   - G3 ≥ 14 → **Low Risk** (0)
   - G3 ≥ 10 → **Medium Risk** (1)
   - G3 < 10 → **High Risk** (2)
5. **Scaling** — `StandardScaler` applied to all 13 numeric features.
6. **Train/test split** — 80/20, stratified by risk class, `random_state=42`.

**Final feature set (13 features):**
`study_time, absences, failures, G1, G2, internet, higher_edu, activities, romantic, family_support_num, gender_bin, high_absence, study_efficiency`

### 2A.3 Model Selection

Three models were selected to provide a baseline-to-advanced comparison:

| Model | Rationale |
|-------|-----------|
| **Logistic Regression** | Simple, interpretable baseline; assumes linear decision boundaries |
| **Random Forest** | Handles non-linear relationships; robust to outliers; provides feature importance |
| **Gradient Boosting** | Sequential ensemble that corrects errors; typically outperforms Random Forest on tabular data |

### 2A.4 Model Comparison

| Iteration | Objective | Key Changes | Model | Metric | Result |
|-----------|-----------|-------------|-------|--------|--------|
| 1 | Establish baseline | Basic preprocessing, no feature engineering | Logistic Regression | Accuracy / F1 | First reference point |
| 2 | Improve performance | Added `study_efficiency`, `high_absence` | Random Forest | F1 (weighted) | Improved over baseline |
| 3 | Compare boosted ensemble | Tuned `n_estimators=200`, `learning_rate=0.05` | Gradient Boosting | CV F1 | Slightly below Random Forest after leakage fix |

### 2A.5 Evaluation

**Results on held-out test set (130 students) after removing target leakage:**

| Model | Accuracy | F1 (weighted) | CV F1 (5-fold) |
|-------|----------|---------------|----------------|
| Logistic Regression | 76.2% | 76.2% | 82.1% |
| **Random Forest** | **80.0%** | **79.9%** | **84.3%** |
| Gradient Boosting | 78.5% | 78.3% | 83.7% |

**Confusion matrix summary (Random Forest):**
- Low Risk: 28/39 correct (72% recall)
- Medium Risk: 61/71 correct (86% recall)
- High Risk: 15/20 correct (75% recall)

**Error analysis:**
- The model occasionally confuses Medium and Low Risk at the boundary (G3 ≈ 14), which is expected given the hard threshold.
- `G2` is the strongest feature (~39% importance), reflecting the strong temporal correlation between recent and final grades. This is realistic but means earlier effort is harder to detect.
- Self-reported data (study time, absences) may introduce measurement error.
- Dataset size (649 samples) limits generalisation; cross-validation was used to mitigate this.

### 2A.6 Integration with NLP

**Output passed to NLP module:**
- `risk_label` — predicted class string (e.g. "High Risk")
- `confidence` — predicted class probability (e.g. 0.91)
- `top_features` — ranked list of `[(feature_name, importance_score), …]`

These outputs form the input prompt context for the NLP coaching module, ensuring a tight data-driven connection between ML prediction and natural language explanation.

---

## 2B. NLP Block

### 2B.1 Data Sources

| # | Source | Type | Size | Role |
|---|--------|------|------|------|
| 1 | `study_tips.md` | Text / Markdown | ~60 paragraphs | Study strategy knowledge base |
| 2 | ML model output | Structured text | 1 prediction per inference | Explanation input |
| 3 | User goal (text field) | User text input | Short free-text | Personalise coaching tone |

### 2B.2 Preprocessing and Prompt Design

**Study tips text preprocessing:**
- Tips written in structured Markdown with topic headers (Time Management, Active Recall, Exam Preparation, etc.)
- First 1500 characters injected into system prompt as reference context
- No tokenisation needed; passed as raw text to the API

**Prompt design (final version):**

*System prompt:*
> "You are StudySmart Coach — a study support assistant. Explain the ML prediction in simple language. Give exactly 3 specific, actionable recommendations based on the top contributing factors. Never judge the student. Never claim certainty about outcomes."

*User prompt structure:*
```text
Risk Level: {risk_label}
Confidence: {confidence}
Student goal: "{student_goal}"
Top factors:
  - {factor_1} (importance: {score})
  - {factor_2} (importance: {score})
  ...
Reference study strategies: [first 1500 chars of study_tips.md]
```

### 2B.3 Approach Selection

**Chosen approach: Prompt Engineering with OpenAI API (+ template fallback)**

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **Prompt engineering (LLM)** | Flexible, personalised, human-readable | Requires API key | Primary |
| Rule-based templates | Fast, no API needed | Rigid, less personalised | Fallback |
| RAG with study-tips corpus | Grounded in source material | More complex infrastructure | Out of scope |
| Text classification | Fast inference | Cannot generate free text | Wrong task |

Prompt engineering was chosen because it produces natural, personalised explanations that adapt to the specific combination of risk level, top factors, and student goal. The template fallback ensures the app works without an API key.

### 2B.4 Comparison and Iterations

| Iteration | Objective | Key Changes | Output quality | Change |
|-----------|-----------|-------------|----------------|--------|
| 1 | Basic explanation | Zero-shot prompt, risk label only | Readable but generic | Baseline |
| 2 | More specific advice | Included top 3 factors and their scores | Advice references actual factors | Improved relevance |
| 3 | Safer, warmer tone | Added "no judgment", "no certainty" constraints; student goal injected | Less prescriptive, more encouraging | Final version |

### 2B.5 Evaluation

**Evaluation strategy:** Manual qualitative review of 12 sample outputs (4 per risk class).

| Criterion | Description | Rating (1–5) |
|-----------|-------------|--------------|
| Correctness | Correctly identifies and explains risk level | 4 / 5 |
| Personalisation | References actual top factors | 4 / 5 |
| Helpfulness | Gives concrete, actionable tips | 4 / 5 |
| Safety | No harsh judgements; uses hedging language | 5 / 5 |
| Clarity | Plain language, no jargon | 4 / 5 |

**Known error patterns:**
- If G2 is the dominant factor (common, ~39% feature importance), recommendations may focus excessively on grades rather than study habits.
- Occasionally produces slightly generic advice when multiple factors have similar importance scores.
- The weekly study plan is rule-based and not directly predicted by the ML model; it is included as a simple support feature.
- If the ML model misclassifies, the NLP explanation inherits that error — the system is only as good as its prediction.

### 2B.6 Integration

**Input from ML:**
- `risk_label` (str), `confidence` (float), `top_features` (list of tuples)

**Output to user/app:**
- 3–4 sentence explanation paragraph
- Three numbered, actionable study recommendations
- Displayed in the Streamlit coaching card

---

## 2C. Computer Vision

N/A — Computer Vision was not selected as a primary block.

---

## 3. Deployment

### Deployment URL

```text
https://huggingface.co/spaces/DKatheesrupan/studysmart
```

The application is deployed on Hugging Face Spaces using Docker. Docker is used because the user interface is implemented with Streamlit and the Docker setup reproduces the local environment reliably.

### Main User Flow

1. User opens the Hugging Face Space.
2. User fills in their academic profile (grades, study habits, personal factors).
3. User enters their study goal in free text.
4. App preprocesses input through the same pipeline as training.
5. ML model predicts performance risk class and probabilities.
6. NLP module (OpenAI API or template fallback) generates personalised coaching.
7. User sees: risk level, probability chart, feature importance, coaching text, weekly study plan.

---

## 4. Execution Instructions

### Prerequisites

- Python 3.10+
- OpenAI API key (optional — app works without it using template fallback)

### Setup

```bash
git clone https://github.com/kathden/studysmart.git
cd studysmart
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Set API key (optional)

```bash
export OPENAI_API_KEY=your_api_key_here
```

For Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

### Prepare data and train model

```bash
python data/raw/download_datasets.py  # Standardise the raw datasets
python src/train_model.py             # Train all models, save best + scaler
```

If the real raw datasets are not available, `data/raw/generate_data.py` can be used as a synthetic fallback for local testing.

### Run the app

```bash
streamlit run app.py
```

### Run the EDA notebook

```bash
jupyter notebook notebooks/analysis.ipynb
```

---

## Project Structure

```text
studysmart/
│
├── app.py                    # Main Streamlit application
├── Dockerfile                # Hugging Face Docker deployment
├── requirements.txt          # Python dependencies
├── README.md                 # Quick-start guide
├── documentation.md          # This file
│
├── data/
│   ├── raw/
│   │   ├── generate_data.py          # Script to create synthetic datasets
│   │   ├── student_performance.csv   # Primary ML dataset (649 rows)
│   │   ├── student_habits.csv        # Supplementary habits dataset (1000 rows)
│   │   └── study_tips.md             # NLP knowledge base
│   └── processed/
│       ├── X_train.csv, X_test.csv   # Scaled feature matrices
│       └── y_train.csv, y_test.csv   # Target labels
│
├── notebooks/
│   └── analysis.ipynb        # Full EDA notebook
│
├── src/
│   ├── preprocessing.py      # Data loading, encoding, scaling, feature engineering
│   ├── train_model.py        # Model training, evaluation, saving
│   ├── predict.py            # Inference pipeline for single student
│   └── nlp_coach.py          # NLP coaching: OpenAI API with template fallback
│
├── models/
│   ├── best_model.pkl        # Serialised Random Forest model
│   ├── scaler.pkl            # Fitted StandardScaler
│   └── model_meta.json       # Model name, metrics, feature importances
│
└── screenshots/
    ├── eda_target.png
    ├── eda_correlation.png
    ├── eda_key_questions.png
    ├── eda_risk_analysis.png
    ├── eda_habits.png
    ├── app_input.png
    └── app_result.png
```

---

## 5. Optional Bonus Evidence

## Ethics & Fairness Analysis

This section addresses responsible AI considerations, eligible for bonus points.

### Bias and Fairness

- **Socioeconomic proxy features:** `internet`, `family_support`, and `activities` may encode socioeconomic status. Students with fewer resources are systematically disadvantaged through no personal fault.
- **Gender:** Included as a feature; should be monitored for disparate impact. If the model performs significantly differently across gender groups, the feature should be removed.
- **Self-reported data bias:** Study time and absences are self-reported and may be inaccurate. Students who underreport absences may receive misleadingly low risk scores.
- **Historical inequity:** Training on past data embeds existing educational inequalities. A student from a well-resourced background may receive a lower risk score for the same academic performance.

### Mitigation Measures

- The model is framed as a **support tool**, not a gatekeeping mechanism.
- All predictions include confidence scores and uncertainty language ("this is an estimate").
- The app explicitly states: *"This prediction is not a definitive assessment of your abilities."*
- Feature importance is shown so students can understand and contest the basis of their prediction.

### Privacy

- No personally identifiable information is stored.
- All user inputs are processed in-memory and discarded after the session.
- In a production deployment, inputs should never be logged or used for retraining without explicit consent.

### Recommendations

- Conduct regular fairness audits comparing prediction accuracy across demographic subgroups.
- Allow students to flag inaccurate predictions.
- Never use this tool for consequential decisions (e.g. scholarship eligibility, academic probation).

---

## Extended Evaluation (Bonus)

### Random Forest Feature Importance

Feature importances from the selected Random Forest model:

| Feature | Importance |
|---------|-----------|
| G2 (Grade Period 2) | 0.390 |
| G1 (Grade Period 1) | 0.258 |
| Study Efficiency | 0.115 |
| Past Failures | 0.051 |
| Absences | 0.040 |
| Study Time | 0.035 |
| Higher Education Aspiration | 0.029 |
| Family Support | 0.024 |

**Insight:** G2 is the strongest feature because it is the most recent signal before the final grade. This is realistic (teachers also weight recent performance most), but means the model has less power to help students who are struggling from the very beginning (when G2 is low and time to improve is limited).

### Cross-Validation

5-fold stratified cross-validation was used for model selection to prevent overfitting to the train/test split. The selected model reached a CV F1-score of 84.3% and a held-out test F1-score of 79.9%. The difference is acceptable for a small educational dataset and is more realistic after removing target leakage.

### Qualitative NLP Evaluation (10 samples)

| Sample | Risk | Top factor mentioned? | Advice specific? | Tone appropriate? |
|--------|------|-----------------------|-----------------|-------------------|
| 1 | High | correct | useful | clear |
| 2 | Medium | correct | useful | clear |
| 3 | Low | correct | useful | clear |
| 4 | High | correct | slightly generic | clear |
| 5 | Medium | correct | useful | clear |
| 6 | High | correct | useful | clear |
| 7 | Low | correct | useful | clear |
| 8 | Medium | too focused on G2 | useful | clear |
| 9 | High | correct | useful | clear |
| 10 | Medium | correct | useful | clear |

**Result:** 9/10 samples correctly personalised to top factors; 10/10 used appropriate encouraging tone; 9/10 gave specific actionable advice.
