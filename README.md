---
title: StudySmart
sdk: docker
app_port: 7860
---

# StudySmart

Student Performance Prediction and Personalised Study Coaching

StudySmart is a Streamlit application that estimates academic performance risk from structured student data and gives short study recommendations based on the model output.

## Quick Start

```bash
git clone https://github.com/<your-github-username>/studysmart.git
cd studysmart
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate
pip install -r requirements.txt

# Optional: only needed for OpenAI-based coaching
export OPENAI_API_KEY=your_key_here

python data/raw/download_datasets.py
python src/train_model.py
streamlit run app.py
```

## Deployment on Hugging Face Spaces

Create a new Hugging Face Space and choose **Docker** as the SDK. The included `Dockerfile` starts the Streamlit app on port `7860`, which is the port expected by Hugging Face Spaces.

Required files for deployment:

```text
app.py
Dockerfile
requirements.txt
models/
src/
data/
```

The app also works without an OpenAI key because it includes a template-based fallback for the coaching text. To enable OpenAI-based coaching in the deployed Space, add this secret in the Space settings:

```text
OPENAI_API_KEY
```

Do not commit API keys to GitHub or Hugging Face files.

## Selected AI Blocks

| Block | Technology | Purpose |
|-------|------------|---------|
| ML Numeric Data | Random Forest with scikit-learn | Predict Low, Medium, or High performance risk |
| NLP | OpenAI API with template fallback | Explain the prediction and provide study advice |

## Model Performance

| Model | Accuracy | F1 weighted | CV F1 |
|-------|----------|-------------|-------|
| Logistic Regression | 76.2% | 76.2% | 82.1% |
| Random Forest | 80.0% | 79.9% | 84.3% |
| Gradient Boosting | 78.5% | 78.3% | 83.7% |

Final model: Random Forest. It achieved the best cross-validated F1 score after removing target leakage from the engineered features.

## Project Structure

```text
studysmart/
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
├── documentation.md        # Project documentation
├── data/raw/               # Raw datasets and study tips
├── data/processed/         # Train/test splits
├── notebooks/              # EDA notebook
├── src/                    # Preprocessing, training, inference and NLP code
├── models/                 # Saved model artefacts
└── screenshots/            # EDA and app screenshots
```

## Notes

The OpenAI key must not be committed to GitHub. For Streamlit Cloud or Hugging Face Spaces, add it through the platform's secret settings.

Add GitHub users `jasminh` and `bkuehnis` as collaborators before submitting.
