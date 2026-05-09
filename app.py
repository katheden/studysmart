"""
StudySmart Streamlit application.
Run with: streamlit run app.py
"""

from pathlib import Path
import sys
import os

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Page config
st.set_page_config(
    page_title="StudySmart",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #ffffff !important;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #d1d5db !important;
        margin-top: 0;
    }
    .risk-box {
        border-radius: 12px;
        padding: 20px 28px;
        margin: 12px 0;
        font-size: 1.3rem;
        font-weight: 700;
    }
    .low-risk    { background: #d4edda; color: #155724; border-left: 6px solid #28a745; }
    .medium-risk { background: #fef3c7; color: #78350f; border-left: 6px solid #f59e0b; }
    .high-risk   { background: #f8d7da; color: #721c24; border-left: 6px solid #dc3545; }
    .coaching-box {
        background: #111827 !important;
        color: #ffffff !important;
        border-radius: 12px;
        padding: 22px 26px;
        border-left: 6px solid #2563eb;
        border: 1px solid #374151;
        font-size: 1rem;
        line-height: 1.7;
        white-space: pre-wrap;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #ffffff;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }

    .result-card {
        background: #111827 !important;
        border: 1px solid #374151 !important;
        border-radius: 12px !important;
        padding: 18px 18px !important;
        min-height: 112px !important;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.25) !important;
    }
    .result-card-title {
        color: #d1d5db !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        margin-bottom: 12px !important;
    }
    .result-card-value {
        color: #ffffff !important;
        font-size: 2.05rem !important;
        font-weight: 700 !important;
        line-height: 1.15 !important;
        overflow-wrap: break-word !important;
    }

    /* Sidebar model information cards */
    section[data-testid="stSidebar"] .model-card {
        background: #1f2937;
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 14px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.18);
    }
    section[data-testid="stSidebar"] .model-card-title {
        color: #d1d5db;
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    section[data-testid="stSidebar"] .model-card-value {
        color: #ffffff;
        font-size: 2.1rem;
        font-weight: 750;
        line-height: 1.1;
    }
    section[data-testid="stSidebar"] .selected-model-box {
        background: #173a2c;
        border: 1px solid #295943;
        border-radius: 12px;
        padding: 14px 16px;
        margin: 10px 0 16px 0;
        color: #ffffff;
        font-weight: 600;
    }
    section[data-testid="stSidebar"] .selected-model-label {
        color: #d1d5db;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


def sidebar_model_card(title, value):
    st.markdown(
        f"""
        <div class="model-card">
            <div class="model-card-title">{title}</div>
            <div class="model-card-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def result_card(title, value):
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-card-title">{title}</div>
            <div class="result-card-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Sidebar
with st.sidebar:
    st.title("StudySmart")
    st.markdown("Academic risk prediction with study coaching.")
    st.divider()
    st.markdown("#### How it works")
    st.markdown("""
1. Fill in your academic profile
2. The model estimates your performance risk
3. The coaching section explains the result and suggests next steps
    """)
    st.divider()
    st.markdown("#### Model info")
    try:
        import json
        with open(os.path.join("models", "model_meta.json")) as f:
            meta = json.load(f)
        st.markdown(
            f"""<div class="selected-model-box"><span class="selected-model-label">Selected model:</span> {meta['best_model_name']}</div>""",
            unsafe_allow_html=True,
        )
        sidebar_model_card("Accuracy", f"{meta['accuracy']:.1%}")
        sidebar_model_card("F1-Score (weighted)", f"{meta['f1_weighted']:.1%}")
        sidebar_model_card("CV F1-Score", f"{meta['cv_f1']:.1%}")
    except Exception:
        st.info("Train the model first: `python src/train_model.py`")
    st.divider()
    use_api = st.toggle("Use OpenAI for coaching", value=False)


# Main header
st.markdown('<p class="main-title">StudySmart</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Student Performance Prediction & Personalised Study Coaching</p>',
            unsafe_allow_html=True)
st.divider()

# Input form
st.markdown("### Your Academic Profile")
st.caption("Fill in your details below. All fields are kept private and used only for prediction.")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Personal Info**")
    gender = st.selectbox("Gender", ["F", "M"], index=0)
    family_support = st.selectbox(
        "Family support level",
        ["none", "low", "medium", "high"],
        index=2,
        help="How much academic support do you receive at home?"
    )
    internet = st.checkbox("Internet access at home", value=True)
    higher_edu = st.checkbox("Aiming for higher education", value=True)

with col2:
    st.markdown("**Academic History**")
    G1 = st.slider("Grade Period 1 (0–20)", 0.0, 20.0, 11.0, 0.5,
                   help="Your first periodic grade")
    G2 = st.slider("Grade Period 2 (0–20)", 0.0, 20.0, 11.0, 0.5,
                   help="Your second periodic grade (most recent)")
    failures = st.selectbox("Previous course failures", [0, 1, 2, 3], index=0,
                            help="Number of classes you had to repeat")

with col3:
    st.markdown("**Study Habits**")
    study_time = st.select_slider(
        "Weekly study time",
        options=[1, 2, 3, 4],
        value=2,
        format_func=lambda x: {1: "<2 hrs", 2: "2–5 hrs", 3: "5–10 hrs", 4: ">10 hrs"}[x],
    )
    absences = st.slider("School absences (this year)", 0, 40, 5,
                         help="Total number of absences")
    activities = st.checkbox("Participates in extra-curricular activities", value=False)
    romantic = st.checkbox("Currently in a romantic relationship", value=False)

st.markdown("**Additional study context**")
context_col1, context_col2, context_col3 = st.columns(3)
with context_col1:
    sleep_hours = st.slider("Average sleep per night", 4.0, 10.0, 7.0, 0.5)
with context_col2:
    social_media_hours = st.slider("Social media use per day", 0.0, 8.0, 2.0, 0.5)
with context_col3:
    motivation_level = st.selectbox("Current motivation level", ["low", "medium", "high"], index=1)

st.markdown("**Your goal**")
student_goal = st.text_input(
    "What is your main academic goal?",
    value="pass my final exams with a good grade",
    placeholder="e.g. improve my grade, pass the exam, reduce absences",
)

st.divider()

# Predict button
predict_clicked = st.button("Predict My Performance Risk", type="primary", use_container_width=True)

if predict_clicked:
    user_input = {
        "gender": gender,
        "family_support": family_support,
        "internet": int(internet),
        "higher_edu": int(higher_edu),
        "G1": G1,
        "G2": G2,
        "failures": failures,
        "study_time": study_time,
        "absences": absences,
        "activities": int(activities),
        "romantic": int(romantic),
    }

    # Run prediction
    with st.spinner("Analysing your profile..."):
        try:
            from predict import predict_student
            result = predict_student(user_input)
        except Exception as e:
            st.error(
                f"Prediction error: {e}\n\n"
                "Check that the folders `src/` and `models/` were uploaded. "
                "If you run locally, execute `python src/train_model.py` first."
            )
            st.stop()

    risk_label   = result["risk_label"]
    confidence   = result["confidence"]
    probabilities = result["probabilities"]
    top_features = result["top_features"]
    model_name   = result["model_name"]

    # Risk display
    st.markdown("---")
    st.markdown("## Prediction Result")

    risk_class = risk_label.lower().replace(" ", "-")

    st.markdown(
        f'<div class="risk-box {risk_class}">'
        f'{risk_label} &nbsp;-&nbsp; Confidence: {confidence:.1%}'
        f'</div>',
        unsafe_allow_html=True
    )

    # Result cards
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        result_card("Risk Level", risk_label)
    with m2:
        result_card("Confidence", f"{confidence:.1%}")
    with m3:
        result_card("G2 (latest)", f"{G2:.1f} / 20")
    with m4:
        result_card("Absences", str(absences))

    # Probability chart
    col_chart, col_fi = st.columns(2)

    with col_chart:
        st.markdown('<p class="section-header">Risk Probability Distribution</p>',
                    unsafe_allow_html=True)
        labels = list(probabilities.keys())
        values = list(probabilities.values())
        colors = ["#28a745", "#ffc107", "#dc3545"]
        fig_prob = go.Figure(go.Bar(
            x=labels, y=values,
            marker_color=colors,
            text=[f"{v:.1%}" for v in values],
            textposition="outside",
        ))
        fig_prob.update_layout(
            yaxis=dict(title="Probability", range=[0, 1]),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
            height=300,
        )
        st.plotly_chart(fig_prob, use_container_width=True)

    with col_fi:
        st.markdown('<p class="section-header">Top Influencing Factors</p>',
                    unsafe_allow_html=True)

        FEATURE_LABELS_DISPLAY = {
            "G2":                 "Grade Period 2 (latest)",
            "G1":                 "Grade Period 1",
            "study_efficiency":   "Study Efficiency",
            "study_time":         "Weekly Study Time",
            "absences":           "Number of Absences",
            "failures":           "Past Course Failures",
            "family_support_num": "Family Support",
            "internet":           "Internet Access",
            "higher_edu":         "Higher Ed Aspiration",
            "activities":         "Extra-curricular Activities",
            "romantic":           "In a Relationship",
            "gender_bin":         "Gender",
            "high_absence":       "High Absence Flag",
        }

        fi_data = [(FEATURE_LABELS_DISPLAY.get(f, f), imp) for f, imp in top_features[:6]]
        fi_df   = pd.DataFrame(fi_data, columns=["Feature", "Importance"])

        fig_fi = px.bar(
            fi_df.sort_values("Importance"),
            x="Importance", y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Blues",
        )
        fig_fi.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=10, b=10),
            height=300,
            showlegend=False,
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    # Coaching
    st.markdown("---")
    st.markdown("## Study Coach")

    with st.spinner("Preparing study advice..."):
        try:
            from nlp_coach import get_coaching
            coaching_text = get_coaching(
                risk_label=risk_label,
                confidence=confidence,
                top_features=top_features,
                student_goal=student_goal,
                use_api=use_api,
                habit_context={
                    "sleep_hours": sleep_hours,
                    "social_media_hours": social_media_hours,
                    "motivation_level": motivation_level,
                },
            )
        except Exception as e:
            coaching_text = f"Coaching could not be generated: {e}"

    st.markdown(
        f'<div class="coaching-box">{coaching_text}</div>',
        unsafe_allow_html=True
    )

    report_text = f"""StudySmart Report

Prediction
Risk level: {risk_label}
Confidence: {confidence:.1%}
Model: {model_name}

Student context
Goal: {student_goal}
Latest grade period: {G2:.1f} / 20
Absences: {absences}
Weekly study time category: {study_time}
Average sleep per night: {sleep_hours:.1f} hours
Social media use per day: {social_media_hours:.1f} hours
Motivation level: {motivation_level}

Top influencing factors
"""
    for feature, importance in top_features[:5]:
        report_text += f"- {FEATURE_LABELS_DISPLAY.get(feature, feature)}: {importance:.3f}\n"

    report_text += f"""
Study coach recommendation
{coaching_text}

Note
This tool is for educational support only. It is not a final academic assessment.
"""

    st.download_button(
        label="Download study report",
        data=report_text,
        file_name="studysmart_report.txt",
        mime="text/plain",
    )

    # Weekly study plan
    st.markdown("---")
    st.markdown("## Suggested Weekly Study Plan")

    plan_hours = {
        "Low Risk":    {"Mon": 1.5, "Tue": 2.0, "Wed": 1.5, "Thu": 2.0, "Fri": 1.0, "Sat": 2.0, "Sun": 0.5},
        "Medium Risk": {"Mon": 2.0, "Tue": 2.5, "Wed": 2.0, "Thu": 2.5, "Fri": 2.0, "Sat": 3.0, "Sun": 1.0},
        "High Risk":   {"Mon": 3.0, "Tue": 3.0, "Wed": 3.0, "Thu": 3.0, "Fri": 2.5, "Sat": 3.5, "Sun": 2.0},
    }
    plan = plan_hours[risk_label]
    plan_df = pd.DataFrame({"Day": list(plan.keys()), "Study Hours": list(plan.values())})

    fig_plan = px.bar(
        plan_df, x="Day", y="Study Hours",
        color="Study Hours",
        color_continuous_scale=["#a8d8ea", "#4361ee"],
        text="Study Hours",
    )
    fig_plan.update_traces(texttemplate="%{text}h", textposition="outside")
    fig_plan.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        height=300,
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_plan, use_container_width=True)
    total_h = sum(plan.values())
    st.caption(f"Total recommended study time: **{total_h:.1f} hours/week** for {risk_label} students.")

    # Ethics note
    st.info(
        "**Important:** This prediction is a statistical estimate to support your learning - "
        "it is not a definitive assessment of your abilities. Many factors influence academic "
        "success, and this tool is meant to empower, not to judge."
    )

    # Model info footer
    st.caption(f"Prediction made by **{model_name}** - trained on student performance data.")

else:
    
    st.markdown("### Welcome to StudySmart")
    st.markdown("""
Fill in your academic profile above and click **Predict My Performance Risk** to receive:

- **Risk assessment** - Low / Medium / High performance risk  
- **Visual breakdown** - Probability distribution & key factors  
- **Study coaching** - recommendations based on the prediction and your goal  
- **Weekly study plan** - Recommended hours tailored to your risk level  
    """)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("#### Low Risk")
        st.markdown("Your profile suggests strong performance. Focus on maintaining consistency and preparing efficiently.")
    with col_b:
        st.markdown("#### Medium Risk")
        st.markdown("There's room to improve. A few targeted changes to your study habits can make a big difference.")
    with col_c:
        st.markdown("#### High Risk")
        st.markdown("Early action matters most. Personalised coaching will help you identify and tackle the key barriers.")
