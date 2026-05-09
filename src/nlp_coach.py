"""
Generate study coaching text for the Streamlit app.
The app can use the OpenAI API, but it also has a deterministic fallback.
"""
import os, re

# Study tips corpus
TIPS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "study_tips.md")

def _load_tips() -> str:
    try:
        with open(TIPS_PATH, "r") as f:
            return f.read()
    except FileNotFoundError:
        return ""


# Human-readable feature names
FEATURE_LABELS = {
    "G2":                 "mid-year grade (period 2)",
    "G1":                 "mid-year grade (period 1)",
    "study_efficiency":   "study efficiency (grade per study hour)",
    "study_time":         "weekly study time",
    "absences":           "number of absences",
    "failures":           "past course failures",
    "family_support_num": "level of family support",
    "internet":           "internet access at home",
    "higher_edu":         "aspiration for higher education",
    "activities":         "participation in extra-curricular activities",
    "romantic":           "being in a romantic relationship",
    "gender_bin":         "gender",
    "high_absence":       "high absence flag",
}


def _top_features_text(top_features: list[tuple]) -> str:
    lines = []
    for feat, imp in top_features[:5]:
        label = FEATURE_LABELS.get(feat, feat)
        lines.append(f"- {label} (importance: {imp:.3f})")
    return "\n".join(lines)


def _habit_context_text(habit_context: dict | None) -> str:
    if not habit_context:
        return "No additional context provided."
    parts = []
    if "sleep_hours" in habit_context:
        parts.append(f"average sleep: {habit_context['sleep_hours']} hours per night")
    if "social_media_hours" in habit_context:
        parts.append(f"social media use: {habit_context['social_media_hours']} hours per day")
    if "motivation_level" in habit_context:
        parts.append(f"motivation level: {habit_context['motivation_level']}")
    return "; ".join(parts) if parts else "No additional context provided."


# Prompt-based generation (OpenAI API)
def generate_coaching_api(
    risk_label: str,
    confidence: float,
    top_features: list[tuple],
    student_goal: str = "pass my exams",
    model: str | None = None,
    habit_context: dict | None = None,
) -> str:
    """Call OpenAI API to generate personalised coaching text.

    The OpenAI client reads OPENAI_API_KEY from the environment.
    Set OPENAI_MODEL to override the default model. If the API is unavailable,
    the function falls back to deterministic template-based coaching.
    """
    try:
        from openai import OpenAI
        client = OpenAI()
        model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    except Exception:
        return generate_coaching_template(risk_label, confidence, top_features, student_goal)

    study_tips = _load_tips()
    top_text   = _top_features_text(top_features)
    context_text = _habit_context_text(habit_context)

    system_prompt = """You are a study support assistant.
Explain a student performance prediction in plain language and suggest practical next steps.

Rules:
- Do not judge the student.
- Do not claim that the prediction is certain.
- Keep the explanation short.
- Give exactly three specific recommendations.
- Base the recommendations on the top model factors and the additional study context.
- Format the output as one short explanation paragraph followed by three numbered recommendations.
"""

    user_prompt = f"""Student performance prediction:
- Risk Level: {risk_label}
- Confidence: {confidence:.1%}
- Student goal: "{student_goal}"

Top factors influencing this prediction:
{top_text}

Additional study context:
{context_text}

Reference study strategies (use selectively):
{study_tips[:1500]}

Generate a StudySmart coaching response with:
1. A 3-4 sentence explanation of the prediction in plain language.
2. Three specific, numbered study recommendations tailored to the top factors.
"""

    try:
        response = client.responses.create(
            model=model,
            instructions=system_prompt,
            input=user_prompt,
            max_output_tokens=600,
        )
        return response.output_text.strip()
    except Exception as e:
        return generate_coaching_template(risk_label, confidence, top_features, student_goal, habit_context=habit_context)


# Template-based fallback
ADVICE_BY_RISK = {
    "Low Risk": [
        "Keep up your consistent study routine - it's clearly paying off.",
        "Challenge yourself with past exam papers to maintain and extend your performance.",
        "Use spaced repetition (e.g. Anki) to lock in knowledge for the long term.",
    ],
    "Medium Risk": [
        "Try to add one extra focused study session of 30–45 minutes per week.",
        "Review your two weakest topics this week using active recall instead of re-reading.",
        "Reduce unplanned absences - even one missed session can create gaps that compound over time.",
    ],
    "High Risk": [
        "Book a session with your lecturer or tutor this week to identify your biggest knowledge gaps.",
        "Switch to active study methods: practice questions, mind maps, or teaching a concept aloud.",
        "Create a structured weekly timetable with fixed study blocks and stick to it for 2 weeks.",
    ],
}

FACTOR_ADVICE = {
    "absences":       "Consider attending every session - attendance is one of the strongest predictors of success.",
    "study_time":     "Aim for at least 2–3 hours of focused study per day in the weeks before exams.",
    "failures":       "Past failures are not predictors of future results - targeted revision can change outcomes quickly.",
    "study_efficiency":"Focus on quality over quantity: use active recall and avoid passive re-reading.",
    "G1":             "Your earlier grade suggests there may be foundational gaps - revisit core concepts first.",
    "G2":             "Your recent grade trend is the clearest signal - consistent effort now makes a big difference.",
    "family_support_num": "If you lack support at home, seek peer study groups or college support services.",
}


def generate_coaching_template(
    risk_label: str,
    confidence: float,
    top_features: list[tuple],
    student_goal: str = "pass my exams",
    habit_context: dict | None = None,
) -> str:

    # Explanation paragraph
    top_names = [FEATURE_LABELS.get(f, f) for f, _ in top_features[:3]]
    factor_str = ", ".join(top_names)
    explanation = (
        f"Based on the information provided, the model identifies a **{risk_label}** "
        f"(confidence: {confidence:.0%}). "
        f"The most influential factors in this assessment are: {factor_str}. "
        f"This is an estimate to help guide your preparation - it is not a definitive judgement. "
        f"The stated goal is to {student_goal}."
    )

    context_text = _habit_context_text(habit_context)
    if habit_context:
        explanation += f" Additional context considered: {context_text}."

    # Recommendations: generic + factor-specific
    generic = ADVICE_BY_RISK.get(risk_label, ADVICE_BY_RISK["Medium Risk"])
    specific = []
    for feat, _ in top_features[:2]:
        if feat in FACTOR_ADVICE:
            specific.append(FACTOR_ADVICE[feat])

    recs = (specific + generic)[:3]
    rec_text = "\n".join(f"{i+1}. {r}" for i, r in enumerate(recs))

    return f"""{explanation}

**Recommendations for you:**
{rec_text}
"""


# Main entry point for the app
def get_coaching(
    risk_label: str,
    confidence: float,
    top_features: list[tuple],
    student_goal: str = "pass my exams",
    use_api: bool = True,
    habit_context: dict | None = None,
) -> str:
    if use_api:
        return generate_coaching_api(risk_label, confidence, top_features, student_goal, habit_context=habit_context)
    return generate_coaching_template(risk_label, confidence, top_features, student_goal, habit_context=habit_context)


if __name__ == "__main__":
    # Quick test
    result = get_coaching(
        risk_label="Medium Risk",
        confidence=0.72,
        top_features=[("G2", 0.88), ("absences", 0.05), ("study_time", 0.03)],
        student_goal="improve my grade",
        use_api=False,
    )
    print(result)
