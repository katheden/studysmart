"""
Generate synthetic student performance datasets for StudySmart.
Run once to create the raw data files.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
n = 650

# ── Student Performance Dataset ──────────────────────────────────────────────
gender = np.random.choice(["M", "F"], n)
age = np.random.randint(15, 22, n)
family_support = np.random.choice(["none", "low", "medium", "high"], n,
                                   p=[0.05, 0.2, 0.45, 0.3])
internet = np.random.choice([0, 1], n, p=[0.25, 0.75])
absences = np.clip(np.random.exponential(6, n).astype(int), 0, 40)
study_time = np.random.choice([1, 2, 3, 4], n, p=[0.2, 0.35, 0.3, 0.15])
failures = np.random.choice([0, 1, 2, 3], n, p=[0.65, 0.2, 0.1, 0.05])
higher_edu = np.random.choice([0, 1], n, p=[0.2, 0.8])
activities = np.random.choice([0, 1], n, p=[0.4, 0.6])
romantic = np.random.choice([0, 1], n, p=[0.6, 0.4])

family_map = {"none": 0, "low": 1, "medium": 2, "high": 3}
family_num = np.array([family_map[f] for f in family_support])

# Realistic grade formula
noise = np.random.normal(0, 1.5, n)
g1 = np.clip(
    10 + study_time * 1.5 - absences * 0.25 - failures * 1.8
    + family_num * 0.5 + internet * 0.4 + higher_edu * 0.6
    + activities * 0.3 - romantic * 0.4 + noise,
    0, 20
).round(1)

noise2 = np.random.normal(0, 1.2, n)
g2 = np.clip(g1 + noise2 * 0.6, 0, 20).round(1)

noise3 = np.random.normal(0, 1.0, n)
g3 = np.clip(g2 + noise3 * 0.5, 0, 20).round(1)

df_perf = pd.DataFrame({
    "gender": gender,
    "age": age,
    "family_support": family_support,
    "internet": internet,
    "absences": absences,
    "study_time": study_time,   # 1=<2h 2=2-5h 3=5-10h 4=>10h
    "failures": failures,
    "higher_edu": higher_edu,
    "activities": activities,
    "romantic": romantic,
    "G1": g1,
    "G2": g2,
    "G3": g3,
})

df_perf.to_csv("/home/claude/studysmart/data/raw/student_performance.csv", index=False)
print("student_performance.csv:", df_perf.shape)

# ── Student Habits Dataset ────────────────────────────────────────────────────
n2 = 800
sleep_hours   = np.clip(np.random.normal(6.8, 1.2, n2), 3, 10).round(1)
social_media  = np.clip(np.random.normal(3.0, 1.5, n2), 0, 8).round(1)
daily_study   = np.clip(np.random.normal(2.5, 1.2, n2), 0, 8).round(1)
motivation    = np.random.choice(["low", "medium", "high"], n2, p=[0.25, 0.45, 0.30])
stress        = np.random.randint(1, 11, n2)
part_time_job = np.random.choice([0, 1], n2, p=[0.65, 0.35])
learning_style= np.random.choice(["visual", "auditory", "reading", "kinesthetic"], n2)
exam_anxiety  = np.random.randint(1, 11, n2)
mental_load   = np.clip(stress * 0.6 + exam_anxiety * 0.4 + np.random.normal(0, 0.5, n2), 1, 10).round(1)

df_habits = pd.DataFrame({
    "sleep_hours": sleep_hours,
    "social_media_hours": social_media,
    "daily_study_hours": daily_study,
    "motivation": motivation,
    "stress_level": stress,
    "part_time_job": part_time_job,
    "learning_style": learning_style,
    "exam_anxiety": exam_anxiety,
    "mental_load": mental_load,
})

df_habits.to_csv("/home/claude/studysmart/data/raw/student_habits.csv", index=False)
print("student_habits.csv:", df_habits.shape)

# ── Study Tips Text ───────────────────────────────────────────────────────────
tips = """# Study Strategies & Learning Tips

## Time Management
- Use time-blocking: assign fixed study slots in your calendar each week.
- Apply the Pomodoro Technique: 25 minutes of focused work, then a 5-minute break.
- Prioritise tasks using the Eisenhower Matrix (urgent vs. important).
- Avoid multitasking; single-task to maintain focus and retention.
- Review your weekly schedule every Sunday to plan the upcoming week.

## Active Recall & Spaced Repetition
- Test yourself after every study session instead of re-reading notes.
- Use flashcards (Anki) with spaced repetition to retain information long-term.
- Write practice questions after each lecture and answer them 24 hours later.
- The Feynman Technique: explain a concept in simple language to uncover gaps.

## Exam Preparation
- Start exam revision at least 3 weeks in advance.
- Solve past exam papers under timed conditions.
- Identify your weakest topics first and allocate more time to them.
- Form study groups to quiz each other and explain difficult concepts.
- Sleep at least 7-8 hours before an exam; memory consolidates during sleep.

## Managing Absences
- If you miss a lecture, obtain notes from classmates within 24 hours.
- Watch recorded lectures or alternative online resources the same week.
- Book office hours with the lecturer to catch up on missed content.
- Create a summary sheet for every missed session to stay on track.

## Motivation & Mental Health
- Set small, achievable weekly goals to build momentum.
- Reward yourself after completing study blocks (e.g., a short walk, a snack).
- Limit social-media use to defined breaks to avoid distraction.
- Talk to a counsellor or trusted person if stress feels overwhelming.
- Physical exercise (even 20 minutes daily) significantly improves focus.

## Sleep & Recovery
- Aim for 7-9 hours of sleep every night; sleep deprivation cuts memory by up to 40%.
- Avoid screens 30 minutes before bed to improve sleep quality.
- Take a 10-20 minute nap if you feel fatigued during the day.
- Maintain a consistent sleep schedule even on weekends.

## Reducing Exam Anxiety
- Practice deep breathing (4-7-8 method) before and during exams.
- Visualise a successful exam scenario the night before.
- Arrive early to the exam venue to settle your nerves.
- Focus on what you know, not what you don't, when reading questions first.
"""

with open("/home/claude/studysmart/data/raw/study_tips.md", "w") as f:
    f.write(tips)

print("study_tips.md written")
print("Done!")
