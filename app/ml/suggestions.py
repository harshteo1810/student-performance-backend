"""
Personalized suggestion engine (Objective 4).

Design approach: hybrid of the model's feature_importances_ (which tells us
WHICH features matter most for risk prediction in general) with per-student
rule-based thresholds (which tell us WHETHER this specific student is weak
on that feature). This is deliberately not another ML model -- suggestion
generation from a black-box classifier is hard to justify/explain in a
report, whereas "the model says GPA matters most, and your GPA is below
the healthy threshold, so improve GPA" is fully explainable and defensible
in a viva.

Only features a student can actually act on are included (age, gender,
years_in_program are excluded -- you can't "improve" your age).
"""

import os
import joblib

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
_model = joblib.load(os.path.join(ARTIFACTS_DIR, "model.pkl"))
_feature_columns = joblib.load(os.path.join(ARTIFACTS_DIR, "feature_columns.pkl"))

_IMPORTANCE = dict(zip(_feature_columns, _model.feature_importances_))

# Actionable features only, with a "healthy" threshold and advice template.
# Thresholds are simple domain heuristics (documented here, not hidden),
# not derived from the data itself -- keep them adjustable to your
# institution's actual benchmarks if they differ.
_RULES = {
    "cumulative_gpa": {
        "check": lambda v: v < 6.5,
        "message": "Your cumulative GPA ({v:.2f}) is below a healthy threshold (6.5). "
                    "Prioritize revising fundamentals in your weakest subjects.",
    },
    "backlogs_count": {
        "check": lambda v: v > 0,
        "message": "You have {v} pending backlog(s). Clearing these should be a "
                    "higher priority than new coursework.",
    },
    "attendance_percentage": {
        "check": lambda v: v < 75,
        "message": "Attendance is {v:.1f}%, below the typical 75% minimum. "
                    "Low attendance strongly correlates with lower internal marks in this model.",
    },
    "consecutive_absences_max": {
        "check": lambda v: v > 4,
        "message": "You've had a streak of {v} consecutive absences. Long gaps make "
                    "it harder to catch up -- consider attending consistently rather than in bursts.",
    },
    "daily_study_hours": {
        "check": lambda v: v < 2,
        "message": "Daily study time ({v:.1f} hrs) is on the lower side. Even a modest "
                    "increase (30-45 min/day) shows up in the model as a meaningful risk reduction.",
    },
    "class_participation_score": {
        "check": lambda v: v < 5,
        "message": "Class participation score ({v:.1f}/10) is low. Actively asking "
                    "questions/answering in class is linked with better internal marks here.",
    },
    "peer_collaboration_score": {
        "check": lambda v: v < 5,
        "message": "Peer collaboration score ({v:.1f}/10) is low. Consider joining or "
                    "forming a study group for upcoming assessments.",
    },
    "ct1_marks_25": {
        "check": lambda v: v < 15,
        "message": "CT1 score ({v:.1f}/25) suggests gaps from early in the term -- "
                    "revisit that material before it compounds into the mid-sem.",
    },
    "ct2_marks_25": {
        "check": lambda v: v < 15,
        "message": "CT2 score ({v:.1f}/25) is below a comfortable margin -- targeted "
                    "revision here has a direct, near-term payoff on the mid-sem.",
    },
    "assignment_marks_15": {
        "check": lambda v: v < 9,
        "message": "Assignment marks ({v:.1f}/15) are low -- these are typically the "
                    "easiest marks to recover through consistent, on-time submission.",
    },
    "mid_sem_marks_35": {
        "check": lambda v: v < 21,
        "message": "Mid-sem score ({v:.1f}/35) is below 60% -- this is the single most "
                    "influential factor in this model's prediction, worth prioritizing.",
    },
}


def generate_suggestions(features: dict, risk_category: str, top_n: int = 4) -> list[str]:
    """
    features: dict of the student's submitted values (same keys as StudentFeaturesIn).
    risk_category: the model's predicted class, used only for the good-performance message.
    Returns: list of human-readable suggestion strings, ranked by the model's
             feature_importances_ among the features this student is actually weak on.
    """
    if risk_category == "Good-Performance":
        return [
            "Current trajectory is strong across attendance, study habits, and "
            "assessment marks -- keep the routine consistent through the final exam."
        ]

    triggered = []
    for feature, rule in _RULES.items():
        value = features.get(feature)
        if value is None:
            continue
        if rule["check"](value):
            triggered.append((feature, rule["message"].format(v=value)))

    # Rank triggered suggestions by how much this feature matters to the model
    triggered.sort(key=lambda x: -_IMPORTANCE.get(x[0], 0))

    suggestions = [msg for _, msg in triggered[:top_n]]

    if not suggestions:
        suggestions = [
            "No specific weak areas detected against standard thresholds, but the "
            "model still flags elevated risk -- consider a manual review with an academic advisor."
        ]

    return suggestions
