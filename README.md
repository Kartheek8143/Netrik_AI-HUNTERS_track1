Team: AI Hunters
College: G. Pulla Reddy Engineering College

Team Members

S. Mythili

T. Yaswanth

S. Rohan

T. Kartheek

Project Title

AI HR Agent — an audit-ready, deterministic HR automation engine for resume screening, interview scheduling, structured interview generation, leave management, and escalation handling.

Project Description (implementation summary)

AI HR Agent converts raw resumes, job descriptions, availability windows, and leave requests into a fully auditable hiring pipeline that outputs the exact JSON required by the hackathon scoring system. Key capabilities implemented in code:

Deterministic resume screening using a fixed-vocabulary TF-IDF pipeline combined with weighted skill matching and experience scoring. Every candidate receives an explainable score_breakdown.

Constraint-aware interview scheduling that respects business hours (10:00–17:30 IST), enforces a 10-minute buffer between interviews, matches expertise & interview type, and uses load-balanced selection with deterministic fallbacks.

Interview question generation with an LLM-first path (Groq) and a deterministic template fallback, producing exactly 8 structured questions (3 technical, 2 behavioral, 2 situational, 1 candidate-specific) each with 4 measurable evaluation points.

Policy-first leave management that counts working days, enforces notice and consecutive-day limits, checks team capacity, and includes an advisory ML risk scorer (if model artifacts are present). Rule violations always override ML suggestions.

Rule-based escalation detection with severity levels (high/medium/low) and urgent/distress detection.

A strict finite-state machine (FSM) controlling candidate transitions with audit logging, idempotency checks, and terminal-state protection.

Final export_results() returns the exact hackathon JSON format.

Resume Screening Engine — Technical Details

The resume ranking system is deterministic, explainable, and tuned for reproducibility.

TF-IDF + Semantic Matching

Fixed vocabulary derived from curated SKILL_MAP (no vocabulary drift).

max_features = 6000 (explicitly fixed).

n-gram range = (1, 2) (unigrams + bigrams).

sublinear_tf = True and norm = "l2".

Vector comparison uses Cosine Similarity and cosine values are used directly (no min-max rescaling).

Scoring breakdown

Semantic score (TF-IDF cosine) — 50% weight

Weighted skill score — 35% weight (required skills = 3 pts, preferred = 1 pt)

Experience score — 15% weight (ratio based, capped at 1.2 then normalized)

Other features

Required skills are boosted (JD skills repeated 3×) and echoed into resumes that already mention them to tighten semantic alignment.

Two-stage sort: primary = required-skill coverage, secondary = final match score.

Per-candidate score_breakdown (semantic_score, skill_score, experience_score, bonuses, penalties, final_score) for explainability.

Interview Question Generator

Primary: Groq LLaMA 3.3 — llama-3.3-70b-versatile (Groq integration used when GROQ_API_KEY is present).
Fallback: Deterministic templates (offline safe).

Generator guarantees:

Exactly 8 structured questions (3 technical, 2 behavioral, 2 situational, 1 candidate-specific).

Each question includes: question, type, category, difficulty and exactly 4 measurable evaluation_points.

Strict JSON validation and auto-regeneration attempts for compliance.

(Explicit model reference: Groq LLaMA 3.3, model id used: llama-3.3-70b-versatile.)

Interview Scheduling System

Constraint-aware deterministic scheduler:

Business hours enforced: 10:00 — 17:30 IST

Timezone aware (Asia/Kolkata) datetimes

10-minute buffer between interviews

Expertise/type matching and duration fitting

Load-balanced interviewer selection (fewest bookings preferred)

Stepwise deterministic fallbacks (including an earliest-available search)

Leave Management

Policy-first leave processing with robust checks:

Working-day counting (Mon–Fri)

Minimum notice and max consecutive day checks

Team capacity check (configurable max_leave_per_day)

Overlap detection against approved leaves

Optional ML advisory (RandomForest, random_state=42) that provides ml_confidence and risk_score — advisory only (rules take precedence)

Escalation Handler

Keyword and compound detection rules for severity:

HIGH / MEDIUM / LOW categories

Harassment + emotional distress → HIGH

Urgency word detection for escalation prioritization

Structured logging of escalations for audit

FSM Architecture (detailed)

Finite State Machine drives pipeline correctness and auditability.

States (PipelineStatus)
applied → processing → shortlisted → interview_scheduled → interviewed → selected (terminal) / rejected (terminal)

Allowed transitions (programmatically enforced)

applied → processing | rejected

processing → shortlisted | rejected

shortlisted → interview_scheduled | rejected

interview_scheduled → interviewed | rejected

interviewed → selected | rejected

Preconditions & Guards

Candidate must exist in pipeline before transitions.

Terminal states (selected, rejected) are locked — no further changes.

Idempotent transitions (same → same) are rejected.

Whitelist enforcement prevents invalid jumps.

interview_scheduled requires a booked slot entry in _booked_slots.

Every successful transition is recorded to audit_trail with timestamp & reason (enum-driven).

Auditability

Audit entries include: candidate_id, from, to, timestamp (ISO), and reason.

Allows judges/auditors to replay decision history and verify correctness.

Export / Hackathon Format

export_results() returns the exact required structure:

{
  "team_id": "AI_Hunters",
  "track": "track_2_hr_agent",
  "results": {
    "resume_screening": {"ranked_candidates": [...], "scores": [...]},
    "scheduling": {"interviews_scheduled": [...], "conflicts": [...]},
    "questionnaire": {"questions": [...]},
    "pipeline": {"candidates": {id: status}},
    "leave_management": {"processed_requests": [...]},
    "escalations": [...]
  }
}
How to run (quick)

Clone the repo.

(Optional) Create venv and install deps:

python -m venv venv
source venv/bin/activate   # macOS / Linux
venv\Scripts\activate      # Windows
pip install -r requirements.txt

Dependencies (example): pandas, numpy, scikit-learn, joblib, groq (only if using Groq API).
