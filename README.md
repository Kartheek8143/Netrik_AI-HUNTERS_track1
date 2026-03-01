Team Details

Team Name: AI Hunters

Team Members:
S. Mythili
T. Yaswanth
S. Rohan
T. Kartheek

College Name:
G. Pulla Reddy Engineering College, Kurnool

Project Title

AI HR Agent — Intelligent Recruitment & HR Automation System

Project Description

The AI HR Agent is a fully integrated, enterprise-grade Human Resource Automation System designed to streamline and optimize the entire recruitment and HR management workflow.

This system simulates a real-world Applicant Tracking System (ATS) and HR Operations Engine by combining:

Resume Screening and Ranking
Interview Scheduling
AI-Based Questionnaire Generation
Leave Management with Policy Enforcement
Escalation Detection System
FSM-Based Candidate Pipeline Tracking

The solution is deterministic, scalable, and designed with production-level architecture principles.

1. Resume Screening and Candidate Ranking

The system uses:

TF-IDF Vectorization
Cosine Similarity
Weighted Skill Matching
Experience Normalization
Coverage-based ATS Sorting

Ranking Formula:

Final Score =
0.50 × Semantic Score

* 0.35 × Skill Match Score
* 0.15 × Experience Score

Additional Enhancements:

Required skill coverage prioritization
Experience gap penalty enforcement
Soft skill bonus scoring
Deterministic scoring (no randomness)

This ensures fair, explainable, and reproducible candidate ranking.

2. Intelligent Interview Scheduling

The scheduling engine performs:

Skill-based interviewer matching
Time window overlap detection
Business hour enforcement (10:00 AM – 5:30 PM)
10-minute buffer enforcement
Load-balanced interviewer distribution
Automatic fallback scheduling
Manual time override option

The system also:

Prevents double booking
Logs scheduling conflicts
Ensures FSM state compliance

3. AI Interview Questionnaire Generator

The system generates:

3 Technical Questions (based on required skills)
2 Behavioral Questions (STAR-based)
2 Situational Questions
1 Candidate-Specific Question

Primary Mode:

LLM-powered structured question generation (Groq API)

Fallback Mode:

Deterministic rule-based question templates

Each question includes:

Difficulty level
Category
4 structured evaluation points

4. FSM-Based Recruitment Pipeline

The system uses a Finite State Machine (FSM) to control candidate status transitions:

APPLIED → PROCESSING → SHORTLISTED
→ INTERVIEW_SCHEDULED → INTERVIEWED
→ SELECTED / REJECTED

Features:

Valid transition enforcement
Terminal state lock
Audit trail logging
Precondition checks (e.g., no interview without slot booking)

This ensures strict workflow integrity.

5. Leave Management System

The leave module integrates:

Leave balance auto-calculation from dataset
Working-day calculation (Mon–Fri)
Max consecutive day enforcement
Minimum notice validation
Medical document enforcement (if required)
Team leave capacity control (max 3 per day)
Overlapping leave detection

ML Integration:

A trained model predicts:

Leave approval probability
Risk score
Confidence percentage

However, rule-based policy enforcement remains authoritative.

6. Intelligent Escalation Detection

The system detects and categorizes HR queries based on severity:

High (harassment, legal issues, discrimination)
Medium (policy conflicts, compensation issues)
Low (general queries)

Includes:

Urgency keyword detection
Compound severity detection
Structured escalation logging

Final Output Structure

The system generates hackathon-compliant structured JSON output:

{
"team_id": "AI_Hunters",
"track": "track_2_hr_agent",
"results": {
"resume_screening": {...},
"scheduling": {...},
"questionnaire": {...},
"pipeline": {...},
"leave_management": {...},
"escalations": [...]
}
}

Fully deterministic and evaluation-ready.

Technologies Used

Python 3
Pandas
Scikit-Learn
TF-IDF and Cosine Similarity
Random Forest (Leave ML)
Groq LLM (llama-3.3-70b-versatile)
Dataclasses
FSM Architecture
Rule-Based Engines
