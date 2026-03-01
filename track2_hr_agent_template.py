#!/usr/bin/env python3
"""
=====================================================
HACKATHON TEMPLATE — Track 2
AI HR Agent
=====================================================
Starter template. Build on top of this.
DO NOT change class interfaces or output format.

Required output format for scoring:
{
    "team_id": "your_team_name",
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
=====================================================
"""

import os
import pandas as pd
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────
CONFIG = {
    "team_id": "AI_Hunters",
    "llm_provider": "groq",
    "llm_model": "llama-3.3-70b-versatile",
    "embedding_model": None,   # not used
}


# ─────────────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────────────
class PipelineStatus(Enum):
    APPLIED = "applied"
    PROCESSING = "processing"
    SHORTLISTED = "shortlisted"
    INTERVIEW_SCHEDULED = "interview_scheduled"
    INTERVIEWED = "interviewed"
    SELECTED = "selected"
    REJECTED = "rejected"

    # Valid transitions (enum → enum)
    @staticmethod
    def valid_transitions():
        return {
            PipelineStatus.APPLIED: [PipelineStatus.PROCESSING, PipelineStatus.REJECTED],
            PipelineStatus.PROCESSING: [PipelineStatus.SHORTLISTED, PipelineStatus.REJECTED],
            PipelineStatus.SHORTLISTED: [PipelineStatus.INTERVIEW_SCHEDULED, PipelineStatus.REJECTED],
            PipelineStatus.INTERVIEW_SCHEDULED: [PipelineStatus.INTERVIEWED, PipelineStatus.REJECTED],
            PipelineStatus.INTERVIEWED: [PipelineStatus.SELECTED, PipelineStatus.REJECTED],
            PipelineStatus.SELECTED: [],
            PipelineStatus.REJECTED: [],
        }


class TransitionReason(Enum):
    """Structured reasons for FSM state transitions."""
    SCREENING_STARTED = "screening_started"
    AUTO_SHORTLISTED = "auto_shortlisted"
    SLOT_BOOKED = "slot_booked"
    INTERVIEW_PASSED = "interview_passed"
    INTERVIEW_FAILED = "interview_failed"
    MANUAL_REJECTION = "manual_rejection"
    MANUAL_SELECTION = "manual_selection"
    MANUAL_TRANSITION = "manual_transition"


@dataclass
class Candidate:
    candidate_id: str
    name: str
    email: str
    resume_text: str
    skills: List[str] = field(default_factory=list)
    experience_years: float = 0.0
    match_score: float = 0.0
    status: PipelineStatus = PipelineStatus.APPLIED
    score_breakdown: Optional[Dict[str, Any]] = None
    coverage: float = 0.0

@dataclass
class JobDescription:
    job_id: str
    title: str
    description: str
    required_skills: List[str] = field(default_factory=list)
    preferred_skills: List[str] = field(default_factory=list)
    min_experience: float = 0.0

@dataclass
class InterviewSlot:
    slot_id: str
    interviewer_id: str
    start_time: datetime
    end_time: datetime
    is_available: bool = True

@dataclass
class LeaveRequest:
    request_id: str
    employee_id: str
    leave_type: str         # casual, sick, earned, etc.
    start_date: datetime
    end_date: datetime
    reason: str
    status: str = "pending"  # pending, approved, rejected
    policy_violations: List[str] = field(default_factory=list)

@dataclass
class LeavePolicy:
    leave_type: str
    annual_quota: int
    max_consecutive_days: int
    min_notice_days: int
    requires_document: bool = False  # e.g., medical certificate for sick leave


# ─────────────────────────────────────────────────────
# ABSTRACT INTERFACES — Implement these
# ─────────────────────────────────────────────────────

class ResumeScreener(ABC):
    @abstractmethod
    def extract_skills(self, resume_text: str) -> List[str]:
        """Extract skills from resume text."""
        pass

    @abstractmethod
    def rank_candidates(self, candidates: List[Candidate], jd: JobDescription) -> List[Candidate]:
        """Rank candidates against job description. Returns sorted list."""
        pass

class InterviewScheduler(ABC):
    @abstractmethod
    def schedule_interview(self, candidate: Candidate, available_slots: List[InterviewSlot]) -> Optional[InterviewSlot]:
        """Find and book an interview slot. Returns booked slot or None."""
        pass

class QuestionnaireGenerator(ABC):
    @abstractmethod
    def generate_questions(self, jd: JobDescription, candidate: Optional[Candidate] = None) -> List[Dict]:
        """Generate structured interview questions. Returns list of {question, type, category}."""
        pass

class LeaveManager(ABC):
    @abstractmethod
    def process_leave_request(self, request: LeaveRequest, policy: LeavePolicy,
                              current_balance: int) -> Dict:
        """Process a leave request. Returns {approved: bool, reason: str, violations: [...]}."""
        pass

class EscalationHandler(ABC):
    @abstractmethod
    def should_escalate(self, query: str, context: Dict) -> tuple:
        """Returns (should_escalate: bool, reason: str, priority: str)."""
        pass


# ─────────────────────────────────────────────────────
# REFERENCE IMPLEMENTATIONS
# ─────────────────────────────────────────────────────

class RuleBasedResumeScreener(ResumeScreener):
    """Deterministic rule-based resume screening with TF-IDF + skill matching."""

    # ── Comprehensive alias → canonical skill map ──────────────
    SKILL_MAP = {
        # Programming languages
        "python": "Python", "python3": "Python", "py": "Python",
        "java": "Java", "javascript": "JavaScript", "js": "JavaScript",
        "typescript": "TypeScript", "ts": "TypeScript",
        "c++": "C++", "cpp": "C++", "c#": "C#", "csharp": "C#",
        "go": "Go", "golang": "Go", "rust": "Rust", "ruby": "Ruby",
        "php": "PHP", "swift": "Swift", "kotlin": "Kotlin",
        "scala": "Scala", "r": "R", "matlab": "MATLAB",
        "shell": "Shell Scripting", "bash": "Shell Scripting",
        # Web / API
        "html": "HTML", "html5": "HTML", "css": "CSS", "css3": "CSS",
        "react": "React", "reactjs": "React", "react.js": "React",
        "angular": "Angular", "angularjs": "Angular",
        "vue": "Vue.js", "vuejs": "Vue.js", "vue.js": "Vue.js",
        "node": "Node.js", "nodejs": "Node.js", "node.js": "Node.js",
        "express": "Express.js", "expressjs": "Express.js",
        "django": "Django", "flask": "Flask", "fastapi": "FastAPI",
        "spring": "Spring", "spring boot": "Spring Boot",
        "rest api": "REST APIs", "rest apis": "REST APIs",
        "restful": "REST APIs", "restful api": "REST APIs",
        "graphql": "GraphQL",
        # Data / ML / AI
        "machine learning": "Machine Learning", "ml": "Machine Learning",
        "deep learning": "Deep Learning", "dl": "Deep Learning",
        "artificial intelligence": "Artificial Intelligence", "ai": "Artificial Intelligence",
        "nlp": "NLP", "natural language processing": "NLP",
        "computer vision": "Computer Vision", "cv": "Computer Vision",
        "data science": "Data Science", "data analysis": "Data Analysis",
        "data engineering": "Data Engineering",
        "tensorflow": "TensorFlow", "tf": "TensorFlow",
        "pytorch": "PyTorch", "torch": "PyTorch",
        "keras": "Keras", "scikit-learn": "Scikit-learn", "sklearn": "Scikit-learn",
        "pandas": "Pandas", "numpy": "NumPy", "scipy": "SciPy",
        "matplotlib": "Matplotlib", "seaborn": "Seaborn",
        "spark": "Apache Spark", "pyspark": "Apache Spark",
        "hadoop": "Hadoop", "hive": "Hive", "kafka": "Kafka",
        "tableau": "Tableau", "power bi": "Power BI", "powerbi": "Power BI",
        # Databases
        "sql": "SQL", "mysql": "MySQL", "postgresql": "PostgreSQL",
        "postgres": "PostgreSQL", "sqlite": "SQLite",
        "mongodb": "MongoDB", "mongo": "MongoDB",
        "redis": "Redis", "elasticsearch": "Elasticsearch",
        "cassandra": "Cassandra", "dynamodb": "DynamoDB",
        "oracle": "Oracle DB", "oracle db": "Oracle DB",
        # Cloud & DevOps
        "aws": "AWS", "amazon web services": "AWS",
        "azure": "Azure", "microsoft azure": "Azure",
        "gcp": "GCP", "google cloud": "GCP", "google cloud platform": "GCP",
        "docker": "Docker", "kubernetes": "Kubernetes", "k8s": "Kubernetes",
        "terraform": "Terraform", "ansible": "Ansible",
        "jenkins": "Jenkins", "ci/cd": "CI/CD", "ci cd": "CI/CD",
        "devops": "DevOps",
        # Tools & Version Control
        "git": "Git", "github": "GitHub", "gitlab": "GitLab",
        "jira": "Jira", "confluence": "Confluence",
        "linux": "Linux", "unix": "Unix",
        "agile": "Agile", "scrum": "Scrum",
        "microservices": "Microservices",
        # Security
        "cybersecurity": "Cybersecurity", "cyber security": "Cybersecurity",
        "penetration testing": "Penetration Testing",
        "network security": "Network Security",
        # Soft skills
        "leadership": "Leadership", "communication": "Communication",
        "problem-solving": "Problem Solving", "problem solving": "Problem Solving",
        "teamwork": "Teamwork", "team work": "Teamwork",
        "project management": "Project Management",
        "time management": "Time Management",
        "critical thinking": "Critical Thinking",
        # Statistics
        "statistics": "Statistics", "statistical analysis": "Statistics",
        "cloud computing": "Cloud Computing",
    }

    def extract_skills(self, resume_text: str) -> List[str]:
        """
        Deterministic rule-based skill extractor.

        Uses a curated SKILL_MAP (alias -> canonical name) with regex
        word-boundary matching.  No LLM, no pretrained models, no APIs.

        Returns a sorted, deduplicated list of canonical skill names.
        """
        import re

        if not resume_text:
            return []

        text_lower = resume_text.lower()
        found: set = set()

        # Match longest aliases first to avoid partial shadowing
        # (e.g., "machine learning" before "ml")
        for alias in sorted(self.SKILL_MAP.keys(), key=len, reverse=True):
            pattern = r'(?<!\w)' + re.escape(alias) + r'(?!\w)'
            if re.search(pattern, text_lower):
                found.add(self.SKILL_MAP[alias])

        return sorted(found)

    def rank_candidates(self, candidates: List[Candidate], jd: JobDescription) -> List[Candidate]:
        """
        Enterprise-grade deterministic candidate ranking.

            final_score = 0.60 * semantic_score
                        + 0.25 * weighted_skill_score
                        + 0.15 * exp_score

        Key design choices
        ------------------
        • TF-IDF: max_features=6000, bigrams, sublinear TF, L2 norm.
        • Cosine similarity used *directly* (already in [0,1] for L2-normed
          non-negative TF-IDF vectors) — no min-max rescaling.
        • Required skills boosted 3× in JD text AND echoed once into resumes
          that already mention them (tighter semantic alignment).
        • Required skill = 3 pts, preferred = 1 pt.
        • Experience: ratio-based (min(exp/req, 1.2)), not min-max across
          candidates, so ranking is stable even for single-candidate calls.
        • Heavy penalty if candidate has < 50 % of required experience.
        • Fully deterministic, no external APIs, no randomness.
        """
        import re
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

        if not candidates:
            return candidates

        # ── skill tables ───────────────────────────────────────────
        required_lower  = [s.lower() for s in (jd.required_skills  or [])]
        preferred_lower = [s.lower() for s in (jd.preferred_skills or [])]

        # ── helper: strict word-boundary skill match ────────────────
        def _has_skill(skill: str, candidate_skills: set) -> bool:
            """True if *skill* matches any candidate skill using word boundaries."""
            for cs in candidate_skills:
                if skill == cs:
                    return True
                pattern = r'(?<!\w)' + re.escape(skill) + r'(?!\w)'
                if re.search(pattern, cs):
                    return True
                # Reverse: check if candidate skill is in the jd skill
                pattern_r = r'(?<!\w)' + re.escape(cs) + r'(?!\w)'
                if re.search(pattern_r, skill):
                    return True
            return False

        # ── Part A: JD-skill enrichment ────────────────────────────
        #   Scan raw resume_text for JD skills not yet in candidate.skills.
        jd_all_skills = list(dict.fromkeys(required_lower + preferred_lower))

        for candidate in candidates:
            existing = {s.lower() for s in (candidate.skills or [])}
            text_lower = (candidate.resume_text or "").lower()
            for skill in jd_all_skills:
                if skill not in existing:
                    pattern = r'(?<!\w)' + re.escape(skill) + r'(?!\w)'
                    if re.search(pattern, text_lower):
                        candidate.skills.append(skill)
                        existing.add(skill)

        # ================================================================
        # STEP 1 — Semantic similarity (TF-IDF cosine)
        # ================================================================
        resume_texts_raw = [c.resume_text or "" for c in candidates]

        # Required-skill boosting in JD: append each skill 3×
        jd_text = jd.description or ""
        if required_lower:
            jd_text += " " + " ".join(s for s in required_lower for _ in range(3))

        # Required-skill echo in resumes: if a resume already mentions a
        # required skill, append it once more to tighten cosine alignment.
        boosted_resumes = []
        for i, candidate in enumerate(candidates):
            text = resume_texts_raw[i]
            c_skills = {s.lower() for s in (candidate.skills or [])}
            extras = [s for s in required_lower if _has_skill(s, c_skills)]
            if extras:
                text += " " + " ".join(extras)
            boosted_resumes.append(text)

        # FIX 5: fixed vocabulary from SKILL_MAP ensures TF-IDF is deterministic
        # regardless of candidate pool size (no vocabulary drift between runs).
        fixed_vocab = list(set(
            token
            for alias, canonical in self.SKILL_MAP.items()
            for token in [alias.lower(), canonical.lower()]
        ))
        tfidf = TfidfVectorizer(
            vocabulary=fixed_vocab,          # FIX 5: fixed vocab — never changes
            stop_words=None,                  # vocabulary already curated
            ngram_range=(1, 2),
            sublinear_tf=True,
            norm="l2",
        )
        matrix = tfidf.fit_transform(boosted_resumes + [jd_text])

        resume_vecs = matrix[:-1]
        jd_vec      = matrix[-1:]

        # Cosine similarity is already in [0, 1] for L2-normed non-negative
        # vectors — use directly, no min-max.
        semantic_scores = np.maximum(
            sk_cosine(resume_vecs, jd_vec).flatten(), 0.0
        )

        # ================================================================
        # STEP 2 — Weighted skill score
        # ================================================================
        #   Required = 3 pts, Preferred = 1 pt
        total_pts = max(len(required_lower) * 3 + len(preferred_lower) * 1, 1)

        def _weighted_skill(candidate_skills: List[str]) -> float:
            c_skills = {s.lower() for s in (candidate_skills or [])}
            pts = 0
            for skill in required_lower:
                if _has_skill(skill, c_skills):
                    pts += 3
            for skill in preferred_lower:
                if _has_skill(skill, c_skills):
                    pts += 1
            return pts / total_pts

        skill_scores = np.array([_weighted_skill(c.skills) for c in candidates])

        # ================================================================
        # STEP 3 — Experience score (ratio-based, not min-max)
        # ================================================================
        req_exp = max(jd.min_experience or 0.0, 1.0)   # floor to 1 to avoid /0

        def _exp_score(exp_years: float) -> float:
            return min((exp_years or 0.0) / req_exp, 1.2)

        exp_scores = np.array([_exp_score(c.experience_years) for c in candidates])
        # Normalize 1.2-cap range to [0, 1] for the formula
        exp_scores_norm = exp_scores / 1.2

        # ================================================================
        # STEP 4 — Base final score (skill-emphasised weights)
        # ================================================================
        final_scores = (
            0.50 * semantic_scores +
            0.35 * skill_scores    +
            0.15 * exp_scores_norm
        )

        # ================================================================
        # STEP 5 — Two-stage ATS adjustments
        # ================================================================
        n_required = max(len(required_lower), 1)
        required_exp = jd.min_experience or 0.0

        def _coverage(candidate_skills: List[str]) -> float:
            c_skills = {s.lower() for s in (candidate_skills or [])}
            matched = sum(1 for s in required_lower if _has_skill(s, c_skills))
            return matched / n_required

        SOFT_SKILLS = {"leadership", "communication", "problem-solving"}

        # Store per-candidate coverage for two-stage sort
        coverages = []

        for i, candidate in enumerate(candidates):
            bonuses   = 0.0
            penalties = 0.0

            # ── STAGE 1: Required-skill coverage (hard filter) ────
            cov = _coverage(candidate.skills)
            coverages.append(cov)
            cov_adjustment = 0.0

            if cov == 0.0:
                # Zero required skills matched → strong multiplicative penalty
                final_scores[i] *= 0.6
                cov_adjustment = -(final_scores[i] * 0.4)  # track the reduction
            elif cov < 0.5:
                penalties += 0.15
                cov_adjustment = -0.15
            elif cov == 1.0:
                bonuses += 0.10
                cov_adjustment = +0.10

            # ── missing required skill penalty: -0.03 each ────────
            c_skills_lower = {s.lower() for s in (candidate.skills or [])}
            missing = sum(
                1 for s in required_lower if not _has_skill(s, c_skills_lower)
            )
            penalties += 0.03 * missing

            # ── soft skill bonus: +0.01 each, max +0.03 ──────────
            text_lower = (candidate.resume_text or "").lower()
            soft_count = sum(1 for s in SOFT_SKILLS if s in text_lower)
            bonuses += min(0.03, 0.01 * soft_count)

            # ── STAGE 2: Experience-gap enforcement ───────────────
            candidate_exp = candidate.experience_years or 0.0
            experience_gap = candidate_exp - required_exp
            exp_adjustment = 0.0

            if candidate_exp < 0.5 * required_exp:
                # Heavy penalty: less than half the required experience
                exp_adjustment = -0.10
                penalties += 0.10
            elif candidate_exp < required_exp:
                # Moderate penalty proportional to gap
                exp_adjustment = -min(0.08, 0.02 * abs(experience_gap))
                penalties += abs(exp_adjustment)
            else:
                # Small bonus proportional to surplus (capped)
                exp_adjustment = +min(0.07, 0.01 * experience_gap)
                bonuses += exp_adjustment

            final_scores[i] += bonuses - penalties

            # ── explainability breakdown (rubric-compliant schema) ──────
            # FIX 6: always populate all required score_breakdown keys
            n_req = max(len(required_lower), 1)
            n_pref = max(len(preferred_lower), 1)
            c_set_lower = {s.lower() for s in (candidate.skills or [])}
            req_matched   = sum(1 for s in required_lower  if _has_skill(s, c_set_lower))
            pref_matched  = sum(1 for s in preferred_lower if _has_skill(s, c_set_lower))
            candidate.score_breakdown = {
                # Legacy internal keys (retained for backward compat)
                "semantic":                    round(float(semantic_scores[i]), 4),
                "skill":                       round(float(skill_scores[i]),   4),
                "experience":                  round(float(exp_scores_norm[i]),4),
                "coverage":                    round(cov, 4),
                "experience_gap":              round(experience_gap, 2),
                "coverage_bonus_or_penalty":   round(cov_adjustment, 4),
                "experience_adjustment":       round(exp_adjustment, 4),
                "bonuses":                     round(bonuses,   4),
                "penalties":                   round(penalties, 4),
                # FIX 6: rubric-required keys
                "semantic_score":              round(float(semantic_scores[i]), 4),
                "skill_score":                 round(float(skill_scores[i]),   4),
                "experience_score":            round(float(exp_scores_norm[i]),4),
                "required_skills_matched":     req_matched,
                "preferred_skills_matched":    pref_matched,
                "total_required":              len(required_lower),
                "total_preferred":             len(preferred_lower),
                "final_score":                 round(float(final_scores[i]), 6),
                "weights": {
                    "semantic":    0.50,
                    "skill":       0.35,
                    "experience":  0.15,
                },
            }

        # ================================================================
        # STEP 6 — Clip, round, two-stage sort
        # ================================================================
        final_scores = np.clip(final_scores, 0.0, 1.0)

        for i, candidate in enumerate(candidates):
            candidate.match_score = round(float(final_scores[i]), 6)
            candidate.coverage    = coverages[i]

        # Two-stage ATS sort:
        #   PRIMARY   = coverage  (descending) — required-skill priority
        #   SECONDARY = match_score (descending) — within same coverage tier
        return sorted(
            candidates,
            key=lambda c: (c.coverage, c.match_score),
            reverse=True,
        )



class BasicInterviewScheduler(InterviewScheduler):
    """Production-grade deterministic constraint-based interview scheduler.

    Implements a full ATS-style scheduling engine:
      Step 1  — Load & validate Excel data with timezone normalization
      Step 2  — Business hours enforcement (10:00 – 17:30)
      Step 3  — Candidate prioritization (match_score + coverage)
      Step 4  — Interviewer filtering (type + expertise)
      Step 5  — Time overlap computation
      Step 6  — Business hours clipping
      Step 7  — Duration fit validation
      Step 8  — 10-minute buffer enforcement
      Step 9  — Load-balanced slot selection
      Step 10 — Conflict diagnostics
    """

    # ── system constants ──────────────────────────────────────
    BUSINESS_START_HOUR   = 10
    BUSINESS_END_HOUR     = 17
    BUSINESS_END_MINUTE   = 30
    MIN_BUFFER_MINUTES    = 10

    def __init__(self):
        # interviewer_id -> list of (start, end) tuples already booked
        self._interviewer_bookings: Dict[str, List[tuple]] = {}

    # ── data loading & validation ─────────────────────────────
    def load_sample_availability(self) -> tuple:
        """Load realistic embedded sample datasets for interviewers and candidates.
        
        Returns (interviewer_df, candidate_df) with timezone-aware datetimes.
        """
        from zoneinfo import ZoneInfo
        from datetime import datetime, timedelta

        DEFAULT_TZ = ZoneInfo("Asia/Kolkata")
        # FIX 2: use dynamic dates so slots are always in the future
        now = datetime.now(DEFAULT_TZ)
        d1 = (now + timedelta(days=1)).strftime("%Y-%m-%d")   # tomorrow
        d2 = (now + timedelta(days=2)).strftime("%Y-%m-%d")   # day after tomorrow

        # ── 1. Create Interviewer Dataset (5 Interviewers) ────
        interviewer_data = [
            ["INT001", "technical", "Python",           f"{d1} 10:00", f"{d1} 13:00", "Asia/Kolkata", True],
            ["INT002", "technical", "Machine Learning", f"{d1} 11:00", f"{d1} 16:00", "Asia/Kolkata", False], # Unavailable
            ["INT003", "hr",        "Behavioral",        f"{d1} 10:00", f"{d1} 17:30", "Asia/Kolkata", True],
            ["INT004", "technical", "Docker",            f"{d1} 14:00", f"{d1} 17:30", "Asia/Kolkata", True],
            ["INT005", "technical", "SQL",               f"{d2} 10:00", f"{d2} 15:00", "Asia/Kolkata", False], # Unavailable
        ]
        i_df = pd.DataFrame(interviewer_data, columns=[
            "Interviewer_ID", "Interview_Type", "Expertise", "Start_Time", "End_Time", "Timezone", "Is_Available"
        ])

        # ── 2. Create Candidate Dataset (3 Candidates) ──────
        candidate_data = [
            ["C001", "technical", 60, f"{d1} 11:00", f"{d1} 15:00", "Asia/Kolkata"],
            ["C002", "technical", 45, f"{d1} 10:00", f"{d1} 12:00", "Asia/Kolkata"],
            ["C003", "technical", 60, f"{d2} 10:30", f"{d2} 14:00", "Asia/Kolkata"],
        ]
        c_df = pd.DataFrame(candidate_data, columns=[
            "Candidate_ID", "Preferred_Type", "Duration_Minutes", "Start_Time", "End_Time", "Timezone"
        ])

        # ── 3. Timezone conversion and internal formatting ────
        def _parse_df(df):
            df["_start"] = [pd.Timestamp(t).tz_localize(ZoneInfo(tz)).to_pydatetime() 
                            for t, tz in zip(df["Start_Time"], df["Timezone"])]
            df["_end"]   = [pd.Timestamp(t).tz_localize(ZoneInfo(tz)).to_pydatetime() 
                            for t, tz in zip(df["End_Time"], df["Timezone"])]
            return df

        return _parse_df(i_df), _parse_df(c_df)

    def find_fallback_slot(
        self,
        interviewer_df: pd.DataFrame,
        duration_min: int,
        slot_counter: List[int],
    ) -> Optional[InterviewSlot]:
        """Rule 2 Fallback: Find the earliest available slot in business hours, ignoring candidate preference.
        
        Always looks at 'Tomorrow' starting from 10:00 AM.
        """
        from zoneinfo import ZoneInfo
        from datetime import datetime, timedelta
        
        DEFAULT_TZ = ZoneInfo("Asia/Kolkata")
        tomorrow = datetime.now(DEFAULT_TZ) + timedelta(days=1)
        
        # Sort interviewers by load for fairness
        i_sorted = interviewer_df.copy()
        i_sorted["load"] = i_sorted["Interviewer_ID"].apply(lambda x: len(self._interviewer_bookings.get(x, [])))
        i_sorted = i_sorted.sort_values("load")

        for _, irow in i_sorted.iterrows():
            i_id = irow["Interviewer_ID"]
            # We use a standard window: Tomorrow 10:00 - 17:30
            win_start = tomorrow.replace(hour=self.BUSINESS_START_HOUR, minute=0, second=0, microsecond=0)
            win_end = tomorrow.replace(hour=self.BUSINESS_END_HOUR, minute=self.BUSINESS_END_MINUTE, second=0, microsecond=0)
            
            # Step through the window in 15-min increments to find a fit
            curr = win_start
            while curr + timedelta(minutes=duration_min) <= win_end:
                t_start = curr
                t_end = curr + timedelta(minutes=duration_min)
                
                if not self._has_buffer_conflict(i_id, t_start, t_end):
                    # Found a spot!
                    slot_counter[0] += 1
                    is_available = irow.get("Is_Available", True)
                    slot = InterviewSlot(
                        slot_id=f"SLOT-{slot_counter[0]:03d}",
                        interviewer_id=i_id,
                        start_time=t_start,
                        end_time=t_end,
                        is_available=is_available,
                    )
                    # Register booking only if slot is available
                    if is_available:
                        self._interviewer_bookings.setdefault(i_id, []).append((t_start, t_end))
                    return slot
                curr += timedelta(minutes=15)
        return None

    # ── business hours clip ───────────────────────────────────
    @classmethod
    def _clip_business_hours(cls, start: datetime, end: datetime) -> tuple:
        """Clip a time window into [10:00, 17:30] on the same date."""
        bh_start = start.replace(
            hour=cls.BUSINESS_START_HOUR, minute=0, second=0, microsecond=0,
        )
        bh_end = start.replace(
            hour=cls.BUSINESS_END_HOUR, minute=cls.BUSINESS_END_MINUTE,
            second=0, microsecond=0,
        )
        clipped_start = max(start, bh_start)
        clipped_end   = min(end, bh_end)
        return clipped_start, clipped_end

    # ── buffer validation ─────────────────────────────────────
    def _has_buffer_conflict(self, interviewer_id: str,
                             new_start: datetime, new_end: datetime) -> bool:
        """True if the new slot violates the 10-min buffer constraint."""
        buf = timedelta(minutes=self.MIN_BUFFER_MINUTES)
        for (ex_start, ex_end) in self._interviewer_bookings.get(interviewer_id, []):
            if not (new_start >= ex_end + buf or new_end + buf <= ex_start):
                return True
        return False

    def _booking_count(self, interviewer_id: str) -> int:
        """Number of interviews already booked for this interviewer."""
        return len(self._interviewer_bookings.get(interviewer_id, []))

    # ── abstract interface (simple slot picker) ───────────────
    def schedule_interview(self, candidate: Candidate,
                           available_slots: List[InterviewSlot]) -> Optional[InterviewSlot]:
        """Pick the first available InterviewSlot from a pre-built list."""
        for slot in available_slots:
            if slot.is_available:
                if not self._has_buffer_conflict(
                    slot.interviewer_id, slot.start_time, slot.end_time
                ):
                    slot.is_available = False
                    bookings = self._interviewer_bookings.setdefault(
                        slot.interviewer_id, []
                    )
                    bookings.append((slot.start_time, slot.end_time))
                    return slot
        return None

    # ── main scheduling engine ────────────────────────────────
    def schedule_from_availability(
        self,
        candidate: Candidate,
        candidate_rows: List[Dict],
        interviewer_df,
        *,
        slot_counter: List[int],
    ) -> tuple:
        """Find a valid slot for *candidate* using constraint-aware matching.

        Load-balanced selection priority (deterministic):
          1. Earliest start time
          2. Fewest existing bookings for that interviewer (fairness)
          3. Shortest interviewer window (efficiency — fill tight windows first)

        Returns:
            (InterviewSlot, None)   on success
            (None, reason_string)   on failure
        """
        c_skills = {s.lower() for s in (candidate.skills or [])}
        if not candidate_rows:
            return None, "No candidate availability data"

        # Diagnostic flags for precise conflict reporting
        had_type_match   = False
        had_expert_match = False
        had_overlap      = False
        had_bh_fit       = False
        had_duration_fit = False

        # Collect valid slots: (start, end, interviewer_id, booking_count, window_mins)
        valid_slots: List[tuple] = []

        for crow in candidate_rows:
            pref_type    = crow.get("Preferred_Type", "")
            duration_min = int(crow.get("Duration_Minutes", 30))
            c_start      = crow.get("_start")
            c_end        = crow.get("_end")

            if not c_start or not c_end or c_start >= c_end:
                continue

            # ── STEP 4: filter by Interview_Type ──────────────
            type_matched = interviewer_df[
                interviewer_df["Interview_Type"] == pref_type
            ]
            if type_matched.empty:
                continue
            had_type_match = True

            # Expertise match (case-insensitive, substring-safe)
            if not c_skills:
                continue  # no skills → cannot match expertise

            def _expertise_match(expertise_str):
                import re as _re_sched
                exp_lower = str(expertise_str).lower()
                for s in c_skills:
                    if s == exp_lower:
                        return True
                    if _re_sched.search(r'(?<!\w)' + _re_sched.escape(s) + r'(?!\w)', exp_lower):
                        return True
                    if _re_sched.search(r'(?<!\w)' + _re_sched.escape(exp_lower) + r'(?!\w)', s):
                        return True
                return False

            expert_matched = type_matched[
                type_matched["Expertise"].apply(_expertise_match)
            ]
            if expert_matched.empty:
                continue
            had_expert_match = True

            # ── STEP 5-8: overlap + clip + duration + buffer ──
            for _, irow in expert_matched.iterrows():
                i_start = irow["_start"]
                i_end   = irow["_end"]
                i_id    = irow["Interviewer_ID"]

                # Step 5: time overlap
                overlap_start = max(c_start, i_start)
                overlap_end   = min(c_end, i_end)
                if overlap_end <= overlap_start:
                    continue
                had_overlap = True

                # Step 6: business hours
                overlap_start, overlap_end = self._clip_business_hours(
                    overlap_start, overlap_end,
                )
                if overlap_end <= overlap_start:
                    continue
                had_bh_fit = True

                # Step 7: duration
                available_mins = (overlap_end - overlap_start).total_seconds() / 60
                if available_mins < duration_min:
                    continue
                had_duration_fit = True

                # Trim to exact duration
                slot_end = overlap_start + timedelta(minutes=duration_min)

                # Step 8: buffer
                # Window size for efficiency bias
                window_mins = (i_end - i_start).total_seconds() / 60
                is_available = irow.get("Is_Available", True)

                valid_slots.append((
                    overlap_start,
                    slot_end,
                    i_id,
                    self._booking_count(i_id),
                    window_mins,
                    is_available
                ))

        # ── STEP 10: precise conflict diagnosis ───────────────
        if not valid_slots:
            if not had_type_match:
                all_types = {r.get("Preferred_Type", "") for r in candidate_rows}
                return None, f"No interviewer type match for {all_types}"
            if not had_expert_match:
                return None, f"No expertise match for skills {candidate.skills}"
            if not had_overlap:
                return None, "No overlapping time window between candidate and interviewer"
            if not had_bh_fit:
                return None, "All overlaps fall outside business hours (10:00-17:30)"
            if not had_duration_fit:
                return None, "Duration insufficient in all valid windows"
            return None, "All valid slots rejected by 10-minute buffer constraint"

        # ── STEP 9: load-balanced, deterministic selection ────
        # Sort key (all ascending):
        #   1. Earliest start time
        #   2. Fewest bookings for that interviewer (load balance / fairness)
        #   3. Shortest interviewer window (efficiency — fill tight windows first)
        valid_slots.sort(key=lambda s: (s[0], s[3], s[4]))

        chosen_start, chosen_end, chosen_interviewer, _, _, chosen_is_available = valid_slots[0]

        # ── build InterviewSlot ───────────────────────────────
        slot_counter[0] += 1
        slot = InterviewSlot(
            slot_id=f"SLOT_{slot_counter[0]:04d}",
            interviewer_id=chosen_interviewer,
            start_time=chosen_start,
            end_time=chosen_end,
            is_available=chosen_is_available,
        )

        # Mark time as consumed only if available
        if chosen_is_available:
            bookings = self._interviewer_bookings.setdefault(chosen_interviewer, [])
            bookings.append((chosen_start, chosen_end))

        return slot, None



class LLMQuestionnaireGenerator(QuestionnaireGenerator):
    """Groq-powered structured interview question generator.

    Primary path:  Groq LLM (llama-3.3-70b-versatile) → structured JSON
    Fallback path: Rule-based templates (deterministic, no API)

    Produces exactly 8 questions:
      3 Technical  (per required skill)
      2 Behavioral (STAR format)
      2 Situational (role-keyword driven)
      1 Candidate-specific (personalised from candidate.skills)
    """

    # ── initialisation ────────────────────────────────────────
    def __init__(self):
        """Set up Groq client (lazy — only fails when actually called)."""
        self._groq_client = None
        self._last_generated: List[Dict] = []
        try:
            import os
            api_key = os.environ.get("GROQ_API_KEY", "")
            if api_key:
                from groq import Groq
                self._groq_client = Groq(api_key=api_key)
                logger.info("Groq client initialised successfully")
            else:
                logger.warning("GROQ_API_KEY not set — will use rule-based fallback")
        except ImportError:
            logger.warning("groq package not installed — will use rule-based fallback")
        except Exception as exc:
            logger.warning("Groq client init failed (%s) — will use rule-based fallback", exc)

    # ── prompt builder ────────────────────────────────────────
    def _build_prompt(self, jd: JobDescription, candidate: Optional[Candidate] = None) -> str:
        """Build the structured enterprise-grade prompt for the Groq LLM."""
        c_skills = ', '.join(candidate.skills) if candidate and candidate.skills else 'Not provided'
        c_exp = candidate.experience_years if candidate else 'Not provided'
        return (
            "==================================================\n"
            "ROLE CONTEXT\n"
            "==================================================\n"
            f"Job Title: {jd.title}\n"
            f"Job Description: {jd.description}\n"
            f"Required Skills: {', '.join(jd.required_skills or [])}\n"
            f"Preferred Skills: {', '.join(jd.preferred_skills or [])}\n"
            f"Minimum Experience Required: {jd.min_experience} years\n\n"
            "==================================================\n"
            "CANDIDATE CONTEXT\n"
            "==================================================\n"
            f"Candidate Skills: {c_skills}\n"
            f"Candidate Experience: {c_exp} years\n\n"
            "==================================================\n"
            "TASK\n"
            "==================================================\n\n"
            "Generate EXACTLY 8 structured interview questions.\n\n"
            "The order MUST be:\n\n"
            "1. Technical Question (Required Skill #1)\n"
            "2. Technical Question (Required Skill #2)\n"
            "3. Technical Question (Required Skill #3)\n"
            "4. Behavioral Question\n"
            "5. Behavioral Question\n"
            "6. Situational Question\n"
            "7. Situational Question\n"
            "8. Candidate-Specific Question\n\n"
            "==================================================\n"
            "QUALITY REQUIREMENTS\n"
            "==================================================\n\n"
            "TECHNICAL QUESTIONS:\n"
            "- Focus ONLY on REQUIRED skills first.\n"
            "- Must include real-world system thinking.\n"
            "- Must assess trade-offs, scalability, reliability, performance, or architecture decisions.\n"
            "- Difficulty must be \"medium\" or \"hard\".\n\n"
            "BEHAVIORAL QUESTIONS:\n"
            "- Must follow STAR evaluation thinking.\n"
            "- Must assess measurable business impact.\n"
            "- Must evaluate ownership, teamwork, or leadership.\n\n"
            "SITUATIONAL QUESTIONS:\n"
            "- Must simulate realistic enterprise constraints.\n"
            "- Must involve decision-making under pressure.\n"
            "- Must assess structured reasoning and prioritization.\n\n"
            "CANDIDATE-SPECIFIC QUESTION:\n"
            "- Must reference a strong skill from candidate profile.\n"
            "- Must ask about real project execution.\n"
            "- Must assess measurable outcome and impact.\n\n"
            "==================================================\n"
            "STRICT OUTPUT FORMAT\n"
            "==================================================\n\n"
            "Return ONLY valid JSON.\n\n"
            "Do NOT include:\n"
            "- Markdown\n"
            "- Backticks\n"
            "- Explanation\n"
            "- Comments\n"
            "- Additional text\n"
            "- Trailing commas\n\n"
            'The JSON MUST follow this exact structure:\n\n'
            '{"questions": [{"question": "string", "type": "technical | behavioral | situational", '
            '"category": "skill_or_topic", "difficulty": "medium | hard", '
            '"evaluation_points": ["specific measurable evaluation point 1", '
            '"specific measurable evaluation point 2", '
            '"specific measurable evaluation point 3", '
            '"specific measurable evaluation point 4"]}]}\n\n'
            "==================================================\n"
            "VALIDATION CONSTRAINTS (MANDATORY)\n"
            "==================================================\n\n"
            "- Exactly 8 questions.\n"
            "- Exactly 4 evaluation_points per question.\n"
            "- All keys must be present.\n"
            "- evaluation_points must be measurable and specific.\n"
            "- Avoid vague phrases like \"good understanding\".\n"
            "- JSON must be syntactically valid.\n\n"
            "If the output does not satisfy these constraints, internally regenerate before responding.\n\n"
            "Return only final valid JSON."
        )

    # ── question template banks ────────────────────────────
    _TECHNICAL_TEMPLATES = [
        "Explain how you would design a production-grade system using {skill}. What trade-offs would you consider?",
        "Describe a challenging problem you solved with {skill}. Walk us through your approach step by step.",
        "What are the best practices for {skill} in a large-scale enterprise environment? Give concrete examples.",
    ]

    _BEHAVIORAL_TEMPLATES = [
        {
            "category": "teamwork",
            "question": "Describe a time when you worked with a cross-functional team to deliver a project under a tight deadline. "
                        "(Use the STAR format: Situation, Task, Action, Result.)",
            "evaluation_points": ["Clarity of situation", "Defined role & responsibility",
                                  "Specific actions taken", "Measurable outcome"],
        },
        {
            "category": "conflict resolution",
            "question": "Tell me about a situation where you disagreed with a colleague on a technical approach. "
                        "How did you resolve it? (Use STAR format.)",
            "evaluation_points": ["Acknowledgement of differing views", "Communication approach",
                                  "Compromise or data-driven resolution", "Relationship outcome"],
        },
        {
            "category": "leadership",
            "question": "Give an example of when you took initiative to improve a process or mentor a teammate. "
                        "(Use STAR format.)",
            "evaluation_points": ["Proactiveness", "Impact on team", "Measurable improvement",
                                  "Leadership without authority"],
        },
    ]

    _SITUATIONAL_TEMPLATES: Dict[str, List[Dict]] = {
        "developer": [
            {
                "question": "A critical production service is down and you're the on-call engineer. Walk us through your incident response.",
                "evaluation_points": ["Triage methodology", "Communication plan", "Root-cause analysis", "Post-mortem thinking"],
            },
            {
                "question": "You discover that a recently merged PR introduced a subtle data-corruption bug. How do you handle it?",
                "evaluation_points": ["Rollback vs. hot-fix decision", "Stakeholder communication", "Regression prevention", "Testing strategy"],
            },
        ],
        "data": [
            {
                "question": "Your ML model's accuracy drops by 5% after a new data pipeline deployment. How do you diagnose and fix it?",
                "evaluation_points": ["Data drift detection", "Pipeline debugging", "Model retraining strategy", "Monitoring setup"],
            },
            {
                "question": "A stakeholder requests a model that must be both highly accurate and fully explainable. How do you approach this trade-off?",
                "evaluation_points": ["Model selection rationale", "Explainability techniques", "Stakeholder management", "Documentation quality"],
            },
        ],
        "default": [
            {
                "question": "You are assigned to a project with unclear requirements and a tight deadline. How do you proceed?",
                "evaluation_points": ["Requirement gathering", "Prioritisation", "Stakeholder alignment", "Iterative delivery"],
            },
            {
                "question": "A teammate is consistently missing deadlines, impacting the whole team. How would you handle this?",
                "evaluation_points": ["Empathy", "Direct communication", "Escalation judgement", "Constructive support"],
            },
        ],
    }

    _ROLE_KEYWORDS: Dict[str, List[str]] = {
        "developer": ["developer", "engineer", "software", "backend", "frontend", "fullstack", "full-stack", "sde", "swe", "devops"],
        "data": ["data", "machine learning", "ml", "ai", "analyst", "scientist", "deep learning"],
    }

    def _detect_role_category(self, jd: JobDescription) -> str:
        """Map JD title / description to a role category key."""
        text = (jd.title + " " + (jd.description or "")).lower()
        for category, keywords in self._ROLE_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                return category
        return "default"

    # ── rule-based fallback ─────────────────────────────────
    def _fallback_questions(self, jd: JobDescription, candidate: Optional[Candidate] = None) -> List[Dict]:
        """Deterministic rule-based question generation (no API needed)."""
        questions: List[Dict] = []

        # 1. Technical questions (3)
        skills = list(jd.required_skills or [])[:3]
        if len(skills) < 3 and (jd.preferred_skills or []):
            for ps in jd.preferred_skills:
                if ps not in skills:
                    skills.append(ps)
                if len(skills) >= 3:
                    break

        for i, skill in enumerate(skills[:3]):
            template = self._TECHNICAL_TEMPLATES[i % len(self._TECHNICAL_TEMPLATES)]
            questions.append({
                "question": template.format(skill=skill),
                "type": "technical",
                "category": skill,
                "difficulty": "medium",
                "evaluation_points": [
                    f"Depth of {skill} knowledge",
                    "System design thinking",
                    "Trade-off awareness",
                    "Real-world application examples",
                ],
            })

        # 2. Behavioral questions (2)
        for bt in self._BEHAVIORAL_TEMPLATES[:2]:
            questions.append({
                "question": bt["question"],
                "type": "behavioral",
                "category": bt["category"],
                "difficulty": "medium",
                "evaluation_points": list(bt["evaluation_points"]),
            })

        # 3. Situational questions (2)
        role_cat = self._detect_role_category(jd)
        sit_templates = self._SITUATIONAL_TEMPLATES.get(
            role_cat, self._SITUATIONAL_TEMPLATES["default"]
        )
        for st in sit_templates[:2]:
            questions.append({
                "question": st["question"],
                "type": "situational",
                "category": role_cat,
                "difficulty": "hard",
                "evaluation_points": list(st["evaluation_points"]),
            })

        # 4. Candidate-specific question (1)
        if candidate and candidate.skills:
            top_skill = candidate.skills[0]
            questions.append({
                "question": f"You listed '{top_skill}' among your skills. "
                            f"Can you walk us through a real project where {top_skill} "
                            f"was the critical technology, and what impact it had?",
                "type": "technical",
                "category": f"candidate-specific ({top_skill})",
                "difficulty": "medium",
                "evaluation_points": [
                    f"Depth of hands-on {top_skill} experience",
                    "Project scope & complexity",
                    "Measurable business impact",
                    "Lessons learned",
                ],
            })
        else:
            questions.append({
                "question": "What is the most technically challenging project you have "
                            "worked on, and what was your specific contribution?",
                "type": "technical",
                "category": "general",
                "difficulty": "medium",
                "evaluation_points": [
                    "Technical depth",
                    "Role clarity",
                    "Problem-solving approach",
                    "Outcome and learnings",
                ],
            })

        return questions

    # ── main entry point ──────────────────────────────────
    def generate_questions(self, jd: JobDescription, candidate: Optional[Candidate] = None) -> List[Dict]:
        """
        Generate 8 structured interview questions.

        Primary:  Groq LLM (llama-3.3-70b-versatile)
        Fallback: Deterministic rule-based templates

        Returns list of dicts with keys:
          question, type, category, difficulty, evaluation_points
        """
        import re as _re

        # ── attempt Groq API call (with 1 retry) ──────────────
        _REQUIRED_KEYS = {"question", "type", "category", "difficulty", "evaluation_points"}
        _VALID_DIFFICULTIES = {"easy", "medium", "hard"}
        _MAX_ATTEMPTS = 2

        if self._groq_client is not None:
            prompt = self._build_prompt(jd, candidate)
            for attempt in range(1, _MAX_ATTEMPTS + 1):
                try:
                    chat_completion = self._groq_client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": (
                                "You are a senior enterprise technical interviewer designing structured hiring evaluations. "
                                "Your output will be consumed by an automated HR system. "
                                "If your JSON is invalid, incomplete, or incorrectly structured, it will be rejected automatically. "
                                "You must strictly follow all instructions. Return ONLY valid JSON."
                            )},
                            {"role": "user", "content": prompt},
                        ],
                        model="llama-3.3-70b-versatile",
                        temperature=0.3,
                        max_tokens=3000,
                        top_p=0.9,
                    )
                    raw = chat_completion.choices[0].message.content.strip()

                    # Strip markdown fences if present
                    cleaned = _re.sub(r'^```(?:json)?\s*', '', raw)
                    cleaned = _re.sub(r'\s*```$', '', cleaned).strip()

                    parsed = json.loads(cleaned)
                    questions = parsed.get("questions", parsed) if isinstance(parsed, dict) else parsed

                    # ── Validation gate (strict) ──────────────
                    valid = True
                    reason = ""

                    # 1. Structure check
                    if not isinstance(questions, list) or len(questions) != 8:
                        valid, reason = False, "Expected list of 8 questions"
                    elif not all(isinstance(q, dict) for q in questions):
                        valid, reason = False, "All items must be dicts"
                    elif not all(_REQUIRED_KEYS.issubset(q.keys()) for q in questions):
                        valid, reason = False, "Missing required keys"
                    elif not all(
                        isinstance(q.get("evaluation_points"), list)
                        and len(q["evaluation_points"]) == 4
                        for q in questions
                    ):
                        valid, reason = False, "evaluation_points must be list of 4"

                    # 2. Difficulty enforcement
                    if valid:
                        for q in questions:
                            if q.get("difficulty", "").lower() not in _VALID_DIFFICULTIES:
                                q["difficulty"] = "medium"  # auto-correct invalid difficulty

                    # 3. Duplicate detection (>60% identical → reject)
                    if valid:
                        q_texts = [q.get("question", "").strip().lower() for q in questions]
                        unique_count = len(set(q_texts))
                        if unique_count < len(q_texts) * 0.6:
                            valid, reason = False, f"Too many duplicate questions ({len(q_texts) - unique_count} dupes)"

                    # 4. Skill-to-question verification
                    if valid and jd.required_skills:
                        tech_questions = [
                            q for q in questions
                            if q.get("type", "").lower() == "technical"
                        ]
                        if tech_questions:
                            all_q_text = " ".join(
                                q.get("question", "").lower() for q in tech_questions
                            )
                            matched_skills = sum(
                                1 for s in jd.required_skills
                                if s.lower() in all_q_text
                            )
                            skill_coverage = matched_skills / len(jd.required_skills)
                            if skill_coverage < 0.3:
                                logger.info(
                                    "Groq attempt %d: low skill coverage %.0f%% — regenerating",
                                    attempt, skill_coverage * 100,
                                )
                                if attempt < _MAX_ATTEMPTS:
                                    continue  # retry
                                # On last attempt, accept anyway (structure is valid)

                    if valid:
                        self._last_generated = questions
                        logger.info(
                            "Groq: generated %d questions — validation passed (attempt %d)",
                            len(questions), attempt,
                        )
                        return questions

                    logger.warning(
                        "Groq attempt %d/%d failed validation: %s",
                        attempt, _MAX_ATTEMPTS, reason,
                    )

                except json.JSONDecodeError as e:
                    logger.warning("Groq attempt %d JSON parse error (%s)", attempt, e)
                except Exception as e:
                    logger.warning("Groq attempt %d API call failed (%s)", attempt, e)

        # ── fallback: deterministic rule-based ────────────────
        questions = self._fallback_questions(jd, candidate)
        self._last_generated = questions
        return questions


class PolicyLeaveManager(LeaveManager):
    """Leave management with policy enforcement."""

    @staticmethod
    def _count_working_days(start: datetime, end: datetime) -> int:
        """Count business days (Mon-Fri) between start and end inclusive."""
        count = 0
        current = start
        one_day = timedelta(days=1)
        while current <= end:
            if current.weekday() < 5:  # 0=Mon … 4=Fri
                count += 1
            current += one_day
        return max(count, 1)  # at least 1 day

    def process_leave_request(self, request: LeaveRequest, policy: LeavePolicy,
                              current_balance: int) -> Dict:
        # FIX 4: guard against malformed / None date fields
        try:
            _ = request.start_date - request.end_date   # triggers TypeError if either is None
        except (TypeError, AttributeError) as exc:
            return {
                "approved": False,
                "reason": f"Invalid leave dates: {exc}",
                "violations": ["Malformed or missing start_date / end_date"],
                "days_requested": 0,
                "remaining_balance": current_balance,
                "ml_used": False,
                "ml_confidence": None,
                "risk_score": None,
            }
        # FIX 4: explicit chronological guard
        if request.start_date > request.end_date:
            return {
                "approved": False,
                "reason": "Start date cannot be after end date",
                "violations": ["start_date > end_date"],
                "days_requested": 0,
                "remaining_balance": current_balance,
                "ml_used": False,
                "ml_confidence": None,
                "risk_score": None,
            }

        violations = []
        days_requested = self._count_working_days(request.start_date, request.end_date)
        ml_confidence = None
        risk_score = None
        ml_used = False

        # ── ML INTEGRATION: PREDICT APPROVAL & RISK ──
        # FIX 7: ML is advisory-only — rule violations always checked first;
        #        RandomForestClassifier must use random_state=42 for determinism.
        try:
            import os, joblib, pandas as pd, numpy as np
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "leave_approval_model.pkl")
            feat_path = os.path.join(base_dir, "feature_columns.pkl")
            enc_path = os.path.join(base_dir, "label_encoders.pkl")

            if os.path.exists(model_path) and os.path.exists(enc_path) and os.path.exists(feat_path):
                model = joblib.load(model_path)
                # FIX 7: patch random_state if the loaded model lacks it
                if hasattr(model, 'random_state') and model.random_state is None:
                    model.random_state = 42
                encoders = joblib.load(enc_path)
                features = joblib.load(feat_path)

                now = datetime.now(request.start_date.tzinfo) if request.start_date.tzinfo else datetime.now()
                notice_days = (request.start_date - now).days

                # Transform categorical variables safely
                leave_type = getattr(request, 'leave_type', 'casual').strip().lower()
                dept = getattr(request, 'department', 'engineering').strip().lower()
                pos = getattr(request, 'position', 'developer').strip().lower()

                le_lt = encoders.get("leave_type")
                le_dept = encoders.get("department")
                le_pos = encoders.get("position")

                lt_enc = le_lt.transform([leave_type])[0] if le_lt and leave_type in le_lt.classes_ else 0
                dept_enc = le_dept.transform([dept])[0] if le_dept and dept in le_dept.classes_ else 0
                pos_enc = le_pos.transform([pos])[0] if le_pos and pos in le_pos.classes_ else 0

                # Build feature dictionary
                data_point = {
                    "leave_duration_days": days_requested,
                    "notice_period_days": notice_days,
                    "days_taken": 0, # Approximation for single request
                    "total_leave_entitlement": current_balance + days_requested,
                    "leave_taken_so_far": 0,
                    "remaining_leaves": current_balance,
                    "leave_days": days_requested,
                    "leave_type_encoded": lt_enc,
                    "department_encoded": dept_enc,
                    "position_encoded": pos_enc,
                    "month_of_year": request.start_date.month,
                    "day_of_week": request.start_date.weekday(),
                    "leave_balance_ratio": days_requested / max(current_balance + days_requested, 1),
                    "past_leaves_ratio": 0.0,
                }

                X_infer = pd.DataFrame([data_point])[features].fillna(0)
                prob = model.predict_proba(X_infer)[0][1] # Probability of "Approved"

                ml_confidence = float(prob)
                risk_score = float((1.0 - prob) * 100)
                ml_used = True

                # FIX 7: ML is advisory — only adds a denial, never overrides rule violations
                # (approved is set below after rule checks; ML can only add a violation)
                if prob <= 0.70:
                    violations.append(f"ML Model rejected request (Confidence: {prob*100:.1f}%)")

        except Exception as e:
            logger.error(f"ML integration failed, falling back to rules: {e}")
            ml_used = False

        # ── RULE-BASED LOGIC (always authoritative) ──
        days_requested = self._count_working_days(request.start_date, request.end_date)

        # Check balance
        if days_requested > current_balance:
            violations.append(f"Insufficient balance: requested {days_requested}, available {current_balance}")

        # Check max consecutive days
        if days_requested > policy.max_consecutive_days:
            violations.append(f"Exceeds max consecutive days ({policy.max_consecutive_days})")

        # Check notice period (timezone-safe)
        now = datetime.now(request.start_date.tzinfo) if request.start_date.tzinfo else datetime.now()
        notice_days = (request.start_date - now).days
        if notice_days < policy.min_notice_days:
            violations.append(f"Insufficient notice: {notice_days} days (min: {policy.min_notice_days})")

        # Check documentation requirement
        if policy.requires_document and not request.reason:
            violations.append("Medical certificate/documentation required")

        # FIX 7: rule-based result is always authoritative
        approved = len(violations) == 0

        return {
            "approved": approved,
            "reason": "Approved" if approved else "Denied due to policy violations or high risk",
            "violations": violations,
            "days_requested": days_requested,
            "remaining_balance": max(0, current_balance - days_requested) if approved else current_balance,
            "ml_used": ml_used,
            "ml_confidence": ml_confidence,
            "risk_score": risk_score,
        }


class RuleBasedEscalation(EscalationHandler):
    """Intelligent severity-aware escalation handler.

    Severity rules (deterministic, no ML):
      HIGH   — any high-severity keyword OR harassment + emotional distress combo
      MEDIUM — 2+ medium keywords OR 1 medium keyword + urgency word
      LOW    — any low-severity keyword
    """

    ESCALATION_PATTERNS = {
        "high": [
            "grievance", "harassment", "sexual harassment",
            "discrimination", "termination", "legal",
            "legal action", "lawsuit", "ethics breach",
            "code of conduct", "retaliation", "physical threat",
            "hostile work environment", "whistleblower",
        ],
        "medium": [
            "compensation", "salary revision", "policy exception",
            "transfer", "unfair treatment", "bias",
            "demotion", "wrongful", "victimised", "victimized",
            "formal complaint", "violation",
        ],
        "low": [
            "general complaint", "feedback", "suggestion",
            "query", "clarification", "leave policy",
        ],
    }

    URGENCY_WORDS = [
        # Immediate urgency
        "urgent", "immediately", "asap", "right away",
        "as soon as possible", "priority", "critical",
        "emergency", "escalate", "action required",
        # Safety / threat indicators
        "unsafe", "threat", "threatened", "intimidated",
        "harassed", "harassment", "abuse", "abusive",
        "bullying", "bullied", "hostile",
        # Emotional distress
        "mental stress", "mental health", "anxiety",
        "depressed", "distressed", "trauma",
        "toxic", "toxic manager", "toxic environment",
        # Workplace conflict severity
        "unfair treatment", "discrimination",
        "bias", "retaliation", "victimised",
        "grievance", "complaint", "formal complaint",
        # Legal / compliance risk
        "legal action", "lawsuit", "violation",
        "code of conduct", "ethics breach",
    ]

    # Harassment + distress combo triggers HIGH
    _HARASSMENT_PHRASES = ["harassment", "harassed", "bullying", "bullied", "hostile"]
    _DISTRESS_PHRASES = ["mental stress", "mental health", "anxiety", "depressed",
                         "distressed", "trauma", "toxic"]

    def should_escalate(self, query: str, context: Dict) -> tuple:
        """Determine if query requires escalation with intelligent severity."""
        query_lower = query.lower()

        # ── collect matched keywords per priority ──
        high_matches = [kw for kw in self.ESCALATION_PATTERNS["high"] if kw in query_lower]
        medium_matches = [kw for kw in self.ESCALATION_PATTERNS["medium"] if kw in query_lower]
        low_matches = [kw for kw in self.ESCALATION_PATTERNS["low"] if kw in query_lower]
        urgency_matches = [w for w in self.URGENCY_WORDS if w in query_lower]

        # ── compound severity detection ──
        # HIGH: any high keyword
        if high_matches:
            return (True, f"High-severity keyword matched: {high_matches[0]}", "high")

        # HIGH: harassment + emotional distress combo
        has_harassment = any(p in query_lower for p in self._HARASSMENT_PHRASES)
        has_distress = any(p in query_lower for p in self._DISTRESS_PHRASES)
        if has_harassment and has_distress:
            return (True, "Harassment combined with emotional distress detected", "high")

        # MEDIUM: 2+ medium keywords
        if len(medium_matches) >= 2:
            return (True, f"Multiple medium-severity keywords: {', '.join(medium_matches[:3])}", "medium")

        # MEDIUM: 1 medium keyword + urgency word
        if len(medium_matches) >= 1 and urgency_matches:
            return (True, f"Medium keyword '{medium_matches[0]}' with urgency '{urgency_matches[0]}'", "medium")

        # LOW: any low keyword
        if low_matches:
            return (True, f"Low-severity keyword matched: {low_matches[0]}", "low")

        return (False, "No escalation needed", "none")


# ─────────────────────────────────────────────────────
# MAIN HR AGENT
# ─────────────────────────────────────────────────────
class HRAgent:
    """Main HR Agent orchestrator with enterprise FSM pipeline control."""

    # Terminal states — no further transitions allowed (enum-driven)
    _TERMINAL_STATES = {PipelineStatus.SELECTED, PipelineStatus.REJECTED}

    def __init__(self):
        self.screener = RuleBasedResumeScreener()
        self.scheduler = BasicInterviewScheduler()
        self.questionnaire = LLMQuestionnaireGenerator()
        self.leave_mgr = PolicyLeaveManager()
        self.escalation = RuleBasedEscalation()
        self.pipeline: Dict[str, Candidate] = {}   # candidate_id -> Candidate
        self.audit_trail: List[Dict] = []           # FSM transition log
        self._booked_slots: Dict[str, Dict] = {}     # candidate_id -> full slot dict
        self._processed_leave_requests: List[Dict] = []
        self._escalation_log: List[Dict] = []
        self.conflicts_log: List[Dict] = []
        # FIX 1: leave overlap detection dict  employee_id -> list of (start, end) tuples
        self._approved_leaves: Dict[str, List[tuple]] = {}
        
        # Leave dataset (auto-balance source of truth)
        try:
            self._leave_df = pd.read_excel("employee leave tracking data.xlsx")
        except Exception:
            self._leave_df = pd.DataFrame()
            
        self._max_leave_per_day = 3  # capacity rule
        
        logger.info("HR Agent initialized")

    def _calculate_employee_balance(self, employee_id: str, policy: LeavePolicy) -> int:
        """Calculate remaining leave balance from the leave dataset."""
        if self._leave_df.empty:
            return policy.annual_quota

        df = self._leave_df.copy()

        # Try to resolve employee_id to a name for matching (e.g. C001 -> 'Priya Sharma')
        name_to_match = employee_id
        if employee_id in self.pipeline:
            name_to_match = self.pipeline[employee_id].name

        # Column handling for 'employee_name' instead of 'employee_id'
        target_col = "employee_name" if "employee_name" in df.columns else "employee_id"
        if target_col not in df.columns:
            return policy.annual_quota

        # Only count approved leaves for this employee
        df = df[
            (df[target_col].astype(str).str.lower() == str(name_to_match).lower()) &
            (df["leave_status"].astype(str).str.lower() == "approved")
        ]

        if df.empty:
            return policy.annual_quota

        days_taken = 0
        for _, row in df.iterrows():
            try:
                start = pd.to_datetime(row["start_date"])
                end   = pd.to_datetime(row["end_date"])
                days_taken += (end - start).days + 1
            except (ValueError, TypeError):
                continue

        remaining = max(policy.annual_quota - days_taken, 0)
        return int(remaining)

    # ── FSM helpers ────────────────────────────────────────────
    @staticmethod
    def _resolve_status(value) -> PipelineStatus:
        """Convert a string or PipelineStatus to PipelineStatus enum."""
        if isinstance(value, PipelineStatus):
            return value
        try:
            return PipelineStatus(value)
        except ValueError:
            raise ValueError(f"Invalid pipeline status: '{value}'")

    def _record_transition(self, candidate_id: str, from_status: PipelineStatus,
                           to_status: PipelineStatus,
                           reason: TransitionReason = TransitionReason.MANUAL_TRANSITION) -> None:
        """Append an immutable audit record for every state change."""
        self.audit_trail.append({
            "candidate_id": candidate_id,
            "from": from_status.value,
            "to": to_status.value,
            "timestamp": datetime.now().isoformat(),
            "reason": reason.value if isinstance(reason, TransitionReason) else str(reason),
        })

    # ── core pipeline methods ─────────────────────────────────
    def screen_resumes(self, candidates: List[Candidate], jd: JobDescription) -> List[Candidate]:
        """Screen and rank candidates. Entry point for resume screening evaluation.

        FSM flow: each candidate moves  APPLIED → PROCESSING  as soon as
        their resume enters the screening pipeline.
        """
        import re as _re_exp

        # FIX 3: deduplicate by candidate_id — warn and keep last occurrence
        seen_ids: Dict[str, int] = {}
        for idx, c in enumerate(candidates):
            if c.candidate_id in seen_ids:
                logger.warning(
                    "Duplicate candidate_id '%s' detected — keeping last occurrence (index %d, overwriting index %d).",
                    c.candidate_id, idx, seen_ids[c.candidate_id],
                )
            seen_ids[c.candidate_id] = idx
        candidates = [candidates[idx] for idx in sorted(seen_ids.values())]  # FIX 3: deduplicated list

        for c in candidates:
            # FIX 8: guard against empty/whitespace-only resume_text
            if not (c.resume_text or "").strip():
                logger.warning("Candidate '%s' has empty resume_text — assigning match_score=0.0.", c.candidate_id)
                c.resume_text = "(no resume provided)"  # FIX 8: placeholder prevents TF-IDF NaN
                c.match_score = 0.0

            c.skills = self.screener.extract_skills(c.resume_text)

            # ── auto-extract experience_years from resume_text ──
            if c.experience_years == 0.0 and c.resume_text:
                # Matches: "5 years", "5+ years", "over 6 years", "6 yrs",
                #          "5.5 years", "10+ yrs of experience"
                exp_patterns = _re_exp.findall(
                    r'(?:over\s+|more\s+than\s+)?(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)',
                    c.resume_text.lower(),
                )
                if exp_patterns:
                    c.experience_years = max(float(v) for v in exp_patterns)

        ranked = self.screener.rank_candidates(candidates, jd)

        for c in ranked:
            # Register in pipeline first (must exist before FSM call)
            self.pipeline[c.candidate_id] = c
            # Transition: APPLIED → PROCESSING (via FSM, not direct mutation)
            self.update_pipeline_status(
                c.candidate_id, "processing",
                reason=TransitionReason.SCREENING_STARTED
            )
        logger.info("Screened %d candidates successfully", len(ranked))
        return ranked

    def shortlist_and_schedule(self, ranked_candidates: List[Candidate],
                                top_n: int, slots: List[InterviewSlot]) -> List[Dict]:
        """Shortlist top N and schedule interviews via FSM."""
        results = []
        for candidate in ranked_candidates[:top_n]:
            # Transition: PROCESSING -> SHORTLISTED (through FSM)
            res = self.update_pipeline_status(
                candidate.candidate_id, "shortlisted",
                reason=TransitionReason.AUTO_SHORTLISTED
            )
            if "error" in res:
                results.append({"candidate": candidate.name, "slot": None,
                                "status": f"shortlist_failed: {res['error']}"})
                continue

            slot = self.scheduler.schedule_interview(candidate, slots)
            if slot:
                # Record the booked slot so FSM precondition is satisfied
                slot_dict = asdict(slot)
                slot_dict["start_time"] = slot.start_time.isoformat() if hasattr(slot.start_time, 'isoformat') else str(slot.start_time)
                slot_dict["end_time"] = slot.end_time.isoformat() if hasattr(slot.end_time, 'isoformat') else str(slot.end_time)
                self._booked_slots[candidate.candidate_id] = slot_dict
                res2 = self.update_pipeline_status(
                    candidate.candidate_id, "interview_scheduled",
                    reason=TransitionReason.SLOT_BOOKED
                )
                if "error" in res2:
                    results.append({"candidate": candidate.name, "slot": slot_dict,
                                    "status": f"schedule_failed: {res2['error']}"})
                else:
                    results.append({"candidate": candidate.name, "slot": slot_dict,
                                    "status": "scheduled"})
            else:
                results.append({"candidate": candidate.name, "slot": None,
                                "status": "no_slot_available"})
        return results

    def schedule_candidates(
        self,
        ranked_candidates: List[Candidate],
        *,
        top_n: Optional[int] = None,
    ) -> List[Dict]:
        """Schedule interviews using realistic internal sample datasets.

        Orchestrates the full flow:
          1. Load sample availability (No Excel needed)
          2. Match candidates to interviewers (type + expertise + time)
          3. Enforce business hours, duration, buffer
          4. Book slots via FSM: PROCESSING → SHORTLISTED → INTERVIEW_SCHEDULED
          5. Log conflicts
          6. Auto-generate availability for missing candidates

        Args:
            ranked_candidates: Output of screen_resumes()
            top_n:             Optional limit on candidates to schedule
        """
        # ── load datasets ─────────────────────────────────────
        i_df, c_df = self.scheduler.load_sample_availability()

        results: List[Dict] = []
        slot_counter = [0]  # mutable counter shared with scheduler
        candidates_to_schedule = ranked_candidates[:top_n] if top_n else ranked_candidates

        # Sort by match_score (desc) then coverage (desc) for deterministic priority
        candidates_to_schedule = sorted(
            candidates_to_schedule,
            key=lambda c: (c.coverage, c.match_score),
            reverse=True,
        )

        scheduled_count = 0
        conflict_count  = 0

        # ── RULE 1: Single Candidate Auto-Schedule ────────────
        if len(candidates_to_schedule) == 1:
            candidate = candidates_to_schedule[0]
            cid = candidate.candidate_id

            # GUARD: skip auto-schedule if candidate already has a booking
            if cid in self._booked_slots:
                logger.info(
                    "Candidate '%s' already has a booked slot — skipping RULE 1 auto-schedule.", cid
                )
                return [{
                    "candidate":    candidate.name,
                    "candidate_id": cid,
                    "slot":         self._booked_slots[cid],
                    "status":       "already_scheduled",
                }]

            from zoneinfo import ZoneInfo
            from datetime import datetime, timedelta
            DEFAULT_TZ = ZoneInfo("Asia/Kolkata")
            tomorrow = datetime.now(DEFAULT_TZ) + timedelta(days=1)
            start_t = tomorrow.replace(hour=10, minute=0, second=0, microsecond=0)
            end_t = start_t + timedelta(minutes=60)
            
            # CHECK AVAILABILITY FOR INT001
            int001_row = i_df[i_df["Interviewer_ID"] == "INT001"]
            is_avail = True
            if not int001_row.empty:
                is_avail = int001_row.iloc[0].get("Is_Available", True)

            slot = InterviewSlot(
                slot_id=f"SLOT-{slot_counter[0]:03d}",
                interviewer_id="INT001",
                start_time=start_t,
                end_time=end_t,
                is_available=is_avail,
            )

            if is_avail:
                # GUARD: only shortlist if not already in a more advanced state
                current = self.pipeline.get(cid)
                if current and current.status not in (
                    PipelineStatus.SHORTLISTED,
                    PipelineStatus.INTERVIEW_SCHEDULED,
                    PipelineStatus.INTERVIEWED,
                    PipelineStatus.SELECTED,
                    PipelineStatus.REJECTED,
                ):
                    self.update_pipeline_status(cid, "shortlisted", reason=TransitionReason.AUTO_SHORTLISTED)
                slot_dict = asdict(slot)
                slot_dict["start_time"] = slot.start_time.isoformat()
                slot_dict["end_time"] = slot.end_time.isoformat()
                self._booked_slots[cid] = slot_dict
                self.update_pipeline_status(cid, "interview_scheduled", reason=TransitionReason.SLOT_BOOKED)
                
                return [{
                    "candidate": candidate.name,
                    "candidate_id": cid,
                    "slot": slot_dict,
                    "status": "scheduled",
                }]
            else:
                # Case 2: Slot not available
                slot_dict = asdict(slot)
                slot_dict["start_time"] = slot.start_time.isoformat()
                slot_dict["end_time"] = slot.end_time.isoformat()
                return [{
                    "candidate": candidate.name,
                    "candidate_id": cid,
                    "slot": slot_dict,
                    "status": "not_scheduled",
                }]

        for candidate in candidates_to_schedule:
            cid = candidate.candidate_id

            # GUARD: skip if already booked (e.g. via request_manual_time or RULE 1)
            if cid in self._booked_slots:
                logger.info("Candidate '%s' already has a booked slot — skipping.", cid)
                results.append({
                    "candidate":    candidate.name,
                    "candidate_id": cid,
                    "slot":         self._booked_slots[cid],
                    "status":       "already_scheduled",
                })
                continue
            c_rows = c_df[c_df["Candidate_ID"] == cid]
            is_auto_gen = False
            if c_rows.empty:
                from datetime import datetime, timedelta
                from zoneinfo import ZoneInfo
                import pandas as pd

                DEFAULT_TZ = ZoneInfo("Asia/Kolkata")

                # Generate default "Tomorrow at 11:00-16:00"
                tomorrow = datetime.now(DEFAULT_TZ) + timedelta(days=1)
                start_time = tomorrow.replace(hour=11, minute=0, second=0, microsecond=0)
                end_time = tomorrow.replace(hour=16, minute=0, second=0, microsecond=0)

                auto_row = {
                    "Candidate_ID": cid,
                    "Preferred_Type": "technical",
                    "Duration_Minutes": 60,
                    "_start": start_time,
                    "_end": end_time,
                    "Timezone": "Asia/Kolkata",
                }
                logger.info("Auto-generated availability for candidate %s", cid)
                candidate_rows = [auto_row]
                is_auto_gen = True
            else:
                candidate_rows = c_rows.to_dict("records")

            # ── attempt scheduling ────────────────────────────
            slot, failure_reason = self.scheduler.schedule_from_availability(
                candidate,
                candidate_rows,
                i_df,
                slot_counter=slot_counter,
            )

            # ── RULE 2: Fallback Strategy for No Overlap ──────
            is_fallback = False
            if slot is None and ("No overlapping time window" in failure_reason or "Duration insufficient" in failure_reason):
                logger.info("Applying fallback scheduling for candidate %s", cid)
                slot = self.scheduler.find_fallback_slot(i_df, 60, slot_counter)
                if slot:
                    is_fallback = True

            if slot is None:
                # ── RULE 3: Still No Slot Available ───────────
                reason = "Manual time confirmation required" if not failure_reason else f"Fallback failed: {failure_reason}"
                self.conflicts_log.append({
                    "candidate_id": cid,
                    "reason": reason,
                })
                conflict_count += 1
                results.append({
                    "candidate": candidate.name,
                    "candidate_id": cid,
                    "slot": None,
                    "status": "manual_time_required",
                })
                continue

            # ── FSM transitions and Booking: Only if slot is available
            slot_dict = asdict(slot)
            slot_dict["start_time"] = slot.start_time.isoformat() if hasattr(slot.start_time, 'isoformat') else str(slot.start_time)
            slot_dict["end_time"] = slot.end_time.isoformat() if hasattr(slot.end_time, 'isoformat') else str(slot.end_time)

            if slot.is_available:
                # Book slot
                self._booked_slots[cid] = slot_dict

                # GUARD: only shortlist if not already in a more advanced state
                current = self.pipeline.get(cid)
                if current and current.status not in (
                    PipelineStatus.SHORTLISTED,
                    PipelineStatus.INTERVIEW_SCHEDULED,
                    PipelineStatus.INTERVIEWED,
                    PipelineStatus.SELECTED,
                    PipelineStatus.REJECTED,
                ):
                    self.update_pipeline_status(cid, "shortlisted", reason=TransitionReason.AUTO_SHORTLISTED)

                self.update_pipeline_status(cid, "interview_scheduled", reason=TransitionReason.SLOT_BOOKED)
                
                scheduled_count += 1
                results.append({
                    "candidate": candidate.name,
                    "candidate_id": cid,
                    "slot": slot_dict,
                    "status": "scheduled_via_score_priority" if is_fallback else "scheduled",
                })
            else:
                # Case 2: Slot not available
                results.append({
                    "candidate": candidate.name,
                    "candidate_id": cid,
                    "slot": slot_dict,
                    "status": "not_scheduled",
                })

        # ── STEP 14: summary logging with interviewer distribution ──
        logger.info(
            "Scheduling complete: %d scheduled, %d conflicts out of %d candidates",
            scheduled_count, conflict_count, len(candidates_to_schedule),
        )
        # Interviewer booking distribution
        if self.scheduler._interviewer_bookings:
            dist = {
                iid: len(bookings)
                for iid, bookings in sorted(self.scheduler._interviewer_bookings.items())
            }
            logger.info("Interviewer booking distribution: %s", dist)
        return results

    def request_manual_time(self, candidate_id: str, preferred_time: datetime) -> Dict:
        """RULE 4: Manually book an interview slot at a specific time.

        Validates:
          - candidate_id exists in pipeline
          - preferred_time is timezone-aware (Asia/Kolkata)
          - preferred_time is not in the past
          - preferred_time is within business hours (10:00–17:30)

        Updates FSM and stores in _booked_slots.
        Returns full slot dict with ISO-format start/end times.
        """
        from zoneinfo import ZoneInfo
        KOLKATA_TZ = ZoneInfo("Asia/Kolkata")

        if candidate_id not in self.pipeline:
            return {"error": "Candidate not found in pipeline"}

        # Ensure timezone-aware datetime using ZoneInfo
        if preferred_time.tzinfo is None:
            preferred_time = preferred_time.replace(tzinfo=KOLKATA_TZ)
        else:
            preferred_time = preferred_time.astimezone(KOLKATA_TZ)

        # Validate: not in the past
        now = datetime.now(KOLKATA_TZ)
        if preferred_time < now:
            return {"error": "Cannot schedule an interview in the past. Please choose a future date and time."}

        # Validate: within business hours (10:00 – 17:30)
        bh_start = preferred_time.replace(hour=10, minute=0, second=0, microsecond=0)
        bh_end   = preferred_time.replace(hour=17, minute=30, second=0, microsecond=0)
        if not (bh_start <= preferred_time <= bh_end):
            return {"error": "Interview time must be within business hours (10:00 AM – 5:30 PM IST)."}

        candidate = self.pipeline[candidate_id]

        # Assign first available interviewer (INT001 default)
        i_id = "INT001"
        duration = 60
        end_time = preferred_time + timedelta(minutes=duration)

        slot = InterviewSlot(
            slot_id=f"MANUAL-{candidate_id}",
            interviewer_id=i_id,
            start_time=preferred_time,
            end_time=end_time,
        )

        # Force FSM transitions
        self.update_pipeline_status(candidate_id, "shortlisted", reason=TransitionReason.AUTO_SHORTLISTED)

        slot_dict = asdict(slot)
        slot_dict["start_time"] = preferred_time.isoformat()
        slot_dict["end_time"]   = end_time.isoformat()
        self._booked_slots[candidate_id] = slot_dict

        res = self.update_pipeline_status(candidate_id, "interview_scheduled", reason=TransitionReason.SLOT_BOOKED)

        if "error" in res:
            return {"error": f"FSM Transition failed: {res['error']}"}

        return {
            "candidate":    candidate.name,
            "candidate_id": candidate_id,
            "slot":         slot_dict,
            "status":       "manually_scheduled",
        }

    def generate_interview_questions(self, jd: JobDescription,
                                     candidate: Optional[Candidate] = None) -> List[Dict]:
        """Generate interview questionnaire for a role (optionally personalised)."""
        return self.questionnaire.generate_questions(jd, candidate)

    def process_leave(self, request: LeaveRequest, policy: LeavePolicy) -> Dict:
        """Process a leave request with policy checks."""
        # Auto-calculate balance from dataset
        balance = self._calculate_employee_balance(request.employee_id, policy)
        
        # FIX 1: check for overlapping approved leaves before processing
        emp_id = request.employee_id
        try:
            req_start = request.start_date
            req_end   = request.end_date
            existing_leaves = self._approved_leaves.get(emp_id, [])
            for (ex_start, ex_end) in existing_leaves:
                # Overlap: new interval starts before existing ends AND new interval ends after existing starts
                if req_start <= ex_end and req_end >= ex_start:
                    overlap_result = {
                        "approved": False,
                        "reason": "Denied due to policy violations or high risk",
                        "violations": [
                            f"Overlapping approved leave already exists for {emp_id}: "
                            f"{ex_start.date()} to {ex_end.date()}"
                        ],
                        "days_requested": 0,
                        "remaining_balance": balance,
                        "ml_used": False,
                        "ml_confidence": None,
                        "risk_score": None,
                    }
                    self._processed_leave_requests.append({
                        "request_id":        request.request_id,
                        "employee_id":       emp_id,
                        "leave_type":        getattr(request, 'leave_type', 'general'),
                        "start_date":        req_start.isoformat() if hasattr(req_start, 'isoformat') else str(req_start),
                        "end_date":          req_end.isoformat()   if hasattr(req_end,   'isoformat') else str(req_end),
                        "days_requested":    0,
                        "approved":          False,
                        "reason":            overlap_result["reason"],
                        "violations":        overlap_result["violations"],
                        "remaining_balance": balance,
                    })
                    return overlap_result
                    
            # ── Team Capacity Check ──
            from datetime import timedelta
            date_cursor = req_start
            while date_cursor <= req_end:
                count_on_day = 0

                # Count approved leaves in dataset (using employee_name or employee_id)
                if not self._leave_df.empty:
                    for _, row in self._leave_df.iterrows():
                        status_col = "leave_status" if "leave_status" in row else "status"
                        if str(row.get(status_col, "")).lower() != "approved":
                            continue
                        try:
                            start = pd.to_datetime(row["start_date"])
                            end   = pd.to_datetime(row["end_date"])
                            if start <= date_cursor <= end:
                                count_on_day += 1
                        except (ValueError, TypeError):
                            continue

                # Count newly approved leaves in this session
                for e_id, leaves in self._approved_leaves.items():
                    for (ex_start, ex_end) in leaves:
                        if ex_start <= date_cursor <= ex_end:
                            count_on_day += 1

                if count_on_day >= self._max_leave_per_day:
                    capacity_result = {
                        "approved": False,
                        "reason": "Denied due to team leave capacity limit",
                        "violations": [
                            f"Maximum employees on leave reached for {date_cursor.date()}"
                        ],
                        "days_requested": 0,
                        "remaining_balance": balance,
                        "ml_used": False,
                        "ml_confidence": None,
                        "risk_score": None,
                    }
                    self._processed_leave_requests.append({
                        "request_id":        request.request_id,
                        "employee_id":       emp_id,
                        "leave_type":        getattr(request, 'leave_type', 'general'),
                        "start_date":        req_start.isoformat() if hasattr(req_start, 'isoformat') else str(req_start),
                        "end_date":          req_end.isoformat()   if hasattr(req_end,   'isoformat') else str(req_end),
                        "days_requested":    0,
                        "approved":          False,
                        "reason":            capacity_result["reason"],
                        "violations":        capacity_result["violations"],
                        "remaining_balance": balance,
                    })
                    return capacity_result
                date_cursor += timedelta(days=1)

        except (TypeError, AttributeError):
            pass  # malformed dates handled downstream by process_leave_request

        result = self.leave_mgr.process_leave_request(request, policy, balance)

        # FIX 1: register newly approved leave so future overlap checks see it
        if result["approved"]:
            self._approved_leaves.setdefault(emp_id, []).append(
                (request.start_date, request.end_date)
            )

        self._processed_leave_requests.append({
            "request_id": request.request_id,
            "employee_id": request.employee_id,
            "leave_type": getattr(request, 'leave_type', 'general'),
            "start_date": request.start_date.isoformat() if hasattr(request.start_date, 'isoformat') else str(request.start_date),
            "end_date": request.end_date.isoformat() if hasattr(request.end_date, 'isoformat') else str(request.end_date),
            "days_requested": result["days_requested"],
            "approved": result["approved"],
            "reason": result["reason"],
            "violations": result["violations"],
            "remaining_balance": result["remaining_balance"],
            "ml_confidence": result.get("ml_confidence"),
            "risk_score": result.get("risk_score"),
            "ml_used": result.get("ml_used", False),
        })
        return result

    def update_pipeline_status(self, candidate_id: str, new_status,
                               reason: TransitionReason = TransitionReason.MANUAL_TRANSITION) -> Dict:
        """
        Enterprise FSM — the *only* method that mutates candidate.status.

        Accepts new_status as string or PipelineStatus enum.

        Enforces:
        1. Candidate existence check
        2. Valid enum conversion (rejects unknown status strings)
        3. Terminal-state lock (SELECTED / REJECTED are final)
        4. Idempotency (same-state transition blocked)
        5. Transition whitelist via PipelineStatus.valid_transitions()
        6. Precondition: INTERVIEW_SCHEDULED requires a booked slot
        7. Audit trail for every successful transition
        """
        # ── existence ──
        if candidate_id not in self.pipeline:
            return {"error": f"Candidate {candidate_id} not found"}

        # ── resolve incoming status to enum ──
        try:
            target = self._resolve_status(new_status)
        except ValueError as e:
            return {"error": str(e)}

        candidate = self.pipeline[candidate_id]
        current   = candidate.status                  # already a PipelineStatus enum
        valid_map = PipelineStatus.valid_transitions()

        # ── terminal-state lock ──
        if current in self._TERMINAL_STATES:
            return {"error": f"Terminal state '{current.value}' — no further transitions allowed"}

        # ── idempotency ──
        if target == current:
            return {"error": f"Already in state '{current.value}' — idempotent rejection"}

        # ── transition whitelist ──
        if target not in valid_map.get(current, []):
            return {"error": f"Invalid transition: {current.value} → {target.value}"}

        # ── precondition: INTERVIEW_SCHEDULED needs booked slot ──
        if target == PipelineStatus.INTERVIEW_SCHEDULED:
            if candidate_id not in self._booked_slots:
                return {"error": "Cannot move to 'interview_scheduled' — no slot booked"}

        # ── apply transition ──
        candidate.status = target
        self._record_transition(candidate_id, current, target, reason)
        logger.info("FSM %s: %s → %s", candidate_id, current.value, target.value)
        return {"success": True, "candidate": candidate.name,
                "from": current.value, "new_status": target.value}

    def handle_query(self, query: str, context: Dict = None) -> Dict:
        """Handle an HR query — check for escalation with structured logging."""
        ctx = context or {}
        should_esc, reason, priority = self.escalation.should_escalate(query, ctx)
        if should_esc:
            entry = {
                "employee_id": ctx.get("employee_id", "unknown"),
                "query_text": query,
                "priority": priority,
                "escalation_reason": reason,
                "timestamp": datetime.now().isoformat(),
                "context": ctx,
            }
            self._escalation_log.append(entry)
            return {
                "escalated": True,
                "priority": priority,
                "message": "Your concern has been escalated to HR for review.",
            }
        return {"escalated": False, "response": "Your query has been recorded and processed."}

    def export_results(self) -> Dict:
        """Export results in hackathon EVALUATION FORMAT.

        Produces the exact structure required by the scoring system.
        Does NOT mutate any internal state.  Deterministic output.
        """
        # ── 1. resume_screening ───────────────────────────────
        # Sort pipeline candidates by match_score descending
        sorted_candidates = sorted(
            self.pipeline.values(),
            key=lambda c: c.match_score,
            reverse=True,
        )
        ranked_candidates = [c.candidate_id for c in sorted_candidates]
        scores = [c.match_score for c in sorted_candidates]

        # ── 2. scheduling ─────────────────────────────────────
        # Enrich each booked slot with candidate_id + candidate_name (no state mutation)
        interviews_scheduled = [
            {
                "candidate_id":   cid,
                "candidate_name": self.pipeline[cid].name if cid in self.pipeline else "Unknown",
                "slot_id":        str(slot.get("slot_id", "")),
                "interviewer_id": str(slot.get("interviewer_id", "")),
                "start_time":     str(slot.get("start_time", "")),
                "end_time":       str(slot.get("end_time", "")),
            }
            for cid, slot in self._booked_slots.items()
        ]
        conflicts = getattr(self, 'conflicts_log', [])

        # ── 3. questionnaire ──────────────────────────────────
        questions = getattr(self.questionnaire, '_last_generated', []) or []

        # ── 4. pipeline ───────────────────────────────────────
        pipeline_candidates = {
            cid: c.status.value if isinstance(c.status, PipelineStatus) else str(c.status)
            for cid, c in sorted(self.pipeline.items())
        }

        # ── 5. leave_management ───────────────────────────────
        processed_requests = getattr(self, '_processed_leave_requests', [])

        # ── 6. escalations ────────────────────────────────────
        escalations = getattr(self, '_escalation_log', [])

        return {
            "team_id": CONFIG["team_id"],
            "track": "track_2_hr_agent",
            "results": {
                "resume_screening": {
                    "ranked_candidates": ranked_candidates,
                    "scores": scores,
                },
                "scheduling": {
                    "interviews_scheduled": interviews_scheduled,
                    "conflicts": list(conflicts),
                },
                "questionnaire": {
                    "questions": list(questions),
                },
                "pipeline": {
                    "candidates": pipeline_candidates,
                },
                "leave_management": {
                    "processed_requests": list(processed_requests),
                },
                "escalations": list(escalations),
            },
        }


# ─────────────────────────────────────────────────────
# SAMPLE DATA FOR TESTING
# ─────────────────────────────────────────────────────
SAMPLE_JD = JobDescription(
    job_id="JD_001",
    title="Senior Python Developer",
    description="We are looking for an experienced Python developer with expertise in "
                "building REST APIs, microservices, and cloud deployments. "
                "Experience with ML/AI pipelines is a plus.",
    required_skills=["Python", "REST APIs", "Docker", "SQL", "Git"],
    preferred_skills=["Kubernetes", "AWS", "Machine Learning", "FastAPI"],
    min_experience=4.0,
)

SAMPLE_CANDIDATES = [
    Candidate("C001", "Priya Sharma", "priya@email.com",
              "5 years Python, Django, REST APIs, Docker, AWS, PostgreSQL. Built ML pipelines."),
    Candidate("C002", "Rahul Verma", "rahul@email.com",
              "3 years Java, Spring Boot, MySQL. Learning Python and Docker."),
    Candidate("C003", "Anita Reddy", "anita@email.com",
              "6 years Python, FastAPI, Kubernetes, AWS, ML, TensorFlow. Open source contributor."),
]

SAMPLE_LEAVE_POLICY = LeavePolicy(
    leave_type="casual", annual_quota=12,
    max_consecutive_days=3, min_notice_days=2,
)


if __name__ == "__main__":
    agent = HRAgent()
    print("=" * 60)
    print("🚀  AI HR Agent — Hackathon Live Demo")
    print("=" * 60)

    # 1. Demo Resume Screening
    print("\n🔍 1. Resume Screening Demo:")
    ranked = agent.screen_resumes(SAMPLE_CANDIDATES, SAMPLE_JD)
    suitable_names = [c.name for c in ranked if c.match_score >= 0.6]
    print(f"Suitable Candidates (Score >= 60%): {json.dumps(suitable_names, indent=2)}")

    # 2. Demo Interview Scheduling (New Zero-Dependency System)
    print("\n📅 2. Interview Scheduling Demo:")
    # Now zero-dependency: no paths needed!
    sched_results = agent.schedule_candidates(ranked)
    print(f"Scheduling Results (All): {json.dumps(sched_results, indent=2)}")

    # 3. Demo Leave Management
    print("\n🏖️ 3. Leave Management Demo (w/ ML Risk Scoring):")
    leave_req = LeaveRequest(
        request_id="LR001", employee_id="EMP042",
        leave_type="casual", start_date=datetime.now() + timedelta(days=5),
        end_date=datetime.now() + timedelta(days=7), reason="Family event"
    )
    # Request 1: Standard approval (auto-calculated balance)
    result = agent.process_leave(leave_req, SAMPLE_LEAVE_POLICY)
    print("Request 1 Status:", result["approved"], "| Reason:", result["reason"])

    # Request 2-4: Testing team capacity (max 3/day)
    for i in range(2, 5):
        req_i = LeaveRequest(
            request_id=f"LR00{i}", employee_id=f"EMP{100+i}",
            leave_type="casual", start_date=leave_req.start_date,
            end_date=leave_req.end_date, reason="Team meeting overlap test"
        )
        res_i = agent.process_leave(req_i, SAMPLE_LEAVE_POLICY)
        print(f"Request {i} (Status):", res_i["approved"], "| Note:", "Rejection expected for Request 4" if i==4 else "Approval expected")

    # 4. Demo Escalation
    print("\n🚨 4. Escalation Demo:")
    esc_result = agent.handle_query("This is a harassment")
    print(json.dumps(esc_result, indent=2))

    # 5. Final Hackathon Export
    print("\n📊 5. Full Hackathon Evaluation Export (JSON):")
    export = agent.export_results()
    print(json.dumps(export, indent=2))
    print("\n" + "=" * 60)
    print("✅ Demo Complete.")
    print("=" * 60)

