"""
AI Recruit — SaaS-Grade HR Pipeline Dashboard (v3)
Glassmorphism · Animations · AMOLED Dark · Score Rings · Premium UI
Backend logic 100% unchanged.
"""
import os, sys, json, io, re, math, pandas as pd
from types import SimpleNamespace
from datetime import datetime, timezone, timedelta

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track2_hr_agent_template import (
    HRAgent, Candidate, LeaveRequest, LeavePolicy,
    SAMPLE_JD, SAMPLE_CANDIDATES, PipelineStatus,
)

# ── Resume helpers (unchanged) ──
def extract_text_from_file(f):
    raw = f.read()
    name = f.name.lower()
    if name.endswith(".pdf"):
        try:
            from PyPDF2 import PdfReader
            return "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(raw)).pages).strip()
        except ImportError:
            return raw.decode("utf-8", errors="ignore")
    elif name.endswith(".docx"):
        try:
            from docx import Document
            return "\n".join(p.text for p in Document(io.BytesIO(raw)).paragraphs).strip()
        except ImportError:
            return raw.decode("utf-8", errors="ignore")
    return raw.decode("utf-8", errors="ignore")

def extract_name(text, fallback="Unknown"):
    for line in text.split("\n")[:5]:
        line = line.strip()
        if line and len(line) < 60 and not any(k in line.lower() for k in ["resume","cv","objective","email","phone","@","http"]):
            words = line.split()
            if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w.isalpha()):
                return line
    return fallback

# ── Score ring SVG generator ──
def score_ring_svg(pct, size=80, stroke=7):
    r = (size - stroke) / 2
    circ = 2 * math.pi * r
    offset = circ * (1 - pct / 100)
    if pct >= 80: color = "#22C55E"
    elif pct >= 60: color = "#6366F1"
    elif pct >= 40: color = "#F59E0B"
    else: color = "#EF4444"
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" style="transform:rotate(-90deg)">
        <circle cx="{size/2}" cy="{size/2}" r="{r}" fill="none" stroke="rgba(255,255,255,0.08)" stroke-width="{stroke}"/>
        <circle cx="{size/2}" cy="{size/2}" r="{r}" fill="none" stroke="{color}" stroke-width="{stroke}"
                stroke-dasharray="{circ}" stroke-dashoffset="{offset}" stroke-linecap="round"
                style="transition: stroke-dashoffset 1s ease;"/>
    </svg>
    <div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
                font-weight:800;font-size:{size*0.22}px;color:{color};">{pct}%</div>
    """

# ── Page Config ──
st.set_page_config(page_title="AI Recruit", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

# ── Session State ──
if "agent" not in st.session_state:
    st.session_state.agent = HRAgent()
    st.session_state.screened = []
    st.session_state.scheduled = []
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
if "filter_status" not in st.session_state:
    st.session_state.filter_status = "All"
if "sort_by" not in st.session_state:
    st.session_state.sort_by = "Score"

agent = st.session_state.agent
dark = st.session_state.dark_mode

# ── Theme Tokens ──
if st.session_state.dark_mode:
    BG       = "#000000"; BG2      = "#050505"
    CARD     = "rgba(10,10,10,0.7)"; CARD_S   = "rgba(0,0,0,0.9)"
    BORDER   = "rgba(255,255,255,0.08)"; TEXT     = "#FFFFFF"
    TEXT2    = "#A1A1AA"; TEXT3    = "#3F3F46"
    SIDEBAR  = "#000000"; INPUT_BG = "rgba(255,255,255,0.03)"
    GLASS    = "rgba(255,255,255,0.02)"; BLUR     = "blur(20px)"
    GLOW_P   = "0 0 40px rgba(124,58,237,0.12)"; CHART_BG = "rgba(0,0,0,0)"
    GRID_C   = "rgba(255,255,255,0.03)"
else:
    BG       = "#FAFBFC"; BG2      = "#FFFFFF"
    CARD     = "rgba(255,255,255,0.7)"; CARD_S   = "rgba(255,255,255,0.9)"
    BORDER   = "rgba(0,0,0,0.06)"; TEXT     = "#09090B"
    TEXT2    = "#71717A"; TEXT3    = "#D4D4D8"
    SIDEBAR  = "#FFFFFF"; INPUT_BG = "rgba(0,0,0,0.02)"
    GLASS    = "rgba(255,255,255,0.6)"; BLUR     = "blur(20px)"
    GLOW_P   = "0 0 40px rgba(124,58,237,0.08)"; CHART_BG = "rgba(0,0,0,0)"
    GRID_C   = "rgba(0,0,0,0.04)"

ACCENT = "#7C3AED"; ACCENT2 = "#A78BFA"; GRAD = "linear-gradient(135deg, #7C3AED, #6D28D9)"

# ═══════════════════════════════════════════════════
# GLOBAL CSS — Premium Design System
# ═══════════════════════════════════════════════════
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Base ── */
*, *::before, *::after {{ box-sizing: border-box; }}
.stApp    * {{ box-sizing: border-box; }}
    body {{
        background-color: {BG};
        color: {TEXT};
        font-family: 'Inter', sans-serif;
        transition: background 0.4s ease, color 0.4s ease;
    }}
    .truncate {{
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }}
    .break-word {{
        word-break: break-all;
        word-wrap: break-word;
    }}
    h1, h2, h3, h4, h5, h6 {{ color: {TEXT} !important; font-family: 'Inter', sans-serif; }}
    p, span, label, div {{ font-family: 'Inter', sans-serif; }}
    
    /* Restore Streamlit Icons */
    .stIcon, [data-testid="stExpander"] svg, [data-testid="stExpander"] span {{
        font-family: inherit !important;
    }}

/* ── Animated BG gradient ── */
.stApp::before {{
    content: '';
    position: fixed; inset: 0; z-index: -1;
    background:
        radial-gradient(ellipse 600px 400px at 20% 20%, rgba(124,58,237,0.06), transparent),
        radial-gradient(ellipse 500px 300px at 80% 80%, rgba(99,102,241,0.04), transparent),
        radial-gradient(ellipse 400px 300px at 50% 50%, rgba(168,85,247,0.03), transparent);
    animation: bgShift 20s ease-in-out infinite alternate;
}}
@keyframes bgShift {{
    0%   {{ opacity: 0.6; }}
    50%  {{ opacity: 1; }}
    100% {{ opacity: 0.7; }}
}}

/* ── Page fade-in ── */
@keyframes fadeSlideIn {{
    from {{ opacity: 0; transform: translateY(12px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
.block-container {{ animation: fadeSlideIn 0.5s ease-out !important; }}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 5px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: {TEXT3}; border-radius: 10px; }}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {{
    background: {SIDEBAR} !important;
    border-right: 1px solid {BORDER} !important;
    transition: background 0.4s ease;
}}
section[data-testid="stSidebar"] * {{ color: {TEXT} !important; }}
section[data-testid="stSidebar"] .stRadio > div {{ gap: 2px !important; }}
section[data-testid="stSidebar"] .stRadio input {{ display: none !important; }}
section[data-testid="stSidebar"] .stRadio label {{
    padding: 11px 18px !important;
    border-radius: 12px !important;
    font-weight: 500 !important;
    font-size: 0.92rem !important;
    transition: all 0.25s cubic-bezier(0.4,0,0.2,1) !important;
    border: 1px solid transparent !important;
}}
section[data-testid="stSidebar"] .stRadio label:hover {{
    background: {CARD_S} !important;
    border-color: {BORDER} !important;
}}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:has(input:checked) {{
    background: {GRAD} !important;
    color: white !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.35), inset 0 1px 0 rgba(255,255,255,0.15);
    border-color: transparent !important;
}}
section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:has(input:checked) * {{
    color: white !important;
}}

/* ── Glass Cards ── */
.glass {{
    background: {GLASS};
    backdrop-filter: {BLUR};
    -webkit-backdrop-filter: {BLUR};
    border: 1px solid {BORDER};
    border-radius: 20px;
    padding: 28px;
    transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
    position: relative;
    overflow: hidden;
}}
.glass::before {{
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
}}
.glass:hover {{
    transform: translateY(-3px) scale(1.005);
    box-shadow: {GLOW_P};
    border-color: rgba(124,58,237,0.15);
}}

/* ── KPI Tiles ── */
.kpi-tile {{
    background: {GLASS};
    backdrop-filter: {BLUR};
    border: 1px solid {BORDER};
    border-radius: 20px;
    padding: 24px 22px;
    text-align: left;
    position: relative;
    overflow: hidden;
    transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
}}
.kpi-tile::before {{
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
}}
.kpi-tile::after {{
    content: '';
    position: absolute; bottom: -30px; right: -30px;
    width: 100px; height: 100px;
    border-radius: 50%;
    opacity: 0.06;
    transition: opacity 0.3s;
}}
.kpi-tile:hover {{
    transform: translateY(-4px) scale(1.02);
    box-shadow: {GLOW_P};
    border-color: rgba(124,58,237,0.12);
}}
.kpi-tile:hover::after {{ opacity: 0.12; }}
.kpi-icon {{
    width: 44px; height: 44px;
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem;
    margin-bottom: 16px;
}}
.kpi-value {{
    font-size: 2.4rem; font-weight: 900; line-height: 1;
    background: linear-gradient(135deg, {TEXT}, {TEXT2});
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}}
.kpi-label {{ font-size: 0.82rem; color: {TEXT2}; font-weight: 500; letter-spacing: 0.02em; }}
.kpi-trend {{
    display: inline-flex; align-items: center; gap: 4px;
    font-size: 0.75rem; font-weight: 700;
    padding: 3px 10px; border-radius: 20px;
    margin-top: 12px;
}}
.trend-up {{ background: rgba(34,197,94,0.1); color: #22C55E; }}
.trend-down {{ background: rgba(239,68,68,0.1); color: #EF4444; }}

/* ── Section Header ── */
.section-hdr {{
    font-size: 1.1rem; font-weight: 700; color: {TEXT};
    padding-bottom: 12px; margin-bottom: 20px;
    border-bottom: 2px solid transparent;
    border-image: linear-gradient(90deg, {ACCENT}, transparent) 1;
    display: flex; align-items: center; gap: 10px;
}}

/* ── Status Pills ── */
.pill {{
    display: inline-flex; align-items: center; gap: 5px;
    padding: 5px 14px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 600;
    letter-spacing: 0.01em;
    transition: all 0.2s;
}}
.pill::before {{ content: ''; width: 6px; height: 6px; border-radius: 50%; }}
.pill-shortlisted {{ background: rgba(99,102,241,0.1); color: #818CF8; }}
.pill-shortlisted::before {{ background: #818CF8; }}
.pill-interview_scheduled {{ background: rgba(34,197,94,0.1); color: #4ADE80; }}
.pill-interview_scheduled::before {{ background: #4ADE80; }}
.pill-processing {{ background: rgba(251,191,36,0.1); color: #FBBF24; }}
.pill-processing::before {{ background: #FBBF24; }}
.pill-selected {{ background: rgba(16,185,129,0.1); color: #34D399; }}
.pill-selected::before {{ background: #34D399; }}
.pill-rejected {{ background: rgba(239,68,68,0.1); color: #F87171; }}
.pill-rejected::before {{ background: #F87171; }}
.pill-applied {{ background: rgba(113,113,122,0.1); color: #A1A1AA; }}
.pill-applied::before {{ background: #A1A1AA; }}
.pill-high {{ background: rgba(239,68,68,0.1); color: #F87171; }}
.pill-high::before {{ background: #F87171; }}
.pill-medium {{ background: rgba(251,191,36,0.1); color: #FBBF24; }}
.pill-medium::before {{ background: #FBBF24; }}
.pill-low {{ background: rgba(99,102,241,0.1); color: #818CF8; }}
.pill-low::before {{ background: #818CF8; }}
.pill-approved {{ background: rgba(34,197,94,0.1); color: #4ADE80; }}
.pill-approved::before {{ background: #4ADE80; }}
.pill-denied {{ background: rgba(239,68,68,0.1); color: #F87171; }}
.pill-denied::before {{ background: #F87171; }}

.cand-card {{
    background: {GLASS};
    backdrop-filter: {BLUR};
    border: 1px solid {BORDER};
    border-radius: 18px;
    padding: 22px;
    transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
    position: relative;
    overflow: hidden;
    width: 100%;
}}
.cand-card::before {{
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.06), transparent);
}}
.cand-card:hover {{
    transform: translateY(-3px);
    box-shadow: {GLOW_P};
    border-color: rgba(124,58,237,0.12);
}}
.cand-name {{ 
    font-weight: 700; 
    font-size: 1rem; 
    color: {TEXT}; 
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 100%;
}}
.cand-meta {{ 
    font-size: 0.82rem; 
    color: {TEXT2}; 
    margin-top: 3px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}}

.avatar-ring {{
    width: 48px; height: 48px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.9rem; color: white;
    flex-shrink: 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}}

.score-ring {{
    flex-shrink: 0;
    min-width: 72px;
    display: flex;
    justify-content: flex-end;
}}

/* ── Escalation Glow ── */
.esc-card {{
    background: {GLASS};
    backdrop-filter: {BLUR};
    border: 1px solid {BORDER};
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 12px;
    transition: all 0.3s;
    position: relative;
}}
.esc-card:hover {{ transform: translateY(-2px); }}
.esc-high {{ border-left: 3px solid #EF4444; box-shadow: -4px 0 20px rgba(239,68,68,0.08); }}
.esc-medium {{ border-left: 3px solid #F59E0B; box-shadow: -4px 0 20px rgba(245,158,11,0.08); }}
.esc-low {{ border-left: 3px solid #6366F1; box-shadow: -4px 0 20px rgba(99,102,241,0.08); }}

/* ── Upload Zone ── */
.upload-zone {{
    border: 2px dashed {BORDER};
    border-radius: 20px;
    padding: 40px;
    text-align: center;
    background: {GLASS};
    transition: all 0.3s;
}}
.upload-zone:hover {{
    border-color: {ACCENT};
    background: rgba(124,58,237,0.03);
}}

/* ── Leave Timeline ── */
.timeline-item {{
    position: relative;
    padding-left: 28px;
    padding-bottom: 20px;
    border-left: 2px solid {BORDER};
    margin-left: 10px;
}}
.timeline-item::before {{
    content: '';
    position: absolute; left: -6px; top: 4px;
    width: 10px; height: 10px;
    border-radius: 50%;
    border: 2px solid {ACCENT};
    background: {BG};
}}
.timeline-content {{
    background: {GLASS};
    backdrop-filter: {BLUR};
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 16px 18px;
}}

/* ── Empty State ── */
.empty-state {{
    text-align: center; padding: 60px 20px;
    color: {TEXT2};
}}
.empty-state .empty-icon {{ font-size: 3rem; margin-bottom: 16px; opacity: 0.4; }}
.empty-state p {{ font-size: 0.95rem; max-width: 400px; margin: 0 auto; }}

/* ── Inputs ── */
.stTextInput input, .stTextArea textarea, .stNumberInput input {{
    background: {INPUT_BG} !important;
    border: 1px solid {BORDER} !important;
    color: {TEXT} !important;
    border-radius: 14px !important;
    transition: border-color 0.2s !important;
    font-family: 'Inter', sans-serif !important;
}}
.stTextInput input:focus, .stTextArea textarea:focus {{
    border-color: {ACCENT} !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,0.1) !important;
}}

/* ── Buttons ── */
.stButton > button[kind="primary"] {{
    background: {GRAD} !important;
    border: none !important;
    border-radius: 14px !important;
    font-weight: 600 !important;
    padding: 12px 28px !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.3) !important;
    transition: all 0.25s cubic-bezier(0.4,0,0.2,1) !important;
    font-family: 'Inter', sans-serif !important;
}}
.stButton > button[kind="primary"]:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(124,58,237,0.4) !important;
}}
.stButton > button[kind="primary"]:active {{
    transform: translateY(0) scale(0.98) !important;
}}

/* ── Progress ── */
.stProgress > div > div {{ background: {GRAD} !important; border-radius: 8px !important; }}
.stProgress > div {{ background: rgba(255,255,255,0.05) !important; border-radius: 8px !important; height: 6px !important; }}

/* ── Divider ── */
hr {{ border-color: {BORDER} !important; opacity: 0.5 !important; }}

/* ── Brand ── */
.brand-wrap {{
    display: flex; align-items: center; gap: 14px;
    padding: 4px 0 20px;
}}
.brand-glyph {{
    width: 48px; height: 48px;
    background: {GRAD};
    border-radius: 16px;
    display: flex; align-items: center; justify-content: center;
    box-shadow: 0 4px 20px rgba(124,58,237,0.3), inset 0 1px 0 rgba(255,255,255,0.15);
}}
.brand-txt {{ font-weight: 800; font-size: 1.25rem; color: {TEXT}; }}
.brand-sub {{ font-size: 0.78rem; color: {ACCENT}; font-weight: 600; letter-spacing: 0.04em; }}

/* ── Navbar ── */
.navbar {{
    display: flex; align-items: center; justify-content: flex-end;
    gap: 12px; padding: 8px 0 20px;
}}
.nav-btn {{
    width: 40px; height: 40px;
    border-radius: 12px;
    background: {GLASS};
    border: 1px solid {BORDER};
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
    cursor: pointer;
    transition: all 0.2s;
}}
.nav-btn:hover {{ background: {CARD_S}; transform: scale(1.05); }}

/* ── Score ring container ── */
.score-ring {{ position: relative; display: inline-block; }}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ──
with st.sidebar:
    st.markdown(f"""
    <div class="brand-wrap">
        <div class="brand-glyph">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.2" stroke-linecap="round">
                <path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/>
            </svg>
        </div>
        <div>
            <div class="brand-txt">AI Recruit</div>
            <div class="brand-sub">PIPELINE DASHBOARD</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "nav",
        ["📊 Dashboard", "📄 Resume Screening", "📅 Interview Scheduling",
         "🎯 Interview Questions", "🏖️ Leave Management", "🚨 Escalations", "📦 Export Results"],
        label_visibility="collapsed",
    )

    st.divider()
    dark_toggle = st.toggle("🌙 Dark Mode", value=st.session_state.dark_mode)
    if dark_toggle != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_toggle
        st.rerun()


# ── Chart helper (Premium Styling) ──
def styled_layout(fig, title="", h=360):
    is_bar = any(t.type in ("bar",) for t in fig.data)
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=15, color=TEXT, family="Inter, sans-serif"),
            x=0, y=0.97, xanchor="left",
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT2, family="Inter, sans-serif", size=12),
        margin=dict(l=20, r=20, t=52, b=20),
        height=h,
        bargap=0.28,
        bargroupgap=0.08,
        xaxis=dict(
            gridcolor="rgba(0,0,0,0)",
            zerolinecolor=GRID_C,
            zerolinewidth=1,
            showgrid=False,
            tickfont=dict(size=12, color=TEXT2, family="Inter, sans-serif"),
            linecolor=GRID_C,
        ),
        yaxis=dict(
            gridcolor=GRID_C,
            zerolinecolor="rgba(0,0,0,0)",
            showgrid=True,
            gridwidth=1,
            tickfont=dict(size=12, color=TEXT2, family="Inter, sans-serif"),
        ),
        legend=dict(
            font=dict(size=11, family="Inter, sans-serif"),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
        ),
        hoverlabel=dict(
            bgcolor="#1C1C28",
            bordercolor="rgba(255,255,255,0.08)",
            font=dict(size=12, color="#F0F0F5", family="Inter, sans-serif"),
        ),
    )
    return fig

def status_str(c):
    return c.status.value if isinstance(c.status, PipelineStatus) else str(c.status)

def status_label(s):
    return {"applied":"Pending","processing":"Processing","shortlisted":"Shortlisted",
            "interview_scheduled":"Interview","selected":"Offer","rejected":"Rejected"}.get(s, s)

AV_COLORS = ['#6366F1','#8B5CF6','#EC4899','#14B8A6','#F59E0B','#EF4444','#06B6D4','#22C55E']
def av_color(name): return AV_COLORS[sum(ord(c) for c in name) % len(AV_COLORS)]
def initials(name): return "".join(w[0] for w in name.split()[:2]).upper()


# ═════════════════════════════════════════════════════
# 📊 DASHBOARD
# ═════════════════════════════════════════════════════
if page == "📊 Dashboard":
    # Navbar
    st.markdown(f"""
    <div class="navbar">
        <div class="nav-btn">🔔</div>
        <div class="nav-btn">👤</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f'<h1 style="font-size:1.8rem;font-weight:800;margin-bottom:4px;">Dashboard</h1>', unsafe_allow_html=True)
    st.caption("Real-time overview of your recruitment pipeline")

    pipeline = agent.pipeline
    total = len(pipeline)
    counts = {"shortlisted":0,"interview_scheduled":0,"selected":0,"rejected":0,"processing":0,"applied":0}
    for c in pipeline.values():
        s = c.status.value if isinstance(c.status, PipelineStatus) else str(c.status)
        if s in counts: counts[s] += 1

    # KPI Tiles
    cols = st.columns(5)
    kpis = [
        ("Total", total, "#7C3AED", "👥", "+12%", True, "rgba(124,58,237,0.15)"),
        ("Shortlisted", counts["shortlisted"], "#6366F1", "✅", "+8%", True, "rgba(99,102,241,0.15)"),
        ("Interviews", counts["interview_scheduled"], "#3B82F6", "📅", "+5%", True, "rgba(59,130,246,0.15)"),
        ("Offers", counts["selected"], "#22C55E", "🎉", "+3%", True, "rgba(34,197,94,0.15)"),
        ("Rejected", counts["rejected"], "#EF4444", "❌", "-2%", False, "rgba(239,68,68,0.15)"),
    ]
    for col, (label, val, color, icon, trend, up, glow) in zip(cols, kpis):
        trend_cls = "trend-up" if up else "trend-down"
        col.markdown(f"""
        <div class="kpi-tile">
            <div class="kpi-icon" style="background:{glow}">{icon}</div>
            <div class="kpi-value" style="background:linear-gradient(135deg,{color},{TEXT2});-webkit-background-clip:text;-webkit-text-fill-color:transparent;">{val}</div>
            <div class="kpi-label">{label}</div>
            <span class="kpi-trend {trend_cls}">{"↑" if up else "↓"} {trend}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.session_state.screened:
        ch1, ch2 = st.columns(2)

        with ch1:
            names = [c.name for c in st.session_state.screened]
            scores = [round(c.match_score * 100, 1) for c in st.session_state.screened]
            colors = ["#22C55E" if s>=80 else "#6366F1" if s>=60 else "#F59E0B" if s>=40 else "#EF4444" for s in scores]
            fig = go.Figure(go.Bar(
                x=scores, y=names, orientation='h',
                marker=dict(color=colors, cornerradius=8, line=dict(width=0)),
                text=[f"{s}%" for s in scores], textposition='outside',
                textfont=dict(size=12, color=TEXT, family="Inter, sans-serif"),
                width=0.6, opacity=0.95,
            ))
            fig.update_layout(yaxis=dict(autorange="reversed"))
            styled_layout(fig, "Match Scores")
            st.plotly_chart(fig, width="stretch")

        with ch2:
            skill_map = {}
            for c in st.session_state.screened:
                for s in (c.skills or []): skill_map[s] = skill_map.get(s, 0) + 1
            if skill_map:
                top = sorted(skill_map.items(), key=lambda x: -x[1])[:8]
                fig2 = go.Figure(go.Pie(
                    labels=[s[0] for s in top], values=[s[1] for s in top],
                    hole=0.62,
                    marker=dict(
                        colors=['#7C3AED','#6366F1','#3B82F6','#06B6D4','#14B8A6','#22C55E','#F59E0B','#EC4899'],
                        line=dict(color='rgba(0,0,0,0)', width=0),
                    ),
                    textinfo="label+percent",
                    textfont=dict(size=11, family="Inter, sans-serif"),
                    hoverinfo="label+value",
                    insidetextorientation="radial",
                ))
                fig2.add_annotation(
                    text=f"<b style='font-size:22px'>{sum(s[1] for s in top)}</b><br><span style='font-size:11px;color:{TEXT2}'>skills</span>",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=22, color=TEXT, family="Inter, sans-serif"),
                    align="center",
                )
                styled_layout(fig2, "Skill Distribution")
                st.plotly_chart(fig2, width="stretch")

        # Status chart
        smap = {}
        for c in st.session_state.screened:
            s = status_str(c); smap[s] = smap.get(s, 0) + 1
        if smap:
            sc = {"processing":"#F59E0B","shortlisted":"#6366F1","interview_scheduled":"#8B5CF6",
                  "selected":"#22C55E","rejected":"#EF4444","applied":"#94A3B8"}
            fig3 = go.Figure(go.Bar(
                x=[status_label(k) for k in smap], y=list(smap.values()),
                marker=dict(color=[sc.get(k,"#94A3B8") for k in smap], cornerradius=10),
                text=list(smap.values()), textposition='outside',
                textfont=dict(size=13, color=TEXT, family="Inter"),
            ))
            styled_layout(fig3, "Pipeline Status", h=300)
            st.plotly_chart(fig3, width="stretch")

        # Candidate list
        st.markdown(f'<div class="section-hdr">📋 Recent Candidates</div>', unsafe_allow_html=True)
        for c in st.session_state.screened:
            s = status_str(c)
            score_pct = int(c.match_score * 100)
            col1, col2, col3, col4 = st.columns([2.5, 1.2, 3, 1.5])
            col1.markdown(f"""
            <div style="display:flex;align-items:center;gap:14px;">
                <div class="avatar-ring" style="background:{av_color(c.name)}">{initials(c.name)}</div>
                <div>
                    <div class="cand-name">{c.name}</div>
                    <div class="cand-meta">{c.candidate_id}</div>
                </div>
            </div>""", unsafe_allow_html=True)
            col2.markdown(f"<span style='color:{TEXT2};font-size:0.88rem'>{c.experience_years} yrs</span>", unsafe_allow_html=True)
            col3.progress(min(score_pct, 100), text=f"{score_pct}% match")
            col4.markdown(f'<span class="pill pill-{s}">{status_label(s)}</span>', unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="empty-state">
            <div class="empty-icon">📭</div>
            <h3 style="color:{TEXT2}; font-weight:600; margin-bottom:8px;">No candidates yet</h3>
            <p>Upload resumes in <strong>Resume Screening</strong> to start building your pipeline.</p>
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════
# 📄 RESUME SCREENING
# ═════════════════════════════════════════════════════
elif page == "📄 Resume Screening":
    st.markdown(f'<div class="navbar"><div class="nav-btn">🔔</div><div class="nav-btn">👤</div></div>', unsafe_allow_html=True)
    st.markdown(f'<h1 style="font-size:1.8rem;font-weight:800;">Resume Screening</h1>', unsafe_allow_html=True)
    st.caption("Upload multiple resumes · AI-powered batch analysis")

    # Job Description
    st.markdown(f'<div class="section-hdr">📝 Job Description</div>', unsafe_allow_html=True)
    jc1, jc2 = st.columns([3, 1])
    jd_title = jc1.text_input("Job Title", value="Senior Python Developer")
    jd_exp = jc2.number_input("Min Experience", min_value=0, value=3)
    jd_desc = st.text_area("Description", value="Looking for a senior Python developer with expertise in REST APIs, Docker, AWS, and machine learning.", height=80)
    jd_skills = st.text_input("Required Skills", value="Python, Docker, AWS, REST APIs, Machine Learning")

    st.markdown("<br>", unsafe_allow_html=True)

    # Upload
    st.markdown(f'<div class="section-hdr">📎 Upload Resumes</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drag & drop resume files — PDF, DOCX, or TXT",
        type=["pdf", "docx", "txt", "md"], accept_multiple_files=True,
    )
    if uploaded:
        st.success(f"📎 {len(uploaded)} resume(s) ready")

    if st.button("🚀 Screen All Resumes", type="primary", width="stretch", disabled=not uploaded):
        required = [s.strip() for s in jd_skills.split(",") if s.strip()]
        jd = SimpleNamespace(title=jd_title, description=jd_desc, required_skills=required, preferred_skills=[], min_experience=jd_exp)

        candidates = []
        prog = st.progress(0, text="Extracting resumes...")
        for i, f in enumerate(uploaded):
            text = extract_text_from_file(f)
            name = extract_name(text, fallback=f.name.rsplit(".", 1)[0])
            candidates.append(Candidate(
                candidate_id=f"C{1001+i:04d}", name=name,
                email=f"{name.lower().replace(' ','.')}@email.com", resume_text=text,
            ))
            prog.progress((i+1)/len(uploaded), text=f"Extracting {i+1}/{len(uploaded)}...")

        prog.progress(100, text="Screening with AI engine...")
        ranked = agent.screen_resumes(candidates, jd)
        st.session_state.screened = ranked
        prog.empty()
        st.success(f"✅ {len(ranked)} candidates screened!")

    # Results
    if st.session_state.screened:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="section-hdr">📊 Results</div>', unsafe_allow_html=True)

        # Filters
        fc1, fc2 = st.columns([1, 1])
        filt = fc1.selectbox("Filter", ["All","Shortlisted","Processing","Rejected","Selected"], key="filt_sel")
        sort = fc2.selectbox("Sort by", ["Score ↓","Score ↑","Experience ↓","Name A-Z"], key="sort_sel")

        # Fix 5: use explicit status_map so filter matches backend lowercase values
        status_map = {
            "Shortlisted": "shortlisted",
            "Processing": "processing",
            "Rejected": "rejected",
            "Selected": "selected",
        }
        filtered = list(st.session_state.screened)
        if filt != "All":
            filtered = [c for c in filtered if status_str(c) == status_map.get(filt, filt.lower())]
        if sort == "Score ↓": filtered.sort(key=lambda c: -c.match_score)
        elif sort == "Score ↑": filtered.sort(key=lambda c: c.match_score)
        elif sort == "Experience ↓": filtered.sort(key=lambda c: -c.experience_years)
        elif sort == "Name A-Z": filtered.sort(key=lambda c: c.name)

        # Score chart
        if filtered:
            names = [c.name for c in filtered]
            scores = [round(c.match_score*100,1) for c in filtered]
            colors = ["#22C55E" if s>=80 else "#6366F1" if s>=60 else "#F59E0B" if s>=40 else "#EF4444" for s in scores]
            fig = go.Figure(go.Bar(
                x=scores, y=names, orientation='h',
                marker=dict(color=colors, cornerradius=8),
                text=[f"{s}%" for s in scores], textposition='outside',
                textfont=dict(size=12, color=TEXT, family="Inter"),
            ))
            fig.update_layout(yaxis=dict(autorange="reversed"))
            styled_layout(fig, f"Scores — {len(filtered)} candidates")
            st.plotly_chart(fig, width="stretch")
            
            # Suitability Summary
            suitable = [c for c in filtered if c.match_score >= 0.6]
            if suitable:
                st.markdown('<div class="section-hdr">✨ Best Fit Recommendations</div>', unsafe_allow_html=True)
                excellent = [c for c in suitable if c.match_score >= 0.8]
                good = [c for c in suitable if 0.6 <= c.match_score < 0.8]
                
                sum_cols = st.columns(2)
                with sum_cols[0]:
                    if excellent:
                        st.markdown('<div style="font-weight:600;color:#22C55E;margin-bottom:8px;">🌟 Excellent Fit</div>', unsafe_allow_html=True)
                        for c in excellent:
                            st.markdown(f"""
                            <div class="glass" style="margin-bottom:8px;padding:12px;display:flex;justify-content:space-between;align-items:center;">
                                <span style="font-weight:600;color:{TEXT};">{c.name}</span>
                                <span style="font-weight:700;color:#22C55E;">{int(c.match_score*100)}%</span>
                            </div>""", unsafe_allow_html=True)
                with sum_cols[1]:
                    if good:
                        st.markdown('<div style="font-weight:600;color:#6366F1;margin-bottom:8px;">✅ Good Fit</div>', unsafe_allow_html=True)
                        for c in good:
                            st.markdown(f"""
                            <div class="glass" style="margin-bottom:8px;padding:12px;display:flex;justify-content:space-between;align-items:center;">
                                <span style="font-weight:600;color:{TEXT};">{c.name}</span>
                                <span style="font-weight:700;color:#6366F1;">{int(c.match_score*100)}%</span>
                            </div>""", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

        # Card grid
        if "shortlisted_ids" not in st.session_state:
            st.session_state.shortlisted_ids = set()

        cols = st.columns(3)
        for i, c in enumerate(filtered):
            score_pct = int(c.match_score * 100)
            s = status_str(c)
            is_shortlisted = c.candidate_id in st.session_state.shortlisted_ids
            with cols[i % 3]:
                ring = score_ring_svg(score_pct, 72, 6)
                st.markdown(f"""
                <div class="cand-card">
                    <div style="display:flex;justify-content:space-between;align-items:center;padding-bottom:14px;border-bottom:1px solid {BORDER};">
                        <div style="display:flex;gap:12px;align-items:center;flex:1;min-width:0;">
                            <div class="avatar-ring" style="background:{av_color(c.name)}">{initials(c.name)}</div>
                            <div style="flex:1;min-width:0;">
                                <div class="cand-name" title="{c.name}" style="font-size:1.1rem;margin-bottom:2px;">{c.name}</div>
                                <div class="cand-meta" title="{c.candidate_id}" style="color:{TEXT2};">{c.candidate_id} &bull; {c.experience_years} yrs</div>
                            </div>
                        </div>
                        <div class="score-ring" style="flex-shrink:0;">{ring}</div>
                    </div>
                     <div style="margin-top:14px;display:flex;gap:6px;flex-wrap:wrap;min-height:30px;">
                         {''.join(f'<span class="truncate" title="{sk}" style="background:{INPUT_BG};border:1px solid {BORDER};border-radius:6px;padding:2px 8px;font-size:0.7rem;color:{TEXT2};max-width:100px;">{sk}</span>' for sk in (sorted(c.skills)[:12] if c.skills else []))}
                     </div>
                    <div style="margin-top:14px;display:flex;justify-content:space-between;align-items:center;">
                        <span class="pill pill-{s}" style="font-size:0.75rem;">{status_label(s)}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

                # Shortlist toggle
                btn_label = "✅ Shortlisted" if is_shortlisted else "⬆️ Shortlist for Interview"
                if st.button(btn_label, key=f"sl_{c.candidate_id}", width="stretch"):
                    if is_shortlisted:
                        st.session_state.shortlisted_ids.discard(c.candidate_id)
                        if c.candidate_id in agent.pipeline:
                            agent.pipeline[c.candidate_id].status = PipelineStatus.PROCESSING
                    else:
                        st.session_state.shortlisted_ids.add(c.candidate_id)
                        if c.candidate_id in agent.pipeline:
                            agent.pipeline[c.candidate_id].status = PipelineStatus.SHORTLISTED
                    st.rerun()

                # Safe resume text expander — uses st.code() to prevent HTML injection
                with st.expander(f"📄 View Resume — {c.name}"):
                    raw_text = getattr(c, 'resume_text', '') or '(No resume text available)'
                    # Strip excessive whitespace for a cleaner display
                    clean_text = '\n'.join(line for line in raw_text.splitlines() if line.strip())
                    st.code(clean_text, language=None)

                st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════
# 📅 SCHEDULING
# ═════════════════════════════════════════════════════
elif page == "📅 Interview Scheduling":
    st.markdown(f'<div class="navbar"><div class="nav-btn">🔔</div><div class="nav-btn">👤</div></div>', unsafe_allow_html=True)
    st.markdown(f'<h1 style="font-size:1.8rem;font-weight:800;">Interview Scheduling</h1>', unsafe_allow_html=True)
    st.caption("Automated conflict-free interview scheduling")

    if not st.session_state.screened:
        st.markdown(f"""<div class="empty-state"><div class="empty-icon">📅</div><h3 style="color:{TEXT2};font-weight:600;">Screen resumes first</h3><p>Upload and screen candidates before scheduling interviews.</p></div>""", unsafe_allow_html=True)
    else:
        st.info("💡 **Smart Scheduling Enabled**: The system now uses internal availability datasets and automatically generates slots for any new candidates. No file uploads required.")

        # Only schedule shortlisted candidates
        shortlisted_cands = [c for c in st.session_state.screened if c.candidate_id in st.session_state.get("shortlisted_ids", set())]
        if not shortlisted_cands:
            shortlisted_cands = st.session_state.screened
            st.caption(f"ℹ️ No candidates shortlisted — scheduling all {len(shortlisted_cands)} screened candidates.")
            st.caption(f"📋 {len(shortlisted_cands)} shortlisted candidate(s) will be scheduled.")

        if st.button("📅 Schedule Interviews", type="primary", width="stretch"):
            with st.spinner("Running scheduling engine..."):
                results = agent.schedule_candidates(shortlisted_cands)
                st.session_state.scheduled = results

            scheduled = [r for r in results if r.get("status") in ["scheduled", "scheduled_via_score_priority", "manually_scheduled"]]
            conflicts = [r for r in results if r.get("status") not in ["scheduled", "scheduled_via_score_priority", "manually_scheduled"]]

            if scheduled:
                st.markdown(f'<div class="section-hdr">✅ Scheduled ({len(scheduled)})</div>', unsafe_allow_html=True)
                for r in scheduled:
                    slot = r['slot']
                    is_priority = r.get("status") == "scheduled_via_score_priority"
                    is_manual = r.get("status") == "manually_scheduled"
                    
                    bg = f"rgba(99, 102, 241, 0.05)" if is_priority else f"rgba(34, 197, 94, 0.05)"
                    border = f"#6366F1" if is_priority else f"#22C55E"
                    tag_text = "PRIORITY MATCH" if is_priority else "MANUAL OVERRIDE" if is_manual else "AUTO SCHEDULED"
                    
                    st.markdown(f"""
                    <div class="glass" style="margin-bottom:12px;padding:22px;border-left:4px solid {border};background:{bg};overflow:hidden;">
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;gap:12px;">
                            <strong class="truncate" style="font-size:1rem;color:{TEXT};flex:1;min-width:0;" title="{r['candidate']}">{r['candidate']}</strong>
                            <span style="font-size:0.7rem;font-weight:700;color:{border};padding:4px 12px;border-radius:20px;background:rgba({int(border[1:3],16)},{int(border[3:5],16)},{int(border[5:7],16)},0.1);white-space:nowrap;flex-shrink:0;">{tag_text}</span>
                        </div>
                        <div style="display:grid;grid-template-columns:1fr 1fr;gap:15px;font-size:0.85rem;">
                            <div style="color:{TEXT2};">📅 {pd.to_datetime(slot['start_time']).strftime('%b %d, %Y')}</div>
                            <div style="color:{TEXT2};">🕒 {pd.to_datetime(slot['start_time']).strftime('%I:%M %p')} - {pd.to_datetime(slot['end_time']).strftime('%I:%M %p')}</div>
                            <div style="color:{TEXT2};">👤 {slot.get('interviewer_id', 'TBD')}</div>
                            <div style="color:{TEXT2};">⚙️ {r['status'].replace('_', ' ').title()}</div>
                        </div>
                    </div>""", unsafe_allow_html=True)

            if conflicts:
                st.markdown(f'<div class="section-hdr">⚠️ Action Required ({len(conflicts)})</div>', unsafe_allow_html=True)
                for r in conflicts:
                    cid = r['candidate_id']
                    st.markdown(f"""
                    <div class="glass" style="margin-bottom:15px;padding:18px;border-left:4px solid #F59E0B;">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <strong style="color:{TEXT}">{r.get('candidate','')}</strong>
                            <span style="color:#F59E0B;font-size:0.75rem;font-weight:700;text-transform:uppercase;">{r.get('status','').replace('_', ' ')}</span>
                        </div>
                    </div>""", unsafe_allow_html=True)
                    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                    
                    # Manual Override Form
                    with st.expander(f"🛠️ Manually Schedule {r.get('candidate','')}"):
                        from zoneinfo import ZoneInfo
                        from datetime import time as dtime
                        KOLKATA_TZ = ZoneInfo("Asia/Kolkata")
                        m_date = st.date_input("Interview Date", key=f"md_{cid}",
                                               min_value=datetime.now(KOLKATA_TZ).date())
                        m_time = st.time_input("Interview Time", key=f"mt_{cid}")
                        if st.button("Confirm Manual Booking", key=f"mb_{cid}", type="primary"):
                            now_dt = datetime.now(KOLKATA_TZ)
                            # Fix 1a: past-date guard
                            if m_date < now_dt.date():
                                st.error("❌ Date cannot be in the past.")
                            # Fix 1b: business hours check using time objects
                            elif not (dtime(10, 0) <= m_time <= dtime(17, 30)):
                                st.error("❌ Interview must be between 10:00 AM – 5:30 PM IST.")
                            # Fix 1c: if today, reject times already passed (clean comparison, no tzinfo juggle)
                            elif m_date == now_dt.date() and m_time <= now_dt.time():
                                st.error("❌ Selected time has already passed today. Please choose a later time.")
                            else:
                                pref_dt = datetime.combine(m_date, m_time).replace(tzinfo=KOLKATA_TZ)
                                res = agent.request_manual_time(cid, pref_dt)
                                if "error" in res:
                                    st.error(res["error"])
                                else:
                                    st.success(f"✅ Interview booked for {m_date.strftime('%b %d, %Y')} at {m_time.strftime('%I:%M %p')}!")
                                    updated = False
                                    for idx, s_res in enumerate(st.session_state.get("scheduled", [])):
                                        if s_res['candidate_id'] == cid:
                                            st.session_state.scheduled[idx] = res
                                            updated = True
                                            break
                                    if not updated:
                                        st.session_state.setdefault("scheduled", []).append(res)
                                    st.rerun()


# ═════════════════════════════════════════════════════
# 🎯 INTERVIEW QUESTIONS (Groq LLM)
# ═════════════════════════════════════════════════════
elif page == "🎯 Interview Questions":
    st.markdown(f'<div class="navbar"><div class="nav-btn">🔔</div><div class="nav-btn">👤</div></div>', unsafe_allow_html=True)
    st.markdown(f'<h1 style="font-size:1.8rem;font-weight:800;">Interview Questions</h1>', unsafe_allow_html=True)
    st.caption("AI-generated structured questions powered by Groq LLM")

    if not st.session_state.screened:
        st.markdown(f"""<div class="empty-state"><div class="empty-icon">🎯</div><h3 style="color:{TEXT2};font-weight:600;">Screen candidates first</h3><p>Upload and screen resumes before generating interview questions.</p></div>""", unsafe_allow_html=True)
    else:
        # Pick candidate
        shortlisted_ids = st.session_state.get("shortlisted_ids", set())
        shortlisted_cands = [c for c in st.session_state.screened if c.candidate_id in shortlisted_ids]
        pool = shortlisted_cands if shortlisted_cands else st.session_state.screened
        cand_options = {f"{c.name} ({c.candidate_id})": c for c in pool}

        st.markdown(f'<div class="section-hdr">👤 Select Candidate</div>', unsafe_allow_html=True)
        selected_name = st.selectbox("Candidate", list(cand_options.keys()), label_visibility="collapsed")
        selected_cand = cand_options[selected_name]

        # JD inputs
        st.markdown(f'<div class="section-hdr">📝 Job Context</div>', unsafe_allow_html=True)
        q_jc1, q_jc2 = st.columns([3, 1])
        q_title = q_jc1.text_input("Job Title", value="Senior Python Developer", key="q_title")
        q_exp = q_jc2.number_input("Min Exp", min_value=0, value=3, key="q_exp")
        q_skills = st.text_input("Required Skills", value="Python, Docker, AWS, REST APIs, Machine Learning", key="q_skills")

        if st.button("🎯 Generate Questions", type="primary", width="stretch"):
            req_skills = [s.strip() for s in q_skills.split(",") if s.strip()]
            jd = SimpleNamespace(title=q_title, description=f"Looking for {q_title}",
                                required_skills=req_skills, preferred_skills=[], min_experience=q_exp)

            with st.spinner("Generating interview questions with Groq LLM..."):
                try:
                    questions = agent.generate_interview_questions(jd, selected_cand)
                    st.session_state.generated_questions = questions
                    st.session_state.question_candidate = selected_cand.name
                except Exception as e:
                    st.error(f"Error generating questions: {e}")
                    st.session_state.generated_questions = []

        # Display questions
        if st.session_state.get("generated_questions"):
            st.markdown("<br>", unsafe_allow_html=True)
            cand_name = st.session_state.get("question_candidate", "")
            st.markdown(f'<div class="section-hdr">📋 Questions for {cand_name}</div>', unsafe_allow_html=True)

            type_icons = {"technical": "💻", "behavioral": "🧠", "situational": "🎭", "candidate_specific": "🎯"}
            type_colors = {"technical": "#6366F1", "behavioral": "#22C55E", "situational": "#F59E0B", "candidate_specific": "#EC4899"}

            for i, q in enumerate(st.session_state.generated_questions, 1):
                q_type = q.get("type", "technical").lower().replace("-", "_").replace(" ", "_")
                icon = type_icons.get(q_type, "❓")
                color = type_colors.get(q_type, ACCENT)
                category = q.get("category", q_type.replace("_", " ").title())
                difficulty = q.get("difficulty", "medium")

                diff_color = "#EF4444" if difficulty == "hard" else "#F59E0B" if difficulty == "medium" else "#22C55E"

                st.markdown(f"""
                <div class="glass" style="margin-bottom:12px;padding:22px;overflow:hidden;">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;gap:12px;">
                        <div style="display:flex;align-items:center;gap:10px;flex:1;min-width:0;">
                            <span style="font-size:1.3rem;flex-shrink:0;">{icon}</span>
                            <span style="font-weight:700;color:{TEXT};font-size:0.95rem;flex-shrink:0;">Q{i}</span>
                            <span class="pill truncate" title="{category}" style="background:rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.12);color:{color};flex:1;min-width:0;">{category}</span>
                        </div>
                        <span style="font-size:0.75rem;font-weight:600;color:{diff_color};padding:3px 10px;border-radius:20px;background:rgba({int(diff_color[1:3],16)},{int(diff_color[3:5],16)},{int(diff_color[5:7],16)},0.1);flex-shrink:0;">{difficulty.upper()}</span>
                    </div>
                    <div style="color:{TEXT};font-size:0.95rem;line-height:1.6;">
                        {q.get('question', 'No question text')}
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════
# 🏖️ LEAVE MANAGEMENT
# ═════════════════════════════════════════════════════
elif page == "🏖️ Leave Management":
    st.markdown(f'<div class="navbar"><div class="nav-btn">🔔</div><div class="nav-btn">👤</div></div>', unsafe_allow_html=True)
    st.markdown(f'<h1 style="font-size:1.8rem;font-weight:800;">Leave Management</h1>', unsafe_allow_html=True)
    st.caption("Working-day aware leave processing")

    st.markdown(f'<div class="section-hdr">📝 New Request</div>', unsafe_allow_html=True)
    lc1, lc2 = st.columns(2)
    emp_id = lc1.text_input("Employee ID", value="E100")
    leave_type = lc2.selectbox("Leave Type", ["annual", "sick", "casual"])
    dc1, dc2 = st.columns(2)
    start_date = dc1.date_input("Start Date")
    end_date = dc2.date_input("End Date")
    reason = st.text_input("Reason", placeholder="Reason for leave")
    # Balance is now auto-calculated by the backend
    # balance = st.number_input("Current Balance", min_value=0, value=20)
    balance = 12  # Standard annual quota fallback

    if st.button("Submit Leave Request", type="primary", width="stretch"):
        from zoneinfo import ZoneInfo
        tz = ZoneInfo("Asia/Kolkata")
        today_date = datetime.now(tz).date()

        # Fix 3: Leave Form Validation
        if start_date < today_date:
            st.error("❌ Start date cannot be in the past.")
            st.stop()
        # Validation: end before start (only check — days<0 is identical, removed as redundant)
        if end_date < start_date:
            st.error("❌ End date cannot be before the start date.")
            st.stop()

        # Fix 3d: default reason
        safe_reason = reason.strip() if reason and reason.strip() else "Personal"

        try:
            req_start_date = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=tz)
            req_end_date = datetime.combine(end_date, datetime.min.time()).replace(tzinfo=tz)
        except TypeError:
            req_start_date = datetime.now(tz)
            req_end_date = datetime.now(tz)
            
        req = LeaveRequest(
            request_id=f"LR-{len(agent._processed_leave_requests)+1:03d}",
            employee_id=emp_id, leave_type=leave_type,
            start_date=req_start_date,
            end_date=req_end_date,
            reason=safe_reason,
        )
        pol = LeavePolicy(leave_type=leave_type, annual_quota=balance, max_consecutive_days=10,
                          min_notice_days=1 if leave_type != "sick" else 0, requires_document=(leave_type=="sick"))
        # Backend now auto-calculates balance and checks team capacity
        result = agent.process_leave(req, pol)
        
        # DISPLAY ML PREDICTIONS
        if result.get("ml_used"):
            conf = result.get('ml_confidence', 0) * 100
            risk = result.get('risk_score', 0)
            # Fix 7: clamp risk to [0, 100] and derive approval_percentage
            risk = min(max(float(risk), 0.0), 100.0)
            approval_percentage = 100.0 - risk
            
            risk_color = "#22C55E" if risk < 30 else "#F59E0B" if risk < 70 else "#EF4444"
            st.markdown(f"""
<div class="glass" style="margin-top:10px;margin-bottom:20px;padding:20px;border-left:4px solid {risk_color};">
    <div style="display:flex;justify-content:space-between;align-items:center;">
        <div>
            <div style="color:{TEXT2};font-size:0.85rem;font-weight:600;text-transform:uppercase;letter-spacing:1px;">Leave Risk Score</div>
            <div style="font-size:1.8rem;font-weight:800;color:{risk_color};">{risk:.1f}%</div>
        </div>
        <div style="text-align:right;">
            <div style="color:{TEXT2};font-size:0.85rem;font-weight:600;text-transform:uppercase;letter-spacing:1px;">Approval Confidence</div>
            <div style="font-size:1.4rem;font-weight:700;color:{TEXT};">{conf:.1f}%</div>
        </div>
    </div>
    <div style="margin-top:15px;height:6px;background:rgba(255,255,255,0.05);border-radius:3px;overflow:hidden;">
        <div style="height:100%;width:{approval_percentage:.1f}%;background:{risk_color};"></div>
    </div>
</div>
""", unsafe_allow_html=True)

        if result["approved"]:
            st.success(f"✅ Approved — {result['days_requested']} working days | Remaining: {result['remaining_balance']}")
        else:
            st.error(f"❌ Denied — {result['reason']}")
            for v in result.get("violations", []): st.warning(f"⚠️ {v}")

    if agent._processed_leave_requests:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="section-hdr">📜 Leave History</div>', unsafe_allow_html=True)

        for lr in reversed(agent._processed_leave_requests):
            approved = lr["approved"]
            icon = "✅" if approved else "❌"
            status = "approved" if approved else "denied"
            
            # Fix 4: use backend-provided risk_score directly (not recalculated from confidence)
            risk_badge = ""
            raw_risk = lr.get('risk_score')
            if raw_risk is not None:
                risk = min(max(float(raw_risk), 0.0), 100.0)
                r_col = "#22C55E" if risk < 30 else "#F59E0B" if risk < 70 else "#EF4444"
                risk_badge = f'<span style="font-size:0.75rem;padding:2px 8px;border-radius:10px;background:{r_col}20;color:{r_col};margin-left:10px;font-weight:600;">Risk: {risk:.0f}%</span>'
                
            # Fix 4: format dates from ISO strings to human-readable
            try:
                start_fmt = pd.to_datetime(lr.get('start_date', '')).strftime('%b %d, %Y')
                end_fmt   = pd.to_datetime(lr.get('end_date', '')).strftime('%b %d, %Y')
            except Exception:
                start_fmt = str(lr.get('start_date', ''))
                end_fmt   = str(lr.get('end_date', ''))

            leave_html = f"""
<div class="timeline-item">
    <div class="timeline-content">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <div>
                <strong style="color:{TEXT}">{icon} {lr['request_id']} &mdash; {lr['employee_id']}</strong>
                {risk_badge}
            </div>
            <span class="pill pill-{status}">{status.title()}</span>
        </div>
        <div style="margin-top:8px;font-size:0.85rem;color:{TEXT2};">
            {lr['leave_type'].title()} &middot; {lr['days_requested']} days &middot;
            {start_fmt} &rarr; {end_fmt}
        </div>
    </div>
</div>"""
            st.markdown(leave_html, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════
# 🚨 ESCALATIONS
# ═════════════════════════════════════════════════════
elif page == "🚨 Escalations":
    st.markdown(f'<div class="navbar"><div class="nav-btn">🔔</div><div class="nav-btn">👤</div></div>', unsafe_allow_html=True)
    st.markdown(f'<h1 style="font-size:1.8rem;font-weight:800;">Escalation Center</h1>', unsafe_allow_html=True)
    st.caption("Intelligent severity detection · Audit-ready logs")

    st.markdown(f'<div class="section-hdr">📝 Submit Concern</div>', unsafe_allow_html=True)
    ec1, ec2 = st.columns([1, 3])
    esc_emp = ec1.text_input("Employee ID", value="E100", key="esc_emp")
    esc_query = ec2.text_area("Describe your concern", placeholder="Describe the issue...", height=100)

    if st.button("Submit Escalation", type="primary", width="stretch"):
        if not esc_query.strip():
            st.warning("Please describe your concern.")
        else:
            result = agent.handle_query(esc_query, {"employee_id": esc_emp})
            if result["escalated"]:
                st.error(f"🚨 Escalated — Priority: **{result['priority'].upper()}**")
                st.info(result.get("message", ""))
            else:
                st.success(f"✅ {result.get('response','Query processed.')}")

    if agent._escalation_log:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f'<div class="section-hdr">📜 Escalation Log</div>', unsafe_allow_html=True)

        sev_icon = {"high": "🔴", "medium": "🟡", "low": "🔵"}
        for entry in reversed(agent._escalation_log):
            pri = entry.get("priority", "low")
            icon = sev_icon.get(pri, "⚪")
            st.markdown(f"""
                 <div class="esc-card esc-{pri}" style="overflow:hidden;">
                    <div style="display:flex;justify-content:space-between;align-items:center;gap:10px;">
                        <div style="display:flex;align-items:center;gap:10px;flex:1;min-width:0;">
                            <span style="font-size:1.3rem;flex-shrink:0;">{icon}</span>
                            <strong class="truncate" style="color:{TEXT};flex:1;min-width:0;" title="{entry.get('employee_id','')}">{entry.get('employee_id','')}</strong>
                        </div>
                        <span class="pill pill-{pri}" style="flex-shrink:0;">{pri.upper()}</span>
                    </div>
                    <div class="truncate" style="margin-top:10px;font-size:0.8rem;color:{TEXT2};" title="{entry.get('escalation_reason','') or ''}">
                        {entry.get('escalation_reason','') or ''} &middot; {entry.get('timestamp','')}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            # Fix 5: render user query_text safely — never via unsafe_allow_html
            if entry.get('query_text'):
                st.write(entry['query_text'])


# ═════════════════════════════════════════════════════
# 📦 EXPORT
# ═════════════════════════════════════════════════════
elif page == "📦 Export Results":
    st.markdown(f'<div class="navbar"><div class="nav-btn">🔔</div><div class="nav-btn">👤</div></div>', unsafe_allow_html=True)
    st.markdown(f'<h1 style="font-size:1.8rem;font-weight:800;">Export Results</h1>', unsafe_allow_html=True)
    st.caption("Download pipeline data in hackathon format")

    export = agent.export_results()
    r = export["results"]

    cols = st.columns(3)
    metrics = [
        ("Candidates", len(r["resume_screening"]["ranked_candidates"]), "👥"),
        ("Interviews", len(r["scheduling"]["interviews_scheduled"]), "📅"),
        ("Escalations", len(r["escalations"]), "🚨"),
    ]
    for col, (label, val, icon) in zip(cols, metrics):
        col.markdown(f"""
        <div class="kpi-tile" style="text-align:center;">
            <div style="font-size:1.5rem;margin-bottom:8px;">{icon}</div>
            <div class="kpi-value" style="font-size:1.8rem;">{val}</div>
            <div class="kpi-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    cols2 = st.columns(3)
    metrics2 = [
        ("Questions", len(r["questionnaire"]["questions"]), "❓"),
        ("Leave Requests", len(r["leave_management"]["processed_requests"]), "🏖️"),
        ("Pipeline Entries", len(r["pipeline"]["candidates"]), "🔄"),
    ]
    for col, (label, val, icon) in zip(cols2, metrics2):
        col.markdown(f"""
        <div class="kpi-tile" style="text-align:center;">
            <div style="font-size:1.5rem;margin-bottom:8px;">{icon}</div>
            <div class="kpi-value" style="font-size:1.8rem;">{val}</div>
            <div class="kpi-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    json_str = json.dumps(export, indent=2, default=str)
    st.download_button("⬇️ Download export_results.json", json_str, "export_results.json", "application/json", type="primary", width="stretch")
    with st.expander("View JSON"):
        st.code(json_str, language="json")
