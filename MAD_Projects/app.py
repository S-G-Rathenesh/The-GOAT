import base64
import hashlib
import io
import sys
import uuid
import xml.sax.saxutils
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Spacer, Table, TableStyle, Paragraph

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from goat_analysis import DATA_DIR, DEFAULT_BREEDS, GoatAnalyzer

LOGO_PATH = ROOT / "logo.png"
if not LOGO_PATH.exists() and (ROOT.parent / "logo.png").exists():
    LOGO_PATH = ROOT.parent / "logo.png"

st.set_page_config(page_title="THE GOAT", page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "🐐", layout="wide")

# Note: Logo is rendered with custom responsive branding in render_header and render_navigation

STEP_LABELS = [
    "Goat Profile",
    "Image Capture",
    "Goat Detection & Segmentation",
    "Body Feature Extraction",
    "Breed-Aware Weight Estimation",
    "Temporary Weight Anomaly Detection",
    "Result & Fair-Trade Analysis",
    "History & Reports",
]


def init_session_state():
    defaults = {
        "current_step": 0,
        "goat_id": "G-101",
        "owner_name": "Demo Farmer",
        "breed": DEFAULT_BREEDS[0],
        "age_months": 12,
        "sex": "Male",
        "observed_weight": 28.0,
        "image_path": None,
        "analysis_result": None,
        "current_features": None,
        "history_list": [],
        "splash_shown": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def can_access_step(step_index):
    if step_index in (0, 1):
        return True

    if step_index == 7:
        # History & Reports can always be visited to view records or current state
        return True

    if not st.session_state.get("image_path") or not Path(st.session_state.image_path).exists():
        return False

    if step_index in (2, 3):
        return True

    if step_index >= 4 and st.session_state.get("current_features") is None:
        try:
            analyzer = GoatAnalyzer()
            st.session_state["current_features"] = analyzer._extract_features_from_image(st.session_state.image_path)
        except Exception:
            return False

    return True


def sync_profile():
    if "widget_goat_id" in st.session_state:
        st.session_state.goat_id = str(st.session_state.widget_goat_id)
    if "widget_owner_name" in st.session_state:
        st.session_state.owner_name = str(st.session_state.widget_owner_name)
    if "widget_breed" in st.session_state:
        st.session_state.breed = str(st.session_state.widget_breed)
    if "widget_age_months" in st.session_state:
        st.session_state.age_months = int(st.session_state.widget_age_months)
    if "widget_sex" in st.session_state:
        st.session_state.sex = str(st.session_state.widget_sex)
    if "widget_observed_weight" in st.session_state:
        st.session_state.observed_weight = float(st.session_state.widget_observed_weight)


def go_next():
    sync_profile()
    target = st.session_state.current_step + 1
    if target < len(STEP_LABELS) and can_access_step(target):
        st.session_state.current_step = target


def go_back():
    sync_profile()
    if st.session_state.current_step > 0:
        st.session_state.current_step -= 1


def set_step(idx):
    sync_profile()
    if can_access_step(idx):
        st.session_state.current_step = idx


def go_home():
    sync_profile()
    st.session_state.current_step = 0


def start_new_assessment():
    sync_profile()
    current_id = str(st.session_state.get("goat_id", "G-101"))
    next_id = "G-102"
    if current_id.startswith("G-"):
        parts = current_id.split("-")
        if len(parts) == 2 and parts[1].isdigit():
            next_id = f"G-{int(parts[1]) + 1}"
        else:
            next_id = f"G-{uuid.uuid4().hex[:4].upper()}"
    else:
        next_id = f"G-{uuid.uuid4().hex[:4].upper()}"

    st.session_state.goat_id = next_id
    if "widget_goat_id" in st.session_state:
        st.session_state.widget_goat_id = next_id
    st.session_state.image_path = None
    st.session_state.current_features = None
    st.session_state.analysis_result = None
    st.session_state.current_step = 0


def get_weight_authenticity(anomaly_score: float, deviation_pct: float) -> dict:
    if anomaly_score < 35.0:
        return {
            "verdict": "REAL WEIGHT VERIFIED",
            "badge": "REAL / GENUINE WEIGHT",
            "badge_color": "#15803d",
            "border_color": "#22c55e",
            "bg_color": "#f0fdf4",
            "text_color": "#14532d",
            "icon": "🟢",
            "summary": "Weight is biologically genuine and consistent with anatomical morphometrics.",
            "detail": "Observed live scale weight aligns closely with height, body length, and chest girth. No indication of artificial water/salt loading or pre-sale weight inflation.",
            "trade_action": "Safe for fair-trade transaction at reported scale weight.",
        }
    elif anomaly_score < 65.0:
        return {
            "verdict": "QUESTIONABLE / REVIEW NEEDED",
            "badge": "QUESTIONABLE WEIGHT",
            "badge_color": "#b45309",
            "border_color": "#f59e0b",
            "bg_color": "#fffbeb",
            "text_color": "#78350f",
            "icon": "🟡",
            "summary": "Moderate divergence between skeletal frame and live scale weight.",
            "detail": "The physical dimensions indicate a weight moderately different from the scale reading. May be caused by recent heavy feeding/watering, pregnancy, or scale calibration drift.",
            "trade_action": "Recommend re-weighing the animal after a rest period or checking scale accuracy.",
        }
    else:
        return {
            "verdict": "SUSPECTED FAKE / INFLATED WEIGHT",
            "badge": "FAKE / ARTIFICIALLY INFLATED WEIGHT",
            "badge_color": "#b91c1c",
            "border_color": "#ef4444",
            "bg_color": "#fef2f2",
            "text_color": "#7f1d1d",
            "icon": "🔴",
            "summary": "High anomaly detected: Observed weight severely mismatches physical skeleton.",
            "detail": "Live scale reading is disproportionately heavier than what the anatomical frame can support. Strong commercial indicator of artificial pre-market water/salt bloating or weighing inaccuracy.",
            "trade_action": "Reject transaction at claimed scale weight. Negotiate based on image-estimated weight.",
        }


def save_uploaded_image(uploaded_file):
    if uploaded_file is None:
        return None

    temp_dir = ROOT / "temp_uploads"
    temp_dir.mkdir(exist_ok=True)

    try:
        raw_name = getattr(uploaded_file, "name", "goat_capture.png")
        stem = Path(raw_name).stem.replace(" ", "_")[:25]
        ext = Path(raw_name).suffix.lower()
        if ext not in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            ext = ".png"

        data = None
        if hasattr(uploaded_file, "getvalue"):
            data = uploaded_file.getvalue()
        elif hasattr(uploaded_file, "read"):
            uploaded_file.seek(0)
            data = uploaded_file.read()
        elif isinstance(uploaded_file, np.ndarray):
            with Image.fromarray(uploaded_file) as img:
                out = io.BytesIO()
                img.save(out, format="PNG")
                data = out.getvalue()

        if data is not None and len(data) > 0:
            file_hash = hashlib.md5(data).hexdigest()[:12]
            save_path = temp_dir / f"goat_{file_hash}_{stem}{ext}"
            if not save_path.exists() or save_path.stat().st_size != len(data):
                save_path.write_bytes(data)
            return str(save_path)

        file_id = uuid.uuid4().hex[:8]
        save_path = temp_dir / f"goat_{file_id}_{stem}{ext}"
        with Image.open(uploaded_file) as img:
            img.convert("RGB").save(save_path)
        return str(save_path)
    except Exception as err:
        st.error(f"Error saving image: {err}")
        return None


def build_report_pdf(report_df):
    if report_df is None or report_df.empty:
        return None

    buffer = io.BytesIO()
    page_width, page_height = landscape(A4)
    document = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=10 * mm,
        leftMargin=10 * mm,
        topMargin=10 * mm,
        bottomMargin=10 * mm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=16,
        textColor=colors.HexColor("#163b27"),
    )
    cell_style = ParagraphStyle(
        "ReportCell",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=7,
        leading=8,
        textColor=colors.HexColor("#102a1a"),
    )
    header_style = ParagraphStyle(
        "ReportHeader",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=7,
        leading=8,
        textColor=colors.white,
    )

    def clean_cell(val):
        return xml.sax.saxutils.escape(str(val)) if val is not None else ""

    table_data = [[Paragraph(clean_cell(column), header_style) for column in report_df.columns]]
    for row in report_df.itertuples(index=False, name=None):
        table_data.append([Paragraph(clean_cell(value), cell_style) for value in row])

    num_cols = len(report_df.columns)
    available_width = page_width - document.leftMargin - document.rightMargin
    col_width = available_width / max(num_cols, 1)
    column_widths = [col_width] * num_cols

    table = Table(table_data, colWidths=column_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#163b27")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#9aa9a0")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f5faf6")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#edf5ef")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    document.build([
        Paragraph("THE GOAT - History & Reports", title_style),
        Spacer(1, 6 * mm),
        table,
    ])
    return buffer.getvalue()



def get_base64_logo() -> str:
    if LOGO_PATH.exists():
        return base64.b64encode(LOGO_PATH.read_bytes()).decode()
    return ""


def render_splash_screen():
    logo_b64 = get_base64_logo()
    logo_img_tag = (
        f'<img src="data:image/png;base64,{logo_b64}" class="goat-splash-logo-img" alt="THE GOAT App" />'
        if logo_b64
        else '<span style="font-size: 3.5rem;">🐐</span>'
    )
    st.markdown(
        f"""
        <div id="goat-splash-screen" class="goat-splash-screen" onclick="this.style.opacity='0'; this.style.pointerEvents='none'; setTimeout(function(){{var s=document.getElementById('goat-splash-screen'); if(s) s.remove();}}, 300);">
            <div class="goat-splash-backdrop"></div>
            <div class="goat-splash-card">
                <div class="goat-splash-halo-wrap">
                    <div class="goat-splash-halo"></div>
                    <div class="goat-splash-halo goat-splash-halo-delay"></div>
                    <div class="goat-splash-avatar">
                        {logo_img_tag}
                    </div>
                </div>
                <div class="goat-splash-brand">
                    <h1 class="goat-splash-title">THE GOAT</h1>
                    <div class="goat-splash-pill">AI LIVESTOCK INTELLIGENCE</div>
                </div>
                <div class="goat-splash-progress-track">
                    <div class="goat-splash-progress-bar"></div>
                </div>
                <div class="goat-splash-footer">
                    <span class="goat-splash-chip">⚡ Fair-Weight Engine Ready</span>
                    <span class="goat-splash-ver">Mobile App Edition</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    st.markdown(
        """
        <style>
            /* Native mobile webview & touch smoothness (Median.co optimized) */
            * {
                -webkit-tap-highlight-color: transparent !important;
                -webkit-font-smoothing: antialiased;
            }
            button, input, select, textarea {
                touch-action: manipulation;
            }

            /* Base layout & high-contrast background */
            .main .block-container {
                padding-top: max(1.2rem, env(safe-area-inset-top));
                max-width: 1350px;
                padding-bottom: max(3rem, env(safe-area-inset-bottom));
                padding-left: max(1rem, env(safe-area-inset-left));
                padding-right: max(1rem, env(safe-area-inset-right));
            }
            .stApp { background: linear-gradient(180deg, #f3faf4 0%, #ebf4ee 100%); color: #0a1f12; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; -webkit-overflow-scrolling: touch; }

            /* Text styling - clear & high contrast */
            .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
                color: #092012 !important;
                font-weight: 800 !important;
            }
            .stApp p, .stApp label, [data-testid="stWidgetLabel"] {
                color: #0f2b19 !important;
                font-weight: 600 !important;
            }
            [data-testid="stCaptionContainer"] p, [data-testid="stCaptionContainer"] {
                color: #2d4a36 !important;
                font-weight: 500 !important;
            }

            /* ========================================================== */
            /* APP OPENING / SPLASH SCREEN ANIMATION (MEDIAN.CO READY)    */
            /* ========================================================== */
            .goat-splash-screen {
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                z-index: 9999999;
                display: flex;
                align-items: center;
                justify-content: center;
                background: radial-gradient(circle at 50% 45%, #184c2a 0%, #0d2817 65%, #07170c 100%);
                overflow: hidden;
                animation: goatSplashFadeOut 0.65s cubic-bezier(0.16, 1, 0.3, 1) 1.35s forwards;
                pointer-events: auto;
            }

            @keyframes goatSplashFadeOut {
                0% { opacity: 1; transform: scale(1); visibility: visible; }
                99% { opacity: 0; transform: scale(1.06); visibility: visible; }
                100% { opacity: 0; transform: scale(1.06); visibility: hidden; pointer-events: none; display: none; }
            }

            .goat-splash-backdrop {
                position: absolute;
                inset: -50%;
                background: radial-gradient(circle, rgba(34, 197, 94, 0.16) 0%, transparent 60%);
                animation: splashBackdropPulse 4s ease-in-out infinite alternate;
            }

            @keyframes splashBackdropPulse {
                from { transform: scale(0.9) rotate(0deg); opacity: 0.6; }
                to { transform: scale(1.2) rotate(15deg); opacity: 1; }
            }

            .goat-splash-card {
                position: relative;
                z-index: 2;
                display: flex;
                flex-direction: column;
                align-items: center;
                text-align: center;
                padding: 2rem 2.2rem;
                max-width: 360px;
                width: 90%;
                animation: splashCardEntrance 0.75s cubic-bezier(0.16, 1, 0.3, 1) both;
            }

            @keyframes splashCardEntrance {
                0% { opacity: 0; transform: translateY(22px) scale(0.92); }
                100% { opacity: 1; transform: translateY(0) scale(1); }
            }

            .goat-splash-halo-wrap {
                position: relative;
                width: 130px;
                height: 130px;
                display: flex;
                align-items: center;
                justify-content: center;
                margin-bottom: 1.1rem;
            }

            .goat-splash-halo {
                position: absolute;
                inset: -12px;
                border-radius: 50%;
                border: 2.5px solid rgba(74, 222, 128, 0.5);
                animation: splashHaloRipple 2s cubic-bezier(0.2, 0.8, 0.2, 1) infinite;
            }
            .goat-splash-halo-delay {
                animation-delay: 0.75s;
            }

            @keyframes splashHaloRipple {
                0% { transform: scale(0.85); opacity: 0.9; }
                100% { transform: scale(1.45); opacity: 0; }
            }

            .goat-splash-avatar {
                position: relative;
                width: 110px;
                height: 110px;
                border-radius: 50%;
                background: #ffffff;
                padding: 6px;
                box-shadow: 0 16px 36px rgba(0, 0, 0, 0.4), 0 0 25px rgba(34, 197, 94, 0.5);
                display: flex;
                align-items: center;
                justify-content: center;
                animation: splashAvatarBounce 2.5s ease-in-out infinite alternate;
            }

            @keyframes splashAvatarBounce {
                0% { transform: translateY(0) scale(1); }
                100% { transform: translateY(-5px) scale(1.02); }
            }

            .goat-splash-logo-img {
                width: 100%;
                height: 100%;
                border-radius: 50%;
                object-fit: contain;
                display: block;
            }

            .goat-splash-brand {
                margin-bottom: 1.1rem;
            }

            .goat-splash-title {
                color: #ffffff !important;
                font-size: 2.3rem !important;
                font-weight: 900 !important;
                letter-spacing: 0.08em !important;
                margin: 0.1rem 0 0.3rem 0 !important;
                text-shadow: 0 4px 18px rgba(0, 0, 0, 0.5);
                background: linear-gradient(135deg, #ffffff 30%, #86efac 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }

            .goat-splash-pill {
                display: inline-block;
                padding: 0.35rem 0.95rem;
                background: rgba(255, 255, 255, 0.12);
                border: 1px solid rgba(134, 239, 172, 0.35);
                border-radius: 999px;
                font-size: 0.72rem;
                font-weight: 800;
                letter-spacing: 0.12em;
                color: #bbf7d0 !important;
                text-transform: uppercase;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
            }

            .goat-splash-progress-track {
                width: 160px;
                height: 5px;
                background: rgba(255, 255, 255, 0.18);
                border-radius: 999px;
                overflow: hidden;
                margin-bottom: 0.9rem;
                box-shadow: inset 0 1px 3px rgba(0,0,0,0.3);
            }

            .goat-splash-progress-bar {
                width: 0%;
                height: 100%;
                background: linear-gradient(90deg, #22c55e 0%, #4ade80 50%, #ffffff 100%);
                border-radius: 999px;
                box-shadow: 0 0 12px #22c55e;
                animation: splashFillBar 1.25s cubic-bezier(0.4, 0, 0.2, 1) forwards;
            }

            @keyframes splashFillBar {
                0% { width: 0%; }
                40% { width: 55%; }
                100% { width: 100%; }
            }

            .goat-splash-footer {
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 0.3rem;
            }

            .goat-splash-chip {
                font-size: 0.8rem;
                color: #e2e8f0;
                font-weight: 600;
                letter-spacing: 0.02em;
            }

            .goat-splash-ver {
                font-size: 0.68rem;
                color: #94a3b8;
                font-weight: 500;
            }

            /* Sidebar styling - Dark Theme with high contrast */
            [data-testid="stSidebar"] {
                background: #182026 !important;
                border-right: 1px solid #2a353f !important;
            }
            [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4,
            [data-testid="stSidebar"] h5, [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] span {
                color: #f8fafc !important;
            }
            [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
                color: #94a3b8 !important;
            }

            /* Sidebar buttons */
            [data-testid="stSidebar"] [data-testid="stButton"] > button {
                background: #27313a !important;
                color: #ffffff !important;
                border: 1px solid #3e4c59 !important;
                border-radius: 10px !important;
                padding: 0.6rem 0.9rem !important;
                font-weight: 700 !important;
                text-align: left !important;
                justify-content: flex-start !important;
                box-shadow: none !important;
                transition: transform 0.15s ease, background 0.15s ease !important;
            }
            [data-testid="stSidebar"] [data-testid="stButton"] > button:hover {
                background: #364450 !important;
                color: #ffffff !important;
                border-color: #5c6f80 !important;
            }
            [data-testid="stSidebar"] [data-testid="stButton"] > button:active {
                transform: scale(0.97) !important;
            }
            [data-testid="stSidebar"] [data-testid="stButton"] > button p,
            [data-testid="stSidebar"] [data-testid="stButton"] > button span {
                color: #ffffff !important;
                font-weight: 700 !important;
            }
            [data-testid="stSidebar"] button[kind="primary"] {
                background: linear-gradient(135deg, #15803d 0%, #22c55e 100%) !important;
                border: 1px solid #4ade80 !important;
                color: #ffffff !important;
                box-shadow: 0 4px 12px rgba(34, 197, 94, 0.35) !important;
            }
            [data-testid="stSidebar"] button[kind="primary"] p {
                color: #ffffff !important;
                font-weight: 800 !important;
            }

            /* ========================================================== */
            /* UNIVERSAL BUTTONS IN MAIN CONTENT - CRISP WHITE TEXT       */
            /* ========================================================== */
            [data-testid="stButton"] > button,
            [data-testid="stDownloadButton"] > button,
            .stButton > button,
            .stDownloadButton > button {
                background: linear-gradient(135deg, #166534 0%, #15803d 100%) !important;
                color: #ffffff !important;
                border: 1px solid #22c55e !important;
                border-radius: 12px !important;
                padding: 0.75rem 1.2rem !important;
                font-weight: 700 !important;
                font-size: 0.96rem !important;
                box-shadow: 0 6px 16px rgba(22, 101, 52, 0.22) !important;
                transition: transform 0.16s ease, box-shadow 0.2s ease, background 0.2s ease !important;
            }
            [data-testid="stButton"] > button:hover,
            [data-testid="stDownloadButton"] > button:hover {
                background: linear-gradient(135deg, #14532d 0%, #166534 100%) !important;
                color: #ffffff !important;
                transform: translateY(-2px) !important;
                box-shadow: 0 8px 22px rgba(22, 101, 52, 0.35) !important;
            }
            [data-testid="stButton"] > button:active,
            [data-testid="stDownloadButton"] > button:active {
                transform: scale(0.96) !important;
                filter: brightness(0.93) !important;
            }

            /* Animated sheen on primary action buttons */
            [data-testid="stButton"] > button[kind="primary"],
            .stButton > button[kind="primary"] {
                background: linear-gradient(135deg, #15803d 0%, #16a34a 50%, #15803d 100%) !important;
                background-size: 200% 200% !important;
                animation: primaryBtnGleam 4s ease-in-out infinite alternate !important;
                box-shadow: 0 6px 18px rgba(34, 197, 94, 0.3) !important;
            }
            @keyframes primaryBtnGleam {
                0% { background-position: 0% 50%; }
                100% { background-position: 100% 50%; }
            }

            /* Explicitly override internal text wrappers so text is ALWAYS pure white */
            [data-testid="stButton"] > button p,
            [data-testid="stButton"] > button span,
            [data-testid="stButton"] > button div,
            [data-testid="stButton"] > button [data-testid="stMarkdownContainer"] p,
            [data-testid="stDownloadButton"] > button p,
            [data-testid="stDownloadButton"] > button span,
            [data-testid="stDownloadButton"] > button div,
            [data-testid="stDownloadButton"] > button [data-testid="stMarkdownContainer"] p {
                color: #ffffff !important;
                font-weight: 700 !important;
            }

            /* Disabled button state - clear readable contrast */
            [data-testid="stButton"] > button:disabled,
            .stButton > button:disabled {
                background: #e2e8f0 !important;
                color: #64748b !important;
                border: 1px solid #cbd5e1 !important;
                box-shadow: none !important;
                cursor: not-allowed !important;
                transform: none !important;
            }
            [data-testid="stButton"] > button:disabled p,
            [data-testid="stButton"] > button:disabled span,
            [data-testid="stButton"] > button:disabled [data-testid="stMarkdownContainer"] p {
                color: #64748b !important;
                font-weight: 600 !important;
            }

            /* Form Inputs & Selects with sleek focus transitions */
            .stTextInput input, .stNumberInput input, .stTextArea textarea {
                background: #ffffff !important;
                color: #0f172a !important;
                border: 1.5px solid #94a3b8 !important;
                border-radius: 10px !important;
                font-weight: 600 !important;
                padding: 0.55rem 0.8rem !important;
                transition: border-color 0.2s ease, box-shadow 0.2s ease, transform 0.15s ease !important;
            }
            .stTextInput input:focus, .stNumberInput input:focus {
                border-color: #16a34a !important;
                box-shadow: 0 0 0 3px rgba(34, 197, 94, 0.25) !important;
                transform: translateY(-1px) !important;
            }

            /* Selectbox styling (BaseWeb & React-Aria for Streamlit) */
            .stSelectbox [data-baseweb="select"] > div,
            [data-testid="stSelectbox"] div[role="group"] {
                background: #ffffff !important;
                color: #0f172a !important;
                border: 1.5px solid #94a3b8 !important;
                border-radius: 10px !important;
                font-weight: 600 !important;
                transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
            }
            [data-testid="stSelectbox"] div[role="group"]:focus-within {
                border-color: #16a34a !important;
                box-shadow: 0 0 0 3px rgba(34, 197, 94, 0.22) !important;
            }
            .stSelectbox [data-baseweb="select"] span,
            [data-testid="stSelectbox"] input {
                color: #0f172a !important;
                font-weight: 600 !important;
            }
            .stSelectbox svg,
            [data-testid="stSelectbox"] svg {
                fill: #0f172a !important;
                color: #0f172a !important;
            }
            [data-testid="stSelectbox"] button {
                background: transparent !important;
                border: none !important;
                box-shadow: none !important;
            }

            /* Number input plus/minus buttons */
            .stNumberInput button {
                background: #f1f5f9 !important;
                color: #0f172a !important;
                border-color: #cbd5e1 !important;
                transition: background 0.15s ease, transform 0.15s ease !important;
            }
            .stNumberInput button:active {
                transform: scale(0.92) !important;
            }
            .stNumberInput button svg {
                fill: #0f172a !important;
            }

            /* Slider */
            .stSlider [data-testid="stSliderValue"] {
                color: #15803d !important;
                font-weight: 800 !important;
            }

            /* File Uploader with interactive dropzone */
            [data-testid="stFileUploader"] {
                background: #ffffff !important;
                border: 2px dashed #16a34a !important;
                border-radius: 16px !important;
                padding: 1rem !important;
                transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease !important;
            }
            [data-testid="stFileUploader"]:hover {
                border-color: #22c55e !important;
                box-shadow: 0 6px 18px rgba(34, 197, 94, 0.12) !important;
            }
            [data-testid="stFileUploader"] section {
                background: #f8fafc !important;
                border-radius: 12px !important;
                padding: 0.9rem !important;
            }
            [data-testid="stFileUploader"] section button {
                background: #16a34a !important;
                color: #ffffff !important;
                border-radius: 8px !important;
                border: none !important;
                font-weight: 700 !important;
                transition: transform 0.15s ease !important;
            }
            [data-testid="stFileUploader"] section button:active {
                transform: scale(0.95) !important;
            }
            [data-testid="stFileUploader"] section button span,
            [data-testid="stFileUploader"] section button p {
                color: #ffffff !important;
                font-weight: 700 !important;
            }
            [data-testid="stFileUploader"] p,
            [data-testid="stFileUploader"] span,
            [data-testid="stFileUploader"] small {
                color: #0f172a !important;
                font-weight: 600 !important;
            }

            /* Tabs */
            .stTabs [data-baseweb="tab-list"] {
                gap: 0.5rem;
                background: #e2e8f0;
                padding: 0.3rem 0.5rem;
                border-radius: 12px;
            }
            .stTabs [data-baseweb="tab"] {
                color: #334155 !important;
                font-weight: 700 !important;
                border-radius: 8px !important;
                padding: 0.45rem 1rem !important;
                background: transparent !important;
                transition: background 0.2s ease, transform 0.15s ease !important;
            }
            .stTabs [data-baseweb="tab"]:active {
                transform: scale(0.97) !important;
            }
            .stTabs [aria-selected="true"] {
                background: #ffffff !important;
                color: #166534 !important;
                box-shadow: 0 4px 10px rgba(0,0,0,0.06) !important;
            }
            .stTabs [aria-selected="true"] p,
            .stTabs [aria-selected="true"] span {
                color: #166534 !important;
                font-weight: 800 !important;
            }

            /* Metric Cards with hover lift & active feedback */
            .metric-card {
                background: #ffffff;
                border: 1.5px solid #cbd5e1;
                border-radius: 18px;
                padding: 1.1rem 1.2rem;
                box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
                transition: transform 0.22s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.22s ease !important;
            }
            .metric-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 22px rgba(15, 23, 42, 0.08) !important;
            }
            .metric-card:active {
                transform: scale(0.98);
            }
            .metric-title {
                color: #475569 !important;
                font-size: 0.8rem;
                font-weight: 700;
                letter-spacing: 0.05em;
                text-transform: uppercase;
                margin-bottom: 0.3rem;
            }
            .metric-value {
                color: #0f172a !important;
                font-size: 1.85rem;
                font-weight: 800;
                margin: 0;
            }

            [data-testid="stMetric"] {
                background: #ffffff !important;
                border: 1.5px solid #cbd5e1 !important;
                border-radius: 16px !important;
                padding: 0.8rem 1rem !important;
                box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04) !important;
                transition: transform 0.22s cubic-bezier(0.4, 0, 0.2, 1), box-shadow 0.22s ease !important;
            }
            [data-testid="stMetric"]:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 20px rgba(15, 23, 42, 0.08) !important;
            }
            [data-testid="stMetric"]:active {
                transform: scale(0.98);
            }
            [data-testid="stMetricValue"] {
                color: #0f172a !important;
                font-weight: 800 !important;
                font-size: 1.6rem !important;
            }
            [data-testid="stMetricLabel"] {
                color: #475569 !important;
                font-size: 0.78rem !important;
                font-weight: 700 !important;
                letter-spacing: 0.05em !important;
                text-transform: uppercase !important;
            }

            /* Hero Banner with ambient gradient flow */
            .hero-panel {
                background: linear-gradient(135deg, #1b4d2a 0%, #2e7d4e 55%, #60af60 100%);
                background-size: 140% 140%;
                animation: heroAmbientShift 8s ease-in-out infinite alternate;
                border-radius: 26px; padding: 2rem 2.2rem; box-shadow: 0 16px 36px rgba(27, 73, 45, 0.2);
                margin-bottom: 1.4rem; border: 1px solid rgba(255,255,255,0.18);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }

            @keyframes heroAmbientShift {
                0% { background-position: 0% 50%; }
                100% { background-position: 100% 50%; }
            }

            .hero-panel h1 { color: #ffffff !important; font-size: 2.6rem; margin: 0.2rem 0 0.4rem 0; font-weight: 800; letter-spacing: 0.04em; }
            .hero-panel p { color: rgba(255,255,255,0.95) !important; font-size: 1rem; margin: 0; font-weight: 500 !important; }
            
            .hero-logo-img {
                animation: heroLogoFloat 3.5s ease-in-out infinite alternate;
                transition: transform 0.3s ease;
            }

            @keyframes heroLogoFloat {
                0% { transform: translateY(0px) rotate(0deg); }
                100% { transform: translateY(-4px) rotate(1deg); }
            }

            .badge { display: inline-block; padding: 0.4rem 0.8rem; background: rgba(255,255,255,0.18); border: 1px solid rgba(255,255,255,0.25); border-radius: 999px; font-size: 0.76rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: #ffffff !important; }
            .chip-row { display: flex; gap: 0.7rem; flex-wrap: wrap; margin: 1rem 0 0 0; }
            .chip {
                display: inline-block; background: rgba(255,255,255,0.18); border: 1px solid rgba(255,255,255,0.22);
                color: #ffffff !important; border-radius: 999px; padding: 0.4rem 0.85rem; font-size: 0.76rem; font-weight: 600;
                transition: transform 0.2s ease, background 0.2s ease, box-shadow 0.2s ease;
                cursor: default;
            }
            .chip:hover {
                transform: translateY(-2px);
                background: rgba(255,255,255,0.28) !important;
                box-shadow: 0 4px 10px rgba(0,0,0,0.12);
            }
            .chip:active {
                transform: scale(0.95);
            }

            /* Step Progress Bar Dynamic Animations */
            .step-progress-container {
                background: #ffffff;
                border: 1.5px solid #cbd5e1;
                border-radius: 14px;
                padding: 0.75rem 1rem;
                margin-bottom: 1.1rem;
                box-shadow: 0 2px 8px rgba(0,0,0,0.03);
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }
            .step-progress-bar-fill {
                background: linear-gradient(90deg, #16a34a 0%, #22c55e 50%, #4ade80 100%);
                background-size: 200% 100%;
                animation: progressShimmerWave 3s linear infinite;
                height: 100%;
                border-radius: 999px;
                transition: width 0.5s cubic-bezier(0.34, 1.56, 0.64, 1);
            }
            @keyframes progressShimmerWave {
                0% { background-position: 100% 0; }
                100% { background-position: -100% 0; }
            }
            .step-badge-pill {
                animation: stepBadgePop 0.4s cubic-bezier(0.16, 1, 0.3, 1);
            }
            @keyframes stepBadgePop {
                0% { transform: scale(0.85); opacity: 0.5; }
                100% { transform: scale(1); opacity: 1; }
            }

            /* Alerts */
            [data-testid="stAlert"] {
                border-radius: 14px !important;
                font-weight: 600 !important;
                animation: alertSlideIn 0.35s cubic-bezier(0.16, 1, 0.3, 1) both;
            }
            @keyframes alertSlideIn {
                0% { opacity: 0; transform: translateY(8px); }
                100% { opacity: 1; transform: translateY(0); }
            }
            [data-testid="stAlert"] p, [data-testid="stAlert"] div {
                color: #0f172a !important;
            }

            /* Result & Authenticity Cards with Breathing Aura */
            .result-summary-card {
                background: #ffffff;
                border: 1.5px solid #cbd5e1;
                border-radius: 20px;
                padding: 1.2rem 1.3rem;
                margin-bottom: 1rem;
                box-shadow: 0 8px 24px rgba(15, 45, 29, 0.06);
                color: #0f172a;
                animation: cardEntrance 0.45s cubic-bezier(0.16, 1, 0.3, 1) both;
            }
            @keyframes cardEntrance {
                0% { opacity: 0; transform: translateY(12px); }
                100% { opacity: 1; transform: translateY(0); }
            }
            .result-summary-header {
                font-size: 1.15rem;
                font-weight: 800;
                color: #0f172a;
                margin-bottom: 0.6rem;
                letter-spacing: 0.02em;
            }
            .result-summary-body {
                font-size: 0.98rem;
                line-height: 1.85;
                color: #0f172a;
            }
            .result-summary-body strong { color: #0f172a; }

            .weight-auth-card {
                transition: transform 0.25s ease, box-shadow 0.3s ease;
                animation: cardEntrance 0.5s cubic-bezier(0.16, 1, 0.3, 1) both;
            }
            .auth-badge-pulse {
                animation: authBadgePulse 2.2s ease-in-out infinite alternate;
                display: inline-block;
            }
            @keyframes authBadgePulse {
                0% { transform: scale(1); box-shadow: 0 0 0 rgba(0,0,0,0); }
                100% { transform: scale(1.04); box-shadow: 0 4px 12px rgba(0,0,0,0.16); }
            }

            /* Smooth Step Transition */
            .page-step-wrapper {
                animation: pageStepFadeIn 0.35s cubic-bezier(0.16, 1, 0.3, 1) both;
            }
            @keyframes pageStepFadeIn {
                0% { opacity: 0; transform: translateY(8px); }
                100% { opacity: 1; transform: translateY(0); }
            }

            /* Prevent Streamlit default navbar logo from showing or overflowing */
            [data-testid="stLogo"], .stLogo {
                display: none !important;
            }

            /* ========================================================== */
            /* MOBILE SPECIFIC RESPONSIVE OVERRIDES (screens <= 768px)    */
            /* ========================================================== */
            @media (max-width: 768px) {
                .main .block-container {
                    padding-top: max(0.4rem, env(safe-area-inset-top)) !important;
                    padding-left: max(0.75rem, env(safe-area-inset-left)) !important;
                    padding-right: max(0.75rem, env(safe-area-inset-right)) !important;
                    padding-bottom: max(2.5rem, env(safe-area-inset-bottom)) !important;
                }
                .hero-panel {
                    padding: 0.85rem 1rem !important;
                    border-radius: 18px !important;
                    margin-bottom: 0.85rem !important;
                    display: flex !important;
                    flex-direction: row !important;
                    align-items: center !important;
                    justify-content: space-between !important;
                    gap: 0.75rem !important;
                    flex-wrap: nowrap !important;
                }
                .hero-content {
                    min-width: 0 !important;
                    flex: 1 1 auto !important;
                }
                .hero-logo-container {
                    flex: 0 0 auto !important;
                    text-align: center !important;
                }
                .hero-panel h1 {
                    font-size: 1.55rem !important;
                    margin: 0.1rem 0 0.2rem 0 !important;
                }
                .hero-panel p {
                    font-size: 0.84rem !important;
                    line-height: 1.35 !important;
                    display: -webkit-box;
                    -webkit-line-clamp: 2;
                    -webkit-box-orient: vertical;
                    overflow: hidden;
                }
                .hero-logo-img {
                    width: 58px !important;
                    height: 58px !important;
                    min-width: 58px !important;
                    min-height: 58px !important;
                    border-width: 2px !important;
                    object-fit: contain !important;
                    display: block !important;
                }
                .badge {
                    font-size: 0.65rem !important;
                    padding: 0.22rem 0.55rem !important;
                }
                .chip-row {
                    gap: 0.35rem !important;
                    margin-top: 0.5rem !important;
                }
                .chip {
                    font-size: 0.68rem !important;
                    padding: 0.22rem 0.55rem !important;
                }

                /* Mobile multi-column wrapping (2x2 grid for metrics, side-by-side for buttons) */
                [data-testid="stHorizontalBlock"] {
                    flex-wrap: wrap !important;
                    gap: 0.5rem !important;
                }
                [data-testid="stHorizontalBlock"] > [data-testid="column"] {
                    flex: 1 1 calc(50% - 0.5rem) !important;
                    min-width: 130px !important;
                }

                .metric-card {
                    padding: 0.75rem 0.85rem !important;
                    border-radius: 14px !important;
                }
                .metric-title {
                    font-size: 0.72rem !important;
                }
                .metric-value {
                    font-size: 1.35rem !important;
                }
                [data-testid="stMetric"] {
                    padding: 0.65rem 0.8rem !important;
                    border-radius: 14px !important;
                }
                [data-testid="stMetricValue"] {
                    font-size: 1.3rem !important;
                }
                [data-testid="stMetricLabel"] {
                    font-size: 0.72rem !important;
                }

                /* Touch-friendly buttons (WCAG min touch target ~44px, no overflow) */
                [data-testid="stButton"] > button,
                [data-testid="stDownloadButton"] > button {
                    padding: 0.7rem 0.85rem !important;
                    font-size: 0.9rem !important;
                    min-height: 44px !important;
                    white-space: normal !important;
                    text-align: center !important;
                    word-break: break-word !important;
                }

                /* Mobile dataframes horizontal scroll */
                [data-testid="stDataFrame"] {
                    max-width: 100% !important;
                    overflow-x: auto !important;
                    -webkit-overflow-scrolling: touch !important;
                }

                [data-testid="stImage"] img,
                .stImage img {
                    max-width: 100% !important;
                    height: auto !important;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    logo_b64 = get_base64_logo()
    logo_img_html = (
        f'<img src="data:image/png;base64,{logo_b64}" class="hero-logo-img" style="width: 110px; height: 110px; border-radius: 50%; box-shadow: 0 10px 25px rgba(0,0,0,0.32); border: 3px solid rgba(255,255,255,0.65); background: #ffffff;" alt="THE GOAT Logo" />'
        if logo_b64
        else ""
    )
    st.markdown(
        f"""
        <div class="hero-panel" style="display: flex; align-items: center; justify-content: space-between; gap: 1rem; flex-wrap: wrap;">
            <div class="hero-content" style="flex: 1; min-width: 200px;">
                <div class="badge">AI Trading & Health System</div>
                <h1>THE GOAT</h1>
                <p>AI-Powered Fair-Weight Estimation & Temporary Anomaly Detection for Livestock Trading.</p>
                <div class="chip-row">
                    <span class="chip">Breed-aware</span>
                    <span class="chip">Mobile capture</span>
                    <span class="chip">Fair-trade alert</span>
                    <span class="chip">Real vs Fake Weight</span>
                </div>
            </div>
            <div class="hero-logo-container" style="flex-shrink: 0; text-align: center;">
                {logo_img_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_step_progress():
    step = st.session_state.current_step
    total = len(STEP_LABELS)
    pct = int(((step + 1) / total) * 100)
    current_label = STEP_LABELS[step]
    st.markdown(
        f"""
        <div class="step-progress-container">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.45rem; flex-wrap: wrap; gap: 0.3rem;">
                <span class="step-badge-pill" style="font-weight: 800; font-size: 0.88rem; color: #166534; letter-spacing: 0.02em;">
                    📍 Step {step + 1} of {total}: {current_label}
                </span>
                <span style="font-weight: 700; font-size: 0.78rem; color: #475569; background: #f1f5f9; padding: 0.2rem 0.6rem; border-radius: 999px;">
                    {pct}% Complete
                </span>
            </div>
            <div style="background: #e2e8f0; height: 7px; border-radius: 999px; overflow: hidden; box-shadow: inset 0 1px 2px rgba(0,0,0,0.06);">
                <div class="step-progress-bar-fill" style="width: {pct}%;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_navigation():
    with st.sidebar:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=105)
        st.markdown("<h2 style='margin: 0.2rem 0 0.1rem 0; font-size: 1.45rem; color: #ffffff !important; font-weight: 800;'>THE GOAT</h2>", unsafe_allow_html=True)
        st.caption("AI Fair-Weight Analyzer")
        st.markdown("---")

        # Quick Navigation Shortcuts
        side_c1, side_c2 = st.columns(2)
        with side_c1:
            st.button("🏠 Home", key="sidebar_nav_home", use_container_width=True, on_click=go_home, help="Go to Goat Profile")
        with side_c2:
            st.button("➕ New", key="sidebar_nav_new", use_container_width=True, on_click=start_new_assessment, help="Start New Goat Assessment")

        st.markdown("<p style='color: #cbd5e1 !important; font-size: 0.82rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; margin: 0.8rem 0 0.3rem 0;'>Workflow Steps</p>", unsafe_allow_html=True)
        for idx, step in enumerate(STEP_LABELS):
            if idx == st.session_state.current_step:
                st.button(step, key=f"nav_{idx}", use_container_width=True, type="primary")
            else:
                st.button(step, key=f"nav_{idx}", use_container_width=True, on_click=set_step, args=(idx,))


def render_profile_page():
    st.subheader("Goat profile")
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Goat ID", value=st.session_state.goat_id, key="widget_goat_id", on_change=sync_profile)
        st.text_input("Owner / Farmer Name", value=st.session_state.owner_name, key="widget_owner_name", on_change=sync_profile)
        st.selectbox("Breed", DEFAULT_BREEDS, index=DEFAULT_BREEDS.index(st.session_state.breed), key="widget_breed", on_change=sync_profile)
    with col2:
        st.slider("Age (months)", 2, 48, value=int(st.session_state.age_months), key="widget_age_months", on_change=sync_profile)
        st.selectbox("Sex", ["Male", "Female"], index=["Male", "Female"].index(st.session_state.sex), key="widget_sex", on_change=sync_profile)
        st.number_input("Observed / Live Weight (kg)", min_value=0.0, max_value=120.0, value=float(st.session_state.observed_weight), step=0.5, key="widget_observed_weight", on_change=sync_profile)

    st.info("Step 1 complete: goat profile is ready. Go to the next page to capture the goat image.")


def render_capture_page():
    st.subheader("Image Capture")
    st.write("Upload a goat image or capture a live photo with your camera to begin analysis.")

    tab_upload, tab_camera = st.tabs(["📁 Upload Image", "📷 Camera Capture"])

    with tab_upload:
        st.markdown("Upload a clear photo of the goat taken from the side profile.")
        uploaded_file = st.file_uploader(
            "Upload goat image from disk",
            type=["png", "jpg", "jpeg", "gif", "bmp", "webp"],
            key="upload_goat_file",
        )
        if uploaded_file is not None:
            saved_path = save_uploaded_image(uploaded_file)
            if saved_path and saved_path != st.session_state.get("image_path"):
                st.session_state.image_path = saved_path
                st.session_state.current_features = None
                st.session_state.analysis_result = None

    with tab_camera:
        st.markdown("Capture a live photo of the goat using your camera.")
        camera_photo = st.camera_input("Capture goat photo via camera", key="camera_goat_photo")
        if camera_photo is not None:
            saved_path = save_uploaded_image(camera_photo)
            if saved_path and saved_path != st.session_state.get("image_path"):
                st.session_state.image_path = saved_path
                st.session_state.current_features = None
                st.session_state.analysis_result = None

    if st.session_state.get("image_path") and Path(st.session_state.image_path).exists():
        st.success(f"Selected image: {Path(st.session_state.image_path).name}")
        try:
            with Image.open(st.session_state.image_path) as img:
                st.image(img.copy(), caption="Active goat image for analysis", width="stretch")
        except Exception as err:
            st.warning(f"Could not preview image: {err}")
        st.button("Continue to goat detection", use_container_width=True, type="primary", on_click=set_step, args=(2,))
    else:
        st.info("Please upload a goat image or capture one using your camera before continuing.")


def render_detection_page():
    st.subheader("Goat Detection & Segmentation")
    if not st.session_state.image_path or not Path(st.session_state.image_path).exists():
        st.warning("No image available. Go back to Image Capture and upload or capture a goat image first.")
        st.button("Go to Image Capture", on_click=set_step, args=(1,))
        return

    analyzer = GoatAnalyzer()
    col1, col2 = st.columns(2)
    with col1:
        try:
            with Image.open(st.session_state.image_path) as orig_image:
                st.image(orig_image.copy(), caption="Original Captured Image", width="stretch")
        except Exception as err:
            st.warning(f"Could not load image: {err}")
    with col2:
        try:
            annotated_img, _ = analyzer.segment_image(st.session_state.image_path)
            st.image(annotated_img, caption="Detected Silhouette & Bounding Box", width="stretch")
        except Exception:
            try:
                with Image.open(st.session_state.image_path) as orig_image:
                    st.image(orig_image.copy(), caption="Detected Subject", width="stretch")
            except Exception:
                pass

    st.success("Goat detection and segmentation complete. The subject has been isolated for body feature extraction.")
    st.markdown(
        """
        - **Foreground Separation:** GrabCut graph-cut isolation with border and ground suppression.
        - **Goat Silhouette Segmentation:** Body contour accurately isolated while excluding terrain and background.
        - **Bounding Geometry:** Physical width and height pixels extracted specifically for the goat body.
        """
    )


def render_feature_page():
    st.subheader("Body Feature Extraction")
    if not st.session_state.image_path:
        st.warning("Please complete the previous steps first.")
        return

    analyzer = GoatAnalyzer()
    try:
        features = analyzer._extract_features_from_image(st.session_state.image_path)
        st.session_state["current_features"] = features
    except Exception as exc:
        st.error(f"Feature extraction failed: {exc}")
        return

    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("Body length", st.session_state["current_features"]["body_length_cm"], "cm"),
        ("Body height", st.session_state["current_features"]["body_height_cm"], "cm"),
        ("Chest girth", st.session_state["current_features"]["chest_girth_cm"], "cm"),
        ("Body fullness", st.session_state["current_features"]["body_fullness"], "index"),
    ]

    for col, (label, value, unit) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">{label}</div>
                    <p class="metric-value">{value:.2f} {unit}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.caption(f"Shape index: {st.session_state['current_features']['aspect_ratio']:.3f}")


def render_weight_page():
    st.subheader("Breed-Aware Weight Estimation")
    if st.session_state.get("current_features") is None:
        st.warning("Complete feature extraction before estimating the weight.")
        return

    analyzer = GoatAnalyzer()
    features = st.session_state["current_features"]
    estimated_weight = analyzer.estimate_expected_weight(
        features,
        breed=st.session_state.breed,
        age_months=st.session_state.age_months,
        sex=st.session_state.sex,
    )

    lower = estimated_weight * 0.9
    upper = estimated_weight * 1.1
    observed_weight = float(st.session_state.get("observed_weight", 0.0))
    diff = observed_weight - estimated_weight
    dev_pct = abs(diff) / max(estimated_weight, 1.0) * 100

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Estimated weight", f"{estimated_weight:.2f} kg", help="Estimated purely from image silhouette & morphometrics")
    with col2:
        st.metric("Expected range", f"{lower:.2f} - {upper:.2f} kg", help="±10% biological confidence interval")
    with col3:
        st.metric("Observed weight", f"{observed_weight:.2f} kg", help="Live scale weight recorded in profile")
    with col4:
        st.metric("Weight difference", f"{diff:+.2f} kg", delta=f"{dev_pct:.1f}% deviation", delta_color="inverse" if dev_pct > 20 else "normal")

    st.write(f"Breed: {st.session_state.breed} | Age: {st.session_state.age_months} months | Sex: {st.session_state.sex}")

    st.markdown("---")
    st.markdown("### 🔍 Image-Estimated vs Observed Weight Comparison")

    comp_col1, comp_col2 = st.columns(2)
    with comp_col1:
        st.markdown(
            f"""
            <div style="background: #ffffff; padding: 1.3rem; border-radius: 16px; border: 1.5px solid #cbd5e1; box-shadow: 0 4px 14px rgba(0,0,0,0.04);">
                <h4 style="margin-top:0; color: #0f172a; font-weight: 800;">Visual Morphometrics (Image)</h4>
                <p style="margin: 0.35rem 0; color: #0f172a;"><strong>Body Length:</strong> {features.get('body_length_cm', 0.0):.1f} cm</p>
                <p style="margin: 0.35rem 0; color: #0f172a;"><strong>Body Height:</strong> {features.get('body_height_cm', 0.0):.1f} cm</p>
                <p style="margin: 0.35rem 0; color: #0f172a;"><strong>Chest Girth:</strong> {features.get('chest_girth_cm', 0.0):.1f} cm</p>
                <p style="margin: 0.35rem 0; color: #0f172a;"><strong>Fullness (Condition):</strong> {features.get('body_fullness', 0.0):.3f}</p>
                <p style="margin: 0.4rem 0 0 0; font-size: 0.84rem; color: #475569;">Derived from image via veterinary Schaeffer's metric formula</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with comp_col2:
        if dev_pct <= 15.0:
            status_title = "✅ Consistent Weight (Fair Trade Match)"
            status_desc = f"Observed live weight ({observed_weight:.2f} kg) aligns closely with visual body estimation ({estimated_weight:.2f} kg). Deviation is within normal biological tolerance ({dev_pct:.1f}%)."
            card_border = "#16a34a"
            card_bg = "#f0fdf4"
            text_c = "#14532d"
        elif observed_weight > upper:
            status_title = "⚠️ Weight Anomaly: Higher than Visual Frame"
            status_desc = f"Observed live weight ({observed_weight:.2f} kg) is {diff:+.2f} kg higher than expected from visual body dimensions ({estimated_weight:.2f} kg, +{dev_pct:.1f}%). Check for artificial water/feed bloating."
            card_border = "#dc2626"
            card_bg = "#fef2f2"
            text_c = "#7f1d1d"
        else:
            status_title = "ℹ️ Lower than Visual Frame"
            status_desc = f"Observed live weight ({observed_weight:.2f} kg) is {diff:+.2f} kg lower than visual morphometrics indicate ({estimated_weight:.2f} kg, -{dev_pct:.1f}%). Subject may be lean or dehydrated."
            card_border = "#2563eb"
            card_bg = "#eff6ff"
            text_c = "#1e3a8a"

        st.markdown(
            f"""
            <div style="background: {card_bg}; padding: 1.3rem; border-radius: 16px; border: 2px solid {card_border}; box-shadow: 0 4px 14px rgba(0,0,0,0.04);">
                <h4 style="margin-top:0; color: {card_border}; font-weight: 800;">{status_title}</h4>
                <p style="margin: 0.35rem 0; color: {text_c};">{status_desc}</p>
                <p style="margin: 0.4rem 0 0 0; color: {text_c}; font-weight: 700;"><strong>Live Scale:</strong> {observed_weight:.2f} kg &nbsp;|&nbsp; <strong>Visual Target:</strong> {estimated_weight:.2f} kg</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_anomaly_page():
    st.subheader("Temporary Weight Anomaly Detection")
    if st.session_state.get("current_features") is None:
        st.warning("Body feature extraction must be completed first.")
        return

    analyzer = GoatAnalyzer()
    features = st.session_state["current_features"]
    estimated_weight = analyzer.estimate_expected_weight(
        features,
        breed=st.session_state.breed,
        age_months=st.session_state.age_months,
        sex=st.session_state.sex,
    )
    deviation_pct = abs(st.session_state.observed_weight - estimated_weight) / max(estimated_weight, 1.0) * 100
    anomaly_score = analyzer._anomaly_score(estimated_weight, st.session_state.observed_weight, features["body_fullness"])

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Observed weight", f"{st.session_state.observed_weight:.2f} kg")
    with m_col2:
        st.metric("Estimated weight", f"{estimated_weight:.2f} kg")
    with m_col3:
        st.metric("Deviation", f"{deviation_pct:.2f}%")
    with m_col4:
        st.metric("Temporary Weight Anomaly Score", f"{anomaly_score:.2f}/100")

    # Real vs Fake Weight Verification Card
    auth = get_weight_authenticity(anomaly_score, deviation_pct)
    st.markdown(
        f"""
        <div class="weight-auth-card" style="background: {auth['bg_color']}; border: 2px solid {auth['border_color']}; border-radius: 18px; padding: 1.4rem; margin: 1.2rem 0; box-shadow: 0 4px 16px rgba(0,0,0,0.05);">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.6rem;">
                <h4 style="margin: 0; color: {auth['text_color']}; font-weight: 800; font-size: 1.2rem;">{auth['icon']} Weight Verification: {auth['badge']}</h4>
                <span class="auth-badge-pulse" style="background: {auth['badge_color']}; color: #ffffff; padding: 0.35rem 0.85rem; border-radius: 999px; font-weight: 700; font-size: 0.82rem;">{auth['verdict']}</span>
            </div>
            <p style="margin: 0.4rem 0; color: {auth['text_color']}; font-weight: 700; font-size: 1.02rem;">{auth['summary']}</p>
            <p style="margin: 0.3rem 0; color: #334155; font-size: 0.94rem; line-height: 1.6;">{auth['detail']}</p>
            <div style="margin-top: 0.8rem; padding-top: 0.7rem; border-top: 1px dashed {auth['border_color']}; font-weight: 700; color: {auth['text_color']}; font-size: 0.94rem;">
                💡 Trade Guidance: {auth['trade_action']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if anomaly_score >= 65:
        st.warning("High temporary anomaly risk: body fullness and observed weight differ significantly from the estimated fair-trade weight.")
    elif anomaly_score >= 35:
        st.warning("Moderate anomaly risk: re-check the goat condition before finalizing trade.")
    else:
        st.success("Fair-trade range is acceptable based on body condition and weight deviation.")

    # Dedicated Return to Home / Next actions
    anom_nav1, anom_nav2 = st.columns([1, 2])
    with anom_nav1:
        st.button("🏠 Return to Home", on_click=go_home, key="anomaly_btn_home", use_container_width=True)
    with anom_nav2:
        st.button("Continue to Fair-Trade Decision →", on_click=set_step, args=(6,), type="primary", key="anomaly_btn_result", use_container_width=True)


def render_result_page():
    st.subheader("Result & Fair-Trade Analysis")
    if not st.session_state.get("image_path") or not Path(st.session_state.image_path).exists():
        st.warning("Goat image is required to complete fair-trade analysis. Please upload or capture an image in Step 1.")
        st.button("Go to Image Capture", on_click=set_step, args=(1,), type="primary")
        return

    analyzer = GoatAnalyzer()
    try:
        result = analyzer.analyze_goat(
            image_path=st.session_state.image_path,
            breed=str(st.session_state.breed),
            age_months=int(st.session_state.age_months),
            sex=str(st.session_state.sex),
            observed_weight_kg=float(st.session_state.observed_weight),
            goat_id=str(st.session_state.goat_id),
            owner_name=str(st.session_state.owner_name),
        )
        st.session_state.analysis_result = result

        # Maintain history list
        history_entry = {
            "Goat ID": str(result.goat_id),
            "Owner": str(result.owner_name),
            "Breed": str(st.session_state.breed),
            "Age (mo)": int(st.session_state.age_months),
            "Sex": str(st.session_state.sex),
            "Observed (kg)": round(float(st.session_state.observed_weight), 2),
            "Estimated (kg)": round(float(result.estimated_weight_kg), 2),
            "Expected Range (kg)": f"{result.expected_range_kg[0]:.2f} - {result.expected_range_kg[1]:.2f}",
            "Deviation (%)": round(float(result.deviation_pct), 2),
            "Anomaly Score": round(float(result.anomaly_score), 2),
            "Health Status": str(result.health_label),
            "Fair-Trade Status": str(result.fair_trade_status),
        }
        if "history_list" not in st.session_state or not isinstance(st.session_state.history_list, list):
            st.session_state.history_list = []
        existing_idx = next(
            (i for i, h in enumerate(st.session_state.history_list) if h.get("Goat ID") == result.goat_id),
            None,
        )
        if existing_idx is not None:
            st.session_state.history_list[existing_idx] = history_entry
        else:
            st.session_state.history_list.append(history_entry)
    except Exception as err:
        st.error(f"Error performing fair-trade analysis: {err}")
        return

    col1, col2 = st.columns([1.1, 1])
    with col1:
        try:
            with Image.open(st.session_state.image_path) as orig_img:
                st.image(orig_img.copy(), caption="Final goat image", width="stretch")
        except Exception as img_err:
            st.warning(f"Could not load image preview: {img_err}")
    with col2:
        health_color = "#1b7e3e" if "Healthy" in result.health_label else "#c0392b"
        status_color = "#1b7e3e" if result.fair_trade_status == "Fair trade" else ("#d35400" if result.fair_trade_status == "Review needed" else "#c0392b")
        st.markdown(
            f"""
            <div class="result-summary-card">
                <div class="result-summary-header">Final Goat Assessment</div>
                <div class="result-summary-body">
                    <div><strong>Goat ID:</strong> {result.goat_id}</div>
                    <div><strong>Owner / Farmer:</strong> {result.owner_name}</div>
                    <div><strong>Health Classification:</strong> <span style="font-weight:700; color:{health_color};">{result.health_label} ({result.confidence * 100:.1f}%)</span></div>
                    <div><strong>Fair-Trade Decision:</strong> <span style="font-weight:700; color:{status_color};">{result.fair_trade_status}</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.metric("Estimated weight", f"{result.estimated_weight_kg:.2f} kg")
        st.metric("Expected range", f"{result.expected_range_kg[0]:.2f} - {result.expected_range_kg[1]:.2f} kg")
        st.metric("Deviation", f"{result.deviation_pct:.2f}%")
        st.metric("Temporary Weight Anomaly Score", f"{result.anomaly_score:.2f}/100")

    if result.anomaly_score >= 65:
        st.warning("High temporary anomaly risk. Trade should be reviewed carefully.")
    elif result.anomaly_score >= 35:
        st.warning("Moderate anomaly risk. Re-check body condition and live weight.")
    else:
        st.success("Fair trade range is acceptable.")

    # Weight Authenticity Card (Real vs Fake Weight)
    auth = get_weight_authenticity(result.anomaly_score, result.deviation_pct)
    st.markdown(
        f"""
        <div class="weight-auth-card" style="background: {auth['bg_color']}; border: 2px solid {auth['border_color']}; border-radius: 18px; padding: 1.4rem; margin: 1.2rem 0; box-shadow: 0 4px 16px rgba(0,0,0,0.05);">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.6rem;">
                <h4 style="margin: 0; color: {auth['text_color']}; font-weight: 800; font-size: 1.2rem;">{auth['icon']} Weight Authenticity Result: {auth['badge']}</h4>
                <span class="auth-badge-pulse" style="background: {auth['badge_color']}; color: #ffffff; padding: 0.35rem 0.85rem; border-radius: 999px; font-weight: 700; font-size: 0.82rem;">{auth['verdict']}</span>
            </div>
            <p style="margin: 0.4rem 0; color: {auth['text_color']}; font-weight: 700; font-size: 1.02rem;">{auth['summary']}</p>
            <p style="margin: 0.3rem 0; color: #334155; font-size: 0.94rem; line-height: 1.6;">{auth['detail']}</p>
            <div style="margin-top: 0.8rem; padding-top: 0.7rem; border-top: 1px dashed {auth['border_color']}; font-weight: 700; color: {auth['text_color']}; font-size: 0.94rem;">
                ⚖️ Fair-Trade Recommendation: <strong>{result.fair_trade_status}</strong> &nbsp;|&nbsp; 💡 Commercial Advice: {auth['trade_action']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Action buttons: Return to Home, Start New Assessment, or View History
    st.markdown("<h4 style='color: #0f2b19; font-weight: 800; margin-top: 1.2rem;'>Assessment Next Actions</h4>", unsafe_allow_html=True)
    res_btn1, res_btn2, res_btn3 = st.columns(3)
    with res_btn1:
        st.button("🏠 Return to Home", on_click=go_home, key="res_action_home", use_container_width=True, help="Return to Goat Profile")
    with res_btn2:
        st.button("➕ Start New Assessment", on_click=start_new_assessment, key="res_action_new", use_container_width=True, help="Assess another goat")
    with res_btn3:
        st.button("📊 View History & Reports →", on_click=set_step, args=(7,), key="res_action_hist", type="primary", use_container_width=True)


def render_history_page():
    st.subheader("History & Reports")

    records = list(st.session_state.get("history_list", []))

    if not records and st.session_state.get("analysis_result") is not None:
        result = st.session_state.analysis_result
        entry = {
            "Goat ID": str(result.goat_id),
            "Owner": str(result.owner_name),
            "Breed": str(st.session_state.breed),
            "Age (mo)": int(st.session_state.age_months),
            "Sex": str(st.session_state.sex),
            "Observed (kg)": round(float(st.session_state.observed_weight), 2),
            "Estimated (kg)": round(float(result.estimated_weight_kg), 2),
            "Expected Range (kg)": f"{result.expected_range_kg[0]:.2f} - {result.expected_range_kg[1]:.2f}",
            "Deviation (%)": round(float(result.deviation_pct), 2),
            "Anomaly Score": round(float(result.anomaly_score), 2),
            "Health Status": str(result.health_label),
            "Fair-Trade Status": str(result.fair_trade_status),
        }
        records = [entry]
        st.session_state.history_list = records

    if not records:
        if st.session_state.get("image_path") and Path(st.session_state.image_path).exists():
            st.info("No report generated yet. Click below to run analysis and generate reports.")
            if st.button("Generate Assessment Report", type="primary", use_container_width=True):
                set_step(6)
                st.rerun()
        else:
            st.info("No report generated yet. Complete the workflow to generate the goat assessment report.")
            st.button("Go to Image Capture", on_click=set_step, args=(1,), use_container_width=True)
        return

    report_df = pd.DataFrame(records)
    st.dataframe(report_df, width="stretch", hide_index=True)

    col1, col2 = st.columns(2)
    current_id = records[-1].get("Goat ID", "assessment")

    with col1:
        csv_data = report_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download complete report as CSV",
            data=csv_data,
            file_name=f"goat_report_{current_id}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col2:
        try:
            pdf_data = build_report_pdf(report_df)
            if pdf_data is not None:
                st.download_button(
                    "Download complete report as PDF",
                    data=pdf_data,
                    file_name=f"goat_report_{current_id}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            else:
                st.warning("Could not generate PDF: table is empty.")
        except Exception as pdf_err:
            st.error(f"Error generating PDF report: {pdf_err}")

    st.markdown("---")
    hist_act1, hist_act2 = st.columns(2)
    with hist_act1:
        st.button("🏠 Return to Home", on_click=go_home, key="hist_action_home", use_container_width=True)
    with hist_act2:
        st.button("➕ Assess Another Goat", on_click=start_new_assessment, key="hist_action_new", type="primary", use_container_width=True)


def main():
    init_session_state()
    if not st.session_state.get("splash_shown", False):
        st.session_state.splash_shown = True
        render_splash_screen()
    render_header()
    render_navigation()
    render_step_progress()

    st.write("")
    step = st.session_state.current_step

    if step == 0:
        render_profile_page()
    elif step == 1:
        render_capture_page()
    elif step == 2:
        render_detection_page()
    elif step == 3:
        render_feature_page()
    elif step == 4:
        render_weight_page()
    elif step == 5:
        render_anomaly_page()
    elif step == 6:
        render_result_page()
    else:
        render_history_page()

    st.write("")
    if st.session_state.current_step > 0:
        col_back, col_home = st.columns(2)
        with col_back:
            st.button("Back", use_container_width=True, on_click=go_back)
        with col_home:
            st.button("🏠 Return to Home", use_container_width=True, on_click=go_home, key="bottom_return_home")

    if st.session_state.current_step < len(STEP_LABELS) - 1:
        next_allowed = can_access_step(st.session_state.current_step + 1)
        if next_allowed:
            st.button("Next", use_container_width=True, on_click=go_next)
        else:
            st.button("Next (requires valid image)", use_container_width=True, disabled=True)
    else:
        st.button("🏠 Start New Assessment", use_container_width=True, on_click=start_new_assessment, type="primary", key="bottom_new_assessment")
if __name__ == "__main__":
    main()