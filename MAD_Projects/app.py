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

if hasattr(st, "logo") and LOGO_PATH.exists():
    st.logo(str(LOGO_PATH))

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


def render_header():
    st.markdown(
        """
        <style>
            .main .block-container { padding-top: 1rem; max-width: 1350px; }
            .stApp { background: linear-gradient(180deg, #f5fdf2 0%, #eef5ff 100%); color: #102a1a; }
            .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
            .stApp p, .stApp li, .stApp label, .stApp [data-testid="stWidgetLabel"],
            .stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stCaptionContainer"] {
                color: #102a1a !important;
            }
            .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
                font-weight: 800 !important;
            }
            [data-testid="stSidebar"] { background: #242832; }
            [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
            [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p,
            [data-testid="stSidebar"] label, [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { color: #ffffff !important; }
            .stTextInput input, .stNumberInput input, .stSelectbox [data-baseweb="select"] > div,
            .stTextArea textarea { background: #ffffff !important; color: #102a1a !important; }
            .stSelectbox [data-baseweb="select"] span { color: #102a1a !important; }
            .stSlider [data-testid="stSlider"] { color: #102a1a !important; }
            .stSlider [data-testid="stSliderValue"] { color: #b42318 !important; font-weight: 800 !important; }
            [data-testid="stAlert"] p, [data-testid="stAlert"] div,
            [data-testid="stFileUploader"] label, [data-testid="stCameraInput"] label {
                color: #102a1a !important;
            }
            [data-testid="stDataFrame"] { color: #102a1a !important; }
            .hero-panel {
                background: linear-gradient(135deg, #1d4d2a 0%, #2e7d4e 55%, #7cc57c 100%);
                border-radius: 26px; padding: 2rem 2.2rem; box-shadow: 0 18px 40px rgba(27, 73, 45, 0.18);
                margin-bottom: 1.2rem; border: 1px solid rgba(255,255,255,0.12);
            }
            .hero-panel h1 { color: #ffffff !important; font-size: 2.6rem; margin: 0.2rem 0 0.4rem 0; font-weight: 800; letter-spacing: 0.04em; }
            .hero-panel p { color: rgba(255,255,255,0.92) !important; font-size: 1rem; margin: 0; }
            .badge { display: inline-block; padding: 0.4rem 0.8rem; background: rgba(255,255,255,0.14); border: 1px solid rgba(255,255,255,0.2); border-radius: 999px; font-size: 0.76rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: #f5fff5 !important; }
            .panel { background: rgba(255,255,255,0.7); border: 1px solid rgba(21, 46, 31, 0.08); border-radius: 22px; padding: 1.25rem; box-shadow: 0 12px 30px rgba(31, 65, 42, 0.08); backdrop-filter: blur(8px); }
            .metric-card { background: linear-gradient(180deg, #ffffff 0%, #f9fbfb 100%); border: 1px solid rgba(20, 48, 34, 0.08); border-radius: 20px; padding: 1rem 1.1rem; text-align: left; box-shadow: 0 8px 24px rgba(15, 26, 18, 0.05); height: 100%; }
            .metric-title { color: #000000; font-size: 0.8rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; margin-bottom: 0.4rem; }
            .metric-value { font-size: 1.9rem; font-weight: 800; color: #000000; margin: 0; }
            .status-box {
                border-radius: 18px;
                padding: 1rem 1.1rem;
                margin-top: 0.9rem;
                border: 1px solid rgba(21, 46, 31, 0.08);
                background: linear-gradient(180deg, #f7fbf7 0%, #eefaf0 100%);
                color: #000000;
                font-weight: 600;
                line-height: 1.7;
                box-shadow: 0 8px 20px rgba(18, 42, 28, 0.05);
            }
            [data-testid="stMetricValue"] {
                color: #000000 !important;
                font-weight: 800 !important;
                font-size: 1.65rem !important;
            }
            [data-testid="stMetricLabel"] {
                color: #000000 !important;
                font-size: 0.74rem !important;
                font-weight: 700 !important;
                letter-spacing: 0.06em !important;
                text-transform: uppercase !important;
            }
            .stMetric { background: rgba(255,255,255,0.8); border-radius: 16px; padding: 0.35rem 0.5rem; }
            .result-summary-card {
                background: linear-gradient(180deg, #ffffff 0%, #f3faf5 100%);
                border: 1px solid rgba(10, 45, 32, 0.12);
                border-radius: 20px;
                padding: 1.1rem 1.2rem;
                margin-bottom: 1rem;
                box-shadow: 0 14px 28px rgba(15, 45, 29, 0.06);
                color: #18372d;
            }
            .result-summary-header {
                font-size: 1.05rem;
                font-weight: 800;
                color: #000000;
                margin-bottom: 0.6rem;
                letter-spacing: 0.02em;
            }
            .result-summary-body {
                font-size: 0.98rem;
                line-height: 1.8;
                color: #000000;
            }
            .result-summary-body strong { color: #000000; }
            [data-testid="stAlert"] {
                font-size: 1rem !important;
                line-height: 1.6 !important;
                color: #000000 !important;
            }
            [data-testid="stButton"] > button { width: 100%; border-radius: 14px; background: linear-gradient(135deg, #1d4d2a 0%, #2d8e57 100%); color: white; border: none; font-weight: 700; padding: 0.8rem 1rem; box-shadow: 0 12px 24px rgba(31, 116, 75, 0.22); }
            [data-testid="stButton"] > button:hover { transform: translateY(-1px); box-shadow: 0 14px 28px rgba(31, 116, 75, 0.28); }
            .chip-row { display: flex; gap: 0.7rem; flex-wrap: wrap; margin: 1rem 0 0 0; }
            .chip { display: inline-block; background: rgba(255,255,255,0.14); border: 1px solid rgba(255,255,255,0.18); color: #ffffff !important; border-radius: 999px; padding: 0.45rem 0.9rem; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.02em; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    logo_b64 = get_base64_logo()
    logo_img_html = (
        f'<img src="data:image/png;base64,{logo_b64}" style="width: 110px; height: 110px; border-radius: 50%; box-shadow: 0 10px 25px rgba(0,0,0,0.32); border: 3px solid rgba(255,255,255,0.65); background: #ffffff;" alt="THE GOAT Logo" />'
        if logo_b64
        else ""
    )
    st.markdown(
        f"""
        <div class="hero-panel" style="display: flex; align-items: center; justify-content: space-between; gap: 1.5rem; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 250px;">
                <div class="badge">AI Trading & Health System</div>
                <h1>THE GOAT</h1>
                <p>AI-Powered Fair-Weight Estimation & Temporary Anomaly Detection for Livestock Trading.</p>
                <div class="chip-row">
                    <span class="chip">Breed-aware</span>
                    <span class="chip">Mobile capture</span>
                    <span class="chip">Fair-trade alert</span>
                </div>
            </div>
            <div style="flex-shrink: 0; text-align: center;">
                {logo_img_html}
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
        st.markdown("##### Workflow")
        for idx, step in enumerate(STEP_LABELS):
            if idx == st.session_state.current_step:
                st.button(step, key=f"nav_{idx}", use_container_width=True, type="primary")
            else:
                st.button(step, key=f"nav_{idx}", use_container_width=True, on_click=set_step, args=(idx,))


def render_profile_page():
    st.markdown(
        """
        <style>
        .stTextInput label, .stNumberInput label, .stSelectbox label, .stSlider label, .stInfo {
            color: #000000 !important;
        }
        .stTextInput input, .stNumberInput input, .stSelectbox div, .stSlider [data-testid="stBaseSlider"] {
            color: #000000 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
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
                st.image(img.copy(), caption="Active goat image for analysis", width=420)
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
                st.image(orig_image.copy(), caption="Original Captured Image", use_container_width=True)
        except Exception as err:
            st.warning(f"Could not load image: {err}")
    with col2:
        try:
            annotated_img, _ = analyzer.segment_image(st.session_state.image_path)
            st.image(annotated_img, caption="Detected Silhouette & Bounding Box", use_container_width=True)
        except Exception:
            try:
                with Image.open(st.session_state.image_path) as orig_image:
                    st.image(orig_image.copy(), caption="Detected Subject", use_container_width=True)
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
            <div style="background: rgba(255,255,255,0.06); padding: 1.2rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.12);">
                <h4 style="margin-top:0;">Visual Morphometrics (Image)</h4>
                <p style="margin: 0.3rem 0;"><strong>Body Length:</strong> {features.get('body_length_cm', 0.0):.1f} cm</p>
                <p style="margin: 0.3rem 0;"><strong>Body Height:</strong> {features.get('body_height_cm', 0.0):.1f} cm</p>
                <p style="margin: 0.3rem 0;"><strong>Chest Girth:</strong> {features.get('chest_girth_cm', 0.0):.1f} cm</p>
                <p style="margin: 0.3rem 0;"><strong>Fullness (Condition):</strong> {features.get('body_fullness', 0.0):.3f}</p>
                <p style="margin: 0.3rem 0; font-size: 0.85rem; color: #888;">Derived from image via veterinary Schaeffer's metric formula</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with comp_col2:
        if dev_pct <= 15.0:
            status_title = "✅ Consistent Weight (Fair Trade)"
            status_desc = f"Observed live weight ({observed_weight:.2f} kg) aligns closely with the visual body estimation ({estimated_weight:.2f} kg). Deviation is within acceptable commercial tolerance ({dev_pct:.1f}%)."
            card_border = "#27ae60"
        elif observed_weight > upper:
            status_title = "⚠️ Weight Anomaly: Higher than Visual Frame"
            status_desc = f"Observed live weight ({observed_weight:.2f} kg) is {diff:+.2f} kg higher than expected from visual body dimensions ({estimated_weight:.2f} kg, +{dev_pct:.1f}%). Re-check for artificial water/feed bloating before trade."
            card_border = "#e67e22"
        else:
            status_title = "ℹ️ Lower than Visual Frame"
            status_desc = f"Observed live weight ({observed_weight:.2f} kg) is {diff:+.2f} kg lower than visual morphometrics indicate ({estimated_weight:.2f} kg, -{dev_pct:.1f}%). Subject may be lean or dehydrated."
            card_border = "#3498db"

        st.markdown(
            f"""
            <div style="background: rgba(255,255,255,0.06); padding: 1.2rem; border-radius: 12px; border: 1px solid {card_border};">
                <h4 style="margin-top:0; color: {card_border};">{status_title}</h4>
                <p style="margin: 0.3rem 0;">{status_desc}</p>
                <p style="margin: 0.3rem 0;"><strong>Live Scale:</strong> {observed_weight:.2f} kg | <strong>Visual Target:</strong> {estimated_weight:.2f} kg</p>
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

    st.metric("Observed weight", f"{st.session_state.observed_weight:.2f} kg")
    st.metric("Estimated weight", f"{estimated_weight:.2f} kg")
    st.metric("Deviation", f"{deviation_pct:.2f}%")
    st.metric("Temporary Weight Anomaly Score", f"{anomaly_score:.2f}/100")

    if anomaly_score >= 65:
        st.warning("High temporary anomaly risk: body fullness and observed weight differ significantly from the estimated fair-trade weight.")
    elif anomaly_score >= 35:
        st.warning("Moderate anomaly risk: re-check the goat condition before finalizing trade.")
    else:
        st.success("Fair-trade range is acceptable based on body condition and weight deviation.")


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
                st.image(orig_img.copy(), caption="Final goat image", width=500)
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
    st.dataframe(report_df, use_container_width=True, hide_index=True)

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


def main():
    init_session_state()
    render_header()
    render_navigation()

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
    col_back, col_next = st.columns([1, 4])
    with col_back:
        if st.session_state.current_step > 0:
            st.button("Back", use_container_width=True, on_click=go_back)
    with col_next:
        if st.session_state.current_step < len(STEP_LABELS) - 1:
            next_allowed = can_access_step(st.session_state.current_step + 1)
            if next_allowed:
                st.button("Next", use_container_width=True, on_click=go_next)
            else:
                st.button("Next (requires valid image)", use_container_width=True, disabled=True)
if __name__ == "__main__":
    main()