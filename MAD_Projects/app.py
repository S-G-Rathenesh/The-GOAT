import sys
import io
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

st.set_page_config(page_title="Goat Fair-Weight Analyzer", page_icon="🐐", layout="wide")

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
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def can_access_step(step_index):
    if step_index in (0, 1):
        return True

    if not st.session_state.image_path:
        return False

    if step_index in (2, 3):
        return True

    if step_index >= 4 and st.session_state.get("current_features") is None:
        return False

    return True


def load_dataset_summary():
    summary = []
    for folder_name in ["healthy_goat", "unhealthy_goat"]:
        folder = DATA_DIR / folder_name
        count = 0
        if folder.exists():
            count = len([p for p in folder.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".gif"}])
        summary.append({"class_name": folder_name.replace("_goat", " goat").title(), "count": count})
    return pd.DataFrame(summary)


def save_uploaded_image(uploaded_file):
    if uploaded_file is None:
        return None

    temp_dir = ROOT / "temp_uploads"
    temp_dir.mkdir(exist_ok=True)

    try:
        if isinstance(uploaded_file, np.ndarray):
            image = Image.fromarray(uploaded_file)
            save_path = temp_dir / "captured_image.png"
            image.save(save_path)
            return str(save_path)

        if hasattr(uploaded_file, "read"):
            uploaded_file.seek(0)
            data = uploaded_file.read()
            file_name = getattr(uploaded_file, "name", "captured_image.png")
            if not file_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                file_name = "captured_image.png"
            save_path = temp_dir / file_name
            save_path.write_bytes(data)
            return str(save_path)

        if hasattr(uploaded_file, "getvalue"):
            data = uploaded_file.getvalue()
            file_name = getattr(uploaded_file, "name", "captured_image.png")
            if not file_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                file_name = "captured_image.png"
            save_path = temp_dir / file_name
            save_path.write_bytes(data)
            return str(save_path)

        if hasattr(uploaded_file, "save"):
            file_name = "captured_image.png"
            save_path = temp_dir / file_name
            uploaded_file.save(save_path)
            return str(save_path)

        image = Image.open(uploaded_file)
        save_path = temp_dir / "captured_image.png"
        image.save(save_path)
        return str(save_path)
    except Exception:
        return None

def build_report_pdf(report_df):
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

    table_data = [[Paragraph(str(column), header_style) for column in report_df.columns]]
    for row in report_df.itertuples(index=False, name=None):
        table_data.append([Paragraph(str(value), cell_style) for value in row])

    column_widths = [
        18 * mm, 28 * mm, 18 * mm, 20 * mm, 14 * mm, 25 * mm,
        25 * mm, 29 * mm, 19 * mm, 20 * mm, 27 * mm,
    ]
    available_width = page_width - document.leftMargin - document.rightMargin
    width_scale = available_width / sum(column_widths)
    column_widths = [width * width_scale for width in column_widths]
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
        Paragraph("Goat Fair-Weight Analyzer - History & Reports", title_style),
        Spacer(1, 6 * mm),
        table,
    ])
    return buffer.getvalue()


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
            .hero-panel h1 { color: #ffffff !important; font-size: 2.5rem; margin: 0.2rem 0 0.5rem 0; font-weight: 800; }
            .hero-panel p { color: rgba(255,255,255,0.9) !important; font-size: 1rem; margin: 0; }
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
            .chip-row { display: flex; gap: 0.7rem; flex-wrap: wrap; margin: 1rem 0 1.2rem 0; }
            .chip { display: inline-block; background: rgba(255,255,255,0.14); border: 1px solid rgba(255,255,255,0.18); color: #ffffff !important; border-radius: 999px; padding: 0.45rem 0.9rem; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.02em; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="hero-panel">
            <div class="badge">AI Trading Assistant</div>
            <h1>Goat Fair-Weight Analyzer</h1>
            <p>Multi-step goat assessment workflow from profile registration to fair-trade decision support.</p>
            <div class="chip-row">
                <span class="chip">Breed-aware</span>
                <span class="chip">Mobile capture</span>
                <span class="chip">Fair-trade alert</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_navigation():
    with st.sidebar:
        st.title("Workflow")
        for idx, step in enumerate(STEP_LABELS):
            if idx == st.session_state.current_step:
                st.button(step, key=f"nav_{idx}", use_container_width=True, type="primary")
            else:
                if st.button(step, key=f"nav_{idx}", use_container_width=True):
                    st.session_state.current_step = idx


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
        st.session_state.goat_id = st.text_input("Goat ID", value=st.session_state.goat_id)
        st.session_state.owner_name = st.text_input("Owner / Farmer Name", value=st.session_state.owner_name)
        st.session_state.breed = st.selectbox("Breed", DEFAULT_BREEDS, index=DEFAULT_BREEDS.index(st.session_state.breed))
    with col2:
        st.session_state.age_months = st.slider("Age (months)", 2, 48, st.session_state.age_months)
        st.session_state.sex = st.selectbox("Sex", ["Male", "Female"], index=["Male", "Female"].index(st.session_state.sex))
        st.session_state.observed_weight = st.number_input("Observed / Live Weight (kg)", min_value=0.0, max_value=120.0, value=float(st.session_state.observed_weight), step=0.5)

    st.info("Step 1 complete: goat profile is ready. Go to the next page to capture the goat image.")


def render_capture_page():
    st.subheader("Image Capture")
    st.write("Capture multiple views of the goat using a smartphone camera or upload a photo from disk.")

    camera_photo = st.camera_input("Capture goat image", key="camera_input")
    uploaded_file = st.file_uploader("Upload goat image", type=["png", "jpg", "jpeg", "gif", "bmp"])

    current_image = None
    if camera_photo is not None:
        current_image = camera_photo
    elif uploaded_file is not None:
        current_image = uploaded_file

    if current_image is not None:
        image_path = save_uploaded_image(current_image)
        if image_path:
            st.session_state.image_path = image_path
            st.session_state.current_features = None
            st.image(current_image, caption="Captured goat image", width=600)
            st.success("Image captured successfully.")
            if st.button("Continue to goat detection", use_container_width=True):
                st.session_state.current_step = 2
        else:
            st.warning("The captured image could not be saved. Please try again or upload a different image.")
    else:
        st.warning("Please capture or upload a goat image before continuing.")


def render_detection_page():
    st.subheader("Goat Detection & Segmentation")
    if not st.session_state.image_path:
        st.warning("No image available. Go back to Image Capture and upload or capture a goat image first.")
        return

    image = Image.open(st.session_state.image_path)
    st.image(image, caption="Detected goat", width=600)
    st.success("Goat detection and segmentation are ready. The subject has been isolated for body analysis.")
    st.markdown(
        """
        - Background-normalized image prepared
        - Goat silhouette segmented for measurement
        - Body boundaries extracted for feature estimation
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

    st.metric("Estimated weight", f"{estimated_weight:.2f} kg")
    st.metric("Expected range", f"{lower:.2f} - {upper:.2f} kg")
    st.write(f"Breed: {st.session_state.breed} | Age: {st.session_state.age_months} months | Sex: {st.session_state.sex}")


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
    if not st.session_state.image_path:
        st.warning("Need a goat image to complete a result.")
        return

    analyzer = GoatAnalyzer()
    result = analyzer.analyze_goat(
        image_path=st.session_state.image_path,
        breed=st.session_state.breed,
        age_months=st.session_state.age_months,
        sex=st.session_state.sex,
        observed_weight_kg=st.session_state.observed_weight,
        goat_id=st.session_state.goat_id,
        owner_name=st.session_state.owner_name,
    )
    st.session_state.analysis_result = result

    col1, col2 = st.columns([1.1, 1])
    with col1:
        image = Image.open(st.session_state.image_path)
        st.image(image, caption="Final goat image", width=600)
    with col2:
        st.markdown(
            f"""
            <div class="result-summary-card">
                <div class="result-summary-header">Final Goat Result</div>
                <div class="result-summary-body">
                    <div><strong>Goat ID:</strong> {result.goat_id}</div>
                    <div><strong>Owner:</strong> {result.owner_name}</div>
                    <div><strong>Fair-trade status:</strong> {result.fair_trade_status}</div>
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
    if st.session_state.analysis_result is None:
        st.info("No report generated yet. Complete the workflow to generate the goat assessment report.")
        return

    result = st.session_state.analysis_result
    report_df = pd.DataFrame([
        {
            "Goat ID": result.goat_id,
            "Owner": result.owner_name,
            "Breed": st.session_state.breed,
            "Age (months)": st.session_state.age_months,
            "Sex": st.session_state.sex,
            "Observed weight (kg)": st.session_state.observed_weight,
            "Estimated weight (kg)": result.estimated_weight_kg,
            "Expected range (kg)": f"{result.expected_range_kg[0]:.2f} - {result.expected_range_kg[1]:.2f}",
            "Deviation (%)": result.deviation_pct,
            "Anomaly score": result.anomaly_score,
            "Fair-trade status": result.fair_trade_status,
        }
    ])
    st.dataframe(report_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download complete report as PDF",
        data=build_report_pdf(report_df),
        file_name=f"goat_report_{result.goat_id}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


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
            if st.button("Back", use_container_width=True):
                st.session_state.current_step -= 1
    with col_next:
        if st.session_state.current_step < len(STEP_LABELS) - 1:
            next_allowed = can_access_step(st.session_state.current_step + 1)
            if next_allowed:
                if st.button("Next", use_container_width=True):
                    st.session_state.current_step += 1
            else:
                st.button("Next (requires valid image)", use_container_width=True, disabled=True)
if __name__ == "__main__":
    main()