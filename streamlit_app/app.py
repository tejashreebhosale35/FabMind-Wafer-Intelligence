
from pathlib import Path

import sys
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import yaml
from PIL import Image


# ---------------------------------------------------------
# 1. PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
TEST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test.pkl"
CHECKPOINT_PATH = (
    PROJECT_ROOT / "outputs" / "models" / "wafer_vit_checkpoint_best.pth"
)
CSS_PATH = PROJECT_ROOT / "streamlit_app" / "styles.css"

# ---------------------------------------------------------
# 2. PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="FabMind | Wafer Intelligence",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# 3. LOAD CSS
# ---------------------------------------------------------

def load_css():
    if CSS_PATH.exists():
        st.markdown(
            f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )


load_css()


# ---------------------------------------------------------
# 4. LOAD CONFIGURATION
# ---------------------------------------------------------

@st.cache_data
def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


# ---------------------------------------------------------
# 5. LOAD PREDICTOR
# ---------------------------------------------------------

@st.cache_resource
def load_predictor():

    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    from ml.predict import WaferPredictor

    return WaferPredictor(
        config_path=str(CONFIG_PATH),
        checkpoint_path=str(CHECKPOINT_PATH),
    )

# ---------------------------------------------------------
# 6. LOAD TEST DATASET
# ---------------------------------------------------------

@st.cache_data
def load_test_dataset():
    if not TEST_DATA_PATH.exists():
        return None

    return pd.read_pickle(TEST_DATA_PATH)


# ---------------------------------------------------------
# 7. WAFER MAP DISPLAY
# ---------------------------------------------------------

def show_wafer_map(wafer_map, title="Wafer Map"):
    wafer_map = np.asarray(wafer_map)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(wafer_map, cmap="viridis", interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")

    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------
# 8. UPLOADED IMAGE CONVERSION
# ---------------------------------------------------------

def convert_uploaded_wafer_image(uploaded_file):
    """
    Converts a specific purple/teal/yellow wafer-map visualization
    into the three categorical values expected by the model.

    This is a prototype conversion, not a general-purpose image
    segmentation algorithm.
    """

    image = Image.open(uploaded_file).convert("RGB")
    image_array = np.asarray(image)

    # Reference colours for the expected wafer-map palette.
    reference_colors = np.array(
        [
            [68, 0, 83],       # Purple
            [31, 144, 140],    # Teal
            [253, 230, 36],    # Yellow
        ],
        dtype=np.float32,
    )

    pixels = image_array.astype(np.float32)

    distances = np.linalg.norm(
        pixels[:, :, None, :] - reference_colors[None, None, :, :],
        axis=3,
    )

    wafer_map = np.argmin(distances, axis=2).astype(np.float32)

    return image, wafer_map


# ---------------------------------------------------------
# 9. RUN PREDICTION
# ---------------------------------------------------------

def run_prediction(wafer_map):
    predictor = load_predictor()
    return predictor.predict_wafer(wafer_map)


# ---------------------------------------------------------
# 10. DISPLAY PREDICTION
# ---------------------------------------------------------

def display_prediction(result):
    if not result:
        st.warning("No prediction result is available.")
        return

    predicted_class = result.get("predicted_class", "Unknown")
    score = result.get("score", 0)

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Predicted Defect", str(predicted_class))

    with col2:
        st.metric("Model Score", f"{float(score):.2%}")

    probabilities = result.get("class_probabilities", {})

    if probabilities:
        st.subheader("Class Scores")

        if isinstance(probabilities, dict):
            probability_df = pd.DataFrame(
                {
                    "Defect Class": list(probabilities.keys()),
                    "Score": list(probabilities.values()),
                }
            )
        else:
            classes = load_config()["data"]["classes"]
            probability_df = pd.DataFrame(
                {
                    "Defect Class": classes[:len(probabilities)],
                    "Score": probabilities,
                }
            )

        probability_df["Score"] = pd.to_numeric(
            probability_df["Score"], errors="coerce"
        )

        probability_df = probability_df.dropna()
        probability_df = probability_df.sort_values(
            "Score", ascending=False
        )

        st.bar_chart(
            probability_df.set_index("Defect Class")["Score"]
        )

        st.dataframe(
            probability_df,
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "Scores are model outputs and are not calibrated probabilities."
    )


# ---------------------------------------------------------
# 11. SIDEBAR NAVIGATION
# ---------------------------------------------------------

MENU_ITEMS = [
    "Wafer Defect Intelligence",
    "Wafer Detection & Analysis",
    "Export Reports",
    "Defect Classification",
    "Root Cause Analysis",
]

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-title">🔬 FabMind</div>
            <div class="brand-subtitle">
                Semiconductor Intelligence Platform
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown('<div class="menu-heading">MAIN MENU</div>',
                unsafe_allow_html=True)


# Sidebar navigation without radio buttons

    MENU_ITEMS = [
        ("🏠", "Wafer Defect Intelligence"),
        ("🔬", "Wafer Detection & Analysis"),
        ("📄", "Export Reports"),
        ("🎯", "Defect Classification"),
        ("🔍", "Root Cause Analysis"),
    ]

    if "selected_page" not in st.session_state:
        st.session_state.selected_page = "Wafer Defect Intelligence"

    for icon, page in MENU_ITEMS:

        if st.button(
            f"{icon}   {page}",
            key=f"menu_{page}",
            use_container_width=True
        ):
            st.session_state.selected_page = page

    selected_page = st.session_state.selected_page


    st.markdown("---")

    st.markdown(
        """
        <div class="menu-heading">MODEL STATUS</div>
        <div class="status-card">WaferViT Ready</div>
        <div class="sidebar-info">Architecture: ViT Tiny</div>
        <div class="sidebar-info">Classes: 9</div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# 12. COMMON HEADER
# ---------------------------------------------------------

st.markdown(
    '<div class="app-header">🔬FabMind – Semiconductor Wafer Defect Intelligence</div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# 13. Wafer Defect Intelligence
# ---------------------------------------------------------

if selected_page == "Wafer Defect Intelligence":

    st.markdown(
        """
        <div class="intro-card">
            FabMind is an AI-powered wafer defect detection and analysis platform designed to help semiconductor manufacturing teams in identifing defect patterns and support yield improvement.
            By leveraging deep learning and computer vision, FabMind analyzes wafer maps, identifies defect patterns, and provides insights to support engineering investigations.
            <div>    
                The platform is being developed in stages: 
                \n1. wafer classification first, followed by agent-based \n2. investigation and report generation.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Platform Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            '<div class="overview-card"><h3>01</h3>'
            '<b>Wafer Analysis</b><p>Inspect and classify wafer maps.</p></div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            '<div class="overview-card"><h3>02</h3>'
            '<b>Defect Intelligence</b><p>Understand predicted defect classes.</p></div>',
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            '<div class="overview-card"><h3>03</h3>'
            '<b>Root Cause Investigation</b><p>Future agent-based investigation.</p></div>',
            unsafe_allow_html=True,
        )

    st.info(
        "Current stage: wafer-map preprocessing and ViT-based classification."
    )


# ---------------------------------------------------------
# 14. WAFER DETECTION & ANALYSIS
# ---------------------------------------------------------

elif selected_page == "Wafer Detection & Analysis":

    st.subheader("Wafer Detection & Analysis")

    input_method = st.radio(
        "Select Wafer Input",
        ["Upload Wafer Image", "Existing Wafer Dataset"],
        horizontal=True
    )

    wafer_map = None

    # OPTION 1: EXISTING DATASET
    if input_method == "Existing Wafer Dataset":

        test_df = load_test_dataset()

        if test_df is None:
            st.error("Test dataset not found.")

        else:
            wafer_column = load_config()["data"].get(
                "wafer_column", "waferMap"
            )

            sample_index = st.selectbox(
                "Select Wafer Sample",
                range(len(test_df)),
                format_func=lambda x: f"Wafer Sample {x + 1}"
            )

            selected_row = test_df.iloc[sample_index]

            wafer_map = np.asarray(selected_row[wafer_column])

            show_wafer_map(wafer_map, "Selected Wafer Map")

            label_column = load_config()["data"].get(
                "label_column", "failureType"
            )

            if label_column in test_df.columns:
                st.caption(
                    f"Actual Dataset Label: {selected_row[label_column]}"
                )

    # OPTION 2: UPLOAD IMAGE
    else:

        uploaded_file = st.file_uploader(
            "Upload Wafer Image",
            type=["png", "jpg", "jpeg"]
        )

        if uploaded_file is not None:

            try:
                original_image, wafer_map = convert_uploaded_wafer_image(
                    uploaded_file
                )

                col1, col2 = st.columns(2)

                with col1:
                    st.image(
                        original_image,
                        caption="Uploaded Wafer Image",
                        use_container_width=True
                    )

                with col2:
                    show_wafer_map(
                        wafer_map,
                        "Converted Wafer Map"
                    )

                st.warning(
                    "Image conversion currently supports the expected "
                    "purple, teal and yellow wafer-map palette."
                )

            except Exception as error:
                st.error(f"Image processing failed: {error}")

    # RUN CLASSIFICATION
    if wafer_map is not None:

        if st.button("Run Wafer Classification", type="primary"):

            try:
                with st.spinner("Analyzing wafer map..."):

                    result = run_prediction(wafer_map)

                st.session_state["latest_prediction"] = result
                st.session_state["latest_wafer_map"] = wafer_map

                st.success("Wafer classification completed!")

                display_prediction(result)

            except Exception as error:
                st.error(f"Prediction failed: {error}")


# ---------------------------------------------------------
# 15. EXPORT REPORTS
# ---------------------------------------------------------

elif selected_page == "Export Reports":

    st.subheader("Export Reports")

    st.write(
        "This section will generate downloadable wafer inspection "
        "and defect investigation reports."
    )

    st.info(
        "Report generation and PostgreSQL storage will be implemented "
        "in a later development phase."
    )

    if "latest_prediction" in st.session_state:
        st.write("A latest prediction is available in this session.")
        st.json(st.session_state["latest_prediction"])
    else:
        st.warning("Run a wafer classification first to create a result.")


# ---------------------------------------------------------
# 16. DEFECT CLASSIFICATION
# ---------------------------------------------------------

elif selected_page == "Defect Classification":

    st.subheader("Defect Classification")

    if "latest_prediction" not in st.session_state:
        st.info(
            "Go to Wafer Detection & Analysis and run a classification first."
        )
    else:
        display_prediction(st.session_state["latest_prediction"])


# ---------------------------------------------------------
# 17. ROOT CAUSE ANALYSIS
# ---------------------------------------------------------

elif selected_page == "Root Cause Analysis":

    st.subheader("Root Cause Analysis")

    st.write(
        "This module will investigate wafer defect patterns and help "
        "engineering teams identify possible contributing factors."
    )

    st.info(
        "Planned future capabilities: LangGraph agent orchestration, "
        "MCP tools, engineering-document retrieval, machine-data "
        "correlation, and evidence-based root-cause hypotheses."
    )

    if "latest_prediction" in st.session_state:
        st.write("Latest classification result:")
        display_prediction(st.session_state["latest_prediction"])
    else:
        st.warning("Run a wafer classification to begin an investigation.")


# ---------------------------------------------------------
# 18. FOOTER
# ---------------------------------------------------------

st.markdown(
    """
    <div class="footer">
        FabMind | Semiconductor Wafer Defect Intelligence
    </div>
    """,
    unsafe_allow_html=True,
)
