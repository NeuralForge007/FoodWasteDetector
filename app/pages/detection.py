"""
FOODWISE AI - Plate Waste Detection Page
Interactive inference interface supporting image upload, test sample selection,
model switching, mask visualizations, KPI metrics, and database recording.
"""
import io
import time
from pathlib import Path
import streamlit as st
from PIL import Image
import pandas as pd

from app.components.ui_elements import render_page_header, render_waste_badge
from app.components.image_utils import load_sample_images
from src.inference.predictor import FoodWisePredictor
from src.database.database import FoodWiseDatabase

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

@st.cache_resource
def get_predictor(weights_path: str, conf_threshold: float = 0.25):
    """Caches predictor instance per model selection to optimize memory and speed."""
    return FoodWisePredictor(weights_path=weights_path, conf_threshold=conf_threshold)

def render_detection_page():
    render_page_header(
        title="Plate Waste Detection & Categorization",
        subtitle="Upload or select a canteen plate image to analyze leftover food, utensils, and waste index.",
        badge_text="Real-Time YOLOv11"
    )

    db = FoodWiseDatabase()

    # Sidebar / controls column
    with st.sidebar:
        st.header("Detection Controls")
        
        # Check which model checkpoints exist
        best_pt = PROJECT_ROOT / "models" / "best.pt"
        benchmark_pt = PROJECT_ROOT / "models" / "yolov11-x-weights-v6.pt"
        default_n = PROJECT_ROOT / "yolo11n-seg.pt"

        model_choices = {}
        if benchmark_pt.exists():
            model_choices["Ferreira et al. YOLOv11x Benchmark (Official)"] = str(benchmark_pt)
        if best_pt.exists():
            model_choices["FOODWISE YOLOv11n (Fine-Tuned Checkpoint)"] = str(best_pt)
        elif default_n.exists():
            model_choices["FOODWISE YOLOv11n (Base Weights)"] = str(default_n)

        selected_model_name = st.selectbox(
            "Select Model Architecture",
            options=list(model_choices.keys()),
            index=0
        )
        selected_model_path = model_choices[selected_model_name]

        conf_thresh = st.slider(
            "Confidence Threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.25,
            step=0.05,
            help="Minimum detection confidence required for predictions."
        )

        st.caption(f"Active Weights: `{Path(selected_model_path).name}`")

    # Image source tabs
    tab_upload, tab_samples = st.tabs(["📤 Upload Image", "🖼️ Select Benchmark Sample"])
    
    input_image = None
    image_source_label = "Upload"

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload plate photograph (JPEG, PNG, WEBP)",
            type=["jpg", "jpeg", "png", "webp"]
        )
        if uploaded_file is not None:
            input_image = Image.open(uploaded_file).convert("RGB")
            image_source_label = uploaded_file.name

    with tab_samples:
        samples = load_sample_images()
        if samples:
            sample_names = [s[0] for s in samples]
            selected_sample_idx = st.selectbox("Choose a canteen plate from test set:", range(len(sample_names)), format_func=lambda x: sample_names[x])
            sample_path = samples[selected_sample_idx][1]
            if st.button("Load Selected Sample"):
                input_image = Image.open(sample_path).convert("RGB")
                image_source_label = sample_path.name
        else:
            st.info("No test samples found in data/raw/roboflow/ directory.")

    if input_image is None:
        st.info("Please upload a photograph or select a sample plate to run analysis.")
        return

    # Image Preview & Analysis Action
    col_input, col_run = st.columns([2, 1])
    with col_input:
        st.image(input_image, caption=f"Input Image: {image_source_label}", use_container_width=True)
    with col_run:
        st.markdown("#### Ready to Analyze")
        st.write("Click below to run instance segmentation, class categorization, and area index calculation.")
        run_btn = st.button("🚀 Analyze Plate Waste", type="primary", use_container_width=True)

    if run_btn:
        with st.spinner("Processing image through YOLOv11 Instance Segmentation pipeline..."):
            start_t = time.time()
            try:
                predictor = get_predictor(selected_model_path, conf_thresh)
                res = predictor.predict(input_image, conf_threshold=conf_thresh)
                inference_time_ms = round((time.time() - start_t) * 1000, 1)
            except Exception as e:
                st.error(f"Inference error: {str(e)}")
                return

        st.success(f"Analysis completed in {inference_time_ms} ms!")

        # Visual Results Tabs
        st.markdown("### Visual Segmentation Analysis")
        tab_overlay, tab_cluster = st.tabs(["🏷️ Annotated Detections", "🎨 Mask Clustering"])
        with tab_overlay:
            st.image(res["annotated_image"], caption="Detections with Class Labels & Bounding Polygons", use_container_width=True)
        with tab_cluster:
            st.image(res["clustering_image"], caption="Pure Mask Clustering Map (Paper Visualization)", use_container_width=True)

        # Waste Estimation KPI Summary
        waste = res["waste_analysis"]
        st.markdown("### Leftover Food Waste Summary")

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.metric("Estimated Leftover Food Area", f"{waste['waste_percentage']:.1f}%")
            st.markdown(render_waste_badge(waste["waste_percentage"]), unsafe_allow_html=True)
        with kpi2:
            st.metric("Total Edible Food Area", f"{waste['food_area']:,} px")
        with kpi3:
            st.metric("Plate Surface Area", f"{waste['plate_area']:,} px")
        with kpi4:
            st.metric("Non-Edible / Utensil Area", f"{waste['garbage_area']:,} px")

        # Safeguard / OOD Notices
        if not res["has_reliable_detections"]:
            st.warning("⚠️ No supported items detected. The image may be out-of-distribution or lacking sufficient lighting.")
        elif not waste["plate_detected"]:
            st.warning("⚠️ " + waste["message"])
        else:
            st.info(f"✅ Safeguards Passed: {waste['message']}")

        # Detailed Detections Table
        st.markdown("### Detected Items & Ingredient Breakdown")
        if res["table_rows"]:
            df_table = pd.DataFrame(res["table_rows"])
            st.dataframe(df_table, use_container_width=True)

            # Option to save scan record to database
            edible_names = [item["class_name"] for item in res["edible_items"]]
            primary_cat = res["edible_items"][0]["category"] if res["edible_items"] else "Non-Edible"
            
            try:
                db.insert_scan(
                    image_filename=image_source_label,
                    waste_percentage=waste["waste_percentage"],
                    food_area=waste["food_area"],
                    plate_area=waste["plate_area"],
                    garbage_area=waste["garbage_area"],
                    detected_items=edible_names,
                    primary_category=primary_cat,
                    model_version=Path(selected_model_path).name,
                    inference_time_ms=inference_time_ms
                )
                st.toast("Scan successfully logged to local SQLite database!", icon="💾")
            except Exception as e:
                st.warning(f"Database log error: {str(e)}")

            # Download CSV button
            csv_data = df_table.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Detection Report (CSV)",
                data=csv_data,
                file_name=f"foodwise_report_{int(time.time())}.csv",
                mime="text/csv"
            )
        else:
            st.write("No individual items were cataloged.")
