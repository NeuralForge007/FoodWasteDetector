"""
FOODWISE AI - Model Information & Technical Architecture
Displays model specifications, parameters, dataset audit details, and evaluation metrics.
"""
import json
from pathlib import Path
import streamlit as st
import pandas as pd
from app.components.ui_elements import render_page_header

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def render_model_info_page():
    render_page_header(
        title="Model Architecture & Evaluation Metrics",
        subtitle="Detailed specifications for YOLOv11 Instance Segmentation, PROJ3 Dataset, and Benchmark Comparisons.",
        badge_text="Ultralytics YOLOv11"
    )

    tab_arch, tab_metrics, tab_dataset = st.tabs(["🧠 YOLOv11 Architecture", "📊 Evaluation Metrics", "📁 PROJ3 Dataset V6"])

    with tab_arch:
        st.markdown("""
        ### YOLOv11 Instance Segmentation Architecture

        YOLOv11 introduces key architectural advancements over previous vision models (YOLOv8, U-Net, Mask R-CNN):

        1. **Enhanced Backbone with C3k2 & SPPF:**
           - Uses modular **C3k2** blocks (Cross-Stage Partial with kernel-2 optimizations) allowing deeper gradient propagation with lower parameter overhead.
           - **SPPF (Spatial Pyramid Pooling - Fast)** pools multi-scale context without increasing computational complexity.
        
        2. **C2PSA (Cross-Stage Partial with Spatial Attention):**
           - Integrates spatial self-attention modules directly into the deep feature representation, allowing the network to distinguish subtle boundaries between overlapping food items (e.g., loose rice grains resting on meat slices).

        3. **Dual-Head Decoupled Architecture:**
           - Decoupled detection head separates classification logits from bounding box regression coordinates.
           - Parallel **Proto Segmentation Head** generates high-resolution prototype mask channels, which are linearly combined with dynamic instance coefficients to produce sharp polygonal masks.

        4. **Multi-Task Loss Function:**
        """)
        st.latex(r"""
        \mathcal{L}_{\text{total}} = \lambda_{\text{box}} \mathcal{L}_{\text{box}} + \lambda_{\text{cls}} \mathcal{L}_{\text{cls}} + \lambda_{\text{dfl}} \mathcal{L}_{\text{dfl}} + \lambda_{\text{seg}} \mathcal{L}_{\text{seg}}
        """)

    with tab_metrics:
        st.markdown("### Test Split Evaluation Performance")
        
        # Load metrics.json if available
        metrics_file = PROJECT_ROOT / "reports" / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            
            bb = metrics["bounding_box_metrics"]
            seg = metrics["instance_segmentation_metrics"]

            st.write(f"**Model:** `{metrics.get('model_name')}` | **Split:** `{metrics.get('split')}` | **Classes:** `{metrics.get('num_classes')}`")

            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.markdown("#### Bounding Box Detection")
                st.write(f"- **Precision:** `{bb['precision']:.4f}`")
                st.write(f"- **Recall:** `{bb['recall']:.4f}`")
                st.write(f"- **F1-Score:** `{bb['f1_score']:.4f}`")
                st.write(f"- **mAP@50:** `{bb['mAP50']:.4f}`")
                st.write(f"- **mAP@50-95:** `{bb['mAP50_95']:.4f}`")

            with col_m2:
                st.markdown("#### Instance Segmentation (Mask)")
                st.write(f"- **Precision:** `{seg['precision']:.4f}`")
                st.write(f"- **Recall:** `{seg['recall']:.4f}`")
                st.write(f"- **F1-Score:** `{seg['f1_score']:.4f}`")
                st.write(f"- **mAP@50:** `{seg['mAP50']:.4f}`")
                st.write(f"- **mAP@50-95:** `{seg['mAP50_95']:.4f}`")

        else:
            st.info("Evaluation metrics file not found. Run `scripts/evaluate.py` to populate.")

        st.markdown("### Benchmark Comparison with Ferreira et al. (MDPI 2025)")
        comp_df = pd.DataFrame([
            {"Model Variant": "Ferreira et al. YOLOv11x (100 Epochs)", "Parameters": "56.9 M", "GFLOPs": "194.9", "Mask mAP@50": "0.784", "Mask Precision": "0.812", "Mask Recall": "0.765"},
            {"Model Variant": "FOODWISE Local YOLOv11n (Fine-Tuned)", "Parameters": "2.87 M", "GFLOPs": "9.8", "Mask mAP@50": "0.418", "Mask Precision": "0.401", "Mask Recall": "0.389"},
            {"Model Variant": "Custom U-Net (Previous Iteration)", "Parameters": "31.0 M", "GFLOPs": "58.2", "Mask mAP@50": "N/A (Dice 0.30)", "Mask Precision": "0.946 (Overfit)", "Mask Recall": "0.299"},
        ])
        st.dataframe(comp_df, use_container_width=True)

    with tab_dataset:
        st.markdown("""
        ### PROJ3 Food Waste Dataset (Roboflow Version 6)
        - **Source:** Canteen plate instance segmentation benchmark (Ferreira et al., 2025).
        - **Images:** 160 total plate photographs at 640x640 resolution (117 train, 35 valid, 8 test).
        - **Annotations:** 1,150 polygon segmentation instances across 85 distinct classes.
        - **Key Ingredients Supported:**
          - `rice` (Class 63, 76 instances in 55 images)
          - `chicken` (Class 18, 55 instances)
          - `plate` (Class 58, 154 instances)
          - `fork`, `knife`, `spoon` (Utensil classes)
          - `vegetables`, `french fries`, `salad`, `fish`, `pasta`
        """)
        
        dist_img = PROJECT_ROOT / "reports" / "class_distribution.png"
        if dist_img.exists():
            st.image(str(dist_img), caption="PROJ3 Dataset Class Distribution (Top 30 Classes)", use_container_width=True)
