"""
FOODWISE AI - Main Streamlit Application Entrypoint
AI-Based Food Waste Detection and Categorization System Using YOLOv11 Instance Segmentation.
"""
import sys
from pathlib import Path
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.components.ui_elements import load_custom_css
from app.pages.home import render_home_page
from app.pages.detection import render_detection_page
from app.pages.analytics import render_analytics_page
from app.pages.model_info import render_model_info_page
from app.pages.sdg_impact import render_sdg_page
from src.database.database import FoodWiseDatabase

def main():
    st.set_page_config(
        page_title="FOODWISE AI | Food Waste Detection",
        page_icon="🍲",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    load_custom_css()

    # Initialize SQLite database
    FoodWiseDatabase()

    # Sidebar Navigation
    st.sidebar.title("🍲 FOODWISE AI")
    st.sidebar.markdown("**Food Waste Detection & Categorization**")
    st.sidebar.caption("Powered by YOLOv11 Instance Segmentation")
    st.sidebar.divider()

    pages = {
        "🏠 Overview & Methodology": render_home_page,
        "🔍 Plate Waste Detection": render_detection_page,
        "📊 Waste Analytics Dashboard": render_analytics_page,
        "🧠 Model Architecture & Metrics": render_model_info_page,
        "🌍 Sustainability & SDG 12": render_sdg_page
    }

    selected_page = st.sidebar.radio(
        "Navigation",
        options=list(pages.keys()),
        index=1  # Default to Detection page for immediate interactivity
    )

    st.sidebar.divider()
    st.sidebar.markdown("""
    **Research Reference:**  
    *Ferreira et al., MDPI Applied Sciences 2025*  
    **Dataset:** Roboflow PROJ3 (v6)  
    **SDG Target:** 12.3  
    """)

    # Render selected page
    render_func = pages[selected_page]
    render_func()

if __name__ == "__main__":
    main()
