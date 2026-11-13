"""
FOODWISE AI - Home & Overview Page
Introduces the project problem statement, research foundation, and system workflow.
"""
import streamlit as st
from app.components.ui_elements import render_page_header

def render_home_page():
    render_page_header(
        title="FOODWISE AI: Overview & Research Methodology",
        subtitle="AI-Based Food Waste Detection and Categorization Using YOLOv11 Instance Segmentation",
        badge_text="SDG 12: Responsible Consumption"
    )

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        ### Problem Statement & Context
        Food wastage in institutional cafeterias, university dining halls, and restaurants is a massive ecological and economic problem.
        After dining, consumers leave varying quantities of food on their plates.
        
        Traditional waste collection aggregates all scrapings into generic bins without identifying:
        1. **Which specific food ingredients** are repeatedly wasted (e.g. rice, poultry, salads, pasta).
        2. **How much food is wasted** per plate session.
        3. **Whether items are edible food or non-edible contaminants** (napkins, toothpicks, cutlery).

        **FOODWISE AI** solves this by capturing photographs of returned plates, deploying **YOLOv11 Instance Segmentation** to classify and delineate exact pixel boundaries of food items, utensils, and plates, and calculating the standardized **Paper-Based Food Waste Area Index (%)**.
        """)

    with col2:
        st.info("""
        #### Primary Alignment
        - **SDG 12 (Target 12.3):** Halve per capita global food waste at retail and consumer levels and reduce food losses along production supply chains.
        
        #### Supporting SDGs
        - **SDG 2 (Zero Hunger):** Redistribution and kitchen inventory optimization.
        - **SDG 13 (Climate Action):** Reducing methane emissions from landfill food decomposition.
        """)

    st.divider()

    st.markdown("### Research Foundation & Methodology")
    st.markdown("""
    This project reproduces and extends the research methodology published in:
    > **"Food Waste Detection in Canteen Plates Using YOLOv11"**  
    > *João Ferreira, Paulino Cerqueira, Jorge Ribeiro*  
    > **MDPI Applied Sciences 2025**, 15(13), 7137.  
    > DOI: [10.3390/app15137137](https://doi.org/10.3390/app15137137)

    The authors established a standardized benchmark using **YOLOv11 instance segmentation** on cafeteria plates, demonstrating that real-time polygon segmentation out-performs bounding boxes by capturing irregular food geometries (e.g., loose rice, sauces, shredded meat).
    """)

    st.markdown("### Standardized Area Estimation Formula")
    st.latex(r"""
    \text{Food Waste Area Index (\%)} = \frac{\text{Area}_{\text{Food}}}{\text{Area}_{\text{Plate}} - \text{Area}_{\text{Garbage}}} \times 100
    """)
    st.caption("Implemented with 9 strict algorithmic safeguards against occlusion, overlap, negative denominators, and missing plates.")

    st.divider()

    st.markdown("### System Workflow Pipeline")
    st.markdown("""
    1. **Plate Image Acquisition:** High-resolution RGB image captured from overhead or angled cafeteria station.
    2. **YOLOv11 Instance Segmentation:** Deep CNN backbone (C3k2, SPPF, C2PSA) extracts features and outputs multi-prototype segmentation masks for 85 food/utensil classes.
    3. **Categorization Engine:** Maps 85 fine-grained classes into 9 primary SDG categories (Grains/Rice, Poultry, Vegetables, Utensils, etc.).
    4. **Waste Indicator Computation:** Calculates precise polygonal pixel area with overlap deduplication.
    5. **Database Storage & Analytics:** Stores scan records in local SQLite database for historical trend analysis and kitchen planning.
    """)
