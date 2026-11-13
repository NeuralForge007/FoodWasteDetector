"""
FOODWISE AI - Sustainability & SDG 12 Impact
Explains the environmental significance, UN SDG 12.3 targets, and institutional cafeteria benefits.
"""
import streamlit as st
from app.components.ui_elements import render_page_header

def render_sdg_page():
    render_page_header(
        title="Sustainability & SDG Impact",
        subtitle="Harnessing Computer Vision to Drive Zero Waste and Responsible Food Consumption.",
        badge_text="UN Sustainable Development Goals"
    )

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### UN SDG Target 12.3
        > *"By 2030, halve per capita global food waste at the retail and consumer levels and reduce food losses along production and supply chains, including post-harvest losses."*

        Food waste accounts for approximately **8–10% of global greenhouse gas emissions**. In university and corporate canteens:
        - Thousands of meals are served daily.
        - High plate leftovers indicate over-portioning or consumer dissatisfaction with specific recipes.
        - Lack of granular feedback forces kitchens to rely on guesswork when ordering ingredients.

        ### How FOODWISE AI Enables Actionable Change
        1. **Automated Granular Auditing:** Rather than merely weighing garbage bins, the system identifies **which** items (e.g. 40% of boiled potatoes vs 2% of chicken) are persistently wasted.
        2. **Dynamic Portion Control:** Kitchen managers can decrease serving sizes for high-waste side dishes and offer on-demand refills.
        3. **Inventory Procurement Optimization:** Accurate food waste tracking prevents over-purchasing perishable produce, directly reducing institutional operating costs and methane landfill impact.
        """)

    with col2:
        st.info("""
        #### Interconnected SDGs
        - **SDG 12:** Responsible Consumption & Production *(Primary)*
        - **SDG 2:** Zero Hunger *(Food Redistribution)*
        - **SDG 13:** Climate Action *(Methane Mitigation)*
        - **SDG 14 & 15:** Life Below Water & Land *(Resource Conservation)*
        """)

    st.divider()

    st.markdown("""
    ### Scientific Transparency: Area Index vs Physical Mass
    
    A critical scientific consideration in food waste estimation:
    
    - **Current Implementation:** FOODWISE AI computes the **Paper-Based Food Waste Area Index (%)** using 2D pixel segmentation:
    """)
    st.latex(r"""
    \text{Food Waste Area Index (\%)} = \frac{\text{Area}_{\text{Food}}}{\text{Area}_{\text{Plate}} - \text{Area}_{\text{Garbage}}} \times 100
    """)
    st.markdown("""
    - **Why We Do NOT Claim Physical Mass (Grams) From Single 2D Images:**  
      A 2D photograph lacks depth ($z$-axis) information. Estimating grams from 2D pixel area alone without calibrated 3D depth cameras (e.g., LiDAR, Intel RealSense) or physical density models introduces massive measurement errors (a flat smear of sauce vs a tall pile of mashed potatoes have identical 2D area but drastically different mass).
      
    - **Future Vision:** Combining YOLOv11 instance segmentation masks with RGB-D depth estimation and ingredient density matrices for calibrated volumetric weight estimation.
    """)
