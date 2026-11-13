"""
UI elements and styling helpers for FOODWISE AI Streamlit application.
"""
import streamlit as st

def load_custom_css():
    """Injects custom CSS styling for modern, clean dashboard appearance."""
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1E3A8A;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.1rem;
            color: #4B5563;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background-color: #F8FAFC;
            border-left: 5px solid #2563EB;
            padding: 1rem 1.2rem;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            margin-bottom: 1rem;
        }
        .waste-badge-low {
            background-color: #DEF7EC;
            color: #03543F;
            padding: 4px 12px;
            border-radius: 9999px;
            font-weight: 600;
            display: inline-block;
        }
        .waste-badge-medium {
            background-color: #FEF08A;
            color: #713F12;
            padding: 4px 12px;
            border-radius: 9999px;
            font-weight: 600;
            display: inline-block;
        }
        .waste-badge-high {
            background-color: #FDE8E8;
            color: #9B1C1C;
            padding: 4px 12px;
            border-radius: 9999px;
            font-weight: 600;
            display: inline-block;
        }
        .sdg-badge {
            background-color: #CFFAFE;
            color: #0E7490;
            font-weight: bold;
            padding: 3px 8px;
            border-radius: 4px;
        }
        </style>
    """, unsafe_allow_html=True)

def render_page_header(title: str, subtitle: str, badge_text: str = "SDG 12: Responsible Consumption"):
    """Renders standard consistent page header."""
    st.markdown(f'<div class="main-header">{title} <span class="sdg-badge">{badge_text}</span></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">{subtitle}</div>', unsafe_allow_html=True)
    st.divider()

def render_waste_badge(waste_pct: float) -> str:
    """Returns HTML markup for waste severity badge."""
    if waste_pct < 5.0:
        return f'<span class="waste-badge-low">Minimal Waste: {waste_pct:.1f}%</span>'
    elif waste_pct < 20.0:
        return f'<span class="waste-badge-low">Low Waste: {waste_pct:.1f}%</span>'
    elif waste_pct < 45.0:
        return f'<span class="waste-badge-medium">Moderate Waste: {waste_pct:.1f}%</span>'
    else:
        return f'<span class="waste-badge-high">High Waste: {waste_pct:.1f}%</span>'

