"""
FOODWISE AI - Waste Analytics Dashboard
Visualizes historical food waste scans, category distributions, and cafeteria trends.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from app.components.ui_elements import render_page_header
from src.database.database import FoodWiseDatabase

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def render_analytics_page():
    render_page_header(
        title="Food Waste Analytics Dashboard",
        subtitle="Aggregated cafeteria telemetry, category waste distributions, and trend analysis.",
        badge_text="SDG 12 Telemetry"
    )

    db = FoodWiseDatabase()
    records = db.get_all_scans()

    # If no records exist yet, offer to populate sample cafeteria data
    if not records:
        st.info("No scan records found in the database yet.")
        if st.button("🌱 Seed Sample Cafeteria Scan Records"):
            sample_entries = [
                ("plate_01_lunch.jpg", 18.5, 45000, 250000, 5000, ["rice", "chicken"], "Grains & Cereals", "yolo11n-seg.pt", 120.5),
                ("plate_02_lunch.jpg", 42.0, 95000, 240000, 12000, ["french fries", "steak"], "Meat & Poultry", "yolo11n-seg.pt", 115.0),
                ("plate_03_dinner.jpg", 4.2, 9000, 230000, 2000, ["salad", "tomato"], "Vegetables & Greens", "yolo11n-seg.pt", 130.2),
                ("plate_04_dinner.jpg", 55.8, 130000, 245000, 8000, ["pasta", "mushrooms"], "Pasta & Noodles", "yolo11n-seg.pt", 125.4),
                ("plate_05_lunch.jpg", 12.1, 28000, 240000, 6000, ["rice", "vegetables"], "Grains & Cereals", "yolo11n-seg.pt", 118.9),
                ("plate_06_lunch.jpg", 28.3, 65000, 238000, 4000, ["fish", "potatoes"], "Fish & Seafood", "yolo11n-seg.pt", 122.1),
            ]
            for s in sample_entries:
                db.insert_scan(*s)
            st.rerun()
        return

    # Convert records to DataFrame
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # High-level KPIs
    total_scans = len(df)
    avg_waste = df["waste_percentage"].mean()
    clean_plates = (df["waste_percentage"] < 5.0).sum()
    clean_plate_rate = (clean_plates / total_scans) * 100
    top_wasted_cat = df["primary_category"].mode()[0] if not df["primary_category"].empty else "N/A"

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Total Plates Analyzed", f"{total_scans}")
    with kpi2:
        st.metric("Average Leftover Area", f"{avg_waste:.1f}%")
    with kpi3:
        st.metric("Clean Plate Rate (<5%)", f"{clean_plate_rate:.1f}%")
    with kpi4:
        st.metric("Most Frequent Waste Category", top_wasted_cat)

    st.divider()

    # Interactive Charts
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("#### Waste Area Index (%) Over Time")
        fig_timeline = px.line(
            df.sort_values("timestamp"),
            x="timestamp",
            y="waste_percentage",
            markers=True,
            title="Scan Sequence vs Leftover Food Area Index (%)",
            labels={"timestamp": "Scan Timestamp", "waste_percentage": "Leftover Food Area (%)"},
            color_discrete_sequence=["#2563EB"]
        )
        fig_timeline.add_hline(y=20.0, line_dash="dash", line_color="orange", annotation_text="Target Waste Ceiling (20%)")
        st.plotly_chart(fig_timeline, use_container_width=True)

    with col_chart2:
        st.markdown("#### Primary Wasted Food Categories")
        cat_counts = df["primary_category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]
        fig_pie = px.pie(
            cat_counts,
            values="Count",
            names="Category",
            title="Proportion of Leftover Items by Food Group",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Waste Severity Distribution Histogram
    st.markdown("#### Distribution of Plate Waste Severity")
    fig_hist = px.histogram(
        df,
        x="waste_percentage",
        nbins=10,
        title="Frequency of Food Waste Percentage Bands",
        labels={"waste_percentage": "Estimated Leftover Food Area (%)"},
        color_discrete_sequence=["#10B981"]
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()

    # Data Table View
    st.markdown("#### Scan Logs & Database Records")
    display_df = df[["id", "timestamp", "image_filename", "waste_percentage", "primary_category", "model_version", "inference_time_ms"]]
    st.dataframe(display_df, use_container_width=True)

    # Export & Management
    c_exp, c_del = st.columns([1, 1])
    with c_exp:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Export Full Database to CSV", data=csv_bytes, file_name="foodwise_scans_history.csv", mime="text/csv")
    with c_del:
        if st.button("🗑️ Clear Scan History"):
            db.clear_all()
            st.success("Database cleared.")
            st.rerun()
