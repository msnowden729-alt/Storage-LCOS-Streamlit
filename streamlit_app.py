import streamlit as st
import master_script
import os
import io
from contextlib import redirect_stdout
import glob
import numpy as np

st.set_page_config(page_title="Arctic Energy Storage LCOS Model", layout="wide")

st.title("Arctic Energy Storage LCOS Model")

# -------------------------------
# USER INPUTS
# -------------------------------
st.header("Input Parameters")

col1, col2 = st.columns(2)

with col1:
    Power = st.number_input("Power (MW)", value=100.0)
    DD = st.number_input("Discharge Duration (hours)", value=1.0)
    charges_per_year = st.number_input("Charges per Year", value=300)
    Powercost = st.number_input("Electricity Cost (USD/kWh)", value=0.05)

with col2:
    interest_rate = st.number_input("Discount Rate", value=0.08)
    project_lifespan = st.number_input("Project Lifespan (years)", value=20)

st.subheader("Temperature Inputs (°C)")

tcol1, tcol2 = st.columns(2)

with tcol1:
    arctic_winter_low = st.number_input("Arctic Winter Low (°C)", value=-40)
    arctic_mean_annual = st.number_input("Arctic Mean Annual (°C)", value=-10)

with tcol2:
    baseline_winter_low = st.number_input("Baseline Winter Low (°C)", value=0)
    baseline_mean_annual = st.number_input("Baseline Mean Annual (°C)", value=20)

# Bundle temperatures into correct list format
selected_Tamb = [
    arctic_winter_low,
    baseline_winter_low,
    arctic_mean_annual,
    baseline_mean_annual
]

# Helper function (updated for Streamlit column)
def safe_metric(col, label, value, fmt="{:,.2f}"):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        col.error(f"{label}: N/A")
    else:
        try:
            formatted = fmt.format(value)
            col.metric(label, formatted)
        except (ValueError, TypeError):
            col.error(f"{label}: Error")
            
common_inputs = {
    "Power": Power,
    "DD": DD,
    "charges_per_year": charges_per_year,
    "selected_Tamb": selected_Tamb,
    "Powercost": Powercost,
    "interest_rate": interest_rate,
    "project_lifespan": project_lifespan,
}


# -------------------------------
# RUN BUTTON
# -------------------------------
# Run the analysis
if st.button("Run Analysis"):
    with st.spinner("Computing..."):
        output = master_script.run(common_inputs)
        results_list = output.get("results_list", [])  # Full list of 5 subprograms
        figures = output.get("figures", [])  # List of plots

    if not results_list:
        st.error("No results generated. Check inputs and subprograms.")
    else:
        # TABLE OUTPUT
        st.subheader("Key Metrics by Storage Technology")
        
        # --- Make table text smaller + tighten layout
        st.markdown("""
        <style>
        .small-font {
            font-size: 12px !important;
            line-height: 1.0 !important;
        }
        .stMetric {
            font-size: 12px !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        num_progs = len(results_list)
        cols = st.columns(num_progs, gap="small")
        
        for idx, res in enumerate(results_list):
            with cols[idx]:
        
                prog_name = res["program"].replace("calcs", "").upper()
                st.markdown(f"<h4 class='small-font'>{prog_name}</h4>", unsafe_allow_html=True)
        
                st.markdown("<div class='small-font'>", unsafe_allow_html=True)
        
                safe_metric(st, "Baseline CAPEX ($M)", res.get("baselineCAPEX", np.nan) / 1e6, "{:,.1f}")
                safe_metric(st, "New CAPEX ($M)", res.get("newCAPEX", np.nan) / 1e6, "{:,.1f}")
                safe_metric(st, "Baseline OPEX ($M)", res.get("baselineOPEX", np.nan) / 1e6, "{:,.1f}")
                safe_metric(st, "New OPEX ($M)", res.get("newOPEX", np.nan) / 1e6, "{:,.1f}")
                safe_metric(st, "Baseline LCOS ($/kWh)", res.get("baseLCOS", np.nan), "{:,.1f}")
                safe_metric(st, "Arctic LCOS Change (%)", res.get("LCOSchange", np.nan), "{:,.1f}%")
        
                st.markdown("</div>", unsafe_allow_html=True)


        # FIXED PLOTS: Display all figures
        st.subheader("Generated Plots")
        plot_descriptions = {
        0: "Best Storage Technology Comparison: Mild (Left) vs. Arctic (Right) Climates",
        1: "Average Levelized Cost Change by Technology",
        2: "Minimum Levelised Cost Gradient in Mild (Left) vs. Arctic (Right) Climates, USD/kWh",
        }

        st.subheader("Generated Plots")
        
        for i, fig in enumerate(figures):
            fig.set_size_inches(5, 3.3)  # shrink while keeping proportions
            st.pyplot(fig, use_container_width=True)
            st.caption(f"Plot {i+1}")
            st.markdown("---")  # Separator
    
