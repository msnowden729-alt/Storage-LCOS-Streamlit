import streamlit as st
import master_script
import os
import io
from contextlib import redirect_stdout
import glob
import numpy as np

st.markdown("""
<style>
/* Compress cells/columns only (no header shrink) */
[data-testid="stDataFrame"] td { 
    font-size: 11px !important; padding: 2px 4px !important;  /* Small font + tight padding for cells */
}
.metric-container div div { 
    font-size: 11px !important; padding: 1px !important;  /* Metric values */
}
.metric-container label { 
    font-size: 13px !important;  /* Headers/labels: Slightly smaller but readable (reverted from 11px) */
}
/* Column gaps: Minimal horizontal space */
.element-container { gap: 2px !important; }  /* Tightens column spacing */
</style>
""", unsafe_allow_html=True)

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
        st.subheader("Key Metrics by Storage Technology")
        
        cols = st.columns(len(results_list), gap="none")  # Zero gap: Packs columns tightly to full width
        
        for idx, res in enumerate(results_list):
            with cols[idx]:
                prog_name = res["program"].replace("calcs", "").upper()
                st.markdown(f"**{prog_name}**")
                
                safe_metric(st.container(), "Base CAPEX ($M)", res.get("baselineCAPEX", np.nan) / 1e6, "{:,.1f}")
                safe_metric(st.container(), "New CAPEX ($M)", res.get("newCAPEX", np.nan) / 1e6, "{:,.1f}")
                safe_metric(st.container(), "Base OPEX ($M)", res.get("baselineOPEX", np.nan) / 1e6, "{:,.1f}")
                safe_metric(st.container(), "New OPEX ($M)", res.get("newOPEX", np.nan) / 1e6, "{:,.1f}")
                safe_metric(st.container(), "Base LCOS ($/kWh)", res.get("baseLCOS", np.nan), "{:,.2f}")
                safe_metric(st.container(), "Change (%)", res.get("LCOSchange", np.nan), "{:,.1f}%")

        # FIXED PLOTS: Display all figures
        st.subheader("Generated Plots")
        plot_descriptions = {
        0: "Best Storage Technology Comparison: Mild (Left) vs. Arctic (Right) Climates",
        1: "Average Levelized Cost Change by Technology",
        2: "Minimum Levelised Cost Gradient in Mild (Left) vs. Arctic (Right) Climates, USD/kWh",
        }

      
        for i, fig in enumerate(figures):
            try:
                # Save to PNG bytes
                output = io.BytesIO()
                fig.savefig(output, format='png', bbox_inches='tight', dpi=100, facecolor='white')  # Explicit facecolor to avoid transparency issues
                output.seek(0)  # CRITICAL: Reset pointer to start (often missed)
                
                # Validate: Check non-empty and PNG header
                img_bytes = output.getvalue()
                if len(img_bytes) == 0 or not img_bytes.startswith(b'\x89PNG'):
                    raise ValueError(f"Invalid PNG for plot {i+1}: {len(img_bytes)} bytes")
                
                # Desc from your dict
                desc = plot_descriptions.get(i, f"Plot {i+1}: Untitled")
                
                # Display scaled
                st.image(img_bytes, caption=desc, width=600)  # Or 500 for slightly larger
                
                output.close()  # Clean up
            except Exception as e:
                st.error(f"Failed to render Plot {i+1}: {e}")
                st.image("https://via.placeholder.com/400x300?text=Plot+Error", width=600)  # Fallback placeholder
            
            st.markdown("---")  # Separator
    
