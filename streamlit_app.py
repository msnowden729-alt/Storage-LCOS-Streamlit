import streamlit as st
import master_script
import numpy as np
import io
import os
from contextlib import redirect_stdout
from PIL import Image
import pandas as pd

st.set_page_config(page_title="Arctic Energy Storage LCOS Model", layout="wide")

# ---------------------------------------------------------
# Custom CSS for compact HTML metrics + responsive columns
# ---------------------------------------------------------
st.markdown("""
<style>

/* --- TABLE CARD --- */
.metric-card {
    border: 1px solid #DDD;
    border-radius: 8px;
    padding: 6px 10px;
    margin-bottom: 8px;
    background-color: #FAFAFA;
}

/* Program name header */
.metric-header {
    font-size: 1.0rem;
    font-weight: 600;
    margin-bottom: 4px;
    text-align: center;
}

/* Metric label */
.metric-label {
    font-size: 0.72rem;
    color: #555;
}

/* Metric value */
.metric-value {
    font-size: 0.80rem;
    font-weight: 600;
    margin-bottom: 4px;
}

/* N/A style */
.metric-value.na {
    color: #AA0000;
    font-weight: 700;
}

/* Scrollable row for mobile */
.metric-row {
    display: flex;
    flex-direction: row;
    overflow-x: auto;
    gap: 12px;
    padding-bottom: 6px;
}

/* Column card */
.metric-col {
    min-width: 150px;
    max-width: 180px;
    flex: 0 0 auto;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Compact metric renderer (HTML version)
# ---------------------------------------------------------
def render_metric(label, value, fmt="{:,.2f}"):
    """Return HTML block for an individual metric."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        formatted = "<div class='metric-value na'>N/A</div>"
    else:
        try:
            formatted_val = fmt.format(value)
            formatted = f"<div class='metric-value'>{formatted_val}</div>"
        except:
            formatted = "<div class='metric-value na'>ERR</div>"

    return f"""
        <div class='metric-card'>
            <div class='metric-label'>{label}</div>
            {formatted}
        </div>
    """


# ---------------------------------------------------------
# UI HEADER
# ---------------------------------------------------------
st.title("Arctic Energy Storage LCOS Model")
#ADD: Smaller-font summary below header
st.markdown("""
<div style="font-size: 14px; color: gray; margin-top: 0px; line-height: 1.2;">
This tool quantifies the Levelized Cost of Storage (LCOS) for five major technologies-- <strong>Hydrogen (H2), Pumped Hydropower (PHS), Lithium-Ion Batteries (BESS), Compressed Air Energy Storage (CAES),</strong>  and <strong>Flywheels</strong>-- to reasses the competitive landscape for energy storage in Arctic climates as compared to to current <a href="https://energystorage.shinyapps.io/LCOSApp/" target="_blank">baseline models</a>.
This tool accounts for Arctic-specific factors including Thermal Management System (TMS) customisation, increased heating and reducded cooling OPEX, rountrip efficiency changes, and storage capacity changes, to name a few.
The full background and methodology for this tool is captured in the <a href="https://github.com/msnowden729-alt/storage-lcos-streamlit/blob/main/A%20Technoeconomic%20Assessment%20of%20Energy%20Storage%20Potential%20in%20Arctic%20Grid%20Systems_SNOWDEN.pdf" target="_blank">attached document</a>
""", unsafe_allow_html=True)

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

# Temperature inputs
st.subheader("Temperature Inputs (°C)")
tcol1, tcol2 = st.columns(2)

with tcol1:
    arctic_winter_low = st.number_input("Arctic Winter Low (°C)", value=-40)
    arctic_mean_annual = st.number_input("Arctic Mean Annual (°C)", value=-10)

with tcol2:
    baseline_winter_low = st.number_input("Baseline Winter Low (°C)", value=0)
    baseline_mean_annual = st.number_input("Baseline Mean Annual (°C)", value=20)

selected_Tamb = [
    arctic_winter_low,
    baseline_winter_low,
    arctic_mean_annual,
    baseline_mean_annual
]

common_inputs = {
    "Power": Power,
    "DD": DD,
    "charges_per_year": charges_per_year,
    "selected_Tamb": selected_Tamb,
    "Powercost": Powercost,
    "interest_rate": interest_rate,
    "project_lifespan": project_lifespan,
}


# ---------------------------------------------------------
# RUN button
# ---------------------------------------------------------
if st.button("Run Analysis"):

     # --- SHOW EXPLANATION WHILE CODE RUNS ---
    explanation_box = st.empty()
    explanation_box.markdown("""
    How the LCOS Model Works:

    This model compares **Levelized Cost of Storage (LCOS)** across five
    storage technologies in both **baseline** and **Arctic** scenarios.

    While the computation runs, the model evaluates:
    - **CAPEX Differences** (insulation, heaters, construction, FEED cost changes)
    - **OPEX Differences** (increased heating, reduced cooling, charging cost changes)
    - **Dispatched Storage Differences** (roundtrip efficiency and/or capacity changes)
    """)
    
    with st.spinner("Computing model outputs..."):
        f = io.StringIO()
        with redirect_stdout(f):
            outs = master_script.run(common_inputs)


    # --- REMOVE THE MESSAGE AFTER CALCULATION ---
    explanation_box.empty()
    
    results_list = outs.get("results_list", [])
    figs = outs.get("figures", [])

    # ---------------------------------------------------------
    # TABLE OF METRICS — COMPACT, RESPONSIVE, HTML-CENTERED
    # ---------------------------------------------------------
    if not results_list:
        st.error("No results were returned.")
    else:
        st.subheader("Key Metrics by Storage Technology")
    
        # Map raw program names to nicer names
        full_names = {
            "H2": "Hydrogen (H2)",
            "PHS": "Pumped Hydropower <br>Storage (PHS)",
            "BESS": "Lithium-Ion Battery<br>(BESS)",
            "CAES": "Adiabatic Compressed Air<br>Energy Storage (CAES)",
            "FLYWHEEL": "Flywheel"
        }
    
        metric_labels = [
            "Baseline CAPEX ($M)",
            "New CAPEX ($M)",
            "Baseline OPEX ($M)",
            "New OPEX ($M)",
            "Baseline LCOS ($/kWh)",
            "Arctic LCOS Change (%)"
        ]
    
        # Build DataFrame dynamically
        data = {}
    
        for res in results_list:
            prog_raw = res.get("program", "").replace("calcs", "").upper()
            prog_name = full_names.get(prog_raw, prog_raw)
    
            data[prog_name] = [
                res.get("baselineCAPEX", np.nan) / 1e6,
                res.get("newCAPEX", np.nan) / 1e6,
                res.get("baselineOPEX", np.nan) / 1e6,
                res.get("newOPEX", np.nan) / 1e6,
                res.get("baseLCOS", np.nan),
                res.get("LCOSchange", np.nan),
            ]
    
        df = pd.DataFrame(data, index=metric_labels)
        # ---- CENTER ALL CELLS + HEADERS ----
        df_styled = df.style.set_properties(**{
            'text-align': 'center'
        })
        df_styled = df_styled.set_table_styles([
            {"selector": "th", "props": [("text-align", "center")]}
        ])

    st.dataframe(
        df_styled.format("{:.2f}"),
        use_container_width=True,
        hide_index=False
    )
    
        # Display responsive dataframe
         st.dataframe(
            df.style.format("{:.2f}"),
            use_container_width=True,
            hide_index=False
        )

    # ---------------------------------------------------------
    # PLOTS — SHRINKED WITHOUT DISTORTION
    # ---------------------------------------------------------
    st.subheader("Generated Plots")
    plot_desc = [
    "Best Storage Technology Comparison: Mild (Left) vs. Arctic (Right) Climates",
    "Average Levelized Cost Change by Technology",
    "Minimum Levelised Cost Gradient in Mild (Left) vs. Arctic (Right) Climates, USD/kWh",
    ]

    for i, fig in enumerate(figs):
        # Save fig to buffer
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        buf.seek(0)

        # Load into PIL and resize proportionally
        img = Image.open(buf)
        w, h = img.size
        scale = 0.80  # shrink uniformly
        new_size = (int(w * scale), int(h * scale))
        img_resized = img.resize(new_size, Image.Resampling.LANCZOS)

        if i < len(plot_desc):
            caption_text = f"Plot {i+1}: {plot_desc[i]}"
        else:
            caption_text = f"Plot {i+1}"


        st.image(img_resized, caption=caption_text, use_container_width=False)
        st.markdown("---")

    # Debug console output
    with st.expander("Console Output (Debug)"):
        st.text(f.getvalue())
