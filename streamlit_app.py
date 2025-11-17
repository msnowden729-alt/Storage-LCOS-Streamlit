import streamlit as st
import master_script
import os

st.title("Arctic Energy Storage LCOS Model")

# User inputs
Power = st.number_input("Power (MW)", value=100)
DD = st.number_input("Discharge Duration (hours)", value=1)
charges_per_year = st.number_input("Charges per Year", value=372.76)
selected_Tamb = st.multiselect("Ambient Temperatures (°C)", [-40, -10, 0, 20], default=[-40, -10, 0, 20])
Powercost = st.number_input("Power Cost (USD/kWh)", value=0.2)
interest_rate = st.number_input("Discount Rate", value=0.08)
project_lifespan = st.number_input("Project Lifespan (years)", value=50)

if st.button("Run LCOS Model"):
    inputs = {
        "Power": Power,
        "DD": DD,
        "charges_per_year": charges_per_year,
        "selected_Tamb": selected_Tamb,
        "Powercost": Powercost,
        "interest_rate": interest_rate,
        "project_lifespan": project_lifespan
    }

    st.write("Running calculations, please wait...")
    results = master_script.run(inputs)

    st.subheader("Console Output")
    st.text(results["console_log"])

    st.subheader("Generated Plots")
    for name, filename in results["plot_files"].items():
        if os.path.exists(filename):
            st.image(filename, caption=name)
