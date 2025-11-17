
import os
import matplotlib.pyplot as plt
import numpy as np
import math
import CoolProp.CoolProp as CP
from tabulate import tabulate  # Added for table formatting
from scipy.optimize import curve_fit

def run(inputs: dict) -> dict:
    
    # Extract inputs
    P = inputs["Power"]  # Output Power
    DD = inputs["DD"]
    charges_per_year = inputs["charges_per_year"]
    selected_Tamb = inputs["selected_Tamb"]
    Powercost = inputs["Powercost"]
    r = inputs["interest_rate"]
    lifespan = inputs["project_lifespan"]

    specificinvcost = 1300 #$/kW
    specificengcost = 40 #$/kWh


    ############################################ COMPRESSION  + EXPANSION CALCS #####################################################
    
    fluid = 'air'
    eta_turbine = 0.85
    Pstorage = 70*1e5 # Pa
    P_initial = 1*1e5 # Pa
    epsilon_intercool = 0.9 #no intercooling, we want the inlet temp to be as high as possible
    eta_comp = 0.85
    CD = DD * 1
    timebetween = np.round(((8760-(DD+CD)*charges_per_year))/charges_per_year/24) #average hours between charging and discharging
    heatlossperday = 0.5/100 # % per day
    remaining_frac = (1-heatlossperday)**timebetween
    print(f'Heat left after {timebetween}days = {remaining_frac*100:0.1f}%')
 

    densitycold = CP.PropsSI('D', 'T', selected_Tamb[1] + 273.15, 'P', Pstorage, 'Air')
    densitywarm = CP.PropsSI('D', 'T',  selected_Tamb[-1] + 273.15, 'P', Pstorage, 'Air')
    co = (densitycold/ densitywarm - 1) * 100
    print(co)

    # Multi Stage Compression Calcs
    # Calculate number of stages for both hydrogen and oxygen at room temperature
    total_pressure_ratio = Pstorage / P_initial
    ideal_ratio = 3
    P3 = Pstorage # turbine inlet pressure
    P4 = 1*1e5 # turbine outlet pressure
    n_stages = math.ceil(math.log(total_pressure_ratio) / math.log(ideal_ratio))
    pressure_ratio = total_pressure_ratio ** (1 / n_stages)
    ideal_exp_ratio = 8  #CHECK THIS
    n_exp_stages = math.ceil(math.log(Pstorage / P4) / math.log(ideal_exp_ratio))  # Fewer expansion stages
    exp_pressure_ratio = (Pstorage / P4) ** (1 / n_exp_stages)
    heatercost = 0.5 #$/W

    # Initialize arrays
    n_points = len(selected_Tamb)
    h2_values = np.zeros(n_points)
    h3 = np.zeros(n_points)
    h4 = np.zeros(n_points)
    s3 = np.zeros(n_points)
    T4s = np.zeros(n_points)
    deltah_exp = np.zeros(n_points)
    comp_power_MW = np.zeros(n_points)
    T3 = np.zeros(n_points)
    flowrate = np.zeros(n_points)  # Fixed initialization
    deltah_comp = np.zeros(n_points)
    airmass = np.zeros(n_points)
    airvol = np.zeros(n_points)
    tankd = np.zeros(n_points)
    tankSA = np.zeros(n_points)
    E_stored = np.zeros(n_points)
    TESvol = np.zeros(n_points)
    TESstoragecost = np.zeros(n_points)

    
    for i, T_C in enumerate(selected_Tamb):
        Tamb_K = T_C + 273.15
        delta_h_total = 0
        delta_h_stored_total = 0  # Total enthalpy extracted for storage
        P_current = P_initial
        T_current = Tamb_K
        T_values = []
        s_values = []
        P_values = []
    
        for stage in range(n_stages):  # cycle through compression stages
            P_next = P_current * pressure_ratio
            if stage == n_stages - 1:
                P_next = Pstorage
            h1 = CP.PropsSI('H', 'T', T_current, 'P', P_current, fluid)
            s1 = CP.PropsSI('S', 'T', T_current, 'P', P_current, fluid)
            T_values.append(T_current)
            s_values.append(s1)
            P_values.append(P_current/1e5)
            T2s_K = CP.PropsSI('T', 'S', s1, 'P', P_next, fluid)
           # T2s_K = T_current * (P_next / P_current) ** ((gamma - 1) / gamma)
            h2s = CP.PropsSI('H', 'T', T2s_K, 'P', P_next, fluid)
            delta_h_isentropic = h2s - h1
            delta_h_actual = delta_h_isentropic / eta_comp
            delta_h_total += delta_h_actual # compression delta h
            h2 = h1 + delta_h_actual  # Enthalpy after compression
            T2_K = CP.PropsSI('T', 'H', h2, 'P', P_next, fluid)  # Actual temperature after compression
            # Extract heat to cool air back to Tamb_K (or partially, based on epsilon_intercool)
            if stage == n_stages:
               T_in_next = Tamb_K
            else:
                T_in_next = T2_K - epsilon_intercool * (T2_K - Tamb_K)
            u2 = CP.PropsSI('U', 'T', T2_K, 'P', P_next, fluid)  # Internal energy after compression
            u_cooled = CP.PropsSI('U', 'T', T_in_next, 'P', P_next, fluid)  # Internal energy after cooling
            delta_u_extracted = u2 - u_cooled  # Thermal energy extracted
            delta_h_stored_total += delta_u_extracted
            P_current = P_next
            T_current = T_in_next
            T_values.append(T2_K)
            s_values.append(s1)
            P_values.append(P_current/1e5)
        
        deltah_comp[i] = delta_h_total
        T_values.append(Tamb_K)
        s_storage = CP.PropsSI('S', 'T', Tamb_K, 'P', Pstorage, fluid)
        s_values.append(s_storage)
        P_values.append(Pstorage/1e5)
        # Apply heat loss to total stored enthalpy
        delta_h_available = delta_h_stored_total * (remaining_frac) 
        delta_h_per_stage = delta_h_available / n_exp_stages 
        delta_h_exp_total = 0
        P_current = Pstorage
        T_current = Tamb_K
        
        for stage in range(n_exp_stages):
            h_in = CP.PropsSI('H', 'T', T_current, 'P', P_current, fluid)
            h3_stage = h_in + delta_h_per_stage 
            T3_stage = CP.PropsSI('T', 'H', h3_stage, 'P', P_current, fluid)
            s3_stage = CP.PropsSI('S', 'T', T3_stage, 'P', P_current, fluid)
            P_next = P_current / exp_pressure_ratio
            if stage == n_exp_stages - 1:
                P_next = P4
            P_values.append(P_current/1e5)
            P_values.append(P_next/1e5)

            T4s_stage = CP.PropsSI('T', 'S', s3_stage, 'P', P_next, fluid) #isentropic expansion
            h4_stage = CP.PropsSI('H', 'T', T4s_stage, 'P', P_next, fluid)
            delta_h_exp_s = h3_stage - h4_stage
            delta_h_exp_stage = delta_h_exp_s * eta_turbine
            delta_h_exp_total += delta_h_exp_stage
            h4 = h3_stage - delta_h_exp_stage
            T4 = CP.PropsSI('T', 'H', h4, 'P', P_next, fluid) 
            P_current = P_next
            T_current = T4
            T_values.append(T3_stage)
            T_values.append(T4)
            s_values.append(s3_stage)
            s_values.append(s3_stage)
        # Reheat air to T3 using available enthalpy
        deltah_exp[i] = delta_h_exp_total 
        
        flowrate[i] = P * 1e6 / deltah_exp[i]  #solve for flowrate required to achieve P output
        powerin = flowrate[i] * deltah_comp[i]   #compressor power = input power
        comp_power_MW[i] = powerin / 1e6 

        ################## Air Storage ###########################
        airmass[i] = flowrate[i]*3600*DD #kg
        rho_storage = CP.PropsSI('D', 'T', Tamb_K, 'P', Pstorage, fluid)
        airvol[i] = airmass[i]/rho_storage

        ################# TES Tank Cost ###########################
        
        E_stored[i] = flowrate[i] * delta_h_stored_total * CD / 1000 # kWh thermal
        capacity = [160e3, 480e3, 28268e3] # kWh, 1-s2.0-S1364032121000290-main.pdf
        scc = [6.105, 2.343, 0.759] # specific capacity cost, $/kWh thermal
        kWh_per_m3 = [58.175, 84.291, 232.174] # kWhth/m3, capacity per unit volume
        energydensity = np.mean(kWh_per_m3)  #assume that energy density doesn't scale with volume if stored in overground tanks
        TESvol[i] = E_stored[i] / energydensity #m3
        num_tanks = math.ceil(TESvol[i]/1570) # limits tank diameter to ~20m per tank
        volpertank = TESvol[i]/num_tanks
        
        tankAR = 1 #assuming cylindrical tanks, this is the heigh/dia
        tankd[i] = (volpertank*4/(np.pi*tankAR))**0.3333  #m
        
        h_cyl = tankd[i]*tankAR
        tankSA[i] = (np.pi * tankd[i] * h_cyl + (np.pi* tankd[i]**2 )/ 4) * num_tanks #m2, surface area for tank insulation PER TANK

        def power_law(x, A, b):
            return A * x**b
        # Fit the model to the data
        params, covariance = curve_fit(power_law, capacity, scc)
        A_fit, b_fit = params
        
        TESstoragecost[i] = (A_fit*(E_stored[i])**(b_fit)) * E_stored[i] # $

    RTE = P/comp_power_MW
    print(f'RTE = {RTE}')
    print(f'Thermal Energy Stored = {np.round(E_stored[-1]):,}kWh')
    print(f'TES cost =${np.round(TESstoragecost[-1]):,}')
    print(f'TES tank diameter = {tankd[-1]:0.2f}m')
    
  
    """
    print(f'\n Thermal Energy Capacity at {selected_Tamb[0]}C = {E_stored[0]:0.0f}kWh ')
    print(f' Thermal Energy Capacity at {selected_Tamb[-1]}C = {E_stored[-1]:0.0f}kWh ')
    print(f' Thermal Storage Cost at {selected_Tamb[0]}C = = ${np.round(TESstoragecost[0]):,}')
    print(f' Thermal Storage Cost at {selected_Tamb[-1]}C = = ${np.round(TESstoragecost[-1]):,}')
    print(f' Thermal Storage Volume at {selected_Tamb[0]}C = {TESvol[0]:0.0f}m3 \n')


    print(f'\n RTE at {selected_Tamb[-1]}C =  {RTE[-1]} ')
    print(f'\n RTE at {selected_Tamb[0]}C =  {RTE[0]} ')
   
        # Tabulate thermodynamic properties
    table_data = []
    for i, T_C in enumerate(selected_Tamb):
        table_data.append([
            T_C,
            f"{deltah_comp[i]:.2f}",  #compression
            f"{deltah_exp[i]:.2f}"  #expansion
        ])
        
    headers = ["Tamb (°C)", "deltah, comp", "deltah, exp"]
    print("\nThermodynamic Properties Table:")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


    # Calculate deltas
    compdelta = (comp_power_MW[0] - comp_power_MW[-1]) * 1e3  # kW
    print(f' Compressor Power at {selected_Tamb[-1]}C = {comp_power_MW[-1]:0.5f}MW ' )
    print(f' Compressor Power at {selected_Tamb[0]}C = {comp_power_MW[0]:0.5f}MW ' )
    print(f' Air flowrate required to achieve {P:0.1f}MW at {selected_Tamb[0]}C= {flowrate[0]:0.3f}kg/s')
    print(f' Air flowrate required to achieve {P:0.1f}MW at {selected_Tamb[-1]}C= {flowrate[-1]:0.3f}kg/s')
    print(f' Compression Power Delta = {np.round(compdelta):,}kW')
    print(f' Capacity increase at {selected_Tamb[0]}C = {co:0.1f}%')

        # Set up figure and axes
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))  # 2 rows, 2 columns
    
    # Flatten the 2D array of axes for easier indexing
    axs = axs.flatten()
    
    axs[0].plot(selected_Tamb, comp_power_MW)
    axs[0].set_title('Compressor Power (MW) vs. Tamb')
    
    axs[1].plot(selected_Tamb, flowrate)
    axs[1].set_title('Flowrate (kg/s) vs. Tamb')

    axs[2].plot(selected_Tamb, deltah_comp)
    axs[2].set_title('Enthalpy Change, Compression vs. Tamb')

    axs[3].plot(selected_Tamb, deltah_exp)
    axs[3].set_title('Enthalpy Change, Expansion vs. Tamb')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))  # 1 row, 2 columns
    axes[0].plot(s_values, T_values)
    axes[0].set_title('T-s Diagram')
    axes[0].set_xlim(2500, 4500)

    axes[1].plot(s_values, P_values)
    axes[1].set_title('P-s Diagram')
    axes[1].set_xlim(2500, 4500)


    # Tabulate thermodynamic properties
    table_data = []
    for i, T in enumerate(T_values):
        table_data.append([
            f"{T}",  #compression
            f"{P_values[i]}",
            f"{s_values[i]}"  #expansion
        ])
        
    headers = [ "Stage Temperature", "Stage Pressure", "Stage Entropy"]
    print(f"\nThermodynamic Properties Table for {n_stages} compression stages & {n_exp_stages} expansion stages:")
    print(tabulate(table_data, headers=headers, tablefmt="grid")) """

    
    ##################################### REPLACEMENT COSTS ##################################################
    
    DoD = 1 #depth of discharge, assumes isobaric (versus isochoric) storage  
    nself = 0.00 #self discharge per cycle
    EoL = 1.0 #capacity at end of life
    cyclelife = 15000 # cycle life, driven by power electronics (from StorageNinja)
    #Cyc_Deg = (1-EoL**(1/cyclelife)) #per-cycle degradation 
    if charges_per_year * lifespan < cyclelife:   #if the project lifespan is shorter than the technology cycle life, extend lifespan to match (NTE 50 yrs)
        lifespan = min(np.round(cyclelife/charges_per_year),50)
    
    Cyc_Deg = 0
    T_Deg = 0  #temporal degradation
    Tc = 2 # construction time, years
    N = int(lifespan)
    num_replacements = math.floor(charges_per_year * lifespan/ cyclelife)
    installationmarkup = 1.2 #installation adds 20%
    specificreplcost = 100 # $/kW, from storageninja
    replacement_cost = (P * 1000 * specificreplcost)* installationmarkup 
    # Calculate replacement intervals
    replacement_interval = cyclelife / charges_per_year  # Years until replacement
    nrepl = [int(i * replacement_interval) for i in range(1, int(N / replacement_interval) + 1) if int(i * replacement_interval) <= N]
    # Ensure replacements don't exceed lifespan
    nrepl = [year for year in nrepl if year < N]
    print(f'Replacement Costs Per Replacement Interval = ${np.round(replacement_cost):,}')
    print(f'Years of Replacement = {nrepl}')
    
    discreplacement_costs = sum(replacement_cost/ (1 + r)**(n + Tc) for n in nrepl)
    print(f'discounted replacement costs = ${np.round(discreplacement_costs):,}')

    ######################################## CAPEX + OPEX COSTS #############################################
    baselineCAPEX = specificinvcost * P * 1000 + specificengcost * P * 1000 * DD + TESstoragecost[-1]  #warm temp CAPEX
    insulationcostpervol = 600 # $/m3, from 1-s2.0-S0960148123007826-main.
    h = 0.1 # baseline insulation thickness

    hk = .04 #W/m*K
    tanktemp = 90 # C
    baselineresistance = h / (tankSA[-1] * hk) # K/W
    baselineheatloss = (tanktemp-selected_Tamb[-1])/ baselineresistance
    newresistance = (tanktemp - selected_Tamb[1]) / baselineheatloss #calculates new resistance value for arctic storage, assuming the same amount of heat loss 
    newthickness = newresistance * tankSA[1] * hk
    insulationdelta = (tankSA[1] * newthickness - tankSA[-1] * h) * insulationcostpervol #$
    powerdensity = 4.4 #kW/m3, combined compressor and turbine specific volume (including ancillary systems + motor/generator)
    enclosurevolume = (P + P/RTE[-1])*1000 / powerdensity #m3
    enclosureh = 7 #m, assume ceiling height
    enclosurel = np.sqrt(enclosurevolume/enclosureh)
    enclosureSA = enclosurel **2 + 4 * enclosureh * enclosurel

    T_target = -10  #C, min operating temp for compressor + turbine
    insulationcostave = 600 # $/m3
    enclosureins = enclosureSA * insulationcostave * h
    enclosureR = h/(hk*enclosureSA)  #thermal resistance of enclosure
    Qenclosure = 0 # initialize
    if selected_Tamb[0] < T_target:
        Qenclosure = (T_target - selected_Tamb[0])/enclosureR  # W
    enclosureheaters = Qenclosure * heatercost
    print(f'Enclosure Heater + Insulation Cost = ${np.round(enclosureheaters + enclosureins):,}')
    timebelowtarg = 0.25 #assumes that XX% of the year is below T_target
    operatingfrac = DD*charges_per_year/8760
    enclosureQcost = Powercost * Qenclosure/1e3 * 8760 * timebelowtarg * (1 - operatingfrac)  #  assumes heating is only required during idle periods
    print(f' Powertrain Enclosure yearly heating cost = ${np.round(enclosureQcost):,}')
    
    constcost_perkW = 226.4 #$/kW from https://docs.google.com/spreadsheets/d/13HWn32tSYMLPjDXOJrflOcg5EeYtLtOeTcO1oFOudxI/edit?gid=977639313#gid=977639313
    constcost_perkWh = 21.43 #$/kWh from https://docs.google.com/spreadsheets/d/13HWn32tSYMLPjDXOJrflOcg5EeYtLtOeTcO1oFOudxI/edit?gid=977639313#gid=977639313
    CCfactor = 0.25 # using NREL locational markup factor, 0.25
    constructioncostdelta = (constcost_perkW * P *1000 + constcost_perkWh * P * 1000 * DD) *  CCfactor 
    CAPEXdelta = insulationdelta + enclosureins #assumes that underground air storage isn't possible in the Arctic
    EPCdelta = 0.3*(CAPEXdelta)
    
    print(f' Cost delta for Arctic Construction = ${np.round(constructioncostdelta):,}')
    print(f' Cost delta for TES insulation = ${np.round(insulationdelta):,}')
    
    newCAPEX = baselineCAPEX + CAPEXdelta + EPCdelta + constructioncostdelta + enclosureheaters + enclosureins

    charging_cost = comp_power_MW * 1000 * DD * charges_per_year * Powercost
    
    OMfactor = 1 #maintenance markup

    baselineOPEX = (14 * P * 1000 + 2 * P * DD)*OMfactor + charging_cost[-1]  # from Storage Ninja

    baselinestorage = DD * P * charges_per_year   # MWh per year    
    disc_OPEX = sum(baselineOPEX / (1 + r)**(n+Tc) for n in range(N))
    discbaselinestorage = baselinestorage * DoD * (1 - nself) * sum(((1 - Cyc_Deg)**((n-1) * charges_per_year) * (1 - T_Deg)**(n-1)) / ((1 + r)**(n + Tc)) for n in range(1, N+1))
    print(f'Yearly Charging Cost at {selected_Tamb[-1]}C = ${np.round(charging_cost[-1]):,}')
    print(f'Yearly Charging Cost at {selected_Tamb[1]}C = ${np.round(charging_cost[1]):,}')

    
    print(f' discounted baseline storage = {np.round(discbaselinestorage):,}MWh ' )
    
    newOPEX =  (14 * P * 1000 + 2 * P * DD)*OMfactor + charging_cost[1] + enclosureQcost 
    discnewOPEX = sum(newOPEX / (1 + r)**(n+Tc) for n in range(N))
    baseLCOS = (baselineCAPEX + discreplacement_costs + disc_OPEX) / (discbaselinestorage * 1000)
    newLCOS = (newCAPEX + discreplacement_costs + discnewOPEX) / (discbaselinestorage * 1000)
    LCOSchange = ((newLCOS-baseLCOS)/baseLCOS) * 100
    
    if DD <= 0.25:
        baseLCOS = 1e6 #$/MWh, making LCOS artificially high because PHS can't respond in time for short discharges
        newLCOS = 1e6 #$/MWh, making LCOS artificially high because PHS can't respond in time for short discharges
        LCOSchange = 0

    constcontr = (constructioncostdelta)/(discbaselinestorage* 1000) # change to LCOS in $/MWh from construction costs
    heatcontr = (CAPEXdelta)/(discbaselinestorage* 1000) # change to LCOS in $/MWh from heating CAPEX
    opexcontr = abs(discnewOPEX-disc_OPEX)/(discbaselinestorage* 1000) # change to LCOS from OPEX increase (lifetime discounted)
    EPCcontr = (EPCdelta) / (discbaselinestorage * 1000)  #change to LCOS from storage cap decrease

    sumchange =  constcontr + heatcontr + opexcontr + EPCcontr

    percentconst = constcontr/sumchange * 100
    percentheat = heatcontr/ sumchange * 100
    percentopex = opexcontr/ sumchange * 100
    percentEPC = EPCcontr/ sumchange * 100

    print(f' Construction Cost Markup = ${np.round(constructioncostdelta):,}')
    print(f' Heater & Insulation CAPEX = ${np.round(CAPEXdelta):,}')
    print(f' EPC Markup = ${np.round(EPCdelta):,}')
    #print(f' OPEX Change (disc. over lifetime) = ${np.round(discnewOPEX-disc_OPEX):,}')
    print(f' OPEX  (disc. over lifetime) = ${np.round(discnewOPEX):,}')

    print(f' % Contribution to LCOS change, Construction Cost Markup = {percentconst:0.2f}%')
    print(f' % Contribution to LCOS change, Heater & Insulation CAPEX = {percentheat:0.2f}%')
    print(f' % Contribution to LCOS change, EPC Markup = {percentEPC:0.2f}%')
    print(f' % Contribution to LCOS change, OPEX (disc. over lifetime) = {percentopex:0.2f}%')


    return {
        "baselinestorage": baselinestorage,
        "baselineCAPEX": baselineCAPEX,
        "baselineOPEX": baselineOPEX,
        "newCAPEX": newCAPEX,
        "newOPEX": newOPEX,
        "baseLCOS": baseLCOS,
        "newLCOS": newLCOS,
        "LCOSchange": LCOSchange,
    }
        


# In[95]:


"""
test_inputs = {
    "Power": 100, #MW
    "DD": 1,  #hrs
    "charges_per_year": 372.76,  
    "selected_Tamb": [-40, -10, 0, 20], #Arctic low, Arctic mean, warm place low, warm place mean temperature
    "Powercost": 0.05, #USD/kWh
    "interest_rate": 0.08,
    "project_lifespan": 50,
}

if __name__ == "__main__":
  results = run(test_inputs)
 
print(f'\nBaseline CAPEX = ${np.round(results['baselineCAPEX']):,}')
print(f'New CAPEX = ${np.round(results['newCAPEX']):,}')
OPEXdelta = results['newOPEX'] - results['baselineOPEX'] 
CAPEXdelta = results['newCAPEX'] - results['baselineCAPEX'] 
print(f'Baseline OPEX = ${np.round(results['baselineOPEX']):,}')
print(f'Change in Yearly OPEX = ${np.round(OPEXdelta):,}')
print(f'Total LCOS Change = {results['LCOSchange']:0.5f}')
print(f'Baseline LCOS = ${results['baseLCOS']:0.5f}/kWh')
print(f'New LCOS = ${results['newLCOS']:0.5f}/kWh') """


# In[ ]:




