

import os
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import curve_fit


## Main Program ##
def run(inputs: dict) -> dict:
    # Extract inputs
    P = inputs["Power"]  # Output Power
    DD = inputs["DD"]
    charges_per_year = inputs["charges_per_year"]
    selected_Tamb = inputs["selected_Tamb"]
    Powercost = inputs["Powercost"]
    r = inputs["interest_rate"]
    lifespan = inputs["project_lifespan"]

    energydensity = 93 #kWh/m3
    insulationcostave = 600 # $/m3
    RTE = .86 # average over lifespan (3500 cycles)
    HEXeff = 0.75
    cyclelife = 3.5e3 # cycle life
    perpackcost = 115 #$/kWh, from https://about.bnef.com/insights/commodities/lithium-ion-battery-pack-prices-see-largest-drop-since-2017-falling-to-115-per-kilowatt-hour-bloombergnef/
    coldmarkup = 0.15 # ave markup on fluidic components rated below -20C
    N = lifespan  
    num_replacements = math.floor(charges_per_year * lifespan/ cyclelife)
    installationmarkup = 1.2 #installation adds 20%
    replacement_cost = (P * DD * 1000 * perpackcost)* installationmarkup # per replacement, assumes that cost is driven by pack replacements, not inverters or other power electronics
    # Calculate replacement intervals
    replacement_interval = cyclelife / charges_per_year  # Years until replacement
    nrepl = [int(i * replacement_interval) for i in range(1, int(N / replacement_interval) + 1) if int(i * replacement_interval) <= N]
    # Ensure replacements don't exceed lifespan
    nrepl = [year for year in nrepl if year < N]
    print(f'Replacement Costs Per Replacement Interval = ${np.round(replacement_cost):,}')
    print(f'Years of Replacement = {nrepl}')


    #################################### HEATING CALCS #################################################
    h = 0.1  # insulation thickness
    hk = .04 #W/m*K, insulation conductivity
    heatercost = 0.5 #$/W
    Prating = 2 #MW, average power rating of grid-scale BESS from Saft, Tesla, and CATL
    Erating = 3.4 #MWh
    percontainervol = 41 #m3, average of grid-scale BESS from Saft, Tesla, and CATL
    num_containers = math.ceil(max(P*DD /Erating, P/Prating))  
    containerl = 7 # assumes 20ft storage container
    containerd = np.sqrt(percontainervol/containerl)
    floorSA = containerl * containerd
    wallSA = (2 * containerd**2 + 4 * containerl * containerd ) - floorSA # assumes a container of dimensions d x d x 40ft
    k_plywood = 0.12  # W/m·K
    L_plywood = 0.028  # m
    L_steel = 0.002 # m
    k_steel = 50  # W/m·K
    R_walls_ceiling = L_steel / (k_steel*wallSA)
    R_floor = L_plywood/(k_plywood * floorSA)
    R_ins = h/(hk * wallSA) #assume insulation on walls, not floor.
    resistance_per_container = 1/(1/(R_walls_ceiling+R_ins) + (1/R_floor)) #thermal resistance in parallel equation

    print(f'num_containers = {num_containers:0.0f}')

    
    #calculate heating energy required to keep BESS >15C during idle cold periods
    deltaT = []
    T_target = 15
    if selected_Tamb[0] < T_target:
         deltaT = T_target - selected_Tamb[0] 
    else:
         deltaT = 0
        
    Q_heat_peak = deltaT / resistance_per_container * num_containers #W, heat loss through container insulation
    Q_heat_peak = Q_heat_peak / HEXeff
    heatercapex = heatercost * Q_heat_peak
    meanyearlytemp = 0 #C
    Q_heat = (T_target-meanyearlytemp) / resistance_per_container * num_containers #W, heat loss through container insulation

    operating_frac = (DD + DD/RTE)*charges_per_year/8760 # % time operational
    print(f'operating_frac = {operating_frac*100:0.2f}%')
    
    heating_period = 8760 * (1-operating_frac) #hrs
    heating_cost = Q_heat/1000 * heating_period * Powercost # USD per year
    print(f'Total Heat Input required to keep BESS at 15C per container during idle = {Q_heat/num_containers:0.0f}W')
    
    print(f'Container Heating Cost per Year = ${np.round(heating_cost):,}')
    print(f'Heater CAPEX  = ${np.round(heatercapex):,}')

    ########### Calculate insulation cost #############
    insulationcost = insulationcostave * (wallSA) * h * num_containers
    print(f'Container Insulation Cost = ${np.round(insulationcost):,}')

    ########## calculate cooling energy during operation #######################
    wasteheat = P/RTE * (1-RTE) #MW,  assumes all energy loss is heat (conservative)
    wasteheatpercontainer = wasteheat/num_containers #MW
    print(f'Waste heat per container = {wasteheatpercontainer* 1000:0.2f}kW')
    T_target = 25 #C
    #deltaP_pump = 200e3 # Pa, standard industrial liquid coolant pressure drop
    cp_coolant = 3300  # J/kg*K, specific heat of 50:50 ethylene glycol-water
    rho_coolant = 1060  # kg/m^3, coolant density (assumed constant)
    eta_pump = 0.85
    
    ydata = [5.5, 5.5, 5.3, 5.0, 4.5, 4, 3.5, 3, 2.5, 2.2] #from https://www.researchgate.net/figure/COP-of-air-cooled-chiller_fig2_332625091
    xdata = np.arange(5,55,5)
    
    coeffs = np.polyfit(xdata, ydata, 2)
    # Create a polynomial function from the coefficients to calculate COP
    poly_func = np.poly1d(coeffs)

    deltaT = []
    COP_cold = poly_func(meanyearlytemp)
    COP_hot = poly_func(selected_Tamb[-1])
    print(f' COP at {meanyearlytemp} = {COP_cold:0.2f}')
    print(f' COP at {selected_Tamb[-1]} = {COP_hot:0.2f}')
        
    deltaQcool = (1/ COP_hot - 1/ COP_cold) * wasteheatpercontainer * 1e3 * num_containers
    time_period = 8760 * (operating_frac) # assume cooling is needed whenever operational
    cooling_cost_op = deltaQcool * time_period * Powercost
    
    print(f'Savings in Operational Cooling Power at {selected_Tamb[0]:0.0f}C = {deltaQcool:0.3f}kW')
    
    COP = poly_func(selected_Tamb)

    #calculate the cooling energy required to keep BESS <25C during idle hot periods
    if selected_Tamb[-1] > T_target:
        deltaT = (selected_Tamb[-1] - T_target) 
        heatinput = deltaT/resistance_per_container  #W
        acpower = heatinput / COP[-1] /1000 * num_containers #kW
        timeabove25 = 0.25 # in a warm place, temps could be >25C for 25% of the time
        time_period = timeabove25 * 8760 * (1-operating_frac)
        cooling_cost = acpower * time_period * Powercost

    else:
        acpower = 0
        cooling_cost = 0
    print(f'Savings in Idle Cooling Power at {selected_Tamb[0]:0.0f}C = {acpower:0.3f}kW')
    print(f'Total Savings in Yearly Cooling Costs at {selected_Tamb[0]:0.0f}C = ${np.round(cooling_cost + cooling_cost_op):,}')

    ######## Calculate Construction Cost ##########
    spec_constcost_P = 49 # $/kW, from https://atb.nrel.gov/electricity/2021/utility-scale_battery_storage
    spec_constcost_E = 12 # $/kWh, from https://atb.nrel.gov/electricity/2021/utility-scale_battery_storage

    constructioncost =(P*1000*DD)* spec_constcost_E + (P*1000)* spec_constcost_P 
    CCfactor = 0.25 #uses the same locational markup that the NREL PHS uses
    constructioncostdelta = constructioncost * CCfactor 

    ####### Calculate the Chiller Cost ############
    specificcost_chiller = 130 #$/kW
    chillercost = wasteheatpercontainer * 1000 * specificcost_chiller * num_containers #
    chiller_delta = chillercost * coldmarkup 
    
    print(f'Construction Cost Delta = ${np.round(constructioncostdelta):,}')
    print(f'Chiller Winterisation Cost = ${np.round(chiller_delta):,}')

    specificinvcost = 250 # $/kW
    specificengcost = 300 #$/kWh
    
    baselineCAPEX = specificinvcost * P * 1e3 + specificengcost * P * 1e3 * DD #from storageninja
    EPCfactor = 0.3
    EPCdelta = (heatercapex + insulationcost + chiller_delta)* EPCfactor

    TotalCAPEXDelta = heatercapex + insulationcost + constructioncostdelta + chiller_delta + EPCdelta

    OPEXdelta = heating_cost - cooling_cost_op - cooling_cost #sum of operational cost deltas
    baselinestorage = DD * P * charges_per_year   # kWh per year
    
    DoD = 0.8 #depth of discharge 
    nself = 0.01 #self discharge per cycle
    EoL = 0.8 #capacity at end of life
    Cyc_Deg = (1-EoL**(1/cyclelife)) #per-cycle degradation 
    T_Deg = 0.01  #temporal degradation per year
    Tc = 1 # construction time, years

    discbaselinestorage = baselinestorage * DoD * (1 - nself) * sum(((1 - Cyc_Deg)**((n-1) * charges_per_year) * (1 - T_Deg)**(n-1)) / ((1 + r)**(n + Tc)) for n in range(1, N+1))

    discreplacement_costs = sum(replacement_cost/ (1 + r)**(n + Tc) for n in nrepl)
    print(f'discounted replacement costs = ${np.round(discreplacement_costs):,}')

    
    chargingcost = Powercost * P*1000/RTE * DD * charges_per_year
    
    OMfactor = 1 #maintenance markup

    baselineOPEX = (5 * P * 1000 + 0.4 * P * DD)*OMfactor + chargingcost
    disc_OPEX = sum(baselineOPEX / (1 + r)**(n+Tc) for n in range(N))
    newCAPEX = baselineCAPEX + TotalCAPEXDelta
    
    newOPEX = baselineOPEX + OPEXdelta
    discnewOPEX = sum(newOPEX / (1 + r)**(n+Tc) for n in range(N))


    baseLCOS = (baselineCAPEX + discreplacement_costs + disc_OPEX) / (discbaselinestorage * 1000)  #$/kWh
    
    newLCOS = (newCAPEX + discreplacement_costs + discnewOPEX) / (discbaselinestorage * 1000)   #$/kWh
    
    LCOSchange = ((newLCOS-baseLCOS)/baseLCOS) * 100

    ################################### PIE CHART ###############################################

    constcontr = (constructioncostdelta)/(discbaselinestorage* 1000) # change to LCOS in $/MWh from construction costs
    heatcontr = (heatercapex + insulationcost)/(discbaselinestorage* 1000) # change to LCOS in $/MWh from heating CAPEX
    opexcontr = abs(discnewOPEX-disc_OPEX)/(discbaselinestorage* 1000) # change to LCOS from OPEX increase (lifetime discounted)
    chillercontr = (chiller_delta) / (discbaselinestorage * 1000)  #change to LCOS from storage cap decrease
    EPCcontr = (EPCdelta) / (discbaselinestorage * 1000)  #change to LCOS from storage cap decrease

    sumchange =  constcontr + heatcontr + chillercontr + opexcontr + EPCcontr

    percentconst = constcontr/sumchange * 100
    percentheat = heatcontr/ sumchange * 100
    percentopex = opexcontr/ sumchange * 100
    percentchiller = chillercontr/ sumchange * 100
    percentEPC = EPCcontr/ sumchange * 100

    print(f' Contribution to LCOS change, Construction Cost Markup = ${constcontr:0.3f}')
    print(f' Contribution to LCOS change, Heater & Insulation CAPEX = ${heatcontr:0.3f}')
    print(f' Contribution to LCOS change, Chiller Markup = ${chillercontr:0.3f}')
    print(f' Contribution to LCOS change, EPC Markup = ${EPCcontr:0.3f}')
    print(f' Contribution to LCOS change, OPEX (disc. over lifetime) = ${opexcontr:0.3f}')

    print(f' % Contribution to LCOS change, Construction Cost Markup = {percentconst:0.2f}%')
    print(f' % Contribution to LCOS change, Heater & Insulation CAPEX = {percentheat:0.2f}%')
    print(f' % Contribution to LCOS change, EPC Markup = {percentEPC:0.2f}%')
    print(f' % Contribution to LCOS change, Chiller Winterisation Markup = {percentchiller:0.2f}%')
    print(f' % Contribution to LCOS change, OPEX (disc. over lifetime) = {percentopex:0.2f}%')

    print(f'Discounted Replacement costs= ${np.round(discreplacement_costs):,}')
    print(f' OPEX= ${np.round(newOPEX-chargingcost):,}')
    print(f'Discounted storage= {np.round(discbaselinestorage):,}MWh')



    return {
        "baselineCAPEX": baselineCAPEX,
        "baselineOPEX": baselineOPEX,
        "newCAPEX": newCAPEX,
        "newOPEX": newOPEX,
        "baseLCOS": baseLCOS,
        "newLCOS": newLCOS,
        "LCOSchange": LCOSchange,
    }


# In[91]:


"""
test_inputs = {
    "Power": 100, #MW
    "DD": 1,  #hrs
    "charges_per_year": 372.76,  
    "selected_Tamb": [-40, -10, 0, 20], #Arctic low, Arctic mean, warm place low, warm place mean temperature
    "Powercost": 0.2, #USD/kWh
    "interest_rate": 0.08,
    "project_lifespan": 50,
}

if __name__ == "__main__":
    import math
    results = run(test_inputs)

print(f'\n\nBaseline CAPEX = ${np.round(results['baselineCAPEX']):,}')
print(f'New CAPEX = ${np.round(results['newCAPEX']):,}')
OPEXdelta = results['newOPEX'] - results['baselineOPEX'] 
CAPEXdelta = results['newCAPEX'] - results['baselineCAPEX'] 
print(f'Baseline OPEX = ${np.round(results['baselineOPEX']):,}')
print(f'Change in Yearly OPEX = ${np.round(OPEXdelta):,}')
#print(f'Change in CAPEX = ${np.round(CAPEXdelta):,}')
print(f'Total LCOS Change = {results['LCOSchange']:0.2f}')
print(f'Baseline LCOS = ${results['baseLCOS']:0.3f}/kWh')
print(f'New LCOS = ${results['newLCOS']:0.3f}/kWh') """


# In[ ]:




