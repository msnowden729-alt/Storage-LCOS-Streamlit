
import os
import matplotlib.pyplot as plt
import numpy as np
import math
import CoolProp.CoolProp as CP


def run(inputs: dict) -> dict:
        # Extract inputs
    P = inputs["Power"]  # Output Power
    DD = inputs["DD"]
    charges_per_year = inputs["charges_per_year"]
    selected_Tamb = inputs["selected_Tamb"]
    Powercost = inputs["Powercost"]
    r = inputs["interest_rate"]
    lifespan = inputs["project_lifespan"]

    specificinvcost = 600 #$/kW
    specificengcost = 3000 #$/kWh
    RTE = 0.88  #Check this


    HEXeff = 0.5  # assume a lower HEX efficiency because only stator can be heated and vacuum reduces heat transfer
    energydensity = 50 #kWh/m3, from storage ninja
    cyclelife = 200e3 # cycle life
    replcost = 3000 #$/kWh, from https://about.bnef.com/insights/commodities/lithium-ion-battery-pack-prices-see-largest-drop-since-2017-falling-to-115-per-kilowatt-hour-bloombergnef/
    coldmarkup = 0.15 # ave markup on fluidic components rated below -20C
    N = lifespan  
    num_replacements = math.floor(charges_per_year * lifespan/ cyclelife)
    installationmarkup = 1.2 #installation adds 20%
    replacement_cost = (P * DD * 1000 * replcost)* installationmarkup # per replacement, assumes that cost is driven by pack replacements, not inverters or other power electronics
    # Calculate replacement intervals
    replacement_interval = cyclelife / charges_per_year  # Years until replacement
    nrepl = [int(i * replacement_interval) for i in range(1, int(N / replacement_interval) + 1) if int(i * replacement_interval) <= N]
    # Ensure replacements don't exceed lifespan
    nrepl = [year for year in nrepl if year < N]
    print(f'Replacement Costs Per Replacement Interval = ${np.round(replacement_cost):,}')
    print(f'Years of Replacement = {nrepl}')


    #################################### HEATING CALCS #################################################
    insulationcostave = 600 # $/m2
    h = 0.1  # insulation thickness
    hk = .04 #W/m*K, insulation conductivity
    heatercost = 0.5 #$/W
    containervolume = P * DD * 1e3 / energydensity  #m3
    powerpercontainer = 2.5 # MW, from dinglun and stephentown FESS architecture
    powerpermodule = .25 # MW
    num_containers = math.ceil(P / powerpercontainer)  
    num_modules = math.ceil(P / powerpermodule)  
    percontainervol = containervolume / num_modules
    AR = 1 # assumes a cylinder with h = 1 * D
    containerd = (percontainervol*4/(np.pi*AR))**0.3333  #m
    h_cyl = containerd * AR
    underground_frac = 0.4 # % of FESS buried underground
    wallSA = (np.pi * containerd * h_cyl + 2 * (np.pi* containerd**2 )/ 4) * underground_frac #m2, surface area for tank insulation PER TANK
    L_steel = 0.002 # m
    k_steel = 50  # W/m·K
    R_walls = L_steel / (k_steel * wallSA)
    R_ins = h/(hk * wallSA) #assume insulation on walls, not floor.
    resistance_per_module = R_walls + R_ins #K/W, thermal resistance 
    print(f' thermal resistance = {resistance_per_module}')

    containerSA = 105.5 #m2, 40ft storage container minus floor
    resistance_per_container = h/(hk*containerSA) + (L_steel/(k_steel*containerSA)) # K/W
    

    print(f'num_modules = {num_modules:0.0f}')
    print(f'num containers = {num_containers:0.0f}')

    # calculate heating energy required to keep BESS >15C during idle cold periods
    deltaT = []
    T_target = -20  # assumed to be the low temp rating for most flywheels
    if selected_Tamb[0] < T_target:
         deltaT = T_target - selected_Tamb[0] 
    else:
         deltaT = 0
        
    Q_heat_pk = deltaT / resistance_per_module * num_modules #W, heat loss through container insulation
    Q_heat_pk = Q_heat_pk / HEXeff
    Q_heatcont_pk = deltaT / resistance_per_container / HEXeff * num_containers

    
    heatercapex = heatercost * Q_heat_pk
    operating_frac = (DD + DD/RTE)*charges_per_year/8760 # % time operational
    print(f'operating_frac = {operating_frac*100:0.2f}%')
    timebelowtarg = 0.1 #10% of the year estimated to be below -20C
    heating_period = 8760 * (1-operating_frac) * timebelowtarg  #hrs,
    heating_cost = (Q_heat_pk+Q_heatcont_pk)/1000 * heating_period * Powercost # USD per year
    print(f'Peak Heat Input required to keep FESS above -20C per container during idle = {Q_heat_pk/num_modules:0.0f}W') 
    print(f'Container Heating Cost per Year = ${np.round(heating_cost):,}')
    print(f'Heater CAPEX  = ${np.round(heatercapex):,}')

    ########### Calculate insulation cost #############
    insulationcost = insulationcostave * (wallSA) * h * num_modules + insulationcostave * (containerSA) * h * num_containers
    print(f'Container Insulation Cost = ${np.round(insulationcost):,}')

    ########## calculate cooling energy during operation #######################
    wasteheat = P/RTE * (1-RTE) #MW,  assumes all energy loss is heat (conservative)
    wasteheatpercontainer = wasteheat / num_containers #MW
    print(f'Waste heat per container = {wasteheatpercontainer * 1000:0.2f}kW')
    T_target = 50 #C
    cp_coolant = 3300  # J/kg*K, specific heat of 50:50 ethylene glycol-water
    rho_coolant = 1060  # kg/m^3, coolant density (assumed constant)
    eta_pump = 0.85
    
    ydata = [5.5, 5.5, 5.3, 5.0, 4.5, 4, 3.5, 3, 2.5, 2.2] #from https://www.researchgate.net/figure/COP-of-air-cooled-chiller_fig2_332625091
    xdata = np.arange(5,55,5)
    
    coeffs = np.polyfit(xdata, ydata, 2)
    # Create a polynomial function from the coefficients to calculate COP
    poly_func = np.poly1d(coeffs)
    cooling_power_op = []

    deltaT = []
  
    for Tamb in selected_Tamb:  # No need for enumerate unless you need the index
        COP = poly_func(Tamb)
        print(f'COP at {Tamb:0.0f}C = {COP:0.2f}')
        power = wasteheatpercontainer * 1e3 / COP * num_containers  # kW
        cooling_power_op.append(power)
        
    deltaQcool = (cooling_power_op[-1] - cooling_power_op[1]) #kW
    time_period = 8760 * (operating_frac) # assume cooling is needed whenever operational
    cooling_cost_op = deltaQcool * time_period * Powercost
    
    print(f'Savings in Operational Cooling Power at {selected_Tamb[0]:0.0f}C = {deltaQcool:0.3f}kW')
    
    COP = poly_func(selected_Tamb)

    ####### Calculate the Chiller Cost ############
    specificcost_chiller = 130 #$/kW
    chillercost = wasteheatpercontainer * 1000 * specificcost_chiller * num_containers #
    chiller_delta = chillercost * coldmarkup 
    
  
    EPCcostdelta = (heatercapex + insulationcost + chiller_delta) * 0.3  # assumes EPC cost change is equal to 30% of installed CAPEX cost change.

    baselineCAPEX = specificinvcost * P * 1e3 + specificengcost * P * 1e3 * DD #from storageninja
    CCfactor = 0.25 #uses the same locational markup that the NREL PHS uses
    constructioncostdelta = 0.2 * baselineCAPEX  * CCfactor # from https://energystorage.pnnl.gov/pdf/pnnl-28866.pdf
    print(f'Construction Cost Delta = ${np.round(constructioncostdelta):,}')
    print(f'Chiller Winterisation Cost = ${np.round(chiller_delta):,}')



    TotalCAPEXDelta = heatercapex + insulationcost + constructioncostdelta + chiller_delta + EPCcostdelta
    OPEXdelta = heating_cost - cooling_cost_op #sum of operational cost deltas
    baselinestorage = DD * P * charges_per_year   # kWh per year
    
    DoD = 1 #depth of discharge 
    nself = 0.1 #self discharge per cycle
    EoL = 0.95 #capacity at end of life
    Cyc_Deg = (1-EoL**(1/cyclelife)) #per-cycle degradation 
    T_Deg = 0.00  #temporal degradation per year
    Tc = 1.0 # construction time, years

    discbaselinestorage = baselinestorage * DoD * (1 - nself) * sum(((1 - Cyc_Deg)**((n-1) * charges_per_year) * (1 - T_Deg)**(n-1)) / ((1 + r)**(n + Tc)) for n in range(1, N+1))
    print(f'discounted storage = {np.round(discbaselinestorage):,}MWh')

    discreplacement_costs = sum(replacement_cost/ (1 + r)**(n + Tc) for n in nrepl)
    print(f'discounted replacement costs = ${np.round(discreplacement_costs):,}')

    
    chargingcost = Powercost * P*1000/RTE * DD * charges_per_year
    OMfactor = 1 #maintenance markup
    baselineOPEX = (5 * P * 1000 + 2 * P * DD)*OMfactor + chargingcost
    
    newCAPEX = baselineCAPEX + TotalCAPEXDelta
    
    newOPEX = baselineOPEX + OPEXdelta
    disc_OPEX = sum(baselineOPEX / (1 + r)**(n+Tc) for n in range(N))
    baseLCOS = (baselineCAPEX + discreplacement_costs + disc_OPEX) / (discbaselinestorage * 1000)  #$/kWh
    discnewOPEX = sum(newOPEX / (1 + r)**(n+Tc) for n in range(N))
    
    newLCOS = (newCAPEX + discreplacement_costs + discnewOPEX) / (discbaselinestorage * 1000)   #$/kWh
    
    LCOSchange = ((newLCOS-baseLCOS)/baseLCOS) * 100
     ################################### PIE CHART ###############################################

    constcontr = (constructioncostdelta)/(discbaselinestorage* 1000) # change to LCOS in $/MWh from construction costs
    heatcontr = (heatercapex + insulationcost)/(discbaselinestorage* 1000) # change to LCOS in $/MWh from heating CAPEX
    opexcontr = abs(discnewOPEX-disc_OPEX)/(discbaselinestorage* 1000) # change to LCOS from OPEX increase (lifetime discounted)
    chillercontr = (chiller_delta) / (discbaselinestorage * 1000)  #change to LCOS from storage cap decrease
    EPCcontr = (EPCcostdelta) / (discbaselinestorage * 1000)  #change to LCOS from storage cap decrease

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

    print(f'\nDiscounted Replacement costs= ${np.round(discreplacement_costs):,}')
    print(f'Discounted OPEX= ${np.round(discnewOPEX):,}')
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


# In[14]:


"""
test_inputs = {
    "Power": 100, #MW
    "DD": 0.0625,  #hrs
    "charges_per_year": 719.686,  
    "selected_Tamb": [-40,-10,0, 20], #temperatures for comparison
    "Powercost": 0.05, #USD/kWh
    "interest_rate": 0.08,
    "project_lifespan": 50
}

if __name__ == "__main__":
  results = run(test_inputs)
 
print(f'\nBaseline CAPEX = ${np.round(results['baselineCAPEX']):,}')
print(f'New CAPEX = ${np.round(results['newCAPEX']):,}')
OPEXdelta = results['newOPEX'] - results['baselineOPEX'] 
CAPEXdelta = results['newCAPEX'] - results['baselineCAPEX'] 
print(f'Baseline OPEX = ${np.round(results['baselineOPEX']):,}')
print(f'Change in Yearly OPEX = ${np.round(OPEXdelta):,}')
#print(f'Change in CAPEX = ${np.round(CAPEXdelta):,}')
print(f'Total LCOS Change = {results['LCOSchange']:0.2f}')
print(f'Baseline LCOS = ${results['baseLCOS']:0.3f}/kWh')
print(f'New LCOS = ${results['newLCOS']:0.3f}/kWh') """

