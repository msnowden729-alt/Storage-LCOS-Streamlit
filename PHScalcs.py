
import os
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import curve_fit, brentq

def run(inputs: dict) -> dict:
    # Extract inputs
    P = inputs["Power"]  # Output Power
    DD = inputs["DD"]
    charges_per_year = inputs["charges_per_year"]
    selected_Tamb = inputs["selected_Tamb"]
    Powercost = inputs["Powercost"]
    r = inputs["interest_rate"]
    lifespan = inputs["project_lifespan"]

    cyclelife = 30e3
    #if charges_per_year * lifespan < cyclelife:   #if the project lifespan is shorter than the technology cycle life, extend lifespan to match 
    #    lifespan = min(np.round(cyclelife/charges_per_year),50)
    head = 300            # Head (m)
    g = 9.81 # Gravitational acceleration (m/s^2)
    usable_frac = 0.85     # Fraction of usable reservoir volume
    rho_water = 1000       # Density of water (kg/m^3)
    k_ice = 2.2            # Thermal conductivity of ice (W/m·K)
    rho_ice = 917          # Density of ice (kg/m^3)
    L_f = 334000           # Latent heat of fusion for water (J/kg)
    T_freeze = 0           # Freezing point of water (°C)
    cp_water = 4184       # Specific heat of water (J/kg·K), approximated for consistency
    specificinvcost = 1100 # $/kW, from storage ninja
    specificengcost = 50 # $/kWh, from storage ninja
    CD = DD * 1   #charging duration, hrs

    
    # Stefan Model Constants (Reservoir Ice Growth)
    r_snowice = 4.9                # Ratio of snow to ice conductivities
    days = int(math.ceil((8760 - (DD + CD) * charges_per_year) / 24 / charges_per_year)) # average idle time between charges, in days
    dt = 86400             # Time step in seconds (1 day)
    
    # Trash Rack Model Constants
    k_water = 0.58         # Thermal conductivity of water (W/m·K)
    k_steel = 16.0         # Thermal conductivity of steel (W/m·K)
    mu_water = 1.787e-3    # Dynamic viscosity of water at 0°C (Pa·s)
    nu_water = mu_water / rho_water  # Kinematic viscosity (m²/s)
    Pr_water = cp_water * mu_water / k_water  # Prandtl number
    flow_velocity = 6     # Flow velocity (m/s), from NREL PHS calculator
    HExEff = 0.75           # Heat exchanger effectiveness
    slantangle = 60        # Trash rack angle (degrees)
    num_racks = 2          # Number of trash racks
    T_target = 5           # Target temperature for bars (°C)
    T_out = T_freeze       # Water temperature (°C), same as T_freeze
    RTE = 0.8              # Roundtrip efficiency for PHS
    Pin = P / RTE  #charging power 
    
    # Lifetime Cost Parameters (for Trash Rack Heating)
    coatingcost = 100e3    # Low-adhesion coating cost ($)
    
    heatercost = 0.5            # Capital cost for resistive heaters ($/W)
    CAPEXhp = 3.5          # Capital cost for ground source heat pump ($/W)
    COP = 3.0              # Coefficient of performance for heat pump
    bar_spacings = np.linspace(0.1, 0.5, 21)  # Range of bar spacings for plot (m)
    
    # Reservoir Calculations (Shared)
    E = P * DD  # Energy per discharge (MWh)
    Volume = E * 1e6 * 3600 / (rho_water * g * head) / usable_frac  # m^3
    mass = Volume * rho_water  # kg
    flowrate = Volume/DD/3600
    print(f' flowrate = {flowrate}m3/s')

    def heat_input_trash_rack(bar_diameter, bar_spacing, num_bars, bar_length, T_target, T_out):

        # Convective heat transfer coefficient for water
        Re = flow_velocity * bar_diameter / nu_water
        term1 = 0.62 * Re**0.5 * Pr_water**(1/3)
        term2 = (1 + (0.4 / Pr_water)**(2/3))**0.25
        term3 = (1 + (Re / 282000)**(5/8))**(4/5)
        Nu = 0.3 + term1 / term2 * term3
        h_water = Nu * k_water / bar_diameter
        
        surface_area_per_bar = surface_area_per_bar = np.pi * bar_diameter * bar_length  #m2
        
        # Heat loss per bar via convection
        Q_conv_per_bar = h_water * surface_area_per_bar * (T_target - T_out)   
        
        # Total heat loss for all bars
        Q_total_bars = Q_conv_per_bar * num_bars / HExEff * num_racks  #W
        
        return Q_total_bars, Q_conv_per_bar, num_bars

    tunnel_area = P*1e6 / (rho_water * g * head * flow_velocity)
    tunneld = np.sqrt(4 * tunnel_area / np.pi)
    bar_length = tunneld / np.sin(slantangle * np.pi / 180)
    bar_diameter = max(bar_length/200,0.05)    # Assumes Outer diameter of hollow steel bars (m), 200:1 L/D down to d = 0.05m
    maxheadloss = 0.01*head #m
    ck = 2.42 #Kirschmer form factor, for sharp edged bars (conservative)
    def headloss_equation(bs):  #Meusberger's equation for head loss as a function of trash rack geometry
        if bs <= bar_diameter or bs <= 0:
            return np.inf  # Return a large value to discourage invalid bs
        term1 = ck*(bar_diameter / bs)**1.33
        term2 = ((bs - bar_diameter) / tunneld)**-0.43
        term3 = flow_velocity**2 / (2 * g) * np.sin(np.radians(slantangle))
        return maxheadloss - term1 * term2 * term3

    lower_bound = bar_diameter * 1.05  # just above diameter
    upper_bound = 0.5  # meters (adjust as needed)
    
    # Solve using brentq
    try:
        bar_spacing = brentq(headloss_equation, lower_bound, upper_bound)
        print(f"Bar spacing to achieve max head loss: {bar_spacing:.4f} m")
    except ValueError as e:
        print("Failed to find a root in the given interval:", e)

    num_bars = np.round(tunneld / bar_spacing)
    (Q_required, Q_per_bar, num_bars) = heat_input_trash_rack(bar_diameter, bar_spacing, num_bars, bar_length, T_target, T_out)
    Eusage = Q_required * (DD + CD)  # Wh
    yearlyEusage = (Eusage * charges_per_year)/2 #assumes that heating is only needed for half the total charge cycles.
    heatingOPEX = yearlyEusage / 1000 * Powercost
    heatingOPEXhp = heatingOPEX / COP
    heatingCAPEX = heatercost * Q_required # $, price for electric heaters
    heatingCAPEXhp = CAPEXhp * Q_required # $, price for GSHP
    #Ecostshp = heatingOPEXhp * lifespan + heatingCAPEXhp
    #Ecosts = heatingOPEX * lifespan + heatingCAPEX
    Tc = 3 # construction time, years
    N = int(lifespan)  # assumes PHS will last longer than project lifespan
    Ecosts = sum(heatingOPEX / (1 + r)**(n+Tc) for n in range(N)) +heatingCAPEX
    Ecostshp = sum(heatingOPEXhp / (1 + r)**(n+Tc) for n in range(N)) +heatingCAPEXhp

    
    Eloss_trash = Eusage / (P*1e6 * DD) * 100  # %

    print(f"Trash Rack Model Results (Single Case, bar_spacing = {bar_spacing:.1f} m):")
    print(f"Tunnel Diameter: {tunneld:.3f} m")
    print(f"Bar Length: {bar_length:.3f} m")
    print(f"Total heat input required: {Q_required:.2f} W ({Q_required/1e6:.2f} MW)")
    print(f"Energy usage per charge cycle: {Eusage:.2f} Wh")
    print(f"Energy loss per charge cycle: {Eloss_trash:.2f} %\n")
    

    if Ecostshp < Ecosts:
        heating_CAPEX = heatingCAPEXhp
        heating_OPEX = heatingOPEXhp
    else:
        heating_CAPEX = heatingCAPEX
        heating_OPEX = heatingOPEX
        
    print(f'Heater cost = ${np.round(heating_CAPEX):,}')
#################################################### STEFAN ICE MODEL #################################################################
    
       
    def stefan_ice_model(selected_Tamb, h_snow_fixed, AR_fixed):

        time = np.arange(days)
        Tamb = (selected_Tamb[0] + selected_Tamb[1]) / 2  #mean winter temp
       
        # Line plot calculations
        # Surface area for fixed AR
        depth = (3 * Volume / (np.pi * (3 * AR_fixed - 1)))**0.3333333333333333
        a = np.sqrt(max(0, 2 * depth * depth * AR_fixed - depth**2))
        SA = np.pi * a * a

        # Ice thickness
        h = np.zeros(days)
        h[0] = 0.01
            
        # Simulate ice growth
        for i in range(1, days):
            T_s = r_snowice * h[i-1] * Tamb / (h_snow_fixed + r_snowice * h[i-1])
            q = k_ice * (T_freeze - T_s) / h[i-1]
            dh = (q * dt) / (rho_ice * L_f)
            h[i] = h[i-1] + max(dh, 0)   #ice thickness
            
        # Energy loss and capacity
        mass_ice = h[-1] * SA * rho_ice
        Eloss = mass_ice / (mass * usable_frac) * 100
        Ecap = E * (1 - Eloss / 100) # MWh
        h_final = h[-1]
        
        return Ecap, h_final, depth

    # Stefan Ice Model Inputs
    h_snow_fixed = 0.5  # Fixed snow depth (m)
    AR_fixed = 50  # Fixed aspect ratio

    Ecap, h_final, depth = stefan_ice_model(selected_Tamb, h_snow_fixed, AR_fixed)

    print(f'Ice thickness after {days:0.0f}days at mean winter temperature = {h_final:0.3f}m')
    missingcap = (E - Ecap) #MWh
    Ecaploss = missingcap * charges_per_year / 2 # kWh, assume that this capacity loss only occurs for half the year
    print(f'Yearly Capacity Loss due to Icing = {Ecaploss:0.0f}MWh')
    print(Ecaploss/E)

    #######################################################################################################################
    ## Baseline CAPEX Cost, Including Direct + Indirect Costs (from NREL PHS Calculator)
    critical_DD = specificinvcost/specificengcost # discharge duration above which Construction Costs are driven by energy, not power

    baselineCAPEX = specificinvcost * P * 1000 + specificengcost * P * 1000 * DD
    
     #######################################################################################################################
    ## Construction Cost
    """
    PPstructurecost_P = 105 #$/kW, approximation from NREL PHS calculator.  True value will depend on dam height
    PPstructurecost = PPstructurecost_P * P * 1000
    
    MWh = [90*10, 1005*18.5]  #upper reservoir volume from NREL calculator, in cubic yards
    Damcost = np.array([25.5e6, 185.8e6])  # $, from NREL PHS calculator
    
    # Define power function: CostperkW = A * kW^(-b)
    def power_func(MWh, A, b):
        return A * MWh**(-b)
    
    # Fit the power function to the data
    popt, _ = curve_fit(power_func, MWh, Damcost, p0=[10000, 0.2])  # Initial guess: A=10000, b=0.2
    A, b = popt
    
    TotalDamCost = A*(E)**(-1*b) 
    print(f'Reservoir Construction Cost = ${np.round(TotalDamCost):,}')
    conveyanceL_ft = head / np.sin(6 * np.pi/180) * 3.28 # water tunnel conveyance length, in feet, assumes an effective slope of 6 deg per NREL avg
    tunnel_area_ft = tunnel_area * 10.76 #m2 to ft2
    
    xdata = [89.9, 934.8] # tunnel area in ft2, from NREl
    ydata = [4798, 24333] # tunnel cost per ft, from NREL
    slope, intercept = np.polyfit(xdata, ydata, 1) # trends a relationship between tunnel cost per ft and tunnel xsec area

    tunnelcostperft = tunnel_area_ft* slope + intercept
    tunnelcost = tunnelcostperft * conveyanceL_ft
    print(f'conveyance length (ft) = {conveyanceL_ft:0.0f}')
    print(f'Waterway Construction Cost per ft = ${np.round(tunnelcostperft):,}')

    print(f'Waterway Construction Cost = ${np.round(tunnelcost):,}')
    
    CCfactor = 0.25 #Arctic Construction markup, from NREL
    locational_costs = (tunnelcost + TotalDamCost + PPstructurecost) * CCfactor """
    specific_CC_P = 594  #$/kW, simplified from calculations above ^
    specific_CC_E = 24 #$/kWh, simplified from calculations above ^
    constructioncost = specific_CC_P * P * 1000 + specific_CC_E * P * DD * 1000
    CCfactor = 0.25
    locational_costs = constructioncost * CCfactor
    
    #######################################################################################################################
    ## Calculate the cost deltas
    
    #locational_costs = baselineCAPEX * 0.25 #NREL tool shows a 25% increase in total CAPEX when factoring in Arctic locational costs
    
    underground_delta = 67e6 #constant offset of building power station underground
    
    print(f'Added cost of an underground power station = ${np.round(underground_delta):,}') 
        
    print(f'Added Locational Costs of a {P:0.0f}MW PHS in the Arctic = ${np.round(locational_costs):,} ')  #This needs an uncertainty
        
    EPCfactor = 0.3
    
    EPCcost = (heating_CAPEX) * EPCfactor

    sumCAPEXdeltas = underground_delta + heating_CAPEX + locational_costs + EPCcost


    
    print(f'Total Cost Delta at -20C = ${np.round(sumCAPEXdeltas):,}')

    chargingcost = Powercost * P*1000/RTE * DD * charges_per_year
    print(f' Charging Cost per Year = ${np.round(chargingcost):,}')

    
    OMfactor = 1
    baselineOPEX = (20 * P * 1000 + 0.4 * P*DD)*OMfactor + chargingcost #from storageninja specific O&M costs
    
    baselinestorage = DD * P * charges_per_year   # MWh per year
    newstorage = baselinestorage - Ecaploss # MWh

    DoD = 1 #depth of discharge 
    nself = 0.00 #self discharge per cycle
    EoL = 0.95 #capacity at end of life
    cyclelife = 10e3 # cycle life
    #Cyc_Deg = (1-EoL**(1/cyclelife)) #per-cycle degradation 
    Cyc_Deg = 0
    T_Deg = 0  #temporal degradation
    discbaselinestorage = baselinestorage * DoD * (1 - nself) * sum(((1 - Cyc_Deg)**((n-1) * charges_per_year) * (1 - T_Deg)**(n-1)) / ((1 + r)**(n + Tc)) for n in range(1, N+1))
    print(f'discounted baseline storage = {np.round(discbaselinestorage):,}MWh')
    discnewstorage = newstorage * DoD * (1 - nself) * sum(((1 - Cyc_Deg)**((n-1) * charges_per_year) * (1 - T_Deg)**(n-1)) / ((1 + r)**(n + Tc)) for n in range(1, N+1))
    print(f'discounted storage in Arctic = {np.round(discnewstorage):,}MWh')

    disc_OPEX = sum(baselineOPEX / (1 + r)**(n+Tc) for n in range(N))

    baseLCOS = (baselineCAPEX + disc_OPEX) / (discbaselinestorage * 1000)  #$/kWh
    print(f'discounted OPEX = ${np.round(disc_OPEX):,}')
    
    newCAPEX = baselineCAPEX + sumCAPEXdeltas
    
    newOPEX = baselineOPEX + heating_OPEX

    discnewOPEX = sum(newOPEX / (1 + r)**(n+Tc) for n in range(N))
    newLCOS = (newCAPEX + discnewOPEX) / (discnewstorage * 1000)   #$/kWh
    
    LCOSchange = ((newLCOS-baseLCOS)/baseLCOS) * 100
    print(f'discounted OPEX change = ${np.round(discnewOPEX-disc_OPEX):,}')

    
    print(f'Total LCOS Change = {LCOSchange:0.2f}%')
   
    if DD <= 0.25:
        baseLCOS = 1e6 #$/MWh, making LCOS artificially high because PHS can't respond in time for short discharges
        newLCOS = 1e6 #$/MWh, making LCOS artificially high because PHS can't respond in time for short discharges
        LCOSchange = 0

    ########################## Make a pie chart for LCOS contributors ###############################################
    
    constcontr = (underground_delta + locational_costs)/(discbaselinestorage* 1000) # change to LCOS in $/MWh from construction costs
    heatcontr = (heating_CAPEX)/(discbaselinestorage* 1000) # change to LCOS in $/MWh from heating CAPEX
    opexcontr = np.abs(discnewOPEX-disc_OPEX)/(discbaselinestorage* 1000) # change to LCOS from OPEX increase (lifetime discounted)
    storagecontr = (baselineCAPEX + disc_OPEX) / (discnewstorage * 1000)- baseLCOS #change to LCOS from storage cap decrease
    EPCcontr = EPCcost/(discbaselinestorage* 1000)
    sumchange =  constcontr + heatcontr + storagecontr + opexcontr + EPCcontr

    percentconst = constcontr/sumchange * 100
    percentheat = heatcontr/ sumchange * 100
    percentEPC = EPCcontr/sumchange * 100
    percentopex = opexcontr/ sumchange * 100
    percentstor = storagecontr/ sumchange * 100
    
    print(f' Contribution to LCOS change, Construction Cost Markup = ${constcontr:0.3f}')
    print(f' Contribution to LCOS change, Heater & Insulation CAPEX = ${heatcontr:0.3f}')
    print(f' Contribution to LCOS change, FEED costs = ${EPCcontr:0.3f}')
    print(f' Contribution to LCOS change, Storage Reduction (disc. over lifetime) = ${storagecontr:0.3f}')
    print(f' Contribution to LCOS change, OPEX (disc. over lifetime) = ${opexcontr:0.3f}')

    print(f' % Contribution to LCOS change, Construction Cost Markup = {percentconst:0.2f}%')
    print(f' % Contribution to LCOS change, Heater & Insulation CAPEX = {percentheat:0.2f}%')
    print(f' % Contribution to LCOS change, FEED costs = {percentEPC:0.2f}%')
    print(f' % Contribution to LCOS change, OPEX (disc. over lifetime) = {percentopex:0.2f}%') 
    print(f' % Contribution to LCOS change, Storage Reduction (disc. over lifetime) = {percentstor:0.2f}%')


    
    return {
            "baselineCAPEX": baselineCAPEX,
            "baselineOPEX": baselineOPEX,
            "newCAPEX": newCAPEX,
            "newOPEX": newOPEX,
            "baseLCOS": baseLCOS,
            "newLCOS": newLCOS,
            "LCOSchange": LCOSchange
        }


# In[17]:


"""
test_inputs = {
    "Power": 100, #MW
    "DD": 4,  #hrs
    "charges_per_year": 791.7,  
    "selected_Tamb": [-40,-10, 0, 20], #temperatures for comparison
    "Powercost": 0.05, #USD/kWh
    "interest_rate": 0.08,
    "project_lifespan": 30
}

if __name__ == "__main__":
    results = run(test_inputs)

print(f'\n\nBaseline CAPEX = ${np.round(results['baselineCAPEX']):,}')
print(f'New CAPEX = ${np.round(results['newCAPEX']):,}')
OPEXdelta = results['newOPEX'] - results['baselineOPEX'] 
CAPEXdelta = results['newCAPEX'] - results['baselineCAPEX'] 
print(f'Baseline OPEX = ${np.round(results['baselineOPEX']):,}')
print(f'Change in Yearly OPEX = ${np.round(OPEXdelta):,}')
#print(f'Change in CAPEX = ${np.round(CAPEXdelta):,}')
print(f'Total LCOS Change = {results['LCOSchange']:0.2f}')
print(f'Baseline LCOS = ${results['baseLCOS']:0.4f}/kWh')
print(f'New LCOS = ${results['newLCOS']:0.4f}/kWh') """


# In[ ]:




