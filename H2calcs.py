
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
    cost = inputs["Powercost"]
    r = inputs["interest_rate"]
    lifespan = inputs["project_lifespan"]
        
    # Uncommon inputs
    specificinvcost = 5000 #USD per kW, per storage ninja
    specificengcost = 30 #USD per kWh, per storage ninja
    coldmarkup = 0.15 #cold-rated components are ~15% on top of OTS price
    HEXeff = 0.75 #assumed for all heat exchangers

    
    # Storage inputs
    Pstorage = 300 * 1e5  # Pa, oxygen storage pressure
    Pstorageh = 300 * 1e5  # Pa, hydrogen storage pressure
    P_initial = 30 * 1e5  # Pa, electrolyzer output
    RT = 25  # deg C, temperature of H2 gas at electrolyser outlet
    RT_K = RT + 273
    eta_comp = 0.85  # isentropic efficiency of compressor
    CF = 0.75 # capacity factor, i.e. how much of added storage is actually used 

    
    # Compressor inputs
    fluid = "oxygen"
    ideal_ratio = 3
    epsilon_intercool = 0.9
    OtoH = 8  # 8X more O2 than H2 (by mass) produced per kg H2O
    
    # Membrane cooling inputs
    T_stack = 70 # C, desired membrane temperature for both electrolyzer and FC
    eff = 55  # kWh/kg, energy input per kg H2 produced
    eta_fc = 0.5  # fuel cell efficiency
    LHVH2 = 33.33 # kWh/kg H2
    parasitic_power = 0  # kW, assumes no extra power draw
    eta_ely = 0.7  # electrolyzer efficiency (HHV)
    cp_coolant = 3300  # J/kg*K, specific heat of 50:50 ethylene glycol-water
    rho_coolant = 1060  # kg/m^3, coolant density (assumed constant)
    eta_pump = 0.85 
    powerdensity = 100 #kW / m3, for container volume calc
    
    # Water heating inputs
    watertoh2 = 15  # kg water per kg H2, including cooling needs
    wdens = 1000 # kg/m^3
    cpwater = 4184 # J/kg*K

    # Derived inputs
    RTE = eta_fc * eta_ely
    Pin = P / RTE  #electrolyser power
    print(f'Pin = {Pin:0.0f}MW')
    flowrate = Pin * 1e3 / eff / 3600  # kg/s, H2 flow production
    CD = DD * 1 # charging duration is assumed to be the same as DD
    flowrate_discharge = flowrate * CD / DD * 3600 #kg/h H2
    operating_frac = (DD + CD) * charges_per_year/8760 # % time operational
    print(f'operating_frac = {operating_frac*100:0.2f}%')

    # COMPRESSION CALCS

    # Capacity gain calc, H2
    density = CP.PropsSI('D', 'T', selected_Tamb[1] + 273, 'P', Pstorageh, 'Hydrogen')
    density700 = CP.PropsSI('D', 'T', selected_Tamb[1] + 273, 'P', 700 * 1e5, 'Hydrogen')
    densityRT = CP.PropsSI('D', 'T', selected_Tamb[-1] + 273, 'P', Pstorageh, 'Hydrogen')
    densityRT700 = CP.PropsSI('D', 'T', selected_Tamb[-1] + 273, 'P', 700 * 1e5, 'Hydrogen')
    c = (density / densityRT - 1) * 100
    c700 = (density700 / densityRT700 - 1) * 100
    h2storagevol = flowrate_discharge * DD / densityRT #m3
    print(f'H2 Storage Volume = {h2storagevol}m3')

    # Multi Stage Compression Calcs
    power_results = {}
    # Calculate number of stages for both hydrogen and oxygen at room temperature
    total_pressure_ratio_h2 = Pstorageh / P_initial
    n_stages_h2 = math.ceil(math.log(total_pressure_ratio_h2) / math.log(ideal_ratio))
    total_pressure_ratio_o2 = Pstorage / P_initial
    n_stages_o2 = math.ceil(math.log(total_pressure_ratio_o2) / math.log(ideal_ratio))
    
    for fluid in ["hydrogen", "oxygen"]:
        target_pressure = Pstorageh if fluid.lower() == "hydrogen" else Pstorage
        n_stages = n_stages_h2 if fluid.lower() == "hydrogen" else n_stages_o2
        flowrate_adjusted = flowrate * OtoH if fluid.lower() == "oxygen" else flowrate
        n_points = len(selected_Tamb)
        initial_density = np.zeros(n_points)
        gamma_array = np.zeros(n_points)
        hact_array = np.zeros(n_points)
        power_MW = np.zeros(n_points)

        for i, T_C in enumerate(selected_Tamb):
            Tamb_K = T_C + 273.15
            # Use fixed inlet pressure (no isochoric cooling)
            Pout = P_initial
            initial_density[i] = CP.PropsSI('D', 'T', Tamb_K, 'P', Pout, fluid)
            
            total_pressure_ratio = target_pressure / Pout
            pressure_ratio = total_pressure_ratio ** (1 / n_stages)
            delta_h_total = 0
            P_current = Pout
            T_current = Tamb_K

            for stage in range(n_stages):
                inlet_density = CP.PropsSI('D', 'T', T_current, 'P', P_current, fluid)
                cp_in = CP.PropsSI('C', 'T', T_current, 'P', P_current, fluid)
                gamma = CP.PropsSI('CPMASS', 'T', T_current, 'P', P_current, fluid) / \
                        CP.PropsSI('CVMASS', 'T', T_current, 'P', P_current, fluid)
                P_next = P_current * pressure_ratio
                if stage == n_stages - 1:
                    P_next = target_pressure
                h1 = CP.PropsSI('H', 'T', T_current, 'P', P_current, fluid)
                T2s_K = T_current * (P_next / P_current) ** ((gamma - 1) / gamma)
                h2s = CP.PropsSI('H', 'T', T2s_K, 'P', P_next, fluid)
                delta_h_isentropic = h2s - h1
                delta_h_actual = delta_h_isentropic / eta_comp
                delta_h_total += delta_h_actual
                cp_in = CP.PropsSI('C', 'T', T_current, 'P', P_current, fluid)
                T_in_next = T2s_K - epsilon_intercool * (T2s_K - Tamb_K)
                P_current = P_next
                T_current = T_in_next

            gamma_array[i] = gamma
            hact_array[i] = delta_h_total / 1000
            power = flowrate_adjusted * delta_h_total
            power_MW[i] = power / 1e6

        power_results[fluid] = power_MW

    # Calculate deltas
    ocompdelta = (power_results["oxygen"][1] - power_results["oxygen"][-1]) * 1e3  # kW
    hcompdelta = (power_results["hydrogen"][1] - power_results["hydrogen"][-1]) * 1e3
    hcapdelta = c # capacity reduction (%) from selected_Tamb[1] to [-1]
    compression_cost = (ocompdelta + hcompdelta) * CD * charges_per_year * cost

    print(f'ocomp/P = {ocompdelta/1e3/Pin*100:0.3f}%')
    print(f'hcomp/P = {hcompdelta/1e3/Pin*100:0.3f}%')
    print(f'Ox comp Power = {power_results["oxygen"][-1]:0.3f}MW')
    print(f'H2 comp Power = {power_results["hydrogen"][-1]:0.3f}MW')

    ######################################################### WATER CALCS ########################################################################
    
    #Calculation of water volume to supply one discharge of H2
    flowrateh = flowrate * 3600 #kg/h H2 production
    totalmass = CD*flowrateh*watertoh2  #kg H20 required for one discharge of H2
    totalvol = totalmass / wdens #m3
    totalvol_L = totalvol*1000 #liters
    maxtanksize = 2e6 #Liters
    #tankmaxcapacity = 1e6 #L or Kg H20, mass per tank
    #numtanks = math.ceil(totalmass / tankmaxcapacity)
    if totalvol_L > maxtanksize:
        numtanks = np.round(totalvol_L/maxtanksize)+1

    else:
        numtanks = 2 #assumes 2 tanks, one pre- and two post DI water treatment / filtering
    mass = totalmass / numtanks #kg, mass per tank
    volume = mass / wdens #m3, volume per tank
    

    tankAR = 1 #assuming cylindrical tanks, this is the heigh/dia
    tankd = (volume*4/(np.pi*tankAR))**0.3333  #m
    h_cyl = tankd*tankAR
    tankSA = (np.pi * tankd * h_cyl + 2 * (np.pi* tankd**2 )/ 4)  #m2, surface area for tank insulation PER TANKz
    
    print(f'Tank Volume required: {totalmass / 1e6 / numtanks: 0.1f} ML')
    print(f'Tank Diameter required: {tankd: 0.1f} m')
    print(f'Number of tanks required: {numtanks: 0.0f} ')
    #print(tankSA)
    
    # Inputs
    heatercost = 0.5 # $/W
    hk = 0.04 #W/m*K, insulation conductivity, assuming cork properties
    Tinit = 10 #C, initialisation temperature of water tank
    idletime = (1-operating_frac)*8760 # assumes conservatively that idle time occurs over winter (if longer than winter, assumed that heating is only needed over 6 months/ 180days)
    insulationcost = 600 # $/m3
    h = 0.1 #m 
    k_values = []
    L_steel = 0.002
    k_steel = 50  # W/m·K
    L_steel / (k_steel)
    Rsteel = L_steel / (k_steel) #thermal resistance of steel tank
    Rins = h/(hk) # thermal resistance of insulation
    R = Rsteel + Rins  # total thermal resistance
    print(f'Thermal Resistance (K/W/m2) = {R}')
    
    # Time array
    dt = 1  # hr, time step (DO NOT CHANGE)
    time = np.arange(0, idletime + dt, dt)  
    n_steps = len(time)
    
    T_tank = np.zeros((len(selected_Tamb), n_steps))  # [Tamb, time]
    heat_input = np.zeros(len(selected_Tamb))  # MWh
    P_heater = np.zeros(len(selected_Tamb))  # W (steady state at T=0°C)
    heating_cost = np.zeros(len(selected_Tamb))  # $
    clamp_time = np.zeros(len(selected_Tamb))  # time steps
    
    # Check for division by zero
    if h <= 0:
        raise ValueError("Insulation thickness h must be positive")
    
    # Loop through ambient temperatures
    for j, T_amb_C in enumerate(selected_Tamb):
        T_current = Tinit
        T_tank[j, 0] = Tinit
        clamped = False  # Flag to track if temperature is clamped
        Q_heater = 0
        for t in range(1, n_steps):
            if clamped:
                # Temperature is clamped to 0°C, heater power is constant
                T_current = 0
                Q_heater = (T_current - T_amb_C) / R * tankSA # W
                Q_heater = max(Q_heater, 0)  # Ensure non-negative
            else:
                # Natural cooling
                Q_loss = (T_current - T_amb_C) / R * tankSA # W
                dT_dt = -Q_loss / (mass * cpwater) * 3600  # °C/hr
                T_current += dt * dT_dt
            if T_current < 0:
                clamped = True
                T_current = 0
                clamp_time[j] = t  # Store time step when clamping occurs
    
            T_tank[j, t] = T_current
    
        if clamp_time[j] > 0:
            # Heater power for maintaining 0°C
            P_heater[j] = max((1 - T_amb_C) / R / HEXeff * tankSA * numtanks, 0)  # W, heat power to keep water at 1C
            # Convert clamp_time to seconds for energy calculation
            heat_input[j] = (P_heater[j] * (idletime - clamp_time[j] * dt)) / 1e6  # MWh
            heating_cost[j] = heat_input[j] * cost * 1000  # Convert MWh to kWh for cost
        else:
            P_heater[j] = 0
            heat_input[j] = 0
            heating_cost[j] = 0
            
    print(f'Steady State Heater Power Required to Keep Water Tanks >0C at {selected_Tamb[0]}C= {P_heater[0]/1e3:0.2f}kW')
    print(f'Yearly Energy Required to Keep Water Tanks >0C = {heat_input[1]:0.2f}MWh')
    tankheatercost = P_heater[0] * heatercost # $ uses yearly low temp
    tank_heating_cost = heat_input[1] * 1e3 * cost # uses mean temp
    print(f'average water tank heat = {P_heater[1]/1e6/Pin * 100:0.3f}%')
    
    ######################################################
    ## Preheating of water entering PEM stack
    h = 0.1
    k_s = (hk * tankSA) / (h * mass * cpwater)  # 1/s
    k = k_s * 3600  # Convert to 1/hr
    k_values.append(k)
    target_velocity = 2 #m/s, standard for industrial waterflow applications
    
    waterflow = flowrate * (9) # kg/s, inlet waterflow to electrolyzer, 9:1 mass ratio
    A = waterflow/(wdens * target_velocity)
    d = np.sqrt(4*A/np.pi)  # m, pipe diameter (example value, adjust as needed)
    L = 100  # m, pipe length (example value, adjust as needed)
    
    # Initialize arrays
    Q_total = np.zeros(len(selected_Tamb))  # MWh
    pipeSA_per_m = np.pi * d  # m^2/m, pipe surface area per unit length (inner surface)
    
    # Calculate heat input for each Tamb
    for i, T_amb_C in enumerate(selected_Tamb):
        # Set initial temperature based on Tamb
        T_initial_w = 1.0 if T_amb_C <= 0 else T_amb_C  # °C
        print(f"Tamb = {T_amb_C}°C, T_initial = {T_initial_w}°C")
        
        # Calculate sensible heat
        Q_sensible = waterflow * cpwater * (T_stack - T_initial_w)  # W
       #  print(f"Sensible heat to raise water from {T_initial}°C to {T_stack}°C: {Q_sensible/1e6:.3f} MW")
        
        # Calculate heat loss
        T_water_avg = (T_initial_w + T_stack) / 2  # °C, average water temperature
        r_inner = d / 2  # m, inner radius
        r_outer = r_inner + h  # m, outer radius
        q_loss_per_m = (2 * np.pi * k * (T_water_avg - T_amb_C)) / np.log(r_outer / r_inner)  # W/m
        Q_loss = q_loss_per_m * L  # W
        
        # Total heat input
        Q_total[i] = (Q_sensible + Q_loss)/HEXeff  # W
        print(f"Total heat input for water entering PEM stack: {Q_total[i]/1e6:.3f} MW")
    
    #################################### OX GAS PRE-HEATING CALCS ####################################################
    
    deltaT = T_stack - selected_Tamb[1]
    
    flowrate_DD_O2 = flowrate_discharge/3600 * OtoH 
    cp_o = 918 # J/kg*K
    
    Qin_ox = flowrate_DD_O2 * cp_o * deltaT / 1e3 / HEXeff # kW
    print(f' Oxygen gas pre-heating (FC) = {Qin_ox/1e3/Pin*100:0.3f}%')
    print(f'ox flow = {flowrate_DD_O2:0.2f}, waterflow = {waterflow:0.2f}kg/s')

     #################################### MEMBRANE HEATING CALCS #################################################

  
    deltaT = []
    T_target = 5
    if selected_Tamb[1] < T_target:
        deltaT = T_target - selected_Tamb[1] 
    else:
        deltaT = 0

    percontainervolume = 80 #m3, assuming standard 40ft storage containers
    num_containers = math.ceil(Pin/10) + math.ceil(P/10) # assumes that a 40ft storage container can hold up to 10MW of capacity
    containerl = 13.3 # assumes 40ft storage container
    containerd = np.sqrt(percontainervolume/containerl)
    containerSA = (2 * containerd**2 + 4 * containerl * containerd ) # assumes a container of dimensions d x d x 40ft
    floorSA = containerl * containerd
    wallSA = (2 * containerd**2 + 4 * containerl * containerd ) - floorSA # assumes a container of dimensions d x d x 40ft
    k_plywood = 0.12  # W/m·K
    L_plywood = 0.028  # m
    R_walls_ceiling = L_steel / (k_steel*wallSA)
    R_floor = L_plywood/(k_plywood * floorSA)
    R_ins = h/(hk * wallSA) #assume insulation on walls, not floor.
    resistance_per_container = 1/(1/(R_walls_ceiling+R_ins) + (1/R_floor)) #thermal resistance in parallel equation
    continsulationcost = wallSA * num_containers * insulationcost * h

    Q_membrane_heat = deltaT / resistance_per_container * num_containers #W
    Q_membrane_heat = Q_membrane_heat / HEXeff / 1000
    Q_membrane_heat_pk = (T_target - selected_Tamb[0]) / resistance_per_container * num_containers
    Q_membrane_heat_pk = Q_membrane_heat_pk / HEXeff / 1000
    
    heatercapex = heatercost * Q_membrane_heat_pk * 1000
    heating_cost_membrane = Q_membrane_heat * idletime * cost  # heat during idle period, assuming Q_membrane_heat represents average year round heat
    
    print(f'Idle Membrane Heating Power = {Q_membrane_heat:0.0f}kW')

    ############################################## COST SUMMARY #####################################################################

    baselineCAPEX = P * 1000 * specificinvcost + P * 1000* DD * specificengcost
     
    stackwaterheat = (Q_total[1]- Q_total[-1]) / 1e3 #kW, pre-heating water before stacks
    tankwaterheat = P_heater[1] / 1e3 
    print(f' Stack Pre-heating = {stackwaterheat/1e3/Pin*100:0.3f}%')

    netpower = (ocompdelta + hcompdelta + stackwaterheat + tankwaterheat + Qin_ox)/1e3/Pin * 100 # membrane cooling delta has been shown to be negligible (see MasterH2calcs)
    netpower_reuse = (ocompdelta + hcompdelta + Q_membrane_heat)/1e3/Pin * 100 # when assuming that FC+ electrolyzer heat is used to pre-heat water

    tankinsulationcost = insulationcost * h * tankSA * numtanks
    drycoolercost = 96 #$ per kW of cooling load, source: file:///C:/Users/msnow/Downloads/0074-DAntoni.pdf
    coolingload = P/RTE*(1-RTE) #MW
    drycoolerdelta = (drycoolercost * coolingload * 1e3) * coldmarkup # assumes a 15% increase due to winterisation
    
    powerdensity = 4.4 #kW/m3, compressor specific volume 
    enclosurevolume = (power_results["oxygen"][-1] + power_results["hydrogen"][-1])*1000 / powerdensity #m3
    enclosureh = 7 #m, assume ceiling height
    enclosurel = np.sqrt(enclosurevolume/enclosureh)
    enclosureSA = enclosurel **2 + 4 * enclosureh * enclosurel

    T_target = -10  #C, min operating temp for compressors
    enclosureins = enclosureSA * insulationcost * h 
    enclosureR = h/(hk*enclosureSA)  #thermal resistance of enclosure
    Qenclosure = 0 # initialize
    if selected_Tamb[0] < T_target:
        Qenclosure_pk = (T_target - selected_Tamb[0])/enclosureR  # W
        Qenclosure = (T_target - selected_Tamb[0])/enclosureR  # W
    enclosureheaters = Qenclosure_pk * heatercost
    print(f'Enclosure Heater + Insulation Cost = ${np.round(enclosureheaters + enclosureins):,}')
    timebelowtarg = 0.25 #assumes that XX% of the year is below T_target
    enclosure_heating_cost = cost * Qenclosure/1e3 * 8760 * timebelowtarg * (1 - operating_frac)  #  assumes heating is only required during idle periods
    print(f' Powertrain Enclosure yearly heating cost at{selected_Tamb[0]}C = ${np.round(enclosure_heating_cost):,}')
    fluidcompcost = .025*(baselineCAPEX*0.6) #assumes BoP is 60% of total CAPEX, and fluid comps makeup 2.5% of BoP
    fluidcompcostdelta = fluidcompcost * coldmarkup   #assumes a 25% cost increase due to low-temp requirements 
    
    insulation_heater_CAPEX = heatercapex + continsulationcost + tankheatercost + tankinsulationcost + enclosureheaters + enclosureins
    sumCAPEXdeltas =  drycoolerdelta + fluidcompcostdelta + insulation_heater_CAPEX
    
    constructioncost = 0.1 * baselineCAPEX # 10% factor from --> "Projecting the levelized cost of large scale hydrogen storage for stationary applications 

    constructioncostdelta = 0.25 * constructioncost
    EPCcostdelta = .3*(sumCAPEXdeltas)  
    TotalCostDelta = sumCAPEXdeltas + EPCcostdelta + constructioncostdelta
    
    newCAPEX = baselineCAPEX + TotalCostDelta
    chargingcost = cost * P * 1000/RTE * DD * charges_per_year

    OMfactor = 1
    baselineOPEX = (30 * P * 1000 + 0.4 * P * DD) * OMfactor + chargingcost  #from storageninja

    baselinestorage = DD * P * charges_per_year   # MWh per year

    DoD = 1 # depth of discharge 
    nself = 0.00 #self discharge per cycle
    EoL = 0.95 #capacity at end of life
    cyclelife = 10e3 # cycle life
    Cyc_Deg = (1-EoL**(1/cyclelife)) #per-cycle degradation 
    T_Deg = 0  #temporal degradation
    Tc = 1 # construction time, years
    if charges_per_year * lifespan < cyclelife:   #if the project lifespan is shorter than the technology cycle life, extend lifespan to match 
        lifespan = min(np.round(cyclelife/charges_per_year),50)
    
    N = int(lifespan)
    num_replacements = math.floor(charges_per_year * lifespan/ cyclelife)
    installationmarkup = 1.2 #installation adds 10%
    specificreplcost = 300 # $/kW,  https://docs.nrel.gov/docs/fy24osti/87625.pdf
    replacement_cost = (P + P/RTE) * 1000 * specificreplcost * installationmarkup # per replacement
    # Calculate replacement intervals
    replacement_interval = cyclelife / charges_per_year  # Years until replacement
    nrepl = [int(i * replacement_interval) for i in range(1, int(N / replacement_interval) + 1) if int(i * replacement_interval) <= N]
    # Ensure replacements don't exceed lifespan
    nrepl = [year for year in nrepl if year < N]

    
    print(f'Replacement Costs Per Replacement Interval = ${np.round(replacement_cost):,}')
    print(f'Years of Replacement = {nrepl}')
    discreplacement_costs = sum(replacement_cost/ (1 + r)**(n + Tc) for n in nrepl)
    print(f'discounted replacement costs = ${np.round(discreplacement_costs):,}')

    discbaselinestorage = charges_per_year * P * DD * DoD * (1 - nself) * sum(((1 - Cyc_Deg)**((n-1) * charges_per_year) * (1 - T_Deg)**(n-1)) / ((1 + r)**(n + Tc)) for n in range(1, N+1))
    print(f'discounted baseline storage = {np.round(discbaselinestorage):,}MWh')
    
    storage_increase = (hcapdelta) / 100 * CF 
    newstorage = baselinestorage * (1  + storage_increase) # MWh per year at low temperature
    discnewstorage = newstorage * DoD * (1 - nself) * sum(((1 - Cyc_Deg)**((n-1) * charges_per_year) * (1 - T_Deg)**(n-1)) / ((1 + r)**(n + Tc)) for n in range(1, N+1))
    print(f'discounted storage at {selected_Tamb[1]}C = {np.round(discnewstorage):,}MWh')

    chargingcostdelta = baselinestorage/RTE * 1000 * storage_increase * cost #If using additional storage capacity, must also include the additional charging cost to fill that capacity
    print(f'charging cost delta = ${np.round(chargingcostdelta):,}')
    OPEXdelta = compression_cost + heating_cost_membrane + tank_heating_cost + enclosure_heating_cost + chargingcostdelta 

    
    print(f'OPEXdelta= ${np.round(OPEXdelta):,}')
    print(f'Compression cost = ${np.round(compression_cost):,}')
    print(f'heating_cost_membrane = ${np.round(heating_cost_membrane):,}')
    print(f'tank_heating_cost = ${np.round(tank_heating_cost):,}')
    print(f'enclosure_heating_cost = ${np.round(enclosure_heating_cost):,}')


    newOPEX = baselineOPEX + OPEXdelta
    disc_OPEX = sum(baselineOPEX / (1 + r)**(n+Tc) for n in range(N))
    
    baseLCOS = (baselineCAPEX + discreplacement_costs + disc_OPEX) / (discbaselinestorage * 1000)

    discnewOPEX = sum(newOPEX / (1 + r)**(n+Tc) for n in range(N))
    newLCOS = (newCAPEX + discreplacement_costs + discnewOPEX) / (discnewstorage * 1000)
    LCOSchange = ((newLCOS-baseLCOS)/baseLCOS) * 100
    print(f'Capacity Change at {selected_Tamb[1]}C = {hcapdelta * CF:0.3f}%')
    
    ################################### PIE CHART ###############################################

    constcontr = (constructioncostdelta)/(discbaselinestorage* 1000) # change to LCOS in $/MWh from construction costs
    heatcontr = (insulation_heater_CAPEX)/(discbaselinestorage* 1000) # change to LCOS in $/MWh from heating CAPEX
    opexcontr = abs(discnewOPEX-disc_OPEX)/(discbaselinestorage* 1000) # change to LCOS from OPEX increase (lifetime discounted)
    coolercontr = (drycoolerdelta) / (discbaselinestorage * 1000)  #change to LCOS from storage cap decrease
    fluidcontr = (fluidcompcostdelta) / (discbaselinestorage * 1000) #change to LCOS from winterisation of BoP fluid components
    EPCcontr = (EPCcostdelta) / (discbaselinestorage * 1000)  #change to LCOS from storage cap decrease
    storagecontr = abs((baselineCAPEX + disc_OPEX) / (discnewstorage * 1000)- baseLCOS) #change to LCOS from storage cap decrease

    sumchange =  constcontr + heatcontr + coolercontr + opexcontr + EPCcontr + storagecontr + fluidcontr

    percentconst = constcontr/sumchange * 100
    percentheat = heatcontr/ sumchange * 100
    percentopex = opexcontr/ sumchange * 100
    percentcooler = coolercontr/ sumchange * 100
    percentfluid = fluidcontr/ sumchange * 100
    percentEPC = EPCcontr/ sumchange * 100
    percentstor = storagecontr/ sumchange * 100

    print(f' Contribution to LCOS change, Construction Cost Markup = ${constcontr:0.3f}')
    print(f' Contribution to LCOS change, Heater & Insulation CAPEX = ${heatcontr:0.3f}')
    print(f' Contribution to LCOS change, Cooler Markup = ${coolercontr:0.3f}')
    print(f' Contribution to LCOS change, Fluid Component Markup = ${fluidcontr:0.3f}')
    print(f' Contribution to LCOS change, EPC Markup = ${EPCcontr:0.3f}')
    print(f' Contribution to LCOS change, OPEX (disc. over lifetime) = ${opexcontr:0.3f}')
    print(f' Contribution to LCOS change, Storage Reduction (disc. over lifetime) = ${storagecontr:0.3f}')    
    

    print(f' % Contribution to LCOS change, Construction Cost Markup = {percentconst:0.2f}%')
    print(f' % Contribution to LCOS change, Heater & Insulation CAPEX = {percentheat:0.2f}%')
    print(f' % Contribution to LCOS change, Cooler Winterisation Markup = {percentcooler:0.2f}%')
    print(f' % Contribution to LCOS change, Fluid Component Markup = {percentfluid:0.2f}%')
    print(f' % Contribution to LCOS change, EPC Markup = {percentEPC:0.2f}%')
    print(f' % Contribution to LCOS change, OPEX (disc. over lifetime) = {percentopex:0.2f}%')
    print(f' % Contribution to LCOS change, Storage Reduction (disc. over lifetime) = {percentstor:0.2f}%')



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


# In[37]:


"""
test_inputs = {
    "Power": 100, #MW
    "DD": 100,  #hrs
    "charges_per_year": 1,  
    "selected_Tamb": [-40, -10, 0, 20], #Arctic low, Arctic mean, warm place low, warm place mean temperature
    "Powercost": 0.05, #USD/kWh
    "interest_rate": 0.08,
    "project_lifespan": 50,
}

if __name__ == "__main__":
  results = run(test_inputs)
 
print(f'\n\nBaseline CAPEX = ${np.round(results['baselineCAPEX']):,}')
print(f'New CAPEX = ${np.round(results['newCAPEX']):,}')
OPEXdelta = results['newOPEX'] - results['baselineOPEX'] 
CAPEXdelta = results['newCAPEX'] - results['baselineCAPEX'] 
print(f'Baseline OPEX (including charging cost) = ${np.round(results['baselineOPEX']):,}')
print(f'Change in Yearly OPEX (including added charging cost) = ${np.round(OPEXdelta):,}')
#print(f'Change in CAPEX = ${np.round(CAPEXdelta):,}')

print(f'Total LCOS Change = {results['LCOSchange']:0.2f}')
print(f'Baseline LCOS = ${results['baseLCOS']:0.3f}/kWh')
print(f'New LCOS = ${results['newLCOS']:0.3f}/kWh') """


# In[ ]:




