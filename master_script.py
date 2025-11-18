import importlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from contextlib import redirect_stdout
import os
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm, Normalize
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.colors as mcolors

def run(common_inputs):

#def run(common_inputs: dict) -> dict:    # Define common inputs if not provided

    # List of subprogram module names
    subprograms = [
        "H2calcs",
        "PHScalcs",
        "BESScalcs",
        "CAEScalcs",
        "Flywheelcalcs"
    ]

    results = []
    # --- Initialize required output variables so they always exist ---
    baselineCAPEX = np.nan
    baselineOPEX = np.nan
    newCAPEX = np.nan
    newOPEX = np.nan
    baseLCOS = np.nan
    newLCOS = np.nan
    LCOSchange = np.nan
    baselinestorage = np.nan


    if common_inputs['DD'] * 2 * common_inputs['charges_per_year'] > 8760:
        raise ValueError(f"DISCHARGE DURATION AND CYCLE FREQUENCY EXCEED THE AVAILABLE HOURS IN A YEAR.")

    for module_name in subprograms:
        try:
            module = importlib.import_module(module_name)
            output = module.run(common_inputs)

            # Validate expected keys
            for key in ["LCOSchange","newLCOS","baselineCAPEX", "newCAPEX", "newOPEX","baseLCOS"]:
                if key not in output:
                    raise ValueError(f"{key} missing in output of {module_name}")

            results.append({ "program": module_name, **output })
            # If this module produced valid outputs, update the top-level variables
            baselineCAPEX = output.get("baselineCAPEX", baselineCAPEX)
            baselineOPEX = output.get("baselineOPEX", baselineOPEX)
            newCAPEX = output.get("newCAPEX", newCAPEX)
            newOPEX = output.get("newOPEX", newOPEX)
            baseLCOS = output.get("baseLCOS", baseLCOS)
            newLCOS = output.get("newLCOS", newLCOS)
            LCOSchange = output.get("LCOSchange", LCOSchange)
            baselinestorage = output.get("baselinestorage", baselinestorage)
        except Exception as e:
            print(f"[ERROR] {module_name}: {e}")
            continue

    # Postprocess: Print results in a formatted table
    # Define table parameters
    title = f"Key Results for P = {common_inputs['Power']:0.0f}MW, DD = {common_inputs['DD']:0.2f}hrs, Frequency = {common_inputs['charges_per_year']:0.0f}/year"
    col_width = 12  # Width for each program column
    desc_width = 35  # Width for metric description column

    # Print table title
    # print("\n" + "=" * (desc_width + len(results) * col_width + (len(results) - 1) * 3 + 4))
    # print(f"{title:^{desc_width + len(results) * col_width + (len(results) - 1) * 3}}")
    # print("=" * (desc_width + len(results) * col_width + (len(results) - 1) * 3 + 4))
    # print()

    # Print table header
    header = f"{'Metric':<{desc_width}} | {' | '.join([res['program'].replace('calcs', '').center(col_width) for res in results])}"
    # print(header)
    # print("-" * (desc_width + len(results) * col_width + (len(results) - 1) * 3))

    # Print table rows
    rows = [
        ("Temperate Climate CAPEX (M$)", [f"{res['baselineCAPEX']/1e6:>6.2f}" for res in results]),
        ("Cold Climate CAPEX (M$)", [f"{res['newCAPEX']/1e6:>6.2f}" for res in results]),
        ("Temperate Climate OPEX (M$)", [f"{res['baselineOPEX']/1e6:>6.2f}" for res in results]),
        ("Cold Climate OPEX (M$)", [f"{res['newOPEX']/1e6:>6.2f}" for res in results]),
        ("Temperate Climate LCOS ($/MWh)", [f"{res['baseLCOS']:>6.3f}" for res in results]),
        (f"LCOS at {common_inputs['selected_Tamb'][1]} C average ($/MWh)", [f"{res.get('newLCOS', 0):>6.3f}" for res in results]),
        (f"LCOS Change at {common_inputs['selected_Tamb'][1]} C average (%)", [f"{res['LCOSchange']:>6.2f}" for res in results])

    ]

    for desc, values in rows:
        # print(f"{desc:<{desc_width}} | {' | '.join([val.rjust(col_width) for val in values])}")
        pass

    # Print table footer
    # print("-" * (desc_width + len(results) * col_width + (len(results) - 1) * 3))
    # print()

    # List of subprogram module names
    subprograms = ["H2calcs", "PHScalcs", "BESScalcs", "CAEScalcs", "Flywheelcalcs"]

    # Define custom colors for subprograms
    subprogram_colors = {
        "H2calcs": "green",
        "PHScalcs": "blue",
        "BESScalcs": "red",
        "CAEScalcs": "pink",
        "Flywheelcalcs": "purple"
    }

    # Define ranges for DD and charges_per_year
    start_exp = np.log(0.0625) / np.log(2)  # 
    end_exp = np.log(1024) / np.log(2)    # 

    # Generate 7 values evenly spaced in log base 4 space
    log_space = np.linspace(start_exp, end_exp, 29)

    # Convert back to linear space
    DD_values = 2 ** log_space
    #DD_values = np.logspace(np.log10(0.0625), np.log10(1024), num=35, base=10)
    charges_values = np.logspace(0, np.log10(10000), num=29, base=10)

    # Initialize arrays to store results
    newLCOS_values = np.full((len(subprograms), len(DD_values), len(charges_values)), np.nan)
    baseLCOS_values = np.full((len(subprograms), len(DD_values), len(charges_values)), np.nan)
    LCOSchange_values = np.full((len(subprograms), len(DD_values), len(charges_values)), np.nan)

    min_newLCOS_indices = np.full((len(DD_values), len(charges_values)), -1, dtype=int)
    min_baseLCOS_indices = np.full((len(DD_values), len(charges_values)), -1, dtype=int)
    min_baseLCOS = np.full((len(DD_values), len(charges_values)), np.nan, dtype=float)
    min_newLCOS = np.full((len(DD_values), len(charges_values)), np.nan, dtype=float)
    min_LCOSchange = np.full((len(DD_values), len(charges_values)), np.nan, dtype=float)
    min_LCOSchange_a = np.full((len(DD_values), len(charges_values)), np.nan, dtype=float)
    # Counter for valid combinations
    valid_combinations = 0
    total_combinations = len(DD_values) * len(charges_values)

    # Iterate over DD and charges_per_year, suppressing print output
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull):
            for i, DD in enumerate(DD_values):
                for j, charges in enumerate(charges_values):
                    common_inputs["DD"] = DD
                    common_inputs["charges_per_year"] = charges
                    
                    # Check if combination is valid
                    if DD * 2 * charges > 8760:
                        continue  # Skip invalid combinations
                    valid_combinations += 1
                    
                    for k, module_name in enumerate(subprograms):
                        try:
                            module = importlib.import_module(module_name)
                            output = module.run(common_inputs)
                            
                            # Validate expected keys (relax baseLCOS requirement)
                            required_keys = ["baseLCOS", "newLCOS"]
                            missing_keys = [key for key in required_keys if key not in output]
                            
                            newLCOS_values[k, i, j] = output["newLCOS"]
                            baseLCOS_values[k, i, j] = output.get("baseLCOS", np.nan)
                            LCOSchange_values[k, i, j] = output.get("LCOSchange", np.nan)

                        
                        except Exception as e:
                            newLCOS_values[k, i, j] = np.nan
                            baseLCOS_values[k, i, j] = np.nan
                            LCOSchange_values[k, i, j] = np.nan
                    
                    # Find the index and value of the subprogram with the lowest newLCOS and baseLCOS
                    
                    valid_newLCOS = newLCOS_values[:, i, j]
                    valid_baseLCOS = baseLCOS_values[:, i, j]
                    valid_LCOSchange = LCOSchange_values[:, i, j]
                    if np.any(~np.isnan(valid_newLCOS)):
                        min_idx = np.nanargmin(valid_newLCOS)
                        min_newLCOS_indices[i, j] = min_idx
                        min_newLCOS[i, j] = valid_newLCOS[min_idx]
                        min_LCOSchange_a[i, j] = valid_LCOSchange[min_idx]  # Use LCOSchange for min newLCOS
                    if np.any(~np.isnan(valid_baseLCOS)):
                        min_baseLCOS_indices[i, j] = np.nanargmin(valid_baseLCOS)
                        min_baseLCOS[i, j] = np.nanmin(valid_baseLCOS)
                        min_LCOSchange[i, j] = valid_LCOSchange[min_baseLCOS_indices[i, j]]
    # Create custom colormap: white for invalid (-1), then specified colors for subprograms
    colors = ['white'] + [subprogram_colors[prog] for prog in subprograms]
    custom_cmap = ListedColormap(colors)

    # Create side-by-side subplots for subprogram indices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Create meshgrid
    X, Y = np.meshgrid(charges_values, DD_values)
    X_flat = X.ravel()
    Y_flat = Y.ravel()

    # Scatter plot for baseLCOS indices
    Z_flat_base = min_baseLCOS_indices.ravel()
    scatter1 = ax1.scatter(X_flat, Y_flat, c=Z_flat_base, cmap=custom_cmap, s=50, marker='o', edgecolors='white', linewidth=0.5)
    ax1.set_xscale('log', base=10)
    ax1.set_yscale('log', base=4)
    ax1.set_xlabel('Charges per Year (CPY)')
    ax1.set_title('Min LCOS Storage Technology in Mild Climates')
    ax1.set_ylabel('Discharge Duration (hours)')

    # Scatter plot for newLCOS indices
    Z_flat = min_newLCOS_indices.ravel()
    scatter2 = ax2.scatter(X_flat, Y_flat, c=Z_flat, cmap=custom_cmap, s=50, marker='o', edgecolors='white', linewidth=0.5)
    ax2.set_xscale('log', base=10)
    ax2.set_yscale('log', base=4)
    ax2.set_xlabel('Charges per Year (CPY)')
    ax2.set_title('Min LCOS Storage Technology in Arctic Climates')

    # Create colorbar with modified tick labels (remove '_calcs')
    cbar = fig.colorbar(scatter2, ax=[ax1, ax2], location='right')
    cbar.set_ticks(np.arange(len(subprograms)))
    cbar.set_ticklabels([prog.replace('calcs', '') for prog in subprograms])

    # Adjust subplot spacing
    fig.subplots_adjust(left=0.05, right=0.75, wspace=0.2)
    #plt.show()
    plt.close(fig)

    #########################################################################################################################
    ##################### AVERAGES THE LCOS CHANGE OVER EACH TECHNOLOGY ##########################
    avg_LCOSchange = np.zeros(len(subprograms))
    counts = np.zeros(len(subprograms), dtype=int)
    # Calculate average min_LCOSchange for each subprogram
    for k in range(len(subprograms)):
        # Find indices where subprogram k has the minimum newLCOS
        #mask = min_newLCOS_indices == k
        mask = min_baseLCOS_indices == k
        valid_LCOSchange = min_LCOSchange[mask]
        # Filter out NaN values
        valid_LCOSchange = valid_LCOSchange[~np.isnan(valid_LCOSchange)]
        counts[k] = len(valid_LCOSchange)
        avg_LCOSchange[k] = np.mean(valid_LCOSchange) if counts[k] > 0 else np.nan

    # Print diagnostics
    # print("Average min_LCOSchange per Subprogram:")
    for k, subprogram in enumerate(subprograms):
        # print(f"{subprogram.replace('calcs', '')}: "
        #       f"Average LCOSchange = {avg_LCOSchange[k]:.1f}%, "
        #       f"Valid points = {counts[k]}")
        pass
    # print("\nmin_LCOSchange Overall Range:")
    # print(f"Min: {np.nanmin(min_LCOSchange):.1f}%, Max: {np.nanmax(min_LCOSchange):.1f}%")

    ###############################################################################
    color_map = {
        'BESS': 'red',
        'H2': 'green',
        'CAES': 'pink',
        'PHS': 'blue',
        'Flywheel': 'purple'
    }

    # Assign colors based on subprogram name (after stripping 'calcs')
    colors = []
    for sp in subprograms:
        sp_clean = sp.replace('calcs', '')
        colors.append(color_map.get(sp_clean, 'gray'))  # default to gray if not found

    # Create bar chart
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(
        range(len(subprograms)),
        avg_LCOSchange,
        color=colors,
        edgecolor='black'
    )

    # X-axis labels
    ax.set_xticks(range(len(subprograms)))
    ax.set_xticklabels([sp.replace('calcs', '') for sp in subprograms], rotation=45, ha='right')

    # Labels and title
    ax.set_ylabel('Average LCOS Change (%)')
    ax.set_title(f'Average LCOS Change in the Arctic by Technology \nP = {common_inputs['Power']:0.0f}MW, Power @ {common_inputs['Powercost']:0.2f} USD/kWh')

    # Add value labels above bars
    for bar, value in zip(bars, avg_LCOSchange):
        if not np.isnan(value):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.1f}%",
                ha='center', va='bottom'
            )

    plt.tight_layout()
    #plt.show()
    plt.close(fig)


    # Define unique markers for each subprogram
    markers = ['o', 's', '^', 'D', '*']  # Circle, square, triangle, diamond, star

    # Define custom diamond path
    diamond_vertices = np.array([
        [0, 1], [1, 0], [0, -1], [-1, 0], [0, 1]  # Top, right, bottom, left, close
    ])
    diamond_codes = [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
    diamond_path = Path(diamond_vertices, diamond_codes)

    # Create side-by-side subplots for subprogram indices

    # Mask invalid data points
    Z_flat_base = np.ma.masked_array(min_baseLCOS_indices.ravel(), mask=np.isnan(min_baseLCOS_indices.ravel()))
    Z_flat_new = np.ma.masked_array(min_newLCOS_indices.ravel(), mask=np.isnan(min_newLCOS_indices.ravel()))

    # Create marker arrays
    marker_array_base = np.array([markers[int(idx)] if idx >= 0 else 'o' for idx in Z_flat_base.filled(-1)])
    marker_array_new = np.array([markers[int(idx)] if idx >= 0 else 'o' for idx in Z_flat_new.filled(-1)])

    # Define marker paths dictionary
    marker_paths = {
        'o': Path.unit_circle(),
        's': Path.unit_rectangle(),
        '^': Path.unit_regular_polygon(3),
        'D': diamond_path,
        '*': Path.unit_regular_polygon(6)
    }


    # Create legend with black markers
    legend_elements = [
        PathPatch(marker_paths['o'], facecolor='black', edgecolor='white', label='H2'),
        PathPatch(marker_paths['s'], facecolor='black', edgecolor='white', label='PHS'),
        PathPatch(marker_paths['^'], facecolor='black', edgecolor='white', label='BESS'),
        PathPatch(marker_paths['D'], facecolor='black', edgecolor='white', label='CAES'),
        PathPatch(marker_paths['*'], facecolor='black', edgecolor='white', label='Flywheel')
    ]
    ax2.legend(handles=legend_elements, title='Storage Technology', loc='upper right', fontsize=8)

    # Adjust subplot spacing
    fig.subplots_adjust(left=0.05, right=0.75, wspace=0.2)
    #plt.show()
    plt.close(fig)


    # Save marker map from newLCOS (right subplot)
    marker_map = marker_array_base  # Array of markers corresponding to min_newLCOS_indices

    # Successive plot for min_LCOSchange using saved marker map
    fig3 = plt.figure(figsize=(6, 4))
    ax3 = fig3.add_axes([0.1, 0.1, 0.7, 0.8])


    # Mask invalid data points
    Z_LCOSchange = np.ma.masked_invalid(min_LCOSchange)
    X_flat_masked = np.ma.masked_array(X_flat, mask=np.isnan(min_LCOSchange.ravel()))
    Y_flat_masked = np.ma.masked_array(Y_flat, mask=np.isnan(min_LCOSchange.ravel()))

    # Scatter plot with saved marker map
    scatter3 = ax3.scatter(X_flat_masked, Y_flat_masked, c=Z_LCOSchange.ravel(), cmap='plasma', 
                          s=50, marker='o', edgecolors='white', linewidth=0.5)
    paths_lcos = [marker_paths[marker] for marker in marker_map]
    scatter3.set_paths(paths_lcos)
    for idx, prog in enumerate(subprograms):
        mask = marker_map == markers[idx]
        # print(f"LCOSchange {prog.replace('calcs', '')}: {np.sum(mask)} points, Marker: {markers[idx]}")
        pass
    # print(f"LCOSchange Range: Min={np.nanmin(Z_LCOSchange):.1f}%, Max={np.nanmax(Z_LCOSchange):.1f}%")

    ax3.set_xscale('log', base=10)
    ax3.set_yscale('log', base=4)
    ax3.set_xlabel('Charges per Year (CPY)')
    ax3.set_ylabel('Discharge Duration (hours)')
    ax3.set_title(f'LCOS Change in Arctic Climates \nfor Baseline Technology (%)')

    # Use specified colorbar range
    vmin = np.min(np.ma.masked_invalid(min_LCOSchange))
    #vmax = np.max(np.ma.masked_invalid(min_LCOSchange))
    vmax = 20
    scatter3.set_norm(Normalize(vmin=vmin, vmax=vmax))

    # Add colorbar
    cbar3 = fig3.colorbar(scatter3, ax=ax3, location='right', pad=0.01)
    cbar3.set_label('Arctic LCOS Change (%)')
    cbar3.formatter = ticker.FuncFormatter(lambda x, pos: f'{x:.1f}%')
    #base_x = 2
    #cbar3.locator = ticker.LogLocator(base=base_x)  

    cbar3.update_ticks()
    #cbar3.ax.text(1.2, 0.00, f'{vmin:.1f}%+', transform=cbar3.ax.transAxes, ha='left', va='top', fontsize=10)
    cbar3.ax.text(1.2, 0.98, f'            +', transform=cbar3.ax.transAxes, 
                  ha='left', va='bottom', fontsize=10)

    # Add legend with black markers
    ax3.legend(handles=legend_elements, title='Storage Technology', loc='upper right', fontsize=8)

    #plt.show()
    plt.close(fig)

    fig3 = plt.figure(figsize=(6, 4))  # Adjust size as needed
    ax3 = fig3.add_axes([0.1, 0.1, 0.7, 0.8])  # [left, bottom, width, height] to 
    # Mask invalid data points
    Z_base = np.ma.masked_invalid(min_newLCOS)
    X_flat_masked = np.ma.masked_array(X_flat, mask=np.isnan(min_newLCOS.ravel()))
    Y_flat_masked = np.ma.masked_array(Y_flat, mask=np.isnan(min_newLCOS.ravel()))
    vmin = np.min(Z_base)
    vmax = np.max(Z_base)


    # Scatter plot for min baseLCOS values
    scatter3 = ax3.scatter(X_flat_masked, Y_flat_masked, c=Z_base.ravel(), norm=mcolors.LogNorm(vmin = vmin,vmax= vmax), cmap='plasma', s=50, marker='o', edgecolors='white', linewidth=0.5)
    ax3.set_xscale('log', base=10)
    ax3.set_yscale('log', base=4)
    ax3.set_xlabel('Charges per Year (CPY)')
    ax3.set_ylabel('Discharge Duration (hours)')
    ax3.set_title('Minimum LCOS in Arctic Climates')

    base_x = 4
    cbar3.locator = ticker.LogLocator(base=base_x)  
    cbar3.formatter = ticker.FuncFormatter(lambda val, pos: f'${val:.2f}/kWh')

    # Add separate colorbars with currency format and min/max labels
    cbar3 = fig3.colorbar(scatter3, ax=ax3, location='right', pad=0.01)
    cbar3.set_label('Arctic LCOS (USD/kWh)')
    cbar3.update_ticks()


    cbar3.ax.text(1.2, 0.00, f'${vmin:.2f}', transform=cbar3.ax.transAxes, 
                  ha='left', va='top', fontsize=10)
    cbar3.ax.text(1.2, 0.95, f'${vmax:.2f}', transform=cbar3.ax.transAxes, 
                  ha='left', va='bottom', fontsize=10)

    #plt.show()
    plt.close(fig)

    fig3 = plt.figure(figsize=(6, 4))  # Adjust size as needed
    ax3 = fig3.add_axes([0.1, 0.1, 0.7, 0.8])  # [left, bottom, width, height] to 
    # Mask invalid (NaN) data points
    Z_base = np.ma.masked_invalid(min_baseLCOS)
    X_flat_masked = np.ma.masked_array(X_flat, mask=np.isnan(min_baseLCOS.ravel()))
    Y_flat_masked = np.ma.masked_array(Y_flat, mask=np.isnan(min_baseLCOS.ravel()))
    vmin = np.min(Z_base)
    vmax = np.max(Z_base)


    # Scatter plot for min baseLCOS values
    scatter3 = ax3.scatter(X_flat_masked, Y_flat_masked, c=Z_base.ravel(), norm=mcolors.LogNorm(vmin = vmin,vmax= vmax), cmap='plasma', s=50, marker='o', edgecolors='white', linewidth=0.5)
    ax3.set_xscale('log', base=10)
    ax3.set_yscale('log', base=4)
    ax3.set_xlabel('Charges per Year (CPY)')
    ax3.set_ylabel('Discharge Duration (hours)')
    ax3.set_title('Minimum LCOS in Temperate Climates (USD/kWh)')

    base_x = 4
    cbar3.locator = ticker.LogLocator(base=base_x)  
    cbar3.formatter = ticker.FuncFormatter(lambda val, pos: f'${val:.2f}/kWh')

    # Add separate colorbars with currency format and min/max labels
    cbar3 = fig3.colorbar(scatter3, ax=ax3, location='right', pad=0.01)
    cbar3.set_label('Temperate LCOS (USD/kWh)')
    cbar3.update_ticks()


    cbar3.ax.text(1.2, 0.0, f'${vmin:.2f}\\n USD/kWh', transform=cbar3.ax.transAxes, 
                  ha='left', va='top', fontsize=10)
    cbar3.ax.text(1.2, 0.92, f'${vmax:.2f}\\n USD/kWh', transform=cbar3.ax.transAxes, 
                  ha='left', va='bottom', fontsize=10)

    #plt.show()
    plt.close(fig)

    # Create custom colormap: white for invalid (-1), then specified colors for subprograms
    colors = ['white'] + [subprogram_colors[prog] for prog in subprograms]
    custom_cmap = ListedColormap(colors)

    # Create side-by-side subplots for subprogram indices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Create meshgrid
    X, Y = np.meshgrid(charges_values, DD_values)
    X_flat = X.ravel()
    Y_flat = Y.ravel()

    # Scatter plot for baseLCOS indices
    Z_flat_base = min_baseLCOS_indices.ravel()
    scatter1 = ax1.scatter(X_flat, Y_flat, c=Z_flat_base, cmap=custom_cmap, s=50, marker='o', edgecolors='white', linewidth=0.5)
    ax1.set_xscale('log', base=10)
    ax1.set_yscale('log', base=4)
    ax1.set_xlabel('Charges per Year (CPY)', fontsize=14)  # Increased font size
    ax1.set_title('Best Storage Technology in Mild Climates', fontsize=14)  # Increased font size
    ax1.set_ylabel('Discharge Duration (hours)', fontsize=14)  # Increased font size

    # Scatter plot for newLCOS indices
    Z_flat = min_newLCOS_indices.ravel()
    scatter2 = ax2.scatter(X_flat, Y_flat, c=Z_flat, cmap=custom_cmap, s=50, marker='o', edgecolors='white', linewidth=0.5)
    ax2.set_xscale('log', base=10)
    ax2.set_yscale('log', base=4)
    ax2.set_xlabel('Charges per Year (CPY)', fontsize=14)  # Increased font size
    ax2.set_title(f'Best Storage Technology in Arctic Climates\n High Power Scenario', fontsize=14)  # Increased font size

    # Create colorbar with modified tick labels (remove '_calcs')
    cbar = fig.colorbar(scatter2, ax=[ax1, ax2], location='right')
    cbar.set_ticks(np.arange(len(subprograms)))
    cbar.set_ticklabels([prog.replace('calcs', '') for prog in subprograms])
    cbar.ax.tick_params(labelsize=12)  # Increased font size for colorbar ticks

    # Adjust subplot spacing
    fig.subplots_adjust(left=0.05, right=0.75, wspace=0.2)
    #plt.show()
    plt.close(fig)


    fig3 = plt.figure(figsize=(6, 4))  # Adjust size as needed
    ax3 = fig3.add_axes([0.1, 0.1, 0.7, 0.8])  # [left, bottom, width, height]

    # Mask invalid (NaN) data points
    Z_base = np.ma.masked_invalid(min_newLCOS)
    X_flat_masked = np.ma.masked_array(X_flat, mask=np.isnan(min_newLCOS.ravel()))
    Y_flat_masked = np.ma.masked_array(Y_flat, mask=np.isnan(min_newLCOS.ravel()))
    vmin = np.min(Z_base)
    vmax = np.max(Z_base)

    # Scatter plot for min baseLCOS values
    scatter3 = ax3.scatter(X_flat_masked, Y_flat_masked, c=Z_base.ravel(), norm=mcolors.LogNorm(vmin=vmin, vmax=vmax), cmap='plasma', s=50, marker='o', edgecolors='white', linewidth=0.5)
    ax3.set_xscale('log', base=10)
    ax3.set_yscale('log', base=4)
    ax3.set_xlabel('Charges per Year (CPY)', fontsize=14)  # Increased font size
    ax3.set_ylabel('Discharge Duration (hours)', fontsize=14)  # Increased font size
    ax3.set_title('Minimum LCOS in Arctic Climates', fontsize=14)  # Increased font size

    base_x = 4
    cbar3.locator = ticker.LogLocator(base=base_x)  
    cbar3.formatter = ticker.FuncFormatter(lambda val, pos: f'${val:.2f}/kWh')

    # Add separate colorbars with currency format and min/max labels
    cbar3 = fig3.colorbar(scatter3, ax=ax3, location='right', pad=0.01)
    cbar3.set_label('Arctic LCOS (USD/kWh)', fontsize=14)  # Increased font size
    cbar3.ax.tick_params(labelsize=12)  # Increased font size for colorbar ticks
    cbar3.update_ticks()

    cbar3.ax.text(1.2, 0.00, f'${vmin:.2f}', transform=cbar3.ax.transAxes, 
                  ha='left', va='top', fontsize=12)  # Increased font size
    cbar3.ax.text(1.2, 0.95, f'${vmax:.2f}', transform=cbar3.ax.transAxes, 
                  ha='left', va='bottom', fontsize=12)  # Increased font size

    #plt.show()
    plt.close(fig)

    fig3 = plt.figure(figsize=(6, 4))  # Adjust size as needed
    ax3 = fig3.add_axes([0.1, 0.1, 0.7, 0.8])  # [left, bottom, width, height]

    # Mask invalid (NaN) data points
    Z_base = np.ma.masked_invalid(min_newLCOS)
    X_flat_masked = np.ma.masked_array(X_flat, mask=np.isnan(min_newLCOS.ravel()))
    Y_flat_masked = np.ma.masked_array(Y_flat, mask=np.isnan(min_newLCOS.ravel()))
    vmin = np.min(Z_base)
    vmax = np.max(Z_base)

    # Scatter plot for min baseLCOS values
    scatter3 = ax3.scatter(X_flat_masked, Y_flat_masked, c=Z_base.ravel(), norm=mcolors.LogNorm(vmin=vmin, vmax=vmax), cmap='plasma', s=50, marker='o', edgecolors='white', linewidth=0.5)
    ax3.set_xscale('log', base=10)
    ax3.set_yscale('log', base=4)
    ax3.set_xlabel('Charges per Year (CPY)', fontsize=14)
    ax3.set_ylabel('Discharge Duration (hours)', fontsize=14)
    ax3.set_title('Minimum LCOS in Arctic Climates', fontsize=14)

    # Add colorbar with logarithmic scale and currency format
    cbar3 = fig3.colorbar(scatter3, ax=ax3, location='right', pad=0.01)
    cbar3.set_label('Arctic LCOS (USD/kWh)', fontsize=14)
    cbar3.locator = ticker.LogLocator(base=10)  # Logarithmic tick placement
    cbar3.formatter = ticker.FuncFormatter(lambda val, pos: f'${val:.0f}')  # Currency format
    cbar3.update_ticks()
    cbar3.ax.tick_params(labelsize=12)

    # Add min/max labels
    cbar3.ax.text(1.2, 0.00, f'${vmin:.2f}', transform=cbar3.ax.transAxes, 
                  ha='left', va='top', fontsize=12)
    cbar3.ax.text(1.2, 0.95, f'${vmax:.2f}', transform=cbar3.ax.transAxes, 
                  ha='left', va='bottom', fontsize=12)

    #plt.show()
    plt.close(fig)

    # Create custom colormap: white for invalid (-1), then specified colors for subprograms
    colors = ['white'] + [subprogram_colors[prog] for prog in subprograms]
    custom_cmap = ListedColormap(colors)

    # Create side-by-side subplots for subprogram indices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Create meshgrid
    X, Y = np.meshgrid(charges_values, DD_values)
    X_flat = X.ravel()
    Y_flat = Y.ravel()

    # Scatter plot for baseLCOS indices
    Z_flat_base = min_baseLCOS_indices.ravel()
    scatter1 = ax1.scatter(X_flat, Y_flat, c=Z_flat_base, cmap=custom_cmap, s=50, marker='o', edgecolors='white', linewidth=0.5)
    ax1.set_xscale('log', base=10)
    ax1.set_yscale('log', base=4)
    ax1.set_xlabel('Charges per Year (CPY)', fontsize=14)
    ax1.set_title('Best Storage Technology in Mild Climates', fontsize=14)
    ax1.set_ylabel('Discharge Duration (hours)', fontsize=14)

    # Scatter plot for newLCOS indices
    Z_flat = min_newLCOS_indices.ravel()
    scatter2 = ax2.scatter(X_flat, Y_flat, c=Z_flat, cmap=custom_cmap, s=50, marker='o', edgecolors='white', linewidth=0.5)
    ax2.set_xscale('log', base=10)
    ax2.set_yscale('log', base=4)
    ax2.set_xlabel('Charges per Year (CPY)', fontsize=14)
    ax2.set_title('Best Storage Technology in Arctic Climates\n Low Discount Rate Scenario', fontsize=14)
    ax2.set_ylabel('Discharge Duration (hours)', fontsize=14)
    ax2.tick_params(axis='y', which='both', labelleft=True)  # Enable y-axis ticks on ax2

    # Create colorbar with modified tick labels (remove '_calcs')
    cbar = fig.colorbar(scatter2, ax=[ax1, ax2], location='right')
    cbar.set_ticks(np.arange(len(subprograms)))
    cbar.set_ticklabels([prog.replace('calcs', '') for prog in subprograms])
    cbar.ax.tick_params(labelsize=12)

    # Adjust subplot spacing
    fig.subplots_adjust(left=0.05, right=0.75, wspace=0.2)
#    plt.show()
    plt.close(fig)

    fig3 = plt.figure(figsize=(6, 4))  # Adjust size as needed
    ax3 = fig3.add_axes([0.1, 0.1, 0.7, 0.8])  # [left, bottom, width, height]

    # Mask invalid (NaN) data points
    Z_base = np.ma.masked_invalid(min_newLCOS)
    X_flat_masked = np.ma.masked_array(X_flat, mask=np.isnan(min_newLCOS.ravel()))
    Y_flat_masked = np.ma.masked_array(Y_flat, mask=np.isnan(min_newLCOS.ravel()))
    vmin = np.min(Z_base)
    vmax = np.max(Z_base)

    # Scatter plot for min baseLCOS values
    scatter3 = ax3.scatter(X_flat_masked, Y_flat_masked, c=Z_base.ravel(), norm=mcolors.LogNorm(vmin=vmin, vmax=vmax), cmap='plasma', s=50, marker='o', edgecolors='white', linewidth=0.5)
    ax3.set_xscale('log', base=10)
    ax3.set_yscale('log', base=4)
    ax3.set_xlabel('Charges per Year (CPY)', fontsize=14)  # Increased font size
    ax3.set_ylabel('Discharge Duration (hours)', fontsize=14)  # Increased font size
    ax3.set_title('Minimum LCOS in Arctic Climates', fontsize=14)  # Increased font size

    base_x = 4
    cbar3.locator = ticker.LogLocator(base=base_x)  
    cbar3.formatter = ticker.FuncFormatter(lambda val, pos: f'${val:.2f}/kWh')

    # Add separate colorbars with currency format and min/max labels
    cbar3 = fig3.colorbar(scatter3, ax=ax3, location='right', pad=0.01)
    cbar3.set_label('Arctic LCOS (USD/kWh)', fontsize=14)  # Increased font size
    cbar3.ax.tick_params(labelsize=12)  # Increased font size for colorbar ticks
    cbar3.update_ticks()

    cbar3.ax.text(1.2, 0.00, f'${vmin:.2f}', transform=cbar3.ax.transAxes, 
                  ha='left', va='top', fontsize=12)  # Increased font size
    cbar3.ax.text(1.2, 0.95, f'${vmax:.2f}', transform=cbar3.ax.transAxes, 
                  ha='left', va='bottom', fontsize=12)  # Increased font size

    #plt.show()
    plt.close(fig)

    fig3 = plt.figure(figsize=(6, 4))  # Adjust size as needed
    ax3 = fig3.add_axes([0.1, 0.1, 0.7, 0.8])  # [left, bottom, width, height]

    # Mask invalid (NaN) data points
    Z_base = np.ma.masked_invalid(min_newLCOS)
    X_flat_masked = np.ma.masked_array(X_flat, mask=np.isnan(min_newLCOS.ravel()))
    Y_flat_masked = np.ma.masked_array(Y_flat, mask=np.isnan(min_newLCOS.ravel()))
    vmin = np.min(Z_base)
    vmax = np.max(Z_base)

    # Scatter plot for min baseLCOS values
    scatter3 = ax3.scatter(X_flat_masked, Y_flat_masked, c=Z_base.ravel(), norm=mcolors.LogNorm(vmin=vmin, vmax=vmax), cmap='plasma', s=50, marker='o', edgecolors='white', linewidth=0.5)
    ax3.set_xscale('log', base=10)
    ax3.set_yscale('log', base=4)
    ax3.set_xlabel('Charges per Year (CPY)', fontsize=14)
    ax3.set_ylabel('Discharge Duration (hours)', fontsize=14)
    ax3.set_title('Minimum LCOS in Arctic Climates', fontsize=14)

    # Add colorbar with logarithmic scale and currency format
    cbar3 = fig3.colorbar(scatter3, ax=ax3, location='right', pad=0.01)
    cbar3.set_label('Arctic LCOS (USD/kWh)', fontsize=14)
    cbar3.locator = ticker.LogLocator(base=10)  # Logarithmic tick placement
    cbar3.formatter = ticker.FuncFormatter(lambda val, pos: f'${val:.0f}')  # Currency format
    cbar3.update_ticks()
    cbar3.ax.tick_params(labelsize=12)

    # Add min/max labels
    cbar3.ax.text(1.2, 0.00, f'${vmin:.2f}', transform=cbar3.ax.transAxes, 
                  ha='left', va='top', fontsize=12)
    cbar3.ax.text(1.2, 0.95, f'${vmax:.2f}', transform=cbar3.ax.transAxes, 
                  ha='left', va='bottom', fontsize=12)

   # plt.show()
    plt.close(fig)

    # Create custom colormap: white for invalid (-1), then specified colors for subprograms
    colors = ['white'] + [subprogram_colors[prog] for prog in subprograms]
    custom_cmap = ListedColormap(colors)

    # Create side-by-side subplots for subprogram indices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Create meshgrid
    X, Y = np.meshgrid(charges_values, DD_values)
    X_flat = X.ravel()
    Y_flat = Y.ravel()

    # Scatter plot for baseLCOS indices
    Z_flat_base = min_baseLCOS_indices.ravel()
    scatter1 = ax1.scatter(X_flat, Y_flat, c=Z_flat_base, cmap=custom_cmap, s=50, marker='o', edgecolors='white', linewidth=0.5)
    ax1.set_xscale('log', base=10)
    ax1.set_yscale('log', base=4)
    ax1.set_xlabel('Charges per Year (CPY)', fontsize=14)
    ax1.set_title('Best Storage Technology in Mild Climates', fontsize=14)
    ax1.set_ylabel('Discharge Duration (hours)', fontsize=14)

    # Scatter plot for newLCOS indices
    Z_flat = min_newLCOS_indices.ravel()
    scatter2 = ax2.scatter(X_flat, Y_flat, c=Z_flat, cmap=custom_cmap, s=50, marker='o', edgecolors='white', linewidth=0.5)
    ax2.set_xscale('log', base=10)
    ax2.set_yscale('log', base=4)
    ax2.set_xlabel('Charges per Year (CPY)', fontsize=14)
    ax2.set_title('Best Storage Technology in Arctic Climates\n Low Discount Rate Scenario', fontsize=14)
    ax2.set_ylabel('Discharge Duration (hours)', fontsize=14)
    ax2.tick_params(axis='y', which='both', labelleft=True)  # Enable y-axis ticks on ax2

    # Create colorbar with modified tick labels (remove '_calcs')
    cbar = fig.colorbar(scatter2, ax=[ax1, ax2], location='right')
    cbar.set_ticks(np.arange(len(subprograms)))
    cbar.set_ticklabels([prog.replace('calcs', '') for prog in subprograms])
    cbar.ax.tick_params(labelsize=12)

    # Adjust subplot spacing
    fig.subplots_adjust(left=0.05, right=0.75, wspace=0.2)
  #  plt.show()
    plt.close(fig)

    fig3 = plt.figure(figsize=(6, 4))  # Adjust size as needed
    ax3 = fig3.add_axes([0.1, 0.1, 0.7, 0.8])  # [left, bottom, width, height]

    # Mask invalid (NaN) data points
    Z_base = np.ma.masked_invalid(min_newLCOS)
    X_flat_masked = np.ma.masked_array(X_flat, mask=np.isnan(min_newLCOS.ravel()))
    Y_flat_masked = np.ma.masked_array(Y_flat, mask=np.isnan(min_newLCOS.ravel()))
    vmin = np.min(Z_base)
    vmax = np.max(Z_base)

    # Scatter plot for min baseLCOS values
    scatter3 = ax3.scatter(X_flat_masked, Y_flat_masked, c=Z_base.ravel(), norm=mcolors.LogNorm(vmin=vmin, vmax=vmax), cmap='plasma', s=50, marker='o', edgecolors='white', linewidth=0.5)
    ax3.set_xscale('log', base=10)
    ax3.set_yscale('log', base=4)
    ax3.set_xlabel('Charges per Year (CPY)', fontsize=14)  # Increased font size
    ax3.set_ylabel('Discharge Duration (hours)', fontsize=14)  # Increased font size
    ax3.set_title('Minimum LCOS in Arctic Climates', fontsize=14)  # Increased font size

    base_x = 4
    cbar3.locator = ticker.LogLocator(base=base_x)  
    cbar3.formatter = ticker.FuncFormatter(lambda val, pos: f'${val:.2f}/kWh')

    # Add separate colorbars with currency format and min/max labels
    cbar3 = fig3.colorbar(scatter3, ax=ax3, location='right', pad=0.01)
    cbar3.set_label('Arctic LCOS (USD/kWh)', fontsize=14)  # Increased font size
    cbar3.ax.tick_params(labelsize=12)  # Increased font size for colorbar ticks
    cbar3.update_ticks()

    cbar3.ax.text(1.2, 0.00, f'${vmin:.2f}', transform=cbar3.ax.transAxes, 
                  ha='left', va='top', fontsize=12)  # Increased font size
    cbar3.ax.text(1.2, 0.95, f'${vmax:.2f}', transform=cbar3.ax.transAxes, 
                  ha='left', va='bottom', fontsize=12)  # Increased font size

   # plt.show()
    plt.close(fig)

    return {
        "baselineCAPEX": baselineCAPEX,
        "baselineOPEX": baselineOPEX,
        "newCAPEX": newCAPEX,
        "newOPEX": newOPEX,
        "baseLCOS": baseLCOS,
        "newLCOS": newLCOS,
        "LCOSchange": LCOSchange,
        "baselinestorage": baselinestorage,
    }
    
        # Note: All prints are commented out, only plots are shown via plt.show()
        # The function implicitly returns None, but displays the figures inline when run in an interactive environment



