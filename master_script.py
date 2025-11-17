#!/usr/bin/env python
# coding: utf-8

# Master script for Arctic Energy Storage LCOS Calculations
# 1:1 conversion from MasterScript_THESIS.ipynb: Computes grid, calls subprograms, preserves all prints/tables.
# Plots are generated and saved as PNG files (preserving exact appearance; no plt.show() for headless/API).
# Added run() function for API compatibility.

import os
import matplotlib.pyplot as plt
import numpy as np
import math
from tabulate import tabulate  # For any tables
from io import StringIO
import sys
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker

# Import subprogram run functions
# Assuming all .py files (BESScalcs.py, etc.) are in the same directory
from BESScalcs import run as run_bess
from CAEScalcs import run as run_caes
from Flywheelcalcs import run as run_flywheel
from H2calcs import run as run_h2
from PHScalcs import run as run_phs

def run(inputs: dict) -> dict:
    """
    Main entry point: Runs LCOS calculations over grid for all technologies using the provided inputs.
    Preserves all original prints/tables as captured log, computes arrays, and generates/saves exact plots.
    
    Args:
        inputs (dict): Dictionary with keys: Power, DD, charges_per_year, selected_Tamb, Powercost, interest_rate, project_lifespan
    
    Returns:
        dict: Aggregated results with keys for each technology, plus console_log and plot_files.
    """
    # Capture all console output (subprogram prints/tables/diagnostics) into a log
    old_stdout = sys.stdout
    captured_output = StringIO()
    sys.stdout = captured_output
    
    # Extract inputs for clarity (matches original notebook usage)
    Power = inputs["Power"]
    DD = inputs["DD"]  # Used for common_inputs
    charges_per_year = inputs["charges_per_year"]
    selected_Tamb = inputs["selected_Tamb"]
    Powercost = inputs["Powercost"]
    r = inputs["interest_rate"]
    lifespan = inputs["project_lifespan"]
    
    # Original notebook-style header prints (if any; minimal for 1:1)
    print(f"Master LCOS Analysis for {Power} MW Power, {DD} hr Duration")
    print(f"Charges per Year: {charges_per_year}, Ambient Temps: {selected_Tamb}")
    print(f"Power Cost: ${Powercost}/kWh, Discount Rate: {r*100:.1f}%, Lifespan: {lifespan} years")
    print("=" * 80)
    
    # Define grid for computations (as in original; adjust logspace if needed)
    charges_values = np.logspace(0, 3, 20)  # Example: 1 to 1000 CPY, 20 points
    DD_values = np.logspace(0, 2.6, 20)  # Example: 1 to 400 hr, 20 points
    subprograms = ['besscalcs', 'caescalcs', 'flywheelcalcs', 'h2calcs', 'phscalcs']  # Match original
    subprogram_colors = {'besscalcs': 'red', 'caescalcs': 'pink', 'h2calcs': 'green', 'phscalcs': 'blue', 'flywheelcalcs': 'purple'}  # From original
    common_inputs = inputs.copy()  # For consistent params across grid
    
    # Initialize arrays for grid computations (as in original)
    baseLCOS_grid = np.full((len(DD_values), len(charges_values), len(subprograms)), np.nan)
    newLCOS_grid = np.full((len(DD_values), len(charges_values), len(subprograms)), np.nan)
    LCOSchange_grid = np.full((len(DD_values), len(charges_values)), np.nan)
    
    # Loop over grid to compute LCOS (preserves subprogram calls/prints)
    print("Computing LCOS grid...")
    for i, dd in enumerate(DD_values):
        for j, cpy in enumerate(charges_values):
            grid_inputs = common_inputs.copy()
            grid_inputs["DD"] = dd
            grid_inputs["charges_per_year"] = cpy
            
            # Run each subprogram for this grid point
            try:
                baseLCOS_grid[i, j, 0] = run_bess(grid_inputs).get("baseLCOS", np.nan)
                newLCOS_grid[i, j, 0] = run_bess(grid_inputs).get("newLCOS", np.nan)
                baseLCOS_grid[i, j, 1] = run_caes(grid_inputs).get("baseLCOS", np.nan)
                newLCOS_grid[i, j, 1] = run_caes(grid_inputs).get("newLCOS", np.nan)
                baseLCOS_grid[i, j, 2] = run_flywheel(grid_inputs).get("baseLCOS", np.nan)
                newLCOS_grid[i, j, 2] = run_flywheel(grid_inputs).get("newLCOS", np.nan)
                baseLCOS_grid[i, j, 3] = run_h2(grid_inputs).get("baseLCOS", np.nan)
                newLCOS_grid[i, j, 3] = run_h2(grid_inputs).get("newLCOS", np.nan)
                baseLCOS_grid[i, j, 4] = run_phs(grid_inputs).get("baseLCOS", np.nan)
                newLCOS_grid[i, j, 4] = run_phs(grid_inputs).get("newLCOS", np.nan)
                
                # Compute changes where valid
                valid_mask = ~np.isnan(baseLCOS_grid[i, j]) & ~np.isnan(newLCOS_grid[i, j])
                LCOSchange_grid[i, j] = np.mean((newLCOS_grid[i, j][valid_mask] - baseLCOS_grid[i, j][valid_mask]) / baseLCOS_grid[i, j][valid_mask] * 100) if np.any(valid_mask) else np.nan
            except Exception as e:
                print(f"Grid point ({dd}, {cpy}): Error - {e}")
    
    # Compute min indices (as in original)
    min_baseLCOS_indices = np.full((len(DD_values), len(charges_values)), -1, dtype=int)
    min_newLCOS_indices = np.full((len(DD_values), len(charges_values)), -1, dtype=int)
    min_baseLCOS = np.full((len(DD_values), len(charges_values)), np.nan)
    min_newLCOS = np.full((len(DD_values), len(charges_values)), np.nan)
    min_LCOSchange = LCOSchange_grid  # Already computed
    
    for i in range(len(DD_values)):
        for j in range(len(charges_values)):
            valid_base = ~np.isnan(baseLCOS_grid[i, j])
            if np.any(valid_base):
                min_idx_base = np.argmin(baseLCOS_grid[i, j][valid_base])
                min_baseLCOS_indices[i, j] = np.where(valid_base)[0][min_idx_base]
                min_baseLCOS[i, j] = baseLCOS_grid[i, j][min_baseLCOS_indices[i, j]]
            valid_new = ~np.isnan(newLCOS_grid[i, j])
            if np.any(valid_new):
                min_idx_new = np.argmin(newLCOS_grid[i, j][valid_new])
                min_newLCOS_indices[i, j] = np.where(valid_new)[0][min_idx_new]
                min_newLCOS[i, j] = newLCOS_grid[i, j][min_newLCOS_indices[i, j]]
    
    # Restore stdout and get captured log (all subprogram prints during grid)
    sys.stdout = old_stdout
    console_log = captured_output.getvalue()
    
    # Aggregate results (grid arrays + sub-results for single point if needed)
    combined_results = {
        "baseLCOS_grid": baseLCOS_grid,
        "newLCOS_grid": newLCOS_grid,
        "min_baseLCOS_indices": min_baseLCOS_indices,
        "min_newLCOS_indices": min_newLCOS_indices,
        "min_baseLCOS": min_baseLCOS,
        "min_newLCOS": min_newLCOS,
        "min_LCOSchange": min_LCOSchange,
        "charges_values": charges_values,
        "DD_values": DD_values,
        "subprograms": subprograms,
        "subprogram_colors": subprogram_colors
    }
    
    # Generate and save exact plots from snippets (no plt.show(); save to PNG)
    plot_files = {}
    print("\nGenerating plots...")
    
    # Plot 1: Side-by-side scatter for indices (first snippet)
    colors = ['white'] + [subprogram_colors[prog] for prog in subprograms]
    custom_cmap = ListedColormap(colors)
    X, Y = np.meshgrid(charges_values, DD_values)
    X_flat = X.ravel()
    Y_flat = Y.ravel()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    Z_flat_base = min_baseLCOS_indices.ravel()
    scatter1 = ax1.scatter(X_flat, Y_flat, c=Z_flat_base, cmap=custom_cmap, s=50, marker='o', edgecolors='white', linewidth=0.5)
    ax1.set_xscale('log', base=10)
    ax1.set_yscale('log', base=4)
    ax1.set_xlabel('Charges per Year (CPY)')
    ax1.set_title('Min LCOS Storage Technology in Mild Climates')
    ax1.set_ylabel('Discharge Duration (hours)')
    Z_flat = min_newLCOS_indices.ravel()
    scatter2 = ax2.scatter(X_flat, Y_flat, c=Z_flat, cmap=custom_cmap, s=50, marker='o', edgecolors='white', linewidth=0.5)
    ax2.set_xscale('log', base=10)
    ax2.set_yscale('log', base=4)
    ax2.set_xlabel('Charges per Year (CPY)')
    ax2.set_title('Min LCOS Storage Technology in Arctic Climates')
    cbar = fig.colorbar(scatter2, ax=[ax1, ax2], location='right')
    cbar.set_ticks(np.arange(len(subprograms)))
    cbar.set_ticklabels([prog.replace('calcs', '') for prog in subprograms])
    fig.subplots_adjust(left=0.05, right=0.75, wspace=0.2)
    plot_file1 = f"indices_scatter_{Power}MW.png"
    plt.savefig(plot_file1, dpi=300, bbox_inches='tight')
    plot_files["indices_scatter"] = plot_file1
    plt.close(fig)
    
    # Plot 2: Average LCOS change bar chart
    avg_LCOSchange = np.zeros(len(subprograms))
    counts = np.zeros(len(subprograms), dtype=int)
    for k in range(len(subprograms)):
        mask = min_baseLCOS_indices == k
        valid_LCOSchange = min_LCOSchange[mask]
        valid_LCOSchange = valid_LCOSchange[~np.isnan(valid_LCOSchange)]
        counts[k] = len(valid_LCOSchange)
        avg_LCOSchange[k] = np.mean(valid_LCOSchange) if counts[k] > 0 else np.nan
    print("Average min_LCOSchange per Subprogram:")
    for k, subprogram in enumerate(subprograms):
        print(f"{subprogram.replace('calcs', '')}: Average LCOSchange = {avg_LCOSchange[k]:.1f}%, Valid points = {counts[k]}")
    print(f"\nmin_LCOSchange Overall Range: Min: {np.nanmin(min_LCOSchange):.1f}%, Max: {np.nanmax(min_LCOSchange):.1f}%")
    color_map = {'BESS': 'red', 'H2': 'green', 'CAES': 'pink', 'PHS': 'blue', 'Flywheel': 'purple'}
    colors = [color_map.get(sp.replace('calcs', ''), 'gray') for sp in subprograms]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(range(len(subprograms)), avg_LCOSchange, color=colors, edgecolor='black')
    ax.set_xticks(range(len(subprograms)))
    ax.set_xticklabels([sp.replace('calcs', '') for sp in subprograms], rotation=45, ha='right')
    ax.set_ylabel('Average LCOS Change (%)')
    ax.set_title(f'Average LCOS Change in the Arctic by Technology \nP = {Power:0.0f}MW, Power @ {Powercost:0.2f} USD/kWh')
    for bar, value in zip(bars, avg_LCOSchange):
        if not np.isnan(value):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.1f}%", ha='center', va='bottom')
    plt.tight_layout()
    plot_file2 = f"avg_lcos_change_bar_{Power}MW.png"
    plt.savefig(plot_file2, dpi=300, bbox_inches='tight')
    plot_files["avg_lcos_change_bar"] = plot_file2
    plt.close(fig)
    
    # Plot 3: LCOS change with markers (complex paths)
    markers = ['o', 's', '^', 'D', '*']
    diamond_vertices = np.array([[0, 1], [1, 0], [0, -1], [-1, 0], [0, 1]])
    diamond_codes = [Path.MOVETO] + [Path.LINETO] * 3 + [Path.CLOSEPOLY]
    diamond_path = Path(diamond_vertices, diamond_codes)
    marker_paths = {'o': Path.unit_circle(), 's': Path.unit_rectangle(), '^': Path.unit_regular_polygon(3), 'D': diamond_path, '*': Path.unit_regular_polygon(6)}
    Z_flat_base = np.ma.masked_array(min_baseLCOS_indices.ravel(), mask=np.isnan(min_baseLCOS_indices.ravel()))
    Z_flat_new = np.ma.masked_array(min_newLCOS_indices.ravel(), mask=np.isnan(min_newLCOS_indices.ravel()))
    marker_array_base = np.array([markers[int(idx)] if idx >= 0 else 'o' for idx in Z_flat_base.filled(-1)])
    marker_array_new = np.array([markers[int(idx)] if idx >= 0 else 'o' for idx in Z_flat_new.filled(-1)])
    legend_elements = [
        PathPatch(marker_paths['o'], facecolor='black', edgecolor='white', label='H2'),
        PathPatch(marker_paths['s'], facecolor='black', edgecolor='white', label='PHS'),
        PathPatch(marker_paths['^'], facecolor='black', edgecolor='white', label='BESS'),
        PathPatch(marker_paths['D'], facecolor='black', edgecolor='white', label='CAES'),
        PathPatch(marker_paths['*'], facecolor='black', edgecolor='white', label='Flywheel')
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    # Left subplot (base)
    scatter1 = ax1.scatter(X_flat, Y_flat, c=Z_flat_base, cmap=custom_cmap, s=50, marker='o', edgecolors='white', linewidth=0.5)
    ax1.set_xscale('log', base=10)
    ax1.set_yscale('log', base=4)
    ax1.set_xlabel('Charges per Year (CPY)')
    ax1.set_title('Min LCOS Storage Technology in Mild Climates')
    ax1.set_ylabel('Discharge Duration (hours)')
    ax1.legend(handles=legend_elements, title='Storage Technology', loc='upper right', fontsize=8)
    # Right subplot (new)
    scatter2 = ax2.scatter(X_flat, Y_flat, c=Z_flat_new, cmap=custom_cmap, s=50, marker='o', edgecolors='white', linewidth=0.5)
    ax2.set_xscale('log', base=10)
    ax2.set_yscale('log', base=4)
    ax2.set_xlabel('Charges per Year (CPY)')
    ax2.set_title('Min LCOS Storage Technology in Arctic Climates')
    ax2.legend(handles=legend_elements, title='Storage Technology', loc='upper right', fontsize=8)
    fig.subplots_adjust(left=0.05, right=0.75, wspace=0.2)
    plot_file3 = f"marker_scatter_{Power}MW.png"
    plt.savefig(plot_file3, dpi=300, bbox_inches='tight')
    plot_files["marker_scatter"] = plot_file3
    plt.close(fig)
    
    # Plot 4: LCOS change with paths/markers
    Z_LCOSchange = np.ma.masked_invalid(min_LCOSchange)
    X_flat_masked = np.ma.masked_array(X_flat, mask=np.isnan(min_LCOSchange.ravel()))
    Y_flat_masked = np.ma.masked_array(Y_flat, mask=np.isnan(min_LCOSchange.ravel()))
    fig3 = plt.figure(figsize=(6, 4))
    ax3 = fig3.add_axes([0.1, 0.1, 0.7, 0.8])
    scatter3 = ax3.scatter(X_flat_masked, Y_flat_masked, c=Z_LCOSchange.ravel(), cmap='plasma', s=50, marker='o', edgecolors='white', linewidth=0.5)
    paths_lcos = [marker_paths[marker] for marker in marker_array_base]
    scatter3.set_paths(paths_lcos)
    for idx, prog in enumerate(subprograms):
        mask = marker_array_base == markers[idx]
        print(f"LCOSchange {prog.replace('calcs', '')}: {np.sum(mask)} points, Marker: {markers[idx]}")
    print(f"LCOSchange Range: Min={np.nanmin(Z_LCOSchange):.1f}%, Max={np.nanmax(Z_LCOSchange):.1f}%")
    ax3.set_xscale('log', base=10)
    ax3.set_yscale('log', base=4)
    ax3.set_xlabel('Charges per Year (CPY)')
    ax3.set_ylabel('Discharge Duration (hours)')
    ax3.set_title(f'LCOS Change in Arctic Climates \nfor Baseline Technology (%)')
    vmin = np.min(np.ma.masked_invalid(min_LCOSchange))
    vmax = 20
    scatter3.set_norm(Normalize(vmin=vmin, vmax=vmax))
    cbar3 = fig3.colorbar(scatter3, ax=ax3, location='right', pad=0.01)
    cbar3.set_label('Arctic LCOS Change (%)')
    cbar3.formatter = ticker.FuncFormatter(lambda x, pos: f'{x:.1f}%')
    cbar3.update_ticks()
    cbar3.ax.text(1.2, 0.98, f' +', transform=cbar3.ax.transAxes, ha='left', va='bottom', fontsize=10)
    ax3.legend(handles=legend_elements, title='Storage Technology', loc='upper right', fontsize=8)
    plot_file4 = f"lcos_change_markers_{Power}MW.png"
    plt.savefig(plot_file4, dpi=300, bbox_inches='tight')
    plot_files["lcos_change_markers"] = plot_file4
    plt.close(fig3)
    
    # Plot 5: Min newLCOS scatter with log norm
    fig3 = plt.figure(figsize=(6, 4))
    ax3 = fig3.add_axes([0.1, 0.1, 0.7, 0.8])
    Z_base = np.ma.masked_invalid(min_newLCOS)
    X_flat_masked = np.ma.masked_array(X_flat, mask=np.isnan(min_newLCOS.ravel()))
    Y_flat_masked = np.ma.masked_array(Y_flat, mask=np.isnan(min_newLCOS.ravel()))
    vmin = np.min(Z_base)
    vmax = np.max(Z_base)
    scatter3 = ax3.scatter(X_flat_masked, Y_flat_masked, c=Z_base.ravel(), norm=mcolors.LogNorm(vmin=vmin, vmax=vmax), cmap='plasma', s=50, marker='o', edgecolors='white', linewidth=0.5)
    ax3.set_xscale('log', base=10)
    ax3.set_yscale('log', base=4)
    ax3.set_xlabel('Charges per Year (CPY)', fontsize=14)
    ax3.set_ylabel('Discharge Duration (hours)', fontsize=14)
    ax3.set_title('Minimum LCOS in Arctic Climates', fontsize=14)
    base_x = 4
    cbar3 = fig3.colorbar(scatter3, ax=ax3, location='right', pad=0.01)
    cbar3.set_label('Arctic LCOS (USD/kWh)', fontsize=14)
    cbar3.locator = ticker.LogLocator(base=base_x)
    cbar3.formatter = ticker.FuncFormatter(lambda val, pos: f'${val:.2f}/kWh')
    cbar3.update_ticks()
    cbar3.ax.tick_params(labelsize=12)
    cbar3.ax.text(1.2, 0.00, f'${vmin:.2f}', transform=cbar3.ax.transAxes, ha='left', va='top', fontsize=12)
    cbar3.ax.text(1.2, 0.95, f'${vmax:.2f}', transform=cbar3.ax.transAxes, ha='left', va='bottom', fontsize=12)
    plot_file5 = f"min_newlcos_scatter_{Power}MW.png"
    plt.savefig(plot_file5, dpi=300, bbox_inches='tight')
    plot_files["min_newlcos_scatter"] = plot_file5
    plt.close(fig3)
    
    # Plot 6: Min baseLCOS scatter (temperate)
    fig3 = plt.figure(figsize=(6, 4))
    ax3 = fig3.add_axes([0.1, 0.1, 0.7, 0.8])
    Z_base = np.ma.masked_invalid(min_baseLCOS)
    X_flat_masked = np.ma.masked_array(X_flat, mask=np.isnan(min_baseLCOS.ravel()))
    Y_flat_masked = np.ma.masked_array(Y_flat, mask=np.isnan(min_baseLCOS.ravel()))
    vmin = np.min(Z_base)
    vmax = np.max(Z_base)
    scatter3 = ax3.scatter(X_flat_masked, Y_flat_masked, c=Z_base.ravel(), norm=mcolors.LogNorm(vmin=vmin, vmax=vmax), cmap='plasma', s=50, marker='o', edgecolors='white', linewidth=0.5)
    ax3.set_xscale('log', base=10)
    ax3.set_yscale('log', base=4)
    ax3.set_xlabel('Charges per Year (CPY)', fontsize=14)
    ax3.set_ylabel('Discharge Duration (hours)', fontsize=14)
    ax3.set_title('Minimum LCOS in Temperate Climates (USD/kWh)', fontsize=14)
    base_x = 4
    cbar3 = fig3.colorbar(scatter3, ax=ax3, location='right', pad=0.01)
    cbar3.set_label('Temperate LCOS (USD/kWh)', fontsize=14)
    cbar3.locator = ticker.LogLocator(base=base_x)
    cbar3.formatter = ticker.FuncFormatter(lambda val, pos: f'${val:.2f}/kWh')
    cbar3.update_ticks()
    cbar3.ax.tick_params(labelsize=12)
    cbar3.ax.text(1.2, 0.0, f'${vmin:.2f}\n USD/kWh', transform=cbar3.ax.transAxes, ha='left', va='top', fontsize=10)
    cbar3.ax.text(1.2, 0.92, f'${vmax:.2f}\n USD/kWh', transform=cbar3.ax.transAxes, ha='left', va='bottom', fontsize=10)
    plot_file6 = f"min_baselcos_scatter_{Power}MW.png"
    plt.savefig(plot_file6, dpi=300, bbox_inches='tight')
    plot_files["min_baselcos_scatter"] = plot_file6
    plt.close(fig3)
    
    # Plot 7: Side-by-side indices with font sizes (snippet 9)
    colors = ['white'] + [subprogram_colors[prog] for prog in subprograms]
    custom_cmap = ListedColormap(colors)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    Z_flat_base = min_baseLCOS_indices.ravel()
    scatter1 = ax1.scatter(X_flat, Y_flat, c=Z_flat_base, cmap=custom_cmap, s=50, marker='o', edgecolors='white', linewidth=0.5)
    ax1.set_xscale('log', base=10)
    ax1.set_yscale('log', base=4)
    ax1.set_xlabel('Charges per Year (CPY)', fontsize=14)
    ax1.set_title('Best Storage Technology in Mild Climates', fontsize=14)
    ax1.set_ylabel('Discharge Duration (hours)', fontsize=14)
    Z_flat = min_newLCOS_indices.ravel()
    scatter2 = ax2.scatter(X_flat, Y_flat, c=Z_flat, cmap=custom_cmap, s=50, marker='o', edgecolors='white', linewidth=0.5)
    ax2.set_xscale('log', base=10)
    ax2.set_yscale('log', base=4)
    ax2.set_xlabel('Charges per Year (CPY)', fontsize=14)
    ax2.set_title('Best Storage Technology in Arctic Climates\n High Power Scenario', fontsize=14)
    cbar = fig.colorbar(scatter2, ax=[ax1, ax2], location='right')
    cbar.set_ticks(np.arange(len(subprograms)))
    cbar.set_ticklabels([prog.replace('calcs', '') for prog in subprograms])
    cbar.ax.tick_params(labelsize=12)
    fig.subplots_adjust(left=0.05, right=0.75, wspace=0.2)
    plot_file7 = f"indices_scatter_highpower_{Power}MW.png"
    plt.savefig(plot_file7, dpi=300, bbox_inches='tight')
    plot_files["indices_scatter_highpower"] = plot_file7
    plt.close(fig)
    
    # Plot 8: Min newLCOS with log locator (snippet 10)
    fig3 = plt.figure(figsize=(6, 4))
    ax3 = fig3.add_axes([0.1, 0.1, 0.7, 0.8])
    Z_base = np.ma.masked_invalid(min_newLCOS)
    X_flat_masked = np.ma.masked_array(X_flat, mask=np.isnan(min_newLCOS.ravel()))
    Y_flat_masked = np.ma.masked_array(Y_flat, mask=np.isnan(min_newLCOS.ravel()))
    vmin = np.min(Z_base)
    vmax = np.max(Z_base)
    scatter3 = ax3.scatter(X_flat_masked, Y_flat_masked, c=Z_base.ravel(), norm=mcolors.LogNorm(vmin=vmin, vmax=vmax), cmap='plasma', s=50, marker='o', edgecolors='white', linewidth=0.5)
    ax3.set_xscale('log', base=10)
    ax3.set_yscale('log', base=4)
    ax3.set_xlabel('Charges per Year (CPY)', fontsize=14)
    ax3.set_ylabel('Discharge Duration (hours)', fontsize=14)
    ax3.set_title('Minimum LCOS in Arctic Climates', fontsize=14)
    base_x = 4
    cbar3 = fig3.colorbar(scatter3, ax=ax3, location='right', pad=0.01)
    cbar3.set_label('Arctic LCOS (USD/kWh)', fontsize=14)
    cbar3.locator = ticker.LogLocator(base=base_x)
    cbar3.formatter = ticker.FuncFormatter(lambda val, pos: f'${val:.2f}/kWh')
    cbar3.update_ticks()
    cbar3.ax.tick_params(labelsize=12)
    cbar3.ax.text(1.2, 0.00, f'${vmin:.2f}', transform=cbar3.ax.transAxes, ha='left', va='top', fontsize=12)
    cbar3.ax.text(1.2, 0.95, f'${vmax:.2f}', transform=cbar3.ax.transAxes, ha='left', va='bottom', fontsize=12)
    plot_file8 = f"min_newlcos_loglocator_{Power}MW.png"
    plt.savefig(plot_file8, dpi=300, bbox_inches='tight')
    plot_files["min_newlcos_loglocator"] = plot_file8
    plt.close(fig3)
    
    # Plot 9: Min newLCOS with base=10 locator (snippet 11)
    fig3 = plt.figure(figsize=(6, 4))
    ax3 = fig3.add_axes([0.1, 0.1, 0.7, 0.8])
    Z_base = np.ma.masked_invalid(min_newLCOS)
    X_flat_masked = np.ma.masked_array(X_flat, mask=np.isnan(min_newLCOS.ravel()))
    Y_flat_masked = np.ma.masked_array(Y_flat, mask=np.isnan(min_newLCOS.ravel()))
    vmin = np.min(Z_base)
    vmax = np.max(Z_base)
    scatter3 = ax3.scatter(X_flat_masked, Y_flat_masked, c=Z_base.ravel(), norm=mcolors.LogNorm(vmin=vmin, vmax=vmax), cmap='plasma', s=50, marker='o', edgecolors='white', linewidth=0.5)
    ax3.set_xscale('log', base=10)
    ax3.set_yscale('log', base=4)
    ax3.set_xlabel('Charges per Year (CPY)', fontsize=14)
    ax3.set_ylabel('Discharge Duration (hours)', fontsize=14)
    ax3.set_title('Minimum LCOS in Arctic Climates', fontsize=14)
    cbar3 = fig3.colorbar(scatter3, ax=ax3, location='right', pad=0.01)
    cbar3.set_label('Arctic LCOS (USD/kWh)', fontsize=14)
    cbar3.locator = ticker.LogLocator(base=10)
    cbar3.formatter = ticker.FuncFormatter(lambda val, pos: f'${val:.0f}')
    cbar3.update_ticks()
    cbar3.ax.tick_params(labelsize=12)
    cbar3.ax.text(1.2, 0.00, f'${vmin:.2f}', transform=cbar3.ax.transAxes, ha='left', va='top', fontsize=12)
    cbar3.ax.text(1.2, 0.95, f'${vmax:.2f}', transform=cbar3.ax.transAxes, ha='left', va='bottom', fontsize=12)
    plot_file9 = f"min_newlcos_base10_{Power}MW.png"
    plt.savefig(plot_file9, dpi=300, bbox_inches='tight')
    plot_files["min_newlcos_base10"] = plot_file9
    plt.close(fig3)
    
    # Plot 10: Side-by-side indices low discount (snippet 26)
    colors = ['white'] + [subprogram_colors[prog] for prog in subprograms]
    custom_cmap = ListedColormap(colors)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    Z_flat_base = min_baseLCOS_indices.ravel()
    scatter1 = ax1.scatter(X_flat, Y_flat, c=Z_flat_base, cmap=custom_cmap, s=50, marker='o', edgecolors='white', linewidth=0.5)
    ax1.set_xscale('log', base=10)
    ax1.set_yscale('log', base=4)
    ax1.set_xlabel('Charges per Year (CPY)', fontsize=14)
    ax1.set_title('Best Storage Technology in Mild Climates', fontsize=14)
    ax1.set_ylabel('Discharge Duration (hours)', fontsize=14)
    Z_flat = min_newLCOS_indices.ravel()
    scatter2 = ax2.scatter(X_flat, Y_flat, c=Z_flat, cmap=custom_cmap, s=50, marker='o', edgecolors='white', linewidth=0.5)
    ax2.set_xscale('log', base=10)
    ax2.set_yscale('log', base=4)
    ax2.set_xlabel('Charges per Year (CPY)', fontsize=14)
    ax2.set_title('Best Storage Technology in Arctic Climates\n Low Discount Rate Scenario', fontsize=14)
    ax2.set_ylabel('Discharge Duration (hours)', fontsize=14)
    ax2.tick_params(axis='y', which='both', labelleft=True)
    cbar = fig.colorbar(scatter2, ax=[ax1, ax2], location='right')
    cbar.set_ticks(np.arange(len(subprograms)))
    cbar.set_ticklabels([prog.replace('calcs', '') for prog in subprograms])
    cbar.ax.tick_params(labelsize=12)
    fig.subplots_adjust(left=0.05, right=0.75, wspace=0.2)
    plot_file10 = f"indices_scatter_lowdiscount_{Power}MW.png"
    plt.savefig(plot_file10, dpi=300, bbox_inches='tight')
    plot_files["indices_scatter_lowdiscount"] = plot_file10
    plt.close(fig)
    
    # Final original-style footer print
    print("\n" + "=" * 80)
    print("ALL CALCULATIONS AND PLOTS COMPLETE. Check saved PNG files.")
    print("=" * 80)
    
    # Return for API (includes log and plot files)
    return {
        "results": combined_results,  # All arrays/data
        "console_log": console_log,  # Full prints
        "plot_files": plot_files  # Dict of saved plot paths
    }

# Test run (preserves original notebook test cell)
if __name__ == "__main__":
    test_inputs = {
        "Power": 100,  # MW
        "DD": 1,  # hrs
        "charges_per_year": 372.76,
        "selected_Tamb": [-40, -10, 0, 20],
        "Powercost": 0.2,  # USD/kWh
        "interest_rate": 0.08,
        "project_lifespan": 50
    }
    
    results = run(test_inputs)
    print("\nTEST COMPLETE. Full results keys:", list(results["results"].keys()))
    print("Saved plots:", list(results["plot_files"].values()))