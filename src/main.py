"""
Main execution script for Avellaneda-Stoikov market making model.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import time

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core import Parameters, MarketMaker, SymmetricStrategy, Simulator
from analysis import ParameterStudy, StatisticalAnalysis, ResultsFormatter
from visualisation import Visualiser


def run_analysis():
    """
    Execute analysis pipeline.
    """
    print("="*60)
    print("AVELLANEDA-STOIKOV MARKET MAKING MODEL")
    print("="*60)
    
    # Define parameters with DAILY time units
    # Paper uses σ=2 in ABSOLUTE PRICE UNITS (not percentage)
    base_params = Parameters(
        gamma=0.1,           # Risk aversion coefficient
        sigma=2.0,           # $2 price movement per day (absolute)
        k=1.5,               # Order arrival decay parameter
        A=140,               # Order arrival rate (orders per day)
        T=1.0,               # 1 trading day (as in paper)
        dt=0.005,            # ~7.2 minutes per time step (as in paper)
        initial_price=100.0,
        max_inventory=50     # Maximum position size
    )
    
    # 1. SINGLE SIMULATION
    print("\n1. Running single simulation...")
    mm = MarketMaker(base_params)
    sim = Simulator(mm)
    single_result = sim.run_single(seed=42)
    
    # Save visualisations
    vis = Visualiser()
    vis.plot_single_simulation(single_result, "simulation.png")
    
    # Create order placement plot with reservation prices
    params_dict = {
        'gamma': base_params.gamma,
        'sigma': base_params.sigma,
        'k': base_params.k,
        'T': base_params.T
    }
    vis.plot_order_placement(single_result, params_dict, "order_placement.png")
    
    print("   Saved: visualisations/simulation.png")
    print("   Saved: visualisations/order_placement.png")
    
    # 2. STRATEGY COMPARISON
    print("\n2. Comparing strategies...")
    print("   Running 20 simulations per configuration...")
    
    study = ParameterStudy(base_params, n_simulations=20)
    gamma_values = [0.01, 0.1, 0.5]  # As in paper
    
    start_time = time.time()
    comparison_results = study.compare_strategies(gamma_values)
    elapsed = time.time() - start_time
    
    print(f"   Completed in {elapsed:.1f} seconds")
    
    # Format and display results
    formatted_table = ResultsFormatter.create_comparison_table(comparison_results)
    print("\n" + formatted_table)
    
    # Save results
    comparison_results.to_csv("visualisations/strategy_comparison.csv", index=False)
    print("   Saved: visualisations/strategy_comparison.csv")
    
    # 3. PARAMETER SENSITIVITY
    print("\n3. Running parameter sensitivity...")
    print("   Testing parameter grid...")
    
    # Grid for heatmap - absolute price units as in paper
    param_grid = {
        'gamma': [0.05, 0.1, 0.2, 0.5],
        'sigma': [1.0, 1.5, 2.0, 3.0]  # $1-3 price movement per day
    }
    
    # Use fewer simulations for grid search
    study_grid = ParameterStudy(base_params, n_simulations=10)
    grid_results = study_grid.run_parameter_grid(param_grid)
    grid_results.to_csv("visualisations/parameter_grid.csv", index=False)
    print("   Saved: visualisations/parameter_grid.csv")
    
    # Create parameter matrix
    param_matrix = ResultsFormatter.create_parameter_matrix(grid_results, metric='sharpe')
    print("\n   Sharpe Ratio Matrix:")
    print(param_matrix)
    
    # 4. VISUALISATIONS
    print("\n4. Generating visualisations...")
    vis.plot_parameter_heatmap(grid_results, metric='sharpe', filename="sharpe_heatmap.png")
    vis.plot_parameter_heatmap(grid_results, metric='mean_pnl', filename="pnl_heatmap.png")
    vis.plot_summary_matrix(comparison_results, "summary_matrix.png")
    
    print("   Saved: visualisations/sharpe_heatmap.png")
    print("   Saved: visualisations/pnl_heatmap.png")
    print("   Saved: visualisations/summary_matrix.png")
    
    # 5. SUMMARY
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    
    print("\nKey Results:")
    print("-" * 40)
    
    # Best configuration
    best_idx = grid_results[grid_results['strategy'] == 'inventory']['sharpe'].idxmax()
    best_config = grid_results.loc[best_idx]
    
    print(f"Best Configuration:")
    print(f"  gamma = {best_config.get('gamma', base_params.gamma):.2f}")
    print(f"  sigma = {best_config.get('sigma', base_params.sigma):.2f}")
    print(f"  Sharpe = {best_config['sharpe']:.3f}")
    
    print("\nStrategy Performance:")
    for gamma in gamma_values:
        gamma_data = comparison_results[comparison_results['gamma'] == gamma]
        inv_sharpe = gamma_data[gamma_data['strategy'] == 'Inventory']['sharpe_ratio'].values[0]
        sym_sharpe = gamma_data[gamma_data['strategy'] == 'Symmetric']['sharpe_ratio'].values[0]
        improvement = (inv_sharpe - sym_sharpe) / sym_sharpe * 100
        print(f"  gamma = {gamma:.2f}: Inventory Sharpe = {inv_sharpe:.3f}, Improvement = {improvement:+.1f}%")
    
    print("\nAll results saved to visualisations/")
    print("="*60)


if __name__ == "__main__":
    np.random.seed(42)
    run_analysis()