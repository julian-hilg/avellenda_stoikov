"""
Visualisation module for Avellaneda-Stoikov model results.
All plots are saved to visualisations/ directory without displaying.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class Visualiser:
    """
    Create and save publication-quality visualisations.
    """
    
    def __init__(self, output_dir: str = "visualisations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_single_simulation(self, results: Dict, filename: str = "simulation.png"):
        """
        Plot single simulation results (reproduces Figure 1 from paper).
        
        Layout:
        - Panel 1: Prices and quotes
        - Panel 2: Inventory evolution
        - Panel 3: P&L trajectory
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        t = results['time']
        
        # Panel 1: Prices
        ax1 = axes[0]
        ax1.plot(t, results['price'], 'k-', label='Mid-price', linewidth=1.5)
        ax1.plot(t[:-1], results['bid'], 'g--', label='Bid', linewidth=0.8, alpha=0.7)
        ax1.plot(t[:-1], results['ask'], 'r--', label='Ask', linewidth=0.8, alpha=0.7)
        ax1.set_ylabel('Price', fontsize=11)
        ax1.set_title('Price Evolution and Optimal Quotes', fontsize=12)
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Inventory
        ax2 = axes[1]
        ax2.plot(t, results['inventory'], 'b-', linewidth=1.5)
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
        ax2.fill_between(t, 0, results['inventory'], alpha=0.3)
        ax2.set_ylabel('Inventory', fontsize=11)
        ax2.set_title('Inventory Position', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: P&L
        ax3 = axes[2]
        ax3.plot(t, results['pnl'], 'purple', linewidth=1.5)
        ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
        ax3.fill_between(t, 0, results['pnl'], 
                         where=(results['pnl'] >= 0), alpha=0.3, color='green')
        ax3.fill_between(t, 0, results['pnl'], 
                         where=(results['pnl'] < 0), alpha=0.3, color='red')
        ax3.set_xlabel('Time (trading days)', fontsize=11)
        ax3.set_ylabel('P&L', fontsize=11)
        ax3.set_title('Cumulative Profit & Loss', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_order_placement(self, results: Dict, params: Optional[Dict] = None, 
                           filename: str = "order_placement.png"):
        """
        Plot order placement pattern showing bid/ask quotes and executions.
        Similar to the paper's Figure 1.
        """
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        t = results['time']
        price = results['price']
        bid = results.get('bid', np.zeros_like(price[:-1]))
        ask = results.get('ask', np.zeros_like(price[:-1]))
        inventory = results['inventory']
        
        # Default parameters if not provided
        if params is None:
            params = {'gamma': 0.1, 'sigma': 0.02, 'k': 1.5, 'T': 10.0}
        
        # Find execution points
        bid_executions = []
        ask_executions = []
        
        for i in range(1, len(inventory)):
            if inventory[i] > inventory[i-1]:  # Buy executed
                bid_executions.append(i-1)
            elif inventory[i] < inventory[i-1]:  # Sell executed
                ask_executions.append(i-1)
        
        # Plot mid-market price
        ax.plot(t, price, 'k-', linewidth=1.5, label='Mid-market price')
        
        # Plot indifference price (reservation price)
        # Using the correct formula: r = s - q*γ*σ²*(T-t)
        gamma = params.get('gamma', 0.1)
        sigma = params.get('sigma', 2.0)
        T = params.get('T', 1.0)
        
        indiff_price = np.zeros_like(price)
        for i in range(len(price)):
            tau = max(0, T - t[i])  # Time to maturity
            indiff_price[i] = price[i] - inventory[i] * gamma * sigma**2 * tau
        
        ax.plot(t, indiff_price, 'g-', linewidth=1.2, alpha=0.7, label='Indifference Price')
        
        # Plot bid and ask quotes as lines
        if len(bid) > 0:
            ax.plot(t[:-1], bid, 'b--', linewidth=0.8, alpha=0.5, label='Bid quotes')
        if len(ask) > 0:
            ax.plot(t[:-1], ask, 'r--', linewidth=0.8, alpha=0.5, label='Ask quotes')
        
        # Highlight executed orders
        if bid_executions and len(bid) > 0:
            ax.scatter(t[bid_executions], bid[bid_executions], c='blue', s=50, 
                      marker='v', label='Buy executions', zorder=5)
        if ask_executions and len(ask) > 0:
            ax.scatter(t[ask_executions], ask[ask_executions], c='red', s=50, 
                      marker='^', label='Sell executions', zorder=5)
        
        ax.set_xlabel('Time (trading days)', fontsize=12)
        ax.set_ylabel('Stock Price', fontsize=12)
        ax.set_title('Order Placement and Execution Pattern', fontsize=14)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add text annotation for parameters with clear units
        param_text = (f'γ={gamma:.2f}, σ={sigma:.1%} daily, k={params.get("k", 1.5):.1f}\n'
                     f'Time units: DAYS')
        ax.text(0.02, 0.98, param_text, 
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_strategy_comparison(self, df_inv: pd.DataFrame, df_sym: pd.DataFrame, 
                                 gamma: float, filename: Optional[str] = None):
        """
        Compare inventory vs symmetric strategies (reproduces Figures 2-4).
        """
        if filename is None:
            filename = f"comparison_gamma_{gamma:.2f}.png"
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Strategy Comparison (gamma = {gamma})', fontsize=14, fontweight='bold')
        
        # P&L Distribution
        ax1 = axes[0, 0]
        ax1.hist(df_inv['final_pnl'], bins=50, alpha=0.6, label='Inventory', 
                density=True, color='blue', edgecolor='black')
        ax1.hist(df_sym['final_pnl'], bins=50, alpha=0.6, label='Symmetric', 
                density=True, color='red', edgecolor='black')
        ax1.axvline(df_inv['final_pnl'].mean(), color='blue', linestyle='--', linewidth=2)
        ax1.axvline(df_sym['final_pnl'].mean(), color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel('Final P&L')
        ax1.set_ylabel('Probability Density')
        ax1.set_title('P&L Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Final Inventory Distribution
        ax2 = axes[0, 1]
        ax2.hist(df_inv['final_inventory'], bins=30, alpha=0.6, label='Inventory', 
                density=True, color='blue', edgecolor='black')
        ax2.hist(df_sym['final_inventory'], bins=30, alpha=0.6, label='Symmetric', 
                density=True, color='red', edgecolor='black')
        ax2.set_xlabel('Final Inventory')
        ax2.set_ylabel('Probability Density')
        ax2.set_title('Final Inventory Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Risk-Return Scatter
        ax3 = axes[1, 0]
        ax3.scatter(df_inv['final_pnl'].std(), df_inv['final_pnl'].mean(), 
                   s=100, label='Inventory', color='blue', marker='o')
        ax3.scatter(df_sym['final_pnl'].std(), df_sym['final_pnl'].mean(), 
                   s=100, label='Symmetric', color='red', marker='s')
        ax3.set_xlabel('Risk (Std Dev of P&L)')
        ax3.set_ylabel('Return (Mean P&L)')
        ax3.set_title('Risk-Return Profile')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Drawdown Comparison
        ax4 = axes[1, 1]
        ax4.hist(df_inv['max_drawdown'], bins=30, alpha=0.6, label='Inventory', 
                density=True, color='blue', edgecolor='black')
        ax4.hist(df_sym['max_drawdown'], bins=30, alpha=0.6, label='Symmetric', 
                density=True, color='red', edgecolor='black')
        ax4.set_xlabel('Maximum Drawdown')
        ax4.set_ylabel('Probability Density')
        ax4.set_title('Drawdown Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_parameter_heatmap(self, df: pd.DataFrame, metric: str = 'sharpe',
                              filename: str = "parameter_heatmap.png"):
        """
        Create detailed heatmap of performance across parameter space.
        """
        # Get parameter columns
        param_cols = [col for col in df.columns if col in ['gamma', 'sigma', 'k', 'A']]
        
        if len(param_cols) >= 2:
            # Create pivot table
            pivot = df[df['strategy'] == 'inventory'].pivot_table(
                values=metric,
                index=param_cols[0],
                columns=param_cols[1]
            )
            
            plt.figure(figsize=(12, 9))
            
            # Use more detailed heatmap with better colormap
            sns.heatmap(pivot, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=pivot.mean().mean(),
                       cbar_kws={'label': metric.replace('_', ' ').title()},
                       linewidths=0.5, linecolor='gray')
            
            plt.title(f'{metric.replace("_", " ").title()} Across Parameters', fontsize=14)
            plt.xlabel(param_cols[1].title(), fontsize=12)
            plt.ylabel(param_cols[0].title(), fontsize=12)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()
    
    def plot_convergence_analysis(self, batch_sizes: List[int], 
                                  metrics: Dict[int, Dict],
                                  filename: str = "convergence.png"):
        """
        Plot convergence of statistics with sample size.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Statistical Convergence Analysis', fontsize=14, fontweight='bold')
        
        sizes = sorted(batch_sizes)
        
        # Mean convergence
        ax1 = axes[0, 0]
        means = [metrics[n]['mean'] for n in sizes]
        stds = [metrics[n]['std'] / np.sqrt(n) for n in sizes]  # Standard error
        ax1.errorbar(sizes, means, yerr=stds, marker='o', capsize=5)
        ax1.set_xlabel('Sample Size')
        ax1.set_ylabel('Mean P&L')
        ax1.set_title('Mean Convergence')
        ax1.set_xscale('log')
        ax1.grid(True, alpha=0.3)
        
        # Variance convergence
        ax2 = axes[0, 1]
        variances = [metrics[n]['std']**2 for n in sizes]
        ax2.plot(sizes, variances, marker='s')
        ax2.set_xlabel('Sample Size')
        ax2.set_ylabel('Variance')
        ax2.set_title('Variance Convergence')
        ax2.set_xscale('log')
        ax2.grid(True, alpha=0.3)
        
        # Sharpe ratio convergence
        ax3 = axes[1, 0]
        sharpes = [metrics[n]['sharpe'] for n in sizes]
        ax3.plot(sizes, sharpes, marker='^', color='green')
        ax3.set_xlabel('Sample Size')
        ax3.set_ylabel('Sharpe Ratio')
        ax3.set_title('Sharpe Ratio Convergence')
        ax3.set_xscale('log')
        ax3.grid(True, alpha=0.3)
        
        # Percentile convergence
        ax4 = axes[1, 1]
        q5 = [metrics[n]['var_95'] for n in sizes]
        q95 = [metrics[n].get('percentile_95', metrics[n]['max']) for n in sizes]
        ax4.fill_between(sizes, q5, q95, alpha=0.3, label='90% Range')
        ax4.plot(sizes, [metrics[n]['median'] for n in sizes], 
                'k-', label='Median', linewidth=2)
        ax4.set_xlabel('Sample Size')
        ax4.set_ylabel('P&L')
        ax4.set_title('Percentile Convergence')
        ax4.set_xscale('log')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_summary_matrix(self, comparison_df: pd.DataFrame, 
                           filename: str = "summary_matrix.png"):
        """
        Create comprehensive summary matrix of all results.
        """
        # Prepare data for visualisation
        metrics = ['profit', 'std_profit', 'sharpe_ratio', 'final_q', 'std_final_q']
        
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)))
        fig.suptitle('Comprehensive Strategy Comparison', fontsize=16, fontweight='bold')
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            # Create grouped bar plot
            pivot = comparison_df.pivot(index='gamma', columns='strategy', values=metric)
            pivot.plot(kind='bar', ax=ax, width=0.8)
            
            ax.set_xlabel('Risk Aversion (gamma)')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} by Strategy and Risk Aversion')
            ax.legend(title='Strategy')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Rotate x-labels
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_all_plots(self, results: Dict, comparison_df: pd.DataFrame):
        """
        Generate and save all standard plots.
        """
        print(f"Saving visualisations to {self.output_dir}/")
        
        # Plot for each gamma value
        for gamma in comparison_df['gamma'].unique():
            df_gamma = comparison_df[comparison_df['gamma'] == gamma]
            df_inv = df_gamma[df_gamma['strategy'] == 'Inventory']
            df_sym = df_gamma[df_gamma['strategy'] == 'Symmetric']
            
            # Note: This would need actual simulation data, not just summary stats
            # For now, we skip the detailed comparison plots
            
        # Summary matrix
        self.plot_summary_matrix(comparison_df, "summary_matrix.png")
        
        print(f"All visualisations saved to {self.output_dir}/")


class ReportGenerator:
    """
    Generate LaTeX-ready tables and reports.
    """
    
    @staticmethod
    def create_latex_table(df: pd.DataFrame, caption: str = "", label: str = "") -> str:
        """
        Convert DataFrame to LaTeX table format.
        """
        latex = df.to_latex(
            index=False,
            float_format="%.2f",
            caption=caption,
            label=label,
            column_format='l' + 'r' * (len(df.columns) - 1)
        )
        return latex
    
    @staticmethod
    def create_summary_report(results: Dict, filename: str = "report.tex"):
        """
        Create comprehensive LaTeX report.
        """
        report = []
        report.append("\\documentclass{article}")
        report.append("\\usepackage{booktabs}")
        report.append("\\usepackage{graphicx}")
        report.append("\\begin{document}")
        report.append("\\title{Avellaneda-Stoikov Model: Comprehensive Analysis}")
        report.append("\\maketitle")
        
        # Add content sections
        report.append("\\section{Executive Summary}")
        report.append("Analysis of market making strategies using the Avellaneda-Stoikov model.")
        
        report.append("\\section{Results}")
        # Add tables and figures references
        
        report.append("\\end{document}")
        
        with open(filename, 'w') as f:
            f.write('\n'.join(report))
        
        return filename