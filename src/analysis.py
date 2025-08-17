"""
Statistical analysis and parameter comparison framework.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from itertools import product
import warnings
warnings.filterwarnings('ignore')

from core import Parameters, MarketMaker, SymmetricStrategy, Simulator


class ParameterStudy:
    """
    Parameter analysis across multiple dimensions.
    """
    
    def __init__(self, base_params: Parameters, n_simulations: int = 1000):
        self.base_params = base_params
        self.n_simulations = n_simulations
        
    def run_parameter_grid(self, param_grid: Dict[str, List]) -> pd.DataFrame:
        """
        Run simulations across parameter grid.
        """
        # Generate parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        
        results = []
        
        for combo in combinations:
            # Create parameters
            params_dict = {name: value for name, value in zip(param_names, combo)}
            params = Parameters(
                gamma=params_dict.get('gamma', self.base_params.gamma),
                sigma=params_dict.get('sigma', self.base_params.sigma),
                k=params_dict.get('k', self.base_params.k),
                A=params_dict.get('A', self.base_params.A),
                T=params_dict.get('T', self.base_params.T),
                dt=params_dict.get('dt', self.base_params.dt)
            )
            
            # Run simulations for both strategies
            mm_inventory = MarketMaker(params)
            mm_symmetric = SymmetricStrategy(params)
            
            sim_inventory = Simulator(mm_inventory)
            sim_symmetric = Simulator(mm_symmetric)
            
            # Run batches
            results_inv = sim_inventory.run_batch(self.n_simulations)
            results_sym = sim_symmetric.run_batch(self.n_simulations)
            
            # Calculate statistics
            stats = {
                **params_dict,
                'strategy': 'inventory',
                'mean_pnl': results_inv['final_pnl'].mean(),
                'std_pnl': results_inv['final_pnl'].std(),
                'sharpe': results_inv['final_pnl'].mean() / results_inv['final_pnl'].std(),
                'mean_final_q': results_inv['final_inventory'].mean(),
                'std_final_q': results_inv['final_inventory'].std(),
                'mean_trades': results_inv['n_trades'].mean(),
                'max_drawdown': results_inv['max_drawdown'].mean(),
                'percentile_5': results_inv['final_pnl'].quantile(0.05),
                'percentile_95': results_inv['final_pnl'].quantile(0.95)
            }
            results.append(stats)
            
            stats_sym = {
                **params_dict,
                'strategy': 'symmetric',
                'mean_pnl': results_sym['final_pnl'].mean(),
                'std_pnl': results_sym['final_pnl'].std(),
                'sharpe': results_sym['final_pnl'].mean() / results_sym['final_pnl'].std(),
                'mean_final_q': results_sym['final_inventory'].mean(),
                'std_final_q': results_sym['final_inventory'].std(),
                'mean_trades': results_sym['n_trades'].mean(),
                'max_drawdown': results_sym['max_drawdown'].mean(),
                'percentile_5': results_sym['final_pnl'].quantile(0.05),
                'percentile_95': results_sym['final_pnl'].quantile(0.95)
            }
            results.append(stats_sym)
        
        return pd.DataFrame(results)
    
    def compare_strategies(self, gamma_values: List[float]) -> pd.DataFrame:
        """
        Compare inventory vs symmetric strategies across risk aversion levels.
        Reproduces Tables 1-3 from paper.
        """
        results = []
        
        for gamma in gamma_values:
            params = Parameters(
                gamma=gamma,
                sigma=self.base_params.sigma,
                k=self.base_params.k,
                A=self.base_params.A,
                T=self.base_params.T,
                dt=self.base_params.dt
            )
            
            # Calculate theoretical spread
            spread = gamma * params.sigma**2 * params.T + \
                    (2/gamma) * np.log(1 + gamma/params.k)
            
            # Run simulations
            for strategy_class, strategy_name in [(MarketMaker, 'Inventory'), 
                                                  (SymmetricStrategy, 'Symmetric')]:
                mm = strategy_class(params)
                sim = Simulator(mm)
                results_df = sim.run_batch(self.n_simulations)
                
                results.append({
                    'gamma': gamma,
                    'strategy': strategy_name,
                    'spread': spread,
                    'profit': results_df['final_pnl'].mean(),
                    'std_profit': results_df['final_pnl'].std(),
                    'final_q': results_df['final_inventory'].mean(),
                    'std_final_q': results_df['final_inventory'].std(),
                    'sharpe_ratio': results_df['final_pnl'].mean() / results_df['final_pnl'].std(),
                    'max_inventory': results_df['max_inventory'].mean(),
                    'min_inventory': results_df['min_inventory'].mean(),
                    'total_trades': results_df['n_trades'].mean()
                })
        
        return pd.DataFrame(results)


class StatisticalAnalysis:
    """
    Statistical analysis of simulation results.
    """
    
    @staticmethod
    def calculate_metrics(results: pd.DataFrame) -> Dict:
        """
        Calculate statistical metrics.
        """
        metrics = {
            # Distribution metrics
            'mean': results['final_pnl'].mean(),
            'std': results['final_pnl'].std(),
            'skew': results['final_pnl'].skew(),
            'kurtosis': results['final_pnl'].kurtosis(),
            
            # Percentiles
            'min': results['final_pnl'].min(),
            'q25': results['final_pnl'].quantile(0.25),
            'median': results['final_pnl'].median(),
            'q75': results['final_pnl'].quantile(0.75),
            'max': results['final_pnl'].max(),
            
            # Risk metrics
            'var_95': results['final_pnl'].quantile(0.05),
            'cvar_95': results[results['final_pnl'] <= results['final_pnl'].quantile(0.05)]['final_pnl'].mean(),
            'sharpe': results['final_pnl'].mean() / results['final_pnl'].std(),
            
            # Inventory metrics
            'mean_final_inventory': results['final_inventory'].mean(),
            'std_final_inventory': results['final_inventory'].std(),
            'inventory_range': results['max_inventory'].mean() - results['min_inventory'].mean(),
            
            # Trading metrics
            'mean_trades': results['n_trades'].mean(),
            'trade_imbalance': (results['n_buys'].mean() - results['n_sells'].mean()) / results['n_trades'].mean()
        }
        
        return metrics
    
    @staticmethod
    def hypothesis_test(results1: pd.DataFrame, results2: pd.DataFrame) -> Dict:
        """
        Statistical hypothesis testing between two strategies.
        """
        from scipy import stats
        
        pnl1 = results1['final_pnl'].values
        pnl2 = results2['final_pnl'].values
        
        # T-test for means
        t_stat, p_value_mean = stats.ttest_ind(pnl1, pnl2)
        
        # F-test for variances
        f_stat = np.var(pnl1, ddof=1) / np.var(pnl2, ddof=1)
        df1, df2 = len(pnl1) - 1, len(pnl2) - 1
        p_value_var = 2 * min(stats.f.cdf(f_stat, df1, df2), 1 - stats.f.cdf(f_stat, df1, df2))
        
        # Kolmogorov-Smirnov test
        ks_stat, p_value_dist = stats.ks_2samp(pnl1, pnl2)
        
        return {
            'mean_difference': pnl1.mean() - pnl2.mean(),
            't_statistic': t_stat,
            'p_value_mean': p_value_mean,
            'variance_ratio': f_stat,
            'p_value_variance': p_value_var,
            'ks_statistic': ks_stat,
            'p_value_distribution': p_value_dist,
            'significant_mean': p_value_mean < 0.05,
            'significant_variance': p_value_var < 0.05,
            'significant_distribution': p_value_dist < 0.05
        }


class ResultsFormatter:
    """
    Format results for display.
    """
    
    @staticmethod
    def create_comparison_table(df: pd.DataFrame) -> str:
        """
        Create formatted comparison table.
        """
        output = []
        output.append("="*80)
        output.append("STRATEGY COMPARISON RESULTS")
        output.append("="*80)
        
        for gamma in df['gamma'].unique():
            gamma_data = df[df['gamma'] == gamma]
            spread = gamma_data['spread'].iloc[0]
            
            output.append(f"\ngamma = {gamma:.2f}, Spread = {spread:.4f}")
            output.append("-"*60)
            output.append(f"{'Strategy':<12} {'Profit':>10} {'Std(P)':>10} {'Final q':>10} {'Std(q)':>10} {'Sharpe':>10}")
            output.append("-"*60)
            
            for _, row in gamma_data.iterrows():
                output.append(
                    f"{row['strategy']:<12} "
                    f"{row['profit']:>10.2f} "
                    f"{row['std_profit']:>10.2f} "
                    f"{row['final_q']:>10.2f} "
                    f"{row['std_final_q']:>10.2f} "
                    f"{row['sharpe_ratio']:>10.3f}"
                )
        
        output.append("="*80)
        return "\n".join(output)
    
    @staticmethod
    def create_parameter_matrix(df: pd.DataFrame, metric: str = 'sharpe') -> pd.DataFrame:
        """
        Create parameter sensitivity matrix.
        """
        params = [col for col in df.columns if col not in 
                 ['strategy', 'mean_pnl', 'std_pnl', 'sharpe', 'mean_final_q', 
                  'std_final_q', 'mean_trades', 'max_drawdown', 'percentile_5', 'percentile_95']]
        
        if len(params) >= 2:
            pivot = df[df['strategy'] == 'inventory'].pivot_table(
                values=metric,
                index=params[0],
                columns=params[1] if len(params) > 1 else 'strategy'
            )
            return pivot
        
        return df