import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from .backtest_engine import BacktestResult


class PerformanceAnalyzer:
    """
    Analyzes and visualizes backtest performance
    """
    
    def __init__(self, result: BacktestResult):
        """
        Initialize performance analyzer
        
        Args:
            result: Backtest result to analyze
        """
        self.result = result
        self.trades_df = pd.DataFrame(result.trade_history) if result.trade_history else pd.DataFrame()
        self.quotes_df = pd.DataFrame(result.quote_history) if result.quote_history else pd.DataFrame()
        
    def generate_report(self) -> str:
        """Generate comprehensive performance report"""
        report = """
=====================================
    BACKTEST PERFORMANCE REPORT
=====================================

SUMMARY STATISTICS
------------------
Initial Capital:     ${:,.2f}
Final P&L:          ${:,.2f}
Total Return:        {:.2%}
Sharpe Ratio:        {:.3f}
Max Drawdown:        {:.2%}
Win Rate:            {:.2%}
Total Trades:        {:,}

RISK METRICS
------------
""".format(
            self.result.config.initial_cash,
            self.result.final_pnl,
            self.result.total_return,
            self.result.sharpe_ratio,
            self.result.max_drawdown,
            self.result.win_rate,
            self.result.total_trades
        )
        
        if self.result.risk_metrics:
            for metric, value in self.result.risk_metrics.items():
                if isinstance(value, float):
                    if 'ratio' in metric.lower():
                        report += f"{metric:20s} {value:.3f}\n"
                    elif 'var' in metric.lower() or 'cvar' in metric.lower():
                        report += f"{metric:20s} {value:.2%}\n"
                    else:
                        report += f"{metric:20s} {value:.2f}\n"
        
        # Add trade analysis
        if not self.trades_df.empty:
            report += self._analyze_trades()
        
        # Add time-based analysis
        report += self._analyze_time_patterns()
        
        return report
    
    def _analyze_trades(self) -> str:
        """Analyze trade statistics"""
        if self.trades_df.empty:
            return "\nNo trades executed\n"
        
        # Calculate trade statistics
        self.trades_df['hour'] = pd.to_datetime(self.trades_df['timestamp']).dt.hour
        
        buys = self.trades_df[self.trades_df['side'] == 'BUY']
        sells = self.trades_df[self.trades_df['side'] == 'SELL']
        
        report = """
TRADE ANALYSIS
--------------
Total Buys:          {:,}
Total Sells:         {:,}
Avg Buy Price:       ${:.2f}
Avg Sell Price:      ${:.2f}
Total Commission:    ${:.2f}
Avg Trade Size:      {:.2f}
""".format(
            len(buys),
            len(sells),
            buys['price'].mean() if not buys.empty else 0,
            sells['price'].mean() if not sells.empty else 0,
            self.trades_df['commission'].sum(),
            self.trades_df['size'].mean()
        )
        
        # Analyze by hour
        hourly_trades = self.trades_df.groupby('hour').size()
        if not hourly_trades.empty:
            report += f"\nMost Active Hour:    {hourly_trades.idxmax()}:00\n"
            report += f"Trades in Peak Hour: {hourly_trades.max()}\n"
        
        return report
    
    def _analyze_time_patterns(self) -> str:
        """Analyze performance patterns over time"""
        if self.result.pnl_series.empty:
            return "\nNo time series data available\n"
        
        # Daily returns
        daily_returns = self.result.returns_series.resample('D').sum()
        
        report = """
TIME ANALYSIS
-------------
"""
        
        if not daily_returns.empty:
            report += f"Best Day P&L:        ${daily_returns.max():.2f}\n"
            report += f"Worst Day P&L:       ${daily_returns.min():.2f}\n"
            report += f"Avg Daily Return:    {daily_returns.mean():.2%}\n"
            report += f"Daily Return Std:    {daily_returns.std():.2%}\n"
            
            # Calculate consecutive winning/losing days
            winning_days = (daily_returns > 0).astype(int)
            consecutive_wins = self._max_consecutive(winning_days.values, 1)
            consecutive_losses = self._max_consecutive(winning_days.values, 0)
            
            report += f"Max Consecutive Win Days:  {consecutive_wins}\n"
            report += f"Max Consecutive Loss Days: {consecutive_losses}\n"
        
        return report
    
    def _max_consecutive(self, arr: np.ndarray, value: int) -> int:
        """Find maximum consecutive occurrences of value"""
        max_count = 0
        current_count = 0
        
        for x in arr:
            if x == value:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        
        return max_count
    
    def plot_performance(self, figsize: Tuple[int, int] = (15, 12)) -> plt.Figure:
        """Create comprehensive performance plots"""
        fig = plt.figure(figsize=figsize)
        
        # Create subplots
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        # 1. P&L over time
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_pnl(ax1)
        
        # 2. Inventory over time
        ax2 = fig.add_subplot(gs[1, :])
        self._plot_inventory(ax2)
        
        # 3. Returns distribution
        ax3 = fig.add_subplot(gs[2, 0])
        self._plot_returns_distribution(ax3)
        
        # 4. Drawdown
        ax4 = fig.add_subplot(gs[2, 1])
        self._plot_drawdown(ax4)
        
        # 5. Trade scatter
        ax5 = fig.add_subplot(gs[2, 2])
        self._plot_trade_scatter(ax5)
        
        # 6. Hourly performance
        ax6 = fig.add_subplot(gs[3, 0])
        self._plot_hourly_performance(ax6)
        
        # 7. Quote spread analysis
        ax7 = fig.add_subplot(gs[3, 1])
        self._plot_spread_analysis(ax7)
        
        # 8. Cumulative trades
        ax8 = fig.add_subplot(gs[3, 2])
        self._plot_cumulative_trades(ax8)
        
        plt.suptitle('Backtest Performance Analysis', fontsize=16, y=0.995)
        
        return fig
    
    def _plot_pnl(self, ax: plt.Axes):
        """Plot P&L over time"""
        if not self.result.pnl_series.empty:
            ax.plot(self.result.pnl_series.index, self.result.pnl_series.values, 
                   label='P&L', linewidth=2)
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax.fill_between(self.result.pnl_series.index, 0, self.result.pnl_series.values,
                           where=self.result.pnl_series.values >= 0, alpha=0.3, color='green')
            ax.fill_between(self.result.pnl_series.index, 0, self.result.pnl_series.values,
                           where=self.result.pnl_series.values < 0, alpha=0.3, color='red')
            ax.set_title('Profit & Loss Over Time')
            ax.set_xlabel('Time')
            ax.set_ylabel('P&L ($)')
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    def _plot_inventory(self, ax: plt.Axes):
        """Plot inventory over time"""
        if not self.result.inventory_series.empty:
            ax.plot(self.result.inventory_series.index, self.result.inventory_series.values,
                   label='Inventory', linewidth=2, color='orange')
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Add inventory limits if available
            if self.result.config.inventory_limits:
                ax.axhline(y=self.result.config.inventory_limits.max_position, 
                         color='r', linestyle='--', alpha=0.5, label='Max Position')
                ax.axhline(y=-self.result.config.inventory_limits.max_position, 
                         color='r', linestyle='--', alpha=0.5)
            
            ax.set_title('Inventory Position Over Time')
            ax.set_xlabel('Time')
            ax.set_ylabel('Inventory')
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    def _plot_returns_distribution(self, ax: plt.Axes):
        """Plot returns distribution"""
        if not self.result.returns_series.empty:
            returns = self.result.returns_series.values * 100  # Convert to percentage
            ax.hist(returns, bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax.axvline(x=np.mean(returns), color='r', linestyle='--', 
                      label=f'Mean: {np.mean(returns):.2f}%')
            ax.set_title('Returns Distribution')
            ax.set_xlabel('Return (%)')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _plot_drawdown(self, ax: plt.Axes):
        """Plot drawdown over time"""
        if not self.result.pnl_series.empty:
            # Calculate drawdown
            cummax = self.result.pnl_series.cummax()
            drawdown = (self.result.pnl_series - cummax) / np.maximum(np.abs(cummax), 1)
            
            ax.fill_between(drawdown.index, 0, drawdown.values * 100,
                           color='red', alpha=0.5)
            ax.set_title('Drawdown Over Time')
            ax.set_xlabel('Time')
            ax.set_ylabel('Drawdown (%)')
            ax.grid(True, alpha=0.3)
    
    def _plot_trade_scatter(self, ax: plt.Axes):
        """Plot trade scatter (buys vs sells)"""
        if not self.trades_df.empty:
            buys = self.trades_df[self.trades_df['side'] == 'BUY']
            sells = self.trades_df[self.trades_df['side'] == 'SELL']
            
            if not buys.empty:
                ax.scatter(range(len(buys)), buys['price'], 
                         color='green', alpha=0.6, label='Buys', s=20)
            if not sells.empty:
                ax.scatter(range(len(sells)), sells['price'], 
                         color='red', alpha=0.6, label='Sells', s=20)
            
            ax.set_title('Trade Execution Prices')
            ax.set_xlabel('Trade Number')
            ax.set_ylabel('Price ($)')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _plot_hourly_performance(self, ax: plt.Axes):
        """Plot performance by hour of day"""
        if not self.trades_df.empty and 'hour' in self.trades_df.columns:
            hourly_counts = self.trades_df.groupby('hour').size()
            hours = list(range(24))
            counts = [hourly_counts.get(h, 0) for h in hours]
            
            ax.bar(hours, counts, color='skyblue', edgecolor='black')
            ax.set_title('Trades by Hour of Day')
            ax.set_xlabel('Hour')
            ax.set_ylabel('Number of Trades')
            ax.set_xticks(range(0, 24, 2))
            ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_spread_analysis(self, ax: plt.Axes):
        """Plot bid-ask spread analysis"""
        if not self.quotes_df.empty and 'spread' in self.quotes_df.columns:
            ax.hist(self.quotes_df['spread'], bins=50, alpha=0.7, 
                   color='purple', edgecolor='black')
            mean_spread = self.quotes_df['spread'].mean()
            ax.axvline(x=mean_spread, color='r', linestyle='--', 
                      label=f'Mean: ${mean_spread:.4f}')
            ax.set_title('Bid-Ask Spread Distribution')
            ax.set_xlabel('Spread ($)')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def _plot_cumulative_trades(self, ax: plt.Axes):
        """Plot cumulative trades over time"""
        if not self.trades_df.empty:
            trades_df_sorted = self.trades_df.sort_values('timestamp')
            trades_df_sorted['cumulative'] = range(1, len(trades_df_sorted) + 1)
            
            buys = trades_df_sorted[trades_df_sorted['side'] == 'BUY'].copy()
            sells = trades_df_sorted[trades_df_sorted['side'] == 'SELL'].copy()
            
            if not buys.empty:
                buys['cum_buys'] = range(1, len(buys) + 1)
                ax.plot(pd.to_datetime(buys['timestamp']), buys['cum_buys'], 
                       label='Cumulative Buys', color='green', linewidth=2)
            
            if not sells.empty:
                sells['cum_sells'] = range(1, len(sells) + 1)
                ax.plot(pd.to_datetime(sells['timestamp']), sells['cum_sells'], 
                       label='Cumulative Sells', color='red', linewidth=2)
            
            ax.set_title('Cumulative Trades Over Time')
            ax.set_xlabel('Time')
            ax.set_ylabel('Number of Trades')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    def export_results(self, filepath: str, format: str = 'csv'):
        """
        Export results to file
        
        Args:
            filepath: Path to save file
            format: Format ('csv', 'excel', 'json')
        """
        # Create comprehensive results dataframe
        results_data = {
            'metric': [],
            'value': []
        }
        
        # Add summary metrics
        results_data['metric'].extend([
            'final_pnl', 'total_return', 'sharpe_ratio', 
            'max_drawdown', 'win_rate', 'total_trades'
        ])
        results_data['value'].extend([
            self.result.final_pnl, self.result.total_return,
            self.result.sharpe_ratio, self.result.max_drawdown,
            self.result.win_rate, self.result.total_trades
        ])
        
        # Add risk metrics
        if self.result.risk_metrics:
            for key, value in self.result.risk_metrics.items():
                results_data['metric'].append(key)
                results_data['value'].append(value)
        
        results_df = pd.DataFrame(results_data)
        
        # Export based on format
        if format == 'csv':
            results_df.to_csv(filepath, index=False)
            # Also export time series
            self.result.pnl_series.to_csv(filepath.replace('.csv', '_pnl.csv'))
            self.result.inventory_series.to_csv(filepath.replace('.csv', '_inventory.csv'))
            if not self.trades_df.empty:
                self.trades_df.to_csv(filepath.replace('.csv', '_trades.csv'), index=False)
        elif format == 'excel':
            with pd.ExcelWriter(filepath) as writer:
                results_df.to_excel(writer, sheet_name='Summary', index=False)
                self.result.pnl_series.to_excel(writer, sheet_name='P&L')
                self.result.inventory_series.to_excel(writer, sheet_name='Inventory')
                if not self.trades_df.empty:
                    self.trades_df.to_excel(writer, sheet_name='Trades', index=False)
        elif format == 'json':
            results_df.to_json(filepath, orient='records')
        else:
            raise ValueError(f"Unsupported format: {format}")