import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timedelta
import pandas as pd
from scipy import stats


@dataclass
class RiskMetrics:
    """Comprehensive risk metrics for market making"""
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    var_95: float  # Value at Risk at 95% confidence
    cvar_95: float  # Conditional VaR (Expected Shortfall)
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    recovery_factor: float
    calmar_ratio: float
    omega_ratio: float
    tail_ratio: float
    

class RiskAnalyzer:
    """
    Analyzes risk and performance metrics for market making strategies
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize risk analyzer
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
        """
        self.risk_free_rate = risk_free_rate
        self.pnl_history = []
        self.returns_history = []
        self.positions_history = []
        self.trades_history = []
        
    def add_pnl_point(self, pnl: float, timestamp: datetime):
        """Add P&L data point"""
        self.pnl_history.append({'timestamp': timestamp, 'pnl': pnl})
        
        if len(self.pnl_history) > 1:
            prev_pnl = self.pnl_history[-2]['pnl']
            if prev_pnl != 0:
                return_pct = (pnl - prev_pnl) / abs(prev_pnl)
                self.returns_history.append({'timestamp': timestamp, 'return': return_pct})
    
    def add_trade(self, price: float, quantity: float, is_buy: bool, timestamp: datetime):
        """Add trade for analysis"""
        self.trades_history.append({
            'timestamp': timestamp,
            'price': price,
            'quantity': quantity,
            'is_buy': is_buy
        })
    
    def add_position(self, position: int, timestamp: datetime):
        """Add position data point"""
        self.positions_history.append({'timestamp': timestamp, 'position': position})
    
    def calculate_sharpe_ratio(self, periods_per_year: float = 252) -> float:
        """
        Calculate Sharpe ratio
        
        Args:
            periods_per_year: Number of trading periods per year
        """
        if len(self.returns_history) < 2:
            return 0.0
        
        returns = [r['return'] for r in self.returns_history]
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualize
        excess_return = avg_return * periods_per_year - self.risk_free_rate
        annualized_std = std_return * np.sqrt(periods_per_year)
        
        return excess_return / annualized_std if annualized_std > 0 else 0.0
    
    def calculate_sortino_ratio(self, periods_per_year: float = 252) -> float:
        """
        Calculate Sortino ratio (Sharpe ratio using only downside volatility)
        """
        if len(self.returns_history) < 2:
            return 0.0
        
        returns = np.array([r['return'] for r in self.returns_history])
        avg_return = np.mean(returns)
        
        # Calculate downside deviation
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return float('inf')  # No downside risk
        
        downside_std = np.std(negative_returns)
        
        if downside_std == 0:
            return 0.0
        
        # Annualize
        excess_return = avg_return * periods_per_year - self.risk_free_rate
        annualized_downside_std = downside_std * np.sqrt(periods_per_year)
        
        return excess_return / annualized_downside_std if annualized_downside_std > 0 else 0.0
    
    def calculate_max_drawdown(self) -> Tuple[float, datetime, datetime]:
        """
        Calculate maximum drawdown
        
        Returns:
            (max_drawdown, peak_date, trough_date)
        """
        if len(self.pnl_history) < 2:
            return (0.0, datetime.now(), datetime.now())
        
        pnls = np.array([p['pnl'] for p in self.pnl_history])
        timestamps = [p['timestamp'] for p in self.pnl_history]
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(pnls)
        
        # Calculate drawdowns
        drawdowns = (pnls - running_max) / np.maximum(np.abs(running_max), 1)
        
        # Find maximum drawdown
        max_dd_idx = np.argmin(drawdowns)
        max_dd = drawdowns[max_dd_idx]
        
        # Find peak before the trough
        peak_idx = np.argmax(pnls[:max_dd_idx+1])
        
        return (abs(max_dd), timestamps[peak_idx], timestamps[max_dd_idx])
    
    def calculate_var(self, confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR)
        
        Args:
            confidence: Confidence level (e.g., 0.95 for 95% VaR)
        """
        if len(self.returns_history) < 10:
            return 0.0
        
        returns = np.array([r['return'] for r in self.returns_history])
        var_percentile = (1 - confidence) * 100
        
        return np.percentile(returns, var_percentile)
    
    def calculate_cvar(self, confidence: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR) / Expected Shortfall
        
        Args:
            confidence: Confidence level
        """
        if len(self.returns_history) < 10:
            return 0.0
        
        returns = np.array([r['return'] for r in self.returns_history])
        var = self.calculate_var(confidence)
        
        # Calculate mean of returns worse than VaR
        tail_returns = returns[returns <= var]
        
        return np.mean(tail_returns) if len(tail_returns) > 0 else var
    
    def calculate_win_rate(self) -> float:
        """Calculate percentage of profitable trades"""
        if len(self.trades_history) < 2:
            return 0.0
        
        # Group trades into round trips
        round_trips = self._identify_round_trips()
        
        if not round_trips:
            return 0.0
        
        wins = sum(1 for rt in round_trips if rt['pnl'] > 0)
        return wins / len(round_trips)
    
    def calculate_profit_factor(self) -> float:
        """Calculate ratio of gross profits to gross losses"""
        round_trips = self._identify_round_trips()
        
        if not round_trips:
            return 0.0
        
        gross_profits = sum(rt['pnl'] for rt in round_trips if rt['pnl'] > 0)
        gross_losses = abs(sum(rt['pnl'] for rt in round_trips if rt['pnl'] < 0))
        
        return gross_profits / gross_losses if gross_losses > 0 else float('inf')
    
    def _identify_round_trips(self) -> List[Dict]:
        """Identify completed round-trip trades"""
        round_trips = []
        position = 0
        entry_price = 0
        entry_quantity = 0
        
        for trade in self.trades_history:
            prev_position = position
            
            if trade['is_buy']:
                if position <= 0:
                    # Opening or reversing to long
                    entry_price = trade['price']
                    entry_quantity = trade['quantity']
                position += trade['quantity']
            else:
                if position >= 0:
                    # Opening or reversing to short
                    entry_price = trade['price']
                    entry_quantity = trade['quantity']
                position -= trade['quantity']
            
            # Check if we closed a position
            if (prev_position > 0 and position <= 0) or (prev_position < 0 and position >= 0):
                # Calculate P&L for the round trip
                if prev_position > 0:
                    pnl = (trade['price'] - entry_price) * min(trade['quantity'], abs(prev_position))
                else:
                    pnl = (entry_price - trade['price']) * min(trade['quantity'], abs(prev_position))
                
                round_trips.append({
                    'entry_price': entry_price,
                    'exit_price': trade['price'],
                    'quantity': min(trade['quantity'], abs(prev_position)),
                    'pnl': pnl,
                    'timestamp': trade['timestamp']
                })
        
        return round_trips
    
    def calculate_consecutive_wins_losses(self) -> Tuple[int, int]:
        """Calculate maximum consecutive wins and losses"""
        round_trips = self._identify_round_trips()
        
        if not round_trips:
            return (0, 0)
        
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0
        
        for rt in round_trips:
            if rt['pnl'] > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
        
        return (max_wins, max_losses)
    
    def calculate_recovery_factor(self) -> float:
        """Calculate recovery factor (net profit / max drawdown)"""
        if len(self.pnl_history) < 2:
            return 0.0
        
        final_pnl = self.pnl_history[-1]['pnl']
        max_dd, _, _ = self.calculate_max_drawdown()
        
        return abs(final_pnl / max_dd) if max_dd != 0 else float('inf')
    
    def calculate_calmar_ratio(self, periods_per_year: float = 252) -> float:
        """Calculate Calmar ratio (annualized return / max drawdown)"""
        if len(self.returns_history) < 2:
            return 0.0
        
        returns = [r['return'] for r in self.returns_history]
        annualized_return = np.mean(returns) * periods_per_year
        max_dd, _, _ = self.calculate_max_drawdown()
        
        return annualized_return / abs(max_dd) if max_dd != 0 else 0.0
    
    def calculate_omega_ratio(self, threshold: float = 0.0) -> float:
        """
        Calculate Omega ratio
        
        Args:
            threshold: Minimum acceptable return threshold
        """
        if len(self.returns_history) < 2:
            return 0.0
        
        returns = np.array([r['return'] for r in self.returns_history])
        
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        
        sum_gains = np.sum(gains) if len(gains) > 0 else 0
        sum_losses = np.sum(losses) if len(losses) > 0 else 0
        
        return sum_gains / sum_losses if sum_losses > 0 else float('inf')
    
    def calculate_tail_ratio(self) -> float:
        """Calculate ratio of 95th percentile to 5th percentile returns"""
        if len(self.returns_history) < 20:
            return 0.0
        
        returns = np.array([r['return'] for r in self.returns_history])
        
        right_tail = np.percentile(returns, 95)
        left_tail = abs(np.percentile(returns, 5))
        
        return right_tail / left_tail if left_tail > 0 else float('inf')
    
    def get_risk_metrics(self) -> RiskMetrics:
        """Calculate and return all risk metrics"""
        
        # Calculate basic metrics
        sharpe = self.calculate_sharpe_ratio()
        sortino = self.calculate_sortino_ratio()
        max_dd, _, _ = self.calculate_max_drawdown()
        var_95 = self.calculate_var(0.95)
        cvar_95 = self.calculate_cvar(0.95)
        
        # Calculate trade metrics
        win_rate = self.calculate_win_rate()
        profit_factor = self.calculate_profit_factor()
        
        # Calculate win/loss statistics
        round_trips = self._identify_round_trips()
        if round_trips:
            wins = [rt['pnl'] for rt in round_trips if rt['pnl'] > 0]
            losses = [rt['pnl'] for rt in round_trips if rt['pnl'] < 0]
            avg_win = np.mean(wins) if wins else 0.0
            avg_loss = np.mean(losses) if losses else 0.0
        else:
            avg_win = 0.0
            avg_loss = 0.0
        
        max_wins, max_losses = self.calculate_consecutive_wins_losses()
        
        # Calculate advanced ratios
        recovery_factor = self.calculate_recovery_factor()
        calmar_ratio = self.calculate_calmar_ratio()
        omega_ratio = self.calculate_omega_ratio()
        tail_ratio = self.calculate_tail_ratio()
        
        return RiskMetrics(
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            var_95=var_95,
            cvar_95=cvar_95,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            max_consecutive_wins=max_wins,
            max_consecutive_losses=max_losses,
            recovery_factor=recovery_factor,
            calmar_ratio=calmar_ratio,
            omega_ratio=omega_ratio,
            tail_ratio=tail_ratio
        )
    
    def generate_risk_report(self) -> str:
        """Generate formatted risk report"""
        metrics = self.get_risk_metrics()
        
        report = """
====================
   RISK REPORT
====================

Performance Metrics:
--------------------
Sharpe Ratio:        {:.3f}
Sortino Ratio:       {:.3f}
Calmar Ratio:        {:.3f}
Omega Ratio:         {:.3f}
Recovery Factor:     {:.3f}

Risk Metrics:
-------------
Max Drawdown:        {:.2%}
VaR (95%):          {:.2%}
CVaR (95%):         {:.2%}
Tail Ratio:          {:.3f}

Trading Statistics:
-------------------
Win Rate:            {:.2%}
Profit Factor:       {:.3f}
Average Win:         ${:.2f}
Average Loss:        ${:.2f}
Max Consecutive Wins:  {}
Max Consecutive Losses: {}

""".format(
            metrics.sharpe_ratio,
            metrics.sortino_ratio,
            metrics.calmar_ratio,
            metrics.omega_ratio,
            metrics.recovery_factor,
            metrics.max_drawdown,
            metrics.var_95,
            metrics.cvar_95,
            metrics.tail_ratio,
            metrics.win_rate,
            metrics.profit_factor,
            metrics.avg_win,
            metrics.avg_loss,
            metrics.max_consecutive_wins,
            metrics.max_consecutive_losses
        )
        
        return report