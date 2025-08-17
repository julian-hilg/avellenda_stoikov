"""
Core implementation of Avellaneda-Stoikov market making model.
Reference: "High-frequency trading in a limit order book" (2008)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
import math


@dataclass
class Parameters:
    """
    Model parameters following the paper's convention.
    
    TIME & PRICE UNITS:
    - Time: All parameters use DAILY units
    - Price: sigma is in ABSOLUTE price units ($)
    
    Paper Parameters (Section 3.3):
    - s = 100 (initial price in $)
    - T = 1 (one trading day)
    - σ = 2 (price moves ±$2 per day)
    - dt = 0.005 (7.2 minutes)
    - γ = 0.1 (risk aversion)
    - k = 1.5 (order decay)
    - A = 140 (orders/day at mid)
    """
    gamma: float          # Risk aversion coefficient
    sigma: float         # Price volatility in $/day (e.g., 2.0 = $2 daily moves)
    k: float            # Order arrival decay parameter  
    A: float            # Base order arrival rate (orders per day)
    T: float            # Terminal time in DAYS (e.g., T=10 = 10 trading days)
    dt: float           # Time step in DAYS (e.g., 0.001 = ~1.4 minutes)
    initial_price: float = 100.0
    initial_inventory: int = 0
    initial_cash: float = 0.0
    max_inventory: int = 100  # Maximum allowed position size
    

class MarketMaker:
    """
    Avellaneda-Stoikov optimal market making strategy.
    
    Mathematical framework:
    1. Price dynamics: Arithmetic Brownian Motion (Eq. 2.1)
    2. Reservation pricing: Utility indifference pricing (Eq. 2.6-2.8)
    3. Optimal spread: Risk-return optimisation (Eq. 3.18)
    4. Order execution: Poisson arrival process (Eq. 2.11)
    """
    
    def __init__(self, params: Parameters):
        self.p = params
        self._validate_parameters()
        
    def _validate_parameters(self):
        """Validate parameter ranges with warnings for suspicious values"""
        import warnings
        
        assert self.p.gamma >= 0, "Risk aversion must be non-negative"
        assert self.p.sigma > 0, "Volatility must be positive"
        assert self.p.k > 0, "Arrival decay must be positive"
        assert self.p.A > 0, "Arrival rate must be positive"
        assert 0 < self.p.dt < self.p.T, "Invalid time discretisation"
        assert self.p.max_inventory > 0, "Max inventory must be positive"
        
        # Warn about suspicious parameter values
        # Note: sigma is in absolute price units, not percentage
        if self.p.sigma > 10:  # Price movement > $10 per day
            warnings.warn(
                f"sigma={self.p.sigma:.2f} implies ${self.p.sigma:.2f} daily price moves. "
                f"For a ${self.p.initial_price:.0f} stock, this is {self.p.sigma/self.p.initial_price*100:.1f}% daily volatility."
            )
        
        if self.p.dt > 0.01:  # Time step > 14 minutes
            warnings.warn(
                f"dt={self.p.dt:.3f} days = {self.p.dt*24*60:.1f} minutes "
                "seems large for HFT. Consider smaller time steps."
            )
        
        if self.p.T > 252:  # More than 1 year of trading days
            warnings.warn(
                f"T={self.p.T:.0f} days is more than 1 year of trading. "
                "Model assumes constant parameters which may be unrealistic."
            )
    
    def reservation_prices(self, s: float, q: int, t: float) -> Tuple[float, float]:
        """
        Calculate reservation bid and ask prices.
        
        Mathematical basis (Equations 2.6-2.7):
        r^a(s,q,t) = s + (1 - 2q) * γσ²(T-t) / 2
        r^b(s,q,t) = s + (-1 - 2q) * γσ²(T-t) / 2
        
        Algorithm:
        1. Calculate time to maturity
        2. Compute inventory adjustment factor
        3. Apply to mid-price
        
        Returns: (bid_reservation, ask_reservation)
        """
        tau = self.p.T - t
        factor = self.p.gamma * self.p.sigma**2 * tau / 2
        
        r_ask = s + (1 - 2*q) * factor
        r_bid = s + (-1 - 2*q) * factor
        
        return r_bid, r_ask
    
    def indifference_price(self, s: float, q: int, t: float) -> float:
        """
        Calculate indifference price.
        
        Mathematical basis (Equation 2.8):
        r(s,q,t) = s - q * γ * σ² * (T-t)
        
        This is the price at which the market maker is indifferent
        between holding and not holding the inventory.
        """
        tau = self.p.T - t
        return s - q * self.p.gamma * self.p.sigma**2 * tau
    
    def optimal_spread(self, t: float) -> float:
        """
        Calculate optimal bid-ask spread.
        
        Mathematical basis (Equation 3.18):
        δ^a + δ^b = γσ²(T-t) + (2/γ)ln(1 + γ/k)
        
        Components:
        1. Inventory risk: γσ²(T-t) - decreases over time
        2. Market making profit: (2/γ)ln(1 + γ/k) - constant
        """
        tau = self.p.T - t
        inventory_component = self.p.gamma * self.p.sigma**2 * tau
        profit_component = (2 / self.p.gamma) * np.log(1 + self.p.gamma / self.p.k)
        
        return inventory_component + profit_component
    
    def optimal_quotes(self, s: float, q: int, t: float) -> Tuple[float, float]:
        """
        Calculate optimal bid and ask quotes.
        
        Algorithm:
        1. Calculate indifference price r(s,q,t)
        2. Calculate optimal spread δ
        3. Position quotes: bid = r - δ/2, ask = r + δ/2
        
        Returns: (optimal_bid, optimal_ask)
        """
        r = self.indifference_price(s, q, t)
        spread = self.optimal_spread(t)
        half_spread = spread / 2
        
        return r - half_spread, r + half_spread
    
    def arrival_intensity(self, delta: float) -> float:
        """
        Order arrival intensity as function of distance from mid-price.
        
        Mathematical basis (Equation 2.11):
        λ(δ) = A * exp(-k * δ)
        
        Interpretation:
        - A: arrival rate at mid-price
        - k: decay rate with distance
        """
        return self.p.A * np.exp(-self.p.k * delta)
    
    def execution_probability(self, delta: float) -> float:
        """
        Probability of order execution in time interval dt.
        
        Mathematical basis:
        P(execution) = 1 - exp(-λ(δ) * dt)
        
        This is the CDF of exponential distribution evaluated at dt.
        """
        lambda_delta = self.arrival_intensity(delta)
        return 1 - np.exp(-lambda_delta * self.p.dt)


class PriceProcess:
    """
    Price dynamics generator using Arithmetic Brownian Motion.
    
    Mathematical basis (Equation 2.1):
    dS_t = σ * dW_t
    
    Note: Arithmetic (not geometric) to ensure bounded utility.
    """
    
    @staticmethod
    def generate(n_steps: int, s0: float, sigma: float, dt: float, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate price path.
        
        Algorithm:
        1. Generate Gaussian increments
        2. Scale by volatility and sqrt(dt)
        3. Cumulative sum for path
        """
        if seed is not None:
            np.random.seed(seed)
        
        dW = np.random.randn(n_steps) * np.sqrt(dt)
        increments = sigma * dW
        
        prices = np.zeros(n_steps + 1)
        prices[0] = s0
        prices[1:] = s0 + np.cumsum(increments)
        
        return prices


class Simulator:
    """
    Monte Carlo simulator for market making strategies.
    
    Workflow:
    1. Generate price paths
    2. Calculate optimal quotes at each step
    3. Simulate order executions
    4. Track inventory and P&L
    """
    
    def __init__(self, market_maker: MarketMaker):
        self.mm = market_maker
        self.p = market_maker.p
        
    def run_single(self, seed: Optional[int] = None) -> Dict:
        """
        Run single simulation path.
        
        Algorithm:
        1. Initialise state (inventory, cash)
        2. Generate price path
        3. For each time step:
           a. Calculate optimal quotes
           b. Simulate executions
           c. Update state
        4. Return results dictionary
        """
        n_steps = int(self.p.T / self.p.dt)
        
        # Generate price path
        prices = PriceProcess.generate(
            n_steps, self.p.initial_price, 
            self.p.sigma, self.p.dt, seed
        )
        
        # Initialise state
        q = self.p.initial_inventory
        x = self.p.initial_cash
        
        # Storage
        results = {
            'time': np.arange(n_steps + 1) * self.p.dt,
            'price': prices,
            'inventory': np.zeros(n_steps + 1),
            'cash': np.zeros(n_steps + 1),
            'pnl': np.zeros(n_steps + 1),
            'bid': np.zeros(n_steps),
            'ask': np.zeros(n_steps),
            'spread': np.zeros(n_steps),
            'n_buys': 0,
            'n_sells': 0
        }
        
        results['inventory'][0] = q
        results['cash'][0] = x
        results['pnl'][0] = x + q * prices[0]
        
        # Simulation loop
        for i in range(n_steps):
            t = i * self.p.dt
            s = prices[i]
            
            # Calculate optimal quotes
            bid, ask = self.mm.optimal_quotes(s, q, t)
            results['bid'][i] = bid
            results['ask'][i] = ask
            results['spread'][i] = ask - bid
            
            # Calculate execution probabilities
            prob_bid = self.mm.execution_probability(s - bid)
            prob_ask = self.mm.execution_probability(ask - s)
            
            # Simulate executions with inventory limits
            if np.random.random() < prob_bid and q < self.p.max_inventory:
                q += 1
                x -= bid
                results['n_buys'] += 1
            
            if np.random.random() < prob_ask and q > -self.p.max_inventory:
                q -= 1
                x += ask
                results['n_sells'] += 1
            
            # Update state
            results['inventory'][i + 1] = q
            results['cash'][i + 1] = x
            results['pnl'][i + 1] = x + q * prices[i + 1]
        
        return results
    
    def run_batch(self, n_simulations: int, parallel: bool = True) -> pd.DataFrame:
        """
        Run batch of simulations.
        
        Returns DataFrame with summary statistics per simulation.
        """
        summaries = []
        
        for seed in range(n_simulations):
            results = self.run_single(seed)
            
            summary = {
                'simulation': seed,
                'final_pnl': results['pnl'][-1],
                'final_inventory': results['inventory'][-1],
                'max_inventory': results['inventory'].max(),
                'min_inventory': results['inventory'].min(),
                'n_trades': results['n_buys'] + results['n_sells'],
                'n_buys': results['n_buys'],
                'n_sells': results['n_sells'],
                'avg_spread': results['spread'].mean(),
                'max_drawdown': self._calculate_drawdown(results['pnl'])
            }
            summaries.append(summary)
        
        return pd.DataFrame(summaries)
    
    @staticmethod
    def _calculate_drawdown(pnl: np.ndarray) -> float:
        """Calculate maximum drawdown from P&L series"""
        cummax = np.maximum.accumulate(pnl)
        drawdown = (pnl - cummax) / np.maximum(cummax, 1)
        return abs(drawdown.min())


class SymmetricStrategy(MarketMaker):
    """
    Symmetric benchmark strategy.
    Places quotes symmetrically around mid-price (not indifference price).
    """
    
    def optimal_quotes(self, s: float, q: int, t: float) -> Tuple[float, float]:
        """Override to centre quotes at mid-price"""
        spread = self.optimal_spread(t)
        half_spread = spread / 2
        return s - half_spread, s + half_spread