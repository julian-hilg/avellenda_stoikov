import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
from datetime import datetime, timedelta
import math
from enum import Enum


class TimeHorizon(Enum):
    FINITE = "finite"
    INFINITE = "infinite"


@dataclass
class ASParameters:
    """Parameters for Avellaneda-Stoikov model"""
    gamma: float  # Risk aversion parameter (0=risk neutral, higher=more risk averse)
    sigma: float  # Volatility of the stock
    k: float = 1.5  # Order arrival intensity decay parameter
    A: float = 140  # Order arrival rate parameter
    dt: float = 0.005  # Time step
    T: Optional[float] = 1.0  # Terminal time for finite horizon
    q_max: int = 10  # Maximum inventory position
    horizon: TimeHorizon = TimeHorizon.FINITE
    
    def validate(self):
        """Validate parameters"""
        assert self.gamma >= 0, "Risk aversion must be non-negative"
        assert self.sigma > 0, "Volatility must be positive"
        assert self.k > 0, "Order arrival decay must be positive"
        assert self.A > 0, "Order arrival rate must be positive"
        assert self.dt > 0, "Time step must be positive"
        if self.horizon == TimeHorizon.FINITE:
            assert self.T is not None and self.T > 0, "Terminal time must be positive for finite horizon"
        assert self.q_max > 0, "Maximum inventory must be positive"


@dataclass
class Quote:
    """Bid and ask quote"""
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    reservation_price: float
    spread: float
    timestamp: datetime
    
    @property
    def mid_price(self) -> float:
        return (self.bid_price + self.ask_price) / 2


@dataclass
class MarketState:
    """Current market state"""
    mid_price: float
    inventory: int
    cash: float
    timestamp: datetime
    time_to_horizon: Optional[float] = None
    

class AvellanedaStoikovModel:
    """
    Implementation of the Avellaneda-Stoikov market making model
    
    Based on the paper: "High-frequency trading in a limit order book"
    by Marco Avellaneda and Sasha Stoikov (2008)
    """
    
    def __init__(self, params: ASParameters):
        """Initialize the model with parameters"""
        self.params = params
        self.params.validate()
        
        # State variables
        self.inventory = 0
        self.cash = 0.0
        self.pnl_history = []
        self.quote_history = []
        self.execution_history = []
        
        # Statistics
        self.total_buys = 0
        self.total_sells = 0
        self.max_inventory = 0
        self.min_inventory = 0
        
    def calculate_reservation_price(self, 
                                   mid_price: float,
                                   inventory: int,
                                   time_to_horizon: Optional[float] = None) -> float:
        """
        Calculate the reservation (indifference) price
        
        r(s,q,t) = s - q * γ * σ²* (T-t)
        
        This is the price at which the agent is indifferent between 
        holding and not holding the inventory
        """
        if self.params.horizon == TimeHorizon.FINITE:
            if time_to_horizon is None:
                time_to_horizon = self.params.T
            adjustment = inventory * self.params.gamma * (self.params.sigma ** 2) * time_to_horizon
        else:
            # Infinite horizon case
            w = 0.5 * (self.params.gamma ** 2) * (self.params.sigma ** 2) * ((self.params.q_max + 1) ** 2)
            adjustment = inventory * self.params.gamma * (self.params.sigma ** 2) / (2 * w)
            
        return mid_price - adjustment
    
    def calculate_optimal_spread(self, time_to_horizon: Optional[float] = None) -> float:
        """
        Calculate the optimal bid-ask spread
        
        δ^a + δ^b = γ*σ²*(T-t) + (2/γ)*ln(1 + γ/k)
        
        The spread consists of:
        1. Inventory risk component: γ*σ²*(T-t)
        2. Market making profit component: (2/γ)*ln(1 + γ/k)
        """
        if self.params.horizon == TimeHorizon.FINITE:
            if time_to_horizon is None:
                time_to_horizon = self.params.T
            inventory_risk = self.params.gamma * (self.params.sigma ** 2) * time_to_horizon
        else:
            inventory_risk = self.params.gamma * (self.params.sigma ** 2) / self.params.q_max
            
        market_making_profit = (2 / self.params.gamma) * math.log(1 + self.params.gamma / self.params.k)
        
        return inventory_risk + market_making_profit
    
    def calculate_order_arrival_probability(self, delta: float) -> float:
        """
        Calculate the probability of order arrival based on distance from mid price
        
        λ(δ) = A * exp(-k*δ)
        P(order arrives) = 1 - exp(-λ*dt)
        """
        intensity = self.params.A * math.exp(-self.params.k * delta)
        probability = 1 - math.exp(-intensity * self.params.dt)
        return min(1.0, max(0.0, probability))  # Ensure in [0,1]
    
    def get_optimal_quotes(self, state: MarketState) -> Quote:
        """
        Calculate optimal bid and ask quotes given current market state
        
        This is the main method that combines:
        1. Reservation price calculation
        2. Optimal spread calculation
        3. Quote positioning around reservation price
        """
        # Calculate time to horizon
        time_to_horizon = None
        if self.params.horizon == TimeHorizon.FINITE:
            time_to_horizon = state.time_to_horizon or self.params.T
            
        # Calculate reservation price
        reservation_price = self.calculate_reservation_price(
            state.mid_price,
            state.inventory,
            time_to_horizon
        )
        
        # Calculate optimal spread
        optimal_spread = self.calculate_optimal_spread(time_to_horizon)
        
        # Position quotes around reservation price
        half_spread = optimal_spread / 2
        
        bid_price = reservation_price - half_spread
        ask_price = reservation_price + half_spread
        
        # Adjust for inventory skew (optional enhancement)
        inventory_skew = self._calculate_inventory_skew(state.inventory)
        bid_price *= (1 - inventory_skew)
        ask_price *= (1 + inventory_skew)
        
        # Default sizes (can be adjusted based on risk limits)
        bid_size = self._calculate_order_size(state.inventory, is_buy=True)
        ask_size = self._calculate_order_size(state.inventory, is_buy=False)
        
        quote = Quote(
            bid_price=bid_price,
            ask_price=ask_price,
            bid_size=bid_size,
            ask_size=ask_size,
            reservation_price=reservation_price,
            spread=optimal_spread,
            timestamp=state.timestamp
        )
        
        self.quote_history.append(quote)
        return quote
    
    def _calculate_inventory_skew(self, inventory: int) -> float:
        """
        Calculate additional skew based on inventory position
        This helps to reduce inventory risk by making quotes more aggressive
        on the side that reduces inventory
        """
        if self.params.q_max == 0:
            return 0.0
            
        # Normalize inventory to [-1, 1]
        normalized_inv = inventory / self.params.q_max
        
        # Apply non-linear skew (tanh for smooth adjustment)
        skew = 0.001 * np.tanh(normalized_inv)  # Max 0.1% skew
        
        return skew
    
    def _calculate_order_size(self, inventory: int, is_buy: bool) -> float:
        """
        Calculate order size based on inventory and risk limits
        
        Reduces size as we approach inventory limits
        """
        base_size = 1.0
        
        if is_buy and inventory >= self.params.q_max:
            return 0.0  # Don't buy if at max long position
        elif not is_buy and inventory <= -self.params.q_max:
            return 0.0  # Don't sell if at max short position
            
        # Reduce size as we approach limits
        inventory_ratio = abs(inventory) / self.params.q_max
        size_multiplier = 1.0 - 0.5 * inventory_ratio  # Reduce up to 50% at limits
        
        return base_size * size_multiplier
    
    def simulate_execution(self, 
                          quote: Quote,
                          mid_price: float,
                          dt: Optional[float] = None) -> Tuple[bool, bool, float, float]:
        """
        Simulate order execution based on arrival probabilities
        
        Returns: (bid_executed, ask_executed, bid_price, ask_price)
        """
        if dt is None:
            dt = self.params.dt
            
        # Calculate distances from mid price
        delta_bid = mid_price - quote.bid_price
        delta_ask = quote.ask_price - mid_price
        
        # Calculate execution probabilities
        prob_bid = self.calculate_order_arrival_probability(delta_bid)
        prob_ask = self.calculate_order_arrival_probability(delta_ask)
        
        # Simulate executions
        bid_executed = np.random.random() < prob_bid
        ask_executed = np.random.random() < prob_ask
        
        return bid_executed, ask_executed, quote.bid_price, quote.ask_price
    
    def update_state(self,
                    bid_executed: bool,
                    ask_executed: bool,
                    bid_price: float,
                    ask_price: float,
                    bid_size: float = 1.0,
                    ask_size: float = 1.0) -> Dict[str, float]:
        """
        Update internal state after order executions
        
        Returns dictionary with updated metrics
        """
        prev_inventory = self.inventory
        prev_cash = self.cash
        
        if bid_executed:
            # We bought (our bid was hit)
            self.inventory += bid_size
            self.cash -= bid_price * bid_size
            self.total_buys += 1
            
        if ask_executed:
            # We sold (our ask was lifted)
            self.inventory -= ask_size
            self.cash += ask_price * ask_size
            self.total_sells += 1
            
        # Update inventory limits tracking
        self.max_inventory = max(self.max_inventory, self.inventory)
        self.min_inventory = min(self.min_inventory, self.inventory)
        
        # Calculate PnL change
        pnl_change = (self.cash - prev_cash) + (self.inventory - prev_inventory) * (bid_price + ask_price) / 2
        
        return {
            'inventory_change': self.inventory - prev_inventory,
            'cash_change': self.cash - prev_cash,
            'pnl_change': pnl_change,
            'current_inventory': self.inventory,
            'current_cash': self.cash
        }
    
    def calculate_pnl(self, current_price: float) -> float:
        """Calculate current profit and loss"""
        return self.cash + self.inventory * current_price
    
    def get_statistics(self) -> Dict[str, float]:
        """Get current statistics"""
        return {
            'total_buys': self.total_buys,
            'total_sells': self.total_sells,
            'max_inventory': self.max_inventory,
            'min_inventory': self.min_inventory,
            'current_inventory': self.inventory,
            'current_cash': self.cash,
            'total_trades': self.total_buys + self.total_sells,
            'buy_sell_ratio': self.total_buys / max(1, self.total_sells)
        }
    
    def reset(self):
        """Reset the model state"""
        self.inventory = 0
        self.cash = 0.0
        self.pnl_history = []
        self.quote_history = []
        self.execution_history = []
        self.total_buys = 0
        self.total_sells = 0
        self.max_inventory = 0
        self.min_inventory = 0
        
    def calculate_reservation_prices_path(self,
                                         mid_prices: np.ndarray,
                                         inventories: np.ndarray,
                                         times_to_horizon: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate reservation prices for entire path (vectorized for efficiency)
        """
        if self.params.horizon == TimeHorizon.FINITE:
            if times_to_horizon is None:
                times_to_horizon = np.full_like(mid_prices, self.params.T)
            adjustments = inventories * self.params.gamma * (self.params.sigma ** 2) * times_to_horizon
        else:
            w = 0.5 * (self.params.gamma ** 2) * (self.params.sigma ** 2) * ((self.params.q_max + 1) ** 2)
            adjustments = inventories * self.params.gamma * (self.params.sigma ** 2) / (2 * w)
            
        return mid_prices - adjustments
    
    def calculate_optimal_quotes_path(self,
                                     mid_prices: np.ndarray,
                                     inventories: np.ndarray,
                                     times_to_horizon: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate optimal bid and ask prices for entire path (vectorized)
        
        Returns: (bid_prices, ask_prices)
        """
        # Calculate reservation prices
        reservation_prices = self.calculate_reservation_prices_path(
            mid_prices, inventories, times_to_horizon
        )
        
        # Calculate optimal spreads
        if self.params.horizon == TimeHorizon.FINITE:
            if times_to_horizon is None:
                times_to_horizon = np.full_like(mid_prices, self.params.T)
            inventory_risk = self.params.gamma * (self.params.sigma ** 2) * times_to_horizon
        else:
            inventory_risk = np.full_like(mid_prices, 
                                        self.params.gamma * (self.params.sigma ** 2) / self.params.q_max)
            
        market_making_profit = (2 / self.params.gamma) * math.log(1 + self.params.gamma / self.params.k)
        optimal_spreads = inventory_risk + market_making_profit
        
        # Calculate quotes
        half_spreads = optimal_spreads / 2
        bid_prices = reservation_prices - half_spreads
        ask_prices = reservation_prices + half_spreads
        
        return bid_prices, ask_prices