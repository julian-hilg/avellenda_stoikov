import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from datetime import datetime
from enum import Enum


class InventoryAction(Enum):
    NONE = "none"
    REDUCE_LONG = "reduce_long"
    REDUCE_SHORT = "reduce_short"
    HALT_TRADING = "halt_trading"


@dataclass
class InventoryLimits:
    """Inventory risk limits"""
    max_position: int  # Maximum absolute position
    soft_limit: int  # Soft limit for position reduction
    hard_limit: int  # Hard limit for trading halt
    max_order_size: float  # Maximum order size
    
    def validate(self):
        assert self.soft_limit <= self.hard_limit <= self.max_position
        assert self.max_order_size > 0


@dataclass
class InventoryState:
    """Current inventory state and metrics"""
    current_position: int
    average_entry_price: float
    total_buys: int
    total_sells: int
    realized_pnl: float
    unrealized_pnl: float
    max_position_today: int
    min_position_today: int
    timestamp: datetime
    
    @property
    def net_position(self) -> int:
        return self.current_position
    
    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl
    
    @property
    def turnover(self) -> int:
        return self.total_buys + self.total_sells


class InventoryManager:
    """
    Manages inventory position and risk for market making
    
    Features:
    - Position tracking and limits
    - Risk-based order sizing
    - Inventory mean reversion
    - P&L tracking
    """
    
    def __init__(self, limits: InventoryLimits):
        self.limits = limits
        self.limits.validate()
        
        # Position tracking
        self.position = 0
        self.positions_history = []
        
        # Price tracking for P&L
        self.total_buy_value = 0.0
        self.total_sell_value = 0.0
        self.total_buy_quantity = 0
        self.total_sell_quantity = 0
        
        # Daily limits
        self.max_position_today = 0
        self.min_position_today = 0
        self.trades_today = []
        
        # P&L tracking
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        
    def get_average_entry_price(self) -> float:
        """Calculate average entry price for current position"""
        if self.position == 0:
            return 0.0
        elif self.position > 0:
            # Long position - use average buy price
            if self.total_buy_quantity > self.total_sell_quantity:
                net_quantity = self.total_buy_quantity - self.total_sell_quantity
                net_value = self.total_buy_value - self.total_sell_value
                return net_value / net_quantity if net_quantity > 0 else 0.0
        else:
            # Short position - use average sell price
            if self.total_sell_quantity > self.total_buy_quantity:
                net_quantity = self.total_sell_quantity - self.total_buy_quantity
                net_value = self.total_sell_value - self.total_buy_value
                return net_value / net_quantity if net_quantity > 0 else 0.0
        return 0.0
    
    def check_inventory_action(self) -> InventoryAction:
        """Determine required inventory management action"""
        abs_position = abs(self.position)
        
        if abs_position >= self.limits.hard_limit:
            return InventoryAction.HALT_TRADING
        elif abs_position >= self.limits.soft_limit:
            if self.position > 0:
                return InventoryAction.REDUCE_LONG
            else:
                return InventoryAction.REDUCE_SHORT
        else:
            return InventoryAction.NONE
    
    def calculate_order_size(self, 
                           is_buy: bool,
                           base_size: float = 1.0,
                           aggressiveness: float = 1.0) -> float:
        """
        Calculate adjusted order size based on inventory risk
        
        Args:
            is_buy: Whether this is a buy order
            base_size: Base order size
            aggressiveness: Aggressiveness factor (0.5 = conservative, 2.0 = aggressive)
        
        Returns:
            Adjusted order size
        """
        # Check position limits
        if is_buy and self.position >= self.limits.max_position:
            return 0.0
        elif not is_buy and self.position <= -self.limits.max_position:
            return 0.0
        
        # Base size with max order size limit
        size = min(base_size, self.limits.max_order_size)
        
        # Adjust based on inventory position
        abs_position = abs(self.position)
        position_ratio = abs_position / self.limits.max_position
        
        # Reduce size as we approach limits
        if position_ratio > 0.5:
            size_reduction = 1.0 - (position_ratio - 0.5) * 2 * 0.7  # Max 70% reduction
            size *= size_reduction
        
        # Apply aggressiveness factor
        size *= aggressiveness
        
        # Additional reduction if we need to reduce position
        action = self.check_inventory_action()
        if action == InventoryAction.REDUCE_LONG and is_buy:
            size *= 0.3  # Significantly reduce buy size
        elif action == InventoryAction.REDUCE_SHORT and not is_buy:
            size *= 0.3  # Significantly reduce sell size
        elif action == InventoryAction.HALT_TRADING:
            # Only allow position-reducing trades
            if (self.position > 0 and is_buy) or (self.position < 0 and not is_buy):
                size = 0.0
        
        return size
    
    def calculate_quote_adjustment(self) -> Tuple[float, float]:
        """
        Calculate bid/ask quote adjustments based on inventory
        
        Returns:
            (bid_adjustment, ask_adjustment) as percentage adjustments
        """
        if self.position == 0:
            return (0.0, 0.0)
        
        # Normalize position
        position_ratio = self.position / self.limits.max_position
        
        # Base adjustment (in basis points)
        base_adjustment = 5  # 5 bps
        
        # Calculate adjustments
        if self.position > 0:
            # Long position - lower bids, raise asks to reduce position
            bid_adjustment = -base_adjustment * abs(position_ratio)
            ask_adjustment = base_adjustment * abs(position_ratio) * 0.5
        else:
            # Short position - raise bids, lower asks to reduce position
            bid_adjustment = base_adjustment * abs(position_ratio) * 0.5
            ask_adjustment = -base_adjustment * abs(position_ratio)
        
        # Convert to percentage
        bid_adjustment /= 10000
        ask_adjustment /= 10000
        
        return (bid_adjustment, ask_adjustment)
    
    def execute_trade(self,
                     price: float,
                     quantity: float,
                     is_buy: bool,
                     timestamp: datetime) -> Dict[str, float]:
        """
        Execute a trade and update inventory
        
        Returns:
            Dictionary with trade metrics
        """
        prev_position = self.position
        
        if is_buy:
            self.position += quantity
            self.total_buy_quantity += quantity
            self.total_buy_value += price * quantity
        else:
            self.position -= quantity
            self.total_sell_quantity += quantity
            self.total_sell_value += price * quantity
        
        # Update daily limits
        self.max_position_today = max(self.max_position_today, self.position)
        self.min_position_today = min(self.min_position_today, self.position)
        
        # Track trade
        trade = {
            'timestamp': timestamp,
            'price': price,
            'quantity': quantity,
            'is_buy': is_buy,
            'position_before': prev_position,
            'position_after': self.position
        }
        self.trades_today.append(trade)
        self.positions_history.append((timestamp, self.position))
        
        # Calculate realized P&L for this trade
        trade_pnl = 0.0
        if prev_position > 0 and not is_buy:
            # Closing long position
            avg_entry = self.get_average_entry_price()
            trade_pnl = (price - avg_entry) * min(quantity, prev_position)
            self.realized_pnl += trade_pnl
        elif prev_position < 0 and is_buy:
            # Closing short position
            avg_entry = self.get_average_entry_price()
            trade_pnl = (avg_entry - price) * min(quantity, abs(prev_position))
            self.realized_pnl += trade_pnl
        
        return {
            'position_change': self.position - prev_position,
            'new_position': self.position,
            'trade_pnl': trade_pnl,
            'total_realized_pnl': self.realized_pnl
        }
    
    def update_unrealized_pnl(self, current_price: float):
        """Update unrealized P&L based on current price"""
        if self.position == 0:
            self.unrealized_pnl = 0.0
        else:
            avg_entry = self.get_average_entry_price()
            if self.position > 0:
                self.unrealized_pnl = (current_price - avg_entry) * self.position
            else:
                self.unrealized_pnl = (avg_entry - current_price) * abs(self.position)
    
    def get_state(self, current_price: float) -> InventoryState:
        """Get current inventory state"""
        self.update_unrealized_pnl(current_price)
        
        return InventoryState(
            current_position=self.position,
            average_entry_price=self.get_average_entry_price(),
            total_buys=self.total_buy_quantity,
            total_sells=self.total_sell_quantity,
            realized_pnl=self.realized_pnl,
            unrealized_pnl=self.unrealized_pnl,
            max_position_today=self.max_position_today,
            min_position_today=self.min_position_today,
            timestamp=datetime.now()
        )
    
    def calculate_inventory_score(self) -> float:
        """
        Calculate inventory health score (0-100)
        Higher score means healthier inventory position
        """
        score = 100.0
        
        # Position size penalty
        position_ratio = abs(self.position) / self.limits.max_position
        score -= position_ratio * 50  # Max 50 point penalty
        
        # Check if approaching limits
        if abs(self.position) >= self.limits.soft_limit:
            score -= 20
        if abs(self.position) >= self.limits.hard_limit:
            score -= 30
        
        return max(0, score)
    
    def reset_daily_stats(self):
        """Reset daily statistics"""
        self.max_position_today = self.position
        self.min_position_today = self.position
        self.trades_today = []
    
    def reset(self):
        """Reset all inventory tracking"""
        self.position = 0
        self.positions_history = []
        self.total_buy_value = 0.0
        self.total_sell_value = 0.0
        self.total_buy_quantity = 0
        self.total_sell_quantity = 0
        self.max_position_today = 0
        self.min_position_today = 0
        self.trades_today = []
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0