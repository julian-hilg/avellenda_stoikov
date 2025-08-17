from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Tuple
import numpy as np
from enum import Enum


class OrderSide(Enum):
    BID = "BID"
    ASK = "ASK"


@dataclass
class Trade:
    timestamp: datetime
    price: float
    volume: float
    side: Optional[OrderSide] = None
    
    
@dataclass
class OrderBookLevel:
    price: float
    volume: float
    order_count: int = 1


@dataclass
class OrderBook:
    timestamp: datetime
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    
    @property
    def mid_price(self) -> float:
        if self.bids and self.asks:
            return (self.bids[0].price + self.asks[0].price) / 2
        return 0.0
    
    @property
    def spread(self) -> float:
        if self.bids and self.asks:
            return self.asks[0].price - self.bids[0].price
        return 0.0
    
    def get_volume_weighted_mid(self, depth: int = 5) -> float:
        """Calculate volume-weighted mid price up to specified depth"""
        bid_sum = 0.0
        bid_volume = 0.0
        ask_sum = 0.0
        ask_volume = 0.0
        
        for i in range(min(depth, len(self.bids))):
            bid_sum += self.bids[i].price * self.bids[i].volume
            bid_volume += self.bids[i].volume
            
        for i in range(min(depth, len(self.asks))):
            ask_sum += self.asks[i].price * self.asks[i].volume
            ask_volume += self.asks[i].volume
            
        if bid_volume > 0 and ask_volume > 0:
            return (bid_sum/bid_volume + ask_sum/ask_volume) / 2
        return self.mid_price
    
    def get_market_impact(self, volume: float, side: OrderSide) -> float:
        """Calculate the price impact of executing a market order of given volume"""
        levels = self.asks if side == OrderSide.BID else self.bids
        remaining_volume = volume
        total_cost = 0.0
        
        for level in levels:
            if remaining_volume <= 0:
                break
            executed = min(remaining_volume, level.volume)
            total_cost += executed * level.price
            remaining_volume -= executed
            
        if remaining_volume > 0:
            # Not enough liquidity
            return float('inf')
            
        return total_cost / volume


@dataclass
class MarketData:
    timestamp: datetime
    symbol: str
    mid_price: float
    bid: float
    ask: float
    bid_volume: float
    ask_volume: float
    last_trade_price: Optional[float] = None
    last_trade_volume: Optional[float] = None
    volume_24h: Optional[float] = None
    order_book: Optional[OrderBook] = None
    trades: Optional[List[Trade]] = None
    
    @property
    def spread(self) -> float:
        return self.ask - self.bid
    
    @property
    def spread_percentage(self) -> float:
        return self.spread / self.mid_price * 100 if self.mid_price > 0 else 0


@dataclass
class DividendEvent:
    ex_date: datetime
    payment_date: datetime
    amount: float
    symbol: str


class DataClient(ABC):
    """Abstract base class for market data clients"""
    
    @abstractmethod
    def get_current_market_data(self, symbol: str) -> MarketData:
        """Get current market data for a symbol"""
        pass
    
    @abstractmethod
    def get_historical_data(self, 
                          symbol: str, 
                          start: datetime, 
                          end: datetime,
                          interval: str = '1m') -> List[MarketData]:
        """Get historical market data"""
        pass
    
    @abstractmethod
    def get_order_book(self, symbol: str, depth: int = 10) -> OrderBook:
        """Get current order book"""
        pass
    
    @abstractmethod
    def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Trade]:
        """Get recent trades"""
        pass
    
    @abstractmethod
    def subscribe_market_data(self, symbol: str, callback):
        """Subscribe to real-time market data updates"""
        pass
    
    @abstractmethod
    def unsubscribe_market_data(self, symbol: str):
        """Unsubscribe from market data updates"""
        pass
    
    @abstractmethod
    def get_dividends(self, symbol: str, 
                     start: datetime, 
                     end: datetime) -> List[DividendEvent]:
        """Get dividend events for a symbol"""
        pass
    
    @abstractmethod
    def is_market_open(self, timestamp: Optional[datetime] = None) -> bool:
        """Check if market is open at given timestamp"""
        pass
    
    @abstractmethod
    def get_next_market_open(self, timestamp: datetime) -> datetime:
        """Get next market open time after given timestamp"""
        pass
    
    @abstractmethod
    def get_next_market_close(self, timestamp: datetime) -> datetime:
        """Get next market close time after given timestamp"""
        pass