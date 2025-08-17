import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Callable
import pandas as pd
from scipy.stats import norm
import math
from .base import (
    DataClient, MarketData, OrderBook, Trade, 
    OrderBookLevel, OrderSide, DividendEvent
)


class SimulatedDataClient(DataClient):
    """Simulated market data client with realistic features"""
    
    def __init__(self,
                 initial_price: float = 100.0,
                 volatility: float = 0.2,
                 drift: float = 0.05,
                 tick_size: float = 0.01,
                 lot_size: float = 100,
                 market_open: str = "09:30",
                 market_close: str = "16:00",
                 timezone: str = "US/Eastern",
                 seed: Optional[int] = None):
        
        self.initial_price = initial_price
        self.volatility = volatility
        self.drift = drift
        self.tick_size = tick_size
        self.lot_size = lot_size
        self.market_open = market_open
        self.market_close = market_close
        self.timezone = timezone
        
        if seed is not None:
            np.random.seed(seed)
            
        self.current_prices = {}
        self.price_history = {}
        self.subscribers = {}
        self.dividend_schedule = {}
        
        # Order book parameters (based on empirical studies)
        self.spread_mean = 0.001  # 0.1% average spread
        self.spread_std = 0.0005
        self.book_depth_lambda = 1.5  # Power law decay for order book depth
        self.order_arrival_rate = 140  # Orders per time unit
        
    def _generate_price_path(self, 
                            symbol: str,
                            start: datetime, 
                            end: datetime,
                            interval: str = '1m') -> np.ndarray:
        """Generate price path using geometric Brownian motion with market hours"""
        
        # Parse interval
        interval_minutes = self._parse_interval(interval)
        
        # Generate trading timestamps (excluding weekends and after hours)
        timestamps = self._generate_trading_timestamps(start, end, interval_minutes)
        n_steps = len(timestamps)
        
        if n_steps == 0:
            return np.array([])
        
        dt = interval_minutes / (252 * 6.5 * 60)  # Convert to years
        
        # Generate returns with GBM
        random_shocks = np.random.randn(n_steps)
        returns = (self.drift - 0.5 * self.volatility**2) * dt + \
                 self.volatility * np.sqrt(dt) * random_shocks
        
        # Add intraday patterns
        intraday_pattern = self._get_intraday_pattern(timestamps)
        returns += intraday_pattern
        
        # Calculate prices
        if symbol not in self.current_prices:
            self.current_prices[symbol] = self.initial_price
            
        prices = self.current_prices[symbol] * np.exp(np.cumsum(returns))
        
        # Round to tick size
        prices = np.round(prices / self.tick_size) * self.tick_size
        
        # Apply dividend adjustments
        prices = self._apply_dividend_adjustments(symbol, timestamps, prices)
        
        return prices, timestamps
    
    def _generate_trading_timestamps(self, 
                                    start: datetime, 
                                    end: datetime,
                                    interval_minutes: int) -> List[datetime]:
        """Generate timestamps only during market hours"""
        timestamps = []
        current = start
        
        market_open_hour, market_open_min = map(int, self.market_open.split(':'))
        market_close_hour, market_close_min = map(int, self.market_close.split(':'))
        
        while current <= end:
            # Skip weekends
            if current.weekday() >= 5:
                current += timedelta(days=1)
                current = current.replace(hour=market_open_hour, 
                                        minute=market_open_min, 
                                        second=0, 
                                        microsecond=0)
                continue
            
            # Check if within market hours
            market_open = current.replace(hour=market_open_hour, 
                                         minute=market_open_min,
                                         second=0,
                                         microsecond=0)
            market_close = current.replace(hour=market_close_hour,
                                          minute=market_close_min,
                                          second=0,
                                          microsecond=0)
            
            if market_open <= current <= market_close:
                timestamps.append(current)
                current += timedelta(minutes=interval_minutes)
            else:
                # Jump to next market open
                if current > market_close:
                    current = current + timedelta(days=1)
                current = current.replace(hour=market_open_hour,
                                        minute=market_open_min,
                                        second=0,
                                        microsecond=0)
                # Skip weekend if necessary
                while current.weekday() >= 5:
                    current += timedelta(days=1)
                    
        return timestamps
    
    def _get_intraday_pattern(self, timestamps: List[datetime]) -> np.ndarray:
        """Generate U-shaped intraday volatility pattern"""
        if not timestamps:
            return np.array([])
            
        market_open_hour = int(self.market_open.split(':')[0])
        market_close_hour = int(self.market_close.split(':')[0])
        
        patterns = []
        for ts in timestamps:
            hours_from_open = (ts.hour + ts.minute/60) - market_open_hour
            hours_to_close = market_close_hour - (ts.hour + ts.minute/60)
            
            # U-shaped pattern: higher volatility at open and close
            if hours_from_open < 1:
                multiplier = 1.5 - 0.5 * hours_from_open
            elif hours_to_close < 1:
                multiplier = 1.5 - 0.5 * hours_to_close
            else:
                multiplier = 0.7
                
            patterns.append(np.random.randn() * 0.0001 * multiplier)
            
        return np.array(patterns)
    
    def _parse_interval(self, interval: str) -> int:
        """Parse interval string to minutes"""
        if interval.endswith('m'):
            return int(interval[:-1])
        elif interval.endswith('h'):
            return int(interval[:-1]) * 60
        elif interval.endswith('d'):
            return int(interval[:-1]) * 6.5 * 60  # 6.5 trading hours per day
        else:
            return 1
    
    def _apply_dividend_adjustments(self, 
                                   symbol: str,
                                   timestamps: List[datetime],
                                   prices: np.ndarray) -> np.ndarray:
        """Apply dividend adjustments to prices"""
        if symbol not in self.dividend_schedule:
            return prices
            
        adjusted_prices = prices.copy()
        for dividend in self.dividend_schedule[symbol]:
            # Find index of ex-dividend date
            for i, ts in enumerate(timestamps):
                if ts.date() >= dividend.ex_date.date():
                    # Adjust all prices from ex-date forward
                    adjustment_factor = 1 - (dividend.amount / adjusted_prices[i])
                    adjusted_prices[i:] *= adjustment_factor
                    break
                    
        return adjusted_prices
    
    def _generate_order_book(self, 
                            mid_price: float,
                            timestamp: datetime) -> OrderBook:
        """Generate realistic order book with power-law decay"""
        
        # Generate spread
        spread = max(self.tick_size, 
                    abs(np.random.normal(self.spread_mean, self.spread_std)) * mid_price)
        
        best_bid = mid_price - spread/2
        best_ask = mid_price + spread/2
        
        # Round to tick size
        best_bid = np.floor(best_bid / self.tick_size) * self.tick_size
        best_ask = np.ceil(best_ask / self.tick_size) * self.tick_size
        
        # Generate order book levels with power law decay
        n_levels = 10
        bids = []
        asks = []
        
        for i in range(n_levels):
            # Price levels
            bid_price = best_bid - i * self.tick_size
            ask_price = best_ask + i * self.tick_size
            
            # Volume follows power law decay
            distance_factor = (i + 1) ** (-self.book_depth_lambda)
            base_volume = np.random.exponential(1000) * distance_factor
            
            bid_volume = max(self.lot_size, 
                           np.round(base_volume / self.lot_size) * self.lot_size)
            ask_volume = max(self.lot_size,
                           np.round(base_volume / self.lot_size) * self.lot_size)
            
            # Order count
            order_count = max(1, np.random.poisson(5 * distance_factor))
            
            bids.append(OrderBookLevel(bid_price, bid_volume, order_count))
            asks.append(OrderBookLevel(ask_price, ask_volume, order_count))
            
        return OrderBook(timestamp, bids, asks)
    
    def _generate_trades(self, 
                        mid_price: float,
                        timestamp: datetime,
                        n_trades: int = 10) -> List[Trade]:
        """Generate realistic trades around mid price"""
        trades = []
        
        for i in range(n_trades):
            # Time offset within the period
            time_offset = timedelta(seconds=np.random.uniform(0, 60))
            trade_time = timestamp - time_offset
            
            # Trade price follows normal distribution around mid
            price_offset = np.random.normal(0, self.spread_mean * mid_price)
            trade_price = mid_price + price_offset
            trade_price = np.round(trade_price / self.tick_size) * self.tick_size
            
            # Volume follows exponential distribution
            volume = np.random.exponential(500)
            volume = np.round(volume / self.lot_size) * self.lot_size
            
            # Determine side based on price relative to mid
            side = OrderSide.BID if trade_price < mid_price else OrderSide.ASK
            
            trades.append(Trade(trade_time, trade_price, volume, side))
            
        return sorted(trades, key=lambda x: x.timestamp, reverse=True)
    
    def get_current_market_data(self, symbol: str) -> MarketData:
        """Get current simulated market data"""
        now = datetime.now()
        
        if symbol not in self.current_prices:
            self.current_prices[symbol] = self.initial_price
            
        # Simulate small price movement
        price_change = np.random.normal(0, self.volatility * self.current_prices[symbol] * 0.0001)
        self.current_prices[symbol] += price_change
        
        mid_price = self.current_prices[symbol]
        order_book = self._generate_order_book(mid_price, now)
        trades = self._generate_trades(mid_price, now)
        
        return MarketData(
            timestamp=now,
            symbol=symbol,
            mid_price=mid_price,
            bid=order_book.bids[0].price,
            ask=order_book.asks[0].price,
            bid_volume=order_book.bids[0].volume,
            ask_volume=order_book.asks[0].volume,
            last_trade_price=trades[0].price if trades else mid_price,
            last_trade_volume=trades[0].volume if trades else 0,
            volume_24h=np.random.exponential(1000000),
            order_book=order_book,
            trades=trades
        )
    
    def get_historical_data(self,
                          symbol: str,
                          start: datetime,
                          end: datetime,
                          interval: str = '1m') -> List[MarketData]:
        """Get historical simulated data"""
        prices, timestamps = self._generate_price_path(symbol, start, end, interval)
        
        market_data = []
        for i, (price, ts) in enumerate(zip(prices, timestamps)):
            order_book = self._generate_order_book(price, ts)
            trades = self._generate_trades(price, ts, n_trades=5)
            
            market_data.append(MarketData(
                timestamp=ts,
                symbol=symbol,
                mid_price=price,
                bid=order_book.bids[0].price,
                ask=order_book.asks[0].price,
                bid_volume=order_book.bids[0].volume,
                ask_volume=order_book.asks[0].volume,
                last_trade_price=trades[0].price if trades else price,
                last_trade_volume=trades[0].volume if trades else 0,
                volume_24h=np.random.exponential(1000000),
                order_book=order_book,
                trades=trades
            ))
            
        # Update current price
        if prices.size > 0:
            self.current_prices[symbol] = prices[-1]
            
        return market_data
    
    def get_order_book(self, symbol: str, depth: int = 10) -> OrderBook:
        """Get current order book"""
        if symbol not in self.current_prices:
            self.current_prices[symbol] = self.initial_price
            
        return self._generate_order_book(self.current_prices[symbol], datetime.now())
    
    def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Trade]:
        """Get recent simulated trades"""
        if symbol not in self.current_prices:
            self.current_prices[symbol] = self.initial_price
            
        return self._generate_trades(
            self.current_prices[symbol], 
            datetime.now(), 
            n_trades=limit
        )
    
    def subscribe_market_data(self, symbol: str, callback: Callable):
        """Subscribe to market data updates"""
        if symbol not in self.subscribers:
            self.subscribers[symbol] = []
        self.subscribers[symbol].append(callback)
    
    def unsubscribe_market_data(self, symbol: str):
        """Unsubscribe from market data"""
        if symbol in self.subscribers:
            del self.subscribers[symbol]
    
    def get_dividends(self, 
                     symbol: str,
                     start: datetime,
                     end: datetime) -> List[DividendEvent]:
        """Get simulated dividend events"""
        if symbol not in self.dividend_schedule:
            # Generate quarterly dividends
            dividends = []
            current = start
            while current <= end:
                # Quarterly dividends
                if current.month in [3, 6, 9, 12]:
                    ex_date = current.replace(day=15)
                    if ex_date.weekday() >= 5:  # Weekend adjustment
                        ex_date += timedelta(days=7-ex_date.weekday())
                    
                    payment_date = ex_date + timedelta(days=14)
                    amount = np.random.uniform(0.5, 2.0)  # $0.50 to $2.00 per share
                    
                    dividends.append(DividendEvent(
                        ex_date=ex_date,
                        payment_date=payment_date,
                        amount=amount,
                        symbol=symbol
                    ))
                    
                current = current + timedelta(days=30)
                
            self.dividend_schedule[symbol] = dividends
            
        return [d for d in self.dividend_schedule[symbol] 
                if start <= d.ex_date <= end]
    
    def is_market_open(self, timestamp: Optional[datetime] = None) -> bool:
        """Check if market is open"""
        if timestamp is None:
            timestamp = datetime.now()
            
        # Check weekend
        if timestamp.weekday() >= 5:
            return False
            
        # Check market hours
        market_open_hour, market_open_min = map(int, self.market_open.split(':'))
        market_close_hour, market_close_min = map(int, self.market_close.split(':'))
        
        market_open = timestamp.replace(hour=market_open_hour,
                                       minute=market_open_min,
                                       second=0,
                                       microsecond=0)
        market_close = timestamp.replace(hour=market_close_hour,
                                        minute=market_close_min,
                                        second=0,
                                        microsecond=0)
        
        return market_open <= timestamp <= market_close
    
    def get_next_market_open(self, timestamp: datetime) -> datetime:
        """Get next market open time"""
        market_open_hour, market_open_min = map(int, self.market_open.split(':'))
        
        next_open = timestamp.replace(hour=market_open_hour,
                                     minute=market_open_min,
                                     second=0,
                                     microsecond=0)
        
        # If already past today's open, move to tomorrow
        if timestamp >= next_open:
            next_open += timedelta(days=1)
            
        # Skip weekends
        while next_open.weekday() >= 5:
            next_open += timedelta(days=1)
            
        return next_open
    
    def get_next_market_close(self, timestamp: datetime) -> datetime:
        """Get next market close time"""
        market_close_hour, market_close_min = map(int, self.market_close.split(':'))
        
        next_close = timestamp.replace(hour=market_close_hour,
                                      minute=market_close_min,
                                      second=0,
                                      microsecond=0)
        
        # If already past today's close, move to tomorrow
        if timestamp >= next_close:
            next_close += timedelta(days=1)
            # Skip to Monday if weekend
            while next_close.weekday() >= 5:
                next_close += timedelta(days=1)
                
        return next_close