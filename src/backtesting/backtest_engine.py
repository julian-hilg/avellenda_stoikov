import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any
from datetime import datetime, timedelta
import logging
from tqdm import tqdm

from ..data_client import DataClient, MarketData
from ..market_making import (
    AvellanedaStoikovModel, 
    ASParameters,
    MarketState,
    InventoryManager,
    InventoryLimits,
    RiskAnalyzer
)


@dataclass
class BacktestConfig:
    """Configuration for backtesting"""
    start_date: datetime
    end_date: datetime
    symbols: List[str]
    initial_cash: float = 100000.0
    data_frequency: str = '1m'  # 1m, 5m, 15m, 30m, 1h
    enable_logging: bool = True
    log_level: str = 'INFO'
    save_results: bool = True
    results_path: str = './backtest_results'
    
    # Market making specific
    as_params: Optional[ASParameters] = None
    inventory_limits: Optional[InventoryLimits] = None
    
    # Execution settings
    slippage_bps: float = 1.0  # Basis points of slippage
    commission_bps: float = 2.0  # Basis points commission
    min_order_size: float = 1.0
    max_order_size: float = 100.0
    
    # Risk limits
    max_daily_loss: Optional[float] = None
    max_position_value: Optional[float] = None
    stop_loss_pct: Optional[float] = None
    

@dataclass
class BacktestResult:
    """Results from backtesting"""
    config: BacktestConfig
    final_pnl: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    
    # Time series data
    pnl_series: pd.Series = field(default_factory=pd.Series)
    returns_series: pd.Series = field(default_factory=pd.Series)
    inventory_series: pd.Series = field(default_factory=pd.Series)
    cash_series: pd.Series = field(default_factory=pd.Series)
    
    # Detailed metrics
    risk_metrics: Optional[Dict[str, float]] = None
    trade_history: List[Dict] = field(default_factory=list)
    quote_history: List[Dict] = field(default_factory=list)
    
    # Performance by symbol
    symbol_performance: Dict[str, Dict] = field(default_factory=dict)
    

class BacktestEngine:
    """
    Main backtesting engine for market making strategies
    """
    
    def __init__(self, 
                 data_client: DataClient,
                 config: BacktestConfig):
        """
        Initialize backtest engine
        
        Args:
            data_client: Data client for market data
            config: Backtest configuration
        """
        self.data_client = data_client
        self.config = config
        
        # Set up logging
        if config.enable_logging:
            logging.basicConfig(
                level=getattr(logging, config.log_level),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = None
            
        # Initialize components
        self.as_model = None
        self.inventory_manager = None
        self.risk_analyzer = None
        
        # State tracking
        self.cash = config.initial_cash
        self.positions = {symbol: 0 for symbol in config.symbols}
        self.pnl_history = []
        self.trade_history = []
        self.quote_history = []
        
    def initialize_models(self):
        """Initialize trading models"""
        # Initialize Avellaneda-Stoikov model
        if self.config.as_params:
            self.as_model = AvellanedaStoikovModel(self.config.as_params)
        
        # Initialize inventory manager
        if self.config.inventory_limits:
            self.inventory_manager = InventoryManager(self.config.inventory_limits)
        
        # Initialize risk analyzer
        self.risk_analyzer = RiskAnalyzer()
        
    def run(self) -> BacktestResult:
        """
        Run the backtest
        
        Returns:
            BacktestResult with performance metrics
        """
        self._log("Starting backtest")
        self.initialize_models()
        
        results = {
            'timestamps': [],
            'pnl': [],
            'cash': [],
            'inventory': [],
            'returns': []
        }
        
        # Process each symbol
        for symbol in self.config.symbols:
            self._log(f"Processing symbol: {symbol}")
            symbol_results = self._backtest_symbol(symbol)
            
            # Aggregate results
            for key in results:
                if key in symbol_results:
                    results[key].extend(symbol_results[key])
        
        # Calculate final metrics
        final_result = self._calculate_final_metrics(results)
        
        if self.config.save_results:
            self._save_results(final_result)
        
        self._log("Backtest completed")
        return final_result
    
    def _backtest_symbol(self, symbol: str) -> Dict:
        """Backtest a single symbol"""
        # Get historical data
        market_data = self.data_client.get_historical_data(
            symbol,
            self.config.start_date,
            self.config.end_date,
            self.config.data_frequency
        )
        
        if not market_data:
            self._log(f"No data available for {symbol}", level='WARNING')
            return {}
        
        results = {
            'timestamps': [],
            'pnl': [],
            'cash': [],
            'inventory': [],
            'returns': [],
            'trades': []
        }
        
        # Process each time step
        for i, data in enumerate(tqdm(market_data, desc=f"Backtesting {symbol}")):
            # Check risk limits
            if self._check_risk_limits():
                self._log("Risk limits breached, halting trading", level='WARNING')
                break
            
            # Update market state
            state = self._create_market_state(data, symbol)
            
            # Get optimal quotes from AS model
            if self.as_model:
                quotes = self.as_model.get_optimal_quotes(state)
                self.quote_history.append({
                    'timestamp': data.timestamp,
                    'symbol': symbol,
                    'bid': quotes.bid_price,
                    'ask': quotes.ask_price,
                    'spread': quotes.spread,
                    'reservation_price': quotes.reservation_price
                })
                
                # Simulate order execution
                executed = self._simulate_execution(quotes, data)
                
                if executed['bid_executed'] or executed['ask_executed']:
                    self._process_execution(executed, data, symbol)
            
            # Update metrics
            current_pnl = self._calculate_pnl(data.mid_price)
            results['timestamps'].append(data.timestamp)
            results['pnl'].append(current_pnl)
            results['cash'].append(self.cash)
            results['inventory'].append(self.positions[symbol])
            
            # Calculate returns
            if i > 0:
                prev_pnl = results['pnl'][-2]
                if prev_pnl != 0:
                    return_pct = (current_pnl - prev_pnl) / abs(prev_pnl)
                    results['returns'].append(return_pct)
                else:
                    results['returns'].append(0.0)
            
            # Update risk analyzer
            if self.risk_analyzer:
                self.risk_analyzer.add_pnl_point(current_pnl, data.timestamp)
                self.risk_analyzer.add_position(self.positions[symbol], data.timestamp)
        
        return results
    
    def _create_market_state(self, data: MarketData, symbol: str) -> MarketState:
        """Create market state for the model"""
        time_to_horizon = None
        if self.config.as_params and self.config.as_params.T:
            elapsed = (data.timestamp - self.config.start_date).total_seconds()
            total = (self.config.end_date - self.config.start_date).total_seconds()
            time_to_horizon = self.config.as_params.T * (1 - elapsed/total)
        
        return MarketState(
            mid_price=data.mid_price,
            inventory=self.positions[symbol],
            cash=self.cash,
            timestamp=data.timestamp,
            time_to_horizon=time_to_horizon
        )
    
    def _simulate_execution(self, quotes, market_data: MarketData) -> Dict:
        """Simulate order execution based on market data"""
        result = {
            'bid_executed': False,
            'ask_executed': False,
            'bid_price': quotes.bid_price,
            'ask_price': quotes.ask_price,
            'bid_size': quotes.bid_size,
            'ask_size': quotes.ask_size
        }
        
        # Simple execution logic based on market trades
        if market_data.trades:
            for trade in market_data.trades:
                # Check if our bid would have been hit
                if trade.price <= quotes.bid_price and quotes.bid_size > 0:
                    result['bid_executed'] = True
                    result['bid_price'] = self._apply_slippage(quotes.bid_price, True)
                    
                # Check if our ask would have been lifted
                if trade.price >= quotes.ask_price and quotes.ask_size > 0:
                    result['ask_executed'] = True
                    result['ask_price'] = self._apply_slippage(quotes.ask_price, False)
        
        # Alternative: Use probability-based execution
        if self.as_model and not result['bid_executed'] and not result['ask_executed']:
            bid_executed, ask_executed, _, _ = self.as_model.simulate_execution(
                quotes, market_data.mid_price
            )
            result['bid_executed'] = bid_executed
            result['ask_executed'] = ask_executed
        
        return result
    
    def _apply_slippage(self, price: float, is_buy: bool) -> float:
        """Apply slippage to execution price"""
        slippage = price * self.config.slippage_bps / 10000
        if is_buy:
            return price + slippage  # Pay more when buying
        else:
            return price - slippage  # Receive less when selling
    
    def _calculate_commission(self, price: float, size: float) -> float:
        """Calculate commission for a trade"""
        return price * size * self.config.commission_bps / 10000
    
    def _process_execution(self, executed: Dict, market_data: MarketData, symbol: str):
        """Process executed orders"""
        if executed['bid_executed']:
            # We bought
            size = min(executed['bid_size'], self.config.max_order_size)
            cost = executed['bid_price'] * size
            commission = self._calculate_commission(executed['bid_price'], size)
            
            self.positions[symbol] += size
            self.cash -= (cost + commission)
            
            # Record trade
            trade = {
                'timestamp': market_data.timestamp,
                'symbol': symbol,
                'side': 'BUY',
                'price': executed['bid_price'],
                'size': size,
                'commission': commission
            }
            self.trade_history.append(trade)
            
            # Update models
            if self.as_model:
                self.as_model.update_state(True, False, executed['bid_price'], 0, size, 0)
            if self.inventory_manager:
                self.inventory_manager.execute_trade(
                    executed['bid_price'], size, True, market_data.timestamp
                )
            if self.risk_analyzer:
                self.risk_analyzer.add_trade(
                    executed['bid_price'], size, True, market_data.timestamp
                )
        
        if executed['ask_executed']:
            # We sold
            size = min(executed['ask_size'], self.config.max_order_size)
            revenue = executed['ask_price'] * size
            commission = self._calculate_commission(executed['ask_price'], size)
            
            self.positions[symbol] -= size
            self.cash += (revenue - commission)
            
            # Record trade
            trade = {
                'timestamp': market_data.timestamp,
                'symbol': symbol,
                'side': 'SELL',
                'price': executed['ask_price'],
                'size': size,
                'commission': commission
            }
            self.trade_history.append(trade)
            
            # Update models
            if self.as_model:
                self.as_model.update_state(False, True, 0, executed['ask_price'], 0, size)
            if self.inventory_manager:
                self.inventory_manager.execute_trade(
                    executed['ask_price'], size, False, market_data.timestamp
                )
            if self.risk_analyzer:
                self.risk_analyzer.add_trade(
                    executed['ask_price'], size, False, market_data.timestamp
                )
    
    def _calculate_pnl(self, current_price: float) -> float:
        """Calculate current P&L"""
        position_value = sum(
            self.positions[symbol] * current_price 
            for symbol in self.positions
        )
        return self.cash + position_value - self.config.initial_cash
    
    def _check_risk_limits(self) -> bool:
        """Check if risk limits are breached"""
        if self.config.max_daily_loss:
            current_pnl = self._calculate_pnl(0)  # Rough estimate
            if current_pnl < -self.config.max_daily_loss:
                return True
        
        if self.config.max_position_value:
            position_value = sum(abs(pos) for pos in self.positions.values())
            if position_value > self.config.max_position_value:
                return True
        
        return False
    
    def _calculate_final_metrics(self, results: Dict) -> BacktestResult:
        """Calculate final performance metrics"""
        # Convert to pandas for easier analysis
        df = pd.DataFrame(results)
        
        if df.empty:
            return BacktestResult(
                config=self.config,
                final_pnl=0,
                total_return=0,
                sharpe_ratio=0,
                max_drawdown=0,
                win_rate=0,
                total_trades=0
            )
        
        # Calculate metrics
        final_pnl = df['pnl'].iloc[-1] if not df['pnl'].empty else 0
        total_return = final_pnl / self.config.initial_cash
        
        # Get risk metrics
        risk_metrics = {}
        if self.risk_analyzer:
            metrics = self.risk_analyzer.get_risk_metrics()
            risk_metrics = {
                'sharpe_ratio': metrics.sharpe_ratio,
                'sortino_ratio': metrics.sortino_ratio,
                'max_drawdown': metrics.max_drawdown,
                'var_95': metrics.var_95,
                'cvar_95': metrics.cvar_95,
                'win_rate': metrics.win_rate,
                'profit_factor': metrics.profit_factor,
                'calmar_ratio': metrics.calmar_ratio,
                'omega_ratio': metrics.omega_ratio
            }
        
        return BacktestResult(
            config=self.config,
            final_pnl=final_pnl,
            total_return=total_return,
            sharpe_ratio=risk_metrics.get('sharpe_ratio', 0),
            max_drawdown=risk_metrics.get('max_drawdown', 0),
            win_rate=risk_metrics.get('win_rate', 0),
            total_trades=len(self.trade_history),
            pnl_series=pd.Series(df['pnl'].values, index=df['timestamps']) if 'timestamps' in df else pd.Series(),
            returns_series=pd.Series(df['returns'].values[1:], index=df['timestamps'][1:]) if 'returns' in df else pd.Series(),
            inventory_series=pd.Series(df['inventory'].values, index=df['timestamps']) if 'inventory' in df else pd.Series(),
            cash_series=pd.Series(df['cash'].values, index=df['timestamps']) if 'cash' in df else pd.Series(),
            risk_metrics=risk_metrics,
            trade_history=self.trade_history,
            quote_history=self.quote_history
        )
    
    def _save_results(self, result: BacktestResult):
        """Save backtest results to file"""
        import os
        import pickle
        
        # Create results directory
        os.makedirs(self.config.results_path, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"backtest_{timestamp}.pkl"
        filepath = os.path.join(self.config.results_path, filename)
        
        # Save results
        with open(filepath, 'wb') as f:
            pickle.dump(result, f)
        
        self._log(f"Results saved to {filepath}")
    
    def _log(self, message: str, level: str = 'INFO'):
        """Log a message"""
        if self.logger:
            log_func = getattr(self.logger, level.lower())
            log_func(message)