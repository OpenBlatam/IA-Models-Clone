"""
Advanced Financial and Trading Data Analysis System
Sistema avanzado de análisis de datos financieros y de trading
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import math
import warnings
warnings.filterwarnings('ignore')

# Financial data imports
try:
    import yfinance as yf
    import pandas_datareader as pdr
    import alpha_vantage
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

try:
    import ta
    import talib
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False

# Machine learning for financial data
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.svm import SVR
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Deep learning for financial data
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Statistical analysis
try:
    import scipy.stats as stats
    from scipy.optimize import minimize
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AssetType(Enum):
    """Tipos de activos financieros"""
    STOCK = "stock"
    BOND = "bond"
    COMMODITY = "commodity"
    CURRENCY = "currency"
    CRYPTOCURRENCY = "cryptocurrency"
    INDEX = "index"
    ETF = "etf"
    OPTION = "option"
    FUTURE = "future"

class TimeFrame(Enum):
    """Marcos temporales"""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAILY = "1d"
    WEEKLY = "1wk"
    MONTHLY = "1mo"

class AnalysisType(Enum):
    """Tipos de análisis financiero"""
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    QUANTITATIVE = "quantitative"
    SENTIMENT = "sentiment"
    RISK = "risk"
    PORTFOLIO = "portfolio"
    BACKTESTING = "backtesting"
    PREDICTION = "prediction"

class TradingStrategy(Enum):
    """Estrategias de trading"""
    BUY_AND_HOLD = "buy_and_hold"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    ARBITRAGE = "arbitrage"
    PAIRS_TRADING = "pairs_trading"
    GRID_TRADING = "grid_trading"
    SCALPING = "scalping"
    SWING_TRADING = "swing_trading"

@dataclass
class FinancialAsset:
    """Activo financiero"""
    id: str
    symbol: str
    name: str
    asset_type: AssetType
    exchange: str
    currency: str
    sector: Optional[str] = None
    market_cap: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class PriceData:
    """Datos de precios"""
    id: str
    asset_id: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    adjusted_close: Optional[float] = None
    timeframe: TimeFrame = TimeFrame.DAILY
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TechnicalIndicator:
    """Indicador técnico"""
    id: str
    asset_id: str
    indicator_name: str
    values: List[float]
    timestamps: List[datetime]
    parameters: Dict[str, Any]
    signal: Optional[str] = None
    strength: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class TradingSignal:
    """Señal de trading"""
    id: str
    asset_id: str
    signal_type: str  # BUY, SELL, HOLD
    strength: float  # 0-1
    price: float
    timestamp: datetime
    strategy: TradingStrategy
    confidence: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Portfolio:
    """Portafolio de inversión"""
    id: str
    name: str
    assets: Dict[str, float]  # asset_id -> weight
    total_value: float
    cash: float
    risk_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class FinancialAnalysis:
    """Análisis financiero"""
    id: str
    asset_id: str
    analysis_type: AnalysisType
    results: Dict[str, Any]
    predictions: Dict[str, Any]
    risk_metrics: Dict[str, float]
    recommendations: List[str]
    created_at: datetime = field(default_factory=datetime.now)

class AdvancedFinancialAnalyzer:
    """
    Analizador avanzado de datos financieros y trading
    """
    
    def __init__(
        self,
        enable_yfinance: bool = True,
        enable_ta: bool = True,
        enable_ccxt: bool = True,
        alpha_vantage_key: Optional[str] = None,
        data_directory: str = "data/financial/"
    ):
        self.enable_yfinance = enable_yfinance and YFINANCE_AVAILABLE
        self.enable_ta = enable_ta and TA_AVAILABLE
        self.enable_ccxt = enable_ccxt and CCXT_AVAILABLE
        self.alpha_vantage_key = alpha_vantage_key
        self.data_directory = data_directory
        
        # Almacenamiento
        self.financial_assets: Dict[str, FinancialAsset] = {}
        self.price_data: Dict[str, List[PriceData]] = {}
        self.technical_indicators: Dict[str, List[TechnicalIndicator]] = {}
        self.trading_signals: Dict[str, List[TradingSignal]] = {}
        self.portfolios: Dict[str, Portfolio] = {}
        self.financial_analyses: Dict[str, FinancialAnalysis] = {}
        
        # Modelos de ML
        self.price_prediction_models: Dict[str, Any] = {}
        self.signal_generation_models: Dict[str, Any] = {}
        
        # Configuración
        self.config = {
            "default_timeframe": TimeFrame.DAILY,
            "lookback_period": 252,  # 1 año de trading
            "risk_free_rate": 0.02,  # 2% anual
            "volatility_window": 20,
            "correlation_threshold": 0.7,
            "min_volume": 1000,
            "max_drawdown_threshold": 0.2
        }
        
        # Crear directorio de datos
        import os
        os.makedirs(self.data_directory, exist_ok=True)
        
        # Inicializar exchanges
        self._initialize_exchanges()
        
        logger.info("Advanced Financial Analyzer inicializado")
    
    def _initialize_exchanges(self):
        """Inicializar exchanges de criptomonedas"""
        try:
            if self.enable_ccxt:
                self.exchanges = {
                    'binance': ccxt.binance(),
                    'coinbase': ccxt.coinbasepro(),
                    'kraken': ccxt.kraken(),
                    'bitfinex': ccxt.bitfinex()
                }
                logger.info("Exchanges de criptomonedas inicializados")
            
        except Exception as e:
            logger.error(f"Error inicializando exchanges: {e}")
    
    async def add_financial_asset(
        self,
        symbol: str,
        name: str,
        asset_type: AssetType,
        exchange: str = "NASDAQ",
        currency: str = "USD",
        sector: Optional[str] = None,
        market_cap: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> FinancialAsset:
        """
        Agregar activo financiero
        
        Args:
            symbol: Símbolo del activo
            name: Nombre del activo
            asset_type: Tipo de activo
            exchange: Exchange
            currency: Moneda
            sector: Sector
            market_cap: Capitalización de mercado
            metadata: Metadatos adicionales
            
        Returns:
            Activo financiero
        """
        try:
            asset_id = f"{symbol}_{exchange}_{asset_type.value}"
            
            financial_asset = FinancialAsset(
                id=asset_id,
                symbol=symbol,
                name=name,
                asset_type=asset_type,
                exchange=exchange,
                currency=currency,
                sector=sector,
                market_cap=market_cap,
                metadata=metadata or {}
            )
            
            # Almacenar activo
            self.financial_assets[asset_id] = financial_asset
            
            logger.info(f"Activo financiero agregado: {asset_id}")
            return financial_asset
            
        except Exception as e:
            logger.error(f"Error agregando activo financiero: {e}")
            raise
    
    async def fetch_price_data(
        self,
        asset_id: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: TimeFrame = None
    ) -> List[PriceData]:
        """
        Obtener datos de precios
        
        Args:
            asset_id: ID del activo
            start_date: Fecha de inicio
            end_date: Fecha de fin
            timeframe: Marco temporal
            
        Returns:
            Lista de datos de precios
        """
        try:
            if asset_id not in self.financial_assets:
                raise ValueError(f"Activo {asset_id} no encontrado")
            
            asset = self.financial_assets[asset_id]
            timeframe = timeframe or self.config["default_timeframe"]
            
            logger.info(f"Obteniendo datos de precios para {asset.symbol} ({timeframe.value})")
            
            price_data_list = []
            
            if self.enable_yfinance and asset.asset_type in [AssetType.STOCK, AssetType.ETF, AssetType.INDEX]:
                # Usar yfinance para acciones y ETFs
                ticker = yf.Ticker(asset.symbol)
                data = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=timeframe.value
                )
                
                for timestamp, row in data.iterrows():
                    price_data = PriceData(
                        id=f"{asset_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                        asset_id=asset_id,
                        timestamp=timestamp,
                        open=float(row['Open']),
                        high=float(row['High']),
                        low=float(row['Low']),
                        close=float(row['Close']),
                        volume=float(row['Volume']),
                        adjusted_close=float(row['Close']) if 'Adj Close' not in row else float(row['Adj Close']),
                        timeframe=timeframe
                    )
                    price_data_list.append(price_data)
            
            elif self.enable_ccxt and asset.asset_type == AssetType.CRYPTOCURRENCY:
                # Usar CCXT para criptomonedas
                exchange = self.exchanges.get('binance')
                if exchange:
                    symbol = f"{asset.symbol}/USDT"
                    ohlcv = exchange.fetch_ohlcv(symbol, timeframe.value, since=int(start_date.timestamp() * 1000))
                    
                    for candle in ohlcv:
                        timestamp = datetime.fromtimestamp(candle[0] / 1000)
                        if start_date <= timestamp <= end_date:
                            price_data = PriceData(
                                id=f"{asset_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}",
                                asset_id=asset_id,
                                timestamp=timestamp,
                                open=float(candle[1]),
                                high=float(candle[2]),
                                low=float(candle[3]),
                                close=float(candle[4]),
                                volume=float(candle[5]),
                                timeframe=timeframe
                            )
                            price_data_list.append(price_data)
            
            # Almacenar datos de precios
            if asset_id not in self.price_data:
                self.price_data[asset_id] = []
            self.price_data[asset_id].extend(price_data_list)
            
            logger.info(f"Obtenidos {len(price_data_list)} registros de precios para {asset.symbol}")
            return price_data_list
            
        except Exception as e:
            logger.error(f"Error obteniendo datos de precios: {e}")
            raise
    
    async def calculate_technical_indicators(
        self,
        asset_id: str,
        indicators: List[str] = None
    ) -> List[TechnicalIndicator]:
        """
        Calcular indicadores técnicos
        
        Args:
            asset_id: ID del activo
            indicators: Lista de indicadores a calcular
            
        Returns:
            Lista de indicadores técnicos
        """
        try:
            if asset_id not in self.price_data:
                raise ValueError(f"No hay datos de precios para {asset_id}")
            
            if not indicators:
                indicators = [
                    "SMA_20", "SMA_50", "EMA_12", "EMA_26",
                    "RSI_14", "MACD", "BB_20", "STOCH_14",
                    "ATR_14", "ADX_14", "CCI_20", "WILLIAMS_R_14"
                ]
            
            price_data = self.price_data[asset_id]
            if len(price_data) < 50:
                raise ValueError("Insuficientes datos para calcular indicadores técnicos")
            
            # Convertir a DataFrame
            df = pd.DataFrame([{
                'timestamp': pd.price_data.timestamp,
                'open': pd.price_data.open,
                'high': pd.price_data.high,
                'low': pd.price_data.low,
                'close': pd.price_data.close,
                'volume': pd.price_data.volume
            } for pd.price_data in price_data])
            
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            technical_indicators = []
            
            for indicator in indicators:
                try:
                    indicator_data = await self._calculate_single_indicator(df, indicator)
                    if indicator_data:
                        technical_indicators.append(indicator_data)
                except Exception as e:
                    logger.warning(f"Error calculando indicador {indicator}: {e}")
            
            # Almacenar indicadores
            if asset_id not in self.technical_indicators:
                self.technical_indicators[asset_id] = []
            self.technical_indicators[asset_id].extend(technical_indicators)
            
            logger.info(f"Calculados {len(technical_indicators)} indicadores técnicos para {asset_id}")
            return technical_indicators
            
        except Exception as e:
            logger.error(f"Error calculando indicadores técnicos: {e}")
            raise
    
    async def _calculate_single_indicator(self, df: pd.DataFrame, indicator: str) -> Optional[TechnicalIndicator]:
        """Calcular un indicador técnico específico"""
        try:
            if not self.enable_ta:
                return None
            
            if indicator.startswith("SMA_"):
                period = int(indicator.split("_")[1])
                values = ta.trend.sma_indicator(df['close'], window=period)
                signal = "BUY" if values.iloc[-1] > df['close'].iloc[-1] else "SELL"
                
            elif indicator.startswith("EMA_"):
                period = int(indicator.split("_")[1])
                values = ta.trend.ema_indicator(df['close'], window=period)
                signal = "BUY" if values.iloc[-1] > df['close'].iloc[-1] else "SELL"
                
            elif indicator.startswith("RSI_"):
                period = int(indicator.split("_")[1])
                values = ta.momentum.rsi(df['close'], window=period)
                rsi_value = values.iloc[-1]
                if rsi_value > 70:
                    signal = "SELL"
                elif rsi_value < 30:
                    signal = "BUY"
                else:
                    signal = "HOLD"
                    
            elif indicator == "MACD":
                macd_line = ta.trend.macd(df['close'])
                signal_line = ta.trend.macd_signal(df['close'])
                histogram = ta.trend.macd_diff(df['close'])
                
                values = macd_line
                if macd_line.iloc[-1] > signal_line.iloc[-1]:
                    signal = "BUY"
                else:
                    signal = "SELL"
                    
            elif indicator.startswith("BB_"):
                period = int(indicator.split("_")[1])
                bb_high = ta.volatility.bollinger_hband(df['close'], window=period)
                bb_low = ta.volatility.bollinger_lband(df['close'], window=period)
                bb_mid = ta.volatility.bollinger_mavg(df['close'], window=period)
                
                values = (df['close'] - bb_low) / (bb_high - bb_low)
                bb_value = values.iloc[-1]
                if bb_value > 0.8:
                    signal = "SELL"
                elif bb_value < 0.2:
                    signal = "BUY"
                else:
                    signal = "HOLD"
                    
            elif indicator.startswith("STOCH_"):
                period = int(indicator.split("_")[1])
                stoch_k = ta.momentum.stoch(df['high'], df['low'], df['close'], window=period)
                stoch_d = ta.momentum.stoch_signal(df['high'], df['low'], df['close'], window=period)
                
                values = stoch_k
                k_value = stoch_k.iloc[-1]
                d_value = stoch_d.iloc[-1]
                if k_value > 80 and d_value > 80:
                    signal = "SELL"
                elif k_value < 20 and d_value < 20:
                    signal = "BUY"
                else:
                    signal = "HOLD"
                    
            else:
                return None
            
            # Calcular fuerza de la señal
            strength = abs(values.iloc[-1] - values.iloc[-2]) / values.iloc[-2] if values.iloc[-2] != 0 else 0
            
            technical_indicator = TechnicalIndicator(
                id=f"ti_{indicator}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                asset_id=df.index[0],  # Usar el primer timestamp como asset_id temporal
                indicator_name=indicator,
                values=values.dropna().tolist(),
                timestamps=df.index[values.notna()].tolist(),
                parameters={"period": period if "period" in locals() else None},
                signal=signal,
                strength=float(strength)
            )
            
            return technical_indicator
            
        except Exception as e:
            logger.error(f"Error calculando indicador {indicator}: {e}")
            return None
    
    async def generate_trading_signals(
        self,
        asset_id: str,
        strategy: TradingStrategy = TradingStrategy.MOMENTUM
    ) -> List[TradingSignal]:
        """
        Generar señales de trading
        
        Args:
            asset_id: ID del activo
            strategy: Estrategia de trading
            
        Returns:
            Lista de señales de trading
        """
        try:
            if asset_id not in self.technical_indicators:
                raise ValueError(f"No hay indicadores técnicos para {asset_id}")
            
            indicators = self.technical_indicators[asset_id]
            if not indicators:
                raise ValueError(f"No hay indicadores disponibles para {asset_id}")
            
            logger.info(f"Generando señales de trading para {asset_id} con estrategia {strategy.value}")
            
            signals = []
            
            if strategy == TradingStrategy.MOMENTUM:
                signals = await self._generate_momentum_signals(asset_id, indicators)
            elif strategy == TradingStrategy.MEAN_REVERSION:
                signals = await self._generate_mean_reversion_signals(asset_id, indicators)
            elif strategy == TradingStrategy.BUY_AND_HOLD:
                signals = await self._generate_buy_and_hold_signals(asset_id)
            else:
                signals = await self._generate_momentum_signals(asset_id, indicators)
            
            # Almacenar señales
            if asset_id not in self.trading_signals:
                self.trading_signals[asset_id] = []
            self.trading_signals[asset_id].extend(signals)
            
            logger.info(f"Generadas {len(signals)} señales de trading para {asset_id}")
            return signals
            
        except Exception as e:
            logger.error(f"Error generando señales de trading: {e}")
            raise
    
    async def _generate_momentum_signals(self, asset_id: str, indicators: List[TechnicalIndicator]) -> List[TradingSignal]:
        """Generar señales de momentum"""
        try:
            signals = []
            
            # Combinar múltiples indicadores para señales de momentum
            rsi_indicator = next((ind for ind in indicators if ind.indicator_name.startswith("RSI")), None)
            macd_indicator = next((ind for ind in indicators if ind.indicator_name == "MACD"), None)
            sma_20 = next((ind for ind in indicators if ind.indicator_name == "SMA_20"), None)
            
            if not all([rsi_indicator, macd_indicator, sma_20]):
                return signals
            
            # Obtener precio actual
            if asset_id in self.price_data and self.price_data[asset_id]:
                current_price = self.price_data[asset_id][-1].close
                current_timestamp = self.price_data[asset_id][-1].timestamp
            else:
                return signals
            
            # Lógica de momentum
            rsi_value = rsi_indicator.values[-1] if rsi_indicator.values else 50
            macd_value = macd_indicator.values[-1] if macd_indicator.values else 0
            sma_value = sma_20.values[-1] if sma_20.values else current_price
            
            # Señal de compra: RSI < 70, MACD positivo, precio > SMA
            if (rsi_value < 70 and 
                macd_value > 0 and 
                current_price > sma_value):
                
                signal = TradingSignal(
                    id=f"signal_{asset_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    asset_id=asset_id,
                    signal_type="BUY",
                    strength=min(1.0, (70 - rsi_value) / 40 + macd_value / 10),
                    price=current_price,
                    timestamp=current_timestamp,
                    strategy=TradingStrategy.MOMENTUM,
                    confidence=0.7,
                    stop_loss=current_price * 0.95,
                    take_profit=current_price * 1.15
                )
                signals.append(signal)
            
            # Señal de venta: RSI > 70, MACD negativo, precio < SMA
            elif (rsi_value > 70 and 
                  macd_value < 0 and 
                  current_price < sma_value):
                
                signal = TradingSignal(
                    id=f"signal_{asset_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    asset_id=asset_id,
                    signal_type="SELL",
                    strength=min(1.0, (rsi_value - 70) / 30 + abs(macd_value) / 10),
                    price=current_price,
                    timestamp=current_timestamp,
                    strategy=TradingStrategy.MOMENTUM,
                    confidence=0.7,
                    stop_loss=current_price * 1.05,
                    take_profit=current_price * 0.85
                )
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generando señales de momentum: {e}")
            return []
    
    async def _generate_mean_reversion_signals(self, asset_id: str, indicators: List[TechnicalIndicator]) -> List[TradingSignal]:
        """Generar señales de reversión a la media"""
        try:
            signals = []
            
            # Usar Bollinger Bands para reversión a la media
            bb_indicator = next((ind for ind in indicators if ind.indicator_name.startswith("BB")), None)
            rsi_indicator = next((ind for ind in indicators if ind.indicator_name.startswith("RSI")), None)
            
            if not all([bb_indicator, rsi_indicator]):
                return signals
            
            # Obtener precio actual
            if asset_id in self.price_data and self.price_data[asset_id]:
                current_price = self.price_data[asset_id][-1].close
                current_timestamp = self.price_data[asset_id][-1].timestamp
            else:
                return signals
            
            bb_value = bb_indicator.values[-1] if bb_indicator.values else 0.5
            rsi_value = rsi_indicator.values[-1] if rsi_indicator.values else 50
            
            # Señal de compra: precio en banda inferior, RSI oversold
            if bb_value < 0.2 and rsi_value < 30:
                signal = TradingSignal(
                    id=f"signal_{asset_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    asset_id=asset_id,
                    signal_type="BUY",
                    strength=min(1.0, (0.2 - bb_value) * 5 + (30 - rsi_value) / 30),
                    price=current_price,
                    timestamp=current_timestamp,
                    strategy=TradingStrategy.MEAN_REVERSION,
                    confidence=0.8,
                    stop_loss=current_price * 0.92,
                    take_profit=current_price * 1.08
                )
                signals.append(signal)
            
            # Señal de venta: precio en banda superior, RSI overbought
            elif bb_value > 0.8 and rsi_value > 70:
                signal = TradingSignal(
                    id=f"signal_{asset_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    asset_id=asset_id,
                    signal_type="SELL",
                    strength=min(1.0, (bb_value - 0.8) * 5 + (rsi_value - 70) / 30),
                    price=current_price,
                    timestamp=current_timestamp,
                    strategy=TradingStrategy.MEAN_REVERSION,
                    confidence=0.8,
                    stop_loss=current_price * 1.08,
                    take_profit=current_price * 0.92
                )
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generando señales de reversión a la media: {e}")
            return []
    
    async def _generate_buy_and_hold_signals(self, asset_id: str) -> List[TradingSignal]:
        """Generar señales de buy and hold"""
        try:
            signals = []
            
            # Obtener precio actual
            if asset_id in self.price_data and self.price_data[asset_id]:
                current_price = self.price_data[asset_id][-1].close
                current_timestamp = self.price_data[asset_id][-1].timestamp
            else:
                return signals
            
            # Señal de compra inicial
            signal = TradingSignal(
                id=f"signal_{asset_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                asset_id=asset_id,
                signal_type="BUY",
                strength=1.0,
                price=current_price,
                timestamp=current_timestamp,
                strategy=TradingStrategy.BUY_AND_HOLD,
                confidence=0.9
            )
            signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generando señales de buy and hold: {e}")
            return []
    
    async def create_portfolio(
        self,
        name: str,
        assets: Dict[str, float],
        initial_cash: float = 100000
    ) -> Portfolio:
        """
        Crear portafolio de inversión
        
        Args:
            name: Nombre del portafolio
            assets: Diccionario de activos y pesos
            initial_cash: Dinero inicial
            
        Returns:
            Portafolio
        """
        try:
            # Validar activos
            for asset_id in assets.keys():
                if asset_id not in self.financial_assets:
                    raise ValueError(f"Activo {asset_id} no encontrado")
            
            # Normalizar pesos
            total_weight = sum(assets.values())
            if total_weight > 1.0:
                assets = {k: v / total_weight for k, v in assets.items()}
            
            portfolio_id = f"portfolio_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Calcular métricas de riesgo iniciales
            risk_metrics = await self._calculate_portfolio_risk_metrics(assets)
            
            # Calcular métricas de rendimiento iniciales
            performance_metrics = await self._calculate_portfolio_performance_metrics(assets)
            
            portfolio = Portfolio(
                id=portfolio_id,
                name=name,
                assets=assets,
                total_value=initial_cash,
                cash=initial_cash * (1 - sum(assets.values())),
                risk_metrics=risk_metrics,
                performance_metrics=performance_metrics
            )
            
            # Almacenar portafolio
            self.portfolios[portfolio_id] = portfolio
            
            logger.info(f"Portafolio creado: {portfolio_id}")
            return portfolio
            
        except Exception as e:
            logger.error(f"Error creando portafolio: {e}")
            raise
    
    async def _calculate_portfolio_risk_metrics(self, assets: Dict[str, float]) -> Dict[str, float]:
        """Calcular métricas de riesgo del portafolio"""
        try:
            risk_metrics = {
                "portfolio_variance": 0.0,
                "portfolio_volatility": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "var_95": 0.0,
                "beta": 1.0
            }
            
            if not assets:
                return risk_metrics
            
            # Obtener datos de precios para todos los activos
            returns_data = {}
            for asset_id in assets.keys():
                if asset_id in self.price_data and len(self.price_data[asset_id]) > 1:
                    prices = [pd.close for pd in self.price_data[asset_id]]
                    returns = np.diff(prices) / prices[:-1]
                    returns_data[asset_id] = returns
            
            if not returns_data:
                return risk_metrics
            
            # Calcular matriz de covarianza
            min_length = min(len(returns) for returns in returns_data.values())
            aligned_returns = {asset_id: returns[-min_length:] for asset_id, returns in returns_data.items()}
            
            if min_length < 10:
                return risk_metrics
            
            # Crear matriz de retornos
            returns_matrix = np.array([aligned_returns[asset_id] for asset_id in assets.keys()])
            weights = np.array([assets[asset_id] for asset_id in assets.keys()])
            
            # Calcular varianza del portafolio
            cov_matrix = np.cov(returns_matrix)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # Calcular Sharpe ratio
            portfolio_return = np.mean(np.dot(weights, returns_matrix))
            sharpe_ratio = (portfolio_return - self.config["risk_free_rate"] / 252) / portfolio_volatility if portfolio_volatility > 0 else 0
            
            # Calcular VaR 95%
            portfolio_returns = np.dot(weights, returns_matrix)
            var_95 = np.percentile(portfolio_returns, 5)
            
            risk_metrics.update({
                "portfolio_variance": float(portfolio_variance),
                "portfolio_volatility": float(portfolio_volatility),
                "sharpe_ratio": float(sharpe_ratio),
                "var_95": float(var_95)
            })
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error calculando métricas de riesgo: {e}")
            return {}
    
    async def _calculate_portfolio_performance_metrics(self, assets: Dict[str, float]) -> Dict[str, float]:
        """Calcular métricas de rendimiento del portafolio"""
        try:
            performance_metrics = {
                "total_return": 0.0,
                "annualized_return": 0.0,
                "volatility": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0
            }
            
            if not assets:
                return performance_metrics
            
            # Obtener datos de precios
            portfolio_values = []
            for asset_id in assets.keys():
                if asset_id in self.price_data and len(self.price_data[asset_id]) > 1:
                    prices = [pd.close for pd in self.price_data[asset_id]]
                    weight = assets[asset_id]
                    asset_values = [price * weight for price in prices]
                    
                    if not portfolio_values:
                        portfolio_values = asset_values
                    else:
                        portfolio_values = [sum(x) for x in zip(portfolio_values, asset_values)]
            
            if len(portfolio_values) < 2:
                return performance_metrics
            
            # Calcular retornos
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            
            # Métricas básicas
            total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
            annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
            volatility = np.std(returns) * np.sqrt(252)
            
            # Sharpe ratio
            sharpe_ratio = (annualized_return - self.config["risk_free_rate"]) / volatility if volatility > 0 else 0
            
            # Maximum drawdown
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = np.min(drawdown)
            
            # Win rate
            win_rate = np.mean(returns > 0)
            
            performance_metrics.update({
                "total_return": float(total_return),
                "annualized_return": float(annualized_return),
                "volatility": float(volatility),
                "sharpe_ratio": float(sharpe_ratio),
                "max_drawdown": float(max_drawdown),
                "win_rate": float(win_rate)
            })
            
            return performance_metrics
            
        except Exception as e:
            logger.error(f"Error calculando métricas de rendimiento: {e}")
            return {}
    
    async def analyze_financial_data(
        self,
        asset_id: str,
        analysis_type: AnalysisType = AnalysisType.TECHNICAL
    ) -> FinancialAnalysis:
        """
        Analizar datos financieros
        
        Args:
            asset_id: ID del activo
            analysis_type: Tipo de análisis
            
        Returns:
            Análisis financiero
        """
        try:
            if asset_id not in self.financial_assets:
                raise ValueError(f"Activo {asset_id} no encontrado")
            
            logger.info(f"Analizando datos financieros para {asset_id} ({analysis_type.value})")
            
            # Realizar análisis según el tipo
            if analysis_type == AnalysisType.TECHNICAL:
                results = await self._perform_technical_analysis(asset_id)
            elif analysis_type == AnalysisType.RISK:
                results = await self._perform_risk_analysis(asset_id)
            elif analysis_type == AnalysisType.PREDICTION:
                results = await self._perform_prediction_analysis(asset_id)
            else:
                results = await self._perform_technical_analysis(asset_id)
            
            # Calcular métricas de riesgo
            risk_metrics = await self._calculate_asset_risk_metrics(asset_id)
            
            # Generar predicciones
            predictions = await self._generate_price_predictions(asset_id)
            
            # Generar recomendaciones
            recommendations = await self._generate_recommendations(asset_id, results, risk_metrics)
            
            # Crear análisis
            analysis = FinancialAnalysis(
                id=f"fa_{asset_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                asset_id=asset_id,
                analysis_type=analysis_type,
                results=results,
                predictions=predictions,
                risk_metrics=risk_metrics,
                recommendations=recommendations
            )
            
            # Almacenar análisis
            self.financial_analyses[analysis.id] = analysis
            
            logger.info(f"Análisis financiero completado: {analysis.id}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analizando datos financieros: {e}")
            raise
    
    async def _perform_technical_analysis(self, asset_id: str) -> Dict[str, Any]:
        """Realizar análisis técnico"""
        try:
            results = {
                "trend_analysis": {},
                "momentum_analysis": {},
                "volatility_analysis": {},
                "volume_analysis": {},
                "support_resistance": {}
            }
            
            if asset_id not in self.price_data or not self.price_data[asset_id]:
                return results
            
            price_data = self.price_data[asset_id]
            prices = [pd.close for pd in price_data]
            volumes = [pd.volume for pd in price_data]
            
            if len(prices) < 20:
                return results
            
            # Análisis de tendencia
            sma_20 = np.mean(prices[-20:])
            sma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else sma_20
            
            if sma_20 > sma_50:
                trend = "UPTREND"
            elif sma_20 < sma_50:
                trend = "DOWNTREND"
            else:
                trend = "SIDEWAYS"
            
            results["trend_analysis"] = {
                "trend": trend,
                "sma_20": sma_20,
                "sma_50": sma_50,
                "trend_strength": abs(sma_20 - sma_50) / sma_50
            }
            
            # Análisis de momentum
            if asset_id in self.technical_indicators:
                rsi_indicator = next((ind for ind in self.technical_indicators[asset_id] if ind.indicator_name.startswith("RSI")), None)
                if rsi_indicator and rsi_indicator.values:
                    rsi_value = rsi_indicator.values[-1]
                    if rsi_value > 70:
                        momentum = "OVERBOUGHT"
                    elif rsi_value < 30:
                        momentum = "OVERSOLD"
                    else:
                        momentum = "NEUTRAL"
                    
                    results["momentum_analysis"] = {
                        "momentum": momentum,
                        "rsi": rsi_value,
                        "strength": abs(rsi_value - 50) / 50
                    }
            
            # Análisis de volatilidad
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns) * np.sqrt(252)  # Anualizada
            
            results["volatility_analysis"] = {
                "volatility": volatility,
                "volatility_percentile": np.percentile(returns, 95) - np.percentile(returns, 5)
            }
            
            # Análisis de volumen
            avg_volume = np.mean(volumes[-20:])
            current_volume = volumes[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            results["volume_analysis"] = {
                "volume_ratio": volume_ratio,
                "volume_trend": "HIGH" if volume_ratio > 1.5 else "LOW" if volume_ratio < 0.5 else "NORMAL"
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error en análisis técnico: {e}")
            return {}
    
    async def _perform_risk_analysis(self, asset_id: str) -> Dict[str, Any]:
        """Realizar análisis de riesgo"""
        try:
            results = {
                "var_analysis": {},
                "stress_testing": {},
                "correlation_analysis": {},
                "liquidity_analysis": {}
            }
            
            if asset_id not in self.price_data or not self.price_data[asset_id]:
                return results
            
            price_data = self.price_data[asset_id]
            prices = [pd.close for pd in price_data]
            volumes = [pd.volume for pd in price_data]
            
            if len(prices) < 20:
                return results
            
            # Value at Risk
            returns = np.diff(prices) / prices[:-1]
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            
            results["var_analysis"] = {
                "var_95": var_95,
                "var_99": var_99,
                "expected_shortfall": np.mean(returns[returns <= var_95])
            }
            
            # Stress testing
            max_drawdown = 0
            peak = prices[0]
            for price in prices:
                if price > peak:
                    peak = price
                drawdown = (peak - price) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            results["stress_testing"] = {
                "max_drawdown": max_drawdown,
                "recovery_time": len(prices)  # Simplificado
            }
            
            # Análisis de liquidez
            avg_volume = np.mean(volumes)
            volume_volatility = np.std(volumes) / avg_volume if avg_volume > 0 else 0
            
            results["liquidity_analysis"] = {
                "avg_volume": avg_volume,
                "volume_volatility": volume_volatility,
                "liquidity_score": min(1.0, avg_volume / 1000000)  # Normalizado
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error en análisis de riesgo: {e}")
            return {}
    
    async def _perform_prediction_analysis(self, asset_id: str) -> Dict[str, Any]:
        """Realizar análisis predictivo"""
        try:
            results = {
                "price_forecast": {},
                "trend_prediction": {},
                "volatility_forecast": {}
            }
            
            if asset_id not in self.price_data or not self.price_data[asset_id]:
                return results
            
            price_data = self.price_data[asset_id]
            prices = [pd.close for pd in price_data]
            
            if len(prices) < 50:
                return results
            
            # Predicción simple usando media móvil
            sma_20 = np.mean(prices[-20:])
            sma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else sma_20
            
            # Predicción de tendencia
            if sma_20 > sma_50:
                trend_prediction = "UPWARD"
                price_forecast = prices[-1] * 1.05  # 5% aumento
            else:
                trend_prediction = "DOWNWARD"
                price_forecast = prices[-1] * 0.95  # 5% disminución
            
            results["price_forecast"] = {
                "next_price": price_forecast,
                "confidence": 0.6
            }
            
            results["trend_prediction"] = {
                "trend": trend_prediction,
                "strength": abs(sma_20 - sma_50) / sma_50
            }
            
            # Predicción de volatilidad
            returns = np.diff(prices) / prices[:-1]
            current_volatility = np.std(returns[-20:]) * np.sqrt(252)
            
            results["volatility_forecast"] = {
                "predicted_volatility": current_volatility,
                "volatility_trend": "STABLE"
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error en análisis predictivo: {e}")
            return {}
    
    async def _calculate_asset_risk_metrics(self, asset_id: str) -> Dict[str, float]:
        """Calcular métricas de riesgo del activo"""
        try:
            risk_metrics = {
                "volatility": 0.0,
                "beta": 1.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "var_95": 0.0
            }
            
            if asset_id not in self.price_data or not self.price_data[asset_id]:
                return risk_metrics
            
            price_data = self.price_data[asset_id]
            prices = [pd.close for pd in price_data]
            
            if len(prices) < 20:
                return risk_metrics
            
            # Calcular retornos
            returns = np.diff(prices) / prices[:-1]
            
            # Volatilidad
            volatility = np.std(returns) * np.sqrt(252)
            
            # Sharpe ratio
            mean_return = np.mean(returns) * 252
            sharpe_ratio = (mean_return - self.config["risk_free_rate"]) / volatility if volatility > 0 else 0
            
            # Maximum drawdown
            cumulative = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = np.min(drawdown)
            
            # VaR 95%
            var_95 = np.percentile(returns, 5)
            
            risk_metrics.update({
                "volatility": float(volatility),
                "sharpe_ratio": float(sharpe_ratio),
                "max_drawdown": float(max_drawdown),
                "var_95": float(var_95)
            })
            
            return risk_metrics
            
        except Exception as e:
            logger.error(f"Error calculando métricas de riesgo: {e}")
            return {}
    
    async def _generate_price_predictions(self, asset_id: str) -> Dict[str, Any]:
        """Generar predicciones de precios"""
        try:
            predictions = {
                "short_term": {},
                "medium_term": {},
                "long_term": {}
            }
            
            if asset_id not in self.price_data or not self.price_data[asset_id]:
                return predictions
            
            price_data = self.price_data[asset_id]
            current_price = price_data[-1].close
            
            # Predicciones simples basadas en tendencia
            if len(price_data) >= 20:
                sma_20 = np.mean([pd.close for pd in price_data[-20:]])
                trend = (current_price - sma_20) / sma_20
                
                # Predicciones a diferentes plazos
                predictions["short_term"] = {
                    "price": current_price * (1 + trend * 0.1),
                    "confidence": 0.6,
                    "timeframe": "1 week"
                }
                
                predictions["medium_term"] = {
                    "price": current_price * (1 + trend * 0.3),
                    "confidence": 0.4,
                    "timeframe": "1 month"
                }
                
                predictions["long_term"] = {
                    "price": current_price * (1 + trend * 0.5),
                    "confidence": 0.2,
                    "timeframe": "3 months"
                }
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error generando predicciones: {e}")
            return {}
    
    async def _generate_recommendations(self, asset_id: str, results: Dict[str, Any], risk_metrics: Dict[str, float]) -> List[str]:
        """Generar recomendaciones"""
        try:
            recommendations = []
            
            # Recomendación basada en tendencia
            trend_analysis = results.get("trend_analysis", {})
            trend = trend_analysis.get("trend", "UNKNOWN")
            
            if trend == "UPTREND":
                recommendations.append("Considerar posición larga debido a tendencia alcista")
            elif trend == "DOWNTREND":
                recommendations.append("Considerar posición corta o evitar entrada debido a tendencia bajista")
            
            # Recomendación basada en momentum
            momentum_analysis = results.get("momentum_analysis", {})
            momentum = momentum_analysis.get("momentum", "NEUTRAL")
            
            if momentum == "OVERSOLD":
                recommendations.append("Posible oportunidad de compra - activo en zona de sobreventa")
            elif momentum == "OVERBOUGHT":
                recommendations.append("Posible oportunidad de venta - activo en zona de sobrecompra")
            
            # Recomendación basada en riesgo
            volatility = risk_metrics.get("volatility", 0)
            if volatility > 0.3:
                recommendations.append("Alta volatilidad - considerar reducir tamaño de posición")
            elif volatility < 0.1:
                recommendations.append("Baja volatilidad - activo relativamente estable")
            
            # Recomendación basada en Sharpe ratio
            sharpe_ratio = risk_metrics.get("sharpe_ratio", 0)
            if sharpe_ratio > 1.0:
                recommendations.append("Excelente ratio riesgo-retorno")
            elif sharpe_ratio < 0:
                recommendations.append("Rendimiento negativo ajustado por riesgo")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generando recomendaciones: {e}")
            return []
    
    async def get_financial_summary(self) -> Dict[str, Any]:
        """Obtener resumen del sistema financiero"""
        try:
            return {
                "total_assets": len(self.financial_assets),
                "total_price_records": sum(len(data) for data in self.price_data.values()),
                "total_indicators": sum(len(indicators) for indicators in self.technical_indicators.values()),
                "total_signals": sum(len(signals) for signals in self.trading_signals.values()),
                "total_portfolios": len(self.portfolios),
                "total_analyses": len(self.financial_analyses),
                "asset_types": {
                    asset_type.value: len([a for a in self.financial_assets.values() if a.asset_type == asset_type])
                    for asset_type in AssetType
                },
                "analysis_types": {
                    analysis_type.value: len([a for a in self.financial_analyses.values() if a.analysis_type == analysis_type])
                    for analysis_type in AnalysisType
                },
                "capabilities": {
                    "yfinance": self.enable_yfinance,
                    "ta": self.enable_ta,
                    "ccxt": self.enable_ccxt,
                    "alpha_vantage": self.alpha_vantage_key is not None
                },
                "last_activity": max([
                    max([a.created_at for a in self.financial_assets.values()]) if self.financial_assets else datetime.min,
                    max([a.created_at for a in self.financial_analyses.values()]) if self.financial_analyses else datetime.min
                ]).isoformat() if any([self.financial_assets, self.financial_analyses]) else None
            }
        except Exception as e:
            logger.error(f"Error obteniendo resumen financiero: {e}")
            return {}
    
    async def export_financial_data(self, filepath: str = None) -> str:
        """Exportar datos financieros"""
        try:
            if filepath is None:
                filepath = f"exports/financial_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            import os
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            export_data = {
                "financial_assets": {
                    asset_id: {
                        "symbol": asset.symbol,
                        "name": asset.name,
                        "asset_type": asset.asset_type.value,
                        "exchange": asset.exchange,
                        "currency": asset.currency,
                        "sector": asset.sector,
                        "market_cap": asset.market_cap,
                        "metadata": asset.metadata,
                        "created_at": asset.created_at.isoformat()
                    }
                    for asset_id, asset in self.financial_assets.items()
                },
                "price_data": {
                    asset_id: [
                        {
                            "timestamp": pd.timestamp.isoformat(),
                            "open": pd.open,
                            "high": pd.high,
                            "low": pd.low,
                            "close": pd.close,
                            "volume": pd.volume,
                            "adjusted_close": pd.adjusted_close,
                            "timeframe": pd.timeframe.value
                        }
                        for pd in price_list
                    ]
                    for asset_id, price_list in self.price_data.items()
                },
                "financial_analyses": {
                    analysis_id: {
                        "asset_id": analysis.asset_id,
                        "analysis_type": analysis.analysis_type.value,
                        "results": analysis.results,
                        "predictions": analysis.predictions,
                        "risk_metrics": analysis.risk_metrics,
                        "recommendations": analysis.recommendations,
                        "created_at": analysis.created_at.isoformat()
                    }
                    for analysis_id, analysis in self.financial_analyses.items()
                },
                "summary": await self.get_financial_summary(),
                "exported_at": datetime.now().isoformat()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Datos financieros exportados a {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error exportando datos financieros: {e}")
            raise
























