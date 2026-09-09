"""
Generic data loader for timeframe-agnostic backtesting.
Reads from Excel/CSV, parses dates, infers frequency, and calculates indicators.
"""

import os
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from signal_system import signal_config as sc

try:
    import pandas_ta as ta
except ImportError:
    raise ImportError("Please install pandas_ta: pip install pandas_ta")

def load_and_prepare_data(file_path):
    """
    Load dataset from CSV or Excel, parse dates, infer frequency, and compute indicators.
    
    Parameters
    ----------
    file_path : str
        Path to the .xlsx or .csv dataset.
        
    Returns
    -------
    df : DataFrame with OHLCV data and computed technical indicators.
    """
    print(f"[SIGNAL-DATA] Loading dataset from {file_path}...")
    
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide .csv or .xlsx")

    # Find and parse date column
    date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
    if date_cols:
        df['date'] = pd.to_datetime(df[date_cols[0]])
        df = df.set_index('date')
    else:
        # Assume first column is date if no clear column name
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.iloc[:, 0])
            df = df.drop(df.columns[0], axis=1)
            
    df = df.sort_index()
    
    # Infer frequency
    inferred_freq = pd.infer_freq(df.index)
    print(f"  → Inferred timeframe/frequency: {inferred_freq}")
    
    # Standardize column names
    col_map = {}
    for c in df.columns:
        c_lower = c.lower()
        if c_lower == 'open' or c_lower == 'o': col_map[c] = 'Open'
        elif c_lower == 'high' or c_lower == 'h': col_map[c] = 'High'
        elif c_lower == 'low' or c_lower == 'l': col_map[c] = 'Low'
        elif c_lower == 'close' or c_lower == 'c': col_map[c] = 'Close'
        elif 'vol' in c_lower: col_map[c] = 'Volume'
    
    df = df.rename(columns=col_map)
    req_cols = ['Open', 'High', 'Low', 'Close']
    missing_cols = [c for c in req_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")
        
    df = df.dropna(subset=req_cols)
    
    # Compute returns
    df['ret'] = df['Close'].pct_change()
    
    print(f"  → Loaded {len(df)} rows.")
    print(f"  → Date range: {df.index.min()} to {df.index.max()}")
    
    # Calculate Indicators using pandas_ta
    print("[SIGNAL-DATA] Computing technical indicators with pandas_ta...")
    
    # Trend
    df['SMAVG'] = ta.sma(df['Close'], length=sc.LOOKBACK_MEDIUM)
    df['EMAVG'] = ta.ema(df['Close'], length=sc.LOOKBACK_SHORT)
    df['WMAVG'] = ta.wma(df['Close'], length=sc.LOOKBACK_MEDIUM)
    df['TMAVG'] = ta.hma(df['Close'], length=sc.LOOKBACK_MEDIUM) # Using HMA as TMA approx or wma
    
    # BB
    bb = ta.bbands(df['Close'], length=sc.LOOKBACK_MEDIUM, std=2)
    if bb is not None:
        df['BB_LOWER'] = bb.iloc[:, 0]
        df['BB_MA'] = bb.iloc[:, 1]
        df['BB_UPPER'] = bb.iloc[:, 2]
        df['BB_WIDTH'] = bb.iloc[:, 3]
        df['BB_PERCENT'] = bb.iloc[:, 4]
    
    # Momentum
    df['RSI'] = ta.rsi(df['Close'], length=sc.LOOKBACK_SHORT)
    df['ROC'] = ta.roc(df['Close'], length=sc.LOOKBACK_SHORT)
    df['MOMENTUM'] = df['Close'].diff(sc.LOOKBACK_SHORT)
    df['MOM_MA'] = ta.sma(df['MOMENTUM'], length=sc.LOOKBACK_SHORT)
    
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    if macd is not None:
        df['MACD'] = macd.iloc[:, 0]
        df['MACD_DIFF'] = macd.iloc[:, 1] # histogram
        df['MACD_SIGNAL'] = macd.iloc[:, 2]
        
    # Volatility / DMI
    adx = ta.adx(df['High'], df['Low'], df['Close'], length=sc.LOOKBACK_SHORT)
    if adx is not None:
        df['ADX'] = adx.iloc[:, 0]
        df['DMI_PLUS'] = adx.iloc[:, 1]
        df['DMI_MINUS'] = adx.iloc[:, 2]
    
    # Stochastics
    stoch = ta.stoch(df['High'], df['Low'], df['Close'], k=14, d=3, smooth_k=3)
    if stoch is not None:
        df['TAS_K'] = stoch.iloc[:, 0]
        df['TAS_D'] = stoch.iloc[:, 1]
    
    # CCI / Williams R
    df['CMCI'] = ta.cci(df['High'], df['Low'], df['Close'], length=sc.LOOKBACK_MEDIUM)
    df['WLPR'] = ta.willr(df['High'], df['Low'], df['Close'], length=sc.LOOKBACK_SHORT)
    
    # Parabolic SAR
    psar = ta.psar(df['High'], df['Low'], df['Close'])
    if psar is not None:
        df['PTPS'] = psar.iloc[:, 0] # PSAR value
        
    df = df.ffill().bfill()
    print(f"  → Indicators successfully computed.")
    return df

