import numpy as np

def computeRSI(pastPriceVec, period=12):
    """
    Function to compute Relative Strength Index (RSI) based on past prices.
    :param pastPriceVec: list of past prices
    :param period: the period for RSI calculation (default 12)
    :return: computed RSI value
    """
    if len(pastPriceVec) < period + 1:
        return 50  # Neutral value if insufficient data
    
    gains = []
    losses = []
    
    for i in range(1, period + 1):
        delta = pastPriceVec[-i] - pastPriceVec[-i-1]
        if delta > 0:
            gains.append(delta)
        else:
            losses.append(-delta)

    avg_gain = np.mean(gains) if gains else 0
    avg_loss = np.mean(losses) if losses else 0
    
    if avg_loss == 0:
        return 100  # If no losses, RSI is 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def myStrategy(pastPriceVec, currentPrice):
    """
    Trading strategy based on RSI indicator.
    :param pastPriceVec: List of past prices
    :param currentPrice: The current price
    :return: 1 (buy), -1 (sell), or 0 (hold)
    """
    rsi_period = 12  # Adjusted RSI period for more sensitivity
    buy_threshold = 25  # Lower buy threshold to capture more buy signals
    sell_threshold = 75  # Higher sell threshold to capture more sell signals
    
    if len(pastPriceVec) < rsi_period:
        return 0
    
    rsi = computeRSI(pastPriceVec, rsi_period)
    
    action = 0
    
    if rsi < buy_threshold:
        action = 1  # Buy signal
    elif rsi > sell_threshold:
        action = -1  # Sell signal
    
    return action


# rr=556.448704% 