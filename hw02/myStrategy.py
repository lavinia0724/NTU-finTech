import numpy as np

def computeRSI(pastPriceVec, period=14):
    if len(pastPriceVec) < period + 1:
        return 50
    
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
        return 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def computeMA(pastPriceVec, period=14):
    if len(pastPriceVec) < period:
        return 0
    return np.mean(pastPriceVec[-period:])

def myStrategy(pastPriceVec, currentPrice):
    rsi_period = 24
    buy_threshold = 33  
    sell_threshold = 70  
    ma_period = 14  
    
    if len(pastPriceVec) < max(rsi_period, ma_period):
        return 0
    
    rsi = computeRSI(pastPriceVec, rsi_period)
    ma = computeMA(pastPriceVec, ma_period)

    action = 0
    
    
    if rsi < buy_threshold and currentPrice < ma:
        action = 1  # Buy signal
    elif rsi > sell_threshold and currentPrice > ma:
        action = -1  # Sell signal
    
    return action
