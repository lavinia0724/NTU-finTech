import numpy as np
from typing import List, Tuple

def myAction01(priceMat, transFeeRate):
    """
    修正版DP解法
    確保所有交易操作都符合規則要求
    """
    days, stocks = priceMat.shape
    initial_capital = 1000.0

    # dp[day][state] 表示在第day天處於state狀態時的最大資產
    # state: -1表示現金, 0~stocks-1表示持有對應股票
    dp = np.zeros((days + 1, stocks + 1))
    prev = np.zeros((days + 1, stocks + 1, 3))  # [前一天, 前一狀態, 交易金額]
    
    # 初始狀態:只有現金
    dp[0][-1] = initial_capital
    for s in range(stocks):
        dp[0][s] = 0

    # 對每一天
    for day in range(days):
        if day == days - 1:  # 最後一天只能賣出
            continue
            
        # 如果現在持有現金
        if dp[day][-1] > 0:
            cash = dp[day][-1]
            # 可以買入任何股票
            for stock in range(stocks):
                if priceMat[day][stock] > 0:
                    value_after_fee = cash * (1 - transFeeRate)
                    shares = value_after_fee / priceMat[day][stock]
                    next_value = shares * priceMat[day + 1][stock]
                    
                    if next_value > dp[day + 1][stock]:
                        dp[day + 1][stock] = next_value
                        prev[day + 1][stock] = [day, -1, cash]

            # 或者繼續持有現金
            if cash > dp[day + 1][-1]:
                dp[day + 1][-1] = cash
                prev[day + 1][-1] = [day, -1, cash]

        # 如果持有股票
        for stock in range(stocks):
            if dp[day][stock] > 0:
                curr_value = dp[day][stock]
                shares = curr_value / priceMat[day][stock]
                
                # 可以賣出變現金
                sell_value = shares * priceMat[day][stock] * (1 - transFeeRate)
                if sell_value > dp[day + 1][-1]:
                    dp[day + 1][-1] = sell_value
                    prev[day + 1][-1] = [day, stock, curr_value]
                
                # 可以轉換到其他股票
                for new_stock in range(stocks):
                    if new_stock != stock:
                        cash_equiv = shares * priceMat[day][stock]
                        after_fees = cash_equiv * (1 - transFeeRate) * (1 - transFeeRate)
                        new_shares = after_fees / priceMat[day][new_stock]
                        next_value = new_shares * priceMat[day + 1][new_stock]
                        
                        if next_value > dp[day + 1][new_stock]:
                            dp[day + 1][new_stock] = next_value
                            prev[day + 1][new_stock] = [day, stock, cash_equiv]
                
                # 可以繼續持有
                next_value = shares * priceMat[day + 1][stock]
                if next_value > dp[day + 1][stock]:
                    dp[day + 1][stock] = next_value
                    prev[day + 1][stock] = [day, stock, curr_value]

    # 回溯構建交易序列
    actionMat = []
    
    # 最後一天的處理
    final_day = days - 1
    max_value = 0
    best_state = -1
    best_prev_state = -1
    best_amount = 0
    
    # 檢查所有可能的最終狀態
    for s in range(stocks):
        if dp[final_day][s] > max_value:
            max_value = dp[final_day][s]
            best_state = s
            best_prev_state = int(prev[final_day][s][1])
            best_amount = prev[final_day][s][2]
            
    # 如果最後持有股票，需要賣出
    if best_state != -1:
        shares = dp[final_day][best_state] / priceMat[final_day][best_state]
        final_value = shares * priceMat[final_day][best_state]
        actionMat.append([final_day, best_state, -1, final_value])

    # 回溯構建完整的交易序列
    current_day = final_day
    current_state = best_state
    
    while current_day > 0:
        prev_day = int(prev[current_day][current_state][0])
        prev_state = int(prev[current_day][current_state][1])
        amount = prev[current_day][current_state][2]
        
        if prev_day == current_day:
            break
            
        if prev_state != current_state:
            if prev_state == -1:  # 現金到股票
                actionMat.append([prev_day, -1, current_state, amount])
            elif current_state == -1:  # 股票到現金
                actionMat.append([prev_day, prev_state, -1, amount])
            else:  # 股票到股票
                actionMat.append([prev_day, prev_state, current_state, amount])
                
        current_day = prev_day
        current_state = prev_state
        
    # 反轉得到時間順序的操作序列
    actionMat = actionMat[::-1]
    
    # 驗證所有操作的合法性
    for action in actionMat:
        day, a, b, z = action
        assert 0 <= day < days
        assert z > 0
        assert (a == -1 and 0 <= b < stocks) or (0 <= a < stocks and b == -1) or (0 <= a < stocks and 0 <= b < stocks)
        
    return actionMat












def myAction02(priceMat, transFeeRate, K):
    # return []
    """
    Optimized trading strategy using dynamic programming with K-day total cash holding constraint.
    Maximizes return rate while ensuring total cash holding days >= K.

    Args:
        priceMat: Price matrix (days x stocks)
        transFeeRate: Transaction fee rate
        K: Required minimum total cash holding days
    """
    days, stocks = priceMat.shape
    initial_capital = 1000.0

    # dp[day][state][cash_days]: max total value at day, in state, with cash_days
    dp = [{} for _ in range(days)]
    prev = [{} for _ in range(days)]

    # Initialize starting state
    dp[0][(-1, 1)] = initial_capital  # On day 0, holding cash, cash_days = 1

    for day in range(days - 1):
        for (state, cash_days), curr_value in dp[day].items():
            if state == -1:  # Holding cash
                # Option 1: Continue holding cash
                new_state = -1
                new_cash_days = min(cash_days + 1, K)
                key = (new_state, new_cash_days)
                value = curr_value
                if dp[day + 1].get(key, 0) < value:
                    dp[day + 1][key] = value
                    prev[day + 1][key] = (state, cash_days, None)

                # Option 2: Buy stock
                for s in range(stocks):
                    if priceMat[day][s] <= 0:
                        continue
                    shares = curr_value * (1 - transFeeRate) / priceMat[day][s]
                    future_value = shares * priceMat[day + 1][s]
                    new_state = s
                    new_cash_days = cash_days  # cash_days remains the same
                    key = (new_state, new_cash_days)
                    if dp[day + 1].get(key, 0) < future_value:
                        dp[day + 1][key] = future_value
                        prev[day + 1][key] = (state, cash_days, curr_value)
            else:  # Holding stock
                shares = curr_value / priceMat[day][state]
                # Option 1: Continue holding stock
                new_state = state
                new_cash_days = cash_days
                future_value = shares * priceMat[day + 1][state]
                key = (new_state, new_cash_days)
                if dp[day + 1].get(key, 0) < future_value:
                    dp[day + 1][key] = future_value
                    prev[day + 1][key] = (state, cash_days, None)

                # Option 2: Sell stock
                sell_value = shares * priceMat[day][state] * (1 - transFeeRate)
                new_state = -1
                new_cash_days = min(cash_days + 1, K)
                key = (new_state, new_cash_days)
                if dp[day + 1].get(key, 0) < sell_value:
                    dp[day + 1][key] = sell_value
                    prev[day + 1][key] = (state, cash_days, curr_value)

                # Option 3: Switch to another stock
                for s_new in range(stocks):
                    if s_new == state or priceMat[day][s_new] <= 0:
                        continue
                    cash_equiv = shares * priceMat[day][state] * (1 - transFeeRate)
                    after_fees = cash_equiv * (1 - transFeeRate)
                    new_shares = after_fees / priceMat[day][s_new]
                    future_value = new_shares * priceMat[day + 1][s_new]
                    new_state = s_new
                    new_cash_days = cash_days
                    key = (new_state, new_cash_days)
                    if dp[day + 1].get(key, 0) < future_value:
                        dp[day + 1][key] = future_value
                        prev[day + 1][key] = (state, cash_days, curr_value)

    # Find the best ending state where total cash holding days >= K
    max_value = 0
    best_state = None
    for (state, cash_days), value in dp[days - 1].items():
        if cash_days >= K:
            final_value = value
            if state >= 0:  # If holding stock, need to sell it
                shares = value / priceMat[days - 1][state]
                final_value = shares * priceMat[days - 1][state] * (1 - transFeeRate)
            if final_value > max_value:
                max_value = final_value
                best_state = (state, cash_days)

    if not best_state:
        return []

    # Reconstruct actions
    actions = []
    day = days - 1
    current_state = best_state
    if current_state[0] >= 0:
        # Sell the stock at the end
        value = dp[day][current_state]
        actions.append([day, current_state[0], -1, value])
    while day > 0:
        prev_state, prev_cash_days, transaction_value = prev[day][current_state]
        if transaction_value is not None:
            if prev_state == -1 and current_state[0] >= 0:
                # Buy stock
                actions.append([day - 1, -1, current_state[0], transaction_value])
            elif prev_state >= 0 and current_state[0] == -1:
                # Sell stock
                actions.append([day - 1, prev_state, -1, transaction_value])
            elif prev_state >= 0 and current_state[0] >= 0 and prev_state != current_state[0]:
                # Switch stock
                actions.append([day - 1, prev_state, current_state[0], transaction_value])
        current_state = (prev_state, prev_cash_days)
        day -= 1
    actions.reverse()
    return actions













































def myAction03(priceMat, transFeeRate, K):
    """
    Optimized trading strategy that ensures at least one period of K consecutive cash-holding days
    while maximizing return rate.
    
    Args:
        priceMat: Price matrix (days x stocks)
        transFeeRate: Transaction fee rate
        K: Required minimum consecutive cash holding days
    """
    days, stocks = priceMat.shape
    initial_capital = 1000.0

    # dp[day][state][consec_cash][done_K]: max value
    # state: -1 for cash, 0~stocks-1 for holding stock
    # consec_cash: current consecutive cash holding days
    # done_K: whether we've completed K consecutive cash days
    dp = [{} for _ in range(days + 1)]
    prev = [{} for _ in range(days + 1)]
    
    # Initialize with cash position
    dp[0][(-1, 0, False)] = initial_capital

    for day in range(days):
        curr_prices = priceMat[day]
        
        # Early skip of clearly unprofitable states
        if not dp[day]:
            continue

        for (state, consec_cash, done_K), value in dp[day].items():
            if state == -1:  # Currently holding cash
                # Option 1: Keep holding cash
                new_consec = min(consec_cash + 1, K)
                new_done_K = done_K or new_consec >= K
                key = (-1, new_consec, new_done_K)
                
                if dp[day + 1].get(key, 0) < value:
                    dp[day + 1][key] = value
                    prev[day + 1][key] = (state, consec_cash, done_K, None)

                # Option 2: Buy stock if we've either met K days requirement or still have time
                if done_K or days - day > K:
                    for stock in range(stocks):
                        if curr_prices[stock] <= 0:
                            continue
                        
                        shares = value * (1 - transFeeRate) / curr_prices[stock]
                        if day + 1 < days:
                            future_value = shares * priceMat[day + 1][stock]
                            key = (stock, 0, done_K)
                            
                            if dp[day + 1].get(key, 0) < future_value:
                                dp[day + 1][key] = future_value
                                prev[day + 1][key] = (state, consec_cash, done_K, value)

            else:  # Holding stock
                shares = value / curr_prices[state]
                
                # Option 1: Continue holding
                if day + 1 < days:
                    future_value = shares * priceMat[day + 1][state]
                    key = (state, 0, done_K)
                    
                    if dp[day + 1].get(key, 0) < future_value:
                        dp[day + 1][key] = future_value
                        prev[day + 1][key] = (state, consec_cash, done_K, None)

                # Option 2: Sell to cash
                sell_value = shares * curr_prices[state] * (1 - transFeeRate)
                key = (-1, 1, done_K)  # Start counting cash days
                
                if dp[day + 1].get(key, 0) < sell_value:
                    dp[day + 1][key] = sell_value
                    prev[day + 1][key] = (state, consec_cash, done_K, value)

                # Option 3: Switch stocks
                if done_K or days - day > K:
                    for new_stock in range(stocks):
                        if new_stock == state or curr_prices[new_stock] <= 0:
                            continue
                            
                        sell_amount = shares * curr_prices[state]
                        new_shares = sell_amount * (1 - transFeeRate) * (1 - transFeeRate) / curr_prices[new_stock]
                        
                        if day + 1 < days:
                            future_value = new_shares * priceMat[day + 1][new_stock]
                            key = (new_stock, 0, done_K)
                            
                            if dp[day + 1].get(key, 0) < future_value:
                                dp[day + 1][key] = future_value
                                prev[day + 1][key] = (state, consec_cash, done_K, sell_amount)

    # Find best final state that meets K-day requirement
    max_value = 0
    best_state = None
    
    for (state, consec_cash, done_K), value in dp[days].items():
        if done_K:
            final_value = value
            if state >= 0:
                shares = value / priceMat[days - 1][state]
                final_value = shares * priceMat[days - 1][state] * (1 - transFeeRate)
            
            if final_value > max_value:
                max_value = final_value
                best_state = (state, consec_cash, done_K)

    if not best_state:
        return []

    # Reconstruct action sequence
    actions = []
    day = days
    current_state = best_state
    
    if current_state[0] >= 0:
        # Sell final stock position
        value = dp[day][current_state]
        actions.append([day - 1, current_state[0], -1, value])

    while day > 0:
        if current_state not in prev[day]:
            break
            
        prev_state, prev_cash, prev_done_K, transaction_value = prev[day][current_state]
        if transaction_value is not None:
            if prev_state == -1 and current_state[0] >= 0:
                # Buy stock
                actions.append([day - 1, -1, current_state[0], transaction_value])
            elif prev_state >= 0 and current_state[0] == -1:
                # Sell stock
                actions.append([day - 1, prev_state, -1, transaction_value])
            elif prev_state >= 0 and current_state[0] >= 0:
                # Switch stocks
                actions.append([day - 1, prev_state, current_state[0], transaction_value])
        
        current_state = (prev_state, prev_cash, prev_done_K)
        day -= 1

    return actions[::-1]