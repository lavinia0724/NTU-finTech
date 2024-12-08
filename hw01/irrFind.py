import numpy as np
from scipy.optimize import fsolve

def irrFind(cashFlowVec, cashFlowPeriod, compoundPeriod):
    # 定義基於 IRR 的淨現值 (NPV) 函數
    def npv(irr):
        total = 0
        for i, cash_flow in enumerate(cashFlowVec):
            # 根據複利周期調整折現因子
            total += cash_flow / (1 + irr / (12 / compoundPeriod))**(i * (cashFlowPeriod / compoundPeriod))
        return total

    # 使用 fsolve 來求解 IRR，初始猜測為 0
    irr = fsolve(npv, 0)[0]  # 初始猜測是 0

    return irr

if __name__ == "__main__":
    import sys
    input = sys.stdin.read
    data = input().strip().split('\n')
    
    for line in data:
        # 解析每一行為現金流向量和期間
        values = list(map(int, line.split()))
        cashFlowVec = values[:-2]
        cashFlowPeriod = values[-2]
        compoundPeriod = values[-1]
        
        # 計算 IRR
        irr = irrFind(cashFlowVec, cashFlowPeriod, compoundPeriod)
        
        # 以百分比格式輸出 IRR，四位小數
        print(f"{irr*100:.4f}")
