import numpy as np
import pandas as pd

def calculate_metrics(cum_returns):
    daily_returns = cum_returns.diff().fillna(0)
    sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
    max_drawdown = np.min(cum_returns - cum_returns.cummax())
    return {"Sharpe Ratio": sharpe_ratio, "Maximum Drawdown": max_drawdown}

def trading_strategy(predictions, test_data, top_k=2):
    strategy_returns = []
    for i in range(len(predictions[list(predictions.keys())[0]])):
        pred_returns = {etf: predictions[etf][i] for etf in predictions.keys()}
        sorted_preds = sorted(pred_returns.items(), key=lambda x: x[1], reverse=True)
        long_positions = [sorted_preds[j][0] for j in range(top_k)]
        short_positions = [sorted_preds[-j - 1][0] for j in range(top_k)]
        step_return = sum(test_data[long].iloc[i] for long in long_positions) - \
                      sum(test_data[short].iloc[i] for short in short_positions)
        strategy_returns.append(step_return)
    return pd.Series(np.cumsum(strategy_returns), index=test_data.index[:len(strategy_returns)])
