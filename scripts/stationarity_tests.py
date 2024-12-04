from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import pickle

def adf_test(log_returns, etfs):
    results = {}
    for etf in etfs:
        result = adfuller(log_returns[etf])
        results[etf] = {'ADF Statistic': result[0], 'p-value': result[1]}
    with open("../data/processed/adf_test_results.pkl", "wb") as file:
        pickle.dump(results, file)
    return results

def seasonal_decomposition(df, etfs, period=252):
    for etf in etfs:
        decomposition = seasonal_decompose(df[etf], model='multiplicative', period=period)
        decomposition.plot()
        plt.suptitle(f"Seasonal Decomposition for {etf}", y=1.02)
        plt.show()
