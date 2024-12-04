import matplotlib.pyplot as plt
import seaborn as sns

def plot_etf_prices(df):
    df.plot(figsize=(14, 7), title='ETF Adjusted Close Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    plt.show()

def plot_log_returns_pairplot(log_returns):
    sns.pairplot(log_returns)
    plt.suptitle("Pairplot of Log Returns", y=1.02)
    plt.show()

def plot_correlation_heatmap(log_returns):
    correlation = log_returns.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix Heatmap')
    plt.show()
