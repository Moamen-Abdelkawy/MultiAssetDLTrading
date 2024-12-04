from scripts import data_processing as dp
from scripts import data_visualization as dv
from scripts import stationarity_tests as st
from scripts import lstm_single_output as lstm_so
from scripts import lstm_multi_output as lstm_mo
from scripts import strategy_backtesting as sb
import pandas as pd

# Configuration
ETFS = ['SPY', 'TLT', 'SHY', 'GLD', 'DBO']
START_DATE = "2018-01-01"
END_DATE = "2022-12-30"
TIME_STEP = 25

def main():
    # Step 1: Data Processing
    print("Step 1: Data Processing")
    df = dp.download_data(ETFS, START_DATE, END_DATE)
    log_returns = dp.calculate_log_returns(df)
    train_data, val_data, test_data = dp.save_split_data(log_returns)
    
    # Step 2: Visualization
    print("Step 2: Data Visualization")
    dv.plot_etf_prices(df)
    dv.plot_log_returns_pairplot(log_returns)
    dv.plot_correlation_heatmap(log_returns)
    
    # Step 3: Stationarity Tests
    print("Step 3: Stationarity Tests")
    adf_results = st.adf_test(log_returns, ETFS)
    print("ADF Test Results:", adf_results)
    st.seasonal_decomposition(df, ETFS)
    
    # Step 4: Single-Output LSTM Models
    print("Step 4: Training Single-Output LSTM Models")
    single_output_models = {}
    for etf in ETFS:
        X_train, y_train = dp.create_dataset(train_data[[etf]], TIME_STEP)
        X_val, y_val = dp.create_dataset(val_data[[etf]], TIME_STEP)
        X_train = X_train.reshape(-1, TIME_STEP, 1)
        X_val = X_val.reshape(-1, TIME_STEP, 1)

        model = lstm_so.create_lstm_model((TIME_STEP, 1))
        model, _ = lstm_so.train_and_save_model(
            model, X_train, y_train, X_val, y_val, etf
        )
        single_output_models[etf] = model
    
    # Step 5: Multi-Output LSTM Model
    print("Step 5: Training Multi-Output LSTM Model")
    X_train, y_train = dp.create_multi_output_dataset(train_data[ETFS].values, TIME_STEP)
    X_val, y_val = dp.create_multi_output_dataset(val_data[ETFS].values, TIME_STEP)
    X_train = X_train.reshape(-1, TIME_STEP, len(ETFS))
    X_val = X_val.reshape(-1, TIME_STEP, len(ETFS))

    multi_output_model = lstm_mo.create_multi_output_lstm((TIME_STEP, len(ETFS)), len(ETFS))
    multi_output_model, _ = lstm_mo.train_and_save_multi_output_model(
        multi_output_model, X_train, y_train, X_val, y_val
    )
    
    # Step 6: Backtesting Trading Strategies
    print("Step 6: Backtesting Trading Strategies")
    predictions = {}
    for etf in ETFS:
        X_test, y_test = dp.create_dataset(test_data[[etf]], TIME_STEP)
        X_test = X_test.reshape(-1, TIME_STEP, 1)
        y_pred = single_output_models[etf].predict(X_test)
        predictions[etf] = y_pred.flatten()

    strategy_cum_returns = sb.trading_strategy(predictions, test_data)
    strategy_metrics = sb.calculate_metrics(strategy_cum_returns)
    print("Single-Output Strategy Metrics:", strategy_metrics)

    # Multi-output strategy
    X_test, y_test = dp.create_multi_output_dataset(test_data[ETFS].values, TIME_STEP)
    X_test = X_test.reshape(-1, TIME_STEP, len(ETFS))
    y_pred = multi_output_model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, index=test_data.index[:len(y_pred)], columns=ETFS)
    
    multi_output_cum_returns = sb.multi_output_trading_strategy(y_pred_df, test_data)
    multi_output_metrics = sb.calculate_metrics(multi_output_cum_returns)
    print("Multi-Output Strategy Metrics:", multi_output_metrics)

if __name__ == "__main__":
    main()
