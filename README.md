# MultiAssetDLTrading

**MultiAssetDLTrading** is a repository that implements deep learning-based trading strategies for multiple assets using Long Short-Term Memory (LSTM) models. It combines financial data preprocessing, model training, and strategy backtesting to analyze the performance of single-output and multi-output models.

## Repository Structure

```
MultiAssetDLTrading/
├── scripts/
│   ├── data_processing.py          # Functions for downloading and processing data
│   ├── data_visualization.py       # Visualization utilities
│   ├── stationarity_tests.py       # Functions for stationarity tests and seasonal decomposition
│   ├── lstm_single_output.py       # Implementation for single-output LSTM models
│   ├── lstm_multi_output.py        # Implementation for multi-output LSTM models
│   ├── strategy_backtesting.py     # Backtesting utilities for trading strategies
├── main.py                         # Main script to execute the entire pipeline
├── multi_asset_trading_analysis.ipynb  # Jupyter Notebook version of the analysis
├── README.md                       # Documentation for the repository
├── requirements.txt                # List of dependencies for the project
├── LICENSE                         # License information (MIT)
└── report.md                       # Markdown report summarizing the analysis
```

## Features

- **Data Processing**:
  - Download and preprocess financial data from Yahoo Finance.
  - Compute log returns and perform stationarity tests.

- **Visualization**:
  - Visualize adjusted close prices, log returns, correlation heatmaps, and seasonal decomposition.

- **LSTM Models**:
  - Train and evaluate single-output LSTM models for individual assets.
  - Implement and evaluate a multi-output LSTM model for simultaneous predictions.

- **Trading Strategy Backtesting**:
  - Develop and backtest trading strategies based on model predictions.
  - Compare the performance of the strategies to a buy-and-hold baseline.

## Getting Started

### Prerequisites

Ensure you have Python 3.8 or later installed. Install the dependencies listed in `requirements.txt` by running:

```bash
pip install -r requirements.txt
```

### Repository Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Moamen-Abdelkawy/MultiAssetDLTrading.git
   cd MultiAssetDLTrading
   ```

2. **Verify Repository Structure**:
   Ensure the repository structure matches the layout described above.

### Running the Project

1. **Using the Main Script**:
   Run the entire pipeline from data processing to backtesting by executing the `main.py` file:
   ```bash
   python main.py
   ```

2. **Using the Jupyter Notebook**:
   Explore and execute the analysis step-by-step using the `multi_asset_trading_analysis.ipynb` notebook. Launch it using:
   ```bash
   jupyter notebook multi_asset_trading_analysis.ipynb
   ```

### Outputs

- **Models**: Trained models are saved in the `data/models/` directory.
- **Processed Data**: Processed datasets (e.g., log returns, train/val/test splits) are saved in `data/processed/`.
- **Visualizations**: Interactive visualizations and plots are generated for analysis.

## Contact

This repository is maintained by **Moamen Abdelkawy**. For questions, suggestions, or collaboration inquiries, please reach out via email at [moamen.abdelkawy@outlook.com](mailto:moamen.abdelkawy@outlook.com).

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

We acknowledge the use of:
- [Yahoo Finance](https://finance.yahoo.com) for financial data.
- TensorFlow and Scikit-learn for deep learning and data preprocessing.
- Matplotlib and Seaborn for visualizations.

---
