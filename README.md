# Market Risk Model: VaR and Stress Testing

Example code for VAR

This Python project implements a Market Risk Model that calculates Value at Risk (VaR) and performs stress testing on a portfolio of assets. It supports three types of VaR calculations: Historical VaR, Parametric (Variance-Covariance) VaR, and Monte Carlo Simulation VaR. Additionally, it offers Conditional VaR (CVaR) and scenario-based stress testing for a portfolio under predefined market shocks.

## Features

•	**Historical VaR**: Based on historical price movements.
•	**Parametric VaR** (Variance-Covariance): Assumes normal distribution of returns.
•	**Monte Carlo VaR**: Simulates thousands of potential future price paths for assets.
•	**Conditional VaR (CVaR)**: Captures tail risk beyond the VaR threshold.
•	**Stress Testing**: Evaluate portfolio under custom market shock scenarios.
•	**Parallel Processing**: For faster Monte Carlo simulations.

## Installation

Clone the repository:
```bash
git clone https://github.com/FIAQuant/VAR-Stress-Test-Model.git
pip install -r requirements.txt
```

## Usage Example

'''python
# Portfolio configuration
tickers = ['AAPL', 'MSFT', 'GOOGL']  # Assets in the portfolio
start_date = '2020-01-01'            # Start date for historical data
end_date = '2023-01-01'              # End date for historical data
confidence_level = 0.95              # VaR confidence level (95%)
holding_period = 1                   # Holding period for VaR (1 day)
'''

```python
from portfolio_optimiser.main
import Main

if __name__ == "__main__":
  python market_risk_model.py
```



### Output
This will:
	1.	Fetch historical market data for the specified tickers.
	2.	Calculate Historical VaR, Parametric VaR, and Monte Carlo VaR.
	3.	Perform Conditional VaR analysis.
	4.	Run stress tests based on custom-defined scenarios.
	5.	Save results as CSV files and generate visual reports in the reports folder.

5. Example Output

After running the script, results will be saved in the reports/ directory.

	•	A plot showing the VaR thresholds.
	•	A CSV file with calculated VaR values and stress test results.

 ## Advanced Features
 
 '''python
 risk_model.run_analysis(shock_factors=shock_factors, simulations=10000)
'''

## Contributing

Contributions are welcome! Please feel free to fork the repository and submit pull requests or open issues for bugs or feature requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This package utilises `yfinance`, `numpy`, `scipy`, and `pandas`. 
Inspiration for portfolio optimisation techniques comes from modern portfolio theory.

## Contact

Azim Patel  
GitHub: [FIAQuant](https://github.com/FIAQuant)  
Email: N/A

