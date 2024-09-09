import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import logging
import os
import multiprocessing
from scipy.stats import norm

# Create a directory for reports
if not os.path.exists('reports'):
    os.makedirs('reports')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s')


class MarketRiskModel:
    def __init__(self, tickers, start_date, end_date, confidence_level=0.95, holding_period=1):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.confidence_level = confidence_level
        self.holding_period = holding_period
        self.data = None
        self.returns = None

    def fetch_data(self):
        """ Fetches historical data for the specified tickers and computes log returns """
        try:
            logging.info(f"Fetching data for {self.tickers} from {self.start_date} to {self.end_date}")
            self.data = yf.download(self.tickers, start=self.start_date, end=self.end_date)['Adj Close']
            self.returns = np.log(self.data / self.data.shift(1)).dropna()
            logging.info("Data fetched and log returns calculated successfully.")
        except Exception as e:
            logging.error(f"Error fetching data: {e}")
            raise Exception("Data fetching failed.")

    def var_historical(self):
        """ Historical simulation of Value at Risk (VaR) """
        if self.returns is None:
            raise Exception("Returns data is missing.")
        var = self.returns.quantile(1 - self.confidence_level, axis=0)
        logging.info(f"Historical VaR: {var}")
        return var

    def var_parametric(self):
        """ Parametric (variance-covariance) VaR calculation """
        if self.returns is None:
            raise Exception("Returns data is missing.")
        mean_returns = self.returns.mean()
        std_dev = self.returns.std()
        z_score = norm.ppf(1 - self.confidence_level)
        var = mean_returns - z_score * std_dev * np.sqrt(self.holding_period)
        logging.info(f"Parametric VaR: {var}")
        return var

    def var_monte_carlo(self, simulations=10000):
        """ Monte Carlo simulation for VaR """
        if self.returns is None:
            raise Exception("Returns data is missing.")
        mean_returns = self.returns.mean()
        cov_matrix = self.returns.cov()

        # Perform Monte Carlo simulation in parallel
        logging.info("Running Monte Carlo simulations...")
        results = np.zeros(simulations)

        def run_simulation(sim_index):
            simulated_returns = np.random.multivariate_normal(mean_returns, cov_matrix, self.holding_period)
            portfolio_return = simulated_returns.sum(axis=0)
            results[sim_index] = portfolio_return

        with multiprocessing.Pool() as pool:
            pool.map(run_simulation, range(simulations))

        var = np.percentile(results, (1 - self.confidence_level) * 100)
        logging.info(f"Monte Carlo VaR: {var}")
        return var

    def cvar(self, var):
        """ Conditional VaR (CVaR) calculation """
        if self.returns is None:
            raise Exception("Returns data is missing.")
        cvar = self.returns[self.returns <= var].mean()
        logging.info(f"Conditional VaR (CVaR): {cvar}")
        return cvar

    def stress_test(self, shock_factors):
        """ Perform scenario analysis for stress testing the portfolio """
        if self.returns is None:
            raise Exception("Returns data is missing.")
        stressed_returns = self.returns.copy()
        for ticker, shock in shock_factors.items():
            if ticker in stressed_returns.columns:
                stressed_returns[ticker] *= (1 + shock)
        return stressed_returns.mean()

    def generate_report(self, historical_var, parametric_var, monte_carlo_var, stressed_mean_return):
        """ Generate a report for VaR analysis """
        plt.figure(figsize=(12, 6))
        plt.plot(self.returns, label='Daily Returns')
        plt.axhline(y=historical_var.mean(), color='r', linestyle='-', label=f'Historical VaR')
        plt.axhline(y=parametric_var.mean(), color='b', linestyle='-', label=f'Parametric VaR')
        plt.axhline(y=monte_carlo_var, color='g', linestyle='-', label=f'Monte Carlo VaR')
        plt.title(f"VaR Analysis for Portfolio: {self.tickers}")
        plt.legend()
        report_path = f'reports/var_analysis_{self.tickers[0]}.png'
        plt.savefig(report_path)
        plt.close()
        logging.info(f"Report saved to {report_path}")

    def save_results(self, historical_var, parametric_var, monte_carlo_var, cvar_value, stressed_mean):
        """ Save the results to a CSV file """
        results = pd.DataFrame({
            "Historical VaR": historical_var,
            "Parametric VaR": parametric_var,
            "Monte Carlo VaR": [monte_carlo_var] * len(historical_var),
            "CVaR": [cvar_value] * len(historical_var),
            "Stressed Mean Return": [stressed_mean] * len(historical_var)
        })
        results_path = f'reports/var_results_{self.tickers[0]}.csv'
        results.to_csv(results_path)
        logging.info(f"Results saved to {results_path}")

    def run_analysis(self, shock_factors=None, simulations=10000):
        """ Main function to run the full VaR and stress test analysis """
        try:
            self.fetch_data()
            historical_var = self.var_historical()
            parametric_var = self.var_parametric()
            monte_carlo_var = self.var_monte_carlo(simulations=simulations)
            cvar_value = self.cvar(monte_carlo_var)

            stressed_mean = None
            if shock_factors:
                stressed_mean = self.stress_test(shock_factors)

            self.generate_report(historical_var, parametric_var, monte_carlo_var, stressed_mean)
            self.save_results(historical_var, parametric_var, monte_carlo_var, cvar_value, stressed_mean)
            logging.info("Market risk analysis completed successfully.")
        except Exception as e:
            logging.error(f"Analysis failed: {e}")


# Portfolio configuration
tickers = ['AAPL', 'MSFT', 'GOOGL']
start_date = '2020-01-01'
end_date = '2023-01-01'
confidence_level = 0.95
holding_period = 1

# Define stress test shock factors
shock_factors = {
    'AAPL': -0.20,  # Simulate a 20% drop in Apple
    'MSFT': -0.15,  # Simulate a 15% drop in Microsoft
    'GOOGL': -0.10  # Simulate a 10% drop in Google
}

# Initialize and run the analysis
risk_model = MarketRiskModel(tickers, start_date, end_date, confidence_level, holding_period)
risk_model.run_analysis(shock_factors=shock_factors, simulations=10000)