#!/usr/bin/env python3

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SimulationConfig:
    """Configuration parameters for the simulation."""
    n_components: int = 100
    n_timesteps: int = 100
    dt: float = 1.0
    drift: float = 0.2
    volatility: float = 0.1
    failure_threshold: float = 10.0
    random_seed: Optional[int] = 42
    plot_style: str = 'default'
    n_paths_to_plot: int = 10
    save_plots: bool = True
    output_dir: str = 'results'
    
    @classmethod
    def from_json(cls, json_path: Union[str, Path]) -> 'SimulationConfig':
        """Load configuration from a JSON file."""
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_json(self, json_path: Union[str, Path]) -> None:
        """Save configuration to a JSON file."""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

class WienerProcess:
    """Implementation of Wiener Process for degradation modeling."""
    
    def __init__(self, drift: float, volatility: float, failure_threshold: float):
        """
        Initialize Wiener Process model.
        
        Args:
            drift: Mean degradation rate
            volatility: Standard deviation of degradation increments
            failure_threshold: Threshold value for failure
        """
        self._validate_parameters(drift, volatility, failure_threshold)
        self.drift = drift
        self.volatility = volatility
        self.failure_threshold = failure_threshold
        
    @staticmethod
    def _validate_parameters(drift: float, volatility: float, failure_threshold: float) -> None:
        """Validate input parameters."""
        if volatility <= 0:
            raise ValueError("Volatility must be positive")
        if failure_threshold <= 0:
            raise ValueError("Failure threshold must be positive")
            
    def simulate_path(self, time_points: np.ndarray) -> np.ndarray:
        """
        Simulate a single degradation path using Wiener Process.
        
        Args:
            time_points: Array of time points for simulation
            
        Returns:
            Array of degradation values
        """
        dt = np.diff(time_points)
        dW = np.random.normal(0, np.sqrt(dt))
        W = np.cumsum(dW)
        return self.drift * time_points + self.volatility * np.concatenate(([0], W))
    
    def simulate_multiple_paths(
        self, 
        n_components: int, 
        n_timesteps: int, 
        dt: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate multiple degradation paths and record failure times.
        
        Args:
            n_components: Number of components to simulate
            n_timesteps: Number of time steps
            dt: Time increment
            
        Returns:
            Tuple of (degradation paths array, failure times array)
        """
        degradation = np.zeros((n_components, n_timesteps))
        failure_times = []
        
        for i in range(n_components):
            for t in range(1, n_timesteps):
                degradation[i, t] = (
                    degradation[i, t-1] + 
                    self.drift * dt + 
                    self.volatility * np.random.randn()
                )
                if degradation[i, t] >= self.failure_threshold:
                    failure_times.append(t)
                    break
            
            if t == n_timesteps - 1 and degradation[i, t] < self.failure_threshold:
                logger.warning(f"Component {i} did not fail within the simulation time")
        
        return degradation, np.array(sorted(failure_times))

    def inverse_gaussian_params(self) -> Tuple[float, float]:
        """Calculate Inverse Gaussian parameters for First Passage Time."""
        mu = self.failure_threshold / self.drift  # mean
        lambda_ = self.failure_threshold**2 / self.volatility**2  # shape
        return mu, lambda_
    
    def fpt_pdf(self, t: np.ndarray) -> np.ndarray:
        """Calculate PDF of First Passage Time."""
        mu, lambda_ = self.inverse_gaussian_params()
        return np.sqrt(lambda_ / (2 * np.pi * t**3)) * np.exp(-lambda_ * (t - mu)**2 / (2 * mu**2 * t))
    
    def fpt_cdf(self, t: np.ndarray) -> np.ndarray:
        """Calculate CDF of First Passage Time."""
        mu, lambda_ = self.inverse_gaussian_params()
        z1 = np.sqrt(lambda_ / t) * ((t / mu) - 1)
        z2 = -np.sqrt(lambda_ / t) * ((t / mu) + 1)
        return stats.norm.cdf(z1) + np.exp(2 * lambda_ / mu) * stats.norm.cdf(z2)
    
    def fpt_hazard(self, t: np.ndarray) -> np.ndarray:
        """Calculate hazard rate of First Passage Time."""
        return self.fpt_pdf(t) / (1 - self.fpt_cdf(t) + 1e-10)

class CrowAMSAA:
    """Implementation of Crow-AMSAA (NHPP) model for reliability growth."""
    
    def __init__(self):
        """Initialize Crow-AMSAA model."""
        self.beta = None
        self.eta = None
        self.fit_statistics = {}
        
    @staticmethod
    def model(t: np.ndarray, beta: float, eta: float) -> np.ndarray:
        """
        Crow-AMSAA model function.
        
        Args:
            t: Time points
            beta: Shape parameter
            eta: Scale parameter
            
        Returns:
            Expected number of cumulative failures
        """
        return eta * t ** beta
    
    def fit(self, failure_times: np.ndarray, cumulative_failures: np.ndarray) -> Dict[str, float]:
        """
        Fit Crow-AMSAA model to failure data.
        
        Args:
            failure_times: Array of failure times
            cumulative_failures: Array of cumulative failure counts
            
        Returns:
            Dictionary containing fitted parameters and statistics
        """
        try:
            params, pcov = curve_fit(self.model, failure_times, cumulative_failures)
            self.beta, self.eta = params
            
            # Calculate goodness of fit statistics
            y_pred = self.predict(failure_times)
            residuals = cumulative_failures - y_pred
            r_squared = 1 - (np.sum(residuals**2) / np.sum((cumulative_failures - np.mean(cumulative_failures))**2))
            
            self.fit_statistics = {
                "beta": self.beta,
                "eta": self.eta,
                "r_squared": r_squared,
                "std_err_beta": np.sqrt(pcov[0,0]),
                "std_err_eta": np.sqrt(pcov[1,1])
            }
            
            return self.fit_statistics
            
        except RuntimeError as e:
            logger.error(f"Failed to fit Crow-AMSAA model: {str(e)}")
            raise
    
    def predict(self, time_points: np.ndarray) -> np.ndarray:
        """
        Predict cumulative failures using fitted model.
        
        Args:
            time_points: Array of time points
            
        Returns:
            Predicted cumulative failures
        """
        if self.beta is None or self.eta is None:
            raise ValueError("Model must be fitted before making predictions")
        return self.model(time_points, self.beta, self.eta)
    
    def calculate_mttf(self, t: float) -> float:
        """
        Calculate Mean Time To Failure (MTTF) at time t.
        
        Args:
            t: Time point
            
        Returns:
            MTTF value
        """
        if self.beta is None or self.eta is None:
            raise ValueError("Model must be fitted before calculating MTTF")
        return 1 / (self.beta * self.eta * t ** (self.beta - 1))

class ReliabilityAnalysis:
    """Class to manage the complete reliability analysis workflow."""
    
    def __init__(self, config: SimulationConfig):
        """
        Initialize reliability analysis.
        
        Args:
            config: Simulation configuration
        """
        self.config = config
        self.wiener = WienerProcess(
            drift=config.drift,
            volatility=config.volatility,
            failure_threshold=config.failure_threshold
        )
        self.crow_amsaa = CrowAMSAA()
        self.setup_output_directory()
        self.selected_times = []
        self.annotation = None
        self.results = None
        
    def setup_output_directory(self) -> None:
        """Create output directory if it doesn't exist."""
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
    def run_analysis(self) -> Dict[str, Union[np.ndarray, Dict[str, float]]]:
        """
        Run the complete reliability analysis.
        
        Returns:
            Dictionary containing analysis results
        """
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
        
        # Run simulation
        degradation, failure_times = self.wiener.simulate_multiple_paths(
            self.config.n_components,
            self.config.n_timesteps,
            self.config.dt
        )
        
        # Calculate FPT parameters
        mu, lambda_ = self.wiener.inverse_gaussian_params()
        
        # Calculate reliability metrics
        mttf = mu  # For Inverse Gaussian, mean is mu
        failure_rate = 1/mttf
        
        reliability_metrics = {
            "mttf": mttf,
            "failure_rate": failure_rate,
            "total_failures": len(failure_times),
            "failure_percentage": (len(failure_times) / self.config.n_components) * 100
        }
        
        results = {
            "degradation": degradation,
            "failure_times": failure_times,
            "fpt_params": {
                "mu": mu,
                "lambda": lambda_
            },
            "reliability_metrics": reliability_metrics
        }
        
        if self.config.save_plots:
            self.create_plots(results)
            
        return results
    
    def create_plots(self, results: Dict[str, Union[np.ndarray, Dict[str, float]]]) -> None:
        """
        Create, save, and display analysis plots.
        
        Args:
            results: Dictionary containing analysis results
        """
        self.results = results

        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Wiener Process Degradation Paths',
                'Fleet-wide Failures (MCF)',
                'Reliability vs Time',
                'MCF Rate'
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        # Plot degradation paths and get fitted distribution
        fitted_dist = self._plot_degradation_paths(fig, results["degradation"])
        
        if fitted_dist:
            # Plot reliability function
            self._plot_reliability_function(fig, results, fitted_dist)
            # Plot fleet-wide failures with confidence bounds
            self._plot_fleet_failures(fig, results, fitted_dist)
            # Plot MCF rate
            self._plot_mcf_rate(fig, results, fitted_dist)
        else:
            # Plot fleet-wide failures without theoretical predictions
            self._plot_fleet_failures(fig, results)

        # Update layout
        fig.update_layout(
            height=800,
            width=1200,
            showlegend=True,
            template='plotly_white',
            title_text="Reliability Analysis",
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.0
            )
        )

        # Save the figure if configured to do so
        if self.config.save_plots:
            save_path = Path(self.config.output_dir) / "reliability_analysis.html"
            fig.write_html(str(save_path))
            logger.info(f"Interactive plot saved to {save_path}")

        # Show the figure
        fig.show()

    def _plot_degradation_paths(self, fig: go.Figure, degradation: np.ndarray) -> Optional[Tuple[str, float, float]]:
        """Plot degradation paths and theoretical distribution."""
        # Find the time when each path crosses the threshold
        failure_times = []
        for path in degradation:
            failure_idx = np.where(path >= self.config.failure_threshold)[0]
            if len(failure_idx) > 0:
                failure_times.append(failure_idx[0])

        max_failure_time = max(failure_times) if failure_times else self.config.n_timesteps
        plot_end_time = int(max_failure_time * 1.1)  # Add 10% margin

        # Plot degradation paths up to failure with a colorscale
        for i in range(len(degradation)):
            path = degradation[i, :plot_end_time]
            failure_idx = np.where(path >= self.config.failure_threshold)[0]
            if len(failure_idx) > 0:
                end_idx = failure_idx[0]
                fig.add_trace(
                    go.Scatter(
                        x=np.arange(end_idx + 1),
                        y=path[:end_idx + 1],
                        mode='lines',
                        name=f'Path {i+1}',
                        line=dict(
                            width=1,
                            color='rgba(100,100,100,0.2)'  # Light gray with transparency
                        ),
                        showlegend=False,
                        hovertemplate="Time: %{x}<br>Degradation: %{y:.2f}<br>Path %{text}<extra></extra>",
                        text=[f"{i+1}"] * (end_idx + 1)
                    ),
                    row=1, col=1
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=np.arange(len(path)),
                        y=path,
                        mode='lines',
                        name=f'Path {i+1}',
                        line=dict(
                            width=1,
                            color='rgba(100,100,100,0.2)'  # Light gray with transparency
                        ),
                        showlegend=False,
                        hovertemplate="Time: %{x}<br>Degradation: %{y:.2f}<br>Path %{text}<extra></extra>",
                        text=[f"{i+1}"] * len(path)
                    ),
                    row=1, col=1
                )

        # Plot failure threshold
        fig.add_trace(
            go.Scatter(
                x=[0, plot_end_time],
                y=[self.config.failure_threshold, self.config.failure_threshold],
                mode='lines',
                name='Failure Threshold',
                line=dict(color='red', dash='dash', width=2),
            ),
            row=1, col=1
        )

        # Get Inverse Gaussian parameters
        if failure_times:
            mu, lambda_ = self.wiener.inverse_gaussian_params()
            
            # Start distribution plot from just before the first failure
            min_failure_time = min(failure_times)
            x = np.linspace(max(0.1, min_failure_time * 0.9), plot_end_time, 100)  # Start from 0.1 to avoid division by zero
            pdf = self.wiener.fpt_pdf(x)
            pdf_scaled = self.config.failure_threshold + (pdf / np.max(pdf)) * (self.config.failure_threshold * 0.2)
            
            # Add fill between distribution and threshold
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=pdf_scaled,
                    mode='lines',
                    name=f'FPT Distribution (μ={mu:.2f}, λ={lambda_:.2f})',
                    line=dict(color='green', width=2),
                    fill='tonexty',
                    fillcolor='rgba(128, 128, 128, 0.2)',
                    hovertemplate="Time: %{x}<br>PDF: %{text:.3f}<extra></extra>",
                    text=pdf
                ),
                row=1, col=1
            )

            # Add the threshold line for fill reference
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=[self.config.failure_threshold] * len(x),
                    mode='lines',
                    showlegend=False,
                    line=dict(width=0),
                    hoverinfo='skip'
                ),
                row=1, col=1
            )

            # Update axes
            fig.update_xaxes(title_text="Time", range=[0, plot_end_time], row=1, col=1)
            fig.update_yaxes(title_text="Degradation", range=[0, self.config.failure_threshold * 1.3], row=1, col=1)

            return ('inverse_gaussian', mu, lambda_)
        return None

    def _plot_reliability_function(self, fig: go.Figure, results: Dict, fpt_params: Optional[Tuple[str, float, float]] = None) -> None:
        """Plot reliability function using FPT distribution."""
        if fpt_params:
            t = np.linspace(0, max(results["failure_times"]) * 1.1, 100)
            reliability = 1 - self.wiener.fpt_cdf(t)
            
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=reliability,
                    mode='lines',
                    name='Reliability (FPT)',
                    line=dict(color='blue'),
                ),
                row=2, col=1
            )
            
            fig.update_xaxes(title_text="Time", row=2, col=1)
            fig.update_yaxes(title_text="Reliability", range=[0, 1.05], row=2, col=1)

    def _plot_mcf_rate(self, fig: go.Figure, results: Dict, fpt_params: Optional[Tuple[str, float, float]] = None) -> None:
        """Plot MCF rate (derivative of MCF) using FPT distribution."""
        if fpt_params:
            _, mu, lambda_ = fpt_params
            failure_times = results["failure_times"]
            n_total = self.config.n_components
            
            # Calculate MCF rate (derivative of MCF)
            t = np.linspace(0.1, max(failure_times) * 1.1, 100)  # Start from 0.1 to avoid division by zero
            mcf_rate = n_total * self.wiener.fpt_pdf(t)
            
            # Plot MCF rate
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=mcf_rate,
                    mode='lines',
                    name='MCF Rate',
                    line=dict(color='purple'),
                    hovertemplate="Time: %{x:.1f}<br>Rate: %{y:.2f}<extra></extra>"
                ),
                row=2, col=2
            )
            
            # Add empirical MCF rate using sliding window
            window_size = max(3, len(failure_times) // 10)
            empirical_rate = []
            centered_times = []
            
            for i in range(len(failure_times) - window_size):
                window = failure_times[i:i+window_size]
                rate = window_size / (window[-1] - window[0])
                centered_times.append(np.mean(window))
                empirical_rate.append(rate)
            
            if empirical_rate:
                fig.add_trace(
                    go.Scatter(
                        x=centered_times,
                        y=empirical_rate,
                        mode='markers',
                        name='Empirical Rate',
                        marker=dict(color='blue', size=8),
                        hovertemplate="Time: %{x:.1f}<br>Rate: %{y:.2f}<extra></extra>"
                    ),
                    row=2, col=2
                )
            
            fig.update_xaxes(title_text="Time", row=2, col=2)
            fig.update_yaxes(title_text="Failure Rate (failures/time)", row=2, col=2)

    def _plot_fleet_failures(self, fig: go.Figure, results: Dict, fpt_params: Optional[Tuple[str, float, float]] = None) -> None:
        """Plot fleet-wide failure analysis using MCF with confidence bounds."""
        failure_times = results["failure_times"]
        n_total = self.config.n_components
        
        # Plot observed cumulative failures
        fig.add_trace(
            go.Scatter(
                x=failure_times,
                y=np.arange(1, len(failure_times) + 1),
                mode='markers',
                name='Observed Failures',
                marker=dict(color='blue', size=8),
                hovertemplate="Time: %{x:.1f}<br>Failures: %{y}<extra></extra>"
            ),
            row=1, col=2
        )
        
        # Plot theoretical MCF with confidence bounds
        if fpt_params:
            _, mu, lambda_ = fpt_params
            t = np.linspace(0, max(failure_times) * 1.1, 100)
            mcf = n_total * self.wiener.fpt_cdf(t)
            
            # Calculate confidence bounds (using delta method approximation)
            alpha = 0.05  # 95% confidence level
            z = stats.norm.ppf(1 - alpha/2)
            std_mcf = np.sqrt(n_total * self.wiener.fpt_cdf(t) * (1 - self.wiener.fpt_cdf(t)))
            lower_bound = mcf - z * std_mcf
            upper_bound = mcf + z * std_mcf
            
            # Plot MCF with confidence bounds
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=mcf,
                    mode='lines',
                    name='Expected MCF',
                    line=dict(color='green'),
                    hovertemplate="Time: %{x:.1f}<br>Expected Failures: %{y:.1f}<extra></extra>"
                ),
                row=1, col=2
            )
            
            # Add confidence bounds
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=upper_bound,
                    mode='lines',
                    name='95% Confidence Bounds',
                    line=dict(color='rgba(0,128,0,0.2)', dash='dash'),
                    hovertemplate="Time: %{x:.1f}<br>Upper Bound: %{y:.1f}<extra></extra>"
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=lower_bound,
                    mode='lines',
                    line=dict(color='rgba(0,128,0,0.2)', dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(0,128,0,0.1)',
                    name='95% Confidence Bounds',
                    hovertemplate="Time: %{x:.1f}<br>Lower Bound: %{y:.1f}<extra></extra>"
                ),
                row=1, col=2
            )
            
            # Calculate and print goodness of fit
            expected_failures = n_total * self.wiener.fpt_cdf(failure_times)
            observed_failures = np.arange(1, len(failure_times) + 1)
            mse = np.mean((expected_failures - observed_failures)**2)
            print("\nFleet-wide Analysis:")
            print("-" * 50)
            print(f"Mean Time to Failure (MTTF): {mu:.3f}")
            print(f"Shape Parameter (λ): {lambda_:.3f}")
            print(f"Mean Square Error: {mse:.3f}")
        
        # Add reference line for total components
        fig.add_trace(
            go.Scatter(
                x=[0, max(failure_times) * 1.1],
                y=[n_total, n_total],
                mode='lines',
                name=f'Total Components (N={n_total})',
                line=dict(color='gray', dash='dot'),
            ),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Time", row=1, col=2)
        fig.update_yaxes(title_text="Cumulative Failures", row=1, col=2)

def main():
    """Main function to run the reliability analysis."""
    # Create default configuration
    config = SimulationConfig()
    
    try:
        # Create and run analysis
        analysis = ReliabilityAnalysis(config)
        results = analysis.run_analysis()
        
        # Print summary statistics
        print("\nAnalysis Results:")
        print("-" * 50)
        print("First Passage Time Parameters:")
        print(f"Mean (μ): {results['fpt_params']['mu']:.3f}")
        print(f"Shape (λ): {results['fpt_params']['lambda']:.3f}")
        print("\nReliability Metrics:")
        print(f"MTTF: {results['reliability_metrics']['mttf']:.2f}")
        print(f"Failure rate: {results['reliability_metrics']['failure_rate']:.4f}")
        print(f"Total failures: {results['reliability_metrics']['total_failures']}")
        print(f"Failure percentage: {results['reliability_metrics']['failure_percentage']:.1f}%")
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main() 