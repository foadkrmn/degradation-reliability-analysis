#!/usr/bin/env python3

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict

class WienerProcess:
    """
    Implementation of Wiener Process for degradation modeling.
    """
    def __init__(self, drift: float, volatility: float, failure_threshold: float):
        self.drift = drift
        self.volatility = volatility
        self.failure_threshold = failure_threshold
        
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
    
    def simulate_multiple_paths(self, n_components: int, n_timesteps: int, dt: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
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
                degradation[i, t] = degradation[i, t-1] + self.drift * dt + self.volatility * np.random.randn()
                if degradation[i, t] >= self.failure_threshold:
                    failure_times.append(t)
                    break
        
        return degradation, np.array(sorted(failure_times))

class CrowAMSAA:
    """
    Implementation of Crow-AMSAA (NHPP) model for reliability growth.
    """
    def __init__(self):
        self.beta = None
        self.eta = None
        
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
            Dictionary containing fitted parameters
        """
        params, _ = curve_fit(self.model, failure_times, cumulative_failures)
        self.beta, self.eta = params
        return {"beta": self.beta, "eta": self.eta}
    
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

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Simulation parameters
    N = 100  # Number of components
    T = 100  # Total time steps
    dt = 1   # Time increment
    mu = 0.2  # Mean degradation rate (drift)
    sigma = 0.1  # Variability in degradation (volatility)
    failure_threshold = 10  # Failure threshold
    
    # Create and run Wiener Process simulation
    wiener = WienerProcess(drift=mu, volatility=sigma, failure_threshold=failure_threshold)
    degradation, failure_times = wiener.simulate_multiple_paths(N, T, dt)
    
    # Prepare data for Crow-AMSAA fitting
    cumulative_failures = np.arange(1, len(failure_times) + 1)
    
    # Fit Crow-AMSAA model
    crow_amsaa = CrowAMSAA()
    params = crow_amsaa.fit(failure_times, cumulative_failures)
    
    # Generate predictions
    t_fit = np.linspace(1, max(failure_times), 100)
    failure_fit = crow_amsaa.predict(t_fit)
    
    # Plotting
    plt.figure(figsize=(15, 5))
    
    # Plot degradation paths
    plt.subplot(1, 2, 1)
    for i in range(min(10, N)):  # Plot first 10 paths
        plt.plot(np.arange(T), degradation[i], alpha=0.5)
    plt.axhline(failure_threshold, color='r', linestyle='--', label='Failure Threshold')
    plt.title('Wiener Process Degradation Paths')
    plt.xlabel('Time')
    plt.ylabel('Degradation')
    plt.legend()
    
    # Plot Crow-AMSAA fit
    plt.subplot(1, 2, 2)
    plt.scatter(failure_times, cumulative_failures, label="Simulated Failures", color="blue", alpha=0.6)
    plt.plot(t_fit, failure_fit, 
             label=f"Crow-AMSAA Fit ($\\beta={params['beta']:.2f}, \\eta={params['eta']:.2f}$)", 
             color="red", linestyle="--")
    plt.axhline(N, color="gray", linestyle=":", label=f"Total Components (N={N})")
    plt.xlabel("Time")
    plt.ylabel("Cumulative Failures")
    plt.title("Crow-AMSAA Model Fit")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Fitted Crow-AMSAA parameters: beta = {params['beta']:.3f}, eta = {params['eta']:.3f}")

if __name__ == "__main__":
    main() 