# Degradation Reliability Analysis

A Python toolkit for analyzing degradation-based reliability using Wiener processes and Mean Cumulative Function (MCF) methods. This tool is particularly useful for modeling and predicting the reliability of systems that degrade over time, such as batteries, mechanical components, or electronic devices.

## Features

- **Wiener Process Degradation Modeling**: Simulate degradation paths with drift and volatility parameters
- **First Passage Time Analysis**: Calculate reliability metrics using Inverse Gaussian distribution
- **Mean Cumulative Function (MCF)**: Analyze fleet-wide failure patterns
- **Interactive Visualizations**:
  - Degradation paths with failure threshold
  - Fleet-wide failure analysis with confidence bounds
  - Reliability function over time
  - MCF rate with empirical validation

## Installation

```bash
# Clone the repository
git clone https://github.com/foadkrmn/degradation-reliability-analysis.git
cd degradation-reliability-analysis

# Create and activate virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
from wiener_mcf_reliability import SimulationConfig, ReliabilityAnalysis

# Create configuration
config = SimulationConfig(
    n_components=100,    # Number of components to simulate
    n_timesteps=100,     # Number of time steps
    drift=0.2,          # Mean degradation rate
    volatility=0.1,     # Degradation volatility
    failure_threshold=10.0  # Failure threshold
)

# Run analysis
analysis = ReliabilityAnalysis(config)
results = analysis.run_analysis()
```

## Example Output

The analysis generates interactive plots showing:
1. Individual degradation paths and their distribution
2. Fleet-wide failure analysis with confidence bounds
3. System reliability over time
4. MCF rate with empirical validation

## Configuration Parameters

- `n_components`: Number of components to simulate
- `n_timesteps`: Number of time steps in simulation
- `dt`: Time step size
- `drift`: Mean degradation rate
- `volatility`: Standard deviation of degradation increments
- `failure_threshold`: Threshold value for failure
- `random_seed`: Seed for reproducibility
- `save_plots`: Whether to save generated plots
- `output_dir`: Directory for saving results

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)
