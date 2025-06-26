# EcoThermoStat - RL-Based HVAC Control Optimization

This project implements a reinforcement learning (RL)-based solution to optimize Heating, Ventilation, and Air Conditioning (HVAC) control in a simulated building environment. Using a Deep Q-Network (DQN) model, it balances indoor comfort (target 22°C), energy efficiency, and thermal storage utilization, addressing sustainability challenges in smart buildings.

## Project Overview

- **Objective**: Develop an intelligent HVAC control system that adapts to dynamic conditions, minimizes energy consumption (130.80 units achieved), and maintains a 90.3% comfort success rate within 20–24°C over 144 timesteps.
- **Approach**: Employs RL with a pre-trained DQN to learn optimal control actions (cool, heat, charge, discharge) in the `AdvancedBuildingEnv` simulation.
- **Tools**: Python, NumPy, Matplotlib, Plotly, Stable-Baselines3, and Streamlit for visualization and interaction.
- **Inspiration**: Builds on research from 2007 (*Reinforcement Learning for Energy Conservation*), 2023 (*Safe RL Architectures*), and 2025 (*Sustainable Energy Survey*).

## Features

- Simulates a single-zone building with HVAC and thermal storage over 24 hours (144 timesteps).
- Dynamically adjusts indoor temperature based on outdoor conditions (sinusoidal 20 ± 10°C).
- Provides real-time 3D (Plotly) and 2D (Matplotlib) visualizations of temperature, energy, and storage trends.
- Interactive Streamlit interface for parameter configuration and result display.

## Getting Started

### Prerequisites

- Python 3.8+
- Required libraries:
  - `numpy`
  - `matplotlib`
  - `plotly`
  - `streamlit`
  - `stable-baselines3`
  - `gym`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/dhanusree15/EcoThermostat.git
   cd Ecothermostat

2.Install dependencies:
  ```bash
pip install -r requirements.txt
 ```
3. Ensure the pre-trained DQN model (dqn_hvac_control.zip) is in the project directory
4. Usage
        - Run the Streamlit app
        - Configure simulation parameters (e.g., initial temperatures, steps) via the interface.
        - View real-time 3D and 2D plots showing temperature regulation, energy usage, and storage utilization.

```bash
   streamlit run hvac_streamlit.py
```
     
## Results
   - **Comfort**: Achieved 90.3% success rate, maintaining 20–24°C, with stabilization around 21–22°C.
   - **Energy**: Total consumption of 130.80 units, with efficient adaptation to outdoor temperature drops (e.g., 34.84°C to 19.56°C).
   - **Storage**: Temperature decreased from 25°C to 13.84°C, indicating effective discharge for energy savings.
   - Visual outputs confirm the system’s ability to balance comfort and efficiency.

## Future Work
  - Integrate safety constraints to prevent equipment overload.
  - Extend to multi-agent coordination for multiple building zones.
  - Incorporate real-time weather data for predictive control.
  - Validate in real-world settings beyond simulation.


## Acknowledgments
Inspired by research from:
   - Dalamagkidis et al. (2007) – Reinforcement Learning for Energy Conservation.
   - McClement et al. (2023) – Safe RL Architectures
   - Ponse et al. (2025) – Sustainable Energy Survey
