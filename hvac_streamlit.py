import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from stable_baselines3 import DQN
import gymnasium as gym
from gymnasium import spaces
import random

# Define the AdvancedBuildingEnv class (unchanged)
class AdvancedBuildingEnv(gym.Env):
    def __init__(self):
        super(AdvancedBuildingEnv, self).__init__()
        self.observation_space = spaces.Box(
            low=np.array([10, -10, 15, 0]),
            high=np.array([30, 40, 35, 24]),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(5)
        self.indoor_temp = 22.0
        self.outdoor_temp = random.uniform(0, 35)
        self.storage_temp = 25.0
        self.energy_usage = 0.0
        self.time_step = 0
        self.max_steps = 144
        self.time_of_day = 0.0
        self.thermal_drift_rate = 0.05
        self.storage_capacity = 10.0
        self.storage_charge_rate = 0.5
        self.storage_discharge_rate = 0.8

    def step(self, action):
        energy_cost = 0.0
        if action == 0:  # Cool
            self.indoor_temp -= 1.0
            energy_cost = 1.5
        elif action == 2:  # Heat
            self.indoor_temp += 1.0
            energy_cost = 2.0
        elif action == 3:  # Charge storage
            if self.storage_temp < 35.0:
                self.storage_temp = min(self.storage_temp + self.storage_charge_rate, 35.0)
                energy_cost = 1.0
        elif action == 4:  # Discharge storage
            if self.storage_temp > 15.0:
                self.storage_temp -= self.storage_discharge_rate
                self.indoor_temp += 0.5
                energy_cost = 0.2
        self.energy_usage += energy_cost
        self.indoor_temp += (self.outdoor_temp - self.indoor_temp) * self.thermal_drift_rate
        self.storage_temp += (self.outdoor_temp - self.storage_temp) * 0.01
        self.time_of_day = (self.time_step * 10 / 60) % 24
        self.time_step += 1
        self.outdoor_temp = 20 + 10 * np.sin(2 * np.pi * self.time_of_day / 24)
        comfort_penalty = abs(self.indoor_temp - 22)
        energy_penalty = energy_cost * 0.5
        storage_bonus = 0.1 * (self.storage_temp - 15) if action in [3, 4] else 0
        reward = -comfort_penalty - energy_penalty + storage_bonus
        done = self.time_step >= self.max_steps
        return (
            np.array([self.indoor_temp, self.outdoor_temp, self.storage_temp, self.time_of_day], dtype=np.float32),
            reward,
            done,
            False,
            {"energy_cost": energy_cost}
        )

    def reset(self, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
        self.indoor_temp = 22.0
        self.outdoor_temp = random.uniform(0, 35)
        self.storage_temp = 25.0
        self.energy_usage = 0.0
        self.time_step = 0
        self.time_of_day = 0.0
        return np.array([self.indoor_temp, self.outdoor_temp, self.storage_temp, self.time_of_day], dtype=np.float32), {}

# Function to run simulation and collect data
def run_simulation(model, env, steps=144):
    obs, _ = env.reset()
    temperatures, energy_usage, storage_temps, time_steps = [], [], [], []
    for _ in range(steps):
        action, _ = model.predict(obs)
        obs, reward, done, _, info = env.step(action)
        temperatures.append(obs[0])
        energy_usage.append(env.energy_usage)
        storage_temps.append(obs[2])
        time_steps.append(env.time_of_day)
        if done:
            break
    return temperatures, energy_usage, storage_temps, time_steps

# Function to create 3D building model with temperature visualization
def create_3d_building_plot(temperatures, time_steps):
    # Define building vertices (simple rectangular prism)
    x = [0, 0, 2, 2, 0, 0, 2, 2]  # Width
    y = [0, 2, 2, 0, 0, 2, 2, 0]  # Depth
    z = [0, 0, 0, 0, 1, 1, 1, 1]  # Height

    # Define triangular faces for Mesh3d
    i = [0, 0, 0, 0, 4, 4, 4, 4, 1, 1, 2, 2]
    j = [1, 2, 4, 7, 5, 6, 7, 3, 5, 2, 6, 3]
    k = [2, 3, 7, 3, 6, 7, 3, 5, 2, 6, 3, 6]

    # Create building mesh
    building = go.Mesh3d(
        x=x, y=y, z=z,
        i=i, j=j, k=k,
        color='lightblue',
        opacity=0.5,
        name='Building'
    )

    # Temperature as a marker inside the building
    temp_normalized = [(t - 10) / 20 for t in temperatures]  # Normalize 10-30°C to 0-1
    temp_scatter = go.Scatter3d(
        x=[1] * len(temperatures),  # Center of building
        y=[1] * len(temperatures),
        z=temp_normalized,  # Height represents temperature
        mode='markers',
        marker=dict(
            size=5,
            color=temperatures,
            colorscale='RdBu',  # Red (hot) to Blue (cold)
            cmin=10,
            cmax=30,
            colorbar=dict(title="Indoor Temp (°C)"),
            showscale=True
        ),
        name='Indoor Temperature'
    )

    # Layout for the 3D plot
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='Width', range=[-1, 3]),
            yaxis=dict(title='Depth', range=[-1, 3]),
            zaxis=dict(title='Height', range=[0, 1.5]),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.5)
        ),
        title="3D Building Model with Temperature Dynamics",
        height=600
    )

    fig = go.Figure(data=[building, temp_scatter], layout=layout)
    return fig

# Streamlit app
def main():
    st.title("HVAC Control System with 3D Building Visualization")
    st.write("Simulate and visualize HVAC control in a 3D building using a trained DQN model.")

    # Load the trained model
    model_path = "dqn_hvac_control.zip"
    try:
        model = DQN.load(model_path)
        st.success(f"Model loaded successfully from {model_path}")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # Create environment
    env = AdvancedBuildingEnv()

    # User inputs
    st.sidebar.header("Simulation Parameters")
    sim_steps = st.sidebar.slider("Simulation Steps (10-min intervals)", 1, 144, 144)
    initial_indoor_temp = st.sidebar.slider("Initial Indoor Temperature (°C)", 10.0, 30.0, 22.0)
    initial_storage_temp = st.sidebar.slider("Initial Storage Temperature (°C)", 15.0, 35.0, 25.0)

    # Override initial conditions
    env.indoor_temp = initial_indoor_temp
    env.storage_temp = initial_storage_temp

    # Run simulation button
    if st.button("Run Simulation"):
        with st.spinner("Running simulation..."):
            temperatures, energy_usage, storage_temps, time_steps = run_simulation(model, env, sim_steps)

        # 3D Building Visualization with Plotly
        st.subheader("3D Building Temperature Dynamics")
        fig_3d = create_3d_building_plot(temperatures, time_steps)
        st.plotly_chart(fig_3d, use_container_width=True)

        # 2D Plots with Matplotlib
        st.subheader("Detailed Simulation Results")
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        ax1.plot(temperatures, label="Indoor Temp (°C)")
        ax1.axhline(y=22, color='r', linestyle='--', label="Target (22°C)")
        ax1.set_xlabel("Time Step (10 min)")
        ax1.set_ylabel("Temperature (°C)")
        ax1.legend()
        ax1.set_title("Temperature Regulation")

        ax2.plot(energy_usage, label="Cumulative Energy Usage", color='g')
        ax2.set_xlabel("Time Step (10 min)")
        ax2.set_ylabel("Energy Consumption")
        ax2.legend()
        ax2.set_title("Energy Usage")

        ax3.plot(storage_temps, label="Storage Temp (°C)", color='b')
        ax3.set_xlabel("Time Step (10 min)")
        ax3.set_ylabel("Temperature (°C)")
        ax3.legend()
        ax3.set_title("Thermal Storage Usage")

        plt.tight_layout()
        st.pyplot(fig)

    # Display current environment state
    st.subheader("Current Environment State")
    st.write(f"Indoor Temperature: {env.indoor_temp:.2f}°C")
    st.write(f"Outdoor Temperature: {env.outdoor_temp:.2f}°C")
    st.write(f"Storage Temperature: {env.storage_temp:.2f}°C")
    st.write(f"Total Energy Usage: {env.energy_usage:.2f}")

if __name__ == "__main__":
    main()