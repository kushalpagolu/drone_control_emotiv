# drone_control_emotiv
A RL agent trained in real time to control DJI Tello drone with a streamer of 14 channels of EEG data from Emotiv Epoch X headset.


# EEG-Based Drone Control Project: Step-by-Step Guide

This guide outlines the steps to train an RL agent using EEG data from the Emotiv EPOC+ headset and control a Tello drone.

## I. Project Overview
This project aims to control a Tello drone using real-time EEG data streamed from an Emotiv EPOC+ headset. The EEG data, along with gyroscope data, is used to train a Reinforcement Learning (RL) agent that learns to map brain signals to drone control commands.

## II. File Summaries
Here's a breakdown of each code file and its role in the project:

### `emotiv_streamer_rl.py`
**Purpose:** This file connects to the Emotiv EPOC+ headset, reads EEG and gyroscope data packets, decrypts the data, performs preprocessing (band power calculation and PCA), and streams the processed data to the RL agent.



**Key Components:**
- `EmotivStreamer` class: Handles device connection, data reading, decryption, preprocessing, and streaming.
- `connect()`: Establishes a connection with the Emotiv headset.
- `read_packet()`: Reads and decrypts a data packet from the headset.
- `preprocess_eeg_data()`: Calculates band power features and applies PCA for dimensionality reduction.
- `stream_data_to_rl_agent()`: Sends the processed data to the RL agent via UDP socket.

### `realtimelearning_rlagent.py`
**Purpose:** Defines the RL environment, initializes the RL agent, and trains the agent using the incoming EEG data.

**Key Components:**
- `DroneControlEnv` class: Defines the environment for the RL agent.
- `reset()`: Resets the environment to a starting state.
- `step()`: Takes an action, calculates the reward, and returns the next state.
- `train_agent()`: Trains the RL agent using the EEG data.

### `control_drone.py`
**Purpose:** Controls the Tello drone based on the actions predicted by the RL agent.

**Key Components:**
- `control_drone()`: Connects to the Tello drone and sends control commands (forward, backward, left, right).

### `visualizer_epoch.py`
**Purpose:** The main entry point of the project. Connects to the Emotiv headset, starts the data visualization, trains the RL agent (if drone control is enabled), and saves the collected data.

**Key Components:**
- `start_data_collection()`: Collects data from the Emotiv headset, updates the real-time EEG visualizer, and trains the RL agent.
- `save_data_continuously()`: Saves the collected data to an Excel file.

### `realtime_visualizer_2D.py`
**Purpose:** Defines the real-time EEG visualizer that displays EEG signals and gyroscope data.

![Figure_1](https://github.com/user-attachments/assets/a28eb3d3-1e4c-4d65-bc22-a56da1793789)


**Key Components:**
- `RealtimeEEGVisualizer` class: Creates and updates the real-time EEG plots.
- `update()`: Updates the EEG and gyroscope plots with new data.

### `kalman_filter.py`
**Purpose:** Implements a Kalman filter to smooth out noisy gyroscope data.

**Key Components:**
- `KalmanFilter` class: Applies Kalman filtering to smooth out noisy gyroscope data.
- `update()`: Updates the filter with a new measurement.

## III. Prerequisites
Before you begin, ensure you have the following:

### **Hardware:**
- Emotiv EPOC+ headset
- Tello drone
- Computer with sufficient processing power

### **Software:**
- Python 3.7+
- Required Python packages (install using `pip install -r requirements.txt`):
  - `hidapi`
  - `pycryptodome`
  - `numpy`
  - `scipy`
  - `pandas`
  - `matplotlib`
  - `stable-baselines3`
  - `gym`
  - `djitellopy`
  - `scikit-learn` (for PCA)

### **Tello Drone Setup:**
- Download and install the Tello app on your smartphone.
- Connect your computer to the Tello drone's Wi-Fi network.

## IV. Step-by-Step Instructions
Follow these steps to train the RL agent and control the Tello drone:

### **1. Connect Emotiv EPOC+ Headset:**
- Ensure the Emotiv EPOC+ headset is properly connected to your computer via USB.
- Verify that the headset is turned on and the sensors are making good contact with your scalp.

### **2. Configure IP Addresses and Ports:**
- In `control_drone.py`, ensure you have the correct IP address of the Tello drone.

### **3. Run `visualizer_epoch.py`:**
```bash
python visualizer_epoch.py
```
- The script will start streaming data from the Emotiv headset and visualizing it in real-time.


### ** 4. Train the RL Agent (and Control the Drone)**
- `visualizer_epoch.py` calls `emotiv_streamer_rl.py`, which preprocesses EEG data and sends it to `realtimelearning_rlagent.py`.
- `realtimelearning_rlagent.py` trains the RL agent.

### ** 5. Enable Drone Control in RL Training**
Uncomment the `control_drone(action)` line in `DroneControlEnv.step()`:

```python
def step(self, action):
    reward = 1 if action == np.argmax(self.current_state[:4]) else 0  # Reward if action matches highest activation
    done = False
    info = {}

    # Execute the action on the drone
    control_drone(action)  # <--- UNCOMMENT THIS LINE TO CONTROL THE DRONE

    return self.current_state, reward, done, info

```

### ** 6. Save the Trained Model: **

- Once the training is complete, the trained RL agent will be saved to a file named drone_rl_full_eeg_agent.

## V. Code Execution Flow
visualizer_epoch.py: This script is the main entry point. It initializes the Emotiv streamer, visualizer, and Kalman filters. It then enters a loop to continuously read data from the Emotiv headset and update the visualizer. It also calls the RL agent's training function with the preprocessed EEG data.

emotiv_streamer_rl.py: This script connects to the Emotiv headset and continuously reads EEG data packets. For each packet, it preprocesses the EEG data (calculates band power features and applies PCA) and sends the processed data to the RL agent.

realtimelearning_rlagent.py: This script receives the preprocessed EEG data from the Emotiv streamer. It then uses this data to train the RL agent. The RL agent learns to map EEG patterns to drone control commands.

control_drone.py: This script receives the action predicted by the RL agent and sends the corresponding command to the Tello drone. The drone then executes the command (e.g., move forward, backward, left, right).

## VI. Important Notes
**Safety First:** Always test the drone control in a safe, open area to prevent accidents.

**EEG Data Quality: **Ensure that the EEG sensors are making good contact with your scalp to obtain high-quality EEG data.

**Training Time: **The training time for the RL agent may vary depending on the complexity of the task and the quality of the EEG data.

**Troubleshooting: **If you encounter any issues, check the console output for error messages and refer to the troubleshooting tips in this guide.

**Hyperparameter Tuning: **Experiment with different RL algorithms and hyperparameters to optimize the performance of the agent. You might need to adjust the learning rate, discount factor, and exploration rate.

**Reward Shaping: **The design of the reward function is crucial for the success of the RL agent. Experiment with different reward functions to encourage the desired drone behaviors.

_By following these steps and referring to the file summaries, you should be able to train your RL agent and control the Tello drone using EEG data from the Emotiv EPOC+ headset. Remember to prioritize safety and experiment with different configurations to optimize performance. _

