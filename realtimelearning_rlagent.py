import gym
from stable_baselines3 import PPO
import numpy as np
from control_drone import TelloController  # Import the new controller
from gym import spaces
import os  # Import the 'os' module

# Define the filename for saving the RL agent model
MODEL_FILENAME = "drone_rl_full_eeg_agent"


class DroneControlEnv(gym.Env):
    def __init__(self):
        super(DroneControlEnv, self).__init__()
        # Continuous action space: forward/backward and left/right speed (-1 to 1)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)  # Gyro X, Gyro Y, 5 PCA components

        self.current_state = None
        self.drone_controller = TelloController()  # Create an instance of the controller
        self.drone_connected = False  # Flag to indicate if the drone is connected
        self.model = None  # RL model attribute

    def connect_drone(self):
        """Connect to the Tello drone."""
        if not self.drone_connected:
            self.drone_connected = self.drone_controller.connect()
            return self.drone_connected
        return True  # Already connected

    def reset(self):
        self.current_state = np.zeros(7)  # Gyro X, Gyro Y, 5 PCA components
        return self.current_state

    def step(self, action):
        # Scale actions to the range -100 to 100 (Tello speed range)
        forward_backward_speed = action[0] * 100  # Assuming action is a single value
        left_right_speed = action[1] * 100
        self.drone_controller.set_forward_backward_speed(forward_backward_speed)
        self.drone_controller.set_left_right_speed(left_right_speed)

        reward = 0  # Initialize reward

        # Simple reward based on action. Make sure drone is moving
        if forward_backward_speed > 0 or left_right_speed > 0:
            reward = 0.1

        done = False
        info = {}
        return self.current_state, reward, done, info  # Dummy for now until done condition

    def update_state(self, new_state):
        self.current_state = new_state

    def load_or_create_model(self):
        """Load a pre-trained model or create a new one if none exists."""
        if os.path.exists(f"{MODEL_FILENAME}.zip"):  # Correctly check for the existence of the .zip file
            try:
                self.model = PPO.load(MODEL_FILENAME, env=self)  # Load the existing model and set the environment
                print("Loaded existing model")
            except Exception as e:
                print(f"Error loading existing model: {e}. Creating a new model instead.")
                self.model = PPO("MlpPolicy", self, verbose=1)  # Create a new model
                print("Created new model")
        else:
            self.model = PPO("MlpPolicy", self, verbose=1)  # Create a new model
            print("Created new model")

        return self.model

    def train_step(self, eeg_data):
        state = np.array(eeg_data)
        self.update_state(state)

        action, _states = self.model.predict(state, deterministic=False)

        obs, reward, done, info = self.step(action)
        self.model.learn(total_timesteps=1, reset_num_timesteps=False)

        return action

    def save_model(self):
        """Saves the current model."""
        if self.model is not None:
            self.model.save(MODEL_FILENAME)  # Save the model using the attribute
            print("Saved model")
        else:
            print("No model to save.")


# Initialize the environment and RL model
env = DroneControlEnv()
model = env.load_or_create_model()  # Load or create the RL model

# train_agent is now a method of the DroneControlEnv class

def train_agent(eeg_data):
    return env.train_step(eeg_data)


# Save the trained model
def save_model():
    env.save_model()


# Load a trained model
def load_model():
    global model
    model = PPO.load(MODEL_FILENAME)
