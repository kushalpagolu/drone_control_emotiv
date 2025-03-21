import gym
from stable_baselines3 import PPO
import numpy as np

class DroneControlEnv(gym.Env):
    def __init__(self):
        super(DroneControlEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(4)  # 4 possible actions: forward, backward, left, right
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)  # Gyro X, Gyro Y, 5 PCA components
        self.current_state = None

    def reset(self):
        self.current_state = np.zeros(7)  # Gyro X, Gyro Y, 5 PCA components
        return self.current_state

    def step(self, action):
        # Here you would map the action to a reward based on how well it matches your intention
        # For now, we'll use a simple reward mechanism
        reward = 1 if action == np.argmax(self.current_state[:4]) else 0  # Reward if action matches highest activation
        done = False
        info = {}
        # Execute the action on the drone
        #control_drone(action) #Commented for Debugging purposes

        return self.current_state, reward, done, info

    def update_state(self, new_state):
        self.current_state = new_state

# Initialize the environment and RL model
env = DroneControlEnv()
model = PPO('MlpPolicy', env, verbose=1)

def train_agent(eeg_data):
    # Preprocess EEG data if necessary
    state = np.array(eeg_data)
    
    # Update the environment with the new state
    env.update_state(state)
    
    # Get the model's action prediction
    action, _states = model.predict(state)
    
    # Perform a training step
    obs, reward, done, info = env.step(action)
    model.learn(total_timesteps=1, reset_num_timesteps=False)
    
    return action

# Save the trained model
def save_model():
    model.save("drone_rl_full_eeg_agent")

# Load a trained model
def load_model():
    global model
    model = PPO.load("drone_rl_full_eeg_agent")
