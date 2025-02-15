import torch
import numpy as np

def state_to_obs(state):
    # Inverse mapping of colors to integers
    color_inverse_mapping = {'W': 0, 'R': 1, 'B': 2, 'O': 3, 'G': 4, 'Y': 5}
    
    # Function to flatten 3x3 face to 1D array
    def flatten_face(face):
        return [color_inverse_mapping[color] for row in face for color in row]
    
    # Reconstruct the observation from the state
    observation = flatten_face(state)
    
    return observation

def answer_to_training_data(answer):
    all_obs = []
    all_rewards = []
    for state_data in answer['states']:
        obs = state_to_obs(state_data['state'])
        reward = state_data['reward']
        all_obs.append(obs)
        all_rewards.append(reward)
    x_train = torch.tensor(all_obs, dtype=torch.float32)
    y_train = torch.tensor((np.array(all_rewards)+1)/2, dtype=torch.float32).view(-1, 1)
    return x_train, y_train