import torch
import torch
from itertools import combinations
import numpy as np
import gym
from gym_rubiks_cube.envs.rubiksCubeEnv import RubiksCubeEnv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sample_random_trajectories(num_episodes, env):
    trajectories_obs = []
    trajectories_actions = []
    for episode in range(num_episodes):

        obs, info = env.reset()
        obs_tensor = torch.tensor(obs[9:18], dtype=torch.float32).unsqueeze(0)
        done = False
        truncated = False
        observations = [obs_tensor]
        actions = []
        
        while not truncated:
            action = np.random.randint(0, 19)
            actions.append(action)
            #actions = actions + [action]
            obs, reward, done, truncated, info = env.step(action)
            obs_tensor = torch.tensor(obs[9:18], dtype=torch.float32).unsqueeze(0)
            observations.append(obs_tensor)
        
        env.close()
        trajectories_actions.append(actions)
        trajectories_obs.append(observations)
    return trajectories_obs, trajectories_actions

def uncertainty_sampling(trajectories_obs, ensemble, n_pairs, trajectories_action):
    # Compute the predictions
    pair_variances = {}
    for i, (traj1, traj2) in enumerate(combinations(trajectories_obs, 2)):
        # Convert trajectories to tensors if they aren't already
        tensor1 = torch.stack(traj1).to(device)
        tensor2 = torch.stack(traj2).to(device)

        # Get predictions from all models in the ensemble for this pair
        predictions = torch.stack([model(tensor1, tensor2) for model in ensemble])

        # Calculate the variance
        variance = torch.var(predictions, dim=0).mean().item()

        # Store the variance with the pair index as the key
        pair_variances[i] = variance

    # Get the pairs with the highest variance
    sorted_pairs = sorted(pair_variances.items(), key=lambda x: x[1], reverse=True)

    # Now, sorted_pairs contains pairs indexes sorted by variance, we take the top n_pairs
    max_variance_pair_indexes = [info[0] for info in sorted_pairs[:n_pairs]]

    # Extract the n_pairs pairs of trajectories with the maximum variance
    top_pairs = [list(combinations(trajectories_obs, 2))[index] for index in max_variance_pair_indexes]
    
    # Get the corresponding action pairs
    # Extract indices for the top n_pairs pairs
    top_indices = [idx for idx, _ in sorted_pairs[:n_pairs]]
    
    # Get the corresponding action trajectory indices (since they should match the observation trajectory indices)
    top_action_indices = []
    for index in top_indices:
        # Retrieve the original indices within trajectories_obs that generate this pair
        original_indices = list(combinations(range(len(trajectories_obs)), 2))[index]
        print(original_indices)
        top_action_indices.append(original_indices)
    top_trajectory_actions = [(trajectories_action[i], trajectories_action[j]) for i, j in top_action_indices]
    
    return top_pairs, top_trajectory_actions