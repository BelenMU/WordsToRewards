import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from config import HIDDEN1, HIDDEN2, LR, GAMMA, EPSILON
import pickle

import os
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
from PPO import _edit_video

class Policy_Network(nn.Module):
    """Parametrized Policy Network."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes a neural network that estimates the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """
        super().__init__()

        hidden_space1 = HIDDEN1
        hidden_space2 = HIDDEN2

        # Shared Network
        self.shared_net = nn.Sequential(
            nn.Linear(obs_space_dims, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh(),
        )

        # Policy Mean specific Linear Layer
        self.policy_mean_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

        # Policy Std Dev specific Linear Layer
        self.policy_stddev_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Conditioned on the observation, returns the mean and standard deviation
         of a normal distribution from which an action is sampled from.

        Args:
            x: Observation from the environment

        Returns:
            action_means: predicted mean of the normal distribution
            action_stddevs: predicted standard deviation of the normal distribution
        """
        shared_features = self.shared_net(x.float())

        action_means = self.policy_mean_net(shared_features)
        action_stddevs = torch.log(
            1 + torch.exp(self.policy_stddev_net(shared_features))
        )

        return action_means, action_stddevs
    
class REINFORCE:
    """REINFORCE algorithm."""

    def __init__(self, obs_space_dims: int, action_space_dims: int):
        """Initializes an agent that learns a policy via REINFORCE algorithm [1]
        to solve the task at hand (Inverted Pendulum v4).

        Args:
            obs_space_dims: Dimension of the observation space
            action_space_dims: Dimension of the action space
        """

        # Hyperparameters
        self.learning_rate = LR  # Learning rate for policy optimization
        self.gamma = GAMMA  # Discount factor
        self.eps = EPSILON  # small number for mathematical stability

        self.probs = []  # Stores probability values of the sampled action
        self.rewards = []  # Stores the corresponding rewards

        self.net = Policy_Network(obs_space_dims, action_space_dims)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)

    def sample_action(self, state: np.ndarray) -> float:
        """Returns an action, conditioned on the policy and observation.

        Args:
            state: Observation from the environment

        Returns:
            action: Action to be performed
        """
        state = torch.tensor(np.array([state]))
        action_means, action_stddevs = self.net(state)

        # create a normal distribution from the predicted
        #   mean and standard deviation and sample an action
        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        action = distrib.sample()
        prob = distrib.log_prob(action)

        action = action.numpy()

        self.probs.append(prob)

        return action

    def update(self):
        """Updates the policy network's weights."""
        running_g = 0
        gs = []

        # Discounted return (backwards) - [::-1] will return an array in reverse
        for R in self.rewards[::-1]:
            running_g = R + self.gamma * running_g
            gs.insert(0, running_g)

        deltas = torch.tensor(gs)

        loss = 0
        # minimize -1 * prob * reward obtained
        for log_prob, delta in zip(self.probs, deltas):
            loss += log_prob.mean() * delta * (-1)

        # Update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Empty / zero out all episode-centric/related variables
        self.probs = []
        self.rewards = []
        
    def save(self, file_path):
        # First, save the neural network state dict
        torch.save(self.net.state_dict(), file_path + '_net.pth')

        # Now save the rest of the object's state excluding the non-pickleable parts
        reinforce_dict = self.__dict__.copy()
        del reinforce_dict['net']  # Remove the non-pickleable net object
        del reinforce_dict['optimizer']  # Remove the non-pickleable optimizer object

        with open(file_path + '.pkl', 'wb') as f:
            pickle.dump(reinforce_dict, f)

def load_reinforce_object(file_path, obs_space_dims, action_space_dims):
    # Create a new instance of REINFORCE
    reinforce = REINFORCE(obs_space_dims, action_space_dims)

    # Load neural network parameters
    net_state_dict = torch.load(file_path + '_net.pth')
    reinforce.net.load_state_dict(net_state_dict)

    # Load the rest of the object's state
    with open(file_path + '.pkl', 'rb') as f:
        reinforce_dict = pickle.load(f)

    for key, value in reinforce_dict.items():
        setattr(reinforce, key, value)

    # Reinitialize the optimizer because it is non-pickleable
    reinforce.optimizer = torch.optim.AdamW(reinforce.net.parameters(), lr=reinforce.learning_rate)

    return reinforce

def get_goal():
    while True:
        goal = np.random.uniform(low=-0.2, high=0.2, size=2)
        if np.linalg.norm(goal) < 0.2:
            return goal
        
def record_video_policy_goal(env, policy_module, video_directory, video_name, goal_init, num_episodes=2):
    """Evaluate a given policy and save a video of the policy_module performance

    Args:
        env: The environment to evaluate the policy on.
        policy_module: The policy to be evaluated.
        video_directory (str): The directory to save the videos.
        video_name (str): The name for the video file.
        num_episodes (int): The number of episodes to evaluate and record.

    Returns:
        str: A formatted string summarizing the evaluation results.
    """
    outputs_to_save = []
    
    # If the directory doesn't exist, create it
    if not os.path.exists(video_directory):
        os.makedirs(video_directory)
    
    # Perform the episodes and record
    for episode in range(num_episodes):
        episode_tensors = []
        print("episode ", episode+1, "out of ", num_episodes)
        # Initialize the episode
        env.reset()
        # Reset Position and velocity but mantain target
        qpos = (
            np.random.uniform(low=-0.1, high=0.1, size=env.model.nq) + env.init_qpos
        )
        env.goal = goal_init
        qpos[-2:] = goal_init
        qvel = env.init_qvel + np.random.uniform(
            low=-0.005, high=0.005, size=env.model.nv
        )
        qvel[-2:] = 0
        env.set_state(qpos, qvel)
        
        theta = env.data.qpos.flat[:2]
        obs = np.concatenate(
            [
                np.cos(theta),
                np.sin(theta),
                env.data.qpos.flat[2:],
                env.data.qvel.flat[:2],
                env.get_body_com("fingertip") - env.get_body_com("target"),
            ]
        )
        done = False
        video_file_path = os.path.join(video_directory, f"{video_name}_episode_{episode+1}.mp4")
        video_recorder = VideoRecorder(env, path=video_file_path)
        
        while not done:
            action = policy_module.sample_action(obs)
            obs, reward_true, terminated, truncated, info = env.step(action)
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            
            done = terminated or truncated
            
            video_recorder.capture_frame()
            episode_tensors.append(torch.cat((torch.from_numpy(action), obs_tensor), dim=0))
            #episode_tensors.append(obs_tensor)
                 
        video_recorder.close()
        outputs_to_save.append(episode_tensors)
        _edit_video(video_file_path)

    env.close()
    try:
        torch.save(outputs_to_save, os.path.join(video_directory,f"{video_name}_data.pth"))
        print("saved")
    except:
        print("It didn't save the trajectory")
    return outputs_to_save

def record_video_policy_goal_noisy(env, policy_module, previous_policy, obs_space_dims, action_space_dims, video_directory, video_name, goal_init, num_episodes=2):
    """Evaluate a given policy and save a video of the policy_module performance

    Args:
        env: The environment to evaluate the policy on.
        policy_module: The policy to be evaluated.
        video_directory (str): The directory to save the videos.
        video_name (str): The name for the video file.
        num_episodes (int): The number of episodes to evaluate and record.

    Returns:
        str: A formatted string summarizing the evaluation results.
    """
    outputs_to_save = []
    
    # If the directory doesn't exist, create it
    if not os.path.exists(video_directory):
        os.makedirs(video_directory)
    
    # Perform the episodes and record
    for episode in range(num_episodes):
        episode_tensors = []
        print("episode ", episode+1, "out of ", num_episodes)
        # Initialize the episode
        env.reset()
        # Reset Position and velocity but mantain target
        qpos = (
            np.random.uniform(low=-0.1, high=0.1, size=env.model.nq) + env.init_qpos
        )
        env.goal = goal_init
        qpos[-2:] = goal_init
        qvel = env.init_qvel + np.random.uniform(
            low=-0.005, high=0.005, size=env.model.nv
        )
        qvel[-2:] = 0
        env.set_state(qpos, qvel)
        
        theta = env.data.qpos.flat[:2]
        obs = np.concatenate(
            [
                np.cos(theta),
                np.sin(theta),
                env.data.qpos.flat[2:],
                env.data.qvel.flat[:2],
                env.get_body_com("fingertip") - env.get_body_com("target"),
            ]
        )
        done = False
        video_file_path = os.path.join(video_directory, f"{video_name}_episode_{episode+1}.mp4")
        video_recorder = VideoRecorder(env, path=video_file_path)
        if episode % 2 == 1:
            policy = policy_module
        elif episode == 2:
            policy = previous_policy
        else:
            policy = REINFORCE(obs_space_dims, action_space_dims)
        
        while not done:
            action = policy.sample_action(obs)
            obs, reward_true, terminated, truncated, info = env.step(action)
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            
            done = terminated or truncated
            
            video_recorder.capture_frame()
            episode_tensors.append(torch.cat((torch.from_numpy(action), obs_tensor), dim=0))
            #episode_tensors.append(obs_tensor)
                 
        video_recorder.close()
        outputs_to_save.append(episode_tensors)
        _edit_video(video_file_path)

    env.close()
    try:
        torch.save(outputs_to_save, os.path.join(video_directory,f"{video_name}_data.pth"))
        print("saved")
    except:
        print("It didn't save the trajectory")
    return outputs_to_save

def record_video_given_trajectory(env, top_action, video_directory, video_name, goal_init, num_pairs):
    """Evaluate a given policy and save a video of the policy_module performance

    Args:
        env: The environment to evaluate on.
        top_action: Actions to recreate.
        video_directory (str): The directory to save the videos.
        video_name (str): The name for the video file.
        goal_init: Position of the target.
        num_pairs (int): The number of pairs of episodes to record.
    """
    outputs_to_save = []
    
    # If the directory doesn't exist, create it
    if not os.path.exists(video_directory):
        os.makedirs(video_directory)
    
    # Perform the episodes and record
    for pair in range(num_pairs):
        pair_action = top_action[pair]
        for ii in range(2): # Each video of the pair
            traj_action = top_action[pair][ii]
            episode_tensors = []
            print("episode ", 2*pair + ii +1, "out of ", num_pairs*2)
            # Initialize the episode
            env.reset()
            # Reset Position and velocity but mantain target
            qpos = env.init_qpos
            env.goal = goal_init
            qpos[-2:] = goal_init
            qvel = env.init_qvel
            qvel[-2:] = 0
            env.set_state(qpos, qvel)

            theta = env.data.qpos.flat[:2]
            obs = np.concatenate(
                [
                    np.cos(theta),
                    np.sin(theta),
                    env.data.qpos.flat[2:],
                    env.data.qvel.flat[:2],
                    env.get_body_com("fingertip") - env.get_body_com("target"),
                ]
            )
            done = False
            video_file_path = os.path.join(video_directory, f"{video_name}_episode_{2*pair + ii +1}.mp4")
            video_recorder = VideoRecorder(env, path=video_file_path)
            
            temp = 0
            while not done:
                action = traj_action[temp]
                obs, reward_true, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                video_recorder.capture_frame()
                outputs_to_save.append(torch.tensor(obs, dtype=torch.float32))

            video_recorder.close()
            _edit_video(video_file_path)

    env.close()
    try:
        torch.save(outputs_to_save, os.path.join(video_directory,f"{video_name}_data.pth"))
        print("saved")
    except:
        print("It didn't save the trajectory")
    return outputs_to_save