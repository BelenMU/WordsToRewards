import numpy as np
import random
from scipy.stats import beta
import spacy
from spacy import displacy 
from tqdm import tqdm # For progress bar
import copy
from transformers import pipeline
import os
import pickle

from science.draw_map import map_reward_estimation

class GridEnvironment:
    def __init__(self, grid_width, grid_height, num_steps, start):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.num_steps = num_steps # Number of steps per trajectory
        # Define the starting position
        self.start = start
        self.current_position = start

    # Function to reset the environment and start a new episode
    def reset(self):
        self.current_position = self.start
        return self.current_position

    # Function to take an action
    def step(self, action):
        x, y = self.current_position
        # action 0 means move up
        if action == 0 and y < self.grid_height - 1:
            y += 1
        # action 1 means move right
        elif action == 1 and x < self.grid_width - 1:
            x += 1
        # new state
        self.current_position = (x, y)

        return self.current_position
    
class QLearningAgent_Bernoulli:
    def __init__(self, env, gamma=0.9, delta=0.85, max_lookahead=10, \
                 alpha_init = 0.5, beta_init = 0.5, scale = 1):
        self.env = env
        self.alpha = alpha_init*np.ones(self.env.grid_width*self.env.grid_height)
        self.beta = beta_init*np.ones(self.env.grid_width*self.env.grid_height)
        self.gamma = gamma # Discount for future
        self.delta = delta # Percentil in UCB
        self.max_lookahead = max_lookahead
        self.scale = scale
        
        self.exp_human_feedback = []
        self.exp_trajectory = []
        self.exp_reward = []
        self.exp_certainty = []
        self.exp_alpha = [self.alpha]
        self.exp_beta = [self.beta]
        
    def save_experiment(self, name_experiment, name_human, date, opttrajectory, road):
        variables_to_save = {
                "human_feedback": self.exp_human_feedback,
                "trajectory": self.exp_trajectory,
                "reward": self.exp_reward,
                "certainty": self.exp_certainty,
                "alpha": self.exp_alpha,
                "beta": self.exp_beta,
                "learned_trajectory": opttrajectory,
                "road": road,
            }
        name_to_save = "./human_experiments/" + date + "_"  + name_human + "_" +name_experiment
        with open(name_to_save, "wb") as file:
            pickle.dump(variables_to_save, file)

    def state_to_index(self, state):
        return state[0] * self.env.grid_height + state[1]

    def choose_action(self, state, step):
         # Compute the optimistic estimate of the rewards
        rewards_upper_bound = beta.ppf(self.delta, self.alpha, self.beta)
        # Remaining steps for the episode (limited to either max_lookahead or less if we are near the episode end)
        remaining_steps = min(self.env.num_steps - step, self.max_lookahead)
        # possible actions and resulting states
        actions = [0, 1]
        next_states = [(state[0], min(state[1] + 1, self.env.grid_height-1)), 
                       (min(state[0] + 1 , self.env.grid_width-1 ), state[1])]
        max_value = -float('inf')
        best_action = None
        # Evaluate each action
        for action, next_state in zip(actions, next_states):
            next_state_index = self.state_to_index(next_state)
            env_copy = copy.deepcopy(self.env)
            env_copy.current_position = next_state

            action_value = 0  # Total return value
            discount_factor = 1  # Initialize discount factor

            for _ in range(remaining_steps): # Greedy roll-outs
                action_value += discount_factor * rewards_upper_bound[next_state_index]
                next_options = [(next_state[0], min(next_state[1] + 1, self.env.grid_height-1)), 
                   (min(next_state[0] + 1 , self.env.grid_width-1 ), next_state[1])]
                r = [0, 0]
                counter = 0
                for option in next_options:
                    option_index = self.state_to_index(option)
                    r[counter] = rewards_upper_bound[option_index]
                    counter += 1
                next_action = np.argmax(r)  # best known action
                next_state = env_copy.step(next_action)
                next_state_index = self.state_to_index(next_state)
                discount_factor *= self.gamma  # Increase discount factor for future steps

            if action_value > max_value:
                max_value = action_value
                best_action = action
            elif action_value == max_value:
                if np.random.uniform(0, 1) < 0.5: # If the reward from both actions is equivalent, flip a coin
                    best_action = action

        return best_action
       

    def learn(self, episodes, reward_model, images, loc_landmarks, road, grid_width, \
              grid_height, car_init, pixel_landmarks, list_landmarks):
        reward_sums = []
        for i in range(episodes):
            # Initialize a new episode
            state = self.env.reset()
            reward_sum = 0
            trajectory_states = []
            # Construct num_steps trajectory
            for step in range(self.env.num_steps): 
                action = self.choose_action(state, step)
                new_state = self.env.step(action)
                trajectory_states.append(new_state)
                state = new_state
            # Query the reward model
            human_feedback, out_reward, out_certainty = reward_model(trajectory_states,\
                                                                     images, loc_landmarks, road, grid_width, \
                                                                     grid_height, car_init, pixel_landmarks, list_landmarks)
            
            # Update alpha and beta
            for ind_reward in out_reward:
                temp_cert = out_certainty[ind_reward]
                temp_reward = out_reward[ind_reward]
                for label_reward in range(len(temp_reward)):
                    if temp_reward[label_reward] == 'POS':
                        self.alpha[ind_reward] += temp_cert[label_reward]*self.scale
                        reward_sum += 1
                    else:
                        self.beta[ind_reward] += temp_cert[label_reward]*self.scale
                        reward_sum += -1
            reward_sums.append(reward_sum)
            
            map_reward_estimation(self.alpha, self.beta, grid_width, grid_height, road)
            # Safe experiment data            
            self.exp_human_feedback.append(human_feedback)
            self.exp_trajectory.append(trajectory_states)
            self.exp_reward.append(out_reward)
            self.exp_certainty.append(out_certainty)
            self.exp_alpha.append(self.alpha)
            self.exp_beta.append(self.beta)
        return reward_sums
    
                
    def get_optimal_trajectory(self):
        # Initialize variables
        optimal_trajectory = []
        state = self.env.reset()
        total_steps = self.env.num_steps
        # Go through all steps
        for current_step in range(total_steps):
            # Calculate lookahead steps
            lookahead_steps = min(10, total_steps - current_step)
            # Choose best action based on mean reward, considering lookahead steps
            best_action = self.choose_best_action_based_on_mean_reward(state, lookahead_steps)
            # Take step with the selected action
            new_state = self.env.step(best_action)
            # Append step to optimal trajectory
            optimal_trajectory.append(new_state)
            state = new_state
        return optimal_trajectory

    def choose_best_action_based_on_mean_reward(self, state, lookahead_steps):
        # Possible actions and next states
        actions = [0, 1]
        next_states = [(state[0], min(state[1] + 1, self.env.grid_height-1)), 
                       (min(state[0] + 1 , self.env.grid_width-1), state[1])]

        max_value = -float('inf')
        best_action = None

        # Evaluate each action
        for action, next_state in zip(actions, next_states):
            action_value = self.calculate_action_value_based_on_mean_reward(next_state, lookahead_steps)
            if action_value > max_value:
                max_value = action_value
                best_action = action

        return best_action

    def calculate_action_value_based_on_mean_reward(self, state, lookahead_steps):
        state_index = self.state_to_index(state)
        expected_reward = self.alpha / (self.alpha + self.beta)
        action_value = expected_reward[state_index]  # mean reward as state value

        # If still some lookahead steps, recursively calculate the action value of the next state
        if lookahead_steps > 1:
            best_next_action = self.choose_best_action_based_on_mean_reward(state, lookahead_steps - 1)
            next_state = (min(state[0] + best_next_action, self.env.grid_width-1), 
                          min(state[1] + 1 - best_next_action, self.env.grid_height-1))
            action_value += self.gamma * self.calculate_action_value_based_on_mean_reward(next_state, lookahead_steps - 1)

        return action_value
    
class QLearningAgent_Bernoulli_greedy(QLearningAgent_Bernoulli):
    def learn(self, episodes, reward_model, images, loc_landmarks, road, grid_width, \
              grid_height, car_init, pixel_landmarks, list_landmarks):
        reward_sums = []
        for i in range(episodes):
            # Initialize a new episode
            state = self.env.reset()
            reward_sum = 0
            # Greedily choose trajectory
            trajectory_states = self.get_optimal_trajectory()
            # Query the reward model
            human_feedback, out_reward, out_certainty = reward_model(trajectory_states,\
                                                                     images, loc_landmarks, road, grid_width, \
                                                                     grid_height, car_init, pixel_landmarks, list_landmarks)
            
            # Update alpha and beta
            for ind_reward in out_reward:
                temp_cert = out_certainty[ind_reward]
                temp_reward = out_reward[ind_reward]
                for label_reward in range(len(temp_reward)):
                    if temp_reward[label_reward] == 'POS':
                        self.alpha[ind_reward] += temp_cert[label_reward]*self.scale
                        reward_sum += 1
                    else:
                        self.beta[ind_reward] += temp_cert[label_reward]*self.scale
                        reward_sum += -1
            reward_sums.append(reward_sum)
            #print("alpha: ", self.alpha)
            #print("beta: ", self.beta)  
            
            map_reward_estimation(self.alpha, self.beta, grid_width, grid_height, road)
            # Safe experiment data            
            self.exp_human_feedback.append(human_feedback)
            self.exp_reward.append(out_reward)
            self.exp_certainty.append(out_certainty)
            self.exp_alpha.append(self.alpha)
            self.exp_beta.append(self.beta)
        return reward_sums

class QLearningAgent_Bernoulli_random(QLearningAgent_Bernoulli):
    def choose_action(self, state, step):
        return random.randint(0, 1)