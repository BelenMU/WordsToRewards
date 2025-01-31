import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.stats import beta
from tqdm import tqdm # For progress bar
import copy
import os
import math
import json

# Function to calculate deviation of trajectory from the road
def calculate_deviation(trajectory, road):
    deviation = 0
    for step in trajectory:
        if not any(np.array_equal(step, road_step) for road_step in road):
            deviation -= 1
    return deviation


def true_trajectory_level_feedback(trajectory, road, grid_height):  
    state_index_prev = -1  # To detect instances where agent is stuck
    NUM_STEPS = len(trajectory)  # Assumes trajectory is a numpy array with a defined length
    if len(road) > NUM_STEPS:
        road = road [-NUM_STEPS:]
    
    # Find label + certainty
    deviation_ratio = -1 * calculate_deviation(trajectory, road) / NUM_STEPS
    if deviation_ratio >= 0.5:
        label_reward = 'NEG'
        certainty_reward = [(deviation_ratio - 0.5) * 2 + 1e-5]
    else:
        label_reward = 'POS'
        certainty_reward = [(0.5 - deviation_ratio) * 2 + 1e-5]
    
    # Format it into dictionary
    out_reward = {}
    out_certainty = {}
    for temp_ind in np.arange(NUM_STEPS):
        # Pass from trajectory to state index
        state_index = trajectory[temp_ind][0] * grid_height + trajectory[temp_ind][1]
        if state_index != state_index_prev: # Only add the reward once, even when it gets stuck
            out_reward[state_index] = label_reward
            out_certainty[state_index] = certainty_reward
            state_index_prev = state_index 
    return [], out_reward, out_certainty

def wrapper_true_trajectory_level_feedback(trajectory_states, images, loc_landmarks, road, grid_width, \
                                            grid_height, car_init, pixel_landmarks, list_landmarks):
    return true_trajectory_level_feedback(trajectory_states, road, grid_height)

def true_state_level_feedback(trajectory, road, grid_height): 
    out_reward = {}
    out_certainty = {}
    state_index_prev = -1  # To detect instances where agent is stuck
    NUM_STEPS = len(trajectory)  # Assumes trajectory is a numpy array with a defined length
    if len(road) > NUM_STEPS:
        road = road [-NUM_STEPS:]
    
    # Compute road indices
    road_ind = []
    for temp_ind in np.arange(NUM_STEPS):
        road_ind.append(road[temp_ind, 0]* grid_height + road[temp_ind, 1])
    
    for temp_ind in np.arange(NUM_STEPS):
        # Pass from trajectory to state index
        state_index = trajectory[temp_ind][0] * grid_height + trajectory[temp_ind][1]
        if state_index != state_index_prev: # Only add the reward once, even when it gets stuck
            if state_index in road_ind:
                out_reward[state_index] = 'POS'  
            else:
                out_reward[state_index] = 'NEG'  
            out_certainty[state_index] = 1
            state_index_prev = state_index 
    return [], out_reward, out_certainty


def wrapper_true_state_level_feedback(trajectory_states, images, loc_landmarks, road, grid_width, \
                                            grid_height, car_init, pixel_landmarks, list_landmarks):
    return true_state_level_feedback(trajectory_states, road, grid_height)