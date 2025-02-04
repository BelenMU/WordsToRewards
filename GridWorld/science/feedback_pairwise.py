import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.stats import beta
from tqdm import tqdm # For progress bar
import copy
import os
import math
import json

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D


# Import functions
from science.draw_map import show_trajectory_on_map, show_two_trajectories_on_map


def get_preference_HF():
    """
    Prompts the user to select their preference between two trajectories, converts the input to
    a binary 0/1 integeer, and then returns it.

    Returns:
        int: Preference.    
    """
    while True:
        text = input("Which trajectory is better? (0: blue, 1: red) ")
        if text in ['0', '1']:
            return int(text)
        else:
            print("Invalid input. Please enter 0 or 1.")


def ask_preference_HF(traj1, traj2, road, grid_width, grid_height, car_init):
    """
    Visualize both trajectories, collect human preference feedback..

    Args:
        traj1 (list): A list of coordinates representing the first trajectory of the agent.
        traj2 (list): A list of coordinates representing the second trajectory of the agent.
        road (np.array): An array containing the road's locations, i.e., the optimal path to be taught to the agent.
        grid_width (int): The width of the grid representation of the map.
        grid_height (int): The height of the grid representation of the map.
        car_init (tuple): The initial position of the agent.

    Returns:
            - human_feedback (int): The preference provided by the human.
    """
    # Show trajectory on map
    #trajectory = show_trajectory_on_map([], [], road, grid_width, grid_height, traj1, car_init)
    #trajectory = show_trajectory_on_map([], [], road, grid_width, grid_height, traj2, car_init)
    show_two_trajectories_on_map([], [], road, grid_width, grid_height, traj1, traj2, car_init)

    # Ask human for feedback
    human_feedback = get_preference_HF()

    return human_feedback