import numpy as np
import random
import spacy
from spacy import displacy 
import os
import math

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# Path images
directory_path = "./icons/"
# Get a list of image files in the directory
image_files = [os.path.join(directory_path, filename) for filename in os.listdir(directory_path)]


# Randomly set landmarks and road
def initialize_landmark_loc(grid_width, grid_height, num_landmarks, distance=2, max_attempts=1000):
    locations = []    
    for _ in range(num_landmarks):
        for attempt in range(max_attempts):
            new_loc = [random.randint(1, grid_width-2), random.randint(1, grid_height-1)]
            if all(abs(loc[0]-new_loc[0])>=distance or abs(loc[1]-new_loc[1])>=distance for loc in locations):
                break
            
            # if max attempts reached append location regardless of distance
            if attempt == max_attempts - 1:
                print(f"Couldn't find a suitable location for the point after {max_attempts} attempts. Ignoring distance constraint.")
        locations.append(new_loc)
    return locations

def random_road(car_init, num_steps, grid_width, grid_height):
    # initialize the path with the car's initial position
    road = [car_init[0]]
    
    for _ in range(num_steps):
        # get the last position
        last_position = road[-1]
        
        # generate possible moves
        possible_moves = []
        if last_position[0] < grid_width - 1:  #if not on the right edge
            possible_moves.append([last_position[0] + 1, last_position[1]])
        
        if last_position[1] < grid_height - 1:  #if not on the upper edge
            possible_moves.append([last_position[0], last_position[1] + 1])
        
        # if there are possible moves, randomly choose one
        if possible_moves:
            new_position = possible_moves[np.random.randint(0, len(possible_moves))]
            road.append(new_position)
        else:
            break   #if stuck (at upper-right corner), then end the loop early
    
    return np.array(road)

def create_grid_map(images, locations, road, grid_width, grid_height, save=False, road_show=True):
    # Create the figure
    fig, ax = plt.subplots(figsize=(555 / 80, 330 / 80))

    # Draw the grid lines
    major_ticks = np.arange(0, max(grid_width, grid_height), 1)
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    ax.grid(which='both')
    # Remove axis labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # For each image and location, add to the plot
    for image, location in zip(images, locations):
        # Open the image
        img = Image.open(image)
        
        # Convert the image into numpy array
        img = np.array(img).astype(np.uint8) / 255.
        
        # Create an offset image artist
        im = OffsetImage(img, zoom=0.45)
        
        # Use AnnotationBbox to put the image on the plot at the specified location
        ab = AnnotationBbox(im, location, frameon=False)

        # Add the artist to the plot
        ax.add_artist(ab)
    
    # Set the limits of the plot to the limits of the grid
    ax.set_xlim(-1, grid_width)
    ax.set_ylim(-1, grid_height)
    
    if save:
        plt.savefig('map.png', dpi=80, bbox_inches='tight')
        
    
    # Add a red Rectangle artist covering the valid area.
    rectangle = Rectangle((0, 0), grid_width-1, grid_height-1, facecolor='green', alpha=0.1)
    ax.add_patch(rectangle)
        
    
    if road_show:
        # Render the road as lines on the grid
        for i in range(1, len(road)):
            line = Line2D((road[i-1][0], road[i][0]), (road[i-1][1], road[i][1]), linewidth=30, color='grey')
            ax.add_line(line)

        # Add a star at the end of the road
        ax.scatter(road[-1][0], road[-1][1], marker='*', color='yellow', s=500, zorder=4)
        # Add a circle at the start of the road
        ax.scatter(road[0][0], road[0][1], marker='o', color='yellow', s=400, zorder=4)

    # Apply a tight layout
    plt.tight_layout()
    
    
    # Return the figure and axes
    return fig, ax

def add_trajectory(map_fig, map_ax, trajectory, start, color='blue', linestyle='-'):
    # print(trajectory)
    line = Line2D((start[0],trajectory[0][0]),
                  (start[1], trajectory[0][1]), linewidth=2, color=color, linestyle=linestyle)
    map_ax.add_line(line)
    for i in range(1, len(trajectory)):
        line = Line2D((trajectory[i-1][0], trajectory[i][0]),
                      (trajectory[i-1][1], trajectory[i][1]), linewidth=2, color=color, linestyle=linestyle, zorder = 4)
        map_ax.add_line(line)
        
    # Create a square at the end of the trajectory
    s = 0.25 # Size of the square
    square = Rectangle((trajectory[-1][0]-s/2, trajectory[-1][1]-s/2), s, s, color=color, alpha=0.5, zorder = 5)
    map_ax.add_patch(square)
    map_fig.canvas.draw()
    #plt.show()

def show_trajectory_on_map(images, loc_landmarks, road, grid_width, grid_height, trajectory, car_init):
    # Show trajectory on map
    map_fig, map_ax = create_grid_map(images, loc_landmarks, road, grid_width, grid_height)
    add_trajectory(map_fig, map_ax, trajectory, car_init)
    plt.show()
    trajectory = np.array(trajectory)
    return trajectory

def show_two_trajectories_on_map(images, loc_landmarks, road, grid_width, grid_height, traj1, traj2, car_init):
    # Show trajectory on map
    map_fig, map_ax = create_grid_map(images, loc_landmarks, road, grid_width, grid_height)
    add_trajectory(map_fig, map_ax, traj1, car_init)
    add_trajectory(map_fig, map_ax, traj2, car_init, color='red')
    plt.show()
    
def map_reward_estimation(alpha, beta, grid_width, grid_height, road):
    state_expected_reward = (alpha / (alpha + beta) - 0.5) * 2
    state_uncertainty = (alpha * beta) /  ((alpha * beta)*(alpha * beta)* (alpha + beta + 1))
    fig, ax = plt.subplots(figsize=(555 / 80, 330 / 80))
    cmap = plt.cm.RdYlGn
     # Render the road as lines on the grid
    for i in range(1, len(road)):
        line = Line2D((road[i-1][0], road[i][0]), (road[i-1][1], road[i][1]), linewidth=30, color='grey')
        ax.add_line(line)
     # For each image and location, add to the plot
    #for image, location in zip(images, loc_landmarks):
    #    # Open the image
    #    img = Image.open(image)
    #    # Convert the image into numpy array
    #    img = np.array(img).astype(np.uint8) / 255.
    #    # Create an offset image artist
    #    im = OffsetImage(img, zoom=0.3)
    #    # Use AnnotationBbox to put the image on the plot at the specified location
    #    ab = AnnotationBbox(im, location, frameon=False)
    #    # Add the artist to the plot
    #    ax.add_artist(ab)
    for ii in range(state_expected_reward.size):
        grid_row = ii // grid_height
        grid_col = ii % grid_height
        size = 500 * state_uncertainty[ii]  # scale factor for visibility of circles
        sc = ax.scatter(grid_row, grid_col,  s=size, c=[[state_expected_reward[ii]]], vmin=-1, vmax=1, cmap=cmap, zorder = 3)
    # Plot reward as color and uncertainty as size of the ball
    plt.colorbar(sc, cmap=cmap, ax=ax)
    plt.title('Expected Reward Map')
    major_ticks = np.arange(0, max(grid_width, grid_height), 1)
    ax.set_xticks(np.arange(-1,grid_width+1))
    ax.set_yticks(np.arange(-1,grid_height+1))
    ax.grid(which='both')
    # Remove axis labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.show()

def map_loc_landmark(ind_loc, ind_landmark, certainty_reward, pixel_landmarks, grid_width, grid_height):
    map_loc = np.zeros([grid_width, grid_height])
    x_left = int(pixel_landmarks[ind_landmark, 0])
    x_right = int(pixel_landmarks[ind_landmark, 1])
    y_down = int(pixel_landmarks[ind_landmark, 2])
    y_up = int(pixel_landmarks[ind_landmark, 3])
    # "above under left right around in"
    if ind_loc == 0: # above
        map_loc[x_left:x_right+1, y_up] = 0.5 * certainty_reward
        map_loc[x_left+1:x_right, y_up] = certainty_reward
        #if y_up < grid_height-1:
        #    map_loc[x_left:x_right+1, y_up+1] = certainty_reward*0.5
        
    elif ind_loc == 1: # under
        map_loc[x_left:x_right+1, y_down] = 0.5 * certainty_reward
        map_loc[x_left+1:x_right, y_down] = certainty_reward
        
    elif ind_loc == 2: # left
        map_loc[x_left, y_down:y_up+1] = certainty_reward * 0.5
        map_loc[x_left, y_down+1:y_up] = certainty_reward
        
    elif ind_loc == 3: # right
        map_loc[x_right, y_down:y_up+1] = certainty_reward * 0.5
        map_loc[x_right, y_down+1:y_up] = certainty_reward
        
    elif ind_loc == 4: # around
        map_loc[x_left:x_right+1, y_up] = certainty_reward
        map_loc[x_left:x_right+1, y_down] = certainty_reward
        map_loc[x_left, y_down:y_up+1] = certainty_reward
        map_loc[x_right, y_down:y_up+1] = certainty_reward
        
    else: # in
        if x_right - x_left > 1 and  y_up - y_down > 1:
            map_loc[x_left+1:x_right, y_down+1:y_up] = certainty_reward
        elif y_up == grid_height - 1:
            map_loc[x_left+1:x_right, y_down+1:y_up+1] = certainty_reward
        elif x_right == grid_width - 1:
            map_loc[x_left+1:x_right+1, y_down+1:y_up] = certainty_reward
        elif y_down == 0:
            map_loc[x_left+1:x_right, y_down:y_up] = certainty_reward
        elif x_left ==0:
            map_loc[x_left:x_right, y_down+1:y_up] = certainty_reward
        else:
            map_loc[x_left:x_right+1, y_down:y_up+1] = certainty_reward*0.5
            
        
    return map_loc