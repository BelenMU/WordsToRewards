import numpy as np
import re
import pickle
import glob
import os
import pandas as pd
import json
import string
from science.agents import GridEnvironment,  QLearningAgent_Bernoulli
import matplotlib.pyplot as plt

def calculate_deviation(trajectory, road):
    """
    Calculate the deviation of a trajectory from the road.

    The function iterates through each step in the trajectory and checks
    if the step is part of the road. For each step that does not exist on
    the road, the deviation count is decremented by 1.

    Parameters:
    trajectory (list): A list of positions (steps) defining the trajectory.
    road (list): A list of positions (steps) defining the road.

    Returns:
    int: The total deviation of the trajectory from the road. Negative values
         indicate a deviation from the road.

    Example:
    >>> calculate_deviation([(0,1), (0,2), (0,3)], [(0,1), (0,2), (0,4)])
    -1

    """    
    deviation = 0
    for step in trajectory:
        if not any(np.array_equal(step, road_step) for road_step in road):
            deviation -= 1
    return deviation

def vector_to_algebraic(trajectory_vector):
    """
    Convert a list of vector coordinates to algebraic notation.

    Each vector coordinate is assumed to be a tuple with two elements: (column, row),
    where row and column are 0-indexed integers representing positions on a grid. This
    function will convert the vector coordinates to algebraic notation, which consists
    of a 1-indexed row number followed by a letter representing the column.

    Parameters:
      trajectory_vector (list of tuple of int): A list of 2D grid coordinates.

    Returns:
      list of str: The algebraic notation for the given trajectory vector.

    Example:
    >>> vector_to_algebraic([(0, 0), (1, 2), (3, 4)])
    ['1a', '3b', '5d']
    """
    algebraic_notation_list = []
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    
    for coordinate in trajectory_vector:
        # Convert the row from 0-indexed to 1-indexed and get the column letter
        algebraic_notation = f"{coordinate[1] + 1}{alphabet[coordinate[0]]}"
        algebraic_notation_list.append(algebraic_notation)

    return algebraic_notation_list

def create_landmark_dict(landmark_pixels, landmark_names):
    """
    Create a dictionary mapping landmark names to their grid locations in algebraic notation.

    Parameters:
      landmark_pixels (list of tuple of float): A list of tuples containing the pixel 
                                                boundaries (x_min, x_max, y_min, y_max) for each landmark.
      landmark_names (list of str): A list of names corresponding to the landmarks defined
                                    by the pixel boundaries.

    Returns:
      dict: A dictionary where each key is a landmark name and the associated value is a
            list of grid locations in algebraic notation representing the area covered by the landmark.
    """
    landmark_grid_dict = {}
    for index, (x_min, x_max, y_min, y_max) in enumerate(landmark_pixels):
        landmark_name = landmark_names[index]
        landmark_grid_positions = []
        # We loop over the ranges given from x_min to x_max and y_min to y_max
        for x in range(int(x_min)+1, int(x_max)):
            for y in range(int(y_min)+1, int(y_max)):
                # Convert grid coordinates to our custom algebraic notation
                landmark_grid_positions.append((x, y))
        if x_max-x_min == 1:
            if x_min == 0:
                x = 0
            else:
                x = 9
            for y in range(int(y_min)+1, int(y_max)):
                # Convert grid coordinates to our custom algebraic notation
                landmark_grid_positions.append((x, y))

        elif y_max-y_min == 1:
            if y_min == 0:
                y = 0
            else:
                y = 4
            for x in range(int(x_min)+1, int(x_max)):
                # Convert grid coordinates to our custom algebraic notation
                landmark_grid_positions.append((x, y))
        # Convert the list of (row, col) pairs into algebraic notation
        landmark_grid_dict[landmark_name] = vector_to_algebraic(landmark_grid_positions)

    return landmark_grid_dict

def split_feedback(feedback):
    """
    Split feedback into sentences.
    
    Parameters:
      feedback (str): A string containing the feedback text to be split.

    Returns:
      list of str: A list of individual sentence strings extracted from the feedback.

    Example:
    >>> split_feedback("The beginning is ok. The rest is horrible!")
    ['The beginning is ok.', 'The rest is horrible!']
    """
    # Split feedback into sentences using regular expression
    sentences = re.split(r'[.,!]+\s*', feedback)
    # Remove any empty strings that may occur due to multiple delimiters
    return [sentence for sentence in sentences if sentence]

class GridState:
    """
    A class used to represent the state of a grid, including feedback, landmarks, and trajectory.

    Attributes:
      feedback (str): The feedback message from the human associated with the grid state.
      landmarks (dict): A dictionary mapping landmark names to the locations they occupy in the grid.
      trajectory (list): A list representing the trajectory taken within the grid.

    Methods:
      get_prompt():
      Converts the stored state information into a JSON-formatted string.
    """

    def __init__(self, feedback, landmarks, trajectory):
        """
        Initializes a new instance of the GridState class.
        """
        self.feedback = feedback
        self.landmarks = landmarks
        self.trajectory = trajectory

    def get_prompt(self):
        """
        Converts the state attributes into a JSON-formatted string.

        Returns:
        str: The JSON-formatted string representation of the grid state.

        """
        # Convert spacy.tokens.doc.Doc keys to strings for JSON serialization
        landmarks_str_keys = {str(key): value for key, value in self.landmarks.items()}

        state_dict = {
            "feedback": self.feedback,
            "landmarks": landmarks_str_keys,
            "trajectory": self.trajectory
        }
        prompt = json.dumps(state_dict)  # Serialize the dictionary to a JSON-formatted string
        return prompt

def load_experiments(directory, file_basename, num_experiments):
    """
    Load experiment setups  and convert landmarks into dictionary with Locations in algebraic notation.

    Parameters:
      directory (str): The path to the directory containing the pickle files.
      file_basename (str): The basename of the pickle files to be loaded.

    Returns:
      list of dicts: A list containing dictionaries for each loaded experiment with the information 
                    about the landmarks. The names as keys, and the locations occupied by the landmarks
                    in algebraic notation.
    
    Raises:
      FileNotFoundError: If a pickle file does not exist for an expected experiment number.
      Exception: Any other exception encountered during file loading or processing.

    """
    experiment_landmarks = []    

    # Loop to load each experiment
    for i in range(1, num_experiments + 1):
        filename = f"{directory}{file_basename}{i}.pkl"
        try:
            with open(filename, "rb") as file:
                # Placeholders for contents that are not used here
                _, _, _, landmark_pixels, landmark_names = pickle.load(file)
                experiment_landmarks.append(create_landmark_dict(landmark_pixels, landmark_names))
        except FileNotFoundError:
            raise FileNotFoundError(f"The file {filename} was not found.")
        except Exception as e:
            raise Exception(f"An error occurred while loading {filename}: {e}")
            
    return experiment_landmarks

def get_experiment_dataframe(base_dir, experiment_landmarks, grid_width, grid_height):
    """
    Load experiment data files and construct metadata and GridState for each.

    Parameters:
      base_dir (str): The base directory where pickle files are stored.
      experiment_landmarks (dict): A mapping from experiment number to landmarks.

    Returns:
      pd.DataFrame: A DataFrame containing columns for Day, Labmate, AgentType, 
      ExperimentNumber, Data, DeviationHardwire, ScoreHardwire, and GridStates.
    """
    # List all .pkl files in the human_experiments directory
    pkl_files = glob.glob(os.path.join(base_dir, 'nov*_*_*.pkl'))
    
    # Regex to extract day, labmate, agent_type, potential version, and experiment number
    pattern = r'nov(\d+)_labmate(\d+)_([a-z]+)(?:_v\d+)?_exp(\d+)\.pkl'
    
    # To store the loaded data along with their metadata
    experiments_data = []
    
    # Loop over each .pkl file
    for file_path in pkl_files:
        # Extract the metadata from the filename using regex
        filename = os.path.basename(file_path)
        match = re.match(pattern, filename.lower())
        if match:
            day, labmate, agent_type, experiment_number = match.groups()
            # Load the .pkl file
            data = pd.read_pickle(file_path)
            labmate = int(labmate)
            experiment_number = int(experiment_number)

            if agent_type =='greedy':
                # Recover the unsaved trajectories
                trajectories, final_trajectory = recover_greedy_trajectories(data, grid_width, grid_height)
                data['trajectory'] = trajectories
                data['learned_trajectory'] = final_trajectory
            # Calculate deviations for each trajectory
            road = data['road'][1:]
            deviations = [calculate_deviation(trajectory, road) for trajectory in data['trajectory']]
            final_score = calculate_deviation(data['learned_trajectory'], road)
            
            # Create GridState objects
            feedbacks = data['human_feedback']
            landmarks = experiment_landmarks[experiment_number]#str(
            trajectory_algebraic = [vector_to_algebraic(trajectory) for trajectory in data['trajectory']]
            
            all_grid_states = []
    
            for feedback, trajectory in zip(feedbacks, trajectory_algebraic):
                # Split feedback into individual sentences
                sentences = split_feedback(feedback)
                # Create a list of GridState objects for each sentence
                grid_states = [
                    GridState(sentence, landmarks, trajectory)
                    for sentence in sentences
                ]
                # Appending the list for the current feedback to the overall list
                all_grid_states.append(grid_states)
            
            # Append a tuple of the metadata and data to the experiments_data list
            experiments_data.append((day, labmate, agent_type, experiment_number, data, deviations, final_score, all_grid_states))
        else:
            print(f"Filename {filename} did not match the pattern and was skipped.")
    # Convert the list to a DataFrame
    experiments_df = pd.DataFrame(experiments_data, columns=['Day', 'Labmate', 'AgentType', 'ExperimentNumber', \
                                                             'Data', 'DeviationHardwire', 'ScoreHardwire', 'GridStates'])
    return experiments_df

def algebraic_to_index(algebraic_vector, grid_height):
    """
    Convert locations from algebraic notation to index positions on a grid.

    Parameters:
      algebraic_vector (list of str): The locations in algebraic notation (e.g., ["1a", "2b"...]).
      grid_height (int): The number of rows in the grid.

    Returns:
      list of int: A list of index positions corresponding to the locations on the grid.
    """
    indices = []
    #print(f'algebraic_vector {algebraic_vector}.')
    for loc in algebraic_vector:
        row = int(loc[0:-1]) - 1  # The row number is taken as the integer part minus 1
        col = ord(loc[-1].lower()) - ord('a')  # The column letter 'a' maps to index 0
        indices.append(col * grid_height + row)  # Calculate the index using column-major order
    return indices

def create_system_message(grid_height, grid_width):
    """
    Create a structured message for the 'system' containing instructions on how to process trajectory feedback.

    Parameters:
      grid_height (int): The height of the grid map that the agent is navigating.
      grid_width (int): The width of the grid map that the agent is navigating.

    Returns:
      dict: A dictionary with two keys: "role" and "content". The "role" is set to "system", and "content" 
            is a string describing the task of translating the feedback into actionable information for 
            an agent navigating a grid.

    The "content" for the system includes descriptions of how to handle different types of feedback: 
    imperative, evaluative, and descriptive, and what to do with these feedback types in terms of mapping 
    to locations and labeling them appropriately.
    """
    example_feedback = '''
                {"feedback": "it should not go below the bed.", 
                "landmarks": {"clock": ["2f"], "bed": ["2c", "2d"]}, 
                "trajectory": ["1b", "1c", "1d", "1e", "1f", "1g", "2g", "3g", "4g", "4h"]}
                '''
    example_output = {'locations': ['1c', '1d'], 'label': 'NEG', 'feedback_type': 'imperative'}

    
    example_feedback2 = '''
                {"feedback": "the last couple steps are good", 
                "landmarks": {"clock": ["2f"], "bed": ["2c", "2d"]}, 
                "trajectory": ["1b", "1c", "1d", "1e", "1f", "1g", "2g", "3g", "4g", "4h"]}
                '''
    example_output2 = {'locations': ['4g', '4h'], 'label': 'POS', 'feedback_type': 'evaluative'}
    
    example_feedback3 = '''
                {"feedback": "Go inside the clock", 
                "landmarks": {"clock": ["2f"], "bed": ["2c", "2d"]}, 
                "trajectory": ["1b", "1c", "1d", "1e", "1f", "1g", "2g", "3g", "4g", "4h"]}
                '''
    example_output3 = {'locations': ['2f'], 'label': 'POS', 'feedback_type': 'imperative'}
    system_content = f"""
    An agent is trying to learn and follow a specific path in a {grid_height}x{grid_width} grid map.
    Your job is to translate the feedback of the current trajectory into feedback types, locations in the map, and a label.

    - If the feedback type is imperative, compute what locations in the map the instructions are referring to and label them as either 'good' (go to) or 'bad' (avoid).

    - If the feedback type is evaluative, determine what locations in the map are being referred to by the feedback and whether the feedback is positive or negative.

    - If the feedback type is descriptive, compute what new locations the agent should have visited, and label them as positive.
    
    Use the getReward function to only return a JSON file with the specified shape enclosed in double quotes.

    For example: if the user's input is {example_feedback}, then the output should be {example_output}.
    
    Another example: if the user's input is {example_feedback2}, then the output should be {example_output2}.
    
    Another example: if the user's input is {example_feedback3}, then the output should be {example_output3}.
    """
    # If feedback are single sentiment words, like "good" or "bad", set locations to the 'trajectory' taken.
    return {"role": "system", "content": system_content.strip()}
    
def create_location_pattern(grid_width, grid_height):
    """
    Generate a regular expression pattern to match valid grid locations.

    Parameters:
      grid_width (int): The width of the grid in terms of number of columns.
      grid_height (int): The height of the grid in terms of number of rows.

    Returns:
      str: A pattern string that can be used in regular expressions to match valid locations 
           on a grid specified by its width and height.
    """
    max_col_letter = string.ascii_lowercase[grid_width - 1]  # Converts a column number to a lowercase letter, 1-based index
    # Create a pattern to restrict row numbers and a letter for columns.
    row_pattern = f"[1-{grid_height}]"
    col_pattern = f"[a-{max_col_letter}]"
    pattern = f"^{row_pattern}{col_pattern}$"
    return pattern

def get_reward_function_structure(grid_width, grid_height):
    """
    Generate a dictionary describing the expect structure for a 'getReward' function.

    Parameters:
      grid_width (int): The width of the grid in terms of number of columns.
      grid_height (int): The height of the grid in terms of number of rows.

    Returns:
      dict: A dictionary that defines the structure of the 'getReward' function, detailing the
            function name, parameters, and their associated properties such as type and description.
    """
    loc_pattern = create_location_pattern(grid_width, grid_height)
    # "Locations in the grid referring to feedback with row numbers (down-up) and a lowercase letter for columns (left-right), e.g., '1b', '3c'."
    return {
        "name": "getReward",
        "parameters": {
            "type": "object",
            "properties": {
                "locations": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "pattern": loc_pattern  # Pattern ensures the locations match the grid format and stay within grid limits.
                    },
                    "description": ("Locations in the grid referring to feedback with row numbers and a lowercase letter "
                                "for columns. The rows are numbered from bottom to top (1 is the lowest row, increasing "
                                "as you move upward), and columns are labeled from left to right (a-j). For example, '1b' "
                                "refers to the lowest row in the second column from the left.")
                },
                "label": {
                    "type": "string",
                    "enum": ["POS", "NEG"],
                    "description": "The feedback's connotation, positive or negative."
                },
                "feedback_type": {
                    "type": "string",
                    "enum": ["imperative", "evaluative", "descriptive"],
                    "description": """
                    Imperative: Feedback includes instructions on what locations are good or should be avoided.
                    Evaluative: Feedback is an assessment of the current trajectory.
                    Descriptive: Feedback is about modifications for an improved trajectory.
                    """
                }
            },
            "required": ["locations", "label"]
        }
    }


def recover_greedy_trajectories(data, grid_width, grid_height):
    """
    Recover trajectories using a greedy strategy with the given data.

    Parameters:
        data (dict): A dictionary  with recorded data from human experiments containing 'reward' and 'certainty' keys with 
                      associated lists of numpy arrays. 'reward' is a list where each element is an array indicating the 
                      grid position and whether it is a positive or negative reward. Certainty provides a confidence
                      level for the corresponding reward at each position.
        grid_width (int): The width of the grid environment.
        grid_height (int): The height of the grid environment.
    
    Returns:
        trajectories (list): A list of trajectories where each trajectory is a sequence of grid positions before
                             the Q-learning agent updates its beliefs.
        final_trajectory (list): The final trajectory after the Q-learning agent has updated its beliefs with
                                 all the rewards and corresponding certainties.
    """
    # Instantiate the grid environment and Q-learning agent
    env = GridEnvironment(grid_width, grid_height, 10, [0,0])
    agent = QLearningAgent_Bernoulli(env, alpha_init = 0.5, beta_init = 0.5)
    
    # Retrieve rewards and certainty data from the input
    rewards = data['reward']
    certainty = data['certainty']
    
    # Initialize empty list for trajectories
    trajectories = []
    
    # Iterate over the dataset to recover trajectories
    for ii in range(len(rewards)):
        # Append the optimal trajectory based on the current Q-values
        trajectories.append(agent.get_optimal_trajectory())
        
        # Iterate over rewards at each location and update agent's beliefs
        for ind_reward in rewards[ii]:
            temp_cert = certainty[ii][ind_reward]
            temp_reward = rewards[ii][ind_reward]
            
            # Update alpha and beta for the agent based on the reward's label and the certainty
            for label_reward in range(len(temp_reward)):
                if temp_reward[label_reward] == 'POS':
                    agent.alpha[ind_reward] += temp_cert[label_reward]*agent.scale
                else:
                    agent.beta[ind_reward] += temp_cert[label_reward]*agent.scale
    
    # After updating beliefs with the entire dataset, get the final trajectory
    final_trajectory = agent.get_optimal_trajectory()
    
    return trajectories, final_trajectory

def plot_mean_scores_bars(list_of_matrices, labels, title):
    """
    Plots bars side by side for the mean of values for each given matrix with standard deviation error bars.
    
    Parameters:
        list_of_matrices: A list of np.arrays of the same shape.
        labels: A list of strings representing the labels for each group in the plot.
        title: String with the title for the plot.
    """
    if not all(matrix.shape == list_of_matrices[0].shape for matrix in list_of_matrices):
        raise ValueError("All matrices must have the same shape.")
    if len(list_of_matrices) != len(labels):
        raise ValueError("The number of matrices must match the number of labels provided.")
    
    n_groups, n_steps = list_of_matrices[0].shape
    bar_width = 0.8 / len(list_of_matrices)  # width of the bars
    opacity = 0.7  # bar opacity
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(9, 5))  # adjust the figure size as needed
    index = np.arange(n_steps)  # array with the number of steps
    
    # Iterate over each matrix and its corresponding label
    for i, matrix in enumerate(list_of_matrices):
        mean_values = np.mean(matrix, axis=0)
        std_deviation = np.std(matrix, axis=0)
        ste_deviation = np.std(matrix, axis=0) / np.sqrt(len(matrix))
        bar_position = index + (i - len(list_of_matrices)/2) * bar_width
        
        # Plot each set of bars
        ax.bar(bar_position, mean_values, bar_width, alpha=opacity, yerr=ste_deviation, label=labels[i])

    # Add some text for labels, title, and axes ticks
    ax.set_xlabel('Steps', fontsize=12)
    ax.set_ylabel('Mean Score', fontsize=12)
    ax.set_title(title)
    ax.set_xticks(index)
    ax.set_xticklabels(['Step 1', 'Step 2', 'Step 3', 'Step 4', 'Step 5'])
    ax.legend(loc='lower right', fontsize=12)
    
    # Show grid and make layout tight
    plt.grid(True)
    plt.tight_layout()
    
    # Show the plot
    plt.show()
    
    
def create_system_message_v2(grid_height, grid_width):
    """
    Create a structured message for the 'system' containing instructions on how to process trajectory feedback.

    Parameters:
      grid_height (int): The height of the grid map that the agent is navigating.
      grid_width (int): The width of the grid map that the agent is navigating.

    Returns:
      dict: A dictionary with two keys: "role" and "content". The "role" is set to "system", and "content" 
            is a string describing the task of translating the feedback into actionable information for 
            an agent navigating a grid.

    The "content" for the system includes descriptions of how to handle different types of feedback: 
    imperative, evaluative, and descriptive, and what to do with these feedback types in terms of mapping 
    to locations and labeling them appropriately.
    """
    
    input1= '''
          {"feedback": "it should go inside the clock, not to its right.", 
          "landmarks": \{"clock": ["2f"], "bed": ["2c", "2d"]\}, 
          "trajectory": ["1b", "1c", "1d", "1e", "1f", "1g", "2g", "3g", "4g", "4h"]}
                '''
    
    system_content = f"""
        You will assess the state-reward pairs of an agent's trajectoy in a {grid_height}x{grid_width} grid map based on human observer feedback. The goal is for the agent to follow a specific path.

        **Objective:** 
        Identify successful and unsuccessful states of the agent in the grid based on observer comments after viewing a 10-move simulation.

        **Input Format:**

        1. **Observer Comments**: Natural language feedback from a human observer who has watched the simulation.
        2. **Agent's Trajectory**: A sequence of states in the grid over timesteps. Each state is defined by a numeric row and a lowercase letter indicating the column. Rows are numbered from bottom to top (1 is the lowest row, increasing as you move upward), and columns are labeled from left to right (a-j). For example, '1b' refers to the lowest row in the second column from the left.

        **Processing Steps:**
        1. For each sentence or comment in the human feedback.
            a. **Classify Human Feedback**:
                - **Imperative**: Indicates states to which to go or avoid (e.g., "Go to the chair.").
                - **Evaluative**: Criticizes the observed simulation (e.g., "The first two steps are wrong, but the end is great").
                - **Description**: Describes whether certain states are good or bad  (e.g., "The area around the bed is bad.").

            b. **Generate State-Reward Pair**:
                - **Imperative**: 
                    - **location**: The state as described by the comment.
                    - **label*: 'POS" if feedback describes place to which to go, and 'NEG' if feedback describes place to avoid.

                - **Evaluative**: .
                    - **location**: Identify the index of the specific state(s) in the trajectory the feedback refers to, and return the input state(s) corresponding to such index.
                    - *label*: 'POS" if connotation of feedback is positive, and 'NEG' if connotation of feedback is negative.

                - **Description**:  
                    - **location**: The state as described by the comment.
                    - *label*: 'POS" if connotation of feedback is positive, and 'NEG' if connotation of feedback is negative.

        2. **Return the Result**:
            - Check your results
            - Use the get_reward_v2 function to only return a JSON file with the specified shape.

        **Example Inputs with Expected Outputs:**

        Example 1:
        Input: {input1}

        Expected Output:
        [
          {{"location": "2f", "label": "POS"}},
          {{"location": "2g", "label": "NEG"}}
        ]

        Example 2:
        Input:
        {{
        "feedback": "The first couple steps are good. Go above the bed.", 
        "landmarks": {{"bed": ["2c", "2d"], "chair": ["3c, 4c"]}}, 
        "trajectory": ["1b", "2b", "3b", "4b", "4b", "4b"]
        }}

        Expected Output:
        [
          {{"location": "1b", "label": "POS"}},
          {{"location": "2b", "label": "POS"}},
          {{"location": "3c", "label": "POS"}},
          {{"location": "3d", "label": "POS"}}
        ]
    """
    return {"role": "system", "content": system_content.strip()}

def get_reward_v2(grid_width, grid_height):
    """
    Generate a dictionary describing the expect structure for a 'getReward' function.

    Parameters:
      grid_width (int): The width of the grid in terms of number of columns.
      grid_height (int): The height of the grid in terms of number of rows.

    Returns:
      dict: A dictionary that defines the structure of the 'getReward' function, detailing the
            function name, parameters, and their associated properties such as type and description.
    """
    loc_pattern = create_location_pattern(grid_width, grid_height)

    return {
        "name": "get_reward_v2",
        "parameters": {
            "type": "object",
            "properties": {
              "states": {
                "type": "array",
                "items": {
                  "type": "object",
                  "properties": {
                        "location": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "pattern": loc_pattern  # Pattern ensures the locations match the grid format and stay within grid limits.
                            },
                            "description": ("Location in the grid referring to feedback with row number and a lowercase letter "
                                        "for columns.")
                        },
                        "label": {
                            "type": "string",
                            "enum": ["POS", "NEG"],
                            "description": "The feedback's connotation, positive or negative."
                        }
                    },
                    "required": ["location", "label"]
        }
      }
    },
    "required": ["states"]
    }
}
