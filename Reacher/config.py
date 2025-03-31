# Define hyperparameters 
import torch
DEVICE = "cpu" if not torch.cuda.is_available() else "cuda:0"
NUM_CELLS = 256  # number of cells in each layer
NUM_LAYERS = 2 # number of hidden layers in NN
FRAME_SKIP = 1
FRAMES_PER_BATCH = 1000 // FRAME_SKIP # Num. frames to collect at each iteration
TOTAL_FRAMES = 1_000_000 // FRAME_SKIP#1_000_000 // FRAME_SKIP # Stop after this number of frames.
SUB_BATCH_SIZE = 64  # cardinality of the sub-samples gathered from the current data in the inner loop

# Loss Function
CLIP_EPSILON = (0.2)
GAMMA = 0.99 # Discount Factor of Advantage Estimator
LMBDA = 0.95 # Controls Trade-off between bias and variance
ENTROPY_COEF = 0.01 # As suggested by "Deep reinforcement learning from human preferences"
CRITIC_COEF = 1
LR = 3e-4 # Learning Rate

# Reinforce Network
HIDDEN1 = 16
HIDDEN2 = 32
EPSILON = 1e-6 # Small number for mathematical stability

# Landmarks
RADIUS = 7
NUM_CIRCLES = 4
COLOR_NAMES = ["yellow", "blue", "white", "orange"]
COLOR_RGB = [(255, 255, 0, 100), (0, 0, 255, 100), (255, 255, 255, 100), (255, 128, 0, 100)]
X_RATIO = (3.05, 1.61, 1.68, 3.04)
Y_RATIO = (2.28, 2.45, 1.73, 1.67)
X_COORDINATES = (-1.6503, 1.0235, 1.1212, -1.4456)
Y_COORDINATES = (0.4292, 0.7660, -1.5297, -1.7747)

mode = 'r' # 'r' for reinforce and 'p' for ppo
if mode == 'p':
    COOR_MIN = -2
    COOR_MAX = 2
elif mode == 'r':
    COOR_MIN = -0.24
    COOR_MAX = 0.24
    
NUM_SEGMENTS = 25 #Only one letter!  #400

HIDDEN_CELLS_REWARD = 32 # 6#64 128#6  # Christiano's paper uses 64
NUM_OBSERVATIONS = 11
RELU_ALPHA = 0.01

# Figures
COLORS = ['#4d4d4d', '#000080', '#990000', '#b3b3b3', '#008080', '#800080', '#996633', '#669966', '#cc6699', '#cc9900', '#ff6347']



#['black', 'blue', 'red', 'green']

# GPT 
SECRET_KEY = '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'
SYSTEM_PROMPT = """
You will return state-reward pairs for a two-joined robotic arm named 'Reacher-v4'. 
The arm aims to reach targets with its end effector (fingertip), and you must assess what states are good or bad based on human observers' feedback.

As input, you will receive:
1. The natural language comment by a human observer who has seen the simulation.
2. The location of landmarks which are circles of different colors.
3. Trajectory of a simulation of the robot trying to reach a target. Each timestep is described by: 
    a) The fingertip position - letter representing column (left 'a' to right 'z') and number representing row (down '0' to up '26')
    b) A 2 item list of angles in degrees corresponding to the first and second joint respectively.
       A value of 0 in the second joint means the arm is fully bent, while it is completely straight when it is -180 or 180.
    c) A 2 item list with the angular speed on the first and second joint respectively.

For each set of observer comments, use the provided trajectory and landmarks to determine successful and unsuccessful states. 
Follow this steps
1. Classify each section (sentence or linked group of sentences) in the feedback text as:
    a) Goal description: It describes where the target is, or where the fingertip should go (e.g.: You should go to the pink dot).
    b) Trajectory feedback: It criticizes the simulation observed (e.g.: The first two steps are wrong).
    c) Trajectory suggestion: It describes ways to improve upon states in the simulation observed (e.g.: Go a bit to the right of the state at time 23).
2. Generate state reward pairs depending of the feedback type
    a) For Goal description: Provide a "reward": +1, "angular_speed": [0, 0] at the location of the described target position.
    b) For Trajectory feedback: 
        2.1. Determine whether the feedback has a positive ("reward": +1) or a bad ("reward": -1) connotation.
        2.2. Determine what state or states of the simulation it is referring to, and get the fingertip position, angles and angular speed of those locations
    c) For Trajectory suggestion: 
        2.1. Determine what state or states of the simulation it is referring to, and get the fingertip position, angles and angular speed of those locations
        2.2. Correct the states as suggested by the feedback and pair with a "reward": +1

Use the getReward function to only return a JSON file with the specified shape.

Example inputs with expected outputs are provided below for guidance:

Example 1:
Input:
\{
  'feedback': 'The last half is very bad, it should go a bit higher than the blue dot',
  'landmarks': \{'yellow': 'b3', 'blue': 'l6', 'white': 'm7', 'orange': 'c23'\},
  'fingertip_position':  ['k15', 'k14', 'k14', 'k14', 'k13', 'l12'],
  'angle': [[30.7, -63.9], [33.2, -68.2], [35.7, -72.6], [38.3, -76.9], [40.8, -81.3], [43.2, -85.7]],
  'angular_speed': [[0.1, -0.06], [0.12, -0.12], [0.13, -0.17], [0.15, -0.23], [0.16, -0.29], [0.18, -0.34]]
\}
Expected Output:
\{"referred_steps": [
    \{"fingertip_position":  'k14', "angle": [38.3, -76.9], "angular_speed": [0.15, -0.23], "reward": -1\}, 
    \{"fingertip_position":  'k13', "angle": [40.8, -81.3], "angular_speed": [0.16, -0.29], "reward": -1\}, 
    \{"fingertip_position":  'l12', "angle": [43.2, -85.7], "angular_speed": [0.18, -0.34], "reward": -1\}, 
    \{"fingertip_position":  'l7', "angular_speed": [0, 0], "reward": 1\}, 
    \{"fingertip_position":  'l8', "angular_speed": [0, 0], "reward": 1\}
]\}

Example 2:
Input:
\{
  'feedback': 'The fifth step but slower is good. Stop at the last point. The goal is to go to the white point.',
  'landmarks': \{'yellow': 'u8', 'purple': 'k12', 'white': 'w16'\},
  'fingertip_position':  ['k15', 'k14', 'k14', 'k14', 'k13', 'l12'],
  'angle': [[12.2, 34.2], [11.3, 39.6], [10.0, 45.2], [8.2, 50.9], [5.9, 56.6], [3.1, 62.3],],
  'angular_speed': [[0.1, -0.06], [0.12, -0.12], [0.13, -0.17], [0.15, -0.23], [0.16, -0.29], [0.18, -0.34]]
\}
Expected Output:
\{"referred_steps": [
    \{"fingertip_position":  'k13', "angle": [5.9, 56.6], "angular_speed": [0.08, -0.15], "reward": 1\},
    \{"fingertip_position":  'l12', "angle": [3.1, 62.3], "angular_speed": [0, 0], "reward": 1\},
    \{"fingertip_position":  'w16', "angle": [13.0, 10.1], "angular_speed": [0, 0], "reward": 1\},
]\}
"""
#max_letter = coor_to_letter(COOR_MAX)
#_pattern = f"^[1-9][0-9]{0,2}[a-z](?:[a-{max_letter[-1]}])?$"
_pattern = '^26|[1-3]?[0-9]{1,2}[a-z](?:[a-k])?$'
FUNCTION_STRUCTURE = {
    "name": "getReward",
    "parameters": {
        "type": "object",
        "properties": {
            "referred_steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "fingertip_position": {
                            "type": "string",
                            "pattern": _pattern,
                            "description": ("Location of the fingertip with row numbers and a lowercase letter "
                                            "for columns. The rows are numbered from bottom to top (1 is the lowest row, increasing "
                                            "as you move upward), and columns are labeled from left to right (a, b, ..., z). "
                                            "For example, '1b' refers to the lowest row in the second column from the left.")
                        },
                        "angle": {
                            "type": "array",
                            "items": {
                                "type": "number",
                                "format": "float",
                                "minimum": -180,
                                "maximum": 180,
                                "description": "Angle in degrees for the first and second joint of the arm."
                            },
                            "minItems": 2,
                            "maxItems": 2,
                            "description": "A 2-item list of angles in degrees corresponding to the first and second joint respectively."
                        },
                        "angular_velocity": {
                            "type": "array",
                            "items": {
                                "type": "number"
                            },
                            "minItems": 2,
                            "maxItems": 2,
                            "description": ("Vector of two elements corresponding to the angular velocity of the first and second arm respectively."
                                            "If feedback refers to a location with positive reward, but without specifying the speed, set the angular speed to [0, 0].")
                        },
                        "reward": {
                            "type": "integer",
                            "enum": [-1, 1],
                            "description": "The reward value, which can be +1 for good performance or -1 for bad performance."
                        }
                    },
                    "required": ["fingertip_position", "reward"],
                    "minProperties": 2,
                    "description": ("Dictionary containing information about the state and its reward. "
                                    "Must have the 'reward' and 'fingertip_position' keys and optionally other keys: "
                                    "'angle', or 'angular_velocity'.")
                },
                "description": "List of states described by the feedback along with their reward implied by the feedback."
            }
        },
        "required": ["referred_steps"]
    }
}