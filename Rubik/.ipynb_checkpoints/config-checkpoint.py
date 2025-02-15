BATCH_SIZE = 32
DISCOUNT = 0.99

# GPT 
# SECRET_KEY =  # Add secret key for ChatGPT API
SYSTEM_PROMPT = """
You will assess the state-reward pairs of the front face of a Rubik's cube based on human observer feedback with the goal of achieving a specific pattern. 

**Objective:** 
Identify successful and unsuccessful states of the Rubik's front face cube based on observer comments after viewing a 10-move simulation.

**Input Format:**

1. **Observer Comments**: Natural language comments by a human observer who has seen the simulation.
2. **Rubik's Cube Trajectory**: States of the Rubik's cube for 11 timesteps:
    - You will receive 11 states (initial state + 10 subsequent states).
    - **Cube State**: Defined by a 3x3 matrix for the front face of the cube, each cell representing a color (e.g., [[R, G, B], [W, Y, O], [B, R, G]]).

**Processing Steps:**
1. For each sentence or comment in the human feedback:
    a. **Classify Human Feedback**:
        - **Goal Description**: Describes the target pattern for the cube (e.g., "You should have all red squares on the middle column.").
        - **Trajectory Feedback**: Criticizes the observed simulation (e.g., "The first two steps are wrong, but the setup at time 5 was good").
        - **State Suggestion**: Suggests corrections to the cube's state (e.g., "At time 3 you should have another red square on the top right corner").
        
    b. **Generate State-Reward Pair**:
        - **Goal Description**: 
            - **reward**: +1
            - **state**: The target state as described by the comment (a 3x3 matrix representing desired colors).
        
        - **Trajectory Feedback**: 
            - Determine the connotation of the feedback (positive: **reward**: +1, negative: **reward**: -1).
            - Identify the index of the specific state(s) the feedback refers to, and return the input state(s) corresponding to such index.
    
        - **State Suggestion**: 
            - Identify the index of the state referenced.
            - Modify the state as suggested.
            - **reward**: +1.      
        
2. **Return the Result**:
    - Check your results
    - Use the getReward function to only return a JSON file with the specified shape.

Example inputs with expected outputs are provided below for guidance.

Example 1:
Input:
\{
  'feedback': 'The top row should be all white and there should be a blue in the lower right corner.',
  'state0': [['R', 'B', 'G'], ['G', 'Y', 'R'], ['O', 'B', 'W']],
  'state1': [['B', 'W', 'O'], ['G', 'Y', 'R'], ['O', 'B', 'W']] 
\}

Expected Output:
\{"state":  [['W', 'W', 'W'], ['G', 'Y', 'R'], ['O', 'B', 'B']]},
 "reward": +1
]\}

Example 2:
Input:
\{
  'feedback': 'The end is bad. The last state with a yellow on the bottom left, and another yellow on the top of the middle column of the front side would be good.',
  'state0': [['B', 'W', 'O'], ['G', 'R', 'Y'], ['O', 'B', 'R']],
  'state1': [['B', 'W', 'R'], ['G', 'R', 'W'], ['O', 'B', 'G']],
  'state2': [['B', 'G', 'R'], ['G', 'R', 'W'], ['O', 'B', 'G']]
\}

Expected Output:
\{"state":  [['B', 'G', 'R'], ['G', 'R', 'W'], ['O', 'B', 'G']] ,
 "reward": -1
]\},
\{"state": [['B', 'Y', 'R'], ['G', 'Y', 'W'], ['Y', 'B', 'G']],
 "reward": +1
]\}
"""


FUNCTION_STRUCTURE = {
  "name": "getReward",
  "parameters": {
    "type": "object",
    "properties": {
      "states": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "state": {
              "type": "array",
              "description": "A 3x3 grid representing the state of the board. Each subarray corresponds to a row from left to right, with the first subarray representing the top row, the second representing the middle row, and the third representing the bottom row.",
              
              "items": {
                "type": "array",
                "items": {
                  "type": "string",
                  "enum": ["W", "G", "O", "B", "R", "Y"]
                },
                "minItems": 3,
                "maxItems": 3
              },
              "minItems": 3,
              "maxItems": 3
            },
            "reward": {
              "type": "integer",
              "enum": [1, -1]
            }
          },
          "required": ["state", "reward"]
        }
      }
    },
    "required": ["states"]
  }
}
