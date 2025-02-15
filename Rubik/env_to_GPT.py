import os

def obs_to_state(observation):
    # Mapping of integers to colors
    color_mapping = {0: 'W', 1: 'R', 2: 'B', 3: 'O', 4: 'G', 5: 'Y'}
    
    # Function to convert 1D array to 3x3 face
    def to_face(array):
        return [array[0:3], array[3:6], array[6:9]]
    
    # Apply color mapping to observation
    colored_observation = [color_mapping[num] for num in observation]
    
    # Define the faces using the correct slices of the observation array
    state = to_face(colored_observation[9:18])
    
    return state


def get_human_feedback(video_directory, video_name):
    """
    Collects and returns human feedback for videos from markdown files.

    Loops through a list of markdown files 
    containing feedback from a specified directory, and aggregates the content 
    into a list after stripping out newline characters and empty lines.

    Args:
        video_directory (str): The path to the directory where feedback file is located.
        video_name (str): File name to collect feedback for.

    Returns:
        str: Human feedback from specified markdown files.

    Raises:
        IOError: If there is an issue opening a file.
    """
    HF_file_path = os.path.join(video_directory,f"HumanFeedback_{video_name}.md")
    with open(HF_file_path, 'r') as file:
        content = file.readlines()
    # Remove newline characters and empty lines
    human_feedback =[line.strip() for line in content if line.strip()]
    return human_feedback