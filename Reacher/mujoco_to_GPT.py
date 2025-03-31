from config import COOR_MIN, COOR_MAX, NUM_SEGMENTS
import string
import random
import numpy as np
import os

def extract_features(observations, num_segments=NUM_SEGMENTS):
    """
    Extracts key features from the given observations at each time step.

    Args:
        observations (list): A list containing the observations at each timestep of the 'Reacher-v4' environment.
        num_segments (int): The number of segments that will be used to divide the width and height to create 
                            the grid. Default is defined by the global NUM_SEGMENTS variable.
    
    Returns:
        list: A list of fingertip positions in algebraic notation for each timestep.
        list: A list containing the angles in degrees of the first and second arms.
        list: A list containing the angular speed (in radians per second) of the first and second arm
            at each timestep, with each speed value rounded to 2 decimal places.
    """
    fingertip_positions = []
    arm_angles = []
    angular_speed = []
    
    for ii in range(1, len(observations)):
        # Calculate fingertip position
        current_observation = observations[ii][2:]
        current_fingertip_position = (current_observation[8]+current_observation[4], current_observation[9]+current_observation[5])
        fingertip_positions.append(map_coordinates_to_algebraic(current_fingertip_position, num_segments))
        # Calculate arm angles
        arm_angles.append(_get_angles(current_observation[:4]))
        # Store angular speed
        angular_speed.append([round(current_observation[6].item(),2), round(current_observation[7].item(),2)])
        
    
    return fingertip_positions, arm_angles, angular_speed

def _get_angles(tensor):
    """
    Convert cosine and sine values of angles to angles in degrees.

    Args:
        tensor (numpy.ndarray): A 1D numpy array with four entries representing
                cosine of the angle of the first arm, cosine of the angle of the second arm,
                sine of the angle of the first arm, and sine of the angle of the second arm.

    Returns:
        tuple: A tuple containing the angles of the first and second arms in degrees rounded to 1 decimal point.
    """

    # Extract cosine and sine values
    cos_angle_1, cos_angle_2, sin_angle_1, sin_angle_2 = tensor

    # Calculate angles in radians using arctangent
    angle_1_rad = np.arctan2(sin_angle_1, cos_angle_1)
    angle_2_rad = np.arctan2(sin_angle_2, cos_angle_2)

    # Convert radians to degrees
    angle_1_deg = np.degrees(angle_1_rad)
    angle_2_deg = np.degrees(angle_2_rad)

    return [round(angle_1_deg.item(), 1), round(angle_2_deg.item(), 1)]
    
def map_coordinates_to_algebraic(coords, num_segments=NUM_SEGMENTS):
    """
    Maps a pair of coordinates to algebraic notation.

    Args:
        coords (tuple): A tuple (x, y) of coordinate values to map.
        num_segments (int, optional): The number of segments that will be used to divide the width and height to create 
                            the grid. Default is defined by the global NUM_SEGMENTS variable.

    Returns:
        str: The algebraic notation of the mapped coordinates.
    """
    col = coor_to_letter(coords[0], num_segments)
    row = _coor_to_row(coords[1], num_segments)

    return '{}{}'.format(col, row)

def coor_to_letter(coor, num_segments=NUM_SEGMENTS):
    """
    Converts a horizontal coordinate to a letter representation.

    Args:
        coor (float): The horizontal coordinate to be converted to letter(s).
        num_segments (int, optional): The number of segments that will be used to divide the width to create 
                            the grid. Default is defined by the global NUM_SEGMENTS variable.

    Returns:
        str: The letter representation of the horizontal coordinate.
    """
    normalized_coor = _check_and_normalize(coor, num_segments)
    index = int(normalized_coor * num_segments)
    
    if num_segments <= 26: # Single letter
        return string.ascii_lowercase[index]
    
    # Calculate the two-letter combination
    first_letter_index = index // 26 
    second_letter_index = index % 26
    
    first_letter = string.ascii_lowercase[first_letter_index]
    second_letter = string.ascii_lowercase[second_letter_index]
    return first_letter + second_letter

def _coor_to_row(coor, num_segments):
    """
    Maps a vertical coordinate to a row number.

    Args:
        coor (float): The vertical coordinate to be converted to row number.
        num_segments (int): The number of segments that will be used to divide the height to create 
                            the grid.

    Returns:
        int: The row number corresponding to the vertical coordinate.
    """
    normalized_coor = _check_and_normalize(coor, num_segments)
    mapped_value = int(normalized_coor * (num_segments - 1) + 1)
    return min(mapped_value, num_segments)

def _check_and_normalize(coor, num_segments):
    """
    Checks the validity of the coordinate and normalizes it.

    Args:
        coor (float): The coordinate to be checked and normalized.
        num_segments (int): Number of segments per height and width in the grid.

    Raises:
        ValueError: If num_segments is less than 1.

    Returns:
        float: The normalized coordinate in the range [0, 1].
    """
    if num_segments < 1:
        raise ValueError("num_segments must be at least 1")
        
    elif coor < COOR_MIN:
        print("Coordinate out of range: ", coor)
        return 0
    elif coor > COOR_MAX:
        print("Coordinate out of range: ", coor)
        return 1    
    # Normalize x to a range of 0 to 1
    normalized_coor = (coor - COOR_MIN) / (COOR_MAX - COOR_MIN)

    return normalized_coor

def get_landmarks():
    """Gets landmark locations and returns them in a dictionary.

    This function reads the number of circles, their X and Y coordinates, and the color names
    from a configuration module. It maps the coordinates to an algebraic format and associates them
    with their corresponding color names in a dictionary.

    Returns:
        dict: A dictionary with color names as keys and their algebraic coordinate positions as values.
    """
    from config import COLOR_NAMES, X_COORDINATES, Y_COORDINATES, NUM_CIRCLES
    landmark_locations = {}
    for ii in range(NUM_CIRCLES):
        pos =  map_coordinates_to_algebraic([X_COORDINATES[ii], Y_COORDINATES[ii]])
        landmark_locations.update({COLOR_NAMES[ii]: pos})
    return landmark_locations

def fill_missing_entries(entries, exp_angle, exp_angular_velocity, exp_target_location):
    """Fills missing 'target_location', 'angle' or 'angular_velocity' values in a list of entries.

    If an entry in the list does not contain the 'angle' or 'angular_velocity' fields,
    this function randomly samples values from provided experiment data lists and
    fills them in correspondingly. It also sets the 'target_location' for each entry
    based on the provided experimental target location coordinates.

    Args:
        entries (list of dict): A list of dictionaries representing the entries that might have
            missing 'angle' or 'angular_velocity' fields.
        exp_angle (list of float): A list of possible 'angle' values to sample from.
        exp_angular_velocity (list of float): A list of possible 'angular_velocity' values to sample from.
        exp_target_location (list of float): The experimental target location coordinates to be set for
            each entry, [x, y].

    Returns:
        list of dict: The updated list of entries with no missing 'angle' or 'angular_velocity' fields
            and with 'target_location' set to the provided experimental target location coordinates.
    """
    for entry in entries:
        if 'angle' not in entry:
            entry['angle'] = random.choice(exp_angle)
        if 'angular_velocity' not in entry:
            entry['angular_velocity'] = random.choice(exp_angular_velocity)
        entry['target_location'] = exp_target_location
    return entries

def get_human_feedback(video_directory, feedback_name):
    """
    Collects and returns human feedback for videos from markdown files.

    Loops through a list of markdown files 
    containing feedback from a specified directory, and aggregates the content 
    into a list after stripping out newline characters and empty lines.

    Args:
        video_directory (str): The path to the directory where feedback files are located.
        feedback_name (list): A list of file names to collect feedback for.

    Returns:
        list: Aggregated human feedback from all specified markdown files.

    Raises:
        IOError: If there is an issue opening a file.
    """
    human_feedback = []
    for video_name in feedback_name:
        HF_file_path = os.path.join(video_directory,f"{video_name}.md")
        with open(HF_file_path, 'r') as file:
            content = file.readlines()
        # Remove newline characters and empty lines
        human_feedback = human_feedback + [line.strip() for line in content if line.strip()]
    return human_feedback
