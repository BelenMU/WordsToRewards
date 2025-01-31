import matplotlib.pyplot as plt
import numpy as np
import random
from scipy.stats import beta
import spacy
from spacy import displacy 
from tqdm import tqdm # For progress bar
import copy
from transformers import pipeline
import os
import math
import json

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# Sentiment analyzers and word embeddders
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
vader_analyzer = SentimentIntensityAnalyzer()

# Import functions
from science.draw_map import map_loc_landmark, show_trajectory_on_map
from science.config import LIST_SEGMENTS, LIST_LOCATIONS, NUM_STEPS, MASK_SEGMENTS, nlp 
from science.config import GPT_MODEL, TEMPERATURE, SYSTEM_MESSAGE, FUNCTION_STRUCTURE, SECRET_KEY, SYSTEM_MESSAGE_NOCERTAINTY, FUNCTION_STRUCTURE_NOCERTAINTY
from science.data_to_GPT import algebraic_to_index, create_landmark_dict, GridState, vector_to_algebraic

import openai
openai.api_key = SECRET_KEY

def get_HF():
    """
    Prompts the user for their critique of the trajectory, converts the input to
    lowercase, and then returns it.

    Returns:
        str: A lowercase string of the user's inputted critique.    
    """
    text = input("Please enter your critique of the trajectory: ")
    text = text.lower()  # All to lower case
    return text

def sentiment_analyzer_vader(text, min_score=0):
    """
    Analyzes the sentiment of a given text using the VADER Sentiment Analysis tool and
    categorizes it as positive or negative based on the comparative scores. It also
    provides a certainty score representing the strength of the sentiment.

    Parameters:
      text (str): The text input whose sentiment needs to be analyzed.
      min_score (float, optional): A minimum threshold for the certainty score. Defaults to 0.

    Returns:
      tuple (str, float): A tuple containing:
        - label_reward (str): The sentiment label ('POS' for positive, 'NEG' for negative).
        - certainty_reward (float): The certainty score representing the strength of
                                the identified sentiment (adjusted by min_score if needed).
    """
    vader_analyzer = SentimentIntensityAnalyzer()
    score_dict = vader_analyzer.polarity_scores(text)
    neg_score = score_dict["neg"]
    pos_score = score_dict["pos"]
    if pos_score >= neg_score:
        label_reward = 'POS'
        certainty_reward = max(pos_score, min_score)
    else:
        label_reward ='NEG'
        certainty_reward = max(neg_score, min_score)

    return label_reward, certainty_reward


def sentiment_trajectory(trajectory, grid_height, out_reward, out_certainty, label_reward, certainty_reward, penalty = 1):
    """Assigns a sentiment label and certainty level to each state in the trajectory.

    Successive occurrences of the same state within the trajectory only 
    have their sentiment and certainty recorded once to avoid redundancy.

    Args:
        trajectory (numpy.ndarray): A 2D array where each row corresponds to a step in the
                                    trajectory. Each row should contain at least two elements,
                                    interpreted as the 2D position (row, column) of the state.
        grid_height (int): The height of the grid that the trajectory is based on; used to
                           calculate the linear state index from 2D positions.
        out_reward (dict): A dictionary to be updated that maps each unique state index
                           to the associated sentiment label ('POS' for positive, 'NEG' for
                           negative). Should be passed as an empty dictionary by the caller.
        out_certainty (dict): A dictionary to be updated that maps each unique state index
                              to the associated certainty level of the sentiment (likelihood
                              of being positive or negative). Should be passed as an empty 
                              dictionary by the caller.
        label_reward (str): The sentiment label to be assigned ('POS' or 'NEG').
        certainty_reward (float): The certainty level of the sentiment (a numerical value).
        penalty (int): Penalty on the certainty level (a numerical value from 0 to 1).

    Returns:
        tuple: A tuple of two dictionaries:
            - out_reward: Updated mapping of state indices to sentiment labels.
            - out_certainty: Updated mapping of state indices to sentiment certainty levels.
    """

    state_index_prev = -1  # To detect instances where agent is stuck
    NUM_STEPS = len(trajectory)  # Assumes trajectory is a numpy array with a defined length
    for temp_ind in np.arange(NUM_STEPS):
        # Pass from trajectory to state index
        state_index = trajectory[temp_ind, 0] * grid_height + trajectory[temp_ind, 1]
        if state_index != state_index_prev: # Only add the reward once, even when it gets stuck
            if state_index not in out_reward: 
                out_reward[state_index] = [label_reward] 
                out_certainty[state_index] = [certainty_reward*penalty] 
            else:
                out_reward[state_index].append(label_reward)  # Append
                out_certainty[state_index].append(certainty_reward*penalty)
            state_index_prev = state_index 

    return out_reward, out_certainty

def sample_reward_HF_Bernoulli_Vader_trajectory(trajectory, images, loc_landmarks, road, grid_width,
                                                grid_height, pos_init, pixel_landmarks, list_landmarks):
    """
    Visualize a trajectory on a map, collect human feedback and then use sentiment analysis (VADER) 
    to estimate rewards and certainty levels associated with the whole trajectory.

    Args:
        trajectory (list): A list of coordinates representing the trajectory of the agent.
        images (list): A sequential list of images representing the landmarks.
        loc_landmarks (dict): A dictionary mapping landmarks to their physical locations.
        road (np.array): An array containing the road's locations, i.e., the optimal path to be taught to the agent.
        grid_width (int): The width of the grid representation of the map.
        grid_height (int): The height of the grid representation of the map.
        pos_init (tuple): The initial position of the agent.
        pixel_landmarks (dict): A dictionary mapping landmarks to their pixel coordinates.
        list_landmarks (list): A list of the names of the landmarks.

    Returns:
        tuple: A 3-element tuple containing:
            - human_feedback (str): The feedback provided by the human.
            - out_reward (dict): A dictionary with trajectory points as keys and associated rewards as values.
            - out_certainty (dict): A dictionary with trajectory points as keys and associated certainty as values.
    """
    # Show trajectory on map
    trajectory = show_trajectory_on_map(images, loc_landmarks, road, grid_width, grid_height, trajectory, pos_init)

    # Ask human for feedback
    human_feedback = get_HF()

    # Initialize dictionaries
    out_reward = {}
    out_certainty = {}
    
    # Measure sentiment of the whole text
    label_reward, certainty_reward = sentiment_analyzer_vader(human_feedback)
    out_reward, out_certainty = sentiment_trajectory(trajectory, grid_height, out_reward, out_certainty, label_reward, certainty_reward)
      
    return human_feedback, out_reward, out_certainty

def sentence_to_locations(sent, label_reward, certainty_reward, trajectory, grid_width,
                          grid_height, list_landmarks, pixel_landmarks, out_certainty, out_reward):
    """
    Translates sentences to locations on a grid map, with their corresponding reward and certainty values.
    
    Args:
        sent (Token Sequence): The sequence of tokens representing the sentence.
        label_reward (float): The reward value associated with the sentence.
        certainty_reward (float): The certainty associated with the reward's label.
        trajectory (np.ndarray): Array representing the trajectory of movement on the grid.
        grid_width (int): Width of the grid map.
        grid_height (int): Height of the grid map.
        list_landmarks (List): List of names of landmarks.
        pixel_landmarks (np.ndarray): Array representing location of landmarks as pixels on the grid.
        out_certainty (dict): Output dictionary mapping state indices to a list of certainty values.
        out_reward (dict): Output dictionary mapping state indices to a list of reward values.

    Returns:
        tuple: 
            state_detected (bool): Indicates if a relevant state has been detected.
            out_reward (dict): Updated dictionary of reward values.
            out_certainty (dict): Updated dictionary of certainty values.
    """
    state_detected = False
    trajectory_map = np.zeros([grid_width, grid_height])
    trajectory_map[trajectory[:, 0], trajectory[:, 1]] = 1
    
    # Process each token in the sentence
    for token in sent:
        # Process non-punctuation, non-stop words
        if not token.is_punct and not token.is_stop:
            similarity_scores = [word.similarity(token) for word in LIST_SEGMENTS]
            max_value = max(similarity_scores)
            if max_value > 0.5:
                state_detected = True                    
                ind_loc = similarity_scores.index(max_value)
                certainty_loc = certainty_reward * MASK_SEGMENTS[ind_loc, :]                                    
                state_index_prev = -1 # To detect instances where agent is stuck
                non_zero_indices = np.nonzero(certainty_loc)
                non_zero_indices = non_zero_indices[0][:]
                for temp_ind in non_zero_indices:         
                    # Pass from trajectory to state index
                    state_index = trajectory[temp_ind, 0] * grid_height + trajectory[temp_ind, 1] 
                    if state_index == state_index_prev: # Stuck
                        stuck_uncertainty = out_certainty[state_index]
                        stuck_uncertainty[-1] = max(stuck_uncertainty[-1], certainty_loc[temp_ind])
                        out_certainty[state_index] = stuck_uncertainty
                    else: 
                        if state_index not in out_reward: 
                            out_reward[state_index] = [label_reward] 
                            out_certainty[state_index] = [certainty_loc[temp_ind]] 
                        else:
                            out_reward[state_index].append(label_reward)  # Append
                            out_certainty[state_index].append(certainty_loc[temp_ind])
                        state_index_prev = state_index    

                #print("\t", token.text, "->",  list_segment[ind_loc].text, "->" , label_reward, ": ", certainty_reward) 
            elif token.pos_ == "NOUN":
                similarity_scores = [word.similarity(token) for word in list_landmarks]
                max_value = max(similarity_scores)
                if max_value > 0.45:
                    state_detected = True
                    ind_landmark = similarity_scores.index(max_value)

                    if token.head.similarity(nlp("of")) > 0.97:
                        loc = token.head.head
                        if loc.similarity(nlp("part")) > 0.97:
                            for child in loc.children:
                                if child.dep_ == 'amod': 
                                    loc = child
                    else:
                        loc = token.head
                        if loc.dep_ == 'ROOT' or loc.dep_ == 'AUX':
                            loc = nlp("in")
                    if loc.similarity(nlp("below")) > 0.99: # Hardwire below, because if not it relates it to above!
                        ind_loc = 1
                    else:
                        similarity_loc = [word.similarity(loc) for word in LIST_LOCATIONS]
                        max_loc = max(similarity_loc)
                        ind_loc = similarity_loc.index(max_loc)
                    feedback_map = map_loc_landmark(ind_loc, ind_landmark, certainty_reward,\
                                                    pixel_landmarks, grid_width, grid_height) # Before trajectory
                    feedback_map_times_trajectory = feedback_map * trajectory_map
                    #print("Feedback map sum:", sum(sum(feedback_map_times_trajectory)))
                    if sum(sum(feedback_map_times_trajectory)) != 0:
                        # Trajectory detected at defined spot -> Evaluative                      
                        feedback_map = feedback_map_times_trajectory
                    #else:          
                        # Trajectory not detected at defined spot -> Descriptive or imperative

                    non_zero_indices = np.nonzero(feedback_map)
                    for loop in np.arange(non_zero_indices[0].size):         
                        # Pass from trajectory to state index
                        temp_ind_x = non_zero_indices[0][loop]
                        temp_ind_y = non_zero_indices[1][loop]
                        state_index = temp_ind_x * grid_height + temp_ind_y
                        if state_index not in out_reward: 
                            out_reward[state_index] = [label_reward] 
                            out_certainty[state_index] = [feedback_map[temp_ind_x, temp_ind_y]] 
                        else:
                            out_reward[state_index].append(label_reward)
                            out_certainty[state_index].append(feedback_map[temp_ind_x, temp_ind_y])
                            
    return state_detected, out_reward, out_certainty
def sample_reward_from_HF(trajectory, human_feedback, grid_width, grid_height, list_landmarks, pixel_landmarks):  
    """
    Processes human feedback to obtain state level rewards. If no state is detected on a sentence, assign to whole trajectory.
    
    Computes the sentiment of each sentence to generate reward values and certainty metrics.
    Performs language processing to map each reward to locations in the grid.

    Args:
        trajectory (list): The list of coordinates representing the agent's trajectory.
        human_feedback (str): The feedback provided by the human.
        grid_width (int): The width of the grid representing the environment.
        grid_height (int): The height of the grid representing the environment.
        list_landmarks (List): List of names of landmarks.
        pixel_landmarks (np.ndarray): Array representing location of landmarks as pixels on the grid.

    Returns:
        tuple: consisting of
            - out_reward (dict): A dictionary mapping locations to reward values.
            - out_certainty (dict): A dictionary mapping locations to certainty values.
    """
    doc = nlp(human_feedback)
    
    # Initialize dictionaries
    out_reward = {}
    out_certainty = {}

    for sent in doc.sents:
        label_reward, certainty_reward = sentiment_analyzer_vader(sent.text, min_score=0.05)
        state_detected, out_reward, out_certainty = sentence_to_locations(
            sent, label_reward, certainty_reward, trajectory, grid_width, grid_height,
            list_landmarks, pixel_landmarks, out_certainty, out_reward
        )

        if not state_detected:
            # Assume (with lower confidence) that sentence refers to the whole trajectory
            out_reward, out_certainty = sentiment_trajectory(
                trajectory, grid_height, out_reward, out_certainty, label_reward, certainty_reward, penalty=0.5
            )
    return out_reward, out_certainty
    
def sample_reward_HF_Bernoulli_Vader_defaultAll(trajectory, images, loc_landmarks, road, grid_width,
                                                grid_height, pos_init, list_landmarks, pixel_landmarks):
    """
    Processes human feedback to obtain state level rewards. If no state is detected on a sentence, assign to whole trajectory.
    
    Computes the sentiment of each sentence to generate reward values and certainty metrics.
    Performs language processing to map each reward to locations in the grid.

    Args:
        trajectory (list): The list of coordinates representing the agent's trajectory.
        images (np.array): The images of the landmarks.
        loc_landmarks (dict): The dict mapping landmarks to their locations.
        road (np.array): An array containing the road's locations, i.e., the optimal path to be taught to the agent.
        grid_width (int): The width of the grid representing the environment.
        grid_height (int): The height of the grid representing the environment.
        pos_init (tuple): The initial position of the agent on the grid.
        list_landmarks (List): List of names of landmarks.
        pixel_landmarks (np.ndarray): Array representing location of landmarks as pixels on the grid.

    Returns:
        tuple: consisting of
            - human_feedback (str): The feedback provided by the human.
            - out_reward (dict): A dictionary mapping locations to reward values.
            - out_certainty (dict): A dictionary mapping locations to certainty values.
    """
    # Show trajectory on map
    trajectory = show_trajectory_on_map(images, loc_landmarks, road, grid_width, grid_height, trajectory, pos_init)

    # Ask human for feedback
    human_feedback = get_HF()    
    
    out_reward, out_certainty = sample_reward_from_HF(trajectory, human_feedback, grid_width, grid_height, list_landmarks, pixel_landmarks)
    return human_feedback, out_reward, out_certainty

def sample_reward_HF_Bernoulli_Vader(trajectory, images, loc_landmarks, road, grid_width, \
                                     grid_height, pos_init, list_landmarks, pixel_landmarks):
    """
    Processes human feedback to obtain state level rewards. If no state is detected on any sentence, assign to whole trajectory.
    
    Computes the sentiment of each sentence to generate reward values and certainty metrics.
    Performs language processing to map each reward to locations in the grid.

    Args:
        trajectory (list): The list of coordinates representing the agent's trajectory.
        images (np.array): The images of the landmarks.
        loc_landmarks (dict): The dict mapping landmarks to their locations.
        road (np.array): An array containing the road's locations, i.e., the optimal path to be taught to the agent.
        grid_width (int): The width of the grid representing the environment.
        grid_height (int): The height of the grid representing the environment.
        pos_init (tuple): The initial position of the agent on the grid.
        list_landmarks (List): List of names of landmarks.
        pixel_landmarks (np.ndarray): Array representing location of landmarks as pixels on the grid.

    Returns:
        tuple: consisting of
            - human_feedback (str): The feedback provided by the human.
            - out_reward (dict): A dictionary mapping locations to reward values.
            - out_certainty (dict): A dictionary mapping locations to certainty values.
    """
    # Show trajectory on map
    trajectory = show_trajectory_on_map(images, loc_landmarks, road, grid_width, grid_height, trajectory, pos_init)

    # Ask human for feedback
    human_feedback = get_HF()
    doc = nlp(human_feedback)
    
    # Initialize dictionaries
    out_reward = {}
    out_certainty = {}
    state_detected = False
    
    for sent in doc.sents: # Divide feedback into separate sentences   
        # Measure sentiment of sentence
        label_reward, certainty_reward = sentiment_analyzer_vader(sent.text)    

        # Translate sentences to locations in the map
        sent_state_detected, out_reward, out_certainty = sentence_to_locations(sent, label_reward, \
                                                                         certainty_reward, trajectory, \
                                                                         grid_width, grid_height,\
                                                                         list_landmarks, pixel_landmarks, \
                                                                         out_certainty, out_reward)
        if sent_state_detected:
            state_detected = True


    if state_detected:
        return human_feedback, out_reward, out_certainty
    else:
        # Assume the whole human feedback refers to the whole trajectory
        label_reward, certainty_reward = sentiment_analyzer_vader(human_feedback)
        out_reward, out_certainty = sentiment_trajectory(trajectory, grid_height, out_reward, out_certainty, \
                                                         label_reward, certainty_reward)
        return human_feedback, out_reward, out_certainty

def sample_reward_HF_GPT(trajectory, images, loc_landmarks, road, grid_width, \
                        grid_height, pos_init, pixel_landmarks, list_landmarks):
    """
    Processes human feedback with GPT4 to obtain state level rewards.

    Args:
        trajectory (list): The list of coordinates representing the agent's trajectory.
        images (np.array): The images of the landmarks.
        loc_landmarks (dict): The dict mapping landmarks to their locations.
        road (np.array): An array containing the road's locations, i.e., the optimal path to be taught to the agent.
        grid_width (int): The width of the grid representing the environment.
        grid_height (int): The height of the grid representing the environment.
        pos_init (tuple): The initial position of the agent on the grid.
        pixel_landmarks (np.ndarray): Array representing location of landmarks as pixels on the grid.
        list_landmarks (List): List of names of landmarks.

    Returns:
        tuple: consisting of
            - human_feedback (str): The feedback provided by the human.
            - out_reward (dict): A dictionary mapping locations to reward values.
            - out_certainty (dict): A dictionary mapping locations to certainty values.
    """
    # Show trajectory on map
    _ = show_trajectory_on_map(images, loc_landmarks, road, grid_width, grid_height, trajectory, pos_init)

    # Ask human for feedback
    human_feedback = get_HF()
    doc = nlp(human_feedback)
    
    # Initialize dictionaries
    out_reward = {}
    out_certainty = {}
    state_detected = False

    max_index = grid_width * grid_height
    landmark_dict = create_landmark_dict(pixel_landmarks, list_landmarks)
    trajectory_algebraic = vector_to_algebraic(np.array(trajectory))
    
    for sent in doc.sents: # Divide feedback into separate sentences   
        state = GridState(sent.text, landmark_dict, trajectory_algebraic)
        gptResponse = openai.chat.completions.create(
                model = GPT_MODEL,
                temperature= TEMPERATURE,
                messages= [SYSTEM_MESSAGE, 
                           {"role": "user", "content": state.get_prompt()}],
                functions=[FUNCTION_STRUCTURE],
                    function_call= { "name": "getReward" })
            
        json_data = gptResponse.choices[0].message.function_call.arguments
        reward_data = json.loads(json_data)
        ind_reward = algebraic_to_index(reward_data['locations'],  grid_height)
        ind_reward = list(np.array(ind_reward)[np.array(ind_reward) < max_index])

        for temp_ind in range(len(ind_reward)):
            state_index = ind_reward[temp_ind]
            if state_index not in out_reward: 
                out_reward[state_index] = [reward_data['label']] 
                out_certainty[state_index] = [reward_data['certainty_locations'][temp_ind]] 
            else: # Append
                out_reward[state_index].append(reward_data['label'])  
                out_certainty[state_index].append(reward_data['certainty_locations'][temp_ind])
                
    return human_feedback, out_reward, out_certainty

def sample_reward_HF_GPT_nocertainty(trajectory, images, loc_landmarks, road, grid_width, \
                        grid_height, pos_init, pixel_landmarks, list_landmarks):
    """
    Processes human feedback with GPT4 to obtain state level rewards.

    Args:
        trajectory (list): The list of coordinates representing the agent's trajectory.
        images (np.array): The images of the landmarks.
        loc_landmarks (dict): The dict mapping landmarks to their locations.
        road (np.array): An array containing the road's locations, i.e., the optimal path to be taught to the agent.
        grid_width (int): The width of the grid representing the environment.
        grid_height (int): The height of the grid representing the environment.
        pos_init (tuple): The initial position of the agent on the grid.
        pixel_landmarks (np.ndarray): Array representing location of landmarks as pixels on the grid.
        list_landmarks (List): List of names of landmarks.

    Returns:
        tuple: consisting of
            - human_feedback (str): The feedback provided by the human.
            - out_reward (dict): A dictionary mapping locations to reward values.
            - out_certainty (dict): A dictionary mapping locations to certainty values.
    """
    # Show trajectory on map
    _ = show_trajectory_on_map(images, loc_landmarks, road, grid_width, grid_height, trajectory, pos_init)

    # Ask human for feedback
    human_feedback = get_HF()
    doc = nlp(human_feedback)
    
    # Initialize dictionaries
    out_reward = {}
    out_certainty = {}
    state_detected = False

    max_index = grid_width * grid_height
    landmark_dict = create_landmark_dict(pixel_landmarks, list_landmarks)
    trajectory_algebraic = vector_to_algebraic(np.array(trajectory))
    
    for sent in doc.sents: # Divide feedback into separate sentences   
        state = GridState(sent.text, landmark_dict, trajectory_algebraic)
        gptResponse = openai.chat.completions.create(
                model = GPT_MODEL,
                temperature= TEMPERATURE,
                messages= [SYSTEM_MESSAGE_NOCERTAINTY, 
                           {"role": "user", "content": state.get_prompt()}],
                functions=[FUNCTION_STRUCTURE_NOCERTAINTY],
                    function_call= { "name": "getReward" })
            
        json_data = gptResponse.choices[0].message.function_call.arguments
        reward_data = json.loads(json_data)
        ind_reward = algebraic_to_index(reward_data['locations'],  grid_height)
        ind_reward = list(np.array(ind_reward)[np.array(ind_reward) < max_index])

        for temp_ind in range(len(ind_reward)):
            state_index = ind_reward[temp_ind]
            if state_index not in out_reward: 
                out_reward[state_index] = [reward_data['label']] 
                out_certainty[state_index] = [1] 
            else: # Append
                out_reward[state_index].append(reward_data['label'])  
                out_certainty[state_index].append(1)
                
    return human_feedback, out_reward, out_certainty

def sample_reward_HF_GPT_nocertainty_v2(trajectory, images, loc_landmarks, road, grid_width, \
                        grid_height, pos_init, pixel_landmarks, list_landmarks):
    """
    Processes human feedback with GPT4 to obtain state level rewards.

    Args:
        trajectory (list): The list of coordinates representing the agent's trajectory.
        images (np.array): The images of the landmarks.
        loc_landmarks (dict): The dict mapping landmarks to their locations.
        road (np.array): An array containing the road's locations, i.e., the optimal path to be taught to the agent.
        grid_width (int): The width of the grid representing the environment.
        grid_height (int): The height of the grid representing the environment.
        pos_init (tuple): The initial position of the agent on the grid.
        pixel_landmarks (np.ndarray): Array representing location of landmarks as pixels on the grid.
        list_landmarks (List): List of names of landmarks.

    Returns:
        tuple: consisting of
            - human_feedback (str): The feedback provided by the human.
            - out_reward (dict): A dictionary mapping locations to reward values.
            - out_certainty (dict): A dictionary mapping locations to certainty values.
    """
    # Show trajectory on map
    _ = show_trajectory_on_map(images, loc_landmarks, road, grid_width, grid_height, trajectory, pos_init)

    # Ask human for feedback
    human_feedback = get_HF()
    
    # Initialize dictionaries
    out_reward = {}
    out_certainty = {}
    state_detected = False

    max_index = grid_width * grid_height
    landmark_dict = create_landmark_dict(pixel_landmarks, list_landmarks)
    trajectory_algebraic = vector_to_algebraic(np.array(trajectory))
     
    state = GridState(human_feedback, landmark_dict, trajectory_algebraic)
    gptResponse = openai.chat.completions.create(
            model = GPT_MODEL,
            temperature= TEMPERATURE,
            messages= [SYSTEM_MESSAGE_NOCERTAINTY, 
                       {"role": "user", "content": state.get_prompt()}],
            functions=[FUNCTION_STRUCTURE_NOCERTAINTY],
                function_call= { "name": "get_reward_v2" })

    json_data = gptResponse.choices[0].message.function_call.arguments 
    
    reward_data = json.loads(json_data)
    #print(f"reward_data: {reward_data}")
    reward_states = reward_data['states']
    
    for state in reward_states:
        #print(f"state: {state}")
        #print(f"state['location']: {state['location']}")
        loc = state['location']
        if not loc:
            print(f"Empty location: {loc}, skipping.")
            continue
            
        ind_reward = algebraic_to_index(loc,  grid_height)
        ind_reward = list(np.array(ind_reward)[np.array(ind_reward) < max_index])
        #print(f"ind_reward: {ind_reward}")
        
        for state_index in ind_reward:
            #print(f"state_index: {state_index}")
            if state_index not in out_reward: 
                out_reward[state_index] = [state['label']] 
                out_certainty[state_index] = [1] 
            else: # Append
                out_reward[state_index].append(state['label'])  
                out_certainty[state_index].append(1)
    
    return human_feedback, out_reward, out_certainty

def get_valence_from_HF(human_feedback):  
    """
    Processes human feedback to obtain the certainty level given by the valence score.

    Args:
        human_feedback (str): The feedback provided by the human.

    Returns:
        out_certainty (list): A list with the computed certainty for each sentence.
    """
    doc = nlp(human_feedback)
    out_certainty = []
    
    for sent in doc.sents: # Divide feedback into separate sentences   
        _, certainty_reward =  sentiment_analyzer_vader(sent.text, min_score=0.05)
        
        out_certainty.append(certainty_reward)
    return out_certainty
    