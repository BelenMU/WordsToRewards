import numpy as np

# Sentiment analyzers and word embeddders
import spacy
from spacy.language import Language

@Language.component("set_custom_boundaries")
def set_custom_boundaries(doc):
    for token in doc[:-1]:
        #if token.text == 'but':
        #    doc[token.i + 1].is_sent_start = True
        if token.text == '.':
            doc[token.i + 1].is_sent_start = True
        if token.text == '!':
            doc[token.i + 1].is_sent_start = True
        if token.text == ',':
            doc[token.i + 1].is_sent_start = True
    return doc

nlp = spacy.load("en_core_web_lg")
nlp.add_pipe('set_custom_boundaries', before='parser')

# Pre-defined lists corresponding to specific masking
LIST_SEGMENTS = nlp("beginning middle end all")
LIST_LOCATIONS = nlp("above under left right around in") # up, above and over seem problematic they aren't tagged as simmilar to each other!

# Predefine-mask for segments: Start/Middle/End
NUM_STEPS = 10
len_mask = round(NUM_STEPS/6) # Length of mask for each segment
MASK_SEGMENTS = np.zeros([4, NUM_STEPS])
# Start
MASK_SEGMENTS[0, 0:len_mask] = 1;
MASK_SEGMENTS[0, len_mask:2*len_mask] = 0.5;
# Middle
ind_mid = round(NUM_STEPS/2)
MASK_SEGMENTS[1, ind_mid - len_mask : ind_mid + len_mask] = 0.5;
MASK_SEGMENTS[1, ind_mid - round(len_mask/2) : ind_mid + round(len_mask/2)] = 1;
# End
MASK_SEGMENTS[2, - 2*len_mask:] = 0.5
MASK_SEGMENTS[2, - len_mask:] = 1;
# All
MASK_SEGMENTS[3, :] = 1    

# GPT variables
from science.data_to_GPT import create_system_message_with_uncertainty, get_reward_function_structure_with_uncertainty, get_reward_v2, create_system_message, create_system_message_v2, get_reward_function_structure
GPT_MODEL = 'gpt-4o' #'gpt-4-1106-preview'
TEMPERATURE = 0.5 # Degree of randomness of GPT's output
grid_height = 5
grid_width = 10
SYSTEM_MESSAGE = create_system_message_with_uncertainty(grid_height, grid_width)
SYSTEM_MESSAGE_NOCERTAINTY = create_system_message_v2(grid_height, grid_width)#create_system_message(grid_height, grid_width)
FUNCTION_STRUCTURE = get_reward_function_structure_with_uncertainty(grid_width, grid_height)
FUNCTION_STRUCTURE_NOCERTAINTY = get_reward_v2(grid_width, grid_height)#get_reward_function_structure(grid_width, grid_height)
SECRET_KEY = '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'