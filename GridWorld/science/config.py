import numpy as np

# Sentiment analyzers and word embeddders
import spacy
from spacy.language import Language

@Language.component("set_custom_boundaries")
def set_custom_boundaries(doc):
    for token in doc[:-1]:
        if token.text == '.':
            doc[token.i + 1].is_sent_start = True
        if token.text == '!':
            doc[token.i + 1].is_sent_start = True
        if token.text == ',':
            doc[token.i + 1].is_sent_start = True
    return doc

nlp = spacy.load("en_core_web_lg")
nlp.add_pipe('set_custom_boundaries', before='parser')

# GPT variables
from science.data_to_GPT import create_system_message_with_uncertainty, get_reward_function_structure_with_uncertainty, get_reward_v2, create_system_message, create_system_message_v2, get_reward_function_structure
GPT_MODEL = 'gpt-4o' #'gpt-4-1106-preview'
TEMPERATURE = 0.5 # Degree of randomness of GPT's output
grid_height = 5
grid_width = 10
SYSTEM_MESSAGE = create_system_message_with_uncertainty(grid_height, grid_width)
SYSTEM_MESSAGE_NOCERTAINTY = create_system_message_v2(grid_height, grid_width)
FUNCTION_STRUCTURE_NOCERTAINTY = get_reward_v2(grid_width, grid_height)
#SECRET_KEY = 