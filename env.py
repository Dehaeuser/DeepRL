import tensorflow as tf
import random
import numpy as np
import csv
import re
import os
import create_data
from vocab import Vocabulary

"""
creates environment and game structure

"""
NUM_DISTRACTORS = 9


def getInput(vocabulary):
    """
    returns name of target concept and list of attribute arrays
    of target and all distractors
    can be used when both sender and receiver see all inputs
    """
    input_concepts = random.sample(vocabulary.concept_list, NUM_DISTRACTORS+1)
    target_concept = random.sample(input_concepts, 1)

    sender_receiver_input = []
    for elem in input_concepts:
        sender_receiver_input.append(vocabulary.parseConcept(elem))
    sample = [sender_receiver_input, target_concept, sender_receiver_input]
    return sample

def getInputDifferent(vocabulary):
    """
    returns target concept and distractors divided into inputs for sender and
    receiver
    can be used when sender only receives target as input
    """
    input_concepts = random.sample(vocabulary.concept_list, NUM_DISTRACTORS+1)
    target_concept = random.sample(input_concepts, 1)

    sender_input = vocabulary.parseConcept(target_concept)
    receiver_input = []
    for elem in input_concepts:
        if elem == target_concept:
            receiver_input.append(sender_input)
        else:
            receiver_input.append(vocabulary.parseConcept(elem))
    sample = [sender_input, target_concept, receiver_input]
    return sample

# pick target random from concepts

# pick random distractors

# send target + distractors through sender network
# TODO: how to implement knowledge of sender what concept the target is


# send message + distractors through receiver

# feedback, target correctly identified
