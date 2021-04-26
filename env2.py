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

class ConceptData():

    def __init__(self, voc, num_distractors):
        self.voc = voc
        self.num_distractors = num_distractors

    def getInput(self):
        """
        returns name of target concept and list of attribute arrays
        of target and all distractors
        can be used when both sender and receiver see all inputs
        """
        input_concepts = random.sample(self.voc.concept_list_name, self.num_distractors+1)
        print("input_concepts:")
        print(input_concepts)
        target_concept = random.sample(input_concepts, 1)[0]
        print("target_concept:")
        print(target_concept)

        sender_input = []
        receiver_input = []

        sender_input.append(self.voc.concept2vector[target_concept])
        for elem in input_concepts:
            if elem != target_concept:
                sender_input.append(self.voc.concept2vector[elem])

        for elem in input_concepts:
            receiver_input.append(self.voc.concept2vector[elem])
        sample = [sender_input, target_concept, receiver_input]
        return sample

    def getInputDifferent(self):
        """
        returns target concept and distractors divided into inputs for sender and
        receiver
        can be used when sender only receives target as input
        """
        input_concepts = random.sample(self.voc.concept_list_name, self.num_distractors+1)
        target_concept = random.sample(input_concepts, 1)

        sender_input = self.voc.concept2vector[target_concept]
        receiver_input = []
        for elem in input_concepts:
            if elem == target_concept:
                receiver_input.append(sender_input)
            else:
                receiver_input.append(self.voc.concept2vector[elem])
        sample = [sender_input, target_concept, receiver_input]
        return sample

# pick target random from concepts

# pick random distractors

# send target + distractors through sender network
# TODO: how to implement knowledge of sender what concept the target is


# send message + distractors through receiver

# feedback, target correctly identified
