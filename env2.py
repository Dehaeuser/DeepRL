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

    def __init__(self, voc, num_distractors, batch_size):
        self.voc = voc
        self.num_distractors = num_distractors
        self.batch_size = batch_size

    def getInput(self):
        """
        returns name of target concept and list of attribute arrays
        of target and all distractors
        can be used when both sender and receiver see all inputs
        """
        sender_input = []
        targets = []
        receiver_input = []

        for _ in range(self.batch_size):

            input_concepts = random.sample(self.voc.concept_list_name, self.num_distractors+1)
            #print("input_concepts:")
            #print(input_concepts)
            target_concept = random.sample(input_concepts, 1)[0]
            targets.append(target_concept)
            #print("target_concept:")
            #print(target_concept)

            sender_input_batch = []
            receiver_input_batch = []

            sender_input_batch.append(self.voc.concept2vector[target_concept])
            for elem in input_concepts:
                if elem != target_concept:
                    sender_input_batch.append(self.voc.concept2vector[elem])

            sender_input.append(sender_input_batch)

            for elem in input_concepts:
                receiver_input_batch.append(self.voc.concept2vector[elem])

            receiver_input.append(receiver_input_batch)

        sample = [sender_input, targets, receiver_input]

        return sample

    def getInputDifferent(self):
        """
        returns target concept and distractors divided into inputs for sender and
        receiver
        can be used when sender only receives target as input
        """
        sender_input = []
        targets = []
        receiver_input = []

        for _ in range(self.batch_size):

            input_concepts = random.sample(self.voc.concept_list_name, self.num_distractors+1)
            target_concept = random.sample(input_concepts, 1)
            targets.append(target_concept)

            sender_input_batch = self.voc.concept2vector[target_concept]
            sender_input.append(sender_input_batch)

            receiver_input_batch = []
            for elem in input_concepts:
                if elem == target_concept:
                    receiver_input_batch.append(sender_input)
                else:
                    receiver_input_batch.append(self.voc.concept2vector[elem])

            receiver_input.append(receiver_input_batch)

        sample = [sender_input, target_concept, receiver_input]

        return sample

# pick target random from concepts

# pick random distractors

# send target + distractors through sender network
# TODO: how to implement knowledge of sender what concept the target is


# send message + distractors through receiver

# feedback, target correctly identified
