import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.optimizers import Adam


class Agents:
    """This class contains the Speaker network and the Listener network"""


    def __init__(self, categories_dim = 573, word_embedding_dim, alphabetSize = 100):
        #tells us the dimension for vector u and z
        self.word_embedding_dim = word_embedding_dim

        #the discrete alphabet
        self.alphabetSize = alphabetSize

        #tells us the dimension of the input for the speaker's encoder and the input
        #for the listener's encoder
        self.categories_dim = categories_dim
        self.build_speaker_encoder()
        self.build_listener_encoder()
        self.build_speaker_LSTM()
        self.build_listener_LSTM()



# class Speaker_Encoder(tf.keras.Model):
#     def __init__(self, units = 50):
#         super(Speaker_Encoder,self).__init__()
#         self.layer_list = [
#             tf.kera
#         ]

    def build_speaker_encoder(self, units =50):
        """Paper:
        single-layer MLP with a sigmoid activation function. (p.4)

        “seeing” prelinguistic feed-forward encoders (see Section 3), have dimension 50 (p.13)
        """
        self.speaker_encoder = tf.keras.Sequential()
        self.speaker_encoder.add(tf.keras.layers.InputLayer(input_shape = (self.categories_dim)))
        self.speaker_encoder.add(tf.keras.layers.Dense(units, activation = 'sigmoid'))


    """
    Function gets one concept and returns the dense representation.
    Forward pass through speaker_encoder
    """
    def get_speaker_representation_u(self, concept):
        u = self.speaker_encoder(concept)
        return u

    def build_listener_encoder(self, units = 50):
        """Paper:
        single-layer MLP with a sigmoid activation function. (p.4)

        “seeing” prelinguistic feed-forward encoders (see Section 3), have dimension 50 (p.13)
        """
        self.listener_encoder = tf.keras.Sequential()
        self.listener_encoder.add(tf.keras.layers.InputLayer(input_shape = (self.categories_dim)))
        self.listener_encoder.add(tf.keras.layers.Dense(units, activation = 'sigmoid'))
        pass

    """
    Function gets list with concept candidates and return list of their Dense
    representation
    """
    def get_listener_representationsCandidates(self, conceptList):
        concepts = []
        for i in conceptList:
            concepts.append(self.listener_encoder(i))
        return concepts


    #one-to many (variable-length) LSTM
    def build_speaker_LSTM(self, input, embed_dim):
        """Paper:
        The sequence generation is terminated either by the production
        of a stop symbol or when the maximum length L has been reached. We implement the decoder
        as a single-layer LSTM (p.3)

        For the speaker’s message, this is generated in a greedy fashion by
        selecting the highest-probability symbol at each step. (p.3)


        """
        self.speaker_LSTM = tf.keras.Sequential()
        self.speaker_LSTM.add(tf.keras.layers.Embedding(input_dim = input, output_dim = embed_dim))
        self.speaker_LSTM.add(tf.keras.LSTM(units, return_sequences = True, return_state =True))

        pass

    def get_speaker_message(self, concept):
        self.speaker_LSTM(concept)
        pass


    #many-to-one LSTM
    """
    agent is listener_encoder
    """
    def build_listener_LSTM(self,  units = 50, num_words, embed_dim):
        """Paper:

        """
        self.listener_LSTM = tf.keras.Sequential()
        self.listener_LSTM.add(tf.keras.layers.Embedding(input_dim = num_words, output_dim = embed_dim) )
        self.listener_LSTM.add(tf.keras.LSTM(units)

        pass

    def get_listeners_message_encoding(self, message):
        z = self.listener_LSTM(message)
        pass







    def calculate_dot_product(self, U, z):
        pass

    #update die weights aus allen nets
    def update(self):
        pass


#brauchen jeweils eine Funktion um m,u,U,z zu bekommen also,dass es durch das network geht

#müssen mit env kommunizieren, ob es ein reward gibt oder nicht

#am besten Datentyp anlegen, der u,m,u,U, reward abspeichert pro Durchgang
