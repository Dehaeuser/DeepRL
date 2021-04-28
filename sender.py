import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp

class SenderEncoder(tf.keras.Model):
    """
    Encoder class for building speaker's Encoder
    The encoder receives a concept as binary vector
    and encodes it into a dense (embedding_dim) representation u
    """
    def __init__(self, units=44, categories_dim=595):
        # TODO: categories_dim anpassen!
        super(SenderEncoder, self).__init__()
        self.categories_dim = categories_dim
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(self.categories_dim))
        self.act = tf.keras.layers.Dense(units, activation='sigmoid')

    def call(self, input_concept):
        """
        input_concept: binary list representing a concept
        """
        x = self.input_layer(input_concept)
        output = self.act(x)

        return output


class Sender_LSTM(tf.keras.Model):
    """
    LSTM network of the sender that creates
    the message of length max_m
    """

    def __init__(self, embed_dim, num_cells, hidden_size, max_len, vocab_size=99, training=True):
        super(Sender_LSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.training = training
        # set max_len-1 to always replace eos with 0
        self.max_len = max_len-1
        self.training = training
        #self.states = None

        self.output_layer = tf.keras.layers.Dense(units=vocab_size, activation='softmax')
        #self.lstm = tf.keras.layers.LSTM(units=hidden_size, activation=None, return_sequences=True, return_state=True)
        self.lstm = tf.keras.layers.RNN([tf.keras.layers.LSTMCell(hidden_size) for _ in range(num_cells)], return_state=True)

    def call(self, input):

        message = []
        entropy = []
        logits = []
        state_h = None

        for i in range(self.max_len):
            states = None
            output = None
            if i == 0:
                output, state_h, state_c = self.lstm(input)
                states = [state_h, state_c]
            else:
                output, state_h, state_c = self.lstm(input, initial_state=states)
                states = [state_h, state_c]
            step_probs = self.output_layer(output)
            dist = tfp.distributions.Categorical(probs=step_probs)

            if self.training:
                single_char = dist.sample()
            else:
                single_char = tf.math.reduce_max(step_probs, 1)
            logits.append(dist.log_prob(single_char))

            message.append(single_char)

            entropy.append(dist.entropy())

        # zip / reshape tensors to have batch_size messages with length max_len
        message = tf.transpose(message, perm=[1, 0])
        # adding zeros to end of message
        zeros = tf.zeros_like(message)
        message = tf.concat([message, zeros], 1)

        return message, logits, entropy, state_h
        # problem: logits has shape seq_length x output_size
        # do we sample only from first output logits[0]
        # logits[0] is output corresponding to target

        # turn lstm output into distribution

        # compute entropy of distribution

        # sample from distribution for every _ in range(self.max_len)
        # append sample to message

        # embed message to max_len, turn eos to 0

        # maybe loop multiple times (max_len) trough network
        # save last hidden state and give as input

        # return message, last_state[0], logits, entropy
