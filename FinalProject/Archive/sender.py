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

class Sender(tf.keras.Model):
    """
    class that calls sender encoder
    """
    def __init__(self, num_options, batch_size):
        super(Sender, self).__init__()
        self.encoder = SenderEncoder()
        self.num_options = num_options
        self.batch_size = batch_size

    def call(self, sender_input):
        encoded_input = []
        for i in range(self.num_options):
            encoded_input.append(self.encoder(sender_input[i]))
        encoded_input = tf.stack(encoded_input)
        return encoded_input

class SenderOnlyTarget(tf.keras.Model):

    def __init__(self, batch_size):
        super(SenderOnlyTarget, self).__init__()
        self.encoder = SenderEncoder()
        self.batch_size = batch_size

    def call(self, sender_input):
        encoded_input = self.encoder(sender_input)
        encoded_input = tf.stack(encoded_input)
        return encoded_input

class Sender_LSTM(tf.keras.Model):
    """
    LSTM network of the sender that creates
    the message of length max_m
    """

    def __init__(self, agent, embed_dim, num_cells, hidden_size, max_len, vocab_size=99, training=True, batch_size=32, see_all_input=True):
        super(Sender_LSTM, self).__init__()
        self.agent = agent
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.training = training
        # set max_len-1 to always replace eos with 0
        self.max_len = max_len-1
        self.training = training
        #self.states = None
        self.batch_size = batch_size
        self.see_all_input = see_all_input

        self.output_layer = tf.keras.layers.Dense(units=vocab_size, activation='softmax')
        #self.lstm = tf.keras.layers.LSTM(units=hidden_size, activation=None, return_sequences=True, return_state=True)
        self.lstm = tf.keras.layers.LSTM(units=num_cells, activation=None, return_sequences=False, return_state=True)

    def call(self, input):

        # preprocess input
        input = self.agent(input)
        if self.see_all_input:
            input = tf.squeeze(input)
        input = tf.transpose(input, [1, 0, 2])

        message = []
        entropy = []
        logits = []
        state_h = None

        #print("input-shape:")
        #print(input.shape)


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

        #print("entropy: ", entropy)
        entropy = tf.transpose(entropy, perm=[1,0])
        zeros_ent = tf.zeros_like(entropy)
        entropy = tf.concat([entropy, zeros_ent], 1)
        entropy = tf.stack(entropy)

        logits = tf.transpose(logits, perm=[1,0])
        zeros_log = tf.zeros_like(logits)
        logits = tf.concat([logits, zeros_log], 1)
        logits = tf.stack(logits)

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
