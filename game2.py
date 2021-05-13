import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from agents import find_lengths

class Game(tf.keras.Model):

    def __init__(self, sender_encoder, sender, receiver, main_loss, sender_entr_coeff, receiver_entr_coeff, batch_size, max_len, sender_all_input):
        super(Game, self).__init__()
        self.sender_encoder = sender_encoder
        self.sender = sender
        self.receiver = receiver
        self.main_loss = main_loss
        self.sender_entropy_coeff = sender_entr_coeff
        self.receiver_entropy_coeff = receiver_entr_coeff
        self.batch_size = batch_size
        self.max_len = max_len
        self.sender_all_input = sender_all_input
# implement game with sender & receiver, apply gradients to both simultaneously
# then make move through Aux_RNN
# apply gradients to Aux_RNN seperately from and after Game

    def call(self, input_concepts, sender_input, targets, receiver_input):

        if self.sender_all_input:
            sender_input = tf.squeeze(tf.convert_to_tensor(sender_input))
        sender_input = tf.transpose(sender_input, [1,0,2])
        sender_input = sender_input.numpy()

        #sender_input = self.sender_encoder(sender_input)

        message, log_prob_s, entropy_s , prev_hidden = self.sender(sender_input)

        #einer- Dimension rausbekommen
        receiver_input = tf.squeeze(tf.convert_to_tensor(receiver_input))
        #richtige Reihenfolge der Dimensionen
        receiver_input2 = tf.transpose(receiver_input, [1,0,2])
        receiver_input2 = receiver_input2.numpy()

        sample, log_prob_r, entropy_r, last_hidden = self.receiver(message=message, batch_size=self.batch_size, max_len=self.max_len, receiver_input=receiver_input2)

        output_receiver = []
        for j in range(self.batch_size):
            output_receiver.append(input_concepts[j][sample[j]])

        loss = self.main_loss(_sender_input=sender_input, _message=message, _receiver_input=receiver_input, input_concepts=input_concepts, receiver_output=sample, targets=targets)
        acc = np.mean(-loss)
        # compute effective entropy and log_prob of output before and including eos
        message_lengths = find_lengths(message)

        effective_entropy_s = tf.zeros_like(entropy_r)
        effective_entropy_s = np.array(effective_entropy_s)
        effective_log_prob_s = tf.zeros_like(log_prob_r)
        effective_log_prob_s = np.array(effective_log_prob_s)

        #print(message.shape)

        for k in range(message.shape[0]):
            for l in range(message.shape[1]):
                not_eosed = float(k < message_lengths[k])
                effective_entropy_s[k] += entropy_s[k, l] * not_eosed
                effective_log_prob_s[k] += log_prob_s[k, l] * not_eosed

        effective_entropy_s = tf.convert_to_tensor(effective_entropy_s)
        effective_log_prob_s = tf.convert_to_tensor(effective_log_prob_s)

        weighted_entropy = entropy_r * self.receiver_entropy_coeff + self.sender_entropy_coeff * effective_entropy_s
        log_prob = log_prob_r + effective_log_prob_s
        loss = tf.math.reduce_mean(loss * log_prob) - weighted_entropy

        return loss, prev_hidden, last_hidden, acc, message
