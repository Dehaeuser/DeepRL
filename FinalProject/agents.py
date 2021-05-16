import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp

class Encoder(tf.keras.Model):
    """
    Encoder class for building speaker's and listener's encoder
    The encoder receives a concept (as binary list) and returns a
    dense representation.
    """
    def __init__(self, units = 44, categories_dim = 597):
        super(Encoder, self).__init__()
        self.categories_dim = categories_dim
        self.input_layer = tf.keras.layers.InputLayer(input_shape = (self.categories_dim))
        self.act = tf.keras.layers.Dense(units, activation = 'sigmoid')

    def call(self, input_concept):
        """
        :param: input_concept: the binary list representing a concept
        """
        x = self.input_layer(input_concept)
        output = self.act(x)
        return output

class AuxiliaryNetwork(tf.keras.Model):
    """
    Class receives last hidden layer of speaker LSTM.
    Returns predection of the listener's hidden layer after hearing the full message.
    """
    def __init__(self, hidden_size):
        super(AuxiliaryNetwork, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape = hidden_size)
        self.empathy = tf.keras.layers.Dense(hidden_size, activation = 'sigmoid')

    def call(self, hidden):
        x = self.input_layer(hidden)
        encoded = self.empathy(x)
        return encoded

class Receiver(tf.keras.Model):
    """
    Receiver's encoder. Receiver receives (in call function) a list of concepts (as binary arrays) and
    returns a list of their dense representation using the Encoder class as encoder.
    """
    #brauche kein voc, siehe Kommentar weiter unten
    def __init__(self, hidden_size, num_options):
        super(Receiver, self).__init__()
        self.encoder = Encoder(units=hidden_size)
        #brauchen wir num_options überhaupt, wenn das sowieso die länge von receieer_input sein müsste?
        self.num_options = num_options

    def call(self, encoded, receiver_input, num_distractors = 7):
        """
        :param: receiver_input: distractor + target concept representations
        """
        c_list = []
        for i in range(self.num_options):
            c_list.append(self.encoder(receiver_input[i]))
        #print(len(c_list))
        sample, log, entropy = receiver_sampling(encoded, c_list,num_distractors)
        return sample, log, entropy


class Receiver_LSTM(tf.keras.Model):
    """
    LSTM network of the receiver. It consists of embedding layer and LSTM layer.
    It receives (in call function) the message and returns a dense representation
    of it.
    """
    def __init__(self, agent , vocab_size, embed_dim, hidden_size, masking_value =999):
        """
        self.embedding: param::vocab_size:: size of the vocabulary
                        param::embed_dim:: Dimension of the dense embedding
        --> Die message besteht also aus zahlen und die höchste Zahl wäre vocab_size
        """
        super(Receiver_LSTM, self).__init__()
        self.agent = agent
        self.masking_value = masking_value
        self.embed_dim = embed_dim
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.mask = tf.keras.layers.Masking(mask_value = masking_value)
        self.cell = tf.keras.layers.LSTM(units =hidden_size )

    def call(self, message, batch_size, max_len, receiver_input):
        """
        param:message: message that is outputted by the sender
        batch_size: needed for get_padded_sequence
        max_len : maximal length of message; needed for get_padded_sequence

        Reihenfolge (zuerst embedding layer dann padding ist von Ossenkopf übernommen)

        Ossenkopf hat das in RnNReceiverReinforce und RnnEncoder
        """
        emb = self.embedding(message)
        #lengths is a list with the repsective length of the messages (ist so lange wie batch_size)
        lengths = find_lengths(message)
        #padded_emb is padded tensor of emb
        padded_emb = get_padded_sequence(emb, lengths,batch_size, max_len, self.embed_dim)
        masked_emb = self.mask(emb)
        output = self.cell(masked_emb)
        sample, logits, entropy = self.agent(output, receiver_input)
        return sample,logits,entropy,output



def find_lengths(message):
    """
    param: messages: the messages outputted by the sender
    function returns tensor that indicates how long the messages are. Function checkes if EOS symbol (which is 0) is reached.
    """
    message_as_list = message.numpy().tolist()
    lengths = []

    for i in message_as_list:
        counter = 1
        for j in i:
            if not(j == 0):
                counter = counter + 1
            else:
                break
        lengths.append(counter)
    return lengths



def get_padded_sequence(embed, lengths, batch_size, max_len, embed_dim, masking_value = 999):

    """
    param:lengths: list that indicates the size of the message
    param:batch_size: hier 32
    param:max_len: maximal length of message; hier 5
    param: embed_dim: hier 50

    after we received the embedding, we need to mask/padd the values that are not actually part of the message because they were outputted after
    the EOS- Symbol
    """
#     emb_list = emb_tensor.numpy().tolist()
    emb_list = embed.numpy().tolist()
    for i in range(batch_size):
        laenge = lengths[i]
        for j in range(max_len):
            for k in  range(embed_dim):
                if j > laenge:
                    emb_list[i][j][k]= masking_value
    return tf.convert_to_tensor(emb_list)


def receiver_sampling(encoding, candidate_list, num_candidates, training = True):
    """
    :param: encoding: listener's LSTM's encoding of speaker's message.
    :param: candidate_list: listener's encoding of distractors and target
    :return: sample: guess about which of the samples is the target
    :return: log_prob: log probability that the chosen sample is indeed the target
    :return: entropy: returns entropy of the distribution
    function calculates the dotproduct between the encoding and the candidate_list. Then it
    forwards it through a softmax layer and samples from it. (that the Gibbs Distribution)
    """

    training = training
    concepts = tf.stack(candidate_list)
    #transpose
    concepts = tf.transpose(concepts, [1,0,2])
    #print(concepts)
    channel_input = encoding[:,:,None]
    dotproduct = tf.matmul(concepts, channel_input)
    dotproduct = tf.squeeze(dotproduct,-1)
    pre_logits = tf.nn.log_softmax(dotproduct,1)
    distr = tfp.distributions.Categorical(logits = pre_logits)

    if training:
        sample = distr.sample()
    else:
        sample = tf.math.reduce_max(pre_logits,1)

    log_prob = distr.log_prob(sample)
    entropy = distr.entropy()

    return sample, log_prob, entropy
