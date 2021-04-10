import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.optimizers import Adam




class Encoder(tf.keras.Model):
    """
    Encoder class for building speaker's and listener's encoder
    The encoder receives a concept (as binary list) and returns a
    dense representation.
    """
    def __init__(self, units = 50):
        super(Encoder, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape = (self.categories_dim))
        self.act = tf.keras.layers.Dense(units, activation = 'sigmoid')

    def call(self, input_concept):
        """
        :param: input_concept: the binary list representing a concept
        """
        x = self.input_layer(input_concept)
        output = self.act(x)
        return output



class Receiver(tf.keras.Model):
    """
    Receiver's encoder. Receiver receives (in call function) a list of concepts (as binary arrays) and
    returns a list of their dense representation using the Encoder class as encoder.
    """
    def __init__(self, num_options, voc):
        super(Receiver, self).__init__()
        self.encoder = Encoder(voc)
        #brauchen wir num_options überhaupt, wenn das sowieso die länge von receieer_input sein müsste?
        self.num_option = num_options

    def call(self, receiver_input):
        """
        :param: receiver_input: distractor + target concept representations
        """
        c_list = []
        for i in range(self.num_options):
            c_list.append(self.encoder(receiver_input[i]))
        return c_list


class Receiver_LSTM(tf.keras.Model):
    """
    LSTM network of the receiver. It consists of embedding layer and LSTM layer.
    It receives (in call function) the message and returns a dense representation
    of it.
    """
    def __init__(self, vocab_size, embed_dim, hidden_size):
        """
        self.embedding: param::vocab_size:: size of the vocabulary
                        param::embed_dim:: Dimension of the dense embedding
        --> Die message besteht also aus zahlen und die höchste Zahl wäre vocab_size
        """
        super(Receiver_LSTM, self).__init__()
        #PyTorch:
        # self.cell = LSTM(input_size=embed_dim, batch_first=True,
        #                        hidden_size=n_hidden, num_layers=num_layers)
        #
        # self.embedding = nn.Embedding(vocab_size, embed_dim)

        #von Ossenkopf
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.cell = tf.keras.LSTM(units =hidden_size )

    def call(self, message, message_length):
        #part in pytorch der noch gemacht wird. Weiß nicht ob wir auch sowas brauchen
            # packed = nn.utils.rnn.pack_padded_sequence(
            #     emb, lengths, batch_first=True, enforce_sorted=False)
        emb = self.embedding(message)
        output = self.cell(emb)
        pass

class Sender_LSTM(tf.keras.Model):
    def __init__(self, vocab:size, embed_dim, hidden_size, max_len): #force_eos =True (pytorch; weiß nicht, ob wir das brauchen)
        super(Sender_LSTM, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.cell = tf.keras.LSTM(units = hidden_size)

    def call(encoding):
        """
        :param:encoding: Sender's Encoder encoding of target concept
        """
        pass

def receiver_sampling(encoding, candidate_list, num_candidates):
    """
    :param: encoding: listener's LSTM's encoding of speaker's message.
    :param: candidate_list: listener's encoding of distractors and target
    :return: guess of receiver about target
    function calculates the dotproduct between the encoding and the candidate_list. Then it
    forwards it through a softmax layer and samples from it. (that the Gibbs Distribution)

    --> Weiß noch nicht, ob das alles so hinhaut mit den Dimensionen etc.
    """

    #habe hier eine Mischung von Ossenkopf und Lazarido 2016 probiert

    for i in range(num_candidates):
        #dotproduct (von Lazaridou 2016 implementation in tensorflow)
        #ggf muss man noch flatten
        dotproducts.append(tf.matmul(encoding,i))
    # stacks list of tensors into tensor
    dotproducts = tf.stack(dotproducts)

    ###HIER EHER AN LAZARIDOU 2016 ANNGELEHNT
    #stimmt die dimension? Ossenkopf benutzt log_softmax, Lazarido2016 softmax
    probs = tf.nn.softmax(dotproduct, dim=1)
    distr = tfp.distributions.Categorical(logits=pre_logits)

    if self.training:
        sample = distr.sample()
    else:
        sample = pre_logits.argmax(dim=1)
    log_prob = distr.log_prob(sample)

    entropy = distr.entropy()
    #für was braucht man log_prob und entropy? Vielleicht für den Loss?
    return sample, log_prob, entropy








#
# class Agents:
#     """This class contains the Speaker network and the Listener network"""
#
#
#     def __init__(self, categories_dim = 573, word_embedding_dim, alphabetSize = 100):
#         #tells us the dimension for vector u and z
#         self.word_embedding_dim = word_embedding_dim
#
#         #the discrete alphabet
#         self.alphabetSize = alphabetSize
#
#         #tells us the dimension of the input for the speaker's encoder and the input
#         #for the listener's encoder
#         self.categories_dim = categories_dim
#         self.build_speaker_encoder()
#         self.build_listener_encoder()
#         self.build_speaker_LSTM()
#         self.build_listener_LSTM()
#
#
#
# # class Speaker_Encoder(tf.keras.Model):
# #     def __init__(self, units = 50):
# #         super(Speaker_Encoder,self).__init__()
# #         self.layer_list = [
# #             tf.kera
# #         ]
#
#
#     def build_speaker_encoder(self, units =50):
#         """Paper:
#         single-layer MLP with a sigmoid activation function. (p.4)
#
#         “seeing” prelinguistic feed-forward encoders (see Section 3), have dimension 50 (p.13)
#         """
#         self.speaker_encoder = tf.keras.Sequential()
#         self.speaker_encoder.add(tf.keras.layers.InputLayer(input_shape = (self.categories_dim)))
#         self.speaker_encoder.add(tf.keras.layers.Dense(units, activation = 'sigmoid'))
#
#
#     """
#     Function gets one concept and returns the dense representation.
#     Forward pass through speaker_encoder
#     """
#     def get_speaker_representation_u(self, concept):
#         u = self.speaker_encoder(concept)
#         return u
#
#     def build_listener_encoder(self, units = 50):
#         """Paper:
#         single-layer MLP with a sigmoid activation function. (p.4)
#
#         “seeing” prelinguistic feed-forward encoders (see Section 3), have dimension 50 (p.13)
#         """
#         self.listener_encoder = tf.keras.Sequential()
#         self.listener_encoder.add(tf.keras.layers.InputLayer(input_shape = (self.categories_dim)))
#         self.listener_encoder.add(tf.keras.layers.Dense(units, activation = 'sigmoid'))
#         pass
#
#     """
#     Function gets list with concept candidates and return list of their Dense
#     representation
#     """
#     def get_listener_representationsCandidates(self, conceptList):
#         concepts = []
#         for i in conceptList:
#             concepts.append(self.listener_encoder(i))
#         return concepts
#
#
#     """
#     Function takes as arguments u (which gets outputted by speaker_encoder and which is the dense representation of the target concept)
#
#     """
#     def build_speaker_LSTM(self, dim= 50, embed_dim, units = 50):
#         """Paper:
#         The sequence generation is terminated either by the production
#         of a stop symbol or when the maximum length L has been reached. We implement the decoder
#         as a single-layer LSTM (p.3)
#
#         For the speaker’s message, this is generated in a greedy fashion by
#         selecting the highest-probability symbol at each step. (p.3)
#
#
#         """
#         self.speaker_LSTM = tf.keras.Sequential()
#         #hier brauchen wir glaube ich das Embedding gar nicht da speaker_encoder uns das schon gibt?
#         self.speaker_LSTM.add(tf.keras.layers.Embedding(input_dim = dim, output_dim = embed_dim))
#         self.speaker_LSTM.add(tf.keras.LSTM(units = units, return_sequences = True, return_state =True))
#
#         pass
#
#     def get_speaker_message(self, concept):
#         self.speaker_LSTM(concept)
#         pass
#
#
#     #many-to-one LSTM
#     """
#
#     """
#     def build_listener_LSTM(self,  units = 50, num_words, embed_dim):
#         """Paper:
#
#         """
#         self.listener_LSTM = tf.keras.Sequential()
#         self.listener_LSTM.add(tf.keras.layers.Embedding(input_dim = num_words, output_dim = embed_dim) )
#         self.listener_LSTM.add(tf.keras.LSTM(units)
#
#         pass
#
#     def get_listeners_message_encoding(self, message):
#         z = self.listener_LSTM(message)
#         pass
#
#
#
#
#
#
#
#     def calculate_dot_product(self, U, z):
#         pass
#
#     #update die weights aus allen nets
#     def update(self):
#         pass
#
#
# #brauchen jeweils eine Funktion um m,u,U,z zu bekommen also,dass es durch das network geht
#
# #müssen mit env kommunizieren, ob es ein reward gibt oder nicht
#
# #am besten Datentyp anlegen, der u,m,u,U, reward abspeichert pro Durchgang
