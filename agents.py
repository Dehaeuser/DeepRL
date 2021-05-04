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

class AuxillaryNetwork(tf.keras.Model):
    """
    Class receives last hidden layer of speaker LSTM.
    Returns predection of the listener's hidden layer after hearing the full message.
    """
    def __init__(self, hidden_size):
        super(AuxillaryNetwork, self).__init__()
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
    def __init__(self, num_options):
        super(Receiver, self).__init__()
        self.encoder = Encoder()
        #brauchen wir num_options überhaupt, wenn das sowieso die länge von receieer_input sein müsste?
        self.num_options = num_options

    def call(self, encoded, receiver_input, num_distractors = 7):
        """
        :param: receiver_input: distractor + target concept representations
        """
        c_list = []
        for i in range(self.num_options):
            c_list.append(self.encoder(receiver_input[i]))
        print(len(c_list))
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
    #concepts = tf.transpose(concepts, [1,0,2])
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





# #Sender von Ossenkopf
# class RnnSenderReinforce(nn.Module):
#     """
#     Reinforce Wrapper for Sender in variable-length message game. Assumes that during the forward,
#     the wrapped agent returns the initial hidden state for a RNN cell. This cell is the unrolled by the wrapper.
#     During training, the wrapper samples from the cell, getting the output message. Evaluation-time, the sampling
#     is replaced by argmax.
#     >>> agent = nn.Linear(10, 3)
#     >>> agent = RnnSenderReinforce(agent, vocab_size=5, embed_dim=5, hidden_size=3, max_len=10, cell='lstm', force_eos=False)
#     >>> input = torch.FloatTensor(16, 10).uniform_(-0.1, 0.1)
#     >>> message, logprob, entropy = agent(input)
#     >>> message.size()
#     torch.Size([16, 10])
#     >>> (entropy > 0).all().item()
#     1
#     >>> message.size()  # batch size x max_len
#     torch.Size([16, 10])
#     """
#     def __init__(self, agent, vocab_size, embed_dim, hidden_size, max_len, num_layers=1, cell='rnn', force_eos=True):
#         """
#         :param agent: the agent to be wrapped
#         :param vocab_size: the communication vocabulary size
#         :param embed_dim: the size of the embedding used to embed the output symbols
#         :param hidden_size: the RNN cell's hidden state size
#         :param max_len: maximal length of the output messages
#         :param cell: type of the cell used (rnn, gru, lstm)
#         :param force_eos: if set to True, each message is extended by an EOS symbol. To ensure that no message goes
#         beyond `max_len`, Sender only generates `max_len - 1` symbols from an RNN cell and appends EOS.
#         """
#         super(RnnSenderReinforce, self).__init__()
#         self.agent = agent

#         self.force_eos = force_eos

#         self.max_len = max_len
#         if force_eos:
#             self.max_len -= 1

#         self.hidden_to_output = nn.Linear(hidden_size, vocab_size)
#         self.embedding = nn.Embedding(vocab_size, embed_dim)
#         #hat irgendwie einen Gradient; brauchen wir glaube ich weil wir uns selbst das
#         #LSTM zusammenstellen
#         self.sos_embedding = nn.Parameter(torch.zeros(embed_dim))
#         self.embed_dim = embed_dim
#         self.vocab_size = vocab_size
#         self.num_layers = num_layers
#         self.cells = None

#         cell = cell.lower()
#         cell_types = {'rnn': nn.RNNCell, 'gru': nn.GRUCell, 'lstm': nn.LSTMCell}

#         if cell not in cell_types:
#             raise ValueError(f"Unknown RNN Cell: {cell}")

#         cell_type = cell_types[cell]

#         #für uns dann:
#         #self.cell(s) = LSTM(input_size = 50, hidden_size = 50)
#         self.cells = nn.ModuleList([
#             cell_type(input_size=embed_dim, hidden_size=hidden_size) if i == 0 else \
#             cell_type(input_size=hidden_size, hidden_size=hidden_size) for i in range(self.num_layers)])

#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.normal_(self.sos_embedding, 0.0, 0.01)

#     #x ist sender_Input
#     def forward(self, x):
#         #Input geht erstmal durch encoder; gibt dann dense representation of target concept zurück
#         prev_hidden = [self.agent(x)]

#         #wenn wir nur eine layer haben, passiert hier nichts, oder? (1-1 == 0)
#         prev_hidden.extend([torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers - 1)])

#         #das sind nullen
#         prev_c = [torch.zeros_like(prev_hidden[0]) for _ in range(self.num_layers)]  # only used for LSTM

#         #was genau passiert hier?
#         input = torch.stack([self.sos_embedding] * x.size(0))

#         sequence = []
#         logits = []
#         entropy = []

#         for step in range(self.max_len):
#             #self.cells hat bei uns dann länge von 1
#                     #layer IST unser LSTM (das hier ist also forward (in tensorflow call))
#                     #gibt uns hidden state und cell state
#             h_t, c_t = layer(input, (prev_hidden[i], prev_c[i]))
#             prev_c[i] = c_t
#             prev_hidden[i] = h_t
#             input = h_t

#             #dann sampled man das wort aus dem vocabulary
#             step_logits = F.log_softmax(self.hidden_to_output(h_t), dim=1)
#             distr = Categorical(logits=step_logits)
#             entropy.append(distr.entropy())

#             if self.training:
#                 x = distr.sample()
#             else:
#                 x = step_logits.argmax(dim=1)
#             logits.append(distr.log_prob(x))

#             input = self.embedding(x)
#             sequence.append(x)

#         sequence = torch.stack(sequence).permute(1, 0)
#         logits = torch.stack(logits).permute(1, 0)
#         entropy = torch.stack(entropy).permute(1, 0)

#         #verstehe ich nicht
#         if self.force_eos:
#             zeros = torch.zeros((sequence.size(0), 1)).to(sequence.device)

#             sequence = torch.cat([sequence, zeros.long()], dim=1)
#             logits = torch.cat([logits, zeros], dim=1)
#             entropy = torch.cat([entropy, zeros], dim=1)

#         return sequence, logits, entropy, prev_hidden[0]
