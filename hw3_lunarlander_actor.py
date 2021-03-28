import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import numpy as np
import gym
import ray
from really import SampleManager  # important !!
from really.utils import (
    dict_to_dict_of_datasets, dict_to_dataset
)


# Model
class PolicyGrad_Model(tf.keras.Model):

    def __init__(self, output_units=2):
        super(PolicyGrad_Model, self).__init__()
        self.hidden1 = tf.keras.layers.Dense(units=8, activation="relu", dtype="float32")
        self.hidden2 = tf.keras.layers.Dense(units=8, activation="relu", dtype="float32")
        self.layer_mus = tf.keras.layers.Dense(units=output_units, activation="tanh")
        self.layer_sigmas = tf.keras.layers.Dense(units=output_units, activation="sigmoid")
        self.gamma = 0.9


    def call(self, x_in):

        output = {}

        x = self.hidden1(x_in)
        x = self.hidden2(x)
        mus = self.layer_mus(x)
        #sigmas = tf.exp(self.layer_sigmas(x))
        sigmas = self.layer_sigmas(x)

        #output["mu"] = tf.squeeze(mus)
        output["mu"] = mus
        #print("mu: ", output["mu"].numpy())
        #print("mu: ", outputs["mu"])
        #output["sigma"] = tf.squeeze(sigmas)
        output["sigma"] = sigmas
        #print("sigmas: ", output["sigma"].numpy())
        #print("sigma: ", outputs["sigma"])

        return output

    def returns_to_go(self, data):

        #g_values = np.zeros(len(data["state"]))
        g_values = []

        for i in range(len(data["state"])):
            g_value = 0
            not_done = True
            j = i

            while not_done:
                if data["not_done"][j]:
                    g_value = g_value + (self.gamma ** (j - i)) * data["reward"][j]
                    j += 1
                else:
                    g_value = g_value + (self.gamma ** (j - i)) * data["reward"][j]
                    #g_values[i] = g_value
                    g_values.append(g_value)
                    not_done = False

        return g_values

def gaussian_loss(state, action, reward):
    output = agent.model(state)
    mus = output["mu"]
    sigmas = output["sigma"]
    mus = tf.cast(mus, dtype = "float32")
    sigmas = tf.cast(sigmas, dtype = "float32")
    action = tf.cast(action, dtype = "float32")
    reward = tf.cast(reward, dtype ="float32")
    reward = tf.expand_dims(reward, axis=1)

    # Compute Gaussian PDF
    gaussian_log = tf.exp(-0.5 * ((action - mus) / sigmas)**2) * 1 / (sigmas * tf.sqrt(2 * np.pi))

    # Compute log prob
    log_prob = tf.math.log(gaussian_log + 1e-5)

    # compute loss
    loss = reward * log_prob

    return loss


if __name__ == "__main__":

    # define kwargs for model
    kwargs = {
    "model": PolicyGrad_Model,
    "environment": "LunarLanderContinuous-v2",
    "num_parallel": 2,
    "total_steps": 100,
    "action_sampling_type": "continuous_normal_diagonal",
    #"num_episodes": 20
    }

    # initialize ray, manager, saving path
    ray.init(log_to_driver=False)
    manager = SampleManager(**kwargs)
    saving_path = os.getcwd() + "\\progress_test_HW3"
    if not os.path.exists(saving_path):
        os.mkdir(saving_path)

    # initialize parameters and aggregator
    buffer_size = 5000
    test_steps = 100
    epochs = 30
    sample_size = 1000
    optim_batch_size = 1 # when batching to 8 another error appears
    saving_after = 5
    training = False

    optim_keys = ['state', 'action', 'reward', 'state_new', 'not_done']

    #manager.initilize_buffer(buffer_size, optim_keys)

    manager.initialize_aggregator(path=saving_path, saving_after=10, aggregator_keys=['loss', 'time_steps'])

    # test before training
    print('Test before training:')
    manager.test(test_steps, render=False, do_print=True, evaluation_measure="time_and_reward")

    # get initial agent
    agent = manager.get_agent()

    # initialize the optimizer
    adam = tf.keras.optimizers.Adam(learning_rate=0.0001)

    if training:
        # main trainig loop
        for e in range(epochs):
            print("training epoch: ", e)
            losses_list = []
            data = manager.get_data()

            # sample from agent
            sample_dict = manager.sample(sample_size, from_buffer=False)
            g_values = agent.model.returns_to_go(sample_dict)
            sample_dict["g_values"] = g_values
            print("g-values: ", len(g_values))
            #print("g-values: ", g_values)
            data_dict = dict_to_dict_of_datasets(sample_dict, batch_size=optim_batch_size)

            dataset = zip(data_dict["state"], data_dict["action"], data_dict["g_values"])

            for state, action, g_value in dataset:
                with tf.GradientTape() as tape:

                    loss = gaussian_loss(state, action, g_value)
                    losses_list.append(loss)
                    gradients = tape.gradient(loss, agent.model.trainable_variables)

                adam.apply_gradients(zip(gradients, agent.model.trainable_variables))

                new_weights = agent.get_weights()
                manager.set_agent(new_weights)

            # get new agent
            agent = manager.get_agent()

            # update aggregator
            time_steps = manager.test(test_steps,  render=True if e%10 == 0 else False)
            print("time_steps: ", len(time_steps))
            print("losses_list: ", len(losses_list))
            manager.update_aggregator(loss=losses_list,
                                      time_steps=time_steps)
            print(
                f"epoch ::: {e}  loss ::: {np.mean([np.mean(l) for l in losses_list])}   avg env steps ::: {np.mean(time_steps)}"
            )
            #manager.agg.save_graphic()
            print("---")

    print("done")
    print("testing optimized agent")
    manager.test(test_steps, test_episodes=20, render=True, do_print=True, evaluation_measure="time_and_reward")
