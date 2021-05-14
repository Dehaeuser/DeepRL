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
from tensorflow.keras.layers import Layer


# implement model for choosing actions
class Actor_Model(tf.keras.layers.Layer):

    def __init__(self, output_units=2):
        super(Actor_Model, self).__init__()
        self.hidden1 = tf.keras.layers.Dense(units=8, activation="relu", dtype="float32")
        self.hidden2 = tf.keras.layers.Dense(units=8, activation="relu", dtype="float32")
        self.layer_mus = tf.keras.layers.Dense(units=output_units, activation="tanh")
        self.layer_sigmas = tf.keras.layers.Dense(units=output_units, activation="sigmoid")
        #self.gamma = 0.9


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


# implement network for value estimation
class Critic_Model(tf.keras.layers.Layer):

    def __init__(self, output_units=1):

        super(Critic_Model, self).__init__()
        self.hidden1 = tf.keras.layers.Dense(units=8, activation='relu')
        self.hidden2 = tf.keras.layers.Dense(units=8, activation='relu')
        self.output_layer = tf.keras.layers.Dense(units=output_units, activation=None)

    def call(self, x_in):

        output = {}
        x = self.hidden1(x_in)
        x = self.hidden2(x_in)
        out = self.output_layer(x)

        output["value_estimate"] = out

        return output


class ActorCriticAgent(tf.keras.Model):

    def __init__(self, gamma=0.9, lam=0.9):

        super(ActorCriticAgent, self).__init__()
        self.gamma = gamma
        self.lam = lam
        self.actor = Actor_Model()
        self.critic = Critic_Model()

    def call(self, state, **kwargs):
        output = {}

        actor_output = self.actor(state)
        output["mu"] = actor_output["mu"]
        output["sigma"] = actor_output["sigma"]

        output["value_estimate"] = self.critic(state)["value_estimate"]

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

    # defining a "loss" function for a continuous model
    def gaussian_loss(self, state, action, reward):

        output = self.actor(state)
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

    def estimate_advantage(self, data):

        advantages = []

        for i in range(len(data["state"])):
            state_value = data["value_estimate"][i]
            state_value_new = data["value_estimate"][(i+1) % len(data["value_estimate"])]
            adv = - state_value + data["reward"][i] + (data["not_done"][i] * self.gamma * state_value_new)
            advantages.append(adv)

        return advantages



if __name__ == "__main__":

    # define kwargs for model
    kwargs = {
    "model": ActorCriticAgent,
    "environment": "LunarLanderContinuous-v2",
    "num_parallel": 2,
    "total_steps": 100,
    "action_sampling_type": "continuous_normal_diagonal",
    "returns": ['monte_carlo', 'value_estimate']
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
    optim_batch_size = 1
    saving_after = 5
    training = True

    optim_keys = ['state', 'action', 'reward', 'state_new', 'not_done', 'value_estimate']

    # initialize aggregator
    aggregator_keys = ['actor_loss', 'time_steps', 'reward']
    manager.initialize_aggregator(path=saving_path, saving_after=10, aggregator_keys=aggregator_keys)

    # test before training
    print('Test before training:')
    manager.test(test_steps, render=False, do_print=True, evaluation_measure="time_and_reward")

    # get initial agent
    agent = manager.get_agent()

    # initialize the optimizer
    adam = tf.keras.optimizers.Adam(learning_rate=0.0001)

    # initialize mse loss for training critic
    mse = tf.keras.losses.MeanSquaredError()

    if training:
    # main trainig loop
        for e in range(epochs):
            print("training epoch: ", e)
            actor_losses = []
            critic_losses = []
            absolute_losses = 0
            data = manager.get_data()

            # sample from agent
            sample_dict = manager.sample(sample_size, from_buffer=False)
            g_values = agent.model.returns_to_go(sample_dict)
            sample_dict["g_values"] = g_values
            advantages = agent.model.estimate_advantage(sample_dict)
            #print("advantages: ", len(advantages))
            #print("g_values: ", len(g_values))
            #print("states: ", len(sample_dict['state']))
            sample_dict["advantage"] = advantages
            print(f"collected data for: {sample_dict.keys()}")
            #print("g-values: ", len(g_values))
            #print("g-values: ", g_values)
            data_dict = dict_to_dict_of_datasets(sample_dict, batch_size=optim_batch_size)

            dataset = tf.data.Dataset.zip((data_dict["state"], data_dict["action"], data_dict["g_values"], data_dict["advantage"], data_dict["value_estimate"], data_dict["monte_carlo"]))

            for state, action, g_value, advantage, value_estimate, monte_carlo in dataset:

                # train actor
                with tf.GradientTape() as tape:

                    actor_loss = - agent.model.gaussian_loss(state, action, advantage)
                    #actor_losses.append(actor_loss)
                    actor_gradients = tape.gradient(actor_loss, agent.model.actor.trainable_variables)

                adam.apply_gradients(zip(actor_gradients, agent.model.actor.trainable_variables))

                # train critic
                with tf.GradientTape() as tape:

                    critic_loss = mse(g_value, agent.v(state))
                    #print("critic loss: ", critic_loss.shape)
                    #critic_losses.append(critic_loss)
                    critic_gradients = tape.gradient(critic_loss, agent.model.critic.trainable_variables)

                adam.apply_gradients(zip(critic_gradients, agent.model.critic.trainable_variables))

                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
                #absolute_losses += actor_loss + critic_loss

                # update agent
                new_weights = agent.get_weights()
                manager.set_agent(new_weights)

            # get new agent
            agent = manager.get_agent()

            # update aggregator
            time_steps, reward = manager.test(test_steps, render=True if e%10 == 0 else False, evaluation_measure="time_and_reward")
            #print("time_steps: ", len(time_steps))
            #print("reward: ", len(reward))
            #print("actor_loss:", len(actor_losses))
            #print("critic_loss: ", len(critic_losses))
            manager.update_aggregator(actor_loss=actor_losses,
                                      time_steps=time_steps,
                                      reward=reward)
            print(
                f"epoch ::: {e}  actor_loss ::: {np.mean([np.mean(l) for l in actor_losses])}  avg env steps ::: {np.mean(time_steps)}   avg reward ::: {np.mean(reward)}"
            )
            if e % saving_after == 0:
                manager.save_model(saving_path, e)
            #manager.agg.save_graphic()
            print("---")

    manager.load_model(saving_path)
    print("done")
    print("testing optimized agent")
    manager.test(test_steps, test_episodes=10, render=True, do_print=True, evaluation_measure="time_and_reward")
