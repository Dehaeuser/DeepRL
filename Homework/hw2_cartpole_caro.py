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

class DQN_Model(tf.keras.Model):
    def __init__(self, output_units=2):

        super(DQN_Model, self).__init__()

        self.input_layer = tf.keras.layers.Dense(units=16, activation='tanh', dtype='float32')
        self.layer1 = tf.keras.layers.Dense(units=16, activation='tanh', dtype='float32')
        self.layer2 = tf.keras.layers.Dense(units=16, activation='tanh', dtype='float32')
        self.output_layer = tf.keras.layers.Dense(units=output_units, activation=None, use_bias=False, dtype='float32')

    def call(self, x_in):

        output = {}

        x = self.input_layer(x_in)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.output_layer(x)

        output['q_values'] = x

        return output

if __name__ == "__main__":

    # clear session
    #tf.keras.backend.clear_session()

    # define kwargs for model
    kwargs = {
        "model": DQN_Model,
        "environment":  "CartPole-v0",
        "num_parallel": 2,
        "total_steps": 100,
        "action_sampling_type": "epsilon_greedy",
        "num_episodes": 20,
        "epsilon": 0.90
    }

    # initialize ray, manager, saving path
    ray.init(log_to_driver=False)
    manager = SampleManager(**kwargs)
    saving_path = os.getcwd() + "\\progress_test_HW2"
    if not os.path.exists(saving_path):
        os.mkdir(saving_path)

    # initilialize parameters, buffer and aggregator
    gamma = 0.99

    buffer_size = 5000
    test_steps = 1000
    epochs = 20
    sample_size = 1000
    optim_batch_size = 8
    saving_after = 5

    optim_keys = ['state', 'action', 'reward', 'state_new', 'not_done']

    manager.initilize_buffer(buffer_size, optim_keys)

    manager.initialize_aggregator(path=saving_path, saving_after=5, aggregator_keys=['loss', 'time_steps'])

    # test before training
    print('Test before training:')
    manager.test(test_steps, render=False, do_print=True, evaluation_measure="time_and_reward")

    # get initial agent
    agent = manager.get_agent()
    #print("trainable variables:")
    #print(agent.model.trainable_variables)

    # initialize the loss
    mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)

    # initialize the optimizer
    adam = tf.keras.optimizers.Adam()

    # main training loop
    for e in range(epochs):
        print("training epoch ", e)
        losses_list = []
        # collect experience and store in buffer
        data = manager.get_data()
        manager.store_in_buffer(data)

        # sample from buffer
        sample_dict = manager.sample(sample_size)
        #print("Sample_dict: ", sample_dict)

        # create tf dataset and batch
        data_dict = dict_to_dict_of_datasets(sample_dict, batch_size=optim_batch_size)
        dataset = tf.data.Dataset.zip((data_dict['state'], data_dict['action'], data_dict['reward'], data_dict['state_new'], data_dict['not_done']))
        #data_tf = data_tf.batch(batch_size=optim_batch_size)


        for (state, action, reward, state_new, not_done) in dataset:
            with tf.GradientTape() as tape:
                q_target = agent.model(state)['q_values'].numpy()
                q_net_vals = agent.model(state)['q_values']
                max_q_subs = agent.max_q(state_new).numpy()
                for i in range(state.shape[0]):
                    im_reward = reward[i]
                    q_target_val = im_reward
                    if not_done[i]:
                        q_target_val = q_target_val + (gamma * max_q_subs[i])
                    q_target[i][action[i]] = q_target_val
                # compute loss
                q_target = tf.convert_to_tensor(q_target)
                q_target = tf.gather(q_target, action, batch_dims=1)
                q_net_vals = tf.gather(q_net_vals, action, batch_dims=1)
                #q_net_vals = tf.convert_to_tensor(q_net_vals)
                #loss = tf.keras.losses.mean_squared_error(y_true=q_target, y_pred=q_net_vals)
                loss = mse(q_target, q_net_vals)
                #loss = tf.convert_to_tensor(loss)
                #print("loss ", loss)
                losses_list.append(loss)
                gradients = tape.gradient(loss, agent.model.trainable_variables)
                #print("gradients: ", gradients)
            # apply gradients
            adam.apply_gradients(zip(gradients, agent.model.trainable_variables))

            # get and apply new weights
            new_weights = agent.get_weights()
            manager.set_agent(new_weights)

        # get new agent
        agent = manager.get_agent()

        # update aggregator
        time_steps = manager.test(test_steps, render=True if e%5 == 0 else False)
        manager.update_aggregator(loss=losses_list, time_steps=time_steps)
        print(
            f"epoch ::: {e}  loss ::: {np.mean([np.mean(l) for l in losses_list])}   avg env steps ::: {np.mean(time_steps)}"
        )
        manager.agg.save_graphic()
        print("---")

        # alter epsilon parameter
        new_epsilon = 0.90 + (0.05/(e+1))
        manager.set_epsilon(epsilon=new_epsilon)

    # test after training
    print("done")
    print("testing optimized agent")
    manager.test(test_steps, test_episodes=10, render=True, do_print=True, evaluation_measure="time_and_reward")
