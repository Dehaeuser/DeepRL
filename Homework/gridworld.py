import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import gym
import numpy as np
import ray
from really import SampleManager
from gridworlds import GridWorld

"""
Your task is to solve the provided Gridword with tabular Q learning!
In the world there is one place where the agent cannot go, the block.
There is one terminal state where the agent receives a reward.
For each other state the agent gets a reward of 0.
The environment behaves like a gym environment.
Have fun!!!!

"""


class TabularQ(object):

    #our constructor
    def __init__(self, h, w, action_space, alpha = 0.99, gamma = 0.99):
        self.action_space = action_space
        #height of the gridworld
        self.h = h
        #width of the gridworld
        self.w = w
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        # initialise Q-table with zeros
        self.q_table = np.zeros((self.h,self.w,self.action_space))

        pass


    #our model needs to be callable to mimic a tensorflow model
    #function returns the q-values for function parameter "state"
    def __call__(self, state):
        #remove Batch Size dimension
        state= np.squeeze(state, axis = 0)
        output = {}
        output["q_values"] = self.q_table[int(state[0]), int(state[1])]
        #Add Batch Size dimension
        output["q_values"] = np.expand_dims(output["q_values"], axis = 0)
        return output

    #return all the weights for all the states
    def get_weights(self):
        return self.q_table

    #set the new weights
    def set_weights(self, q_vals):
        self.q_table = q_vals
        pass

    #function returns the maxmimal q-value for the state "state"
    def max_qVal(self, state):
        q_vals = self.q_table[int(state[0]), int(state[1])]
        max = np.max(q_vals)
        return max

    #function that optimizes the q-table and returns the mean over the losses
    def optimize(self, sample_dict):
        loss = []
        for counter, value in enumerate(sample_dict['state']):
            immediate_reward = sample_dict['reward'][counter]
            subs_state = sample_dict['state_new'][counter]
            max = self.max_qVal(subs_state)
            old_q = self.q_table[int(value[0]), int(value[1]), (sample_dict['action'][counter])]
            new_q = old_q + self.alpha*(immediate_reward + self.gamma * max - old_q)
            copy_q_table = np.copy(self.q_table)
            copy_q_table[int(value[0]), int(value[1]), (sample_dict['action'][counter])] = new_q
            self.set_weights(copy_q_table)
            loss.append(abs(old_q - new_q))
        return np.mean(loss)



if __name__ == "__main__":
    action_dict = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}

    env_kwargs = {
        "height": 3,
        "width": 4,
        "action_dict": action_dict,
        "start_position": (2, 0),
        "reward_position": (0, 3),
    }

    env = GridWorld(**env_kwargs)

    model_kwargs = {"h": env.height, "w": env.width, "action_space": 4}

    kwargs = {
        "model": TabularQ,
        "environment": GridWorld,
        "num_parallel": 2,
        "total_steps": 100,
        "model_kwargs": model_kwargs,
        "action_sampling_type": "epsilon_greedy",
    }

    ray.init(log_to_driver=False)

    saving_path=os.getcwd() + "/progress_test_HW1"

    manager = SampleManager(**kwargs)
    sample_size = 1000
    buffer_size = 5000
    saving_after = 5
    test_steps = 1000


    optim_keys = ["state", "action", "reward", "state_new", "not_done"]


    manager.initilize_buffer(buffer_size, optim_keys)

    manager.initialize_aggregator(path= saving_path, saving_after = 5, aggregator_keys=["loss", "time_steps"])


    print("test before training: ")
    manager.test(
        max_steps=12, #100
        test_episodes=1,#10,
        render=True,
        do_print=True,
        evaluation_measure="time_and_reward",
    )



    agent = manager.get_agent()

    epochs = 20

    for e in range(epochs):
        print("collecting experience..")
        data = manager.get_data()
        manager.store_in_buffer(data)
        sample_dict = manager.sample(sample_size)
        print(f"collected data for: {sample_dict.keys()}")
        print("optimizing...")

        loss = agent.model.optimize(sample_dict)
        new_weights = agent.model.get_weights()
        manager.set_agent(new_weights)
        agent = manager.get_agent()
        time_steps = manager.test(test_steps)
        manager.update_aggregator(loss=loss, time_steps=time_steps)

        # print progress
        print(
            f"epoch ::: {e}  loss ::: {loss}   avg env steps ::: {np.mean(time_steps)}"
        )


    print("done")
    print("testing optimized agent")
    manager.test(test_steps, test_episodes=10, render=True)
