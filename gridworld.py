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

        # print("Q- Table:")
        # print(self.q_table)
        #
        # print("End Q-Table")

        pass


    #our model needs to be callable to mimic a tensorflow model
    #function returns the q-values for function parameter "state"
    def __call__(self, state):

        # print("State")
        # print(state)
        # print("State Ende")
        ## # TODO:
        #remove Batch Size dimension
        state= np.squeeze(state, axis = 0)
        output = {}

        output["q_values"] = self.q_table[int(state[0]), int(state[1])]

        #Add Batch Size dimension
        output["q_values"] = np.expand_dims(output["q_values"], axis = 0)

        # print("Output:")
        # print(output)
        # print("Output Ende")
        return output

    #return all the weights for all the states
    def get_weights(self):
        return self.q_table

    #set the new weights
    def set_weights(self, q_vals):
        self.q_table = q_vals
        pass

    # what else do you need?

    #state without batch dimension
    def max_qVal(self, state):
        #[1,1]
        q_vals = self.q_table[int(state[0]), int(state[1])]
        max = np.max(q_vals)
        return max

    #do the optimization here
    def optimize(self, sample_dict):
        loss = []
        for counter, value in enumerate(sample_dict['state']):
            # print(counter)
            # print(value)
            # print("......")
            immediate_reward = sample_dict['reward'][counter]
            # print("Immediate_reward:")
            # print(immediate_reward)
            # print("-----")
            subs_state = sample_dict['state_new'][counter]
            # print("Next state:")
            # print(subs_state)
            # print("*****")
            ##HIER FEHLER
            max = self.max_qVal(subs_state)
            # q_vals = agent.model.q_table[int(subs_state[0]), int(subs_state[1])]
            # max = np.max(q_vals)
            # print("Max:")
            # print(max)
            # print("###########")
            old_q = self.q_table[int(value[0]), int(value[1]), (sample_dict['action'][counter])]
            # print("Old Q:")
            # print(old_q)
            #Is the function correct???
            new_q = old_q + self.alpha*(immediate_reward + self.gamma * max - old_q)
            # print("New Q:")
            # print(new_q)
            #
            # print(self.q_table[int(value[0]), int(value[1]), (sample_dict['action'][counter])])
            #change q-values; how to tell them which action?
            copy_q_table = np.copy(self.q_table)
            copy_q_table[int(value[0]), int(value[1]), (sample_dict['action'][counter])] = new_q
            self.set_weights(copy_q_table)
            #agent.model.q_table[int(value[0]), int(value[1]), (sample_dict['action'][counter])] = new_q
            loss.append((new_q - old_q)*(new_q - old_q))
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

    # you can also create your environment like this after installation: env = gym.make('gridworld-v0')
    env = GridWorld(**env_kwargs)

    model_kwargs = {"h": env.height, "w": env.width, "action_space": 4}

    kwargs = {
        "model": TabularQ,
        "environment": GridWorld,
        "num_parallel": 1,#2,
        "total_steps": 100,
        "model_kwargs": model_kwargs,
        # and more
        "action_sampling_type": "epsilon_greedy",
    }

    # initilize
    ray.init(log_to_driver=False)

    saving_path=os.getcwd() + "/progress_test_HW1"

    manager = SampleManager(**kwargs)
    sample_size = 1000
    buffer_size = 5000
    saving_after = 5
    test_steps = 1000

    # keys for replay buffer -> what you will need for optimization
    optim_keys = ["state", "action", "reward", "state_new", "not_done"]

    #initialize buffer
    # buffer is not thaaaat important for this task though
    #have to remove "a" -> Schreibfehler
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



    # do the rest!!!!
    agent = manager.get_agent()

    epochs = 20
    # we will sample in each epoch and optimize with the samples
    for e in range(epochs):
        print("collecting experience..")
        data = manager.get_data()
        manager.store_in_buffer(data) # maybe this still gives an error? We could then just remove
        # the buffer for now

        #that is the data we want to use for optimizing the q-values
        sample_dict = manager.sample(sample_size)
        print(f"collected data for: {sample_dict.keys()}")
        print("optimizing...")

        loss = agent.model.optimize(sample_dict)
    ###THE FOLLOWING IS NOW IN FUNCTION OPTIMIZE
        # for counter, value in enumerate(sample_dict['state']):
        #     print(counter)
        #     print(value)
        #     print("......")
        #     immediate_reward = sample_dict['reward'][counter]
        #     print("Immediate_reward:")
        #     print(immediate_reward)
        #     print("-----")
        #     subs_state = sample_dict['state_new'][counter]
        #     print("Next state:")
        #     print(subs_state)
        #     print("*****")
        #     max = agent.model.max_qVal(value)
        #     # q_vals = agent.model.q_table[int(subs_state[0]), int(subs_state[1])]
        #     # max = np.max(q_vals)
        #     print("Max:")
        #     print(max)
        #     print("###########")
        #     old_q = agent.model.q_table[int(value[0]), int(value[1]), (sample_dict['action'][counter])]
        #     print("Old Q:")
        #     print(old_q)
        #     new_q = old_q + agent.model.alpha*(immediate_reward + agent.model.gamma * max - old_q)
        #     print("New Q:")
        #     print(new_q)
        #
        #     print(agent.model.q_table[int(value[0]), int(value[1]), (sample_dict['action'][counter])])
        #     #change q-values; how to tell them which action?
        #     copy_q_table = np.copy(agent.model.q_table)
        #     copy_q_table[int(value[0]), int(value[1]), (sample_dict['action'][counter])] = new_q
        #     agent.model.set_weights(copy_q_table)
        #     #agent.model.q_table[int(value[0]), int(value[1]), (sample_dict['action'][counter])] = new_q



        #afterwards set new weights and tell agent about new weights
        new_weights = agent.model.get_weights()
        manager.set_agent(new_weights)
        agent = manager.get_agent()
        time_steps = manager.test(test_steps)
        manager.update_aggregator(loss=loss, time_steps=time_steps)

        # print progress
        print(
            f"epoch ::: {e}  loss ::: {loss}   avg env steps ::: {np.mean(time_steps)}"
        )

        # yeu can also alter your managers parameters
        #manager.set_epsilon(epsilon=0.99)

        # if e % saving_after == 0:
        #     # you can save models
        #     manager.save_model(saving_path, e)

    # and load mmodels
    manager.load_model(saving_path)
    print("done")
    print("testing optimized agent")
    manager.test(test_steps, test_episodes=10, render=True)
