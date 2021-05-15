# Deep RL
Repository for the course Deep reinforcement learning at University of Osnabrueck in the winter term 2020/2021.
Repository contains solutions to the homework assignments and the final project.
All homeworks are solved using the [ReAlly framework](https://github.com/geronimocharlie/ReAllY) by Charlie Lange.

## Homework
Solving Homeworks of the Deep reinforcement learning course at University of Osnabrueck in the winter term 2020/2021.

### Homework 1
Solving a gridworld environment using Watkins Q-Learning.

### Homework 2
Solving the OpenAI gym environment [CartPole-v0](https://gym.openai.com/envs/CartPole-v0/) using 
a Vanilla-DQN approach.

### Homework 3
Solving the OpenAI gym environment [LunarLanderContinuous-v2](https://gym.openai.com/envs/LunarLanderContinuous-v2/) using a Policy Gradient Algorithm. Implemented in two ways: 1. Actor only model, 2. Advantage Actor-Critic model

## Final Project
The folder contains all files needed for our reimplementation of Language Emergence in Multi-Agent Systems as published in [Ossenkopf (2020)](https://openreview.net/forum?id=Hke1gySFvB).

### Structure

[Contribution guidelines for this project](FinalProject/Archiv.py)
The Archiv contains older versions of our files tracking our progress.

The visa_dataset contains all necessary files to acces the visA dataset.
Agents.py contains the implementations of the receiver and the auxillary network.

Sender2.py contains the implementation of the sender.
Files needed for environment are create_data.py, vocab.py and env2.py.

To run implementation, execute file Game_Sender_sees_all_input.ipynb or Game_Sender_sees_only_target.ipynb.

Game_Error_Recreation.ipynb is used to illustrate the error message we get when we try to follow the paper exactly in the implementation.
