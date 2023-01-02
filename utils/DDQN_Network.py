from utils import DQN, ReplayBuffer, greedy_action, epsilon_greedy, update_target, loss

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import math
import numpy as np

import gym
import matplotlib.pyplot as plt
from matplotlib import ticker, cm


def run_ddqn(policy_neurons=[4,20,20,2], target_neurons=[4,20,20,2], num_episodes=300, 
            learning_rate=0.001, EPSILON=0.3, buffer_max=20000, buffer_sample_no=64, 
            update_target_rate=10, epsilon_decay=0.999, use_decay=True, stop_epsilon=0.1,
            NUM_RUNS=10, display=False):

            """ Runs the DDQN algorithm for the CartPole environment.

            Args:
                policy_neurons: The number of neurons in each layer of the policy network.
                target_neurons: The number of neurons in each layer of the target network.
                num_episodes: The number of episodes to run the algorithm for.
                learning_rate: The learning rate for the Adam optimizer.
                EPSILON: The initial epsilon value for epsilon-greedy action selection.
                buffer_max: The maximum size of the replay buffer.
                buffer_sample_no: The number of samples to take from the replay buffer.
                update_target_rate: The number of episodes between updating the target network.
                epsilon_decay: The rate at which to decay epsilon.
                use_decay: Whether to use epsilon decay.
                stop_epsilon: The minimum value for epsilon.
                NUM_RUNS: The number of runs to average over.
                display: Whether to display the results of each run.

            Returns:
                tuple of (policy_net, target_net, runs_results, epsilons):
                    policy_net: The final policy network.
                    target_net: The final target network.
                    runs_results: The results for each run.
                    epsilons: The epsilon values for each run.
            """
    #Set up environment
    env = gym.make('CartPole-v1')
    runs_results = []
    EPSILON_ORIGINAL = EPSILON
    epsilons = [EPSILON]

    #Run the algorithm NUM_RUNS times
    for run in range(NUM_RUNS):
        if display:
          print(f"Starting run {run+1} of {NUM_RUNS}")

        #Set up the networks and replay buffer
        policy_net = DQN(policy_neurons)
        target_net = DQN(target_neurons)
        update_target(target_net, policy_net)
        target_net.eval()

        #Set up the optimizer and replay buffer
        optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
        memory = ReplayBuffer(buffer_max)

        steps_done = 0
        episode_durations = []
        EPSILON = EPSILON_ORIGINAL

        #Run the algorithm for num_episodes
        for i_episode in range(num_episodes):
            
            #Decay epsilon
            if i_episode > 0 and use_decay and EPSILON > stop_epsilon:
              EPSILON = epsilon_decay*EPSILON
              epsilons.append(EPSILON)
            
            if display:
              if (i_episode+1) % 50 == 0:
                  print("episode ", i_episode+1, "/", 300)

            #Reset the environment and state
            observation, info = env.reset()
            state = torch.tensor(observation).float()

            done = False
            terminated = False
            t = 0
            
            #Run the episode
            while not (done or terminated):

                #Select and perform an action
                action = epsilon_greedy(EPSILON, policy_net, state)

                #Observe new state
                observation, reward, done, terminated, info = env.step(action)
                reward = torch.tensor([reward])
                action = torch.tensor([action])
                next_state = torch.tensor(observation).reshape(-1).float()

                #Store the transition in memory
                memory.push([state, action, next_state, reward, torch.tensor([done])])

                #Move to the next state
                state = next_state

                #Perform one step of the optimization (on the policy network)
                if not len(memory.buffer) < buffer_sample_no:
                    transitions = memory.sample(buffer_sample_no)
                    state_batch, action_batch, nextstate_batch, reward_batch, dones = (torch.stack(x) for x in zip(*transitions))
                    
                    #Compute loss and perform backprop
                    mse_loss = loss(policy_net, target_net, state_batch, action_batch, reward_batch, nextstate_batch, dones, DDQN=True)
                    optimizer.zero_grad()
                    mse_loss.backward()
                    optimizer.step()
                
                #Check if the episode has terminated
                if done:
                    episode_durations.append(t + 1)
                t += 1

            #Update the target network, copying all weights and biases in DQN
            if i_episode % update_target_rate == 0: 
                update_target(target_net, policy_net)
        runs_results.append(episode_durations)
    
    print('Complete')
    return policy_net, target_net, runs_results, epsilons