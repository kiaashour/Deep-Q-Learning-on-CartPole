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


def run_dqn(policy_neurons=[4,20,20,2], target_neurons=[4,20,20,2], num_episodes=300, 
            learning_rate=0.001, EPSILON=0.3, buffer_max=20000, buffer_sample_no=64, 
            update_target_rate=10, epsilon_decay=0.999, use_decay=True, stop_epsilon=0.1,
            NUM_RUNS=10, return_losses = False, display=False, show_ranges=False, return_ranges=False,
            return_ranges_v_d=False, show_ranges_v_d=False):

            """ Runs the DQN algorithm for the CartPole environment.

            Args:
                policy_neurons (list): The number of neurons in each layer of the policy network.
                target_neurons (list): The number of neurons in each layer of the target network.
                num_episodes (int): The number of episodes to run for.
                learning_rate (float): The learning rate for the Adam optimizer.
                EPSILON (float): The epsilon value for the epsilon greedy policy.
                buffer_max (int): The maximum size of the replay buffer.
                buffer_sample_no (int): The number of samples to take from the replay buffer.
                update_target_rate (int): The number of episodes between each update of the target network.
                epsilon_decay (float): The rate of decay for epsilon.
                use_decay (bool): Whether to use epsilon decay.
                stop_epsilon (float): The minimum value for epsilon.
                NUM_RUNS (int): The number of runs to average over.
                return_losses (bool): Whether to return the losses.
                display (bool): Whether to display the progress of the algorithm.
                show_ranges (bool): Whether to show the ranges of the pole angle and pole velocity.
                return_ranges (bool): Whether to return the ranges of the pole angle and pole velocity.
                return_ranges_v_d (bool): Whether to return the ranges of the cart position and velocity.
                show_ranges_v_d (bool): Whether to show the ranges of the cart position and velocity.

            Returns:
                tuple of (runs_results, losses, ranges_of_pole_v, ranges_of_pole_angle, ranges_v, ranges_d):
                    runs_results (list): The average reward for each episode over the number of runs.
                    losses (list): The average loss for each episode over the number of runs.
                    ranges_of_pole_v (list): The ranges of the pole velocity.
                    ranges_of_pole_angle (list): The ranges of the pole angle.
                    ranges_v (list): The ranges of the cart velocity.
                    ranges_d (list): The ranges of the cart position.
            """
    #Create the environment
    env = gym.make('CartPole-v1')
    runs_results = []
    EPSILON_ORIGINAL = EPSILON
    epsilons = [EPSILON]

    #If we are returning the losses, create a list to store them
    if return_losses:
      losses = []

    ranges_of_pole_v = []
    ranges_of_pole_angle = []
    ranges_v = []
    ranges_d = []    

    #Run the algorithm for the number of runs
    for run in range(NUM_RUNS):

        if display:
          print(f"Starting run {run+1} of {NUM_RUNS}")

        #Create the policy and target networks
        policy_net = DQN(policy_neurons)
        target_net = DQN(target_neurons)
        update_target(target_net, policy_net)
        target_net.eval()

        #Create the optimizer and replay buffer
        optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
        memory = ReplayBuffer(buffer_max)

        steps_done = 0

        episode_durations = []
        EPSILON = EPSILON_ORIGINAL

        #Run the algorithm for the number of episodes
        for i_episode in range(num_episodes):

            if i_episode > 0 and use_decay and EPSILON > stop_epsilon:
              EPSILON = epsilon_decay*EPSILON
              epsilons.append(EPSILON)

            #Display the progress
            if display:
              if (i_episode+1) % 50 == 0:
                  print("episode ", i_episode+1, "/", 300)

            #Removed info after obsevation for Google Colab
            observation, info = env.reset()
            state = torch.tensor(observation).float()

            #If we are returning the ranges, store the ranges
            if show_ranges:
              ranges_of_pole_v.append(state[3].numpy())
              ranges_of_pole_angle.append(state[2].numpy())

            #If we are returning the ranges, store the ranges
            if show_ranges_v_d:
              ranges_d.append(state[0].numpy())
              ranges_v.append(state[1].numpy())

            #If we are returning the losses, create a list to store them
            if return_losses:
              episode_losses = []

            done = False
            terminated = False
            t = 0                      

            #Run the algorithm until the episode is done
            while not (done or terminated):

                #Select and perform an action
                action = epsilon_greedy(EPSILON, policy_net, state)

                #Removed terminated for google colab, after done
                observation, reward, done, terminated, info = env.step(action)
                reward = torch.tensor([reward])
                action = torch.tensor([action])
                next_state = torch.tensor(observation).reshape(-1).float()

                #Store the transition in memory
                memory.push([state, action, next_state, reward, torch.tensor([done])])

                #If we are returning the ranges, store the ranges
                if show_ranges:
                  ranges_of_pole_v.append(observation[3])
                  ranges_of_pole_angle.append(observation[2])

                #If we are returning the ranges, store the ranges
                if show_ranges_v_d:
                  ranges_d.append(state[0].numpy())
                  ranges_v.append(state[1].numpy())

                #Move to the next state
                state = next_state

                #Perform one step of the optimization (on the policy network)
                running_loss = 0
                if not len(memory.buffer) < buffer_sample_no:
                    transitions = memory.sample(buffer_sample_no)
                    state_batch, action_batch, nextstate_batch, reward_batch, dones = (torch.stack(x) for x in zip(*transitions))
                    
                    #Compute loss
                    mse_loss = loss(policy_net, target_net, state_batch, action_batch, reward_batch, nextstate_batch, dones)
                    
                    #Optimize the model
                    if return_losses:
                      running_loss += mse_loss.item()                      
                    optimizer.zero_grad()
                    mse_loss.backward()
                    optimizer.step()

                #If we are returning the losses, store the losses
                if return_losses:
                  running_loss = running_loss/buffer_sample_no
                  episode_losses.append(running_loss)
                
                #Removed or terminated
                if done:
                    episode_durations.append(t + 1)
                    
                    if return_losses:
                      losses.append(np.mean(episode_losses))
                t += 1

            #Update the target network, copying all weights and biases in DQN
            if i_episode % update_target_rate == 0: 
                update_target(target_net, policy_net)
        runs_results.append(episode_durations)
    
    print('Complete')

    #If we are returning the ranges, print them
    if show_ranges:
      print("Minimum and maximum pole angular velocity:")
      print(np.min(ranges_of_pole_v), np.max(ranges_of_pole_v))
      print("Minimum and maximum pole angle:")
      print(np.min(ranges_of_pole_angle), np.max(ranges_of_pole_angle))

    if return_losses:
      return policy_net, target_net, runs_results, epsilons, losses
    
    if return_ranges:
        return policy_net, target_net, runs_results, epsilons, ranges_of_pole_v, ranges_of_pole_angle

    if return_ranges_v_d:
      return policy_net, target_net, runs_results, epsilons, ranges_d, ranges_v

    return policy_net, target_net, runs_results, epsilons