import random
from collections import deque

import torch
import torch.nn.functional as F
from gym.core import Env
from torch import nn

class ReplayBuffer():
    def __init__(self, size:int):
        """Replay buffer initialisation

        Args:
            size: maximum numbers of objects stored by replay buffer
        """
        self.size = size
        self.buffer = deque([], size)
    
    def push(self, transition)->list:
        """Push an object to the replay buffer

        Args:
            transition: object to be stored in replay buffer. Can be of any type
        
        Returns:
            The current memory of the buffer (any iterable object e.g. list)
        """  
        self.buffer.append(transition)
        return self.buffer

    def sample(self, batch_size:int)->list:
        """Get a random sample from the replay buffer
        
        Args:
            batch_size: size of sample

        Returns:
            iterable (e.g. list) with objects sampled from buffer without replacement
        """
        return random.sample(self.buffer, batch_size)

class DQN(nn.Module):
    def __init__(self, layer_sizes:list[int]):
        """
        DQN initialisation

        Args:
            layer_sizes: list with size of each layer as elements
        """
        super(DQN, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)])
    
    def forward (self, x:torch.Tensor)->torch.Tensor:
        """Forward pass through the DQN

        Args:
            x: input to the DQN
        
        Returns:
            outputted value by the DQN
        """
        for layer in self.layers:
            x = F.relu(layer(x))
        return x

def greedy_action(dqn:DQN, state:torch.Tensor)->int:
    """Select action according to a given DQN
    
    Args:
        dqn: the DQN that selects the action
        state: state at which the action is chosen

    Returns:
        Greedy action according to DQN
    """
    return int(torch.argmax(dqn(state)))

def epsilon_greedy(epsilon:float, dqn:DQN, state:torch.Tensor)->int:
    """Sample an epsilon-greedy action according to a given DQN
    
    Args:
        epsilon: parameter for epsilon-greedy action selection
        dqn: the DQN that selects the action
        state: state at which the action is chosen
    
    Returns:
        Sampled epsilon-greedy action
    """
    q_values = dqn(state)
    num_actions = q_values.shape[0]
    greedy_act = int(torch.argmax(q_values))
    p = float(torch.rand(1))
    if p>epsilon:
        return greedy_act
    else:
        return random.randint(0,num_actions-1)

def update_target(target_dqn:DQN, policy_dqn:DQN):
    """Update target network parameters using policy network.
    Does not return anything but modifies the target network passed as parameter
    
    Args:
        target_dqn: target network to be modified in-place
        policy_dqn: the DQN that selects the action
    """

    target_dqn.load_state_dict(policy_dqn.state_dict())

def loss(policy_dqn:DQN, target_dqn:DQN,
         states:torch.Tensor, actions:torch.Tensor,
         rewards:torch.Tensor, next_states:torch.Tensor, dones:torch.Tensor,
         DDQN=False)->torch.Tensor:
    """Calculate Bellman error loss
    
    Args:
        policy_dqn: policy DQN
        target_dqn: target DQN
        states: batched state tensor
        actions: batched action tensor
        rewards: batched rewards tensor
        next_states: batched next states tensor
        dones: batched Boolean tensor, True when episode terminates
        DDQN: whether to use Double DQN or not
    
    Returns:
        Float scalar tensor with loss value
    """
    if DDQN:
        policy_actions = policy_dqn(next_states).max(1).indices.reshape([-1,1])
        bellman_targets = (~dones).reshape(-1)*(target_dqn(next_states)).gather(1, policy_actions).reshape(-1) + rewards.reshape(-1)
        q_values = policy_dqn(states).gather(1, actions).reshape(-1)
        return ((q_values - bellman_targets)**2).mean()

    else:
        bellman_targets = (~dones).reshape(-1)*(target_dqn(next_states)).max(1).values + rewards.reshape(-1)
        q_values = policy_dqn(states).gather(1, actions).reshape(-1)
        return ((q_values - bellman_targets)**2).mean()

def run_policy_greedily(policy_net, num_episodes=300, NUM_RUNS=10, 
                     display=False):
    """Run a policy greedily for a number of episodes and return the average episode duration.

    Args:
        policy_net: the policy network to be run
        num_episodes: number of episodes to run
        NUM_RUNS: number of runs to average over
        display: whether to display progress or not

    Returns:
        Average episode duration
    """
    #Create environment
    env = gym.make('CartPole-v1')
    runs_results = []

    #Run policy for NUM_RUNS times
    for run in range(NUM_RUNS):
        if display:
          print(f"Starting run {run+1} of {NUM_RUNS}")

        #Run policy for num_episodes
        steps_done = 0
        episode_durations = []
        for i_episode in range(num_episodes):          
            if display:
              if (i_episode+1) % 50 == 0:
                  print("episode ", i_episode+1, "/", 300)

            #Reset environment and state
            observation, info = env.reset()
            state = torch.tensor(observation).float()
            
            #Run policy until done
            done = False
            terminated = False
            t = 0                      
            while not (done or terminated):

                #Select and perform an action
                action = greedy_action(policy_net, state)

                #Observe new state
                observation, reward, done, terminated, info = env.step(action)
                reward = torch.tensor([reward])
                action = torch.tensor([action])
                next_state = torch.tensor(observation).reshape(-1).float()

                #Move to the next state
                state = next_state      
                
                #Update statistics
                if done:
                    episode_durations.append(t + 1)                    
                t += 1

        runs_results.append(episode_durations)
    print('Complete')
    
    return runs_results

def plot_policy(policy_net, q=False, angle_range=0.2095, omega_range=3, num_episodes=300, v=0, plot_locator=False,
                plot_quads=False):
    """Plot the policy and Q values according to the network currently stored in the variable "policy_net".
    Uses a velocity v as argument.

    Args:
        policy_net: the policy network to be run
        q: whether to plot Q values or not  
        angle_range: range of angles to plot
        omega_range: range of angular velocities to plot
        num_episodes: number of episodes to run
        v: velocity to use
        plot_locator: whether to plot locator or not
        plot_quads: whether to plot quads or not
    """
  #Create range of angles and angular velocities                                
  angle_samples = 100
  omega_samples = 100
  angles = torch.linspace(angle_range, -angle_range, angle_samples)
  omegas = torch.linspace(-omega_range, omega_range, omega_samples)

  #Create grid of angles and angular velocities and calculate Q values and policy
  greedy_q_array = torch.zeros((angle_samples, omega_samples))
  policy_array = torch.zeros((angle_samples, omega_samples))
  for i, angle in enumerate(angles):
    for j, omega in enumerate(omegas):

        #Observe state
        state = torch.tensor([0., v, angle, omega])

        #Calculate Q values and policy
        with torch.no_grad():
            q_vals = policy_net(state)
            greedy_action = q_vals.argmax()
            greedy_q_array[i, j] = q_vals[greedy_action]
            policy_array[i, j] = greedy_action

  #Plot Q values or policy
  plt.figure(figsize=(6,6))

  #Plot locator
  if q:
        if plot_locator:
            cs = plt.contourf(angles, omegas, greedy_q_array.T, cmap="cividis")
            cbar = plt.colorbar(cs)
            q_title = "Q values"
        else:
            plt.contourf(angles, omegas, greedy_q_array.T, cmap='cividis')
            q_title = "Q values"
  
  #Plot policy
  else:
        if plot_locator:
            cs = plt.contourf(angles, omegas, policy_array.T, cmap='cividis')
            cbar = plt.colorbar(cs)
            q_title = "Policy"
        else:
            plt.contourf(angles, omegas, policy_array.T, cmap='cividis')
            q_title = "Policy"

  #Plot quadrants
  if plot_quads:
    texts = ["A", "B", "C", "D"]
    x_coords = [-0.1, 0.1, -0.1, 0.1]
    y_coords = [1.5, 1.5, -1.5, -1.5]

    plt.axvline(x=0, color='r', linestyle='--')
    plt.axhline(y=0, color='r', linestyle='--')

    for i in range(4):
      plt.text(x_coords[i], y_coords[i], texts[i], color="r")

  plt.xlabel("angle")
  plt.ylabel("angular velocity")
  plt.title(f"{q_title} for velocity: {v}")
  plt.show()
  