import gymnasium as gym
#env = gym.make('FrozenLake-v1', render_mode="human",map_name="8x8", is_slippery = False)
#env = gym.make('CliffWalking-v0')
env_name = "CartPole-v1"
#env = gym.make(env_name, render_mode="human")
env = gym.make(env_name)
env2  = gym.make(env_name)
#env.action_space.seed(42)
#from Frozen_lake import simulate_env
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
import os
import random
import scipy
from scipy.stats import gaussian_kde
import matplotlib as mpl
# Create the FrozenLake environment
#env = gym.make('FrozenLake-v0')
def custom_reward(state,action):
    n = len(state)
    q = np.ones(n)
    Q = np.diag(q)
    reward = state@Q@state.T
    return reward
class monte_carlo:
    def __init__(self,action_size,gamma,episode_length,state,env,env_name):
        self.state = state
        self.gamma = gamma
        self.episode_length = episode_length
        self.score_life_function = []
        self.env = env
        self.R_array = np.empty(1)
        self.l_array = np.empty(1)
        self.env_name = env_name
        self.action_size = action_size
#        for state in range(state_size):
#            self.score_life_function.append(np.array([[0,0],[1,0]]))
#        self.action_size = action_size
#        self.epsilon = 0.00000001
    def reset(self):
        self.R_array = np.empty(1)
        self.l_array = np.empty(1)
    def run_monte_carlo(self,desired_state,max_iterations):
        iterations = 0
        env.reset()
        while iterations < max_iterations:
            l  = 0
            R = 0
            env.state = env.unwrapped.state = desired_state
            print(env.state)
            for i in range(self.episode_length):
                #sample actions:
#                print(i)
#                print(self.action_size)
                action = env.action_space.sample()
                l = l + ((int(self.action_size))**(-i-1))*action
                observation, reward, terminated, truncated, info = env.step(action)
                reward = custom_reward(observation,action)
                R = (self.gamma**(i))*reward + R
#                if terminated == True:
#                    env.reset()
#                    break
            self.R_array = np.append(self.R_array,R)
            self.l_array = np.append(self.l_array,l)
            iterations = iterations + 1
        
    def plot(self,iteration_no):
        if not os.path.exists(f'results_monte_carlo_simulation_2{self.env_name}'):
            os.makedirs(f'results_monte_carlo_simulation_2{self.env_name}')
        sns.set(style="ticks")
        x = self.l_array
        y = self.R_array
        kernel = gaussian_kde(np.vstack([x, y]))
        c = kernel(np.vstack([x, y]))
        plt.xlim(0,1)
        plt.scatter(x, y, s=1, c=c, cmap=mpl.cm.viridis, edgecolor='none')
#        sns.scatterplot(x=self.l_array, y=self.R_array, color='purple')
        # Add labels and title to the plot
        plt.xlabel('life Values')
        plt.ylabel('S(l,x)')
#        plt.title(f'x:{iteration_no}')
        # Create a folder to save the image
        # Save the plot as a high-quality jpg image
        plt.savefig(f'results_monte_carlo_simulation_2{self.env_name}/monte_carlo_{iteration_no}.jpg', dpi=300)
        plt.close()
            # Show the plot
#            plt.show()
   

### define variables:
gamma = 0.5
action_size = env.action_space.n
episode_length = 100
max_iterations = 1000
#####
i = 0
N = 500
state = np.array([0, 0,  0, 0]) #initialize state

experiment = monte_carlo(action_size, gamma,episode_length,state,env,env_name) #initialize class
experiment.run_monte_carlo(state,max_iterations) #run monte carlo simulations
experiment.plot(state) #plot results
experiment.reset()



#env2.reset()
#env2.state = env2.unwrapped.state = initial_state
#while i < N:
#    for a in range(action_size):
#        state, reward, terminated, truncated, info = env2.step(a)
#        experiment = monte_carlo(action_size, gamma,episode_length,state,env,env_name) #initialize class
#        experiment.run_monte_carlo(state,max_iterations) #run monte carlo simulations
#        experiment.plot(state) #plot results
#        experiment.reset()
#    i = i + 1
#
######random score life functions
#
#lower_bound = -10
#upper_bound =  10
#i = 0
#N = 20
#while i < N:
#    random_state = np.random.uniform(lower_bound, upper_bound, size=4)
#    experiment = monte_carlo(action_size, gamma,episode_length,random_state,env,env_name) #initialize class
#    experiment.run_monte_carlo(random_state,max_iterations) #run monte carlo simulations
#    experiment.plot(random_state) #plot results
#    experiment.reset()
#    i = i + 1
