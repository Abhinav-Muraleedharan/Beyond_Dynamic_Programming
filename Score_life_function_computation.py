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
    q = np.array([2,1,8,1])
    Q = np.diag(q)
    reward = state@Q@state.T
    return reward

def fraction_to_binary(fraction, num_bits=20):
    if fraction == 0:
        return '.' + '0' * (num_bits - 1)
    elif fraction == 1:
        return  '.' + '1' * num_bits
    else:
        binary = ''
        # Check if the fraction is less than 1
        if fraction < 1:
            binary += '.'
        for i in range(num_bits):
            fraction *= 2
            if fraction >= 1:
                binary += '1'
                fraction -= 1
            else:
                binary += '0'
        return binary


def S(l,X,gamma,N,env):
#    env = gym.make("CartPole-v0")
    env.reset()
    R = 0
    action_sequence = fraction_to_binary(l,num_bits = N)
#    print(action_sequence)
    env.state = env.unwrapped.state = X
    for i in range(len(action_sequence)-1):
        action = int(action_sequence[i+1])
        state, reward, terminated, truncated, info  = env.step(action)
        reward = custom_reward(state,action)
#        reward = -reward
        R = (gamma**(i))*reward + R
    env.close()
    return R
    
def compute_a_ij(i,j,X,gamma,N,env):
  l_1 = (2*i + 1)/(2**(j+1))
  l_2 = i/(2**j)
  l_3 = (i+1)/(2**j)
  a_ij = S(l_1,X, gamma,N,env) - 0.5*(S(l_2,X, gamma,N,env)+ S(l_3,X,gamma,N,env))
  return a_ij

env = gym.make("CartPole-v0")
N = 100
X = np.array([0,0,0,0])
gamma = 0.5
a_0 = S(0,X,gamma,N,env)
#print(a_0)
a_1 = S(1,X,gamma,N,env) - S(0,X,gamma,N,env)
#print(a_1)
def compute_faber_schauder_coefficients(X,gamma,N,j_max,env):
    a_0 = S(0,X,gamma,N,env)
    print(a_0)
    a_1 = S(1,X,gamma,N,env) - S(0,X,gamma,N,env)
    print(a_1)
    ####compute a_i,j
    i = 0
    j = 0
    coefficients = []
    while j < j_max:
        i = 0
        c_j = []
        while i <= 2**j - 1:
            a_i_j = compute_a_ij(i,j,X,gamma,N,env)
            c_j.append(a_i_j)
            i = i + 1
        coefficients.append(c_j)
        j = j + 1
    return a_0,a_1, coefficients
j_max = 10
#a_0,a_1,coefficients = compute_faber_schauder_coefficients(X,gamma,N,j_max,env)
#print(a_0)
#print(a_1)
#print(coefficients[0])
#print(coefficients[1])

def derivative_mod_x(a,b,x):
    ##function to compute derivative of |ax - b|
    if x == b/a:
        derivative = -a
    else:
        derivative = a*(abs(a*x - b)/(a*x - b))
    return derivative
    
def d_S_i_j(l,i,j):
    derivative = (2**j)*(derivative_mod_x(1,(i/(2**j)),l) + derivative_mod_x(1,((i+1)/(2**j)),l) - derivative_mod_x(2,((2*i+1)/(2**j)),l))
    return derivative

def grad_score_life_function(a_0,a_1,coefficients,l):
    grad_f = a_1
    j_max = len(coefficients)
    j = 0
    while j < j_max:
        i = 0
        while i <=2**j - 1:
            grad_f = grad_f + d_S_i_j(l,i,j)*coefficients[j][i]
            i = i + 1
        j = j + 1
    return grad_f


def S_i_j(l,i,j):
    val = (2**j)*(abs(l-(i/(2**j))) + abs(l-((i+1)/(2**j))) - abs(2*l-((2*i+1)/(2**(j)))))
    return val
    
def compute_score_life_function(a_0,a_1,coefficients,l):
    f = a_0 + a_1*l
    j_max = len(coefficients)
    j = 0
    while j < j_max:
        i = 0
        while i <= 2**j - 1:
            f = f + S_i_j(l,i,j)*coefficients[j][i]
            i = i + 1
        j = j + 1
    
    return f
#S_l = compute_score_life_function(a_0,a_1,coefficients,0.3333333)
#print(S_l)
####plot faber schauder function
a_0,a_1,coefficients = compute_faber_schauder_coefficients(X,gamma,N,j_max,env)
# Generate data for x and y values
l = np.linspace(0, 1, 1000)
y = []
for val in l:
    y_val = compute_score_life_function(a_0,a_1,coefficients,val)
    y.append(y_val)

# Create a figure and axis object
fig, ax = plt.subplots()

# Plot the data
ax.plot(l, y, color='blue', linewidth=0.5)

# Set the title and axis labels
ax.set_title('Exact Representation of Score-life function')
ax.set_xlabel('l')
ax.set_ylabel('S(l,x)')

# Set the x-axis limits
ax.set_xlim([0, 1])

# Display the plot
#plt.show()
plt.savefig("Score_life_function")
#print(S_i_j(0,0,0))
def compute_optimal_l(a_0,a_1,coefficients):
    ####optimize Score-life function:
    max_iter = 6000
    i = 0
    lr = 0.01 # learning rate
    l = 0.001 #initialize l
    grad_prev = 0
    l_array = []
    grad_array = []
    i_array = []
    while i < max_iter:
        if i == 0:
            l = random.random()
        grad = grad_score_life_function(a_0,a_1,coefficients,l)
        l = l - grad*lr
        lr = lr*(2**(-i))
#        print(l)
#        print("square of gradient:")
        grad_sq = grad**2
#        print(grad**2)
        grad_array.append(grad_sq)
        l_array.append(l)
        i_array.append(i)
        if grad*grad_prev < 0:
#            print("grad square:")
#            print(grad**2)
            if grad**2 < 0.01:
                break
        grad_prev = grad
        if l < 0:
            l = 0
            break
        if l > 1:
            l = 0.9999999
            break
        i = i +1
    print("Optimal l:!!")
    print("iterations:",i)
    print(l)
    print("Optimal Cost:")
    J_optimal = compute_score_life_function(a_0,a_1,coefficients,l)
    print(J_optimal)
    return l, J_optimal,i_array,l_array,grad_array
    
l_optimal, J_optimal,i_array,l_array,grad_array = compute_optimal_l(a_0,a_1,coefficients)

fig, ax = plt.subplots()

# Plot the data
ax.plot(i_array,grad_array, color='blue', linewidth=2)
ax.plot(i_array,l_array, color='red', linewidth=2)

# Set the title and axis labels
ax.set_title('Fractal Optimization Convergence Plot')
ax.set_xlabel('l')
ax.set_ylabel('grad_squared')

# Set the x-axis limits
ax.set_xlim([0, 1])

# Display the plot
plt.show()
#plt.savefig("")
env_2 = gym.make("CartPole-v1", render_mode="human")
gamma = 0.5
N = 100
j_max = 10
observation, info = env_2.reset(seed=42)
k = 0
N_action_horizon = 10
x_array =[]
x_dot_array = []
theta_array = []
theta_dot_array = []
for i in range(1000):
    #compute faber schauder coefficients:
    if k == 0:
        a_0,a_1,coefficients = compute_faber_schauder_coefficients(observation,gamma,N,j_max,env)
        l_optimal, J_optimal,i_array,l_array,grad_array = compute_optimal_l(a_0,a_1,coefficients)
        action_sequence = fraction_to_binary(l_optimal,N_action_horizon)
        print(action_sequence)
    if k < N_action_horizon - 1:
        action = int(action_sequence[k+1])
        k = k + 1
        print(k)
    if k == N_action_horizon-1:
        k = 0
#    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env_2.step(action)
    print(observation)
    x_array.append(observation[0])
    x_dot_array.append(observation[1])
    theta_array.append(observation[2])
    theta_dot_array.append(observation[3])
    env_2.render()
    if terminated or truncated:
        print("terminating...Iterations:",i)
        break
        observation, info = env.reset()

fig, ax = plt.subplots()
ax.plot(x_array, label=f'Trajectory - x')
ax.plot(x_dot_array, label=f'Trajectory  - x_dot')
ax.plot(theta_array, label=f'Trajectory - theta')
ax.plot(theta_dot_array, label=f'Trajectory  - theta_dot')
ax.set_title('Simulation Trajectories')
ax.set_xlabel('Time')
ax.set_ylabel('Values')
# Show legend
ax.legend()
# Display the plot
plt.savefig(f'Simulation_results_exact.jpg', dpi=300)
plt.show()
env.close()
env_2.close()
