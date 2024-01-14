import gymnasium as gym
import time
#import Score_life_function_computation
from Score_life_function_computation import all
env = gym.make("CartPole-v1", render_mode="human")
gamma = 0.5
N = 100
j_max = 10
observation, info = env.reset(seed=42)
k = 0
N_action_horizon = 10
for i in range(1000):
    #compute faber schauder coefficient:
    if k == 0:
        a_0,a_1,coefficients = compute_faber_schauder_coefficients(observation,gamma,N,j_max,env)
        l_optimal, J_optimal,i_array,l_array,grad_array = compute_optimal_l(a_0,a_1,coefficients)
        action_sequence = fraction_to_binary(l_optimal,N_action_horizon)
    if k < N_action_horizon:
        action = int(action_sequence[k+1])
        k = k + 1
    if k == N_action_horizon:
        k = 0
#    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(observation)
    env.render()
    if terminated or truncated:
        print("terminating...Iterations:",i)
        break
        observation, info = env.reset()
env.close()
