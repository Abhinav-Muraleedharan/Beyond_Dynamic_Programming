from  Approximate_Score_life_function_computation import *
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
#env = gym.make('FrozenLake-v1', render_mode="human",map_name="8x8", is_slippery = False)
#env = gym.make('CliffWalking-v0')
env_name = "CartPole-v1"
#env = gym.make(env_name, render_mode="human")
env = gym.make(env_name)

#env = gym.make("CartPole-v0")
N = 200
X = np.array([ 1.3156445,  0.33272818, 0.01678211, -0.11442678])  #####state
gamma = 0.8
a_0 = S(0,X,gamma,N,env)
a_1 = S(1,X,gamma,N,env) - S(0,X,gamma,N,env)
j_max = 10
n = 100
a_0,a_1,coefficients = compute_faber_schauder_coefficients(X,gamma,N,j_max,env)
a_opt,b_opt,c_opt = evaluate_quadratic_score_life_function(X,n,N,gamma,env)
plt = plot_quadratic(a_opt,b_opt,c_opt)

# Generate data for x and y values
l = np.linspace(0, 1, 1000)
y = []
for val in l:
    y_val = compute_score_life_function(a_0,a_1,coefficients,val)
    y.append(y_val)

# Create a figure and axis object
#fig, ax = plt.subplots()
# Plot the data
plt.plot(l, y, color='blue', linewidth=0.5)
plt.xlabel('life Values')
plt.ylabel('S(l,x)')
plt.savefig(f'Score_life_function_{X}.jpg', dpi=300)
plt.show()

coefficients_quad = [a_opt,b_opt,c_opt]
J = optimize_quadratic(coefficients_quad)
print("Cost to go: ")
print(J)
#print(S_i_j(0,0,0))
#plt.savefig("")
env_2 = gym.make("CartPole-v1", render_mode="human")
gamma = 0.8
N = 200
j_max = 10
observation, info = env_2.reset()
k = 0
#N_action_horizon = 10
n = 200
### simulation::::
x_array =[]
x_dot_array = []
theta_array = []
theta_dot_array = []
for i in range(1000):
    #compute faber schauder coefficients:
#    action = env.action_space.sample()
    env.reset()
    Q = compute_Q(observation,env,n,N,gamma)
    print(Q)
    action = Q.index(min(Q))
    print("action:")
    print(action)
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
plt.savefig(f'Simulation_results.jpg', dpi=300)
plt.show()
env.close()
env_2.close()
