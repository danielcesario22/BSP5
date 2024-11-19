
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow as tf
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks.q_network import QNetwork
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.drivers import dynamic_step_driver
from tf_agents.policies import random_tf_policy
from tf_agents.trajectories import trajectory

# Set up the environment
env_name = 'CartPole-v1'
train_py_env = suite_gym.load(env_name)
eval_py_env = suite_gym.load(env_name)

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)





results = {}


##########
# Agent 1
##########

# Define the Q-network
fc_layer_params = (200,)
q_net = QNetwork(
     train_env.observation_spec(),
     train_env.action_spec(),
     fc_layer_params=fc_layer_params)

# Create the DQN Agent
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)


agent.initialize()

# Set Up the Replay Buffer and Data Collection

replay_buffer_capacity = 10000 
batch_size = 64

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_capacity)

def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    buffer.add_batch(traj)


# Train the Agent
num_iterations = 800
collect_steps_per_iteration = 1
log_interval = 100
eval_interval = 400

# Collect initial data
for _ in range(1000):
     collect_step(train_env, random_tf_policy.RandomTFPolicy(
          train_env.time_step_spec(), train_env.action_spec()), replay_buffer)

dataset = replay_buffer.as_dataset(
     num_parallel_calls=3, 
     sample_batch_size=batch_size, 
     num_steps=2).prefetch(3)

iterator = iter(dataset)

# Evaluate the Agent

def avg_return(environment, policy, num_episodes=10):
     total_return = 0.0
     for _ in range(num_episodes):
          time_step = environment.reset()
          episode_return = 0.0
          while not time_step.is_last():
               action_step = policy.action(time_step)
               time_step = environment.step(action_step.action)
               episode_return += time_step.reward
          total_return += episode_return
     
     avg_return = total_return / num_episodes
     return avg_return.numpy()[0]

# Train the agent
print(f'Agent1 training:')
for iteration in range(num_iterations):
     for _ in range(collect_steps_per_iteration):
          collect_step(train_env, agent.collect_policy, replay_buffer)
     experience, _ = next(iterator)
     train_loss = agent.train(experience).loss
     step = agent.train_step_counter.numpy()
     if step % log_interval == 0:
          print(f'Step {step}: loss = {train_loss}')
     if step % eval_interval == 0:
          avg_return_value = avg_return(eval_env, agent.policy, num_episodes=10)
          print(f'Step {step}: avg_return = { avg_return_value }')
print("---------------------------")

# Evaluate the trained agent
result={}
avg_return_value = avg_return(eval_env, agent.policy, num_episodes=10)
result["avg_return"]= avg_return_value
results["Agent1"]=result

##########
# Agent 2
##########

# Define the Q-network
fc_layer_params = (200, 70,)
q_net = QNetwork(
     train_env.observation_spec(),
     train_env.action_spec(),
     fc_layer_params=fc_layer_params)

# Create the DQN Agent
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)


agent.initialize()

# Set Up the Replay Buffer and Data Collection

replay_buffer_capacity = 10000 
batch_size = 64

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_capacity)

def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    buffer.add_batch(traj)


# Train the Agent
num_iterations = 800
collect_steps_per_iteration = 1
log_interval = 100
eval_interval = 400

# Collect initial data
for _ in range(1000):
     collect_step(train_env, random_tf_policy.RandomTFPolicy(
          train_env.time_step_spec(), train_env.action_spec()), replay_buffer)

dataset = replay_buffer.as_dataset(
     num_parallel_calls=3, 
     sample_batch_size=batch_size, 
     num_steps=2).prefetch(3)

iterator = iter(dataset)

# Evaluate the Agent

def avg_return(environment, policy, num_episodes=10):
     total_return = 0.0
     for _ in range(num_episodes):
          time_step = environment.reset()
          episode_return = 0.0
          while not time_step.is_last():
               action_step = policy.action(time_step)
               time_step = environment.step(action_step.action)
               episode_return += time_step.reward
          total_return += episode_return
     
     avg_return = total_return / num_episodes
     return avg_return.numpy()[0]

# Train the agent
print(f'Agent2 training:')
for iteration in range(num_iterations):
     for _ in range(collect_steps_per_iteration):
          collect_step(train_env, agent.collect_policy, replay_buffer)
     experience, _ = next(iterator)
     train_loss = agent.train(experience).loss
     step = agent.train_step_counter.numpy()
     if step % log_interval == 0:
          print(f'Step {step}: loss = {train_loss}')
     if step % eval_interval == 0:
          avg_return_value = avg_return(eval_env, agent.policy, num_episodes=10)
          print(f'Step {step}: avg_return = { avg_return_value }')
print("---------------------------")

# Evaluate the trained agent
result={}
avg_return_value = avg_return(eval_env, agent.policy, num_episodes=10)
result["avg_return"]= avg_return_value
results["Agent2"]=result


#####################################
# Display results
line=f'{"Results":<15}'
for agent in results.keys():
     line+=f'{agent:<10}'
print(line)
for metric in results[list(results.keys())[0]].keys():
     line=f'{metric:<15}'
     for agent in results.keys():
          line+=f'{results[agent][metric]:<10.2g}'
     print(line)


# Output example:
# Step 100: loss = 16.346210479736328
# Step 200: loss = 25.265539169311523
# Step 300: loss = 35.050594329833984
# Step 400: loss = 80.39982604980469
# Step 400: avg_return = 9.199999809265137
# Step 500: loss = 47.706417083740234
# Step 600: loss = 51.23229217529297
# Step 700: loss = 28.510663986206055
# Step 800: loss = 19.141983032226562
# Step 800: avg_return = 18.600000381469727
# ---------------------------
# Agent2 training:
# Step 100: loss = 17.615901947021484
# Step 200: loss = 336.9310302734375
# Step 300: loss = 9069.4501953125
# Step 400: loss = 1375.24169921875
# Step 400: avg_return = 9.399999618530273
# Step 500: loss = 99.68546295166016
# Step 600: loss = 247.015625
# Step 700: loss = 39.80617904663086
# Step 800: loss = 227.0438995361328
# Step 800: avg_return = 27.700000762939453
# ---------------------------
# Results        Agent1    Agent2    
# avg_return     22        30     
     
     

