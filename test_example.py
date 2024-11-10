from rl import *

learning_rate = 1e-3
optimizer ='adam'
num_iterations = 1000
collect_steps_per_iteration = 1
log_interval = 200
eval_interval = 1000
replay_buffer_capacity = 10000
batch_size = 64

hyperparameters = Hyperparameters(
   learning_rate,
   optimizer,
   num_iterations,
   collect_steps_per_iteration,
   log_interval,
   eval_interval,
   replay_buffer_capacity,
   batch_size
)

layer_params = [100]
loss_function = 'mse'

agent_config = DQNConfiguration( layer_params, loss_function)

agents = [RLAgent("DQN",hyperparameters,agent_config)]

env = Environment("CartPole-v1")

metrics = ['avg_return']
num_eval_episodes = 10

result = Result( metrics, num_eval_episodes)

rl = RL(result, agents, env)

print(rl)

# Output:
# RL(Result(['avg_return'], 10), [RLAgent(DQN, Hyperparameters(1000, 1, 200, 1000, 10000, 64), 
# DQNConfiguration([100], mse))], Environment(CartPole-v1))