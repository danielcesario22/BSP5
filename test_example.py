from Besser.BUML.metamodel.rl import Hyperparameters, Agent, Result, Environment, DQNConfiguration,RL
from Besser.generators.rl import RLGenerator

learning_rate = 1e-3
optimizer ='adam'
num_iterations = 800
collect_steps_per_iteration = 1
log_interval = 100
eval_interval = 400
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

layer_params_1 = [200]
layer_params_2 = [200,70]
loss_function = 'mse'

agent_config_1 = DQNConfiguration( layer_params_1, loss_function)
agent_config_2 = DQNConfiguration( layer_params_2, loss_function)

agent_1= Agent(name="DQN",agent_config=agent_config_1,hyper_param=hyperparameters)
agent_2= Agent(name="DQN",agent_config=agent_config_2,hyper_param=hyperparameters)
agents = [agent_1,agent_2]

env = Environment("CartPole-v1")

metrics = ['avg_return']
num_eval_episodes = 10

result = Result( metrics, num_eval_episodes)

rl = RL(result, agents, env)

print(rl)
# Output:
# RL(Result(['avg_return'], 10), 
# [Agent(DQN, DQNConfiguration([200], mse), Hyperparameters(800, 1, 100, 400, 10000, 64)), 
# Agent(DQN, DQNConfiguration([200, 70], mse), Hyperparameters(800, 1, 100, 400, 10000, 64))], 
# Environment(CartPole-v1))

generator = RLGenerator(rl)
generator.generate()