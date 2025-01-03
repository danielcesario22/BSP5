from Besser.BUML.metamodel.rl import Hyperparameters, Agent, Result, Environment, DQNConfiguration,RLTrainer,EvaluationSettings
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

layer_params_1 = [128,128,128,128]

loss_function = 'mse'

agent_config_1 = DQNConfiguration( layer_params_1, loss_function)


agent_1= Agent(id="agent1",name="dqn",agent_config=agent_config_1,hyper_param=hyperparameters)
agent_1.result.video = True
agents = [agent_1]

env = Environment("CartPole-v1")

metrics = ['avg_return']
num_eval_episodes = 10

evaluationSettings = EvaluationSettings( metrics, num_eval_episodes)

rl = RLTrainer(evaluationSettings, agents, env)

print(rl)
# Output:
# RLTrainer(EvaluationSettings(['avg_return'], 10),
# [Agent(dqn, DQNConfiguration([128, 128, 128, 128], mse), 
# Hyperparameters(800, 1, 100, 400, 10000, 64), None, 
# Result(agent1,2025-01-03,True, /Users/danielcesario/Documents/Uni/Semester5/BSP/GitProject/BSP5))],
#  Environment(CartPole-v1))

generator = RLGenerator(rl)
generator.generate()