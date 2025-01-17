from Besser.BUML.metamodel.rl import Hyperparameters, Agent, Result, Environment, DQNConfiguration,RLTrainer,EvaluationSettings
from Besser.generators.rl import RLGenerator



hyperparameters = Hyperparameters(
   learning_rate=1e-3,
   optimizer='adam',
   num_iterations=1000,
   collect_steps_per_iteration=2,
   log_interval= 100,
   eval_interval= 200,
   replay_buffer_capacity= 10000,
   batch_size= 64
)

agent_config_1 = DQNConfiguration( layer_params=[128,128,128,128],
                                 loss_function = 'mse')

result = Result(video=True)

agent_1= Agent(id="agent1",name="dqn",agent_config=agent_config_1,hyper_param=hyperparameters,result=result)

env = Environment("CartPole-v1")

evaluationSettings = EvaluationSettings( metrics = ['avg_return'], num_eval_episodes = 10)

rl = RLTrainer(evaluationSettings, [agent_1], env)

generator = RLGenerator(rl)
generator.generate()

print(rl)
# Output:
# RLTrainer(EvaluationSettings(['avg_return'], 10),
# [Agent(dqn, DQNConfiguration([128, 128, 128, 128], mse), 
# Hyperparameters(800, 1, 100, 400, 10000, 64), None, 
# Result(agent1,2025-01-03,True, /Users/danielcesario/Documents/Uni/Semester5/BSP/GitProject/BSP5))],
#  Environment(CartPole-v1))