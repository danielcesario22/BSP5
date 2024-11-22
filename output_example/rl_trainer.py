import tensorflow as tf
from tf_agents.utils import common
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from metrics import avg_return
from dqn_trainer import DQNTrainer

# Agent trainer configuartions
trainer_configs=[]
dqn_trainer_config_1 = {
     "trainer":DQNTrainer,
     "agent_name":"DQN Agent 1",
     "agent_config":{
          "fc_layer_params":(200,),
          "loss_function": common.element_wise_squared_loss 
     },
     "hyper_param":{
          "learning_rate":0.001,
          "optimizer": tf.keras.optimizers.Adam ,
          "num_iterations":800,
          "collect_steps_per_iteration":1,
          "log_interval":100,
          "eval_interval":400,
          "replay_buffer_capacity":10000,
          "batch_size":64
     }
}
trainer_configs.append(dqn_trainer_config_1)
dqn_trainer_config_2 = {
     "trainer":DQNTrainer,
     "agent_name":"DQN Agent 2",
     "agent_config":{
          "fc_layer_params":(200, 70,),
          "loss_function": common.element_wise_squared_loss 
     },
     "hyper_param":{
          "learning_rate":0.001,
          "optimizer": tf.keras.optimizers.Adam ,
          "num_iterations":800,
          "collect_steps_per_iteration":1,
          "log_interval":100,
          "eval_interval":400,
          "replay_buffer_capacity":10000,
          "batch_size":64
     }
}
trainer_configs.append(dqn_trainer_config_2)
 
eval_param = {
     "metrics":{ "avg_return":avg_return },
     "num_eval_episodes": 10
}

def main():
     # Set up the environment
     env_name = 'CartPole-v1'
     train_py_env = suite_gym.load(env_name)
     eval_py_env = suite_gym.load(env_name)
     train_env = tf_py_environment.TFPyEnvironment(train_py_env)
     eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

     trainers = []

     for trainer_config in trainer_configs:
          trainers.append(trainer_config["trainer"](
                    trainer_config,
                    eval_param,
                    train_env,
                    eval_env))
     results={}
     for trainer in trainers:
          trainer.train()
          result= trainer.evaluate()
          results[trainer.agent_name]= result


     # Display results
     column_width= 20
     line=f'{"Results":<{column_width}}'
     for agent_name in results.keys():
          line+=f'{agent_name:<{column_width}}'
     print(line)
     for metric in results[list(results.keys())[0]].keys():
          line=f'{metric:<{column_width}}'
          for agent_name in results.keys():
               formatted_result = f"{results[agent_name][metric]:.2f}"  
               trimmed_result = formatted_result.rstrip('0').rstrip('.') if '.' in formatted_result else formatted_result
               line+=f'{trimmed_result:<{column_width}}'
          print(line)


if __name__ == "__main__":
     main()


     
# Output example
# -- Training DQN Agent 1 --
# Step 100: loss = 7.159775733947754
# Step 200: loss = 17.266582489013672
# Step 300: loss = 60.947608947753906
# Step 400: loss = 88.33880615234375
# Step 400: avg_return = 9.300000190734863
# Step 500: loss = 30.99675750732422
# Step 600: loss = 40.30192947387695
# Step 700: loss = 71.80934143066406
# Step 800: loss = 43.50737762451172
# Step 800: avg_return = 35.79999923706055
# -- Training DQN Agent 2 --
# Step 100: loss = 187.54759216308594
# Step 200: loss = 1850.8167724609375
# Step 300: loss = 2903.263671875
# Step 400: loss = 818.1119384765625
# Step 400: avg_return = 10.0
# Step 500: loss = 636.1046142578125
# Step 600: loss = 140.20465087890625
# Step 700: loss = 64.1087646484375
# Step 800: loss = 25.392793655395508
# Step 800: avg_return = 9.399999618530273
# Results             DQN Agent 1         DQN Agent 2         
# avg_return          35.4                9.1 