import os
# Keep using keras-2 (tf-keras) rather than keras-3 (keras).
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow as tf
from tf_agents.utils import common
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from metrics import avg_return
from dqn_trainer import DQNTrainer
import pandas as pd
from datetime import date

# Agent trainer configuartions and save configuartions
trainer_configs=[]
save_configs=[]
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
dqn_save_config_1 = {
     "agent_id":"agent1",
     "timestamp":date.today(),
     "filepath":"/Users/danielcesario/Documents/Uni/Semester5/BSP/GitProject/BSP5"
} 
trainer_configs.append(dqn_trainer_config_1)
save_configs.append(dqn_save_config_1)
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
dqn_save_config_2 = {
     "agent_id":"agent2",
     "timestamp":date.today(),
     "filepath":"/Users/danielcesario/Documents/Uni/Semester5/BSP/GitProject/BSP5"
} 
trainer_configs.append(dqn_trainer_config_2)
save_configs.append(dqn_save_config_2)
 
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

     train_results=[]
     eval_results=[]
     for trainer in trainers:
          train_result =trainer.train()
          eval_result=trainer.evaluate()
          train_results.append(train_result)
          eval_results.append( eval_result)

     # Save Training and Evaluation data
     for save_config,train_result,eval_result in zip(save_configs,train_results,eval_results):
          df1=pd.DataFrame(train_result)
          df2=pd.DataFrame([eval_result])
          filepath = save_config["filepath"]
          date= save_config["timestamp"]
          agent_id = save_config["agent_id"]
          name = eval_result["Agent"]
          with pd.ExcelWriter(f'/{filepath}/{date}-{agent_id}-{name}.xlsx') as writer:
               df1.to_excel(writer, sheet_name='Training', index=False)
               df2.to_excel(writer, sheet_name='Evaluation', index=False)


     # Display results
     column_width= 20
     line=f'{"Agent":<{column_width}}'
     for metric in list(eval_results[0].keys())[1:]:
          line+=f'{metric:<{column_width}}'
     print(line)
     for result in eval_results:
          line=f'{result["Agent"]:<{column_width}}'
          for metric in list(result.keys())[1:]:
               formatted_result = f"{result[metric]:.2f}"  
               trimmed_result = formatted_result.rstrip('0').rstrip('.') if '.' in formatted_result else formatted_result
               line+=f'{trimmed_result:<{column_width}}'
          print(line)


if __name__ == "__main__":
     main()


     
     

