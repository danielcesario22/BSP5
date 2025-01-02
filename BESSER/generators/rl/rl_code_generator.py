import os
from jinja2 import Environment, FileSystemLoader
from Besser.BUML.metamodel.rl import RLTrainer 

from besser.generators import GeneratorInterface  

class RLGenerator(GeneratorInterface):
    """
    RLGenerator is a class that implements the GeneratorInterface and is responsible for generating
    the Python reinforcement learning trainer code based on the input B-UML model.

    Args:
        model (RL): An instance of the RL Model class representing the B-UML model.
        output_dir (str, optional): The output directory where the generated code will be saved. Defaults to None.
    """
    def __init__(self, model: RLTrainer, output_dir: str = None):
        super().__init__(model, output_dir)
    


    def generate(self):
        """
        Generates Python reinforcement learning trainer code based on the provided B-UML model and saves it to the specified output directory.
        If the output directory was not specified, the code generated will be stored in the <current directory>/output
        folder.

        Returns:
            None, stores generateed code in files
        """
        templates_path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "templates")
        env = Environment(loader=FileSystemLoader(templates_path))

        # Agent Trainers
        agent_types = [agent.name for agent in self.model.agents]
        for agent_type in list(set(agent_types)):
            template_name = f"{agent_type}_template.py.j2"
            template = env.get_template(template_name)

            file_path = self.build_generation_path(file_name=f"{agent_type}_trainer.py")
            with open(file_path, mode="w") as f:
                generated_code = template.render()
                f.write(generated_code)
                print(f"Code for '{agent_type}' trainer generated in the location: {file_path}")

        # Main RL Trainer
        template_name = f"rl_template.py.j2"
        template = env.get_template(template_name)

        file_path = self.build_generation_path(file_name=f"rl_trainer.py")

        self.model.agents = sorted(self.model.agents, key=lambda agent: agent.name)
        unique_agent_types = list(set([agent.name for agent in self.model.agents]))

        with open(file_path, mode="w") as f:
            generated_code = template.render(model=self.model, agent_types=unique_agent_types)
            f.write(generated_code)
            print(f"Code for trainer generated in the location: {file_path}")

        # Metrics
        template_name = f"metrics_template.py.j2"
        template = env.get_template(template_name)

        file_path = self.build_generation_path(file_name=f"metrics.py")

        with open(file_path, mode="w") as f:
            generated_code = template.render(metrics=self.model.evaluationSettings.metrics)
            f.write(generated_code)
            print(f"Code for metrics generated in the location: {file_path}")

   