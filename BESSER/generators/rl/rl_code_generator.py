import os
from jinja2 import Environment, FileSystemLoader
from Besser.BUML.metamodel.rl import RL 

from besser.generators import GeneratorInterface  

class RLGenerator(GeneratorInterface):
    """
    RLGenerator is a class that implements the GeneratorInterface and is responsible for generating
    the Python reinforcement learning trainer code based on the input B-UML model.

    Args:
        model (RL): An instance of the RL Model class representing the B-UML model.
        output_dir (str, optional): The output directory where the generated code will be saved. Defaults to None.
    """
    def __init__(self, model: RL, output_dir: str = None):
        super().__init__(model, output_dir)

    def generate(self):
        """
        Generates Python reinforcement learning trainer code based on the provided B-UML model and saves it to the specified output directory.
        If the output directory was not specified, the code generated will be stored in the <current directory>/output
        folder.

        Returns:
            None, but store the generated code as a file named rl.py 
        """
        file_path = self.build_generation_path(file_name="rl.py")
        templates_path = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "templates")
        env = Environment(loader=FileSystemLoader(templates_path))
        template = env.get_template('rl_template.py.j2')
        with open(file_path, mode="w") as f:
            generated_code = template.render(model=self.model)
            f.write(generated_code)
            print("Code generated in the location: " + file_path)