"Evaluation"

from .network import NeuralNetwork
from .manager import NeuralManager, Genome

class NeuroLoader(NeuralManager):
    "Class used to load a neural network from a file"

    def __init__(self, name="neuro", folder="../data/", filename: str=None):
        self.network = None

        super().__init__(name, folder)

        data = self._load_data_from_file(filename)

        networkdict = data["networks"][0]
        self.network = NeuralNetwork.from_dict(networkdict)

    def get_genome(self, genome_class, *genome_setup_args,  **genome_setup_kwargs):
        genome = genome_class(self.network.clone())
        genome.setup(*genome_setup_args, **genome_setup_kwargs)
        return genome
