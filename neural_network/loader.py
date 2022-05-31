"Evaluation"

import json

from .network import NeuralNetwork
from .manager import NeuralManager

class NeuroLoader(NeuralManager):
    "Class used to load a NeuralNetwork from a NeuralManager export"

    def __init__(self, name="neuro", folder="../data/"):
        super().__init__(name, folder)

        filename = self._get_filename(for_export=True)

        print(f"Loading from '{filename}'...", end=" ")

        with open(folder+filename, "r", encoding="utf-8") as file:
            data = json.loads(file.read())

        self.generation = data["generation"]
        print(f"Found generation {self.generation}!")

        self.network = NeuralNetwork.from_dict(data["network"])

    def get_genome(self, genome_class, *genome_setup_args,  **genome_setup_kwargs):
        genome = genome_class(self.network)
        genome.setup(*genome_setup_args, **genome_setup_kwargs)
        return genome
