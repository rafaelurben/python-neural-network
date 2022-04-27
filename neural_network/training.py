"Training"

import json
import random
import os
from tqdm import tqdm

from .network import NeuralNetwork

class Genome():
    "Genome"

    def __init__(self, network: NeuralNetwork):
        self.network = network
        self.obj = None

    def setup(self, *args, **kwargs):
        raise NotImplementedError

    def run_evaluation(self, generation: int):
        raise NotImplementedError

    @property
    def score(self):
        if hasattr(self.obj, "score"):
            if callable(self.obj.score):
                return self.obj.score()
            return self.obj.score
        raise NotImplementedError

class NeuroEvolution():
    "Base class for neural network training using NeuroEvolution"

    def __init__(self, genome_class, *genome_setup_args, name="neuro", folder="../data/", **genome_setup_kwargs):
        self.learning_rate_base = 0.01
        self.learning_rate_factor = 0.95
        self.mutation_chance = 0.05
        self.population_size = 100

        self.generation = -1

        self.genomes = []
        self.genome_class = genome_class
        self.genome_setup_args = genome_setup_args
        self.genome_setup_kwargs = genome_setup_kwargs

        self.name = name
        self.folder = folder

    def _new_genome(self, network: NeuralNetwork):
        genome = self.genome_class(network)
        genome.setup(*self.genome_setup_args, **self.genome_setup_kwargs)
        return genome

    def _get_filename(self):
        return f"neuro-{self.name}-gen{str(self.generation).zfill(3)}.json"

    def _find_latest_filename(self):
        files = os.listdir(self.folder)
        filestart = f"neuro-{self.name}-gen"
        generations = [int(f.split(filestart)[1].split(".json")[0]) for f in files if f.startswith(filestart) and f.endswith(".json")]
        youngest = max(generations)
        return f"neuro-{self.name}-gen{str(youngest).zfill(3)}.json"

    def setup_from_scratch(self):
        for _ in range(self.population_size):
            self.genomes.append(self._new_genome(self._get_default_network()))

    def setup_from_file(self, filename:str=None):
        filename = filename or self._find_latest_filename()

        print(f"Loading from file '{filename}'...", end=" ")

        with open(self.folder+filename, "r", encoding="utf-8") as file:
            data = json.loads(file.read())

        self.generation = data["generation"]
        network = NeuralNetwork.from_dict(data["network"])
        self.genomes.append(self._new_genome(network))
        self.genomes.append(self._new_genome(network.clone()))

        for _ in range(self.population_size-2):
            self.genomes.append(None)

        self._generate_genomes(self._get_learning_rate())

        print("Loaded!")

    def to_file(self, filename:str=None):
        filename = filename or self._get_filename()
        
        print(f"Saving to file '{filename}'...", end=" ")

        data = {
            "network": self.genomes[0].network.to_dict(),
            "generation": self.generation,
        }
        with open(self.folder+filename, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)

        print("Saved!")

    def run_generation(self):
        self.generation += 1
        learning_rate = self._get_learning_rate()

        print(f"Generation {self.generation} generating... (learning rate: {learning_rate})", end=" ")

        if self.generation > 0:
            # No need to generate genomes in generation 0!
            self._generate_genomes(learning_rate)

        print("Done! Running training...")

        for genome in tqdm(self.genomes, desc=f"Generation {self.generation}"):
            genome.run_evaluation(self.generation)

        self._sort_genomes()
        highscore = self.genomes[0].score

        print(f"Generation {self.generation} ended! Highscore: {highscore}")

    def _generate_genomes(self, learning_rate):
        for i in range(2, self.population_size):
            orig = self.genomes[int(random.random())]

            network = orig.network.clone()
            network.mutate(learning_rate, self.mutation_chance)
            self.genomes[i] = self._new_genome(network)

    def _sort_genomes(self):
        self.genomes.sort(key=lambda genome: genome.score, reverse=True)

    def _get_learning_rate(self):
        return self.learning_rate_base * (self.learning_rate_factor ** self.generation)

    def _get_default_network(self):
        raise NotImplementedError
