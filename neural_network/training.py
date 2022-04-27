"Training"

import json
import random
from tqdm import tqdm

from .network import NeuralNetwork

class Genome():
    "Genome"

    def __init__(self, network: NeuralNetwork):
        self.network = network
        self.obj = None

    def clone(self):
        return self.__class__(self.network.clone())

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

    def __init__(self, genome_class, *genome_setup_args, **genome_setup_kwargs):
        self.learning_rate_base = 0.005
        self.learning_rate_factor = 0.95
        self.mutation_chance = 0.05
        self.population_size = 100

        self.generation = 0

        self.genomes = []
        self.genome_class = genome_class
        self.genome_setup_args = genome_setup_args
        self.genome_setup_kwargs = genome_setup_kwargs

    def setup_from_scratch(self):
        for _ in range(self.population_size):
            genome = self.genome_class(self.get_default_network())
            genome.setup(*self.genome_setup_args, **self.genome_setup_kwargs)
            self.genomes.append(genome)

    def setup_from_file(self, filename):
        with open(filename, "r", encoding="utf-8") as file:
            data = json.loads(file.read())

        self.generation = data["generation"]
        network = NeuralNetwork.from_dict(data["network"])

        for _ in range(1, self.population_size):
            genome = self.genome_class(network.clone())
            genome.setup(*self.genome_setup_args, **self.genome_setup_kwargs)
            self.genomes.append(genome)

    def to_file(self, filename, network: NeuralNetwork):
        data = {
            "network": network.to_dict(),
            "generation": self.generation,
        }
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)


    def run_generation(self):
        learning_rate = self.get_learning_rate()

        print(f"Generation {self.generation} generating... (learning rate: {learning_rate})", end=" ")

        self.generate_genomes(learning_rate)
        
        print(f"Done! Running training...")

        for genome in tqdm(self.genomes, desc=f"Generation {self.generation}"):
            genome.run_evaluation(self.generation)

        self.sort_genomes()
        highscore = self.genomes[0].score

        print(f"Generation {self.generation} ended! Highscore: {highscore}")
        
        self.generation += 1

    def generate_genomes(self, learning_rate):
        if self.generation == 0:
            # No need to generate genomes in generation 0! Skipping...
            return
        
        for i in range(2, self.population_size):
            orig_index = int(random.random())

            genome = self.genomes[orig_index].clone()
            genome.network.mutate(learning_rate, self.mutation_chance)
            genome.setup(*self.genome_setup_args, **self.genome_setup_kwargs)
            self.genomes[i] = genome

    def sort_genomes(self):
        self.genomes.sort(key=lambda genome: genome.score, reverse=True)

    def get_learning_rate(self):
        return self.learning_rate_base * (self.learning_rate_factor ** self.generation)

    def get_default_network(self):
        raise NotImplementedError
