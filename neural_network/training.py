"Training"

import random
import typing
from tqdm import tqdm

from .network import NeuralNetwork
from .manager import NeuralManager, Genome

class NeuroEvolution(NeuralManager):
    "Base class for neural network training using NeuroEvolution"

    def __init__(self, genome_class, *genome_setup_args, name="neuro", folder="../data/", **genome_setup_kwargs):
        self.learning_rate_base = 0.01
        self.learning_rate_factor = 0.95
        self.mutation_chance = 0.05

        self.set_population_size(100)

        self.genomes: typing.List[Genome] = []
        self.genome_class = genome_class
        self.genome_setup_args = genome_setup_args
        self.genome_setup_kwargs = genome_setup_kwargs

        self.default_network: NeuralNetwork = None

        super().__init__(name, folder)

        self.__is_setup_done = False

    def set_population_size(self, size: int):
        "Set the size of the population and the repopulation options relative to it"
        self.population_size = size

        # Best _ genomes will be kept in the population
        self.repopulate_keep = int(0.05*size)
        # _ genomes will be added randomly
        self.repopulate_random_add = int(0.05*size)
        # _ genomes will be mutations of any old genomes
        self.repopulate_random_mutate = int(0.05*size)
        # The rest of the population will be mutations of the top _ genomes.
        self.repopulate_best_n = int(0.05*size)

    def _get_repopulate_rest(self):
        return self.population_size - self.repopulate_keep - self.repopulate_random_add - self.repopulate_random_mutate

    def _new_genome(self, network: NeuralNetwork):
        genome = self.genome_class(network)
        genome.setup(*self.genome_setup_args, **self.genome_setup_kwargs)
        return genome

    def _create_random_genomes(self, amount):
        return [self._new_genome(self._get_default_network()) for _ in range(amount)]

    def setup_from_scratch(self):
        if self.__is_setup_done:
            raise AssertionError("Already setup!")

        for _ in range(self.population_size):
            self.genomes.append(self._new_genome(self._get_default_network()))

        self.__is_setup_done = True

    def setup_from_file(self, filename:str=None):
        if self.__is_setup_done:
            raise AssertionError("Already setup!")

        data = self._load_data_from_file(filename)

        for networkdict in data["networks"]:
            network = NeuralNetwork.from_dict(networkdict)
            self.genomes.append(self._new_genome(network))

        self.__is_setup_done = True

    def setup_auto(self):
        try:
            self.setup_from_file()
        except FileNotFoundError:
            self.setup_from_scratch()

    def save_to_file(self, filename:str=None):
        data = {
            "networks": list(map(lambda g: g.network.to_dict(), self.genomes)),
            "generation": self.generation,
        }
        self._save_data_to_file(data, filename)

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
        oldgenomes = self.genomes
        newgenomes = []

        newgenomes += self._create_random_genomes(self.repopulate_random_add)

        indexes_to_keep = range(self.repopulate_keep)

        for i in indexes_to_keep:
            orig = oldgenomes[i]

            network = orig.network.clone()
            newgenomes.append(self._new_genome(network))

        indexes_to_mutate = random.choices(range(self.population_size), k=self.repopulate_random_mutate)
        indexes_to_mutate += random.choices(range(self.repopulate_best_n), k=self._get_repopulate_rest())

        for i in indexes_to_mutate:
            orig = oldgenomes[i]

            network = orig.network.clone_and_mutate(learning_rate, self.mutation_chance)
            newgenomes.append(self._new_genome(network))

        self.genomes = newgenomes

    def _sort_genomes(self):
        self.genomes.sort(key=lambda genome: genome.score, reverse=True)

    def _get_learning_rate(self):
        return self.learning_rate_base * (self.learning_rate_factor ** self.generation)

    def _get_default_network(self):
        if isinstance(getattr(self, "default_network", None), NeuralNetwork):
            return self.default_network.clone()
        raise NotImplementedError
