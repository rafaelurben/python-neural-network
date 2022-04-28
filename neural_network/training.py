"Training"

import json
import random
import os
import typing
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

        # Size of the population
        self.population_size = 100

        # REPOPULATION
        # Best _ genomes will be kept in the population
        self.repopulate_keep = int(0.05*self.population_size)
        # _ genomes will be added randomly
        self.repopulate_random_add = int(0.05*self.population_size)
        # _ genomes will be mutations of any old genomes
        self.repopulate_random_mutate = int(0.05*self.population_size)
        # The rest of the population will be mutations of the top _ genomes.
        self.repopulate_best_n = int(0.05*self.population_size)

        self.generation = -1

        self.genomes: typing.List[Genome] = []
        self.genome_class = genome_class
        self.genome_setup_args = genome_setup_args
        self.genome_setup_kwargs = genome_setup_kwargs

        self.name = name
        self.folder = folder

        self.__is_setup_done = False
        self.__after_init()

    def __after_init(self):
        os.makedirs(self.folder, exist_ok=True)

    def _get_repopulate_rest(self):
        return self.population_size - self.repopulate_keep - self.repopulate_random_add - self.repopulate_random_mutate

    def _new_genome(self, network: NeuralNetwork):
        genome = self.genome_class(network)
        genome.setup(*self.genome_setup_args, **self.genome_setup_kwargs)
        return genome

    def _create_random_genomes(self, amount):
        return [self._new_genome(self._get_default_network()) for _ in range(amount)]

    def _get_filename(self):
        return f"neuro-{self.name}-gen{str(self.generation).zfill(3)}.json"

    def _find_latest_filename(self):
        files = os.listdir(self.folder)
        filestart = f"neuro-{self.name}-gen"
        generations = [int(f.split(filestart)[1].split(".json")[0]) for f in files if f.startswith(filestart) and f.endswith(".json")]
        if not generations:
            raise FileNotFoundError(f"No files found in '{self.folder}' with prefix '{filestart}'")
        youngest = max(generations)
        return f"neuro-{self.name}-gen{str(youngest).zfill(3)}.json"

    def setup_from_scratch(self):
        if self.__is_setup_done:
            raise AssertionError("Already setup!")

        for _ in range(self.population_size):
            self.genomes.append(self._new_genome(self._get_default_network()))

        self.__is_setup_done = True

    def setup_from_file(self, filename:str=None):
        if self.__is_setup_done:
            raise AssertionError("Already setup!")

        filename = filename or self._find_latest_filename()

        print(f"Loading from file '{filename}'...", end=" ")

        with open(self.folder+filename, "r", encoding="utf-8") as file:
            data = json.loads(file.read())

        self.generation = data["generation"]

        for networkdict in data["networks"]:
            network = NeuralNetwork.from_dict(networkdict)
            self.genomes.append(self._new_genome(network))

        print(f"Loaded generation {self.generation}!")

        self.__is_setup_done = True

    def setup_auto(self):
        try:
            self.setup_from_file()
        except FileNotFoundError:
            self.setup_from_scratch()

    def save_to_file(self, filename:str=None):
        filename = filename or self._get_filename()

        print(f"Saving to file '{filename}'...", end=" ")

        data = {
            "networks": list(map(lambda g: g.network.to_dict(), self.genomes)),
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
        raise NotImplementedError
