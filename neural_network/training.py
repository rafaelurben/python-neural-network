"Training"

import random
import typing
from tqdm import tqdm

from .network import NeuralNetwork
from .manager import NeuralManager, Genome

class NeuroEvolution(NeuralManager):
    "Neural network training using the neuro evolution technique"

    EDITABLE_FIELDS = [
        'learning_rate_base', 'learning_rate_factor', 'mutation_chance',
        'population_size', 'repop_amount_keep', 'repop_amount_random_add',
        'repop_amount_random_mutate', 'repop_best_n']

    def __init__(self, genome_class, *genome_setup_args, name="neuro", folder="../data/", **genome_setup_kwargs):
        self.learning_rate_base = 0.01
        self.learning_rate_factor = 0.95

        # Chance of mutating a bias or weight
        self.mutation_chance = 0.05

        # Total size of a population
        self.population_size = 100

        # Settings for repopulation (before every generation)
        # 1: Best _ genomes will be kept unmutated.
        self.repop_amount_keep = 5
        # 2: _ genomes will be created randomly.
        self.repop_amount_random_add = 0
        # 3: _ genomes will be mutations of any old genomes.
        self.repop_amount_random_mutate = 0
        # 4: The rest of the population will be mutations of the top _ genomes.
        self.repop_best_n = 5

        # Settings used for the generation of new genomes
        self.genome_class = genome_class
        self.genome_setup_args = genome_setup_args
        self.genome_setup_kwargs = genome_setup_kwargs

        super().__init__(name, folder)

        self.genomes: typing.List[Genome] = []
        self.__is_setup_done = False

    def _get_repopulate_rest(self) -> int:
        "Get the remaining population size after subtracting the other repopulate settings"

        rest = self.population_size - self.repop_amount_keep - self.repop_amount_random_add - self.repop_amount_random_mutate
        if rest < 0:
            raise AssertionError("Population size is too small!")
        return rest

    def _new_genome(self, network: NeuralNetwork) -> Genome:
        "Create a new genome"

        genome = self.genome_class(network)
        genome.setup(*self.genome_setup_args, **self.genome_setup_kwargs)
        return genome

    def _create_random_genomes(self, amount) -> typing.List[Genome]:
        "Create random genomes with new networks via the _get_default_network() method"

        return [self._new_genome(self._get_default_network()) for _ in range(amount)]

    def setup_from_scratch(self) -> None:
        "SETUP: Create a completely new population"

        if self.__is_setup_done:
            raise AssertionError("Already setup!")

        for _ in range(self.population_size):
            self.genomes.append(self._new_genome(self._get_default_network()))

        self.__is_setup_done = True

    def setup_from_file(self, filename:str=None) -> None:
        "SETUP: Load the population from a file"

        if self.__is_setup_done:
            raise AssertionError("Already setup!")

        data = self._load_data_from_file(filename)

        for networkdict in data["networks"]:
            network = NeuralNetwork.from_dict(networkdict)
            self.genomes.append(self._new_genome(network))

        # If population size was made bigger, add random genomes to fill up the gap
        if len(self.genomes) < self.population_size:
            for _ in range(self.population_size - len(self.genomes)):
                self.genomes.append(self._new_genome(self._get_default_network()))

        self.__is_setup_done = True

    def setup_auto(self) -> None:
        "SETUP: Try to load the population from a file, if it exists, otherwise create a new population"

        try:
            self.setup_from_file()
        except FileNotFoundError:
            self.setup_from_scratch()

    def save_to_file(self, filename:str=None) -> None:
        "Save the population to a file"

        data = {
            "networks": list(map(lambda g: g.network.to_dict(), self.genomes)),
            "generation": self.generation,
        }
        self._save_data_to_file(data, filename)

    def run_generation(self) -> float:
        "Run a generation - return highscore"

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
        return highscore

    def _generate_genomes(self, learning_rate) -> None:
        "Generate new genomes based on the success of the previous ones"

        oldgenomes = self.genomes
        newgenomes = []

        newgenomes += self._create_random_genomes(self.repop_amount_random_add)

        indexes_to_keep = range(self.repop_amount_keep)

        for i in indexes_to_keep:
            orig = oldgenomes[i]

            network = orig.network.clone()
            newgenomes.append(self._new_genome(network))

        indexes_to_mutate = random.choices(range(self.population_size), k=self.repop_amount_random_mutate)
        indexes_to_mutate += random.choices(range(self.repop_best_n), k=self._get_repopulate_rest())

        for i in indexes_to_mutate:
            orig = oldgenomes[i]

            network = orig.network.clone_and_mutate(learning_rate, self.mutation_chance)
            newgenomes.append(self._new_genome(network))

        self.genomes = newgenomes

    def _sort_genomes(self) -> None:
        "Sort the genomes by score"

        self.genomes.sort(key=lambda genome: genome.score, reverse=True)

    def _get_learning_rate(self):
        "Calculate the learning rate for the current generation"

        return self.learning_rate_base * (self.learning_rate_factor ** self.generation)

    def _get_default_network(self):
        """
        Create a new network -> HAS to be overriden if creating a network from scratch
        or if repop_amount_random_add > 0
        """
        raise NotImplementedError
