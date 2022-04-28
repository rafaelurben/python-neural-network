"NeuralNetwork by rafaelurben"

import json
import random
from copy import deepcopy

from . import ACFUNCS
from .utils import randplusminus


class NeuralNetwork():
    """A neural network"""

    def __init__(self, sizes: list, *, default_weight: float = None, default_bias: float = None, default_acfunc: str = "relu"):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network. For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron."""

        self.sizes: list = sizes
        self.biases: list[list] = [
            [
                default_bias or randplusminus() for _ in range(sizes[i])
            ] for i in range(1, len(sizes))
        ]
        self.weights: list[list[list]] = [
            [
                [
                    default_weight or randplusminus()
                    for _ in range(sizes[i-1])
                ] for _ in range(sizes[i])
            ] for i in range(1, len(sizes))
        ]
        self.actfuncs: list[list] = [
            [
                default_acfunc for _ in range(sizes[i])
            ] for i in range(1, len(sizes))
        ]

    def __call__(self, inputs: list):
        "Process the inputs through the network"
        return self.feed_forward(inputs)

    def add_layer(self, size, *, default_weight: float = None, default_bias: float = None, default_acfunc: str = "relu"):
        "Add a layer"

        self.sizes.append(size)
        self.biases.append(
            [
                default_bias or randplusminus()
                for _ in range(size)
            ]
        )
        self.weights.append(
            [
                [
                    default_weight or randplusminus()
                    for _ in range(self.sizes[-2])
                ] for _ in range(self.sizes[-1])
            ]
        )
        self.actfuncs.append(
            [
                default_acfunc
                for _ in range(size)
            ]
        )

    # Processing

    def _feed_forward_layer(self, inputs: list, layerindex: int):
        "Process a layer"

        currlaylen = self.sizes[layerindex]
        lastlaylen = self.sizes[layerindex-1]

        if not 0 < layerindex < len(self.sizes):
            raise ValueError(
                f"Invalid layer index! Must be greater than 0 and smaller than {len(self.sizes)}.")
        if len(inputs) != lastlaylen:
            raise ValueError("Invalid number of inputs.")

        result = [
            (
                self._get_actfunc(layerindex, i)(
                    sum(
                        [
                            inputs[j] * self.weights[layerindex-1][i][j]
                            for j in range(lastlaylen)
                        ]
                    ) + self.biases[layerindex-1][i]
                )
            )
            for i in range(currlaylen)
        ]
        return result

    def feed_forward(self, inputs: list):
        "Process the inputs through the network"

        for index in range(1, len(self.sizes)):
            inputs = self._feed_forward_layer(inputs, index)

        return inputs

    def _get_actfunc(self, layerindex: int, neuronindex: int):
        "Get the activation function of a neuron"

        if not 0 < layerindex < len(self.sizes):
            raise ValueError(
                f"Invalid layer index! Must be greater than 0 and smaller than {len(self.sizes)}.")
        if not 0 <= neuronindex < self.sizes[layerindex]:
            raise ValueError(
                f"Invalid neuron index! Must be greater than 0 and smaller than {self.sizes[layerindex]}.")

        acfunc = self.actfuncs[layerindex-1][neuronindex]

        if isinstance(acfunc, str):
            if not acfunc in ACFUNCS:
                raise ValueError(f"Invalid activation function name: {acfunc}")
            return ACFUNCS[acfunc]

        return acfunc

    # Adjusting

    def mutate(self, learning_rate, mutation_chance: float = 0.01):
        "Adjust the weights and biases randomly"
        for layerindex in range(len(self.sizes)-1):
            for neuronindex, _ in enumerate(self.biases[layerindex]):
                if random.random() <= mutation_chance:
                    self.biases[layerindex][neuronindex] += randplusminus(learning_rate)
            for neuronindex, _ in enumerate(self.weights[layerindex]):
                for nextneuronindex, _ in enumerate(self.weights[layerindex][neuronindex]):
                    if random.random() <= mutation_chance:
                        self.weights[layerindex][neuronindex][nextneuronindex] += randplusminus(learning_rate)

    # Import & Export

    @classmethod
    def from_dict(cls, data: dict):
        "Import a network from a dictionary"

        newnetwork = cls(data["sizes"])
        newnetwork.biases = data["biases"]
        newnetwork.weights = data["weights"]
        newnetwork.actfuncs = data["actfuncs"]

        return newnetwork

    @classmethod
    def from_json(cls, jsondata: str):
        "Import the network from a json string"

        data: dict = json.loads(jsondata)
        return cls.from_dict(data)

    @classmethod
    def from_json_file(cls, filename: str):
        "Import the network from a json file"

        with open(filename, "r", encoding="utf8") as file:
            return cls.from_json(file.read())

    def to_dict(self):
        "Export the network to a dictionary"

        data = {
            "_info": "NeuralNetwork generated with python-neural-network by rafaelurben",
            "sizes": self.sizes,
            "biases": self.biases,
            "weights": self.weights,
            "actfuncs": self.actfuncs,
        }
        return data

    def to_json(self, indent: int = 4):
        "Export the network to a json string"

        return json.dumps(self.to_dict(), indent=indent)

    def to_json_file(self, filename: str, indent: int = 4):
        "Export the network to a json file"

        with open(filename, "w", encoding="utf8") as file:
            file.write(self.to_json(indent=indent))

    def clone(self):
        "Get a clone of the network"

        newnetwork = self.__class__(self.sizes.copy())
        newnetwork.biases = deepcopy(self.biases)
        newnetwork.weights = deepcopy(self.weights)
        newnetwork.actfuncs = deepcopy(self.actfuncs)
        return newnetwork

    def clone_and_mutate(self, learning_rate, mutation_chance: float = 0.01):
        "Get a clone of the network and mutate it"

        newnetwork = self.clone()
        newnetwork.mutate(learning_rate=learning_rate, mutation_chance=mutation_chance)
        return newnetwork
