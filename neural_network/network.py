"NeuralNetwork by rafaelurben"

import json

from . import ACFUNCS


class NeuralNetwork():
    """A neural network"""

    def __init__(self, sizes: list, *, default_weight: float = 0.5, default_bias: float = 0, default_acfunc: str = "relu"):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network. For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron."""

        self._default_weight = default_weight
        self._default_bias = default_bias
        self._default_acfunc = default_acfunc

        self.sizes: list = sizes
        self.biases: list[list] = [
            [default_bias for _ in range(sizes[i])] for i in range(len(sizes))
        ]
        self.weights: list[list[list]] = [
            [
                [default_weight for _ in range(sizes[i-1])] for _ in range(sizes[i])
            ] for i in range(1, len(sizes))
        ]
        self.acfuncs: list[list] = [
            [default_acfunc for _ in range(sizes[i])] for i in range(1, len(sizes))
        ]

    def add_layer(self, size):
        "Add a layer"

        self.sizes.append(size)
        self.biases.append([self._default_bias for _ in range(size)])
        self.weights.append(
            [
                [self._default_weight for _ in range(self.sizes[-2])] for _ in range(self.sizes[-1])
            ]
        )
        self.acfuncs.append([self._default_acfunc for _ in range(size)])

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

        acfunc = self.acfuncs[layerindex-1][neuronindex]

        if isinstance(acfunc, str):
            if not acfunc in ACFUNCS:
                raise ValueError(f"Invalid activation function name: {acfunc}")
            return ACFUNCS[acfunc]

        return acfunc


    # Import & Export

    @classmethod
    def from_json(cls, jsondata: str):
        "Import the network from a json file"

        data: dict = json.loads(jsondata)

        newnetwork = cls(data["sizes"])
        newnetwork.biases = data["biases"]
        newnetwork.weights = data["weights"]

        return newnetwork

    @classmethod
    def from_json_file(cls, filename: str):
        "Import the network from a json file"

        with open(filename, "r", encoding="utf8") as file:
            return cls.from_json(file.read())

    def to_json(self, indent: int = None):
        "Export the network to a json file"

        data = {
            "_info": "NeuralNetwork generated with python-neural-network by rafaelurben",
            "sizes": self.sizes,
            "biases": self.biases,
            "weights": self.weights,
        }
        return json.dumps(data, indent=indent)

    def to_json_file(self, filename: str, indent: int = 4):
        "Export the network to a json file"

        with open(filename, "w", encoding="utf8") as file:
            file.write(self.to_json(indent=indent))

    def clone(self):
        "Get a clone of the network"
        return self.from_json(self.to_json())
