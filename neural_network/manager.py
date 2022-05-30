import os
import json

from .network import NeuralNetwork

class Genome():
    "Genome - must be subclassed"

    def __init__(self, network: NeuralNetwork):
        self.network = network
        self.obj = None

    def setup(self, *args, **kwargs):
        raise NotImplementedError

    def run_evaluation(self, generation: int = None):
        raise NotImplementedError

    def feed_forward(self, data):
        return self.network.feed_forward(data)

    @property
    def score(self):
        if hasattr(self.obj, "score"):
            if callable(self.obj.score):
                return self.obj.score()
            return self.obj.score
        raise NotImplementedError

class NeuralManager():
    "Base class for saving and loading neural network data"

    def __init__(self, name="neuro", folder="../data/"):
        self.name = name
        self.folder = folder
        self.generation = -1

        os.makedirs(self.folder, exist_ok=True)

    def _get_filename(self, for_export=False) -> str:
        if for_export:
            return f"neuro-{self.name}-export.json"
        return f"neuro-{self.name}-gen{str(self.generation).zfill(3)}.json"

    def _find_latest_filename(self) -> str:
        files = os.listdir(self.folder)
        filestart = f"neuro-{self.name}-gen"
        generations = [int(f.split(filestart)[1].split(".json")[0])
                       for f in files if f.startswith(filestart) and f.endswith(".json")]
        if not generations:
            raise FileNotFoundError(
                f"No files found in '{self.folder}' with prefix '{filestart}'")
        youngest = max(generations)
        return f"neuro-{self.name}-gen{str(youngest).zfill(3)}.json"

    def _load_data_from_file(self, filename: str = None) -> dict:
        filename = filename or self._find_latest_filename()

        print(f"Loading from file '{filename}'...", end=" ")

        with open(self.folder+filename, "r", encoding="utf-8") as file:
            data = json.loads(file.read())

        self.generation = data["generation"]
        print(f"Found generation {self.generation}!")
        return data

    def _save_state_to_file(self, data: dict, filename: str = None) -> None:
        filename = filename or self._get_filename()

        print(f"Saving state to file '{filename}'...", end=" ")

        data = {
            "_info": "NeuralManager save - can be used to continue training",
            "_generated_by": "https://github.com/rafaelurben/python-neural-network",
            **data
        }

        with open(self.folder+filename, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)

        print("Saved!")

    def _export_network_to_file(self, network: NeuralNetwork, filename: str = None) -> None:
        filename = filename or self._get_filename(for_export=True)

        print(f"Exporting network to file '{filename}'...", end=" ")

        data = {
            "_info": "NeuralNetwork export - can be used for evaluating/using the network",
            "_generated_by": "https://github.com/rafaelurben/python-neural-network",
            "generation": self.generation,
            "network": network.to_dict()
        }

        with open(self.folder+filename, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)

        print("Exported!")
