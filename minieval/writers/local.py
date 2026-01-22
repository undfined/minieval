import json
import os
from dataclasses import asdict

from rich import print
from rich.console import Console

from minieval.datatypes import Response, Writer, WriterConfig

console = Console()


class LocalWriter(Writer):
    def __init__(self, config: WriterConfig):
        self.config = config

    def save_responses(self, task_alias: str, responses: list[Response]):
        os.makedirs(self.config.save_path, exist_ok=True)

        save_path = f"{task_alias}_responses.jsonl"
        save_path = os.path.join(self.config.save_path, save_path)

        with open(save_path, "w") as f:
            for response in responses:
                f.write(json.dumps(asdict(response)) + "\n")

        print(f"Saved responses to [bold purple]{save_path}[/bold purple]")

    def save_metrics(self, task_alias: str, metrics: dict, *, num_instances: int = 0):
        os.makedirs(self.config.save_path, exist_ok=True)

        save_path = f"{task_alias}_metrics.json"
        save_path = os.path.join(self.config.save_path, save_path)

        with open(save_path, "w") as f:
            json.dump(metrics, f)

        print(f"Saved metrics to [bold purple]{save_path}[/bold purple]")
