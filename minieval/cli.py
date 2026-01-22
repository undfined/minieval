import os
import sys
from dataclasses import dataclass
from typing import Optional

from rich.columns import Columns
from rich.console import Console
from rich.pretty import pprint
from rich.text import Text

from minieval.backends import BackendType, init_backend, init_template
from minieval.datatypes import (
    Formatter,
    Instance,
    LauncherConfig,
    LauncherType,
    LMOutput,
    LMRequest,
    Metric,
    ModelConfig,
    RequestType,
    Response,
    Scorer,
    TaskConfig,
    WriterConfig,
)
from minieval.formatters import base_model_chat_template, empty_chat_template
from minieval.utils import WELCOME_MSG, apply_overrides
from minieval.writers import init_writer
from minieval.writers.datalake.datalake import DatalakeConfig

console = Console()


# Manually disable parallization in tokenizers to allow using threading
# @davidh -- If we know a better place to put this, that'd be great.
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class RunnerConfig:
    backend: BackendType
    model: ModelConfig
    tasks: list[TaskConfig]
    launcher: LauncherConfig
    writer: WriterConfig


class EvalRunner:
    def __init__(self, config: RunnerConfig):
        self.config = config

        self.llm = init_backend(backend_type=self.config.backend, model_name=self.config.model.name)

        self.writer = init_writer(
            config=self.config.writer, writer_type=self.config.writer.writer_type
        )

    def run(self, task_config: TaskConfig):
        console.rule(f"[bold red]{task_config.alias}")

        # Instantiate task from Task Registry
        from minieval.task_registry import TaskRegistry
        TaskClass = TaskRegistry.get_task(task_config.alias)

        task = TaskClass(config=task_config)

        formatter: Formatter = task_config.formatter
        scorers: list[Scorer] = task_config.scorer
        metrics: list[Metric] = task_config.metric

        # Choose the template function based on the task's request type. For example,
        # MC/RC tasks will never use a chat template
        match formatter.REQUEST_TYPE:
            case RequestType.CHAT:
                template_func = init_template(
                    backend_type=self.config.backend, model_name=self.config.model.name
                )
            case RequestType.GENERATE | RequestType.PPL:
                template_func = empty_chat_template
            case RequestType.LOGPROBS:
                template_func = base_model_chat_template

        # Build requests
        instances: list[Instance] = task.requests
        _: list[Instance] = task.build_few_shot()

        if task_config.limit:
            assert task_config.limit <= len(instances), \
                f"Limit is larger than the dataset! {task_config.limit=}, {len(instances)=}"
            instances = instances[: task_config.limit]

        # Build messages. E.g. [{"role": "user", "content": ...}]
        messages: list[LMRequest] = formatter.build_messages(instances)

        # Build request strings. E.g. "Question: ...\nAnswer:"
        requests: list[LMRequest] = formatter.build_requests(template_func, messages)

        # Generate / compute PPL from model
        match formatter.REQUEST_TYPE:
            case RequestType.GENERATE | RequestType.CHAT:
                assert task_config.sampling_params is not None, (
                    "Generation requires specifying sampling_params in your task config!"
                )
                generations: list[list[LMOutput]] = self.llm.generate(
                    requests, sampling_params=task_config.sampling_params
                )
            case RequestType.LOGPROBS | RequestType.PPL:
                generations: list[list[LMOutput]] = self.llm.logprobs(requests)

        # Collect Responses
        responses: list[Response] = []
        for inst, req, gen in zip(instances, requests, generations):
            responses += [Response(input=inst, request=req, output=gen)]

        # Extract answers
        responses: list[Response] = task.extract_answers(responses)

        # Score each output. E.g. exact match, logprobs per char
        for score in scorers:
            responses = score.score_responses(responses)

        # Compute metrics from sets of outputs. E.g. pass@k, maj@k, accuracy
        for metric in metrics:
            responses = metric.compute_metrics(responses)

        pprint(responses[0], max_string=10_000, max_length=32)

        # Average scores across all instances
        instance_metrics = [response.scores for response in responses]

        dataset_metrics = self.reduce_metrics(
            instance_metrics = instance_metrics, 
            primary_metric = task_config.primary_metric
        )

        console.print(f"\n[dim]─── results ({task_config.alias}) ───[/dim]")

        pprint(dataset_metrics, expand_all=True)

        # Save results with the Writer
        self.writer.save_responses(task_alias=task_config.alias, responses=responses)
        self.writer.save_metrics(task_alias=task_config.alias, metrics=dataset_metrics, num_instances=len(responses))

        return dataset_metrics

    def reduce_metrics(
        self, instance_metrics: dict, primary_metric: Optional[callable] = None
    ) -> dict:
        def average_dict(dicts: list[dict]) -> dict:
            """Recursively average entries in a list of dicts"""
            result = {}
            first = dicts[0]

            for key in first:
                if isinstance(first[key], dict):
                    values = [d[key] for d in dicts]
                    result[key] = average_dict(values)
                elif isinstance(first[key], (float, int, bool)):
                    values = [float(d[key]) for d in dicts]
                    result[key] = sum(values) / len(values)
                else:
                    raise TypeError(
                        f"Cannot perform reduce in datatype {type(first[key])} for key '{key}'"
                    )

            return result

        averaged_metrics = average_dict(instance_metrics)

        if primary_metric is not None:
            try:
                primary_score = primary_metric(averaged_metrics)
            except KeyError as _:
                raise KeyError("Could not find primary metric in score dict:", averaged_metrics)
        else:
            # ensure there is a primary_score key
            primary_score = None

        averaged_metrics["primary_score"] = primary_score

        return averaged_metrics

    def evaluate(self):
        all_metrics = {}
        for task_config in self.config.tasks:
            all_metrics[task_config.alias] = self.run(task_config)

        self.writer.write_finalized_metrics(self.config)

        return all_metrics


def run_eval(
    aliases: list[str],
    model_name: str,
    backend_type: BackendType,
    launch_type: LauncherType,
):
    from minieval.task_registry import TaskRegistry

    TaskRegistry()  # initialize the registry

    # Find the experiment ID, will be None if we are not on Beaker
    experiment_id = os.getenv("BEAKER_EXPERIMENT_ID", None)

    # Init task config from registry
    tasks = []
    for alias in aliases:
        task_config: TaskConfig = TaskRegistry.get_config(alias)
        tasks.append(task_config)

    # Init model config
    model = ModelConfig(name=model_name)

    # Init launcher config
    match launch_type:
        case LauncherType.LOCAL:
            launcher = LauncherConfig()
            writer = WriterConfig()
        case LauncherType.BEAKER:
            from minieval.launchers.beaker.launcher import (
                BeakerConfig,
                make_beaker_run_name,
            )

            launcher = BeakerConfig(
                description=f"{len(aliases)} task(s): {model_name} on {' '.join(aliases)}",
                task_name=f"minieval.{make_beaker_run_name(model_name)}",
            )

            writer = DatalakeConfig(
                run_name=model_name,
                experiment_id=experiment_id,
                task_idx=0,  # updated during execution
            )

    config = RunnerConfig(
        backend=backend_type, 
        model=model, 
        tasks=tasks, 
        launcher=launcher, 
        writer=writer
    )

    config: RunnerConfig = apply_overrides(config)

    pprint(config, expand_all=True)

    # If local, exit here with a gantry command
    # If on beaker, we will continue (@davidh might not be the best design if we want
    # to be able to launch jobs inside non-minieval beaker jobs)
    if launch_type == LauncherType.BEAKER and not experiment_id:
        from minieval.launchers.beaker.launcher import BeakerConfig, launch_gantry

        assert isinstance(config.launcher, BeakerConfig), (
            "Can only launch Gantry with a Beaker Config!"
        )

        if config.backend == BackendType.vllm and config.launcher.gpus == 0:
            raise RuntimeError(
                "vLLM requires GPUs and you requested 0 GPUs! Use '--launcher.gpus 1'"
            )

        launch_gantry(config.launcher)
        return

    # Launch evals!
    runner = EvalRunner(config)

    all_metrics = runner.evaluate()

    console.print("\n[dim]─── all results ───[/dim]")

    pprint(all_metrics, expand_all=True)


def main():
    import argparse

    # Redirect "minieval datalake" to datalake CLI
    if sys.argv[1:2] == ["datalake"]:
        import runpy

        runpy.run_module("minieval.writers.datalake.datalake_cli", run_name="__main__")
        return

    parser = argparse.ArgumentParser(description="""minieval - A simple eval library""")
    parser.add_argument("-t", "--tasks", action="append", help="Task aliases to evaluate")
    parser.add_argument("-m", "--model", default="mock", help="Model name/path to evaluate")
    parser.add_argument(
        "-b", "--backend", default="mock", help="Backend to use (mock, vllm, litellm)"
    )
    parser.add_argument("-l", "--launcher", default="local", help="Launcher to use (local, beaker)")
    parser.add_argument("--list", action="store_true", help="List available tasks")

    if len(sys.argv) == 1:
        lines = WELCOME_MSG[1:].split("\n")

        # don't print beyond terminal width
        term_width = os.get_terminal_size().columns
        truncated = "\n".join(line[:term_width] for line in lines)

        sys.stdout.write("\033[35m" + truncated + "\033[0m")
        parser.print_help()
        sys.exit(1)

    # Allow unknown arguments to be passed as overrides
    args, unknown = parser.parse_known_args()

    if args.list:
        from minieval.task_registry import TaskRegistry

        task_names = TaskRegistry.names()
        task_texts = [Text(name, style="cyan") for name in task_names]
        columns = Columns(task_texts, width=50, expand=True)

        console.print(columns)
        return

    if not args.tasks:
        parser.error("the following arguments are required: -t/--tasks")

    # Override defaults with CLI args
    aliases = args.tasks
    model_name = args.model
    backend = args.backend
    launch_type = args.launcher

    # Special alias "-t all". Runs all tasks in the Task Registry
    if "all" in aliases:
        from minieval.task_registry import TaskRegistry

        aliases = TaskRegistry.names()

    # Get launcher type
    try:
        launch_type = LauncherType(launch_type.lower())
    except ValueError:
        parser.error(
            f"invalid launcher type '{launch_type}'. Must be one of: {[t.value for t in LauncherType]}"
        )

    # Get backend type
    try:
        backend_type = BackendType(backend.lower())
    except ValueError:
        parser.error(
            f"invalid backend type '{backend}'. Must be one of: {[t.value for t in BackendType]}"
        )

    run_eval(aliases, model_name, backend_type, launch_type)


if __name__ == "__main__":
    main()
