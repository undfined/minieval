from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, ClassVar, Dict, List, Optional, Type, TypeVar, Union

from minieval.writers import WriterType

T = TypeVar("T", bound=Type["TaskConfig"])


@dataclass
class Instance:
    """A single unit of work"""

    question: str = ""
    gold_completion: Optional[str] = None
    choices: Optional[list[str]] = None
    solution: Optional[str|int] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class LMRequest:
    messages: list[dict]
    context: Optional[str] = None
    continuation: Optional[list[str]] = None


@dataclass
class LMOutput:
    text: str
    logprobs: Optional[List[Dict[str, Any]]] = None
    extracted_answer: Optional[list|dict] = None
    score: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass
class Response:
    input: Instance
    request: LMRequest
    output: list[LMOutput]
    scores: dict = field(default_factory=dict)


class RequestType(str, Enum):
    CHAT = "chat"
    GENERATE = "generate"
    LOGPROBS = "logprobs"
    PPL = "perplexity"


class LauncherType(str, Enum):
    LOCAL = "local"
    BEAKER = "beaker"


@dataclass
class Formatter:
    REQUEST_TYPE: RequestType
    template: Optional[str]
    few_shot_n: Optional[int] = 0
    few_shot_alias: Optional[str] = None
    few_shot: Optional[list[Instance]] = None

    def build_messages(self, instances: list[Instance]) -> list[LMRequest]:
        return list(map(self._build_message, instances))

    def build_requests(self, template_func, messages: list[LMRequest]):
        return list(map(lambda msg: self._build_request(template_func, msg), messages))

    def _build_request(self, template_func: callable, request: LMRequest) -> LMRequest:
        context = template_func(request.messages, tokenize=False)
        request.context = context
        return request

    def _build_message(self, instance: Instance) -> LMRequest:
        raise NotImplementedError()


@dataclass
class Scorer:
    name: str
    timeout: int = 10

    def score_responses(self, responses: list[Response]) -> list[Response]:
        """Call scorer with multi-threading and timeout"""
        results: list[Optional[Response]] = [None] * len(responses)
        with ThreadPoolExecutor() as executor:
            future_to_idx = {
                executor.submit(self._score_response, response): idx
                for idx, response in enumerate(responses)
            }
            for future in as_completed(future_to_idx, timeout=self.timeout * len(responses)):
                idx = future_to_idx[future]
                try:
                    result: Response = future.result(timeout=self.timeout)
                    results[idx] = result
                except TimeoutError:
                    raise TimeoutError(
                        f"Scorer failed to complete within timeout of {self.timeout} sec."
                    )

        return results

    def _score_response(self, response: Response) -> Response:
        input: Instance = response.input
        outputs: list[LMOutput] = response.output

        for output in outputs:
            score = self._score_response_single(input, output)
            output.score[self.name] = score

        return response

    def _score_response_single(self, input: Instance, output: LMOutput) -> float:
        raise NotImplementedError()


@dataclass
class Metric:
    """Collates scores into metrics. E.g., accuracy, pass@k"""

    name: str

    def compute_metrics(self, responses: list[Response]) -> list[Response]:
        return list(map(self._compute_metric, responses))

    def _compute_metric(self, response: Response) -> Response:
        response.scores[self.name] = {}
        scorers = response.output[0].score.keys()  # infer scorers from first output

        for scorer in scorers:
            input: Instance = response.input
            outputs: list[LMOutput] = response.output

            response.scores[self.name][scorer] = self._compute_metric_single(input, outputs, scorer)

        return response

    def _compute_metric_single(self, input: Instance, outputs: list[LMOutput], scorer: str) -> Response:
        raise NotImplementedError()


@dataclass
class SamplingParams:
    repeats: Optional[int] = 1
    max_gen_toks: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    logprobs: Optional[int] = None


@dataclass
class LauncherConfig:
    pass


@dataclass
class WriterConfig:
    writer_type: WriterType = WriterType.local
    save_path: str = "/tmp/minieval"


class Writer:
    def save_responses(self, task_alias: str, responses: list[Response]):
        raise NotImplementedError()

    def save_metrics(self, task_alias: str, metrics: dict, *, num_instances: int = 0):
        raise NotImplementedError()

    def write_finalized_metrics(self, config):
        # For additional utilities, like pushing to a datalake
        return


@dataclass
class ModelConfig:
    name: str
    revision: Optional[str] = None


@dataclass
class TaskConfig:
    alias: str
    formatter: Formatter
    scorer: list[Scorer]
    metric: list[Metric]
    subset: Optional[str] = None
    primary_metric: Optional[Any] = None
    sampling_params: Optional[SamplingParams] = None
    limit: Optional[int] = None
    seed: int = 0


class Task:
    config: TaskConfig
    hf_path: str
    subsets: Optional[list[str|int]] = None

    def __init__(self, config: TaskConfig):
        raise NotImplementedError()

    @property
    def requests(self):
        raise NotImplementedError()

    def _process_instance(self, doc: dict) -> Instance:
        raise NotImplementedError()

    def extract_answers(self, responses: list[Response]) -> list[Response]:
        for resp in responses:
            generations: list[LMOutput] = resp.output
            for gen in generations:
                gen.extracted_answer = self.extract_answer(resp, gen)
        return responses

    @classmethod
    def extract_answer(cls, response: Response, generation: LMOutput) -> Any:
        return generation.text

    def _construct_few_shot(self) -> list[dict]:
        raise NotImplementedError(
            "Currently, we require a few shot examples to be specified manually. We do not want to sample few-shot examples from the test set."
        )

    def build_few_shot(self) -> list[Instance]:
        formatter: Formatter = self.config.formatter

        if formatter.few_shot_n is None or formatter.few_shot_n == 0:
            return []

        # If no alias in the FewShotRegistry is provided, fall back to the
        # task's few shot sampler.
        if formatter.few_shot_alias is None:
            few_shot_ds = self._construct_few_shot()
        else:
            from minieval.few_shot import FewShotRegistry

            few_shot_ds: list[dict] = FewShotRegistry.init(formatter.few_shot_alias)

        few_shot_instances = list(map(self._process_instance, few_shot_ds))

        few_shot_instances = few_shot_instances[: formatter.few_shot_n]

        # update formatter with few shot examples
        formatter.few_shot = few_shot_instances

        return few_shot_instances


class TaskRegistry:
    """Registry for task aliases."""

    _instance: ClassVar[Union["TaskRegistry", None]] = None
    _named_tasks: ClassVar[dict[str, Type["TaskConfig"]]] = {}
    _task_mapping: ClassVar[dict[str, Type["Task"]]] = {}

    def __new__(cls, *args, **kwargs):
        # singleton pattern
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def register(cls, task_alias: str, task: Task) -> Callable[[T], T]:
        # instantiate the singleton instance here; it won't get instantiated
        # twice cuz it's a singleton, after all.
        instance = cls()

        def decorator(
            task_config: T,
            _task_alias: str = task_alias,
            _class: Task = task,
            _instance: "TaskRegistry" = instance,
        ) -> T:
            # add to the registry
            _instance._named_tasks[_task_alias] = task_config

            _instance._task_mapping[task_alias] = _class

            # a little bit of a Python crime, but when registering a group,
            # we replace the class name with the task name for the `.name` property.
            task_config.alias = _task_alias  # pyright: ignore
            return task_config

        return decorator

    @classmethod
    def names(cls) -> list[str]:
        return list(cls._named_tasks.keys())

    @classmethod
    def exists(cls, task_alias: str) -> bool:
        return any(cls.search(task_alias))

    @classmethod
    def get_config(cls, task_alias: str) -> TaskConfig:
        assert cls._instance is not None, "TaskRegistry is not initialized"

        if task_alias not in cls._named_tasks:
            raise ValueError(
                f"Task {task_alias} not found in the Task Registry! View available tasks with 'minieval --list'"
            )

        task_config_class = cls._named_tasks[task_alias]

        kwargs = {"alias": task_alias}

        # Get default values from class attributes if they exist
        for attr_key, attr_val in task_config_class.__dict__.items():
            # Don't include class attributes (like __module__)
            if attr_key.startswith("__"):
                continue

            kwargs.update({attr_key: attr_val})

        config = task_config_class(**kwargs)
        return config

    @classmethod
    def get_task(cls, task_alias: str) -> Type["Task"]:
        if task_alias not in cls._task_mapping:
            raise ValueError(f"Task class for {task_alias} not found in the Task Registry!")

        return cls._task_mapping[task_alias]
