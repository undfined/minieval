import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import requests
from rich import print
from rich.console import Console
from rich.pretty import pprint

from minieval.datatypes import Response, Writer, WriterConfig
from minieval.writers import WriterType

logger = logging.getLogger(__name__)

console = Console()


@dataclass
class DatalakeConfig(WriterConfig):
    run_name: Optional[str] = None
    save_path: str = "/results"  # must be "/results" to save to Beaker results
    dashboard: Optional[str] = None
    experiment_id: Optional[str] = None
    tags: Optional[str] = None
    base_url: str = "https://oe-eval-datalake.allen.ai"
    local_mode: bool = True  # True for local development, False for Beaker production
    task_idx: int = 0  # Task index for file naming

    writer_type: WriterType = WriterType.datalake

    def __post_init__(self):
        # Add tags for olmo cookbook
        tags = self.tags if self.tags is not None else ""
        if tags != "" and not tags.endswith(","):
            tags += ","
        if self.run_name is not None and f"checkpoint={self.run_name}" not in tags:
            tags += f"checkpoint={self.run_name}"
        if self.dashboard is not None and f"dashboard={self.dashboard}" not in tags:
            tags += f",dashboard={self.dashboard}"
        tags.replace(",,", ",")  # sanity check
        self.tags = tags


class DatalakeWriter(Writer):
    """Writes to the datalake using the oe-eval-internal format."""

    def __init__(self, config: DatalakeConfig):
        self.config = config

    def _generate_hash(self, config_dict: dict) -> str:
        """Generate MD5 hash of a config dictionary"""
        # Convert to JSON string with sorted keys for consistent hashing
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()

    def save_responses(self, task_alias: str, responses: list[Response]):
        os.makedirs(self.config.save_path, exist_ok=True)

        # Extract task name from alias (remove formatting suffix like :mc, :cot, etc.)
        task_name = task_alias.split(":")[0]

        # Format task index with zero padding
        task_idx_str = f"{self.config.task_idx:03d}"

        # Generate task and model hashes
        task_config = {
            "dataset_path": "unknown",
            "dataset_name": task_name,
            "native_id_field": "id",
            "primary_metric": "acc",
            "fewshot_source": "unknown",
            "random_subsample_seed": 1234,
            "split": "test",
            "num_shots": 5,
            "metadata": {"alias": task_alias},
            "generation_kwargs": None,
            "context_kwargs": None,
            "metric_kwargs": {},
            "fewshot_seed": 1234,
            "task_name": task_name,
            "version": 0,
            "task_core": task_name,
        }

        model_config = {
            "model": "mock",  # This should be populated from actual model config
            "revision": None,
            "trust_remote_code": None,
            "max_length": 2048,
            "model_path": None,
            "model_type": "mock",
            "metadata": {"alias": "mock"},
        }

        task_hash = self._generate_hash(task_config)
        model_hash = self._generate_hash(model_config)

        # Save predictions with correct naming: task-XXX-taskname-predictions.jsonl
        predictions_path = f"task-{task_idx_str}-{task_name}-predictions.jsonl"
        predictions_path = os.path.join(self.config.save_path, predictions_path)

        with open(predictions_path, "w") as f:
            for doc_id, response in enumerate(responses):
                # Extract metrics from response scores
                metrics = {}
                for metric_name, metric_data in response.scores.items():
                    if isinstance(metric_data, dict):
                        for sub_metric, value in metric_data.items():
                            metrics[f"{metric_name}_{sub_metric}"] = value
                    else:
                        metrics[metric_name] = metric_data

                # Find the correct choice based on gold completion or solution
                correct_choice = None
                if hasattr(response.input, "solution") and response.input.solution is not None:
                    correct_choice = response.input.solution
                elif (
                    hasattr(response.input, "choices")
                    and response.input.choices
                    and response.input.gold_completion
                ):
                    try:
                        correct_choice = response.input.choices.index(
                            response.input.gold_completion
                        )
                    except ValueError:
                        correct_choice = None

                # Build model output from LMOutput list
                model_output = []
                for output in response.output:
                    output_data = {}
                    if hasattr(output, "score") and output.score:
                        output_data.update(output.score)
                    if hasattr(output, "logprobs") and output.logprobs:
                        # Sum logprobs if available
                        sum_logits = sum(token.get("logprob", 0) for token in output.logprobs)
                        output_data["sum_logits"] = sum_logits
                        output_data["num_tokens"] = len(output.logprobs)
                    if hasattr(output, "text"):
                        output_data["num_chars"] = len(output.text)
                    model_output.append(output_data)

                prediction_data = {
                    "doc_id": doc_id,
                    "native_id": response.input.metadata.get("id", f"doc_{doc_id}")
                    if hasattr(response.input, "metadata")
                    else f"doc_{doc_id}",
                    "metrics": metrics,
                    "model_output": model_output,
                    "label": correct_choice,
                    "task_hash": task_hash,
                    "model_hash": model_hash,
                }
                f.write(json.dumps(prediction_data) + "\n")

        # Save requests with correct naming: task-XXX-taskname-requests.jsonl
        requests_path = f"task-{task_idx_str}-{task_name}-requests.jsonl"
        requests_path = os.path.join(self.config.save_path, requests_path)

        with open(requests_path, "w") as f:
            for doc_id, response in enumerate(responses):
                # Handle both single LMRequest and list of LMRequest
                requests = (
                    response.request if isinstance(response.request, list) else [response.request]
                )

                for idx, req in enumerate(requests):
                    # Build doc structure
                    doc_data = {
                        "id": response.input.metadata.get("id", f"doc_{doc_id}")
                        if hasattr(response.input, "metadata")
                        else f"doc_{doc_id}",
                        "query": response.input.question
                        if hasattr(response.input, "question")
                        else "",
                    }

                    if hasattr(response.input, "choices") and response.input.choices:
                        doc_data["choices"] = response.input.choices

                    if hasattr(response.input, "solution") and response.input.solution is not None:
                        doc_data["gold"] = response.input.solution
                    elif (
                        hasattr(response.input, "choices")
                        and response.input.choices
                        and response.input.gold_completion
                    ):
                        try:
                            doc_data["gold"] = response.input.choices.index(
                                response.input.gold_completion
                            )
                        except ValueError:
                            doc_data["gold"] = None

                    # Determine request type based on continuation
                    request_type = "loglikelihood"
                    if hasattr(req, "continuation") and req.continuation:
                        request_type = "loglikelihood"

                    # Build request content
                    request_content = {}
                    if hasattr(req, "context") and req.context:
                        request_content["context"] = req.context

                    if hasattr(req, "continuation") and req.continuation:
                        if isinstance(req.continuation, list):
                            # For multiple choice, create separate requests for each choice
                            for choice_idx, continuation in enumerate(req.continuation):
                                request_data = {
                                    "request_type": request_type,
                                    "doc": doc_data,
                                    "request": {
                                        "context": request_content.get("context", ""),
                                        "continuation": continuation,
                                    },
                                    "idx": choice_idx,
                                    "task_name": task_name,
                                    "doc_id": doc_id,
                                    "native_id": doc_data["id"],
                                    "label": doc_data.get("gold"),
                                }
                                f.write(json.dumps(request_data) + "\n")
                        else:
                            request_content["continuation"] = req.continuation
                            request_data = {
                                "request_type": request_type,
                                "doc": doc_data,
                                "request": request_content,
                                "idx": idx,
                                "task_name": task_name,
                                "doc_id": doc_id,
                                "native_id": doc_data["id"],
                                "label": doc_data.get("gold"),
                            }
                            f.write(json.dumps(request_data) + "\n")
                    else:
                        # Single request without continuation
                        request_data = {
                            "request_type": request_type,
                            "doc": doc_data,
                            "request": request_content,
                            "idx": idx,
                            "task_name": task_name,
                            "doc_id": doc_id,
                            "native_id": doc_data["id"],
                            "label": doc_data.get("gold"),
                        }
                        f.write(json.dumps(request_data) + "\n")

        # Save recorded inputs with correct naming: task-XXX-taskname-recorded-inputs.jsonl
        inputs_path = f"task-{task_idx_str}-{task_name}-recorded-inputs.jsonl"
        inputs_path = os.path.join(self.config.save_path, inputs_path)

        with open(inputs_path, "w") as f:
            # Save first 10 as sample
            for doc_id, response in enumerate(responses[:10]):
                # Build doc structure
                doc_data = {
                    "id": response.input.metadata.get("id", f"doc_{doc_id}")
                    if hasattr(response.input, "metadata")
                    else f"doc_{doc_id}",
                    "query": response.input.question if hasattr(response.input, "question") else "",
                }

                if hasattr(response.input, "choices") and response.input.choices:
                    doc_data["choices"] = response.input.choices

                if hasattr(response.input, "solution") and response.input.solution is not None:
                    doc_data["gold"] = response.input.solution
                elif (
                    hasattr(response.input, "choices")
                    and response.input.choices
                    and response.input.gold_completion
                ):
                    try:
                        doc_data["gold"] = response.input.choices.index(
                            response.input.gold_completion
                        )
                    except ValueError:
                        doc_data["gold"] = None

                # Build requests array
                requests_array = []
                requests = (
                    response.request if isinstance(response.request, list) else [response.request]
                )

                for req in requests:
                    if hasattr(req, "continuation") and req.continuation:
                        if isinstance(req.continuation, list):
                            # For multiple choice, create separate requests for each choice
                            for choice_idx, continuation in enumerate(req.continuation):
                                request_item = {
                                    "request_type": "loglikelihood",
                                    "request": {
                                        "context": req.context if hasattr(req, "context") else "",
                                        "continuation": continuation,
                                    },
                                    "idx": choice_idx,
                                }
                                requests_array.append(request_item)
                        else:
                            request_item = {
                                "request_type": "loglikelihood",
                                "request": {
                                    "context": req.context if hasattr(req, "context") else "",
                                    "continuation": req.continuation,
                                },
                                "idx": 0,
                            }
                            requests_array.append(request_item)

                input_data = {
                    "doc": doc_data,
                    "task_name": task_name,
                    "doc_id": doc_id,
                    "native_id": doc_data["id"],
                    "label": doc_data.get("gold"),
                    "requests": requests_array,
                }
                f.write(json.dumps(input_data) + "\n")

        print(f"Saved predictions to [bold purple]{predictions_path}[/bold purple]")
        print(f"Saved requests to [bold purple]{requests_path}[/bold purple]")
        print(f"Saved recorded inputs to [bold purple]{inputs_path}[/bold purple]")

    def save_metrics(self, task_alias: str, metrics: dict, *, num_instances: int = 0):
        os.makedirs(self.config.save_path, exist_ok=True)

        # Extract task name from alias (remove formatting suffix like :mc, :cot, etc.)
        task_name = task_alias.split(":")[0]

        # Format task index with zero padding
        task_idx_str = f"{self.config.task_idx:03d}"

        # Save metrics with correct naming: task-XXX-taskname-metrics.json
        metrics_path = f"task-{task_idx_str}-{task_name}-metrics.json"
        metrics_path = os.path.join(self.config.save_path, metrics_path)

        # Build model and task configs for hashing
        model_config = {
            "model": "mock",  # This should be populated from actual model config
            "revision": None,
            "trust_remote_code": None,
            "max_length": 2048,
            "model_path": None,
            "model_type": "mock",
            "metadata": {"alias": "mock"},
        }

        task_config = {
            "dataset_path": "unknown",
            "dataset_name": task_name,
            "native_id_field": "id",
            "primary_metric": "acc",
            "fewshot_source": "unknown",
            "random_subsample_seed": 1234,
            "split": "test",
            "num_shots": 5,
            "metadata": {"alias": task_alias},
            "generation_kwargs": None,
            "context_kwargs": None,
            "metric_kwargs": {},
            "fewshot_seed": 1234,
            "task_name": task_name,
            "version": 0,
            "task_core": task_name,
        }

        # Generate hashes
        task_hash = self._generate_hash(task_config)
        model_hash = self._generate_hash(model_config)

        # Define expected Beaker environment variables
        beaker_keys = [
            "BEAKER_EXPERIMENT_NAME",
            "BEAKER_WORKSPACE",
            "BEAKER_JOB_KIND",
            "BEAKER_NODE_ID",
            "GIT_REF",
            "BEAKER_JOB_ID",
            "BEAKER_WORKLOAD_ID",
            "BEAKER_EXPERIMENT_ID",
            "BEAKER_TASK_ID",
            "BEAKER_ASSIGNED_CPU_COUNT",
            "BEAKER_RESULT_DATASET_ID",
            "BEAKER_ASSIGNED_GPU_COUNT",
            "BEAKER_NODE_HOSTNAME",
        ]

        # Build beaker info from environment variables
        beaker_info = {}
        for key in beaker_keys:
            beaker_info[key] = os.environ.get(key, None)

        # Build the metrics structure to match legacy format
        metrics_data = {
            "task_name": task_name,
            "task_idx": self.config.task_idx,
            "task_hash": task_hash,
            "model_hash": model_hash,
            "model_config": model_config,
            "task_config": task_config,
            "compute_config": {
                "batch_size": "1",
                "max_batch_size": 32,
                "output_dir": self.config.save_path,
                "num_recorded_inputs": 10,
                "gsheet": None,
                "save_raw_requests": True,
                "check_datalake": False,
            },
            "processing_time": 3.0,  # This should be populated from actual processing time
            "current_date": "2024-01-01 00:00:00 UTC",  # This should be populated from actual date
            "num_instances": num_instances,
            "metrics": metrics,
            "beaker_info": beaker_info,
        }

        # Add primary_score to metrics if not present
        if "primary_score" not in metrics_data["metrics"]:
            # Try to find a suitable primary score
            primary_score = None
            for metric_name, metric_data in metrics.items():
                if isinstance(metric_data, dict):
                    for sub_metric, value in metric_data.items():
                        if "acc" in metric_name.lower() or "accuracy" in metric_name.lower():
                            primary_score = value
                            break
                else:
                    if "acc" in metric_name.lower() or "accuracy" in metric_name.lower():
                        primary_score = metric_data
                        break
                if primary_score is not None:
                    break

            if primary_score is not None:
                metrics_data["metrics"]["primary_score"] = primary_score

        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f, indent=2)

        print(f"Saved metrics to [bold purple]{metrics_path}[/bold purple]")

        # Increment task idx counter
        self.config.task_idx += 1

    def finalize_metrics(self, model_config: dict):
        """
        Generate consolidated metrics files for local runs

        @davidh -- This was generated by cursor to match oe-eval-internal format. Needs a refactor
        """
        os.makedirs(self.config.save_path, exist_ok=True)

        # Find all metrics files
        metrics_files = [
            f for f in os.listdir(self.config.save_path) if f.endswith("_metrics.json")
        ]
        if not metrics_files:
            print(
                f"[bold yellow]Warning: No metrics files found for consolidation in {self.config.save_path}[/bold yellow]"
            )
            return

        # Sort files by name for consistent ordering
        metrics_files.sort()

        # Generate metrics-all.jsonl
        jsonl_path = os.path.join(self.config.save_path, "metrics-all.jsonl")
        with open(jsonl_path, "w") as f:
            for metrics_file in metrics_files:
                metrics_file_path = os.path.join(self.config.save_path, metrics_file)
                try:
                    with open(metrics_file_path, "r") as mf:
                        metrics_data = json.load(mf)
                        # Add some metadata for consistency
                        task_alias = metrics_file.replace("_metrics.json", "")
                        enhanced_data = {
                            "task_name": task_alias.split(":")[0],
                            "task_alias": task_alias,
                            "metrics": metrics_data,
                            "model_config": model_config,
                        }
                        f.write(json.dumps(enhanced_data) + "\n")
                except Exception as e:
                    print(f"[bold red]Error reading metrics file {metrics_file}: {e}[/bold red]")

        # Generate metrics.json
        all_primary_scores = []
        tasks = []

        for metrics_file in metrics_files:
            metrics_file_path = os.path.join(self.config.save_path, metrics_file)
            try:
                with open(metrics_file_path, "r") as mf:
                    metrics_data = json.load(mf)
                    task_alias = metrics_file.replace("_metrics.json", "")

                    # Get primary score
                    primary_score = metrics_data.get("primary_score", None)
                    if primary_score is not None:
                        all_primary_scores.append(f"{task_alias}: {primary_score}")

                    # Build task entry
                    task_entry = {
                        "alias": task_alias,
                        "metrics": metrics_data,
                        "num_instances": 0,  # Not available in local mode
                        "processing_time": 0.0,  # Not available in local mode
                        "task_config": {"task_name": task_alias.split(":")[0]},
                    }

                    tasks.append(task_entry)

            except Exception as e:
                print(f"[bold red]Error reading metrics file {metrics_file}: {e}[/bold red]")

        # Build the consolidated metrics structure
        consolidated_metrics = {
            "all_primary_scores": all_primary_scores,
            "tasks": tasks,
            "model_config": model_config,
        }

        json_path = os.path.join(self.config.save_path, "metrics.json")
        with open(json_path, "w") as f:
            json.dump(consolidated_metrics, f, indent=2)

        print(f"Saved consolidated metrics to [bold purple]{jsonl_path}[/bold purple]")
        print(f"Saved consolidated metrics summary to [bold purple]{json_path}[/bold purple]")

    def write_finalized_metrics(self, config):
        """@davidh -- also cursor"""
        # Generate consolidated metrics after all tasks are completed
        try:
            # Build model config for consolidated metrics
            model_config = {
                "model": config.model.name,
                "revision": getattr(config.model, "revision", None),
                "trust_remote_code": getattr(config.model, "trust_remote_code", None),
                "max_length": getattr(config.model, "max_length", 2048),
                "model_path": getattr(config.model, "model_path", None),
                "model_type": config.backend.value
                if hasattr(config.backend, "value")
                else str(config.backend),
                "metadata": {"tasks": [task.alias for task in config.tasks]},
            }

            console.print("\n[dim]─── generating consolidated metrics ───[/dim]")
            self.finalize_metrics(model_config)
            console.print("[green]✓ Generated consolidated metrics files[/green]")

        except Exception as e:
            console.print(
                f"[bold yellow]Warning: Failed to generate consolidated metrics:[/bold yellow] {e}"
            )

        self.push_finalized_metrics()

    def push_finalized_metrics(self):
        experiment_id = os.getenv("BEAKER_EXPERIMENT_ID", None)

        # Auto-push to datalake if running in Beaker environment
        if experiment_id:
            console.print("\n[dim]─── pushing to datalake ───[/dim]")
            try:
                result = self.push_to_datalake(experiment_id, self.config.tags)
                console.print("[bold green]Successfully pushed to datalake:[/bold green]")
                pprint(result)
                console.print("[bold blue]View your results at:[/bold blue]")
                console.print(
                    f"[bold blue]>>> minieval datalake metrics {experiment_id}[/bold blue]"
                )
                console.print(
                    f"[bold blue]>>> minieval datalake download {experiment_id}[/bold blue]"
                )
                console.print(
                    f"[bold blue]>>> minieval datalake inspect {experiment_id}[/bold blue]"
                )
            except Exception as e:
                # We want the evaluation results even if push fails
                console.print(f"[bold red]Failed to push to datalake:[/bold red] {e}")

    def save_consolidated_metrics_jsonl(self, output_filename: str = "metrics-all.jsonl"):
        """Save all task metrics to a single JSONL file"""
        os.makedirs(self.config.save_path, exist_ok=True)

        # Find all metrics files
        metrics_files = [
            f for f in os.listdir(self.config.save_path) if f.endswith("-metrics.json")
        ]
        if not metrics_files:
            logger.warning("No metrics files found for consolidation")
            return

        # Sort by task index (extracted from filename)
        def extract_task_idx(filename):
            try:
                # Extract task index from filename like "task-000-arc_easy-metrics.json"
                parts = filename.split("-")
                if len(parts) >= 2 and parts[0] == "task":
                    return int(parts[1])
                return 0
            except Exception as _:
                return 0

        metrics_files.sort(key=extract_task_idx)

        consolidated_path = os.path.join(self.config.save_path, output_filename)

        with open(consolidated_path, "w") as f:
            for metrics_file in metrics_files:
                metrics_file_path = os.path.join(self.config.save_path, metrics_file)
                try:
                    with open(metrics_file_path, "r") as mf:
                        metrics_data = json.load(mf)
                        f.write(json.dumps(metrics_data) + "\n")
                except Exception as e:
                    logger.error(f"Error reading metrics file {metrics_file}: {e}")

        print(f"Saved consolidated metrics to [bold purple]{consolidated_path}[/bold purple]")

    def save_consolidated_metrics_json(
        self, model_config: dict, output_filename: str = "metrics.json"
    ):
        """Save consolidated metrics summary in the specified format"""
        os.makedirs(self.config.save_path, exist_ok=True)

        # Find all metrics files
        metrics_files = [
            f for f in os.listdir(self.config.save_path) if f.endswith("-metrics.json")
        ]
        if not metrics_files:
            logger.warning("No metrics files found for consolidation")
            return

        # Sort by task index (extracted from filename)
        def extract_task_idx(filename):
            try:
                # Extract task index from filename like "task-000-arc_easy-metrics.json"
                parts = filename.split("-")
                if len(parts) >= 2 and parts[0] == "task":
                    return int(parts[1])
                return 0
            except Exception as _:
                return 0

        metrics_files.sort(key=extract_task_idx)

        all_primary_scores = []
        tasks = []

        for metrics_file in metrics_files:
            metrics_file_path = os.path.join(self.config.save_path, metrics_file)
            try:
                with open(metrics_file_path, "r") as mf:
                    metrics_data = json.load(mf)

                    # Extract task alias from metadata or task_config
                    task_alias = "unknown"
                    if "task_config" in metrics_data and "metadata" in metrics_data["task_config"]:
                        task_alias = metrics_data["task_config"]["metadata"].get("alias", "unknown")
                    elif "task_name" in metrics_data:
                        task_alias = metrics_data["task_name"]

                    # Get primary score
                    primary_score = None
                    if "metrics" in metrics_data and "primary_score" in metrics_data["metrics"]:
                        primary_score = metrics_data["metrics"]["primary_score"]

                    # Add to all_primary_scores list
                    if primary_score is not None:
                        all_primary_scores.append(f"{task_alias}: {primary_score}")

                    # Build task entry
                    task_entry = {
                        "alias": task_alias,
                        "metrics": metrics_data.get("metrics", {}),
                        "num_instances": metrics_data.get("num_instances", 0),
                        "processing_time": metrics_data.get("processing_time", 0.0),
                        "task_config": metrics_data.get("task_config", {}),
                    }

                    tasks.append(task_entry)

            except Exception as e:
                logger.error(f"Error reading metrics file {metrics_file}: {e}")

        # Build the consolidated metrics structure
        consolidated_metrics = {
            "all_primary_scores": all_primary_scores,
            "tasks": tasks,
            "model_config": model_config,
        }

        consolidated_path = os.path.join(self.config.save_path, output_filename)

        with open(consolidated_path, "w") as f:
            json.dump(consolidated_metrics, f, indent=2)

        print(
            f"Saved consolidated metrics summary to [bold purple]{consolidated_path}[/bold purple]"
        )

    def finalize_metrics(self, model_config: dict):
        """Convenience method to save both consolidated metrics files"""
        self.save_consolidated_metrics_jsonl()
        self.save_consolidated_metrics_json(model_config)

    def push_to_datalake(self, experiment_id: str, tags: Optional[str] = None):
        """Push results to the datalake using the datalake_push_trigger pattern"""
        if not experiment_id:
            raise ValueError("experiment_id is required for pushing to datalake")

        # Find the most recently modified metrics file as the last_output_file
        metrics_files = [
            f for f in os.listdir(self.config.save_path) if f.endswith("-metrics.json")
        ]
        if not metrics_files:
            raise ValueError("No metrics files found to push")

        # Sort by modification time (most recent first)
        metrics_files_with_time = []
        for f in metrics_files:
            file_path = os.path.join(self.config.save_path, f)
            if os.path.exists(file_path):
                mtime = os.path.getmtime(file_path)
                metrics_files_with_time.append((f, mtime))

        if not metrics_files_with_time:
            raise ValueError("No valid metrics files found to push")

        # Sort by modification time (most recent first) and get the most recent
        metrics_files_with_time.sort(key=lambda x: x[1], reverse=True)
        last_metrics_file = metrics_files_with_time[0][0]

        # Use datalake_push_trigger pattern
        return self._datalake_push_trigger(last_metrics_file, experiment_id, tags)

    def _datalake_push_trigger(
        self, beaker_metrics_file: str, experiment_id: str, tags: Optional[str] = None
    ):
        """
        This function will check for the existence of the last_output_file from within the experiment, then
        request output upload to the datalake and exit. The datalake API will recheck on the server side before
        upload to ensure that the last_output_file is made available through the Beaker API.
        """
        for retry in range(30):
            metrics_file_path = os.path.join(self.config.save_path, beaker_metrics_file)
            if os.path.isfile(metrics_file_path):
                logger.info("Results datasets found. Starting upload trigger.")

                params = {
                    "last_output_file": beaker_metrics_file,
                }
                if tags:
                    params["tags"] = tags

                url = f"{self.config.base_url}/greenlake/upload/{experiment_id}"

                try:
                    response = requests.post(url=url, params=params)
                    if response.status_code == 202:
                        logger.info(f"Upload triggered successfully: {response.json()}")
                        return response.json()
                    else:
                        response.raise_for_status()
                except requests.exceptions.HTTPError as e:
                    try:
                        error_detail = e.response.json()
                        error_msg = error_detail.get("detail", str(e))
                    except Exception as _:
                        error_msg = str(e)
                    raise Exception(f"Datalake upload failed: {error_msg}")
                except requests.exceptions.RequestException as e:
                    raise Exception(f"Network error connecting to datalake: {e}")

                return
            else:
                logger.warning(
                    f"File {metrics_file_path} has not been created yet. Re-checking in 10s..."
                )
                time.sleep(10)

        logger.error("Failed to find results datasets after 5 minutes. Exiting.")
        raise Exception("Failed to find results datasets after 5 minutes")

    def _push_local_mode(self, experiment_id: str, tags: Optional[str] = None):
        raise NotImplementedError(
            "Datalake does not support directly pushing results from a local eval job."
        )

    def _push_production_mode(self, experiment_id: str, tags: Optional[str] = None):
        """Push in production mode - uses last_output_file approach for Beaker"""
        logger.info(f"Production mode: Pushing experiment {experiment_id} to datalake")

        # Find the most recently modified metrics file
        metrics_files = [
            f for f in os.listdir(self.config.save_path) if f.endswith("_metrics.json")
        ]
        if not metrics_files:
            raise ValueError("No metrics files found to push")

        # Sort by modification time (most recent first)
        metrics_files_with_time = []
        for f in metrics_files:
            file_path = os.path.join(self.config.save_path, f)
            if os.path.exists(file_path):
                mtime = os.path.getmtime(file_path)
                metrics_files_with_time.append((f, mtime))

        if not metrics_files_with_time:
            raise ValueError("No valid metrics files found to push")

        # Sort by modification time (most recent first) and get the most recent
        metrics_files_with_time.sort(key=lambda x: x[1], reverse=True)
        last_metrics_file = metrics_files_with_time[0][0]

        params = {
            "last_output_file": last_metrics_file,
        }
        if tags:
            params["tags"] = tags

        url = f"{self.config.base_url}/greenlake/upload/{experiment_id}"

        for retry in range(30):
            metrics_file_path = os.path.join(self.config.save_path, last_metrics_file)
            if os.path.isfile(metrics_file_path):
                logger.info("Results datasets found. Starting upload trigger.")

                try:
                    response = requests.post(url=url, params=params)
                    if response.status_code == 202:
                        logger.info(f"Upload triggered successfully: {response.json()}")
                        return response.json()
                    else:
                        response.raise_for_status()
                except requests.exceptions.HTTPError as e:
                    try:
                        error_detail = e.response.json()
                        error_msg = error_detail.get("detail", str(e))
                    except Exception as _:
                        error_msg = str(e)
                    raise Exception(f"Datalake upload failed: {error_msg}")
                except requests.exceptions.RequestException as e:
                    raise Exception(f"Network error connecting to datalake: {e}")
            else:
                logger.warning(f"File {metrics_file_path} not found. Re-checking in 10s...")
                time.sleep(10)

        logger.error("Failed to find results datasets after 5 minutes. Exiting.")
        raise Exception("Failed to find results datasets after 5 minutes")

    def _push_with_kickoff_transform(self, experiment_id: str, tags: Optional[str] = None):
        """Push using the kickoff_transform approach"""
        params = {"kickoff_transform": True}
        if tags:
            params["tags"] = tags

        url = f"{self.config.base_url}/greenlake/upload/{experiment_id}"

        logger.info(f"Pushing experiment {experiment_id} results to the datalake")

        try:
            response = requests.post(url=url, params=params)
            if response.status_code == 202:
                result = response.json()
                logger.info(f"Upload triggered successfully: {result}")

                # Check if any files were actually found
                if self._check_experiment_has_files(experiment_id):
                    return result
                else:
                    logger.warning(
                        f"Push succeeded but no files found in experiment {experiment_id}"
                    )
                    logger.warning(
                        "This is expected for local development - files are not uploaded to Beaker"
                    )
                    return {
                        **result,
                        "warning": "No files found in Beaker experiment (expected for local development)",
                    }
            else:
                # Try to get the error details from the response
                try:
                    error_detail = response.json()
                    error_msg = error_detail.get("detail", f"HTTP {response.status_code}")
                except Exception as _:
                    error_msg = f"HTTP {response.status_code}: {response.text}"

                logger.error(f"Failed to push to datalake: {error_msg}")
                response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            # Re-raise with more context
            try:
                error_detail = e.response.json()
                error_msg = error_detail.get("detail", str(e))
            except Exception as _:
                error_msg = str(e)
            raise Exception(f"Datalake upload failed: {error_msg}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Network error connecting to datalake: {e}")

    def _check_experiment_has_files(self, experiment_id: str) -> bool:
        """Check if the experiment has any files in the datalake"""
        try:
            url = f"{self.config.base_url}/greenlake/inspect/{experiment_id}"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                # Check if any file type has count > 0
                file_counts = [
                    data.get("ALL_METRICS_files", 0),
                    data.get("METRICS_files", 0),
                    data.get("PREDICTIONS_files", 0),
                    data.get("REQUESTS_files", 0),
                    data.get("INPUTS_files", 0),
                    data.get("METADATA_files", 0),
                ]
                return any(count > 0 for count in file_counts)
        except Exception as _:
            pass
        return False
