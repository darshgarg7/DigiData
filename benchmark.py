# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse
import asyncio
import json
import os

from collections import defaultdict

from auto_evaluator.auto_evaluator import AutoEvaluator
from env.emulator import LocalEmulatorPool
from env.remote_emulator import RemoteEmulator
from predictor.llama4_predictor import Llama4Predictor
from predictor.openai_gpt4o_predictor import OpenAIGPT4OPredictor
from task.task import Task
from tqdm import tqdm


def analyze_results(results):
    print(f"Total number of trajectories: {len(results)}")
    successes = [result for result in results if result[0][1] == 1]
    print(f"Number of successful trajectories: {len(successes)}")
    print(f"Success rate: {len(successes) / len(results) * 100}%")


def prepare_task_for_eval(task, emulator):
    metadata_dict = {
        "goal": {},
        "screen_width_height": {},
        "step_dict": defaultdict(dict),
    }

    for i in range(len(task.screenshots)):
        screenshot = task.screenshots[i]
        xml = task.xmls[i]
        step_id = i
        goal = task.goal
        trajectory_id = task.session_id
        screen_size = emulator.get_screen_size()
        screen_size = [screen_size["width"], screen_size["height"]]
        action = str(task.actions[i])

        metadata_dict["goal"][trajectory_id] = goal
        metadata_dict["screen_width_height"][trajectory_id] = screen_size
        metadata_dict["step_dict"][trajectory_id][step_id] = {
            "raw_action": action,
            "raw_screenshot": screenshot,
            "raw_xml": xml,
        }
    return metadata_dict


async def process_trajectories(evaluator, metadata_dict, log=False):
    all_trajectory_ids = list(metadata_dict["step_dict"].keys())

    successes = []
    for trajectory_id in tqdm(all_trajectory_ids):
        cur_trajectory = metadata_dict["step_dict"][trajectory_id]
        screen_width_height = metadata_dict["screen_width_height"][trajectory_id]
        goal = metadata_dict["goal"][trajectory_id]

        cur_trajectory = await evaluator.preprocess(
            trajectory_id,
            cur_trajectory,
            screen_width_height,
        )

        step_progress_list, trajectory_progress = await evaluator.evaluate(
            goal=goal,
            trajectory_id=trajectory_id,
            trajectory=cur_trajectory,
            total_steps=len(cur_trajectory),
        )

        if log:
            for step_progress in step_progress_list:
                print(f"Step {step_progress['step_id']}")
                if "summary" in step_progress:
                    print(f"Summary: {step_progress['summary']}")
                if "reason" in step_progress:
                    print(f"Reason: {step_progress['reason']}")
                if "progress_reward" in step_progress:
                    print(f"Step progress: {step_progress['progress_reward']}\n")

            if trajectory_progress:
                if "progress_reward" in trajectory_progress:
                    print(
                        f"Trajectory progress: {trajectory_progress['progress_reward']}"
                    )
                if "reason" in trajectory_progress:
                    print(f"Reason: {trajectory_progress['reason']}")
                if "summary" in trajectory_progress:
                    print(f"Summary: {trajectory_progress['summary']}")
        successes.append(
            (
                trajectory_id,
                trajectory_progress["progress_reward"],
                trajectory_progress["reason"],
            )
        )
    return successes


def load_config(config_filepath):
    """
    Load config from JSON file into a dictionary.

    Args:
        config_filepath (str): Path to the config file.

    Returns:
        dict: Config dictionary.
    """
    try:
        with open(config_filepath, "r") as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_filepath}")
        exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in config file at {config_filepath}: {e}")
        exit(1)


def load_tasks_from_registry(task_registry_filepath):
    """
    Load tasks from a registry file.

    Args:
        task_registry_filepath (str): Path to the task registry file.

    Returns:
        list: List of task dictionaries.
    """
    try:
        with open(task_registry_filepath, "r") as f:
            task_dicts = [json.loads(line) for line in f.readlines()]
            tasks = [Task.from_dict(task_dict) for task_dict in task_dicts]
            return tasks
    except FileNotFoundError:
        print(f"Error: Task registry file not found at {task_registry_filepath}")
        exit(1)


async def run_task(task, pool, predictor):
    emulator = await pool.acquire()
    try:
        print(f"\nProcessing task: {str(task)}")
        task.do_prework(emulator)
        while not task.is_complete():
            await task.execute(emulator, predictor)
            await asyncio.sleep(3)
    except Exception as e:
        print(f"Error: Failed to execute task {str(task)}: {e}")
    finally:
        pool.release(emulator)


async def driver(config):
    # Initialize emulators
    print(f"=" * 50)
    print("Initializing and connecting to emulators")
    emulator_config = config.get("env", {}).get("emulator", None)
    emulators = []
    if emulator_config is None:
        pool = LocalEmulatorPool(device_name="DigiData")
    elif emulator_config.get("emulator_type", None) == "remote":
        ip = emulator_config.get("ip", None)
        username = emulator_config.get("username", None)
        password = emulator_config.get("password", None)
        emulator = RemoteEmulator(ip, username, password)
        emulator.connect()
        emulators.append(emulator)
        pool = None  # For remote, tasks are executed sequentially on the single remote emulator.
    else:
        raise Exception("Unknown emulator type")

    # Initialize the predictor
    predictor_config = config["predictor_config"]
    temperature = predictor_config["temperature"]
    top_p = predictor_config["top_p"]
    max_new_tokens = predictor_config["max_new_tokens"]
    model_id = predictor_config["model_id"]
    use_screenshot = predictor_config["use_screenshot"]
    api_key_name = predictor_config.get("api_key_name", "")
    base_url = predictor_config.get("base_url", "")
    Predictor = OpenAIGPT4OPredictor if "gpt" in model_id.lower() else Llama4Predictor
    predictor = Predictor(
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
        model_id=model_id,
        use_screenshot=use_screenshot,
        api_key_name=api_key_name,
        base_url=base_url,
    )
    print(f"=" * 50)
    print(
        f"Initalized LLM predictor with model_id={model_id}, temperature={temperature}, top_p={top_p}, max_new_tokens={max_new_tokens}, use_screenshot={use_screenshot}"
    )

    # Initilaize the tasks
    task_registry_filepath = config["task_registry_filepath"]
    task_registry_filepath = os.path.join(
        os.path.dirname(__file__), task_registry_filepath
    )
    tasks = load_tasks_from_registry(task_registry_filepath)
    print(f"=" * 50)
    print(f"Loaded {len(tasks)} tasks from registry {task_registry_filepath}")

    # Execute all tasks in parallel using the emulator pool if available
    print(f"=" * 50)
    print(f"Starting parallel execution of {len(tasks)} tasks")
    task_coroutines = []
    if pool is not None:
        for task in tasks:
            task_coroutines.append(run_task(task, pool, predictor))
        await asyncio.gather(*task_coroutines)
    else:

        async def acquire_emulator():
            return emulators[0]

        pool = type(
            "DummyPool",
            (),
            {
                "acquire": lambda self: acquire_emulator(),
                "release": lambda self, x: None,
            },
        )()
        for task in tasks:
            await run_task(task, pool, predictor)

    evaluator_config = config["evaluator_config"]
    model_id = evaluator_config["model_id"]
    model_host_type = evaluator_config["model_host_type"]
    use_screenshot = evaluator_config["use_screenshot"]
    use_xml = evaluator_config["use_xml"]
    api_key_name = predictor_config.get("api_key_name", "")
    base_url = predictor_config.get("base_url", "")
    evaluator = AutoEvaluator(
        model_id=model_id,
        model_host_type=model_host_type,
        use_screenshot=use_screenshot,
        use_xml=use_xml,
        api_key_name=api_key_name,
        base_url=base_url,
    )
    print(f"=" * 50)
    print(
        f"LLM Judge initialized with model_id={model_id}, model_host_type={model_host_type}, use_screenshot={use_screenshot}, use_xml={use_xml}"
    )

    # Evaluate task trajectories in parallel
    print(f"=" * 50)
    print(f"Starting evaluation of {len(tasks)} tasks in parallel")

    async def eval_task(task, idx):
        print(f"\nProcessing task {idx+1} of {len(tasks)}: {str(task)}")
        if pool is not None:
            emulator_eval = await pool.acquire()
            metadata_dict = prepare_task_for_eval(task, emulator_eval)
            pool.release(emulator_eval)
        else:
            metadata_dict = prepare_task_for_eval(task, emulators[0])
        result = await process_trajectories(evaluator, metadata_dict)
        success = True if result[0][1] == 1 else False
        print(
            f"LLM Judge thinks task {str(task)} is {'successful' if success else 'failed'} because {result[0][2]}"
        )
        return result

    evaluation_results = await asyncio.gather(
        *[eval_task(task, idx) for idx, task in enumerate(tasks)]
    )

    print(f"=" * 50)
    print(f"Analysis of {len(evaluation_results)} tasks")
    analyze_results(evaluation_results)


async def main():
    parser = argparse.ArgumentParser(description="Launch code with config")
    parser.add_argument("--config_filepath", help="Path to the config file")
    args = parser.parse_args()

    config_filepath = args.config_filepath
    config_filepath = os.path.join(os.path.dirname(__file__), config_filepath)
    if not os.path.exists(config_filepath):
        print(f"Error: Config file not found at {config_filepath}")
        exit(1)

    config = load_config(config_filepath)
    await driver(config)


if __name__ == "__main__":
    asyncio.run(main())
