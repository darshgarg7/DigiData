# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
from logging import getLogger


logger = getLogger("auto_evaluator_helper")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def get_traj_ids(trajectories, chunk_idx, chunk_size):
    trajectory_ids = sorted(list(trajectories.keys()))
    if chunk_size != -1:
        trajectory_ids = trajectory_ids[
            chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size
        ]
        logger.info(
            f"Processing from {chunk_idx * chunk_size} to {(chunk_idx + 1) * chunk_size}"
        )
    return trajectory_ids


def process_local_filesystem_annotation(
    annotation, trajectories, action_cnt, screen_width_height, goals, app_names
):
    """adds to trajectories, action_cnt, goals"""
    # parse xml
    for entry in annotation["xml"].keys():
        entry_name_split = entry.split("_")
        traj_id = "_".join(entry_name_split[:-1])
        step_idx = int(entry_name_split[-1])

        xml_file = open(annotation["xml"][entry][0], "rb")
        raw_xml_content = b"".join(xml_file.readlines())

        trajectories[traj_id][step_idx] = {
            "raw_action": annotation["actions"][step_idx][0],
            "raw_xml": raw_xml_content,
            "raw_screenshot": None,
        }

        # count action frequency
        action_cnt[annotation["actions"][step_idx][0].split("(")[0]] += 1

    screen_width_height[traj_id] = (
        annotation["width"].item(),
        annotation["height"].item(),
    )
    goals[traj_id] = annotation["atomic_digital_goal"][0]
    app_names[traj_id] = annotation["app"][0]

    return trajectories, action_cnt, screen_width_height, goals, app_names


def evaluate_trajectories(
    evaluator,
    trajectories,
    trajectory_id,
    screen_width_height,
    goals,
    app_names,
    action_summary,
    use_future_actions,
):

    cur_trajectory = trajectories[trajectory_id]
    total_steps = len(cur_trajectory)

    # use future action summary in the prompt
    cur_action_summary = []
    if use_future_actions:
        cur_action_summary = [
            action_summary[trajectory_id][i] for i in range(total_steps)
        ]

    trajectories[trajectory_id], goals[trajectory_id] = evaluator.preprocess(
        trajectory_id,
        trajectories[trajectory_id],
        screen_width_height[trajectory_id],
        goals[trajectory_id],
        app_names[trajectory_id],
    )

    step_progress_list, trajectory_progress = evaluator.evaluate(
        goal=goals[trajectory_id],
        trajectory_id=trajectory_id,
        trajectory=trajectories[trajectory_id],
        use_future_actions=use_future_actions,
        action_summary=cur_action_summary,
    )

    return step_progress_list, trajectory_progress
