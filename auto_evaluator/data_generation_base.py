# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Step-level progress evaluation prompt for an agent"""

import asyncio
import logging
import os
import re
import xml.etree.ElementTree as ET
from typing import Optional

from auto_evaluator.models import model_mapping

from auto_evaluator.utils import (
    extract_action_and_normalize_coordinates,
    generate_ui_elements_description_list_full,
    parse_action,
    traverse_xml,
    visualize_screenshot,
)
from PIL import Image

logger = logging.getLogger("auto_evaluator.data_generation_base")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


class DataGenerationBase:
    """
    Step-level and trajectory-level evaluator.
    """

    def __init__(
        self,
        model_id: str = "",
        model_host_type: str = "",
        use_screenshot: bool = False,
        use_xml: bool = True,
        visualize_xml: bool = False,
        visualize_xml_dir: str = "",
        temperature: float = 0.01,
        max_new_tokens: int = 512,
        top_p: float = 0.8,
        use_two_screenshots: bool = True,
        api_key_name: str = "",
        base_url: str = "",
    ) -> None:
        self.model_host_type = model_host_type
        self.use_screenshot = use_screenshot
        self.use_xml = use_xml
        self.visualize_xml = visualize_xml
        self.visualize_xml_dir = visualize_xml_dir
        self.use_two_screenshots = use_two_screenshots

        self.model = model_mapping[model_host_type](
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            model_id=model_id,
            use_screenshot=use_screenshot,
            api_key_name=api_key_name,
            base_url=base_url,
        )

    async def preprocess_step(
        self,
        trajectory_id: str,
        step_id: int,
        step_dict: dict,
        screen_width_height: tuple[int, int],
    ) -> dict:

        # get normalized action coordinate
        action_dict_with_normalized_coordinates = (
            extract_action_and_normalize_coordinates(
                step_dict["raw_action"], screen_width_height
            )
        )

        try:
            root_node = ET.fromstring(step_dict["raw_xml"])
        except ET.ParseError:
            logger.info(f"Failed to parse xml for {trajectory_id}_{step_id}")
            root_node = None

        leaf_nodes, interactable_nodes = traverse_xml(root_node, screen_width_height)

        # parse xml (coordinates are normalized to [0, screen_width_height])
        step_dict["state_text"] = generate_ui_elements_description_list_full(
            leaf_nodes, screen_width_height, progress_only=True
        )
        output_width_height = screen_width_height

        # parse action, associate action coordinates with direction or interactable xml node
        step_dict["processed_action"] = parse_action(
            action_dict_with_normalized_coordinates,
            interactable_nodes,
            output_width_height,
            screen_width_height,
        )

        # if failed to associate "tap" location with interactable node, try associating it with leaf_nodes
        if (
            "action_type" in action_dict_with_normalized_coordinates
            and action_dict_with_normalized_coordinates["action_type"] == "tap"
            and "likely tapping on" not in step_dict["processed_action"]
        ):
            step_dict["processed_action"] = parse_action(
                action_dict_with_normalized_coordinates,
                leaf_nodes,
                output_width_height,
                screen_width_height,
            )

        if "raw_screenshot" not in step_dict:
            step_dict["raw_screenshot"] = None

        if "human_annotation" not in step_dict:
            step_dict["human_annotation"] = None

        # visualize on top of screenshot
        if self.visualize_xml and step_dict["raw_screenshot"] is not None:
            vis_path = os.path.join(
                self.visualize_xml_dir,
                f"{trajectory_id}_{step_id}.png",
            )
            visualize_screenshot(
                step_dict["raw_screenshot"].copy(),
                leaf_nodes,
                step_dict["raw_action"],
                vis_path,
            )
            logger.info(
                f"Visualization for {trajectory_id}_{step_id} saved: {step_dict['processed_action']}"
            )

        return step_dict

    async def preprocess(
        self,
        trajectory_id: str,
        trajectory_dict: dict,
        screen_width_height: tuple[int, int],
    ) -> dict:

        tasks = []
        for step_id, step_dict in trajectory_dict.items():
            task = asyncio.create_task(
                self.preprocess_step(
                    trajectory_id, step_id, step_dict, screen_width_height
                )
            )
            tasks.append(task)
        results = await asyncio.gather(*tasks)
        trajectory_dict = {
            step_id: result for step_id, result in zip(trajectory_dict.keys(), results)
        }

        return trajectory_dict

    def step_evaluation_prompt(
        self,
        goal: str,
        raw_action: str,
        processed_action: str,
        before_elements: str,
        after_elements: Optional[str] = None,
        before_screenshot: Optional[Image.Image] = None,
        after_screenshot: Optional[Image.Image] = None,
        step_id: int = 0,
        history: Optional[list[str]] = None,
    ) -> tuple[str, str, Optional[list[tuple[str, Image.Image]]]]:
        """
        Create system, user prompt and optionally image to prompt MLLM
        to evaluate the progress of the agent.
        """
        return "", "", None

    def trajectory_evaluation_prompt(
        self,
        goal_string: str,
        initial_state: str,
        final_state: str,
        initial_screenshot: Optional[Image.Image] = None,
        final_screenshot: Optional[Image.Image] = None,
        summaries: Optional[list[str]] = None,
    ) -> tuple[str, str, Optional[list[tuple[str, Image.Image]]]]:
        """
        Create system, user prompt and optionally image to prompt MLLM
        to evaluate whether goal was achieved in the episode.
        """

        return "", "", None

    def parse_judgment(self, evaluator_output: str) -> int:
        """
        Extract the last word of evaluator_output string,
        It should be Yes or No. If yes, return 1, if no, return 0. If parsing fails, return -1.
        """
        indicator = evaluator_output.split()[-1].lower()
        if indicator in {"yes", "yes.", "**yes**"}:
            success = 1
        elif indicator in {"no", "no.", "**no**"}:
            success = 0
        else:
            indicator = re.search(
                r"(?:Judgment\**:|Answer\**:)(.*?)(Yes|No)", evaluator_output, re.S
            )
            indicator = indicator.group(2).lower() if indicator else ""
            if indicator == "yes":
                success = 1
            elif indicator == "no":
                success = 0
            else:
                success = -1

        return success

    def parse_step_eval_output(self, evaluator_output: str) -> dict:
        return {}

    def parse_trajectory_eval_output(self, evaluator_output: str) -> dict:
        """
        Parse the output of the evaluator to determine whether the goal
        was achieved in the episode.

        Args:
        evaluator_output (str): The output string from evaluator including
                                reasoning and judgment in the specified format.

        Returns:
        tuple: A tuple containing the reasoning string and an
        int indicating if the goal was achieved
            (1 for "Yes", 0 for "No", -1 for parsing failure).
        """

        if not evaluator_output:
            reason = ""
            success = -1
        else:
            success = self.parse_judgment(evaluator_output)

            reason_result = re.search(
                r"(?:Reason\**:)(.*?)(?:\n\**Judgment\**:|\n\**Answer\**:)",
                evaluator_output,
                re.S,
            )
            # First search for "Reason: ... (Judgment: | Answer:)"
            if reason_result:
                reason = reason_result.group(1).strip()
            else:
                # Then search for "... (Judgment: | Answer:)"
                reason_result = re.search(
                    r"(.*?)(?:\n\**Judgment\**:|\n\**Answer\**:)",
                    evaluator_output,
                    re.S,
                )
                if reason_result:
                    reason = reason_result.group(1).strip()
                else:
                    reason = ""

            if reason.startswith("**"):
                reason = reason[2:].strip()

        return_dict = {
            "reason": reason,
            "progress_reward": success,
            "summary_progress": evaluator_output,
        }
        return return_dict

    async def evaluate_step(
        self,
        goal: str,
        raw_action: str,
        processed_action: str,
        before_elements: str,
        after_elements: Optional[str] = None,
        before_screenshot: Optional[Image.Image] = None,
        after_screenshot: Optional[Image.Image] = None,
        step_id: int = 0,
        summary_history: Optional[list[str]] = None,
    ) -> dict:
        """
        Evaluate the progress of the agent.
        """
        system_prompt, user_prompt, image = self.step_evaluation_prompt(
            goal,
            raw_action,
            processed_action,
            before_elements,
            after_elements,
            before_screenshot,
            after_screenshot,
            step_id + 1,
            summary_history,
        )
        try:
            evaluator_output = await self.model.complete_chat(
                system_prompt, user_prompt, image
            )
        except Exception as e:
            logger.info(f"An error occurred: {e}")
            evaluator_output = ""

        parsed_model_output = self.parse_step_eval_output(evaluator_output)

        return parsed_model_output

    async def evaluate_trajectory(
        self,
        goal: str,
        initial_state: str,
        final_state: str,
        initial_screenshot: Optional[Image.Image] = None,
        final_screenshot: Optional[Image.Image] = None,
        summary_history: Optional[list[str]] = None,
    ) -> dict:
        """
        Evaluate the success of the agent in achieving the goal_string.
        """
        system_prompt, user_prompt, image = self.trajectory_evaluation_prompt(
            goal,
            initial_state,
            final_state,
            initial_screenshot,
            final_screenshot,
            summary_history,
        )
        try:
            evaluator_output = await self.model.complete_chat(
                system_prompt, user_prompt, image
            )
        except Exception as e:
            logger.info(f"An error occurred: {e}")
            evaluator_output = ""

        parsed_model_output = self.parse_trajectory_eval_output(evaluator_output)

        return parsed_model_output

    async def evaluate(
        self,
        goal: str,
        trajectory_id: str,
        trajectory: dict,
        total_steps: int,
        use_future_actions: bool = False,
        action_summary: list[str] = [],
        existing_step_results: dict = {},
    ) -> tuple[list, dict]:

        return [], {}
