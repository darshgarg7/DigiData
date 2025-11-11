# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Step-level progress evaluation prompt for an agent"""

import logging
import os
import re
from typing import Optional

from auto_evaluator.data_generation_base import DataGenerationBase

from auto_evaluator.prompts import (
    get_full_trajectory_summary,
    get_initial_and_final_xml,
    get_prev_step_summary,
    get_previous_and_next_xml,
    get_trajectory_system_prompt,
    LASTSTEP_IMG_PROMPT,
    NONLASTSTEP_STITCHSCREENSHOT_IMG_PROMPT,
    NONLASTSTEP_TWOSCREENSHOT_IMG_PROMPT,
    STEP_EVALUATION_SYSTEM_MESSAGE,
    STEP_EVALUATION_USER_MESSAGE,
    TRAJECTORY_EVALUATION_STITCHSCREENSHOT_PROMPT,
    TRAJECTORY_EVALUATION_TWOSCREENSHOT_PROMPT,
    TRAJECTORY_EVALUATION_USER_MESSAGE,
)

from auto_evaluator.utils import get_two_screenshots
from PIL import Image

logger = logging.getLogger("auto_evaluator.auto_evaluator")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


class AutoEvaluator(DataGenerationBase):
    """
    Step-level and trajectory-level evaluator.
    """

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

        system_prompt = STEP_EVALUATION_SYSTEM_MESSAGE

        history_string = get_prev_step_summary(history)

        # Format the before and after screenshots
        if self.use_screenshot:

            if self.use_two_screenshots and after_screenshot:
                screenshot_names = ["Screen A", "Screen B"]
            else:
                screenshot_names = ["Screenshot"]

            screenshot = get_two_screenshots(
                raw_action,
                before_screenshot,
                after_screenshot,
                screenshot_names=screenshot_names,
                return_two_screenshots=self.use_two_screenshots,
            )

            if after_screenshot is not None:
                if self.use_two_screenshots:
                    img_info = NONLASTSTEP_TWOSCREENSHOT_IMG_PROMPT.format(
                        step_id=step_id
                    )
                else:
                    img_info = NONLASTSTEP_STITCHSCREENSHOT_IMG_PROMPT.format(
                        step_id=step_id
                    )
            else:
                img_info = LASTSTEP_IMG_PROMPT
        else:
            img_info = ""
            screenshot = None

        # Format the before and after simplified XML
        if self.use_xml:
            xml_string = get_previous_and_next_xml(before_elements, after_elements)
        else:
            xml_string = ""

        # Fill in the user prompt template
        user_prompt = STEP_EVALUATION_USER_MESSAGE.format(
            goal=goal,
            action=processed_action,
            xml_string=xml_string,
            history=history_string,
            step_id=step_id,
            img_info=img_info,
        )

        return system_prompt, user_prompt, screenshot

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
        # Format the steps details
        steps_details = get_full_trajectory_summary(summaries)

        # Format the before and after screenshots
        if self.use_screenshot:

            if self.use_two_screenshots and final_screenshot:
                screenshot_names = ["Initial Screenshot", "Final Screenshot"]
            else:
                screenshot_names = ["Screenshot"]

            screenshot = get_two_screenshots(
                "",
                initial_screenshot,
                final_screenshot,
                screenshot_names=screenshot_names,
                return_two_screenshots=self.use_two_screenshots,
            )

            if self.use_two_screenshots:
                img_info = TRAJECTORY_EVALUATION_TWOSCREENSHOT_PROMPT
            else:
                img_info = TRAJECTORY_EVALUATION_STITCHSCREENSHOT_PROMPT
        else:
            img_info = ""
            screenshot = None

        # Format the initial and final simplified XML
        if self.use_xml:
            xml_string = get_initial_and_final_xml(initial_state, final_state)
        else:
            xml_string = ""

        system_prompt = get_trajectory_system_prompt(use_xml=self.use_xml)

        # Fill in the user prompt template
        user_prompt = TRAJECTORY_EVALUATION_USER_MESSAGE.format(
            goal=goal_string,
            xml_string=xml_string,
            summaries=steps_details,
            img_info=img_info,
        )

        return system_prompt, user_prompt, screenshot

    def parse_step_eval_output(self, evaluator_output: str) -> dict:
        """
        Parse the output of the evaluator to determine whether the step
        is working towards the goal.

        Args:
        evaluator_output (str): The output string from evaluator including
                                reasoning and judgment in the specified format.

        Returns:
        dict: A dict containing the reasoning string, summarization string and an
        int indicating if the step is working towards the goal
            (1 for "Yes", 0 for "No", -1 for parsing failure).
        """

        if not evaluator_output:
            summary = ""
            reason = ""
            progress = -1
        else:
            progress = self.parse_judgment(evaluator_output)

            summary_result = re.search(
                r"(?:Summary:)(.*?)(?:\n\**Reasoning:)", evaluator_output, re.S
            )
            summary = summary_result.group(1).strip() if summary_result else ""
            if summary.startswith("**"):
                summary = summary[2:].strip()

            reason_result = re.search(
                r"(?:Reasoning:)(.*?)(?:\n\**Judgment:)", evaluator_output, re.S
            )
            reason = reason_result.group(1).strip() if reason_result else ""
            if reason.startswith("**"):
                reason = reason[2:].strip()

        return_dict = {
            "summary": summary,
            "reason": reason,
            "progress_reward": progress,
            "summary_progress": evaluator_output,
        }
        return return_dict

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

        summary_history = []
        step_progress_list = []

        for step_id in range(total_steps):

            logger.info(f"Processing {trajectory_id} step {step_id}")

            if step_id in existing_step_results:
                # skip step-level auto-eval if already evaluated
                logger.info(
                    f"skipping step-level eval for {trajectory_id} with step {step_id} because it has already been evaluated"
                )
                summary_history.append(existing_step_results[step_id]["summary"])
                continue
            if step_id not in trajectory:
                # skip if no current state
                logger.info(
                    f"skipping step-level eval for {trajectory_id} with step {step_id} because it has no step {step_id}"
                )
                continue

            cur_step = trajectory[step_id]
            next_step = trajectory[step_id + 1] if step_id + 1 in trajectory else None

            # remove the current acton
            if use_future_actions:
                action_summary = action_summary[1:]

            # evaluate step progress
            step_model_output = await self.evaluate_step(
                goal=goal,
                raw_action=cur_step["raw_action"],
                processed_action=cur_step["processed_action"],
                before_elements=cur_step["state_text"],
                after_elements=(next_step["state_text"] if next_step else None),
                before_screenshot=cur_step["raw_screenshot"],
                after_screenshot=(next_step["raw_screenshot"] if next_step else None),
                step_id=step_id,
                summary_history=summary_history,
            )
            summary_history.append(step_model_output["summary"])

            if step_model_output["progress_reward"] == -1:
                logger.info(
                    f"No valid progress for {trajectory_id} step {step_id}. This might be caused by a model invocation or parsing error."
                )

            step_model_output["trajectory_id"] = trajectory_id
            step_model_output["step_id"] = step_id
            step_model_output["action"] = cur_step["processed_action"]

            step_progress_list.append(step_model_output)

        if 0 not in trajectory or total_steps - 1 not in trajectory:
            # skip if initial or last step not in the trajectory
            trajectory_progress = {
                "goal": goal,
                "trajectory_id": trajectory_id,
                "reason": "",
                "progress_reward": -1,
                "summary_progress": "",
            }
            logger.info(
                f"skipping trajectory-level eval for {trajectory_id} because it has no initial or final state"
            )
        else:
            initial_state = trajectory[0]["state_text"]
            final_state = trajectory[total_steps - 1]["state_text"]
            initial_screenshot = trajectory[0]["raw_screenshot"]
            final_screenshot = trajectory[total_steps - 1]["raw_screenshot"]

            # evaluate trajectory progress
            trajectory_progress = await self.evaluate_trajectory(
                goal=goal,
                initial_state=initial_state,
                final_state=final_state,
                initial_screenshot=initial_screenshot,
                final_screenshot=final_screenshot,
                summary_history=summary_history,
            )

            trajectory_progress["trajectory_id"] = trajectory_id
            trajectory_progress["goal"] = goal

        return step_progress_list, trajectory_progress
