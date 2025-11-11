# Copyright (c) Meta Platforms, Inc. and affiliates.

from auto_evaluator.models import ApiWrapper

LLAMA4_USER_PROMPT = "Goal: {app_prefix}{goal}. What action should the user take next?"

LLAMA4_SYSTEM_PROMPT = """
Assist an Android user by generating actions based on their conversational input and the current screen image.
Available actions (pick one):
- tap(x, y): Tap at screen location (x, y). Example: tap(0.312, 0.589).
- swipe(x1, y1) to (x2, y2). Example: swipe(0.171, 0.350, 0.899, 0.357).
- type(text): Type text. Example: type('Hello').
- navigate(option): Navigate options: {back, home, enter}. Example: navigate(back).
- end(option): End options: {complete, impossible}. Example: end(complete).
Please respond with a single action, with no additional text.
"""

import io
import os
import sys
import xml.etree.ElementTree as ET
from base64 import b64encode
from typing import Any, List, Optional, Tuple

from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, ".."))
sys.path.insert(0, current_dir)

from action import Action
from action_parser import parse_action
from auto_evaluator.utils import (
    generate_ui_elements_description_list_full,
    traverse_xml,
)
from goal_payload import GoalPayload
from openai import OpenAI
from prompts import format_action_history_for_screen
from ui_state import base64_to_bytes


def image_to_base64(image: Image.Image) -> str:
    image_buffer = io.BytesIO()
    image.save(image_buffer, format="JPEG")
    return b64encode(image_buffer.getvalue()).decode()


class Llama4Predictor(ApiWrapper):
    def __init__(
        self,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        model_id: str,
        use_screenshot: bool,
        api_key_name: str,
        base_url: str,
        **kwargs,
    ):
        super().__init__(
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            model_id=model_id,
            use_screenshot=use_screenshot,
            api_key_name=api_key_name,
            base_url=base_url,
            **kwargs,
        )

    async def predict(self, goal_payload: GoalPayload) -> Optional[Action]:
        ui_elements = None
        screen_width = (
            int(goal_payload.image_size[0])
            if goal_payload.image_size is not None
            else 0
        )
        screen_height = (
            int(goal_payload.image_size[1])
            if goal_payload.image_size is not None
            else 0
        )

        screen_width_height = (screen_width, screen_height)
        json_raw = goal_payload.ui_raw if goal_payload.ui_raw else ""
        xml_raw = json_raw

        if xml_raw:
            root_node = ET.fromstring(xml_raw)
            leaf_nodes, interactable_nodes = traverse_xml(
                root_node, screen_width_height
            )
            ui_elements = generate_ui_elements_description_list_full(
                leaf_nodes, screen_width_height, shorten_ui=True, progress_only=False
            )

        app_prefix = ""
        if goal_payload.app_name is not None and len(goal_payload.app_name) > 0:
            app_prefix = f"In the {goal_payload.app_name} app: "

        action_history_string = format_action_history_for_screen(
            goal_payload.action_history, screen_width, screen_height
        )

        system_message = LLAMA4_SYSTEM_PROMPT
        user_message = LLAMA4_USER_PROMPT.format(
            app_prefix=app_prefix, goal=goal_payload.goal
        )

        if action_history_string:
            user_message += f"\nPrevious actions: {action_history_string}"

        if ui_elements:
            user_message += f"\nScreen elements: {ui_elements}"

        action, _ = await self.call_model_and_parse(
            goal_payload=goal_payload,
            system_message=system_message,
            user_message=user_message,
        )
        return action

    async def call_model_and_parse(
        self,
        goal_payload: GoalPayload,
        system_message: str,
        user_message: str,
    ) -> tuple[Optional[Action], Optional[str]]:
        image = None
        if goal_payload.image_raw is not None and self.use_screenshot:
            image_data = goal_payload.image_raw["data"]
            image_bytes = base64_to_bytes(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            image = [("screenshot", image)]

        response = await self.complete_chat(
            system_prompt=system_message, user_prompt=user_message, image=image
        )

        # Parse the response to extract the action
        action_text = response.strip()

        # Convert the action text into an Action object
        action = parse_action(
            action=action_text,
            image_size=goal_payload.image_size,
            session_id=goal_payload.session_id,
            request_id=goal_payload.request_id,
        )

        prompt = f"{system_message}\n\n{user_message}"
        return action, prompt
