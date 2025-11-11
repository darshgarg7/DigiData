# Copyright (c) Meta Platforms, Inc. and affiliates.

import json
import io
import os
import sys
import xml.etree.ElementTree as ET

from base64 import b64encode
from PIL import Image
from typing import Optional, List, Any

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, '..'))
sys.path.insert(0, current_dir)

from action import Action
from action_parser import parse_action
from auto_evaluator.utils import (
    generate_ui_elements_description_list_full,
    traverse_xml,
)
from goal_payload import GoalPayload
from openai import OpenAI
from prompts import OPENAI_GPT4O_SYSTEM, OPENAI_GPT4O_USER, format_action_history_for_screen
from tools import COMPLETE_TOOL, SWIPE_TOOL, TAP_TOOL, TEXT_INPUT_TOOL
from ui_state import base64_to_bytes


class OpenAIGPT4OPredictor:
    def __init__(
        self,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        model_id: str,
        use_screenshot: bool,
        **kwargs,
    ):
        self.client = OpenAI()
        self.model_id = model_id
        self.temperature = temperature
        self.top_p = top_p
        self.use_screenshot = use_screenshot
        self.max_new_tokens = max_new_tokens

    async def predict(self, goal_payload):
        ui_elements = None
        screen_width = int(goal_payload.image_size[0]) if goal_payload.image_size is not None else 0
        screen_height = int(goal_payload.image_size[1]) if goal_payload.image_size is not None else 0

        screen_width_height = (screen_width, screen_height)
        json_raw = goal_payload.ui_raw if goal_payload.ui_raw else ""
        xml_raw = json_raw
        root_node = ET.fromstring(xml_raw)
        leaf_nodes, interactable_nodes = traverse_xml(
            root_node,
            screen_width_height
        )
        ui_elements = generate_ui_elements_description_list_full(
            leaf_nodes,
            screen_width_height,
            shorten_ui=True,
            progress_only=False
        )
        
        system_message = None
        user_message = None

        system_message = OPENAI_GPT4O_SYSTEM
        app_prefix = ''
        if goal_payload.app_name is not None and len(goal_payload.app_name) > 0:
            app_prefix = f'In the {goal_payload.app_name} app: '
        action_history_string = format_action_history_for_screen(
            goal_payload.action_history,
            screen_width,
            screen_height
        )   
        user_message = OPENAI_GPT4O_USER.format(
            app_prefix=app_prefix,
            goal=goal_payload.goal,
            action_history=action_history_string,
            ui_raw=ui_elements
        )
        action, prompt = await self.call_model_and_parse(
            goal_payload=goal_payload,
            system_message=system_message,
            user_message=user_message
        )
        return action

    async def call_model_and_parse(
        self,
        goal_payload: GoalPayload,
        system_message: Optional[str],
        user_message: Optional[str],
    ) -> tuple[Optional[Action], Optional[str]]:
        if goal_payload.image_raw is None:
            raise Exception("GPT4_O_TOOLS: Image cannot be None")
        response = await self.run_model(
            system_message=system_message,
            user_message=user_message,
            tools=[COMPLETE_TOOL, SWIPE_TOOL, TAP_TOOL, TEXT_INPUT_TOOL],
            tool_choice="required",
            image_data=goal_payload.image_raw["data"],
        )
        tool_name = response[0].function.name
        tool_args = json.loads(response[0].function.arguments)
        generated_action = ""
        screen_width = float(goal_payload.image_size[0])
        screen_height = float(goal_payload.image_size[1])
        if tool_name == "tap":
            x = float(tool_args["x_coordinate"]) / screen_width
            y = float(tool_args["y_coordinate"]) / screen_height
            if x < (1/screen_width):
                x = 0
            if y < (1/screen_height):
                y = 0
            generated_action = f"{tool_name}({x}, {y})"
        elif tool_name == "swipe":
            x_1 = float(tool_args["x_1"]) / screen_width
            x_2 = float(tool_args["x_2"]) / screen_width
            y_1 = float(tool_args["y_1"]) / screen_height
            y_2 = float(tool_args["y_2"]) / screen_height
            if x_1 < (1/screen_width):
                x_1 = 0
            if x_2 < (1/screen_width):
                x_2 = 0
            if y_1 < (1/screen_height):
                y_1 = 0
            if y_2 < (1/screen_height):
                y_2 = 0
            generated_action = f"{tool_name}({x_1}, {y_1}, {x_2}, {y_2})"
        elif tool_name == "text_input":
            text = tool_args["text"]
            generated_action = f"type({text})"
        elif tool_name == "complete":
            generated_action = "complete()"
        action = parse_action(
            action=generated_action,
            image_size=goal_payload.image_size,
            session_id=goal_payload.session_id,
            request_id=goal_payload.request_id
        )
        prompt = f"{system_message}\n\n{user_message}" if system_message and user_message else system_message or user_message
        return action, prompt

    async def run_model(
        self,
        system_message: Optional[str],
        user_message: Optional[str],
        tools: Optional[List],
        tool_choice: Optional[str],
        image_data: Optional[str],
    ) -> Any:
        messages = []
        if system_message is not None and len(system_message) > 0:
            messages.append(
                {
                    "role": "system",
                    "content": [
                        {
                            "text": system_message,
                            "type": "text"
                        }
                    ]
                }
            )
        if user_message is not None and len(user_message) > 0:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "text": user_message,
                            "type": "text"
                        }
                    ]
                }
            )
        attachments = {}
        user_message_index = next((i for i, msg in enumerate(messages) if msg["role"] == "user"), None)

        image = None
        if image_data is not None and user_message_index is not None:
            image = base64_to_bytes(image_data)
            messages[user_message_index]["content"].append(
                {
                    "type": "image"
                }
            )
        try:
            processed_messages = messages.copy()
            for i in range(0,len(messages)):
                message = messages[i]
                for j in range(0,len(message["content"])):
                    content = message["content"][j]
                    if content["type"] == "image":
                        pil_image = Image.open(io.BytesIO(image))
                        image = pil_image.convert("RGB")
                        image_buffer = io.BytesIO()
                        image.save(image_buffer, format="JPEG")
                        image_content = {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64encode(image_buffer.getvalue()).decode()}"
                            },
                        }
                        processed_messages[i]["content"][j] = image_content

            completion = self.client.chat.completions.create(
                model=self.model_id,
                messages=processed_messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_new_tokens,
                tools=tools,
                tool_choice=tool_choice,
            )
            function = completion.choices[0].message.tool_calls
            response_content = function
            return response_content

        except Exception as e:
            print(f"Error calling GPT4o: {e}")

        return None