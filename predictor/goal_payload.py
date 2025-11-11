# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
import sys

from pydantic import BaseModel, Field
from typing import Dict, Optional, Tuple

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from ui_state import image_to_base64

class GoalPayload(BaseModel):

    class Config:
        arbitrary_types_allowed = True

    session_id: str = Field(default="")
    request_id: str = Field(default="")
    goal_timestamp: int = Field(default=0)
    goal: str = Field(default="")
    notify_message_service: bool = Field(default=False)
    android_agent_host: Optional[str] = Field(default=None)
    android_agent_port: Optional[int] = Field(default=None)
    feedback: Optional[str] = Field(default=None)
    auto_confirm_action: bool = Field(default=False)
    image_raw: Optional[Dict[str, str]] = Field(default=None)
    image_size: Optional[Tuple[int, int]] = Field(default=None)
    ui_raw: Optional[str] = Field(default=None)
    max_steps: Optional[int] = Field(default=None)
    app_name: Optional[str] = Field(default=None)
    action_history: Optional[str] = Field(default=None)

    def message_fields(self) -> "GoalPayload":
        message_fields = self.dict()
        return GoalPayload(**message_fields)


def create_goal_payload(screenshot, goal, ui_raw=None, action_history=None):
    """
    Create a GoalPayload object from a screenshot and optional UI raw data.

    Args:
        screenshot (Image): The screenshot image.
        ui_raw (str): Optional UI raw data.
        action_history (str): Optional action history.

    Returns:
        GoalPayload: A GoalPayload object containing the screenshot and optional UI raw data.
    """
    dict_vals = {}
    dict_vals["image_raw"] = {
        "data": image_to_base64(screenshot),
    }
    dict_vals["goal"] = goal
    dict_vals["image_size"] = screenshot.size
    dict_vals["ui_raw"] = ui_raw
    dict_vals["action_history"] = '' if action_history is None else action_history
    goal_payload = GoalPayload(**dict_vals)
    return goal_payload