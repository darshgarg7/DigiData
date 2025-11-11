# Copyright (c) Meta Platforms, Inc. and affiliates.

from typing import List

from langchain_core.tools import BaseTool
from langchain_core.tools import tool


# Dummy tools to reuse the planning chain
@tool()
async def tap() -> None:
    """tap"""
    pass
    
@tool()
async def swipe() -> None:
    """swipe"""
    pass

@tool()
async def text_input() -> None:
    """text_input"""
    pass

STATIC_ANDOJO_VISION_TOOLS: List[BaseTool] = [
    tap, swipe, text_input
]

TAP_TOOL = {
    "type": "function",
    "function": {
        "name": "tap",
        "description": "Tap on a specific place on the phone",
        "parameters": {
            "type": "object",
            "properties": {
                "x_coordinate": {
                    "type": "string",
                    "description": "Based on the image, output the pixel on the HORIZONTAL axis where the user should tap. Value must be related to provided bounding boxes"
                },
                "y_coordinate": {
                    "type": "string",
                    "description": "Based on the image, output the pixel on the VERTICAL axis where the user should tap. Value must be related to provided bounding boxes"
                }
            },
            "required": [
                "x_coordinate",
                "y_coordinate"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
}
SWIPE_TOOL = {
    "type": "function",
    "function": {
        "name": "swipe",
        "description": "Swipe towards a specific direction on the phone",
        "parameters": {
            "type": "object",
            "properties": {
                "x_1": {
                    "type": "string",
                    "description": "Based on the image, output the pixel on the HORIZONTAL axis where the user should START the swipe gesture. Value must be related to provided bounding boxes"
                },
                "x_2": {
                    "type": "string",
                    "description": "Based on the image, output the pixel on the HORIZONTAL axis where the user should END the swipe gesture. Value must be related to provided bounding boxes"
                },
                "y_1": {
                    "type": "string",
                    "description": "Based on the image, output the pixel on the VERTICAL axis where the user should START the swipe gesture. Value must be related to provided bounding boxes"
                },
                "y_2": {
                    "type": "string",
                    "description": "Based on the image, output the pixel on the VERTICAL axis where the user should END the swipe gesture. Value must be related to provided bounding boxes"
                }
            },
            "required": [
                "x_1", "x_2", "y_1", "y_2"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
}
TEXT_INPUT_TOOL = {
    "type": "function",
    "function": {
        "name": "text_input",
        "description": "Input text on a specific text input field of the phone",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text the user should type based on the user goal input"
                },
            },
            "required": [
                "text"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
}
COMPLETE_TOOL = {
    "type": "function",
    "function": {
        "name": "complete",
        "description": "If there is no other action to take, inform the user that the task is completed",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False
        },
        "strict": True
    }
}
