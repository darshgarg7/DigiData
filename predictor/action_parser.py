# Copyright (c) Meta Platforms, Inc. and affiliates.

import re
from typing import Dict, List, Optional, Tuple

from action import Action

SWIPE_DURATION_MS: int = 300
TAP_DURATION_MS: int = 100
MIN_SWIPE_DURATION_MS: int = 100

def parse_action(
        action: str,
        image_size: Tuple[int, int],
        session_id: str,
        request_id: str,
        swipe_duration_ms: int = SWIPE_DURATION_MS
    ) -> Optional[Action]:
    if swipe_duration_ms < MIN_SWIPE_DURATION_MS:
        swipe_duration_ms = MIN_SWIPE_DURATION_MS
    
    match = re.match(r"(\w+)\(([\d.,\s]+)\)", action)
    if (match is not None and match):
        function = match.group(1)
        args = list(map(float, match.group(2).split(',')))
        x_y: Dict[str, List[float]] = {"x":[], "y":[]}
        for i, value in enumerate(args):
            if i % 2 == 0:
                x_y["x"].append(value * image_size[0])
            else:
                x_y["y"].append(value * image_size[1])
        if function == "tap":
            function_name = "tap"
            x = x_y["x"][0]
            y = x_y["y"][0]
            args = [[x, y], True]
            return Action(session_id=session_id, request_id=request_id, function_name=function_name, args=args, generated_action=action)
        elif function == "swipe":
            function_name = "swipe"
            x1 = x_y["x"][0]
            y1 = x_y["y"][0]
            x2 = x_y["x"][1]
            y2 = x_y["y"][1]
            args = [[x1, x2], [y1, y2], [swipe_duration_ms], True]
            return Action(session_id=session_id, request_id=request_id, function_name=function_name, args=args, generated_action=action)

    match = re.match(r"type\([\'\"]*(.*?)[\'\"]*\)", action)
    if (match is not None and match):
        function_name = "type"
        args = [match.group(1)]
        return Action(session_id=session_id, request_id=request_id, function_name=function_name, args=args, generated_action=action)

    match = re.match(r"navigate\([\'\"]*(.*?)[\'\"]*\)", action)
    if (match is not None and match):
        param = match.group(1)
        args = []
        if param == "home":
            function_name = "navigate(home)"
            return Action(session_id=session_id, request_id=request_id, function_name=function_name, args=args, generated_action=action)
        elif param == "back":
            function_name = "navigate(back)"
            return Action(session_id=session_id, request_id=request_id, function_name=function_name, args=args, generated_action=action)

    match = re.match(r"end\([\'\"]*(.*?)[\'\"]*\)", action)
    match = match if match else re.match(r"complete\([\'\"]*(.*?)[\'\"]*\)", action)
    match = match if match else "status(complete)" in action
    if (match is not None and match):
        function_name = "complete"
        args = []
        return Action(session_id=session_id, request_id=request_id, function_name=function_name, args=args, generated_action=action)
    
    return None