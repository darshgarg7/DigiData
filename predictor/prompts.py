# Copyright (c) Meta Platforms, Inc. and affiliates.

import re

OPENAI_GPT4O_SYSTEM = """Imagine that you are imitating humans operating an Android device for a task step by step. At each stage, you
can see the Android screen like humans by a screenshot and know the previous actions before the current
step decided by yourself through recorded history. You need to decide on the first following action to
take. You can tap on an element, long-press an element, swipe, input text, open an app, or use the
keyboard enter, home, or back key. (For your understanding, they are like ‘adb shell input tap’, ‘adb
shell input swipe’, ‘adb shell input text’, ‘adb shell am start -n’, and ‘adb shell input keyevent’).
One next step means one operation within these actions. Unlike humans, for typing (e.g., in text areas,
text boxes), you should try directly typing the input or selecting the choice, bypassing the need for an
initial click. You should not attempt to create accounts, log in or do the final submission. Terminate
when you deem the task complete or if it requires potentially harmful actions."""


OPENAI_GPT4O_USER = """You are asked to complete the following task: {app_prefix}{goal}

Previous Actions:
{action_history}

The screenshot below shows the Android screen you see. Follow the following guidance to think step by step
before outlining the next action step at the current stage:

(Current Screen Identification)
Firstly, think about what the current screen is.
{ui_raw}

(Previous Action Analysis)
Secondly, combined with the screenshot, analyze each step of the previous action history and their intention
one by one. Particularly, pay more attention to the last step, which may be more related to what you
should do now as the next step. Specifically, if the last action involved a INPUT TEXT, always evaluate
whether it necessitates a confirmation step, because typically a single INPUT TEXT action does not make
effect. (often, simply pressing ’Enter’, assuming the default element involved in the last action,
unless other clear elements are present for operation).

(Screenshot Details Analysis)
Closely examine the screenshot to check the status of every part of the screen to understand what you can
operate with and what has been set or completed. You should closely examine the screenshot details to
see what steps have been completed by previous actions even though you are given the textual previous
actions. Because the textual history may not clearly and sufficiently record some effects of previous
actions, you should closely evaluate the status of every part of the screen to understand what you have
done.

(Next Action Based on Android screen and Analysis)
Then, based on your analysis, in conjunction with human phone operation habits and the logic of app design,
decide on the following action. And clearly outline which element on the Android screen users will
operate with as the first next target element, its detailed location, and the corresponding operation.

To be successful, it is important to follow the following rules:
1. You should only issue a valid action given the current observation.
2. You should only issue one action at a time
3. For handling the select dropdown elements on a screen, it’s not necessary for you to provide completely
accurate options right now. The full list of options for these elements will be supplied later."""

def format_action_history_for_screen(
    action_history: str | None,
    screen_width: float | None,
    screen_height: float | None
):
    formatted_action_history = action_history or ''
    if screen_width is not None and screen_height is not None:
        formatted_action_history = convert_actions(
            action_history,
            screen_width,
            screen_height
        )
    return formatted_action_history

def convert_actions(input_str, screen_width, screen_height):
    actions = re.findall(r'(swipe|tap|type)\(([^)]+)\)', input_str)
    output_actions = []
    for action_type, params in actions:
        params_list = [p.strip() for p in params.split(',')]
        processed_params = []
        if action_type == 'swipe':
            x1 = round(float(params_list[0]) * screen_width)
            y1 = round(float(params_list[1]) * screen_height)
            x2 = round(float(params_list[2]) * screen_width)
            y2 = round(float(params_list[3]) * screen_height)
            processed_params = [str(x1), str(y1), str(x2), str(y2)]
        elif action_type == 'tap':
            x = round(float(params_list[0]) * screen_width)
            y = round(float(params_list[1]) * screen_height)
            processed_params = [str(x), str(y)]
        elif action_type == 'type':
            x = params_list[0]
            processed_params = [str(x)]
        output_action = f"{action_type}({', '.join(processed_params)})"
        output_actions.append(output_action)
    if len(output_actions) > 0:
        return ", ".join(output_actions)
    else:
        return ""