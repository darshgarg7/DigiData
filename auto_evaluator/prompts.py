# Copyright (c) Meta Platforms, Inc. and affiliates.

""" Collection of prompts for agent, step-evaluators,
and success-evaluators """

from typing import Optional

##########################################
# FULL TASK SUCCESS AUTO-EVALUTION PROMPTS
##########################################

TRAJECTORY_EVALUATION_INTRO = (
    "You are an expert in evaluating the performance of an Android navigation"
    " agent, whose role is to help a human user navigate the android phone"
    " to complete a goal."
    " The agent takes step-by-step phone actions like clicking,"
    " swiping, typing, navigating home, navigating back, and ending.\n\n"
    "You are given the goal, the initial and final states of the android system."
    " For each action that the agent takes, you are also provided with"
    " a summary of the transition from previous state (State A) to next state (State B) due to the action.\n\n"
)

TRAJECTORY_EVALUATION_OUTPUT = (
    "Your task is to evaluate whether the goal was achieved and stays achieved during the episode."
    " Please respond first with a reasoning of your evaluation"
    " based on (i) the conditions required to fulfill the goal,"
    " and (ii) insights drawn from all relevant state transitions concerning these conditions."
    " Do NOT repeat the summary of the transition for each step in your reasoning."
    " Then, respond with your Yes/No judgment."
    " Your answer should be in the following format:\n"
    "Reason: <reasoning>\n"
    "Judgment: <Yes/No>\n\n"
)

TRAJECTORY_EVALUATION_TIPS = (
    "Tips for deciding your evaluation:\n"
    "- The ending action shows the agent's decision to stop the episode, but it doesn't necessarily indicate goal completion."
    " You should deduce if the goal was achieved based on all previous state transitions and the agent's action.\n"
    "- The agent should always issue an ending action in the final step to stop the episode."
    " Failure to do so indicates goal not achieved.\n"
    "- Consider the screen types of the initial or final states, such as:"
    " home (workspace), list of settings, inside an app, status menu etc.\n"
    "- There may be actions or states at any step that are irrelevant to the goal."
    " Focus only on the relevant state transitions and reason agent's intent behind them, ignore the irrelevant steps.\n"
    "- The goal is considered achieved as long as it is reached at any step and maintained thereafter,"
    " even if subsequent irrelevant actions divert the focus."
    " However, if the goal is initially met but later reversed, it should be considered as not achieved.\n"
    "- For information-seeking tasks, such as retrieving information from the Android phone,"
    " the intermediate or final state should either provide the requested information or state its unavailability."
    " Goal is achieved with information presented in any text format, including nested or secondary text.\n"
    "- Sometimes the goal is achieved by default, not as a result of the agent's actions,"
    " such as when the objective is to activate a feature that is already enabled"
    " (like a toggle that's on or a checkbox that's checked), there's no need to interact with it."
    " If information at any state indicates that the goal is met and maintained thereafter, it should be considered goal achieved.\n"
    "- If the goal is to enable, toggle, or turn on a feature, pay attention to the checkbox or toggle state."
    " Whether it is already enabled or the agent interacts with it to enable it,"
    " both are considered goal achieved.\n"
    "- When the agent follows the necessary steps but is blocked by external factors,"
    " like trying to add an unavailable item to the cart or checking unavailable information,"
    " the effort is still considered as achieving the goal.\n"
)

TRAJECTORY_EVALUATION_TIPS_CONT = (
    "- The full goal must be achieved, not partial. For example, if the goal is to open a website,"
    " ensure it is opened, not just entering the URL in the search bar. \n"
    "- For tasks that specify parameters for the goal,"
    " all the parameters must be correctly applied by the agent.\n"
    "- For tasks that involve saving or creating, ensure that the agent"
    " has executed a save action or a comparable action. You will not"
    " receive confirmation that the save occurred, but you can assume it"
    " was successful if you observe that the agent clicked a save button"
    " or performed an equivalent action.\n"
    "- If the goal involves opening a specific app, don't confuse it with similar apps, such as Google vs. Chrome.\n"
    "- If the goal involves opening a specific setting,"
    " don't confuse it with similar settings, such as privacy vs. security.\n\n"
)

TRAJECTORY_EVALUATION_SYSTEM_MESSAGE = (
    f"{TRAJECTORY_EVALUATION_INTRO}"
    f"{TRAJECTORY_EVALUATION_OUTPUT}"
    f"{TRAJECTORY_EVALUATION_TIPS}\n"
    "Be critical and conservative. Before saying yes, make sure that:\n"
    f"{TRAJECTORY_EVALUATION_TIPS_CONT}"
)

TRAJECTORY_EVALUATION_USER_MESSAGE = (
    "{img_info}"
    'The goal of the trajectory is "{goal}".\n\n'
    "{summaries}"
    "{xml_string}"
)

TRAJECTORY_EVALUATION_STITCHSCREENSHOT_PROMPT = (
    "You are given an image contains two side-by-side screenshots from an Android navigation trajectory."
    " The left screenshot (screen A) shows the initial state of the android system, and the right screenshot (screen B) shows the final state.\n\n"
)

TRAJECTORY_EVALUATION_TWOSCREENSHOT_PROMPT = (
    "You are given two screenshots from an Android navigation trajectory."
    " The first screenshot (Initial Screenshot) shows the initial state of the android system, and the second screenshot (Final Screenshot) shows the final state.\n\n"
)

##########################################
# STEP-LEVEL PROGRESS EVALUATION PROMPTS
##########################################

STEP_EVALUATION_INTRO = (
    "You are an expert in evaluating the progress of an Android navigation"
    " agent, whose role is to help a human user navigate the android phone"
    " to complete a goal."
    " The agent takes step-by-step phone actions like clicking,"
    " swiping, typing, navigating home, navigating back, and ending.\n\n"
    "You are given the goal and the state transition from a previous state (State A) to the next (State B),"
    " resulting from the agent's action at a particular step."
    " State A and B are presented as screenshots and lists of UI elements extracted from the screenshots."
    " If this is the last step of the episode, then only State A is provided."
    " Note that the UI elements may be missing or incomplete,"
    " and the screenshots serve as the source of truth if available."
    " You also have access to summary of each past transitions.\n\n"
)

STEP_EVALUATION_OUTPUT = (
    "Your task is to evaluate whether this step constitutes progress towards the goal."
    " Respond first with a summary of changes from A to B."
    " Then, provide a reasoning of your evaluation."
    " If this is the last step, summarize only State A,"
    " and reason whether State A should be considered the final state as whether the goal has been achieved."
    " Finally, respond with your Yes/No judgment."
    " Your answer should be in the following format:\n"
    "Summary: <summary of A to B or summary of A if this is the last step>\n"
    "Reasoning: <reasoning>\n"
    "Judgment: <Yes/No>\n\n"
)

STEP_EVALUATION_SUMMARIZING = (
    "Tips for summarizing:\n"
    "- What key elements (e.g., icons, texts, buttons, navigation bars, images, forms and input field, pop-ups, switches and toggles etc)"
    " are there in each screenshot. Only mention key elements which are related to the goal and ignore the irrelevant ones.\n"
    "- What did the agent do to get from A to B."
    " If the action involves tapping, infer which key element is tapped based on the action, hollow green circle location and the context of the screenshot."
    " If the action involves swiping, infer the swipe direction based on the action, green arrow and the context of the screenshot.\n"
    "- What are the changes from A to B related to the goal. Pay attention to subtle changes including altered or new key elements related to the goal.\n\n"
)

STEP_EVALUATION_TIPS = (
    "Tips for deciding your evaluation:\n"
    "- The ending action shows the agent's decision to stop the episode, but it doesn't necessarily indicate goal completion."
    " You should deduce if the goal was achieved based on the state transitions and the agent's action.\n"
    "- The agent should always issue an ending action in the final step to stop the episode."
    " Failure to do so indicates goal not achieved.\n"
    "- Reason agent's intent behind the step but do not repeat the summary of each past transitions.\n"
    "- Sometimes the goal is achieved by default, not as a result of the agent's actions."
    " If information at any state indicates that the goal is met,"
    " the agent's ending action at that step should be considered as progress toward the goal.\n"
    "- If the goal is to enable, toggle, or turn on a feature, pay attention to the checkbox or toggle state."
    " Whether it is already enabled (e.g., a toggle that's on or a checkbox that's checked) or the agent interacts with it to enable it,"
    " both are considered progress toward the goal.\n"
)

STEP_EVALUATION_POSITIVE_EXAMPLES = (
    "Examples of making progress towards the goal:\n"
    "- State B achieves the goal or some subgoal towards the goal.\n"
    "- State A achieves the goal and the agent issues an Ending action.\n"
    "- open the apps_list, when goal requires an app.\n"
    "- click and open a goal-relevant app.\n"
    "- make goal-directed clicks, when inside a goal-relevant app.\n"
    "- reset with navigate_home or navigate_back, when inside a goal-irrelevant app.\n"
    "- click an option, when pop-up appears.\n"
    "- search for the target element (e.g., goal-relevant app) by typing keywords in the search bar or swiping through the screen multiple times."
    " The target element may not appear in intermediate UI element lists during the multiple swipes.\n"
    "- install the goal-relevant app if the goal requires using an app that is not currently installed.\n"
    "- click on a text field to focus on it so that it can type in the next step,"
    " such a change may not detected by the UI element list.\n"
    "- follows the action but is blocked by external factors,"
    " like trying to add an unavailable item to the cart or checking unavailable information.\n"
)

STEP_EVALUATION_NEGATIVE_EXAMPLES = (
    "Examples of NOT making progress towards the goal:\n"
    "- nothing changed from State A to B.\n"
    "- State A achieves the goal but the agent doesn't issue an Ending action.\n"
    "- need to open an app, but instead navigate back to home from apps_list\n"
    "- click and open a goal-irrelevant app, such as confusing similar apps like Google and Chrome, and navigating within it.\n"
    "- click and open a goal-irrelevant setting, such as confusing similar settings like privacy vs. security.\n"
    "- end the episode without achieving the goal or only partially achieving it.\n"
    "- repreatly perform actions that do not contribute to goal completion, such as opening and closing the same app multiple times.\n\n"
)

STEP_EVALUATION_SYSTEM_MESSAGE = (
    f"{STEP_EVALUATION_INTRO}"
    f"{STEP_EVALUATION_OUTPUT}"
    f"{STEP_EVALUATION_SUMMARIZING}"
    f"{STEP_EVALUATION_TIPS}"
    f"{STEP_EVALUATION_POSITIVE_EXAMPLES}"
    f"{STEP_EVALUATION_NEGATIVE_EXAMPLES}"
)

STEP_EVALUATION_USER_MESSAGE = (
    "{img_info}"
    'The goal of the trajectory is "{goal}".\n\n'
    "{history}"
    "The agent's current action at step {step_id} is {action}.\n\n"
    "{xml_string}"
)

NONLASTSTEP_STITCHSCREENSHOT_IMG_PROMPT = (
    "You are given an image contains two side-by-side screenshots from an Android navigation trajectory."
    " The left screenshot (screen A) is before agent's action at step {step_id}, and the right screenshot (screen B) is after the action."
    " If the action involves tapping or long press, the tap location is shown on screen A with a hollow green circle."
    " If the action involves swiping, the start and end points of the swipe are shown on Screen A with an arrow.\n\n"
)

NONLASTSTEP_TWOSCREENSHOT_IMG_PROMPT = (
    "You are given two screenshots from an Android navigation trajectory."
    " The first screenshot (screen A) is before agent's action at step {step_id}, and the second screenshot (screen B) is after the action."
    " If the action involves tapping or long press, the tap location is shown on screen A with a hollow green circle."
    " If the action involves swiping, the start and end points of the swipe are shown on Screen A with an arrow.\n\n"
)

LASTSTEP_IMG_PROMPT = (
    "You are given an image contains a screenshot from an Android navigation trajectory."
    " The screenshot shows the final state of the trajectory.\n\n"
)

LLAMA_PROMPT = (
    "<|begin_of_text|>"
    "<|start_header_id|>system<|end_header_id|>"
    "{system_prompt}<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>"
    "{user_prompt}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>"
)


def get_full_trajectory_summary(
    summaries: Optional[list[str]] = None,
) -> str:

    if summaries:
        steps_details = "\n".join(
            f"Step {step_num + 1}{' (final step)' if step_num == len(summaries) - 1 else ''}: {summary}"
            for step_num, summary in enumerate(summaries)
        )
        steps_details = f"The agent took {len(summaries)} actions. Here is a history of what the agent has done:\n{steps_details}\n\n"
    else:
        steps_details = ""

    return steps_details


def get_prev_step_summary(
    summaries: Optional[list[str]] = None,
) -> str:

    if summaries:
        steps_details = "\n".join(
            f"Step {step_num + 1}: {summary}"
            for step_num, summary in enumerate(summaries)
        )
        steps_details = f"The agent took {len(summaries)} actions. Here is a history of what the agent has done:\n{steps_details}\n\n"
    else:
        steps_details = "This is the first step of the trajectory and there is no previous action.\n\n"

    return steps_details


def get_initial_and_final_xml(initial_state: str, final_state: str) -> str:
    xml_string = (
        "Here are the list of UI elements extracted from the initial and final state of trajectory.\n"
        "Initial State:\n"
        f"{initial_state}"
        "Final State:\n"
        f"{final_state}"
        "The list of UI elements may be incomplete and cannot accurately reflect the initial and final state."
        " You should focus on the screenshot if provided and only use the list of UI elements as complementary information.\n"
    )
    return xml_string


def get_previous_and_next_xml(
    before_elements: str, after_elements: Optional[str]
) -> str:
    if after_elements is None:
        xml_string = (
            "Here is the list of UI elements extracted from screenshot:\n"
            f"{before_elements}"
            "The list of UI elements may be incomplete and cannot accurately reflect the screenshot image."
            " You should focus on the screenshot image and only use the list of UI elements as complementary information.\n"
        )
    else:
        xml_string = (
            "Here are the list of UI elements extracted from screen A and screen B.\n"
            "screen A:\n"
            f"{before_elements}"
            "screen B:\n"
            f"{after_elements}"
            "The list of UI elements may be incomplete and cannot accurately reflect the changes in the screenshot image."
            " You should focus on the screenshot image if provided and only use the list of UI elements as complementary information.\n"
        )
    return xml_string


def get_trajectory_system_prompt(use_xml) -> str:
    system_prompt = TRAJECTORY_EVALUATION_SYSTEM_MESSAGE

    if not use_xml:
        system_prompt = system_prompt.replace(
            ", the initial and final states of the android system", ""
        )

    return system_prompt
