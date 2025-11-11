# Copyright (c) Meta Platforms, Inc. and affiliates.

import secrets
import string
import time

from PIL import Image, ImageDraw, ImageFont

from predictor.goal_payload import create_goal_payload


TASK_MAX_STEP = 30
PREWORK_CMD_SETTLE_TIME_IN_SEC = 3
CMD_SETTLE_TIME_IN_SEC = 12


def generate_session_id(length=20):
    """
    Generate a random session ID.
    Args:
        length (int): The length of the session ID. Defaults to 20.
    Returns:
        str: A random session ID.
    """
    characters = string.ascii_letters + string.digits
    return "".join(secrets.choice(characters) for _ in range(length))


class Task:
    def __init__(self, goal=None, app_name=None, app_id=None, task_dict=None):
        """
        Initialize a Task object.

        Args:
            goal (str): A string describing the task.
            app_name (str): The name of the app to be used for the task.
            app_id (str): The ID of the app to be used for the task.
            task_dict (dict): A dictionary containing task data.

        Raises:
            ValueError: If neither task_str nor task_dict is provided.
        """

        if not goal and not task_dict:
            raise ValueError("Either task_str or task_dict must be provided")

        if goal and task_dict:
            raise ValueError("Cannot provide both task_str and task_dict")

        if goal:
            self.goal = goal
            self.app_name = app_name
            self.app_id = app_id
        elif task_dict:
            self.goal = task_dict.get("goal")
            self.app_name = task_dict.get("app_name", None)
            self.app_id = task_dict.get("app_id", None)

        self.setup_commands = []
        self.prework_commands = self.create_prework_commands()
        self.task_max_step = TASK_MAX_STEP
        self.complete = False
        self.step_id = 0
        self.screenshots = []
        self.xmls = []
        self.actions = []

        self.session_id = generate_session_id()

        print(f"{str(self)} initialized")

    @classmethod
    def from_string(cls, goal):
        """
        Create a Task object from a string.

        Args:
            goal (str): A string describing the task.

        Returns:
            Task: A Task object initialized with the given string.
        """
        return cls(goal=goal)

    @classmethod
    def from_dict(cls, task_dict):
        """
        Create a Task object from a dictionary.

        Args:
            task_dict (dict): A dictionary containing task data.

        Returns:
            Task: A Task object initialized with the given dictionary.
        """
        return cls(task_dict=task_dict)

    def create_prework_commands(self):
        """
        Initialize the prework commands for the task.
        """
        return [
            f"am force-stop {self.app_id}",
            f"pm clear {self.app_id}",
            f"monkey -p {self.app_id} -c android.intent.category.LAUNCHER 1",
        ]

    def setup(self, env):
        """
        Run setup commands on the given environment.

        Args:
            env: An environment object that can run commands.

        Raises:
            RuntimeError: If any setup command fails.
        """
        for cmd in self.setup_commands:
            env.run_command(cmd)
            time.sleep(CMD_SETTLE_TIME_IN_SEC)

    def do_prework(self, env):
        """
        Run prework commands on the given environment.

        Args:
            env: An environment object that can run commands.

        Raises:
            RuntimeError: If any prework command fails.
        """
        print(f"Doing prework for {str(self)}")
        for cmd in self.prework_commands:
            env.run_adb_command(cmd)
            time.sleep(PREWORK_CMD_SETTLE_TIME_IN_SEC)

        # Leave enough time for the app to settle
        time.sleep(10)

    async def execute(self, env, predictor):
        """
        Execute the task on the given environment.

        Args:
            env: An environment object that can run commands.
        """
        if self.step_id < self.task_max_step:
            print(f"Executing step {self.step_id} of {self.task_max_step}")
            screenshot = env.get_screenshot()
            xml = env.get_xml()

            goal_payload = create_goal_payload(screenshot, self.goal, xml)
            action = await predictor.predict(goal_payload)
            print(f"- Predicted action: {action}")
            env.execute_action(action)
            self.screenshots.append(screenshot)
            self.xmls.append(xml if xml else "")
            self.actions.append(action)

            if action.is_end_action():
                self.complete = True

            self.step_id += 1
            if self.step_id == self.task_max_step:
                self.complete = True

    def is_complete(self):
        return self.complete

    def visualize_task_trajectory(self):
        total_width = (
            sum(img.width for img in self.screenshots)
            + (len(self.screenshots) - 1) * 10
        )
        max_height = max(img.height for img in self.screenshots)
        new_img = Image.new(
            "RGB", (total_width, max_height + 50), color=(128, 128, 128)
        )
        x = 0
        font = ImageFont.load_default(24)
        for i, img in enumerate(self.screenshots):
            print(type(img))
            new_img.paste(img, (x, (max_height - img.height) // 2))
            draw = ImageDraw.Draw(new_img)
            draw.text(
                (x, max_height + 10),
                str(self.actions[i]),
                font=font,
                fill=(255, 255, 255),
            )
            x += img.width + 10
        new_img.show()

    def __str__(self) -> str:
        return f"Task(goal={self.goal}, app_name={self.app_name})"
