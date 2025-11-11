# Copyright (c) Meta Platforms, Inc. and affiliates.

import base64
import dataclasses
import json
import logging
import math
import os
import random
import re
from io import BytesIO
from typing import Any, Optional, Union
from xml.etree.ElementTree import Element

import cv2

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("auto_evaluator.utils")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


TAP_CIRCLR_SIZE = 20
TAP_CIRCLR_LINE_WIDTH = 10
LONG_PRESS_CIRCLR_BASE_SIZE = 10
LONG_PRESS_CIRCLR_LINE_WIDTH = 4
ARROW_LINE_WIDTH = 5
DELTA_TOP = 100
SPACE_BETWEEN_HORIZONTAL_STITCH = 40
TITLE_FONT_SIZE = 40
KEYBOARD_PACKAGES_NAMES = {
    "com.google.android.inputmethod.latin",
    "com.android.inputmethod.latin",
}


@dataclasses.dataclass
class BoundingBox:
    """Class for representing a bounding box."""

    x_min: Union[float, int]
    x_max: Union[float, int]
    y_min: Union[float, int]
    y_max: Union[float, int]

    @property
    def center(self) -> tuple[float, float]:
        """Gets center of bounding box."""
        return (float(self.x_min) + float(self.x_max)) / 2.0, (
            float(self.y_min) + float(self.y_max)
        ) / 2.0

    @property
    def width(self) -> Union[float, int]:
        """Gets width of bounding box."""
        return float(self.x_max) - float(self.x_min)

    @property
    def height(self) -> Union[float, int]:
        """Gets height of bounding box."""
        return float(self.y_max) - float(self.y_min)

    @property
    def area(self) -> Union[float, int]:
        return float(self.width) * float(self.height)


@dataclasses.dataclass
class UIElement:
    """Represents a UI element."""

    text: Optional[str] = None
    content_description: Optional[str] = None
    class_name: Optional[str] = None
    bbox: Optional[BoundingBox] = None
    bbox_pixels: Optional[BoundingBox] = None
    hint_text: Optional[str] = None
    is_checked: Optional[bool] = None
    is_checkable: Optional[bool] = None
    is_clickable: Optional[bool] = None
    is_editable: Optional[bool] = None
    is_enabled: Optional[bool] = None  # whether user Interaction possible
    is_focused: Optional[bool] = None  # destination of keyboard events
    is_focusable: Optional[bool] = None
    is_long_clickable: Optional[bool] = None
    is_scrollable: Optional[bool] = None
    is_selected: Optional[bool] = None
    is_visible: Optional[bool] = None
    package_name: Optional[str] = None
    resource_name: Optional[str] = None
    tooltip: Optional[str] = None
    resource_id: Optional[str] = None

    def get_string(self) -> str:
        attributes = []
        # Iterate over all fields defined in the dataclass
        for field in dataclasses.fields(self):
            # Get the value of the field
            value = getattr(self, field.name)
            # If the value is not None, add it to the attributes list in the format 'name=value'
            if value is not None:
                attributes.append(f"{field.name}='{value}'")
        # Join all attributes with a comma and space, and enclose in parentheses
        return f"({', '.join(attributes)})"


@dataclasses.dataclass
class MinimalUIElement:
    """Represents a Minimal UI element that skips most of the features."""

    text: Optional[str] = None
    content_description: Optional[str] = None
    class_name: Optional[str] = None
    hint_text: Optional[str] = None
    resource_name: Optional[str] = None
    is_checked: Optional[bool] = None
    is_checkable: Optional[bool] = None
    is_selected: Optional[bool] = None

    def get_string(self) -> str:
        """Convert this UIElement into a str representation"""
        attributes = []
        # Iterate over all fields defined in the dataclass
        for field in dataclasses.fields(self):
            # Get the value of the field
            value = getattr(self, field.name)

            # If the value is not None,
            # add it to the attributes list in the format 'name=value'
            if value is not None:

                if field.name == "class_name" and value in {
                    "ImageView",
                    "TextView",
                    "FrameLayout",
                    "View",
                }:
                    continue
                elif field.name == "is_checkable":
                    continue
                elif field.name == "is_checked" and not self.is_checkable:
                    continue
                elif field.name == "is_selected" and value is False:
                    continue
                elif isinstance(value, str):
                    attributes.append(f"{field.name}='{value}'")
                else:
                    attributes.append(f"{field.name}={value}")

        # Join all attributes with a comma-space, and enclose in parentheses
        return f"({', '.join(attributes)})"

    @staticmethod
    def copy_from(
        element: UIElement,
    ) -> "MinimalUIElement":
        """Extract only the elements in this class from UIElement"""
        # Create a dictionary to hold the values of the fields to be copied
        field_values = {}
        # Iterate over all fields defined in the dataclass
        for field in dataclasses.fields(MinimalUIElement):
            # Get the value of the field from the source element
            value = getattr(element, field.name, None)
            # Add the field and its value to the dictionary
            field_values[field.name] = value
        # Create a new instance of MinimalUIElement using the field values
        return MinimalUIElement(**field_values)


@dataclasses.dataclass
class ShortUIElement:
    """Represents a Short UI element that skips some features."""

    text: Optional[str] = None
    content_description: Optional[str] = None
    class_name: Optional[str] = None
    bbox_pixels: Optional[BoundingBox] = None
    hint_text: Optional[str] = None
    package_name: Optional[str] = None
    resource_name: Optional[str] = None
    is_checked: Optional[bool] = None
    is_checkable: Optional[bool] = None
    is_clickable: Optional[bool] = None
    is_editable: Optional[bool] = None
    is_focused: Optional[bool] = None
    is_selected: Optional[bool] = None

    def get_string(self) -> str:
        """Convert this UIElement into a str representation"""
        attributes = []
        # Iterate over all fields defined in the dataclass
        for field in dataclasses.fields(self):
            # Get the value of the field
            value = getattr(self, field.name)

            # If the value is not None and not False,
            # add it to the attributes list in the format 'name=value'
            if value:
                if isinstance(value, str):
                    attributes.append(f"{field.name}='{value}'")
                else:
                    attributes.append(f"{field.name}={value}")

        # Join all attributes with a comma-space, and enclose in parentheses
        return f"({', '.join(attributes)})"

    @staticmethod
    def copy_from(element: UIElement) -> "ShortUIElement":
        """Extract only the elements in this class from UIElement"""
        # Create a dictionary to hold the values of the fields to be copied
        field_values = {}
        # Iterate over all fields defined in the dataclass
        for field in dataclasses.fields(ShortUIElement):
            # Get the value of the field from the source element
            value = getattr(element, field.name, None)
            # Add the field and its value to the dictionary
            field_values[field.name] = value
        # Create a new instance of ShortUIElement using the field values
        return ShortUIElement(**field_values)


def _normalize_bounding_box(
    node_bbox: BoundingBox,
    screen_width_height_px: tuple[int, int],
) -> BoundingBox:
    width, height = screen_width_height_px
    return BoundingBox(
        node_bbox.x_min / width,
        node_bbox.x_max / width,
        node_bbox.y_min / height,
        node_bbox.y_max / height,
    )


def _xml_nodes_to_ui_element(
    node: Any, screen_size: Optional[tuple[int, int]] = None
) -> UIElement:
    """Converts a node from an accessibility tree to a UIElement."""

    def text_or_none(text: Optional[str]) -> Optional[str]:
        """Returns None if text is None or 0 length."""
        return text if text else None

    bbox_pixels = None
    bbox_normalized = None
    if "bounds" in node.attrib:
        if "][" in node.attrib["bounds"]:
            top_left, bottom_right = node.attrib["bounds"][1:-1].split("][")
            x1, y1 = top_left.split(",")
            x2, y2 = bottom_right.split(",")
        else:
            x1, y1, x2, y2 = node.attrib["bounds"][1:-1].split(",")
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        bbox_pixels = BoundingBox(x1, x2, y1, y2)
        if screen_size is not None:
            bbox_normalized = _normalize_bounding_box(bbox_pixels, screen_size)

    return UIElement(
        text=text_or_none(node.attrib.get("text", None)),
        content_description=text_or_none(node.attrib.get("content-desc", None)),
        class_name=text_or_none(
            node.attrib.get("class", "")
            .replace("android.widget.", "")
            .replace("android.view.", "")
        ),
        bbox=bbox_normalized,
        bbox_pixels=bbox_pixels,
        # hint_text=text_or_none(node.hint_text),
        is_checked=node.attrib.get("checked", None) == "true",
        is_checkable=node.attrib.get("checkable", None) == "true",
        is_clickable=node.attrib.get("clickable", None) == "true",
        is_editable=node.attrib.get("editable", None) == "true",
        is_enabled=node.attrib.get("enabled", None) == "true",
        is_focused=node.attrib.get("focused", None) == "true",
        is_focusable=node.attrib.get("focusable", None) == "true",
        is_long_clickable=node.attrib.get("long-clickable", None) == "true",
        is_scrollable=node.attrib.get("scrollable", None) == "true",
        is_selected=node.attrib.get("selected", None) == "true",
        is_visible=True,
        package_name=text_or_none(node.attrib.get("package", None)),
        resource_name=text_or_none(
            node.attrib.get("resource-id", "").split(":id/")[-1]
        ),
    )


def validate_ui_element(
    ui_element: UIElement,
    screen_width_height_px: tuple[int, int],
) -> bool:
    """Used to filter out invalid UI element."""
    screen_width, screen_height = screen_width_height_px

    # Filters out invisible element and Maestro app related element
    if not ui_element.is_visible:
        return False
    elif ui_element.content_description == "Maestro notification: Andojo Service":
        return False

    # Filters out element with invalid bounding box.
    if ui_element.bbox_pixels is not None:
        bbox_pixels = ui_element.bbox_pixels
        x_min = int(bbox_pixels.x_min)
        x_max = int(bbox_pixels.x_max)
        y_min = int(bbox_pixels.y_min)
        y_max = int(bbox_pixels.y_max)

        if (
            x_min >= x_max
            or x_min >= screen_width
            or x_max <= 0
            or y_min >= y_max
            or y_min >= screen_height
            or y_max <= 0
        ):
            return False

    return True


def traverse_xml(
    root_node: Optional[Element] = None,
    screen_size: Optional[tuple[int, int]] = None,
    print_node: bool = False,
    indent: int = 4,
) -> tuple[list, list]:

    if root_node is None:
        return [], []

    # Print root_node details
    if print_node:
        print(" " * indent + f"Element: {root_node.tag}")
        for name, value in root_node.attrib.items():
            print(" " * indent + f"  Attribute - {name}: {value}")

    leaf_nodes = []
    interactable_nodes = []

    # Recursively parse children elements, and get their text
    children_text = []
    for child in root_node:
        children_leaf_nodes, children_interactable_nodes = traverse_xml(
            child, screen_size, print_node, indent + 4
        )
        leaf_nodes.extend(children_leaf_nodes)
        interactable_nodes.extend(children_interactable_nodes)
        if child.attrib.get(
            "class", ""
        ) == "android.widget.TextView" and child.attrib.get("text", ""):
            children_text.append(child.attrib.get("text", ""))
    children_text = ". ".join(children_text)

    # Check if the current node is interactable
    is_interactable = False
    # Skip keyboard keys
    if root_node.attrib.get("package", "") not in KEYBOARD_PACKAGES_NAMES:
        for field_name in [
            "checkable",
            "clickable",
            "editable",
            "long-clickable",
            "scrollable",
        ]:
            if root_node.attrib.get(field_name, "false") == "true":
                is_interactable = True
                break

    # If the current node is a leaf node or interactable, create a new UIElement object
    if len(root_node) == 0 or is_interactable:
        root_node_ui_element = _xml_nodes_to_ui_element(root_node, screen_size)
        # propogate text from children to parent if parent has no text
        if not root_node_ui_element.text and children_text:
            root_node_ui_element.text = children_text

        # If the current node is a leaf node, add it to the leaf_nodes list
        if len(root_node) == 0:
            leaf_nodes.append(root_node_ui_element)

        # If the current node is interactable, add it to the interactable_nodes list
        if is_interactable:
            interactable_nodes.append(root_node_ui_element)

    return leaf_nodes, interactable_nodes


def visualize_screenshot(
    image: Image.Image, elements: list[UIElement], action: str, save_path: str
):
    draw = ImageDraw.Draw(image)

    # Draw the elements on the image
    for element in elements:
        if element.bbox_pixels is not None:
            bbox_pixels = element.bbox_pixels
            x_min = int(bbox_pixels.x_min)
            x_max = int(bbox_pixels.x_max)
            y_min = int(bbox_pixels.y_min)
            y_max = int(bbox_pixels.y_max)

            # Draw a rectangle around the element
            draw.rectangle((x_min, y_min, x_max, y_max), outline="red", width=2)
            # Draw the text inside the element
            try:
                if element.text:
                    draw.text((x_min, y_min), element.text, fill="red")
                elif element.content_description:
                    draw.text((x_min, y_min), element.content_description, fill="red")
            except UnicodeEncodeError:
                pass

    draw_action_image_only(image, action)

    # Save the new image to a file
    image.save(save_path)


def generate_ui_elements_description_list_full(
    ui_elements: list[UIElement],
    screen_width_height_px: tuple[int, int],
    shorten_ui=False,
    progress_only=False,
) -> str:
    """Generate description for a list of UIElement using full information.

    Args:
      ui_elements: UI elements for the current screen.
      screen_width_height_px: Logical screen size.
      progress_only: If True, then shorten the UI elements significantly.

    Returns:
      Information for each UIElement.
    """
    ui_description_list = []
    keyboard_exists = False
    for ui_element in ui_elements:
        if validate_ui_element(ui_element, screen_width_height_px):
            if progress_only:
                # Skip keyboard keys as UI elements.
                if ui_element.package_name in KEYBOARD_PACKAGES_NAMES:
                    keyboard_exists = True
                    continue
                elif any(
                    attr is not None
                    for attr in [
                        ui_element.text,
                        ui_element.content_description,
                        ui_element.hint_text,
                    ]
                ):
                    curr_string = MinimalUIElement.copy_from(ui_element).get_string()
                else:
                    continue
            elif shorten_ui:
                # Skip keyboard keys as UI elements.
                if ui_element.package_name in KEYBOARD_PACKAGES_NAMES:
                    keyboard_exists = True
                    continue
                curr_string = ShortUIElement.copy_from(ui_element).get_string()
            else:
                curr_string = str(ui_element)

            ui_description_list.append(curr_string)

    if (progress_only or shorten_ui) and keyboard_exists:
        ui_description_list.append("(Keyboard)")

    tree_info = ""
    ui_description_idx = 0
    ui_description_set = set()
    for ui_description in ui_description_list:
        if ui_description not in ui_description_set:
            tree_info += f"{ui_description_idx}: {ui_description}\n"
            ui_description_idx += 1
            ui_description_set.add(ui_description)

    return tree_info


def parse_action(
    action_dict: dict,
    interactable_nodes: list[UIElement],
    output_width_height: tuple[int, int],
    screen_width_height: tuple[int, int],
) -> str:
    """
    Parse the action string and return a more readable string.
    For swipe: return the direction of the swipe and the start and end coordinates (in output_width_height coordinate).
    For tap: return the coordinates of the tap (in output_width_height coordinate) and the text of the interactable node that contains the tap point.
    For type: return the text that was typed.
    For end: return the end state of the action.
    """

    raw_action = action_dict["raw_action"]
    parsed_action = raw_action

    if "action_type" in action_dict:
        if action_dict["action_type"] == "swipe":
            # get x, y in original screen size
            start_x = int(action_dict["start_x"] * screen_width_height[0])
            start_y = int(action_dict["start_y"] * screen_width_height[1])
            end_x = int(action_dict["end_x"] * screen_width_height[0])
            end_y = int(action_dict["end_y"] * screen_width_height[1])

            # determine the direction of the swipe by the angle of the vector
            angle = math.atan2(end_y - start_y, end_x - start_x) * 180 / math.pi
            if angle >= -45 and angle < 45:
                direct = "right"
            elif angle >= 45 and angle < 135:
                direct = "down"
            elif angle >= -135 and angle < -45:
                direct = "up"
            else:
                direct = "left"

            # scale coordinate to [0, output_width_height]
            start_x = int(action_dict["start_x"] * output_width_height[0])
            start_y = int(action_dict["start_y"] * output_width_height[1])
            end_x = int(action_dict["end_x"] * output_width_height[0])
            end_y = int(action_dict["end_y"] * output_width_height[1])
            parsed_action = (
                f"swipe {direct} from ({start_x}, {start_y}) to ({end_x}, {end_y})"
            )
        elif action_dict["action_type"] == "tap":

            # get x1, y1 in original screen size
            x1 = action_dict["x"] * screen_width_height[0]
            y1 = action_dict["y"] * screen_width_height[1]

            # find all interactable nodes that contain the tap point
            interactable_nodes_contain_tap = []
            for node in interactable_nodes:
                if (
                    node.bbox_pixels
                    and node.bbox_pixels.x_min <= x1
                    and node.bbox_pixels.x_max >= x1
                    and node.bbox_pixels.y_min <= y1
                    and node.bbox_pixels.y_max >= y1
                ):
                    interactable_nodes_contain_tap.append(node)

            associated_text = ""
            # select the smallest interactable node that contains the tap point
            if interactable_nodes_contain_tap:
                interactable_nodes_contain_tap = sorted(
                    interactable_nodes_contain_tap,
                    key=lambda node: node.bbox_pixels.area,
                )
                if interactable_nodes_contain_tap[0].text:
                    associated_text = interactable_nodes_contain_tap[0].text
                elif interactable_nodes_contain_tap[0].content_description:
                    associated_text = interactable_nodes_contain_tap[
                        0
                    ].content_description

            # scale coordinate to [0, output_width_height]
            x1 = int(action_dict["x"] * output_width_height[0])
            y1 = int(action_dict["y"] * output_width_height[1])
            if associated_text:
                parsed_action = (
                    f'tap({x1}, {y1}), likely tapping on "{associated_text}"'
                )
            else:
                parsed_action = f"tap({x1}, {y1})"
    else:
        if "end(" in raw_action:
            parsed_action = "Ending action"
            try:
                end_state_dict = json.loads(raw_action[4:-1])
                if (
                    "successOutput" in end_state_dict
                    and end_state_dict["successOutput"]
                ):
                    parsed_action += f" {end_state_dict['successOutput']}"
            except json.decoder.JSONDecodeError:
                pass
        elif "status(" in raw_action:
            # Change "status(complete)" in AitW to "Ending action",
            # otherwise the auto-eval would always determine the last step as achieving the goal
            parsed_action = "Ending action"

    return parsed_action


def extract_action_and_normalize_coordinates(
    raw_action: str, screen_width_height_px: tuple[int, int]
) -> dict:
    """
    Extract action type for swipe and tap, normalize action coordinates.
    """

    action_dict = {}
    if "swipe(" in raw_action:
        # coordinate for swipe is already normalized to [0, 1.0]
        pattern = r"swipe\((?P<start_x>\d+\.?\d*),\s*(?P<start_y>\d+\.?\d*),\s*(?P<end_x>\d+\.?\d*),\s*(?P<end_y>\d+\.?\d*)\)"
        match = re.search(pattern, raw_action)
        if match:
            action_dict["action_type"] = "swipe"
            action_dict["start_x"] = float(match.group("start_x"))
            action_dict["start_y"] = float(match.group("start_y"))
            action_dict["end_x"] = float(match.group("end_x"))
            action_dict["end_y"] = float(match.group("end_y"))
        else:
            logger.info(f"Failed to parse action: {raw_action}")
    elif "GestureDown(); Scroll" in raw_action:
        # coordinate for "GestureDown(); Scroll" is in original screen size
        pattern = r"start=\(x=(?P<start_x>\d+\.?\d*),y=(?P<start_y>\d+\.?\d*), end=\(x=(?P<end_x>\d+\.?\d*),y=(?P<end_y>\d+\.?\d*)\)"
        match = re.search(pattern, raw_action)
        if match:
            action_dict["action_type"] = "swipe"
            action_dict["start_x"] = (
                int(float(match.group("start_x"))) / screen_width_height_px[0]
            )
            action_dict["start_y"] = (
                int(float(match.group("start_y"))) / screen_width_height_px[1]
            )
            action_dict["end_x"] = (
                int(float(match.group("end_x"))) / screen_width_height_px[0]
            )
            action_dict["end_y"] = (
                int(float(match.group("end_y"))) / screen_width_height_px[1]
            )
        else:
            logger.info(f"Failed to parse action: {raw_action}")
    elif "tap(" in raw_action:
        # coordinate for tap is normalized to [0, 1.0]
        pattern = r"tap\((?P<x>\d+\.?\d*),\s*(?P<y>\d+\.?\d*)\)"
        match = re.search(pattern, raw_action)
        if match:
            action_dict["action_type"] = "tap"
            action_dict["x"] = float(match.group("x"))
            action_dict["y"] = float(match.group("y"))
        else:
            logger.info(f"Failed to parse action: {raw_action}")

    action_dict["raw_action"] = raw_action
    return action_dict


def stitch_images_horizontally(
    image1: Image.Image,
    image2: Image.Image,
) -> Image.Image:
    """
    Stitch two images together horizontally.
    Args:
        image1 (Image.Image): The first image.
        image2 (Image.Image): The second image.
    Returns:
        Image.Image: The stitched image.
    """

    # Calculate the width and height of the stitched image
    width = image1.width + image2.width + SPACE_BETWEEN_HORIZONTAL_STITCH
    height = max(image1.height, image2.height) + DELTA_TOP
    # Create a new image with the calculated width and height
    stitched_image = Image.new("RGB", (width, height), color="white")
    # Paste the first image onto mthe stitched image
    stitched_image.paste(image1, (0, DELTA_TOP))
    # Paste the second image onto the stitched image
    stitched_image.paste(
        image2, (image1.width + SPACE_BETWEEN_HORIZONTAL_STITCH, DELTA_TOP)
    )

    try:
        cv2_dir = os.path.dirname(cv2.__file__)
        font_path = os.path.join(cv2_dir, "qt", "fonts", "DejaVuSans.ttf")
        font_title = ImageFont.truetype(font_path, TITLE_FONT_SIZE)
    except OSError:
        font_title = ImageFont.load_default()

    draw = ImageDraw.Draw(stitched_image)

    # write "Screen A" title on the left image
    screenA_bbox = draw.textbbox((0, 0), "Screen A", font=font_title)
    sbox_width = screenA_bbox[2] - screenA_bbox[0]
    sbox_height = screenA_bbox[3] - screenA_bbox[1]
    pos_x = (image1.width - sbox_width) // 2
    pos_y = (DELTA_TOP - sbox_height) // 2
    draw.text((pos_x, pos_y), "Screen A", font=font_title, fill=(0, 0, 0))

    # write "Screen B" title on the left image
    screenB_bbox = draw.textbbox((0, 0), "Screen B", font=font_title)
    sbox_width = screenB_bbox[2] - screenB_bbox[0]
    sbox_height = screenB_bbox[3] - screenB_bbox[1]
    pos_x = (
        image1.width
        + SPACE_BETWEEN_HORIZONTAL_STITCH
        + (image2.width - sbox_width) // 2
    )
    pos_y = (DELTA_TOP - sbox_height) // 2
    draw.text((pos_x, pos_y), "Screen B", font=font_title, fill=(0, 0, 0))

    return stitched_image


def _draw_circle(draw, xy, size=10, width=5):
    """Draw a click on the image"""
    draw.ellipse(
        [xy[0] - size, xy[1] - size, xy[0] + size, xy[1] + size],
        outline=(0, 255, 0),
        width=width,
    )
    return draw


def _draw_arrow(draw, x0, y0, x1, y1, width=5):
    start = (x0, y0)
    end = (x1, y1)
    # Draw arrow body
    draw.line([start, end], fill=(0, 255, 0), width=width)
    # Calculate the angle of the line and the positions of the arrow head ends
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    arrow_head_length = 50
    arrow_angle = math.pi * 5 / 6
    arrow_end1 = (
        end[0] + arrow_head_length * math.cos(angle + arrow_angle),
        end[1] + arrow_head_length * math.sin(angle + arrow_angle),
    )
    arrow_end2 = (
        end[0] + arrow_head_length * math.cos(angle - arrow_angle),
        end[1] + arrow_head_length * math.sin(angle - arrow_angle),
    )
    # Draw arrow head ends
    draw.line([end, arrow_end1], fill=(0, 255, 0), width=width)
    draw.line([end, arrow_end2], fill=(0, 255, 0), width=width)
    return draw


def str_to_action(action_str):
    """Convert action string to dictionary"""
    action = {"str": action_str, "type": "", "params": {}}

    if action_str.startswith("tap"):
        action["type"] = "tap"
        x, y = action_str.split("(")[1].split(")")[0].split(",")
        action["params"]["x"] = float(x.strip())
        action["params"]["y"] = float(y.strip())
    elif action_str.startswith("swipe"):
        action["type"] = "swipe"
        coords = action_str[6:-1].split(", ")
        coords = [float(x) for x in coords]
        action["params"]["x0"] = coords[0]
        action["params"]["y0"] = coords[1]
        action["params"]["x1"] = coords[2]
        action["params"]["y1"] = coords[3]
    elif action_str.startswith("long_press"):
        action["type"] = "long_press"
        x, y = action_str.split("(")[1].split(")")[0].split(",")
        action["params"]["x"] = float(x.strip())
        action["params"]["y"] = float(y.strip())

    return action


def draw_action_image_only(img: Image.Image, action_str):
    draw = ImageDraw.Draw(img)
    width, height = img.size
    action = str_to_action(action_str)

    # Plot visualizable actions
    # Tap
    if action["type"] == "tap":
        x, y = int(action["params"]["x"] * width), int(action["params"]["y"] * height)
        tap_circle_size = max(int(width * 0.02), 1)
        tap_circle_line_width = max(int(width * 0.01), 1)
        _draw_circle(draw, (x, y), size=tap_circle_size, width=tap_circle_line_width)
    # Long press
    elif action["type"] == "long_press":
        x, y = int(action["params"]["x"] * width), int(action["params"]["y"] * height)
        long_press_circle_base_size = max(int(width * 0.01), 1)
        long_press_circle_line_width = max(int(width * 0.004), 1)
        _draw_circle(
            draw,
            (x, y),
            size=int(long_press_circle_base_size * 1.0),
            width=long_press_circle_line_width,
        )
        _draw_circle(
            draw,
            (x, y),
            size=int(long_press_circle_base_size * 2.0),
            width=long_press_circle_line_width,
        )
        _draw_circle(
            draw,
            (x, y),
            size=int(long_press_circle_base_size * 3.0),
            width=long_press_circle_line_width,
        )
    # Swipe
    elif action["type"] == "swipe":
        x0, y0 = (
            int(action["params"]["x0"] * width),
            int(action["params"]["y0"] * height),
        )
        x1, y1 = (
            int(action["params"]["x1"] * width),
            int(action["params"]["y1"] * height),
        )
        arraw_line_width = max(int(width * 0.005), 1)
        _draw_arrow(draw, x0, y0, x1, y1, width=arraw_line_width)


def image_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    if image.format:
        image_format = image.format
    else:
        image_format = "PNG"
    image.save(buffer, format=image_format)
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return img_str


def sample_steps(total_steps, num_keep_steps) -> list[int]:
    if num_keep_steps >= total_steps:
        return list(range(total_steps))

    # Always include the first and last steps
    sampled_steps = {0, total_steps - 1}

    # Calculate how many more steps need to be sampled
    additional_steps_needed = num_keep_steps - len(sampled_steps)

    if additional_steps_needed > 0:
        # Sample additional steps from the middle range
        middle_steps = range(1, total_steps - 1)
        sampled_steps.update(random.sample(middle_steps, additional_steps_needed))

    return sorted(sampled_steps)


def get_two_screenshots(
    raw_action: str = "",
    before_screenshot: Optional[Image.Image] = None,
    after_screenshot: Optional[Image.Image] = None,
    screenshot_names: list[str] = ["Screenshot"],
    return_two_screenshots: bool = True,
) -> Optional[list[tuple[str, Image.Image]]]:

    # draw action on before screenshot
    if before_screenshot is not None and raw_action != "":
        draw_action_image_only(before_screenshot, raw_action)

    if return_two_screenshots:
        if before_screenshot is not None and after_screenshot is not None:
            screenshot = [
                (screenshot_names[0], before_screenshot),
                (screenshot_names[1], after_screenshot),
            ]
        elif before_screenshot is not None:
            screenshot = [(screenshot_names[0], before_screenshot)]
        else:
            screenshot = None
    else:
        if before_screenshot is not None and after_screenshot is not None:
            screenshot = [
                (
                    screenshot_names[0],
                    stitch_images_horizontally(before_screenshot, after_screenshot),
                )
            ]
        elif before_screenshot is not None:
            screenshot = [(screenshot_names[0], before_screenshot)]
        else:
            screenshot = None

    return screenshot

