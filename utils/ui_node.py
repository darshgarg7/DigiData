# Copyright (c) Meta Platforms, Inc. and affiliates.

from __future__ import annotations

import math
import json
import re
import xmltodict

from pydantic import BaseModel
from typing import Any, Callable, Dict, ForwardRef, List, Optional, Sequence, Tuple


VALUE_XML_NAME_TO_FIELD_NAME = {
    "text": "text",
    "resource-id": "viewIdResourceName",
    "class": "className",
    "package": "packageName",
    "content-desc": "contentDescription",
    "content-desc": "contentDescription",
    "window-id": "windowId",
}

BOOLEAN_XML_NAME_TO_FIELD_NAME = {
    "checkable": "isCheckable",
    "checked": "isChecked",
    "clickable": "isClickable",
    "focusable": "isFocusable",
    "focused": "isFocused",
    "scrollable": "isScrollable",
    "long-clickable": "isLongClickable",
    "password": "isPassword",
    "selected": "isSelected",
    "editable": "isEditable",
}

UIForward = ForwardRef("UINode")

class UINode(BaseModel):
    packageName: Optional[str]
    className: Optional[str] = None
    viewIdResourceName: Optional[str] = None

    text: Optional[str] = None
    contentDescription: Optional[str] = None

    isCheckable: Optional[bool] = None
    isChecked: Optional[bool] = None
    isClickable: Optional[bool] = None
    isFocusable: Optional[bool] = None
    isFocused: Optional[bool] = None
    isScrollable: Optional[bool] = None
    isLongClickable: Optional[bool] = None
    isPassword: Optional[bool] = None
    isSelected: Optional[bool] = None
    isEditable: Optional[bool] = None
    windowId: Optional[int] = None
    boundsInScreen: Optional[List[int | float]] = None
    children: Optional[List["UINode"]] = None

    def to_dict(self) -> Dict[str, Any]:
        return (
            self.model_dump(exclude_none=True)
            if hasattr(self, "model_dump")
            else self.dict(exclude_none=True)
        )

    __hash__ = object.__hash__


def is_interactable(node: UINode) -> bool | None:
    return (
        node.isCheckable
        or node.isChecked
        or node.isClickable
        or node.isFocusable
        or node.isFocused
        or node.isScrollable
        or node.isLongClickable
        or node.isPassword
        or node.isSelected
        or node.isEditable
    )


def is_text_important(node: UINode) -> str | None:
    return node.text or node.contentDescription or node.viewIdResourceName


def prune_fields(node: UINode) -> UINode | None:
    new_children = []

    children = node.children
    if children:
        for child in children:
            new_child = prune_fields(child)
            if new_child:
                new_children.append(new_child)
    if new_children:
        node.children = new_children
    else:
        node.children = None
    node_is_interactable = is_interactable(node)

    if not node_is_interactable:
        node.boundsInScreen = None
    if node_is_interactable or new_children or is_text_important(node):
        return node

    return node


def prep_xml_formatting(
    json_node: Dict[str, Any], node_index: int = 0, level: int = 0
) -> Dict[str, Any]:
    def add_boolean(node: Dict[str, Any], xml_name: str, field_name: str) -> None:
        if field_name in json_node and json_node[field_name]:
            node[f"@{xml_name}"] = True

    def add_value(node: Dict[str, Any], xml_name: str, field_name: str) -> None:
        if field_name in json_node and json_node[field_name]:
            node[f"@{xml_name}"] = json_node[field_name]

    node = {}

    for xml_name, field_name in VALUE_XML_NAME_TO_FIELD_NAME.items():
        add_value(node, xml_name, field_name)

    for xml_name, field_name in BOOLEAN_XML_NAME_TO_FIELD_NAME.items():
        add_boolean(node, xml_name, field_name)

    if "boundsInScreen" in json_node:
        # this changed a little - instead of [left, top][right, bottom] it is [left, top, right, bottom] and the values are relative
        node["@bounds"] = (
            f'[{json_node["boundsInScreen"][0]},{json_node["boundsInScreen"][1]},'
            + f'{json_node["boundsInScreen"][2]},{json_node["boundsInScreen"][3]}]'
        )

    if "children" in json_node and len(json_node["children"]) > 0:
        node["node"] = [
            prep_xml_formatting(child, index, level + 1)
            for index, child in enumerate(json_node["children"])
        ]

    return node


def unprep_xml_formatting(xml_node: Dict[str, Any]) -> Dict[str, Any]:
    node = {}

    for xml_name, field_name in {
        **VALUE_XML_NAME_TO_FIELD_NAME,
        **BOOLEAN_XML_NAME_TO_FIELD_NAME,
    }.items():
        new_xml_name = f"@{xml_name}"
        if new_xml_name in xml_node:
            node[field_name] = xml_node[new_xml_name]

    bounds_name = "@bounds"
    if bounds_name in xml_node:
        node["boundsInScreen"] = [
            float(x) if "." in x else int(x)
            for x in re.findall(r"[\d\.]+", xml_node[bounds_name])
        ]

    if "node" not in xml_node or xml_node["node"] is None:
        return node

    if isinstance(xml_node["node"], list):
        node["children"] = [unprep_xml_formatting(child) for child in xml_node["node"]]
    else:
        node["children"] = [unprep_xml_formatting(xml_node["node"])]

    return node


def prepped_json_to_xml(
    prepped_json: List[Dict[str, Any]], pretty: bool = False, react_style: bool = False
) -> str:
    xml = xmltodict.unparse(
        {
            "heirarchy": {
                "@rotation": 0,
                "node": prepped_json,
            }
        },
        pretty=pretty,
        encoding="utf-8",
        short_empty_elements=True,
    )

    return xml.replace('="True"', "") if react_style else xml


def ui_json_to_pruned_ui_nodes(ui_json_raw: str) -> List[UINode]:
    ui_json = json.loads(ui_json_raw)
    return [
        pruned_node
        for pruned_node in [prune_fields(UINode(**ui_node)) for ui_node in ui_json]
        if pruned_node is not None
    ]


def ui_json_to_xml(
    ui_json: str, pretty: bool = False, react_style: bool = False, prune: bool = True
) -> str:
    """
    Converts a UI JSON string to an XML string.
    1. String -> List[Dict]
    2. List[Dict] -> List[UINode]
    3. List[UINode] => List[Dict] (pruned)
    4. List[Dict] pruned => XML
    """
    ui_dicts: List[Dict[str, Any]] = json.loads(ui_json)

    prepped_dict = [
        prep_xml_formatting(ui_node.to_dict())
        for ui_node in [
            prune_fields(UINode(**ui_dict)) if prune else UINode(**ui_dict)
            for ui_dict in ui_dicts
        ]
        if ui_node is not None
    ]

    return prepped_json_to_xml(list(prepped_dict), pretty)


def ui_json_to_xml_pruned(
    ui_json: str, pretty: bool = False, react_style: bool = False
) -> str:
    return ui_json_to_xml(ui_json, pretty, react_style, True)


def ui_json_to_xml_raw(
    ui_json: str, pretty: bool = False, react_style: bool = False
) -> str:
    return ui_json_to_xml(ui_json, pretty, react_style, False)


def ui_nodes_to_xml(
    ui_nodes: Sequence[UINode], pretty: bool = False, react_style: bool = False
) -> str:
    prepped_dict = [prep_xml_formatting(ui_node.to_dict()) for ui_node in ui_nodes]
    return prepped_json_to_xml(list(prepped_dict), pretty, react_style)


def ui_nodes_to_xml_react(ui_nodes: Sequence[UINode], pretty: bool = False) -> str:
    return ui_nodes_to_xml(ui_nodes, pretty, True)


def xml_to_ui_nodes(xml: str) -> List[UINode]:
    xml_dict = xmltodict.parse(xml)

    ui_nodes = xml_dict["heirarchy"]["node"]
    if isinstance(ui_nodes, dict):
        ui_nodes = [ui_nodes]

    return [UINode(**unprep_xml_formatting(ui_node)) for ui_node in ui_nodes]


def get_min_max_coords_for_nodes(
    nodes: List[UINode],
) -> Tuple[float, float, float, float]:
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")

    for node in nodes:
        new_min_x, new_min_y, new_max_x, new_max_y = get_min_max_coords(node)

        min_x = min(min_x, new_min_x)
        min_y = min(min_y, new_min_y)
        max_x = max(max_x, new_max_x)
        max_y = max(max_y, new_max_y)
    return min_x, min_y, max_x, max_y


def get_min_max_coords(node: UINode) -> Tuple[float, float, float, float]:
    """
    Return x_min, y_min, x_max, y_max of children
    """
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")

    if node.children is not None:
        for child in node.children:
            child_min_x, child_min_y, child_max_x, child_max_y = get_min_max_coords(
                child
            )
            min_x = min(min_x, child_min_x)
            max_x = min(max_x, child_max_x)
            min_y = max(min_y, child_min_y)
            max_y = max(max_y, child_max_y)

    if node.boundsInScreen is None:
        return min_x, min_y, max_x, max_y

    left, top, right, bottom = node.boundsInScreen

    return (
        min(min_x, left, right),
        min(min_y, top, bottom),
        max(max_x, left, right),
        max(max_y, top, bottom),
    )


def normalize_ui_node_coords(
    node: UINode, min_x_px: float, min_y_px: float, max_x_px: float, max_y_px: float
) -> UINode:
    copy = node.copy()

    children_copy = []
    if node.children is not None:
        for child in node.children:
            child = normalize_ui_node_coords(
                child, min_x_px, min_y_px, max_x_px, max_y_px
            )
            children_copy.append(child)
    copy.children = children_copy

    if copy.boundsInScreen is None:
        return copy
    left, top, right, bottom = copy.boundsInScreen

    x_range = max_x_px - min_x_px
    y_range = max_y_px - min_y_px

    copy.boundsInScreen = [
        math.floor((left - min_x_px) / x_range * 1000) / 1000,
        math.floor((top - min_y_px) / y_range * 1000) / 1000,
        math.floor((right - min_x_px) / x_range * 1000) / 1000,
        math.floor((bottom - min_y_px) / y_range * 1000) / 1000,
    ]

    return copy


def prune_uninteresting(node: UINode) -> UINode | None:
    children = []
    if node.children is not None:
        for child in node.children:
            pruned_child = prune_uninteresting(child)
            if pruned_child is not None:
                children.append(pruned_child)
    if not is_text_important(node) and len(children) == 0:
        return None
    if len(children) > 0:
        node.children = children
    return node


def prune_uninteresting_nodes(nodes: Sequence[UINode]) -> Sequence[UINode]:
    pruned_nodes = [prune_uninteresting(n) for n in nodes]
    return [n for n in pruned_nodes if n is not None]


def is_scroll(node: UINode) -> bool:
    return node.isScrollable is True


def contains_fn(x: float, y: float) -> Callable[[UINode], bool]:
    def contains(node: UINode) -> bool:
        if node.boundsInScreen is None:
            return False
        left, top, right, bottom = node.boundsInScreen

        if x < left or x > right or y < top or y > bottom:
            return False
        return True

    return contains


def separation_tap_fn(x_norm: float, y_norm: float) -> Callable[[UINode], int]:
    """
    Return the node distance to include around a tap node
    """

    def get_degrees_separation(node_norm: UINode) -> int:
        if node_norm.boundsInScreen is None:
            return -1
        left, top, right, bottom = node_norm.boundsInScreen

        # Skip node if tap is outside of bounds

        if x_norm < left or x_norm > right or y_norm < top or y_norm > bottom:
            return -1
        area = (right - left) * (bottom - top)

        # Skip large nodes covering 85% of the screen (arbitrary)

        if area > 0.85:
            return -1
        return 1

    return get_degrees_separation


def separation_swipe_fn(
    x1_norm: float, y1_norm: float, x2_norm: float, y2_norm: float
) -> Callable[[UINode], int]:
    """
    Return the node distance to include around a swipe node
    """

    mid_x: float = (x1_norm + x2_norm) / 2
    mid_y: float = (y1_norm + y2_norm) / 2

    def get_degrees_separation(node_norm: UINode) -> int:
        if node_norm.boundsInScreen is None:
            return -1
        left, top, right, bottom = node_norm.boundsInScreen

        area = (right - left) * (bottom - top)
        # include node that is specifically contain the x1, y1

        if (
            x1_norm >= left
            and x1_norm <= right
            and y1_norm >= top
            and y1_norm <= bottom
        ):
            # scrollable must include this node

            if node_norm.isScrollable:
                return 1
            # Skip large nodes covering 85% of the screen

            if area > 0.85:
                return -1
            return 2
        # since it's a swipe, the midpoint might be important to see what's around it.

        if mid_x >= left and mid_y <= right and mid_x >= top and mid_y <= bottom:
            return 1
        # might be somewhat interesting where we ended the swipe

        if (
            x2_norm >= left
            and x2_norm <= right
            and y2_norm >= top
            and y2_norm <= bottom
        ):
            return 0
        return -1

    return get_degrees_separation


def get_swipe_direction(x1: float, y1: float, x2: float, y2: float) -> str:
    horizonal_delta = abs(x1 - x2)
    vertical_delta = abs(y1 - y2)
    if horizonal_delta > vertical_delta:
        if x1 > x2:
            return "left"
        else:
            return "right"
    else:
        if y1 > y2:
            return "up"
        else:
            return "down"
