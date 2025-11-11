# Copyright (c) Meta Platforms, Inc. and affiliates.

from abc import ABC
from typing import Any, Dict, List, Optional


class Action(ABC):
    function_name: str
    args: List[Any]
    feedback: Optional[str]
    info: Optional[Dict[str, Any]]
    session_id: str
    request_id: str
    generated_action: str

    def __init__(
        self,
        session_id: str,
        request_id: str,
        function_name: str,
        args: List[Any],
        generated_action: str,
        feedback: Optional[str] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.session_id = session_id
        self.request_id = request_id
        self.function_name = function_name
        self.args = args
        self.generated_action = generated_action
        self.feedback = feedback
        self.info = info

    def needs_feedback(self) -> bool:
        return False
    
    def is_end_action(self) -> bool:
        return self.function_name == "complete"

    def __str__(self):
        return self.generated_action