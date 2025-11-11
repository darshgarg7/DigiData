# Copyright (c) Meta Platforms, Inc. and affiliates.

import base64
from io import BytesIO
from typing import NamedTuple, Optional

from PIL import Image

# Define the UIState NamedTuple with a key, an image (base64), and a string representing a UI tree
UIState = NamedTuple(
    "UIState", [("key", str), ("image", Optional[str]), ("ui_raw", Optional[str])]
)


def image_to_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    if image is not None:
        image.save(buffer, format="JPEG")
    return buffer.getvalue()

def image_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return img_str

def base64_to_image(base64_string) -> Image.Image:
    img_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(img_data))

def base64_to_bytes(base64_string):
    byte_data = base64.b64decode(base64_string)
    return byte_data
