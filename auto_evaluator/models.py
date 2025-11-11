# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
import os
from typing import Optional

from auto_evaluator.utils import image_to_base64

from PIL import Image

logger = logging.getLogger("auto_evaluator.models")
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False

try:
    from openai import OpenAI
except ImportError:
    logger.info("OpenAI not found. Please install openai for using OpenAI API.")


try:
    import transformers
    from vllm import LLM, SamplingParams
except ImportError:
    logger.info(
        "vLLM or transformers not found. Please install vLLM or transformers for using vLLM models."
    )


class ModelWrapper:

    async def complete_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        image: Optional[list[tuple[str, Image.Image]]] = None,
    ) -> str:
        return ""


class vLLMLlamaSyncWrapper(ModelWrapper):
    def __init__(
        self,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        model_id: str,
        use_screenshot: bool,
    ):
        self.llm = LLM(
            model=model_id,
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=128000,
            tensor_parallel_size=8,
            limit_mm_per_prompt={"image": 2},
        )

        self.processor = transformers.AutoProcessor.from_pretrained(model_id)

        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.model_id = model_id
        self.use_screenshot = use_screenshot

    async def complete_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        image: Optional[list[tuple[str, Image.Image]]] = None,
    ) -> str:
        conversation = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_new_tokens,
        )

        if self.use_screenshot and image is not None:
            img_content = []
            for key_image_pair in image:
                if key_image_pair[0]:
                    img_content.append(
                        {"type": "text", "text": f"{key_image_pair[0]}: "}
                    )
                img_content.append({"type": "image"})

            conversation[1]["content"] = img_content + conversation[1]["content"]

            prompt = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )

            outputs = self.llm.generate(
                [
                    {
                        "prompt": prompt,
                        "multi_modal_data": {
                            "image": [key_image_pair[1] for key_image_pair in image]
                        },
                    }
                ],
                sampling_params=sampling_params,
            )
        else:
            outputs = self.llm.chat([conversation], sampling_params)

        response = outputs[0].outputs[0].text

        if "<|eot_id|>" in response:
            response = response.split("<|eot_id|>")[0]

        return response.strip()


class ApiWrapper(ModelWrapper):
    def __init__(
        self,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        model_id: str,
        use_screenshot: bool,
        api_key_name: str,
        base_url: str,
        **kwargs,
    ):
        client_args = {}
        if api_key_name != "":
            client_args["api_key"] = os.environ[api_key_name]
        if base_url != "":
            client_args["base_url"] = base_url
        self.client = OpenAI(**client_args)
        self.model_id = model_id
        self.temperature = temperature
        self.top_p = top_p
        self.use_screenshot = use_screenshot
        self.max_new_tokens = max_new_tokens

    def encode_images(self, images: list[tuple[str, Image.Image]]) -> list[dict]:
        encoded_images = []
        for name, image in images:
            encoded_images.append(
                {
                    "name": name,
                    "url": f"data:image/jpeg;base64,{image_to_base64(image)}",
                }
            )
        return encoded_images

    async def complete_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        image: Optional[list[tuple[str, Image.Image]]] = None,
    ) -> str:
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [],
            },
        ]

        if self.use_screenshot and image is not None:
            try:
                encoded_images = self.encode_images(image)
                for encoded_image in encoded_images:
                    conversation[1]["content"].append(
                        {"type": "text", "text": encoded_image["name"]}
                    )
                    conversation[1]["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {"url": encoded_image["url"]},
                        }
                    )
            except ValueError as e:
                logger.error(f"Invalid image input: {e}")
                raise

        conversation[1]["content"].append({"type": "text", "text": user_prompt})

        completion = self.client.chat.completions.create(
            model=self.model_id,
            messages=conversation,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_new_tokens,
        )
        return completion.choices[0].message.content.strip()


model_mapping = {
    "vllm_model": vLLMLlamaSyncWrapper,
    "api_model": ApiWrapper,
}
