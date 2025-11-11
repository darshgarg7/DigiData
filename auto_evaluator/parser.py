# Copyright (c) Meta Platforms, Inc. and affiliates.

import argparse


def auto_evaluator_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="AutoEvaluator",
        choices=[
            "AutoEvaluator",
            "ScreenshotSummarization",
            "ReasonGenerator",
        ],
    )
    parser.add_argument(
        "--model_id",
        default="Llama-4-Scout-17B-16E-Instruct-FP8",
        help="Model name to either use locally or on an API.",
    )
    parser.add_argument(
        "--use_screenshot",
        action="store_true",
        help="use screenshot to describe the each state of the trajectory",
    )
    parser.add_argument(
        "--use_xml",
        action="store_true",
        help="use simplified xml to describe the each state of the trajectory",
    )
    parser.add_argument("--chunk_size", default=-1, type=int)
    parser.add_argument("--chunk_idx", default=-1, type=int)
    parser.add_argument(
        "--model_host_type",
        default="api_model",
        choices=[
            "vllm_model",
            "api_model"
        ],
    )
    parser.add_argument("--use_future_actions", action="store_true")
    parser.add_argument("--future_action_summary", default="", type=str)
    parser.add_argument("--output_name", default="debug")
    parser.add_argument(
        "--output_dir",
        default="./output",
    )
    args = parser.parse_args()
    return args
