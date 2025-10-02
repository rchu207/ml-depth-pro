#!/usr/bin/env python3
"""Sample script to run DepthPro.

Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""


import argparse
import logging
from pathlib import Path

import torch
import pnnx

from depth_pro import create_model_and_transforms

LOGGER = logging.getLogger(__name__)


def get_torch_device() -> torch.device:
    """Get the Torch device."""
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    return device


def run(args):
    """Run Depth Pro on a sample image."""
    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    # Trace model with FP32.
    LOGGER.info(f"Create model.")
    model_cpu, _ = create_model_and_transforms(
    )
    LOGGER.info(f"Evaluate model.")
    model_cpu.eval()
    LOGGER.info(f"Create example input.")
    input_shape = (1, 3, 1536, 1536)
    example_input = torch.rand(input_shape)
    LOGGER.info(f"Use pnnx to convert model.")
    pnnx.export(model_cpu, "dptpro.pt", example_input)
    LOGGER.info(f"Use pnnx to convert model - done.")


def main():
    """Run DepthPro inference example."""
    parser = argparse.ArgumentParser(
        description="Inference scripts of DepthPro with PyTorch models."
    )
    parser.add_argument(
        "-i", 
        "--image-path", 
        type=Path, 
        default="./data/test_input_image.jpg",
        help="Path to input image.",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=Path,
        default="./output",
        help="Path to store output files.",
    )
    parser.add_argument(
        "--skip-display",
        default=True,
        help="Skip matplotlib display.",
    )
    parser.add_argument(
        "-v", 
        "--verbose", 
        action="store_true", 
        help="Show verbose output."
    )
    
    run(parser.parse_args())


if __name__ == "__main__":
    main()
