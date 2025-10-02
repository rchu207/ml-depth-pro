#!/usr/bin/env python3
"""Sample script to run DepthPro.

Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""


import cv2
import argparse
import logging
from pathlib import Path

import numpy as np
import PIL.Image
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import (
    Compose,
    ConvertImageDtype,
    Lambda,
    Normalize,
    ToTensor,
)
from torch import nn
from depth_pro import create_model_and_transforms, load_rgb

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

    # Load model.
    model, _ = create_model_and_transforms(
        device=get_torch_device(),
        precision=torch.half,
    )
    model.eval()

    # Load image.
    filename = "data/test_input_image2.jpg"
    LOGGER.info(f"Loading image {filename} ...")
    raw_image = cv2.imread(filename)
    # Possible OK when use RGB.
    image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    transform = Compose(
        [
            ToTensor(),
            Lambda(lambda x: x.to(get_torch_device())),
            Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ConvertImageDtype(torch.half),
        ]
    )
    image = transform(image)

    f_px = None

    prediction = model.infer(image, f_px=f_px)
    depth = prediction["depth"].detach().cpu().numpy().squeeze()

    # if len(image.shape) == 3:
    #     image = image.unsqueeze(0)
    # _, _, _, width = image.shape

    # Run prediction.
    # canonical_inverse_depth, fov_deg = model.forward(image)
    # if f_px is None:
    #     f_px = 0.5 * width / torch.tan(0.5 * torch.deg2rad(fov_deg.to(torch.float)))
    #
    # inverse_depth = canonical_inverse_depth * (width / f_px)
    #
    # depth = 1.0 / torch.clamp(inverse_depth, min=1e-4, max=1e4)
    #
    # # Extract the depth and focal length.
    # depth = depth.squeeze().detach().cpu().numpy().squeeze()

    inverse_depth = 1 / depth
    # Visualize inverse depth instead of depth, clipped to [0.1m;250m] range for better visualization.
    max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
    min_invdepth_vizu = max(1 / 250, inverse_depth.min())
    inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (
        max_invdepth_vizu - min_invdepth_vizu
    )

    # Save as color-mapped "turbo" jpg image.
    output_file = "test_heatmap1.jpg"
    depth = (inverse_depth_normalized[..., :3] * 255).astype(
        np.uint8
    )
    PIL.Image.fromarray(depth).save(
        output_file, format="PNG", quality=100
    )

    # Save as color-mapped "turbo" jpg image.
    cmap = plt.get_cmap("turbo")
    color_depth = (cmap(inverse_depth_normalized)[..., :3] * 255).astype(
        np.uint8
    )
    color_map_output_file = "test_heatmap2.png"
    LOGGER.info(f"Saving color-mapped depth to: : {color_map_output_file}")
    PIL.Image.fromarray(color_depth).save(
        color_map_output_file, format="PNG", quality=100
    )

    LOGGER.info("Done predicting depth!")


def main():
    """Run DepthPro inference example."""
    parser = argparse.ArgumentParser(
        description="Inference scripts of DepthPro with PyTorch models."
    )
    parser.add_argument(
        "-i", 
        "--image-path", 
        type=Path, 
        default="./data/example.jpg",
        help="Path to input image.",
    )
    parser.add_argument(
        "-o",
        "--output-path",
        type=Path,
        help="Path to store output files.",
    )
    parser.add_argument(
        "--skip-display",
        action="store_true",
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
