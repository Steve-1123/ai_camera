from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.vision.scene_classifier import SceneClassifier


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run scene classification on one image.")
    parser.add_argument("image_path", type=Path)
    args = parser.parse_args()

    image = load_image(args.image_path)
    result = SceneClassifier().classify(image)

    print(f"image shape: {image.shape}")
    print(f"model_name: {result.model_name}")
    print(f"primary_category: {result.primary_category}")
    print("top 5 candidates:")
    for candidate in result.candidates[:5]:
        print(f"- {candidate.label}: {candidate.score:.3f}")


if __name__ == "__main__":
    main()
