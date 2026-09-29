"""
Forces reproducible GPU inference (bit-exact across runs), at some cost to
speed. Import this FIRST in any entry-point script, before anything that
touches torch/CUDA (including `from ultralytics import YOLO`) - the
CUBLAS_WORKSPACE_CONFIG environment variable it sets only takes effect if
set before CUDA initializes, which happens the moment torch is imported.

Without this, tiny run-to-run floating-point differences in GPU ops can
flip a detection right at a threshold boundary (conf=0.5, iou=0.5) from
one run to the next - and because our scoring requires all 46 objects in
an image to agree simultaneously, that single flipped detection can flip
an entire image's verdict. Measured directly: 29% of images had their
verdict change between two runs of the IDENTICAL model, weights, and
settings, with nothing else different.

warn_only=True rather than strict mode: some GPU ops genuinely have no
deterministic implementation, and we want a warning for those, not a
crash - this fixes the common sources of noise (convolutions, the bulk
of what a detector/classifier actually runs) without risking the whole
pipeline failing over some obscure internal op.
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402 (must follow the env var above)

torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)
