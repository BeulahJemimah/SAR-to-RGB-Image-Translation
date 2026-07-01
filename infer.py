"""
GalaxEye SAR-to-EO Inference Script
Conforms to the I/O contract in Section 2.3 of the assignment:

  Input : directory of single-channel Sentinel-1 SAR (VV) patches,
          256x256, 8-bit PNG, dB-scaled and min-max normalised to [0, 255].
  Output: directory of generated 256x256 RGB PNG images,
          same filenames as the corresponding inputs.
  CLI   : python infer.py --input_dir <path> --output_dir <path> --weights <path/to/checkpoint>

Runs on a single GPU with <=16 GB VRAM (or CPU) and requires no internet access.
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


# ------------------------------------------------------------------
# Generator architecture (must match training exactly)
# ------------------------------------------------------------------

class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        ]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.d1 = UNetDown(1, 64, normalize=False)
        self.d2 = UNetDown(64, 128)
        self.d3 = UNetDown(128, 256)
        self.d4 = UNetDown(256, 512)
        self.d5 = UNetDown(512, 512)
        self.d6 = UNetDown(512, 512)
        self.d7 = UNetDown(512, 512)

        self.bottom = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(True),
        )

        # Decoder
        self.u1 = UNetUp(512, 512, dropout=True)
        self.u2 = UNetUp(1024, 512, dropout=True)
        self.u3 = UNetUp(1024, 512, dropout=True)
        self.u4 = UNetUp(1024, 512)
        self.u5 = UNetUp(1024, 256)
        self.u6 = UNetUp(512, 128)
        self.u7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)

        b = self.bottom(d7)

        u1 = self.u1(b)
        u1 = torch.cat([u1, d7], dim=1)
        u2 = self.u2(u1)
        u2 = torch.cat([u2, d6], dim=1)
        u3 = self.u3(u2)
        u3 = torch.cat([u3, d5], dim=1)
        u4 = self.u4(u3)
        u4 = torch.cat([u4, d4], dim=1)
        u5 = self.u5(u4)
        u5 = torch.cat([u5, d3], dim=1)
        u6 = self.u6(u5)
        u6 = torch.cat([u6, d2], dim=1)
        u7 = self.u7(u6)
        u7 = torch.cat([u7, d1], dim=1)

        return self.final(u7)


# ------------------------------------------------------------------
# Pre/post-processing (must mirror training normalisation exactly)
# ------------------------------------------------------------------

SAR_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])


def load_sar(path):
    """Load an 8-bit, dB-scaled, min-max normalised SAR PNG as a [-1, 1] tensor."""
    img = Image.open(path).convert("L")  # single-channel, matches training
    if img.size != (256, 256):
        img = img.resize((256, 256), Image.BICUBIC)
    tensor = SAR_TRANSFORM(img)  # (1, 256, 256), range [-1, 1]
    return tensor


def tensor_to_rgb_image(tensor):
    """Convert a generator output tensor in [-1, 1], shape (3, H, W), to a PIL RGB image."""
    tensor = tensor.detach().cpu().clamp(-1, 1)
    tensor = (tensor + 1.0) / 2.0  # -> [0, 1]
    array = (tensor.numpy().transpose(1, 2, 0) * 255.0).round().astype("uint8")
    return Image.fromarray(array, mode="RGB")


# ------------------------------------------------------------------
# Main inference routine
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SAR-to-EO inference (Pix2Pix generator)")
    parser.add_argument("--input_dir", type=str, required=True,
                         help="Directory of single-channel SAR PNG patches (256x256, 8-bit)")
    parser.add_argument("--output_dir", type=str, required=True,
                         help="Directory to write generated RGB PNG images")
    parser.add_argument("--weights", type=str, required=True,
                         help="Path to generator checkpoint (.pth)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.isdir(args.input_dir):
        print(f"ERROR: input_dir does not exist: {args.input_dir}")
        sys.exit(1)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load model ---
    generator = Generator().to(device)

    if not os.path.isfile(args.weights):
        print(f"ERROR: weights file not found: {args.weights}")
        sys.exit(1)

    state = torch.load(args.weights, map_location=device)
    # Support both a raw state_dict and a full training checkpoint dict
    if isinstance(state, dict) and "generator" in state:
        state = state["generator"]
    generator.load_state_dict(state)
    generator.eval()

    # --- Gather input files ---
    valid_ext = (".png", ".PNG")
    filenames = sorted(f for f in os.listdir(args.input_dir) if f.endswith(valid_ext))

    if not filenames:
        print(f"WARNING: no PNG files found in {args.input_dir}")
        sys.exit(0)

    print(f"Found {len(filenames)} input SAR tiles. Running inference...")

    with torch.no_grad():
        for i, fname in enumerate(filenames, 1):
            in_path = os.path.join(args.input_dir, fname)
            out_path = os.path.join(args.output_dir, fname)

            sar_tensor = load_sar(in_path).unsqueeze(0).to(device)  # (1, 1, 256, 256)
            fake_rgb = generator(sar_tensor)[0]                      # (3, 256, 256)

            out_img = tensor_to_rgb_image(fake_rgb)
            out_img.save(out_path)

            if i % 25 == 0 or i == len(filenames):
                print(f"  [{i}/{len(filenames)}] {fname} -> saved")

    print(f"Done. {len(filenames)} generated images written to {args.output_dir}")


if __name__ == "__main__":
    main()
