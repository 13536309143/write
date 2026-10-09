"""Shared training/inference preprocessing. No flips: they change characters."""
import numpy as np
import torch
from PIL import Image, ImageOps, ImageFilter
from torchvision import transforms


def normalize_glyph(image, size=128, padding=0.12):
    image = ImageOps.exif_transpose(image)
    if image.mode in ('RGBA', 'LA') or 'transparency' in image.info:
        rgba = image.convert('RGBA')
        image = Image.alpha_composite(Image.new('RGBA', rgba.size, 'white'), rgba)
    image = image.convert('L')
    if image.width < 2 or image.height < 2:
        raise ValueError('图片太小，请提供清晰的单字图片。')
    a = np.asarray(image, dtype=np.float32)
    border = np.concatenate((a[0], a[-1], a[:, 0], a[:, -1]))
    if np.median(border) < 127:
        a = 255 - a
    # Preserve antialiased strokes; normalize modestly uneven paper backgrounds.
    low, high = float(a.min()), float(np.percentile(a, 95))
    if high - low < 12:
        raise ValueError('未检测到清晰笔画，请检查图片。')
    a = np.clip((a - low) * 255 / (high - low), 0, 255)
    mask = a < 210
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError('未检测到笔画。')
    crop = Image.fromarray(a[ys.min():ys.max()+1, xs.min():xs.max()+1].astype(np.uint8))
    inner = max(1, round(size * (1 - 2 * padding)))
    ratio = min(inner / crop.width, inner / crop.height)
    crop = crop.resize((max(1, round(crop.width * ratio)), max(1, round(crop.height * ratio))), Image.Resampling.BICUBIC)
    canvas = Image.new('L', (size, size), 255)
    canvas.paste(crop, ((size-crop.width)//2, (size-crop.height)//2))
    return canvas


class GlyphTransform:
    def __init__(self, size=128, training=False):
        self.size, self.training = size, training
        self.affine = transforms.RandomAffine(8, translate=(0.07, 0.07), scale=(0.92, 1.08), shear=5, fill=255,
                                             interpolation=transforms.InterpolationMode.BILINEAR)

    def __call__(self, image):
        image = normalize_glyph(image, self.size)
        if self.training:
            image = self.affine(image)
            if torch.rand(()).item() < 0.08:
                image = image.filter(ImageFilter.GaussianBlur(0.35))
        a = np.asarray(image, dtype=np.float32).copy()
        # White background -> -1; black ink -> +1. Sobel branch uses this tensor.
        return torch.from_numpy(1 - a / 127.5).unsqueeze(0)
