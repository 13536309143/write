import json
import random
from pathlib import Path
import numpy as np
import torch
import yaml


def load_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
    root = path.parent.parent
    for key in ('data_dir', 'run_dir'):
        p = Path(config[key])
        config[key] = str(p if p.is_absolute() else root / p)
    return config


def device_for(name='auto'):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if name == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS 不可用，请检查 Apple 芯片和 PyTorch 安装。')
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA 不可用。')
    return torch.device(name)


def precision_for(config, device):
    precision = config.get('precision', 'auto')
    if precision == 'auto':
        precision = 'fp16' if device.type == 'cuda' else 'fp32'
    if precision not in ('fp32', 'fp16', 'bf16'):
        raise ValueError('precision must be auto, fp32, fp16 or bf16.')
    if precision != 'fp32' and device.type != 'cuda':
        raise ValueError('fp16/bf16 training requires CUDA; use fp32 on MPS/CPU.')
    if precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('This GPU does not support bf16; use fp16 or fp32.')
    return precision


def training_signature(config):
    # Runtime settings may change without changing model size or LR schedule.
    ignored = {'data_dir', 'run_dir', 'device', 'workers', 'precision'}
    return {k: v for k, v in config.items() if k not in ignored}


def restore_scaler(scaler, state):
    # MPS/FP32 checkpoints have an empty scaler state. Start a fresh scaler
    # when moving them to CUDA FP16; ignore saved scaling for FP32/BF16.
    if scaler.is_enabled() and state:
        scaler.load_state_dict(state)


def optimizer_update(model, optimizer, scaler, max_norm):
    """Return False when AMP skipped an overflowing update; FP32 errors remain fatal."""
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm,
                                         error_if_nonfinite=not scaler.is_enabled())
    if not torch.isfinite(norm) and scaler.is_enabled():
        # A norm can overflow even when each element is finite. That is not an
        # AMP-detected overflow and must not be treated as a successful update.
        if all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
            raise RuntimeError('Non-finite gradient norm with finite gradient elements.')
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    applied = not scaler.is_enabled() or scaler.get_scale() >= previous_scale
    optimizer.zero_grad(set_to_none=True)
    return applied


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def synchronize(device):
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()


def save_checkpoint(path, payload):
    path = Path(path)
    temp = path.with_suffix('.pt.part')
    torch.save(payload, temp)
    temp.replace(path)


def load_checkpoint(path):
    # Our checkpoints contain tensors and primitive metadata, not Python classes.
    return torch.load(path, map_location='cpu', weights_only=True)


def worker_seed(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
