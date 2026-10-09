"""Check the GPU and copied dataset without starting training or writing checkpoints."""
import argparse
import json
import platform
import sys
from pathlib import Path
import torch
from handwriting.runtime import device_for, synchronize


def check(device_name, data_dir):
    device = device_for(device_name)
    # An actual kernel launch also detects GPUs unsupported by the installed wheel.
    a = torch.randn(32, 32, device=device)
    if not bool(torch.isfinite(a @ a).all()):
        raise RuntimeError('GPU calculation returned non-finite values.')
    synchronize(device)
    root = Path(data_dir).resolve()
    if not (root/'metadata.json').is_file() or not (root/'index.npy').is_file():
        raise FileNotFoundError('请复制完整 data/processed，或先运行 prepare_data.py。')
    metadata = json.loads((root/'metadata.json').read_text(encoding='utf-8-sig'))
    missing = [f['path'] for f in metadata['files'] if not (root/f['path']).is_file()]
    if missing:
        raise FileNotFoundError(f'处理数据缺少 {len(missing)} 个文件，例如 {missing[0]}；请同时复制 raw 目录。')
    if len(metadata['classes']) != 7247:
        raise ValueError('数据类别数量不符合 7247 类。')
    result = {'python':sys.version.split()[0],'platform':platform.platform(),
              'torch':str(torch.__version__),'cuda_runtime':torch.version.cuda,
              'device':str(device),'gpu_kernel_check':'passed','classes':len(metadata['classes']),
              'dataset':str(root),'splits':metadata['split_counts']}
    if device.type == 'cuda':
        result.update(gpu=torch.cuda.get_device_name(device),
                      gpu_memory_gib=round(torch.cuda.get_device_properties(device).total_memory/2**30,2),
                      bf16_supported=torch.cuda.is_bf16_supported())
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cuda',choices=['cuda','auto','cpu','mps'])
    parser.add_argument('--data-dir',default='data/processed')
    args = parser.parse_args()
    try:
        check(args.device,args.data_dir)
    except (RuntimeError,ValueError,OSError) as exc:
        parser.exit(2,f'环境检查失败：{exc}\n')
