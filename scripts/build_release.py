"""Package a completed GlyphWeave experiment without including its dataset."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import yaml
from handwriting.model import build_model


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'runs/windows_cuda')
    parser.add_argument('--version', default='v1.0.0')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    output = (args.output or ROOT / 'work/releases' / args.version).resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit('Release output already contains files; choose another --output directory.')
    rows = [json.loads(line) for line in (run / 'history.jsonl').read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    test = json.loads((run / 'test_metrics.json').read_text(encoding='utf-8-sig'))
    best = max(rows, key=lambda row: row['validation']['macro_top1'])
    saved = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
    if saved.get('verification_only') or test.get('verification_only') or test.get('subset'):
        raise SystemExit('A diagnostic/subset result cannot be packaged as a complete experiment.')
    if test['classes_evaluated'] != len(saved['classes']) or len(saved['classes']) != 7247:
        raise SystemExit('Class count mismatch.')
    if saved['next_epoch'] != best['epoch'] or abs(saved['best_macro_top1'] - best['validation']['macro_top1']) > 1e-12:
        raise SystemExit('The best checkpoint does not match the best logged validation epoch.')
    output.mkdir(parents=True, exist_ok=True)
    portable = copy.deepcopy(saved['config'])
    portable['data_dir'] = 'data/processed'
    portable['run_dir'] = 'runs/windows_cuda'
    source_hash = sha256(run / 'best.pt')
    model = build_model(portable, len(saved['classes']))
    parameters = sum(p.numel() for p in model.parameters())
    del model
    common = {
        'project': 'GlyphWeave', 'version': args.version,
        'best_epoch': best['epoch'], 'epochs_completed': len(rows),
        'seed': portable['seed'], 'model_parameters': parameters,
        'classes': len(saved['classes']), 'index_sha256': saved['data_hash'],
        'source_checkpoint_sha256': source_hash,
        'training_code_commit': None,
        'test_report_provenance': 'Existing complete test report; original evaluation checkpoint identity, command, package versions, and timing were not recorded.',
        'test_report_sha256': sha256(run / 'test_metrics.json'),
        'history_sha256': sha256(run / 'history.jsonl'),
        'validation': {k: v for k, v in best['validation'].items() if k not in ('worst_classes', 'confusions')},
        'test': {k: v for k, v in test.items() if k not in ('worst_classes', 'confusions')},
    }
    assets = []
    for kind in ['inference', 'training']:
        folder = output / f'glyphweave-{args.version}-{kind}'
        folder.mkdir()
        if kind == 'inference':
            inference = {
                'format_version': saved['format_version'], 'checkpoint_type': 'inference_ema',
                'inference_only': True, 'release_version': args.version,
                'config': portable, 'classes': saved['classes'], 'ema': saved['ema'],
                'data_hash': saved['data_hash'], 'verification_only': False,
                'best_epoch': best['epoch'], 'source_checkpoint_sha256': source_hash,
            }
            torch.save(inference, folder / 'best.pt')
        else:
            shutil.copy2(run / 'best.pt', folder / 'best.pt')
        (folder / 'config.yaml').write_text(yaml.safe_dump(portable, sort_keys=False, allow_unicode=True), encoding='utf-8')
        (folder / 'classes.txt').write_text('\n'.join(saved['classes']) + '\n', encoding='utf-8')
        for name in ['history.jsonl', 'test_metrics.json']:
            shutil.copy2(run / name, folder / name)
        for name in ['experiment-summary.json', 'experiment-overview.png', 'experiment-overview.svg']:
            shutil.copy2(run / 'analysis' / name, folder / name)
        manifest = {**common, 'package_type': kind, 'checkpoint_sha256': sha256(folder / 'best.pt'),
                    'resume_supported': kind == 'training'}
        (folder / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        checkpoint = f'{folder.name}/best.pt'
        commands = f'```bash\n.venv/bin/python predict.py path/to/glyph.png --checkpoint {checkpoint}\n.venv/bin/python app.py --checkpoint {checkpoint}\n```'
        en = f'# GlyphWeave {args.version} — {kind} package\n\n'
        en += 'Clone the matching source release and install its dependencies, then extract this package into the project root. Single-image inference does not require the dataset.\n\n' + commands + '\n\n'
        en += 'On Windows, replace `.venv/bin/python` with `.\\.venv-win\\Scripts\\python.exe`. Complete evaluation and training require the entire `data/processed/` directory.\n\n'
        zh = f'# GlyphWeave {args.version} — {kind} 包\n\n'
        zh += '克隆对应版本的源码并安装依赖，将本包解压到项目根目录。单图识别不需要数据集。\n\n' + commands + '\n\n'
        zh += 'Windows 将 `.venv/bin/python` 替换为 `.\\.venv-win\\Scripts\\python.exe`。完整评估与训练需要整个 `data/processed/` 目录。\n\n'
        if kind == 'inference':
            en += 'This checkpoint contains EMA weights and inference metadata only. It cannot resume training.\n'
            zh += '此检查点只包含 EMA 权重与推理元信息，不能用于断点续训。\n'
        else:
            en += 'This is the original full best checkpoint, including raw weights, EMA, optimizer, scheduler, and AMP state. The completed 80-epoch run has no remaining scheduled epochs; extending its schedule requires a separately designed experiment. Not all RNG states were saved, so recovery is not bitwise reproducible.\n'
            zh += '这是原始完整最佳检查点，包含原模型、EMA、优化器、调度器与 AMP 状态。80 轮训练已完成，没有剩余计划轮次；延长调度需要另行设计实验。未保存全部 RNG 状态，恢复不保证逐位复现。\n'
        en += '\nOnly one seed was run. No baseline, ablations, calibrated confidence, timing, or external-photo benchmark is claimed. Check SHA256SUMS.txt before loading downloaded checkpoints.\n'
        zh += '\n仅完成单个随机种子实验，未宣称完成基线、消融、分数校准、计时或外部照片基准。加载下载的检查点前，请核对 SHA256SUMS.txt。\n'
        (folder / 'README.md').write_text(en, encoding='utf-8')
        (folder / 'README.zh-CN.md').write_text(zh, encoding='utf-8')
        archive = output / f'{folder.name}.zip'
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for file in sorted(folder.iterdir()):
                z.write(file, f'{folder.name}/{file.name}')
        assets.append({'name': archive.name, 'bytes': archive.stat().st_size, 'sha256': sha256(archive)})
        print(json.dumps(assets[-1]), flush=True)
    (output / 'release-manifest.json').write_text(json.dumps({**common, 'assets': assets}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    checksum_files = [output / item['name'] for item in assets] + [output / 'release-manifest.json']
    (output / 'SHA256SUMS.txt').write_text(''.join(f'{sha256(file)}  {file.name}\n' for file in checksum_files), encoding='utf-8')


if __name__ == '__main__':
    main()
