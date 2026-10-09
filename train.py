"""Train the handwriting network from random initialization on CPU, MPS or CUDA."""
import argparse
import copy
import json
import math
import multiprocessing
import time
from pathlib import Path

if __name__ == '__main__':
    _import_started = time.monotonic()
    print('正在加载 PyTorch 和训练依赖，请稍候…', flush=True)
try:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from handwriting.data import HandwritingDataset
    from handwriting.metrics import evaluate
    from handwriting.model import build_model
    from handwriting.runtime import load_config, device_for, seed_all, synchronize, save_checkpoint, load_checkpoint, worker_seed
    from handwriting.runtime import precision_for, training_signature, restore_scaler, optimizer_update
except KeyboardInterrupt:
    if __name__ == '__main__':
        print('\n已取消依赖加载，训练尚未开始。', flush=True)
        raise SystemExit(130) from None
    raise
if __name__ == '__main__':
    print(f'依赖加载完成，用时 {time.monotonic() - _import_started:.1f} 秒。', flush=True)


def train(config, resume=None, smoke_steps=0):
    seed_all(config['seed'])
    device = device_for(config['device'])
    precision = precision_for(config, device)
    run = Path(config['run_dir']); run.mkdir(parents=True, exist_ok=True)
    if not resume and (run/'last.pt').exists():
        raise ValueError('This run already has a checkpoint. Use --resume or a new --run-dir.')
    print(f'训练设备：{device}；正在读取数据索引和划分…', flush=True)
    train_set = HandwritingDataset(config['data_dir'], 'train', config['image_size'], True,
                                   limit=128 if smoke_steps else config.get('train_limit'), seed=config['seed'])
    val_set = HandwritingDataset(config['data_dir'], 'val', config['image_size'],
                                 limit=16 if smoke_steps else config.get('validation_limit'), seed=config['seed'])
    classes = train_set.classes
    print(f'数据就绪：训练 {len(train_set):,} 张，验证 {len(val_set):,} 张，{len(classes):,} 类。正在初始化网络…', flush=True)
    model = build_model(config, len(classes)).to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    batch_size, accumulation = config['batch_size'], config['accumulation']
    labels = train_set.labels
    counts = np.bincount(labels, minlength=len(classes))
    weights = torch.from_numpy(np.maximum(counts[labels], 1).astype(np.float64) ** (-config['class_balance_power']))
    samples = min(config.get('samples_per_epoch', len(train_set)), len(train_set))
    if smoke_steps:
        samples = min(len(train_set), smoke_steps*batch_size)
        accumulation = 1
    batches_per_epoch = math.ceil(samples/batch_size)
    updates_per_epoch = math.ceil(batches_per_epoch/accumulation)
    epochs = 1 if smoke_steps else config['epochs']
    total_updates = updates_per_epoch*epochs
    warmup = min(total_updates-1, updates_per_epoch*config['warmup_epochs'])

    def lr_schedule(step):
        if step < warmup:
            return max(0.01, (step+1)/max(1,warmup))
        progress = min(1.0, (step-warmup)/max(1,total_updates-warmup))
        return 0.01+0.99*(1+math.cos(math.pi*progress))/2

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])
    # MPS/CPU use FP32. CUDA can use FP16 scaling or BF16 without scaling.
    use_amp = precision != 'fp32'
    amp_dtype = torch.bfloat16 if precision == 'bf16' else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=precision == 'fp16')
    start_epoch, resume_batch, global_step, best, stale = 0, 0, 0, -1.0, 0
    skipped_updates = 0
    state_config = training_signature(config)
    data_hash = train_set.metadata['index_sha256']
    if resume:
        print('正在恢复模型、优化器和训练进度…', flush=True)
        saved = load_checkpoint(resume)
        if saved['classes'] != classes or saved['data_hash'] != data_hash:
            raise ValueError('Checkpoint classes/data split do not match this dataset.')
        if training_signature(saved['training_config']) != state_config or saved['verification_only'] != bool(smoke_steps):
            raise ValueError('Resume settings differ from saved configuration. Use the same config.')
        model.load_state_dict(saved['model']); ema.load_state_dict(saved['ema'])
        optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        restore_scaler(scaler, saved.get('scaler', {}))
        start_epoch, resume_batch = saved['next_epoch'], saved['next_batch']
        global_step, best, stale = saved['global_step'], saved['best_macro_top1'], saved['stale_epochs']
        skipped_updates = saved.get('skipped_updates', 0)
    (run/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    parameters = sum(p.numel() for p in model.parameters())
    print(json.dumps({'event':'start','device':str(device),'precision':precision,'parameters':parameters,'classes':len(classes),
                      'train_samples_available':len(train_set),'sampled_per_epoch':samples,'validation_samples':len(val_set),
                      'effective_batch_size':batch_size*accumulation,'verification_only':bool(smoke_steps),
                      'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
                      'torch_version':str(torch.__version__),'cuda_version':torch.version.cuda},ensure_ascii=False),flush=True)

    def snapshot(next_epoch, next_batch, verification_only=bool(smoke_steps)):
        return {'format_version':1,'model':model.state_dict(),'ema':ema.state_dict(),'optimizer':optimizer.state_dict(),
                'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'config':config,
                'training_config':state_config,'classes':classes,'data_hash':data_hash,'next_epoch':next_epoch,
                'next_batch':next_batch,'global_step':global_step,'best_macro_top1':best,'stale_epochs':stale,
                'verification_only':verification_only,'skipped_updates':skipped_updates,'precision':precision}

    current_epoch, next_batch = start_epoch, resume_batch
    try:
        for epoch in range(start_epoch,epochs):
            current_epoch = epoch
            print(f'开始第 {epoch + 1}/{epochs} 轮，准备数据加载器（{config["workers"]} 个进程）…', flush=True)
            generator = torch.Generator().manual_seed(config['seed']+epoch)
            sampler = WeightedRandomSampler(weights, samples, replacement=True, generator=generator)
            loader = DataLoader(train_set,batch_size=batch_size,sampler=sampler,num_workers=config['workers'],
                                persistent_workers=False,pin_memory=device.type=='cuda',worker_init_fn=worker_seed,
                                generator=torch.Generator().manual_seed(config['seed']+epoch+10000))
            model.train(); optimizer.zero_grad(set_to_none=True)
            start = time.monotonic();seen, loss_sum = 0,0.0
            next_batch = resume_batch if epoch==start_epoch else 0
            for batch_index,(images,target) in enumerate(loader):
                if batch_index < next_batch:
                    continue
                images,target=images.to(device,non_blocking=device.type=='cuda'),target.to(device,non_blocking=device.type=='cuda')
                # Normalize the final partial accumulation window by its true length.
                window_start=(batch_index//accumulation)*accumulation
                window_size=min(accumulation,batches_per_epoch-window_start)
                with torch.autocast(device_type=device.type,dtype=amp_dtype,enabled=use_amp):
                    logits=model(images);raw_loss=loss_fn(logits,target);loss=raw_loss/window_size
                if not torch.isfinite(raw_loss):
                    raise RuntimeError('Non-finite loss. Last completed checkpoint remains available.')
                scaler.scale(loss).backward()
                loss_sum+=float(raw_loss.detach())*len(target);seen+=len(target)
                boundary=(batch_index+1)%accumulation==0 or batch_index+1==len(loader)
                if boundary:
                    applied=optimizer_update(model,optimizer,scaler,config['grad_clip'])
                    if applied:
                        scheduler.step();global_step+=1
                        with torch.no_grad():
                            for averaged,p in zip(ema.parameters(),model.parameters()):
                                averaged.lerp_(p.detach(),1-config['ema_decay'])
                            for averaged,b in zip(ema.buffers(),model.buffers()):
                                averaged.copy_(b)
                    else:
                        skipped_updates+=1
                        print(json.dumps({'event':'amp_overflow','epoch':epoch+1,'batch':batch_index+1,
                                          'scale':scaler.get_scale(),'skipped_updates':skipped_updates}),flush=True)
                    next_batch=batch_index+1
                    if applied and global_step%config['save_every_steps']==0:
                        save_checkpoint(run/'last.pt',snapshot(epoch,next_batch))
                if batch_index == 0 or (batch_index+1)%25==0 or batch_index+1==len(loader):
                    synchronize(device);elapsed=time.monotonic()-start
                    print(json.dumps({'event':'progress','epoch':epoch+1,'batch':batch_index+1,'batches':len(loader),
                                      'loss':round(loss_sum/max(seen,1),5),'learning_rate':optimizer.param_groups[0]['lr'],
                                      'global_step':global_step,'skipped_updates':skipped_updates,'images_per_second':round(seen/elapsed,2),
                                      'eta_minutes':round((len(loader)-batch_index-1)*batch_size/(seen/elapsed)/60,1)},ensure_ascii=False),flush=True)
            validation_loader=DataLoader(val_set,batch_size=batch_size,num_workers=config['workers'],pin_memory=device.type=='cuda')
            print(json.dumps({'event':'validation_start','epoch':epoch+1,'samples':len(val_set)}),flush=True)
            metrics=evaluate(ema,validation_loader,device,classes,progress=True)
            metric=metrics['macro_top1']
            improved=metric>best
            if improved:
                best,stale=metric,0
            else:
                stale+=1
            resume_batch=0;next_batch=0;current_epoch=epoch+1
            save_checkpoint(run/'last.pt',snapshot(epoch+1,0))
            if improved:
                save_checkpoint(run/'best.pt',snapshot(epoch+1,0))
            row={'epoch':epoch+1,'global_step':global_step,'train_loss':loss_sum/max(seen,1),'validation':metrics,
                 'verification_only':bool(smoke_steps),'skipped_updates':skipped_updates}
            with (run/'history.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(row,ensure_ascii=False)+'\n')
            print(json.dumps({'event':'epoch_complete','epoch':epoch+1,'val_top1':metrics['top1'],
                              'val_macro_top1':metric,'val_top5':metrics['top5'],'best_checkpoint':str(run/'best.pt')},ensure_ascii=False),flush=True)
            if stale>=config['patience']:
                break
    except KeyboardInterrupt:
        # next_batch points to the last applied optimizer update, so partial gradients
        # are discarded and those micro-batches are replayed when resuming.
        optimizer.zero_grad(set_to_none=True)
        save_checkpoint(run/'last.pt',snapshot(current_epoch,next_batch))
        print('已保存中断前的完整更新，可用 --resume 继续。',flush=True)
        return
    print(json.dumps({'event':'complete','checkpoint':str(run/'best.pt'),'verification_only':bool(smoke_steps)},ensure_ascii=False),flush=True)


if __name__=='__main__':
    multiprocessing.freeze_support()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/mac.yaml')
    parser.add_argument('--resume')
    parser.add_argument('--run-dir')
    parser.add_argument('--smoke-steps',type=int,default=0,help='流程验证；不生成可用于正式识别的模型')
    parser.add_argument('--workers',type=int)
    args=parser.parse_args()
    try:
        config=load_config(args.config)
        if args.run_dir:config['run_dir']=str(Path(args.run_dir).resolve())
        if args.workers is not None:config['workers']=args.workers
        train(config,args.resume,args.smoke_steps)
    except KeyboardInterrupt:
        print('\n已取消初始化，本次尚未进入训练循环。', flush=True)
        raise SystemExit(130) from None
