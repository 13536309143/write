import argparse
import json
import multiprocessing
from pathlib import Path
from torch.utils.data import DataLoader
from handwriting.data import HandwritingDataset
from handwriting.inference import Recognizer
from handwriting.metrics import evaluate
from handwriting.runtime import load_checkpoint

if __name__=='__main__':
    multiprocessing.freeze_support()
    parser=argparse.ArgumentParser(description='在独立测试集上评估，不用测试集挑选训练模型。')
    parser.add_argument('--checkpoint',default='runs/mac/best.pt')
    parser.add_argument('--data-dir',default='data/processed')
    parser.add_argument('--device',default='auto')
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--limit',type=int,help='仅抽样评估；不提供时评估完整测试集')
    parser.add_argument('--output',default='runs/mac/test_metrics.json')
    parser.add_argument('--allow-verification',action='store_true')
    args=parser.parse_args()
    recognizer=Recognizer(args.checkpoint,args.device,args.allow_verification)
    saved_config=load_checkpoint(args.checkpoint)
    data=HandwritingDataset(args.data_dir,'test',saved_config['config']['image_size'],limit=args.limit)
    if saved_config['data_hash']!=data.metadata['index_sha256'] or data.classes!=recognizer.classes:
        raise ValueError('测试数据或字符映射与训练模型不一致。')
    loader=DataLoader(data,batch_size=args.batch_size,num_workers=args.workers)
    metrics=evaluate(recognizer.model,loader,recognizer.device,recognizer.classes,progress=True)
    metrics['subset']=bool(args.limit)
    metrics['verification_only']=recognizer.verification_only
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in metrics.items() if k not in ('confusions','worst_classes')},ensure_ascii=False,indent=2))
