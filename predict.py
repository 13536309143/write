import argparse
import json
from pathlib import Path
from PIL import Image
from handwriting.inference import Recognizer

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='识别一张手写单字图片，返回候选字。')
    parser.add_argument('image')
    parser.add_argument('--checkpoint',default='runs/mac/best.pt')
    parser.add_argument('--device',default='auto')
    parser.add_argument('--topk',type=int,default=5)
    parser.add_argument('--group',default='any',choices=['any','chinese','digits','letters','uppercase','lowercase'])
    parser.add_argument('--output')
    parser.add_argument('--allow-verification',action='store_true',help='仅用于测试流程，不代表已训练好')
    args=parser.parse_args()
    recognizer=Recognizer(args.checkpoint,args.device,args.allow_verification)
    with Image.open(args.image) as image:
        result=recognizer.predict(image,args.topk,args.group)
    text=json.dumps(result,ensure_ascii=False,indent=2)
    print(text)
    if args.output:
        Path(args.output).write_text(text,encoding='utf-8')
