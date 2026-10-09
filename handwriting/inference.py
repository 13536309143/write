from pathlib import Path
import torch
from .images import GlyphTransform
from .model import build_model
from .runtime import device_for, load_checkpoint


class Recognizer:
    def __init__(self, checkpoint, device='auto', allow_verification=False):
        path=Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError('尚未找到训练好的模型，请先完成训练：'+str(path))
        saved=load_checkpoint(path)
        self.verification_only=saved.get('verification_only',False)
        if self.verification_only and not allow_verification:
            raise ValueError('这是流程验证模型，不能用于正式识别；请先完成正式训练。')
        self.device=device_for(device)
        self.classes=saved['classes']
        self.model=build_model(saved['config'],len(self.classes)).to(self.device)
        self.model.load_state_dict(saved['ema'])
        self.model.eval()
        self.transform=GlyphTransform(saved['config']['image_size'])

    @torch.inference_mode()
    def predict(self, image, topk=5, group='any'):
        if image.width*image.height>16_000_000:
            raise ValueError('图片过大，请裁剪为单字后再上传。')
        x=self.transform(image).unsqueeze(0).to(self.device)
        logits=self.model(x).float()[0]
        scores=logits.softmax(-1).cpu()
        ranges={'any':range(len(self.classes)),'chinese':range(62,len(self.classes)),
                'digits':range(10),'letters':range(10,62),'uppercase':range(10,36),'lowercase':range(36,62)}
        if group not in ranges:
            raise ValueError('Unsupported character group.')
        ids=torch.tensor(list(ranges[group]),dtype=torch.long)
        ranked=scores[ids].topk(min(max(1,topk),len(ids)))
        candidates=[{'character':self.classes[int(ids[i])],'score':float(s)} for s,i in zip(ranked.values,ranked.indices)]
        margin=candidates[0]['score']-(candidates[1]['score'] if len(candidates)>1 else 0)
        return {'character':candidates[0]['character'],'candidates':candidates,
                'needs_review':candidates[0]['score']<0.6 or margin<0.15,
                'verification_only':self.verification_only,
                'note':'候选分数为未校准的模型分数，不保证等于真实正确率。请使用清晰的单字图片。'}
