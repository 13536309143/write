import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from handwriting.data import HandwritingDataset
from handwriting.images import GlyphTransform
from handwriting.inference import Recognizer
from handwriting.model import build_model
from handwriting.runtime import load_config, save_checkpoint, device_for
from app import create_app

ROOT=Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def test_splits_and_class_coverage(self):
        data=HandwritingDataset(ROOT/'data/processed','train')
        meta=data.metadata
        sets=[set(meta['casia_writers'][k]) for k in ('train','val','test')]
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        self.assertEqual(len(data.classes),7247)
        self.assertEqual(data.classes[:62],list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'))
        for split in ('train','val','test'):
            self.assertTrue(all(n>0 for n in meta['class_counts'][split]))
        val=HandwritingDataset(ROOT/'data/processed','val',limit=72470)
        self.assertEqual(len(set(val.labels)),7247)

    def test_polarity_and_empty_image(self):
        image=Image.new('L',(80,120),255)
        ImageDraw.Draw(image).line((15,95,40,20,65,95),fill=0,width=5)
        transform=GlyphTransform()
        torch.testing.assert_close(transform(image),transform(ImageOps.invert(image)))
        self.assertEqual(transform(image).shape,(1,128,128))
        with self.assertRaises(ValueError):transform(Image.new('L',(80,80),255))
        with self.assertRaises(ValueError):transform(Image.new('RGBA',(80,80),(0,0,0,0)))

    def test_real_glyph_learning_and_checkpoint(self):
        config=load_config(ROOT/'configs/mac.yaml')
        config['model']=copy.deepcopy(config['model'])
        config['model'].update(dropout=0,drop_path=0)
        device=device_for('auto')
        torch.manual_seed(17)
        data=HandwritingDataset(ROOT/'data/processed','train')
        selected=['A','a','0','2','国','好','中','人']
        labels=data.labels
        images=[];targets=[]
        for character in selected:
            label=data.classes.index(character)
            pos=int(np.flatnonzero(labels==label)[0])
            image,target=data[pos];images.append(image);targets.append(target)
        x=torch.stack(images).to(device);y=torch.tensor(targets,device=device)
        model=build_model(config,len(data.classes)).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=0.001)
        curve=[]
        for _ in range(25):
            optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.cross_entropy(model(x),y)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()))
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step();curve.append(float(loss.detach()))
        self.assertLess(curve[-1],curve[0]/3)
        self.assertLess(curve[-1],2.0)
        model.eval()
        with torch.inference_mode():reference=model(x).cpu()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint=Path(directory)/'test.pt'
            save_checkpoint(checkpoint,{'config':config,'classes':data.classes,'model':model.state_dict(),
                                        'ema':model.state_dict(),'verification_only':True})
            with self.assertRaises(ValueError):Recognizer(checkpoint,allow_verification=False)
            recognizer=Recognizer(checkpoint,allow_verification=True)
            with torch.inference_mode():restored=recognizer.model(x).cpu()
            torch.testing.assert_close(reference,restored)
            image=data.raw_image(int(np.flatnonzero(labels==data.classes.index('国'))[0]))[0]
            result=recognizer.predict(image,group='chinese')
            self.assertEqual(len(result['candidates']),5)
            self.assertTrue(result['verification_only'])
            # Exercise the same upload path using a disposable checkpoint. This
            # tiny memorization model is never retained as a trained deliverable.
            save_checkpoint(checkpoint,{'config':config,'classes':data.classes,
                                        'ema':model.state_dict(),'verification_only':False})
            client=create_app(checkpoint).test_client()
            upload=io.BytesIO();image.save(upload,format='PNG');upload.seek(0)
            response=client.post('/predict',data={'image':(upload,'glyph.png'),'group':'chinese'})
            self.assertEqual(response.status_code,200)
            self.assertEqual(len(response.json['candidates']),5)
            (ROOT/'work').mkdir(exist_ok=True)
            (ROOT/'work/real_glyph_learning.json').write_text(json.dumps({'device':str(device),'loss_curve':curve,
                'first_loss':curve[0],'last_loss':curve[-1],'purpose':'8 training glyphs memorization / gradient verification, not test-set accuracy'},ensure_ascii=False,indent=2),encoding='utf-8')

    def test_upload_errors_and_no_model(self):
        with tempfile.TemporaryDirectory() as directory:
            app=create_app(Path(directory)/'missing.pt')
            client=app.test_client()
            self.assertEqual(client.get('/').status_code,200)
            self.assertFalse(client.get('/status').json['ready'])
            self.assertEqual(client.post('/predict').status_code,400)
            self.assertEqual(client.post('/predict',data={'image':(io.BytesIO(b'not an image'),'x.png')}).status_code,400)
            image=Image.new('L',(50,50),255);b=io.BytesIO();image.save(b,format='PNG');b.seek(0)
            self.assertEqual(client.post('/predict',data={'image':(b,'x.png')}).status_code,503)


if __name__=='__main__':unittest.main()
