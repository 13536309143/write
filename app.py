"""Local-only image upload demo; uses your trained checkpoint."""
import argparse
import io
import threading
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from flask import Flask, request, jsonify, render_template_string
from handwriting.inference import Recognizer

HTML='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>手写单字识别</title>
<style>body{font-family:system-ui,sans-serif;background:#f5f4ef;color:#202d27;margin:0}main{max-width:760px;margin:70px auto;padding:28px}h1{font-size:38px}p{color:#52645b;line-height:1.7}.card{background:white;padding:30px;border-radius:20px;box-shadow:0 8px 35px #26382b0b}input,select,button{font:inherit;margin:8px 0}button{background:#286a50;color:white;border:0;padding:12px 24px;border-radius:10px;cursor:pointer}button:disabled{opacity:.5}select{padding:10px;border-radius:8px}#preview{max-width:220px;max-height:220px;display:none;border:1px solid #ddd;margin:20px 0}#character{font-size:84px;margin:15px 0}.row{display:flex;justify-content:space-between;padding:12px;border-bottom:1px solid #eee}.error{color:#b04435}small{color:#68786e}</style>
<main><p>你的手写识别模型</p><h1>这个字是什么？</h1><p>上传一张包含单个汉字、英文字母或数字的清晰图片。</p><div class="card"><input id="file" type="file" accept="image/*"><br>
<select id="group"><option value="any">汉字、字母和数字</option><option value="chinese">只看汉字</option><option value="digits">只看数字</option><option value="letters">只看字母</option></select><br><img id="preview" alt="待识别图片"><br><button id="submit">识别图片</button><p id="status"></p><div id="character"></div><div id="candidates"></div><small>候选分数供参考；相似字、模糊图片或字符范围外的内容可能识别错误。多字图片请先裁剪。</small></div></main>
<script>const $=id=>document.getElementById(id);let objectUrl; $('file').onchange=()=>{let f=$('file').files[0];if(f){if(objectUrl)URL.revokeObjectURL(objectUrl);objectUrl=URL.createObjectURL(f);$('preview').src=objectUrl;$('preview').style.display='block';$('character').textContent='';$('candidates').replaceChildren()}};
$('submit').onclick=async()=>{const f=$('file').files[0];if(!f){$('status').textContent='请先选择图片。';return}let body=new FormData();body.append('image',f);body.append('group',$('group').value);$('submit').disabled=true;$('status').className='';$('status').textContent='正在识别…';$('character').textContent='';$('candidates').replaceChildren();try{let r=await fetch('/predict',{method:'POST',body});let x=await r.json();if(!r.ok)throw Error(x.error||'识别失败');$('character').textContent=x.character;$('status').textContent=x.needs_review?'这些候选字比较接近，请核对结果。':'识别完成。';for(const c of x.candidates){let row=document.createElement('div');row.className='row';let a=document.createElement('span'),b=document.createElement('span');a.textContent=c.character;b.textContent=(c.score*100).toFixed(1)+'% 模型分数';row.append(a,b);$('candidates').append(row)}}catch(e){$('status').className='error';$('status').textContent=e.message}finally{$('submit').disabled=false}};
fetch('/status').then(r=>r.json()).then(x=>{if(!x.ready)$('status').textContent='还没有训练完成的模型。请先完成训练，再上传图片。'});</script></html>'''


def create_app(checkpoint='runs/mac/best.pt',device='auto'):
    app=Flask(__name__)
    app.config['MAX_CONTENT_LENGTH']=8*1024*1024
    app.json.ensure_ascii=False
    lock=threading.Lock()
    cache={'model':None,'mtime':None}
    path=Path(checkpoint)

    @app.get('/')
    def home():
        return render_template_string(HTML)

    @app.get('/status')
    def status():
        return jsonify({'ready':path.is_file()})

    @app.post('/predict')
    def predict():
        if 'image' not in request.files:
            return jsonify(error='请上传单字图片。'),400
        try:
            with Image.open(io.BytesIO(request.files['image'].read())) as image:
                if image.width*image.height>16_000_000:
                    raise ValueError('图片过大，请先裁剪为单字。')
                image.load()
                with lock:
                    if not path.is_file():
                        raise FileNotFoundError('尚未找到训练好的模型，请先完成训练。')
                    mtime=path.stat().st_mtime_ns
                    if cache['mtime']!=mtime:
                        cache['model']=Recognizer(path,device)
                        cache['mtime']=mtime
                    result=cache['model'].predict(image,group=request.form.get('group','any'))
            return jsonify(result)
        except FileNotFoundError as exc:
            return jsonify(error=str(exc)),503
        except (ValueError,UnidentifiedImageError,Image.DecompressionBombError) as exc:
            return jsonify(error=str(exc)),400
        except OSError:
            return jsonify(error='图片无法读取，请换一张 PNG 或 JPG 单字图片。'),400

    @app.errorhandler(413)
    def too_large(error):
        return jsonify(error='图片文件超过 8 MB，请裁剪或压缩后再上传。'),413

    return app


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',default='runs/mac/best.pt')
    parser.add_argument('--device',default='auto')
    parser.add_argument('--port',type=int,default=7860)
    args=parser.parse_args()
    create_app(args.checkpoint,args.device).run(host='127.0.0.1',port=args.port,debug=False)
