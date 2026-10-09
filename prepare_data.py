"""Extract raw GNT/IDX once; create a compact mmap index, without millions of PNGs."""
import argparse
import gzip
import hashlib
import json
import shutil
import struct
import time
import zipfile
from pathlib import Path
import numpy as np
from handwriting.data import RECORD


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare(data_root, output, seed=42, validation_fraction=0.1):
    data_root, output = Path(data_root).resolve(), Path(output).resolve()
    if (output / 'metadata.json').exists():
        metadata = json.loads((output / 'metadata.json').read_text(encoding='utf-8-sig'))
        if metadata['seed'] != seed or metadata['validation_fraction'] != validation_fraction:
            raise ValueError('Prepared data use different split settings. Select another --output directory.')
        if digest(output / 'index.npy') != metadata['index_sha256']:
            raise ValueError('Index integrity check failed.')
        print(json.dumps({'status':'already_prepared', 'classes':len(metadata['classes']), 'splits':metadata['split_counts']}, ensure_ascii=False))
        return
    casia = data_root / 'CASIA-HWDB'
    chinese = (casia / 'chinese_characters.txt').read_text(encoding='utf-8-sig').splitlines()
    classes = list('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz') + sorted(set(chinese))
    if len(classes) != 7247:
        raise ValueError('Expected 7,185 Chinese classes + 62 alphanumeric classes.')
    class_id = {c:i for i,c in enumerate(classes)}
    expected = ['Gnt1.0Test.zip'] + [f'Gnt1.0TrainPart{i}.zip' for i in (1,2,3)]
    for version in (1,2):
        expected += [f'Gnt1.{version}Test.zip'] + [f'Gnt1.{version}TrainPart{i}.zip' for i in (1,2)]
    for name in expected:
        if not (casia / name).is_file():
            raise FileNotFoundError(casia / name)
    output.mkdir(parents=True, exist_ok=True)
    files, chunks, writer_sets = [], [], {0:set(), 1:set(), 2:set()}
    val_writers = set()
    rng = np.random.default_rng(seed)
    for version in (0,1,2):
        writers = []
        for name in expected:
            if name.startswith(f'Gnt1.{version}Train'):
                with zipfile.ZipFile(casia/name) as z:
                    writers += [Path(i.filename).name.split('-')[0] for i in z.infolist() if i.filename.endswith('.gnt')]
        chosen = rng.choice(sorted(set(writers)), max(1, round(len(set(writers))*validation_fraction)), replace=False)
        val_writers.update(str(x) for x in chosen)
    tag_cache = {}
    excluded_empty = 0
    excluded_blank = 0
    start = time.monotonic()
    for name in sorted(expected):
        archive_records = 0
        with zipfile.ZipFile(casia / name) as z:
            for entry in z.infolist():
                if not entry.filename.endswith('.gnt'):
                    continue
                # Ignore archive paths; only use the checked basename inside our directory.
                basename = Path(entry.filename).name
                writer = basename.split('-')[0]
                split = 2 if 'Test' in name else 1 if writer in val_writers else 0
                writer_sets[split].add(writer)
                relative = Path('raw') / Path(name).stem / basename
                path = output / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists() or path.stat().st_size != entry.file_size:
                    temp = path.with_suffix('.gnt.part')
                    with z.open(entry) as source, temp.open('wb') as target:
                        shutil.copyfileobj(source, target, 4*1024*1024)
                    temp.replace(path)
                file_id = len(files)
                files.append({'path':str(relative), 'writer':writer, 'source':'CASIA', 'split':split})
                data = np.memmap(path, mode='r', dtype=np.uint8)
                offset, rows = 0, []
                while offset < len(data):
                    if offset+10 > len(data):
                        raise ValueError('Truncated GNT: '+str(path))
                    n, tag, width, height = struct.unpack_from('<I2sHH', data, offset)
                    if n != 10+width*height or offset+n > len(data) :
                        raise ValueError('Invalid GNT: '+str(path))
                    if not width or not height:
                        excluded_empty += 1
                        offset += n
                        continue
                    if tag not in tag_cache:
                        tag_cache[tag] = class_id.get(tag.rstrip(b'\0').decode('gb18030'), -1)
                    label = tag_cache[tag]
                    if label >= 0 and int(data[offset+10:offset+n].min()) >= 243:
                        excluded_blank += 1
                    elif label >= 0:
                        rows.append((file_id, offset+10, width, height, label, split, 0))
                    offset += n
                chunks.append(np.array(rows, dtype=RECORD))
                archive_records += len(rows)
                del data
        print(json.dumps({'archive':name, 'indexed_samples':archive_records, 'seconds':round(time.monotonic()-start)}, ensure_ascii=False), flush=True)
    if any(writer_sets[a] & writer_sets[b] for a,b in ((0,1),(0,2),(1,2))):
        raise ValueError('Writer leakage between CASIA splits.')
    emnist = data_root / 'EMNIST'
    mapping = dict(tuple(map(int,l.split())) for l in (emnist/'emnist-byclass-mapping.txt').read_text(encoding='utf-8-sig').splitlines())
    for split_name, expected_n in (('train',697932),('test',116323)):
        relative = Path('raw') / f'emnist-{split_name}.idx'
        path = output / relative
        if not path.exists() or path.stat().st_size != 16+expected_n*784:
            temp = path.with_suffix('.idx.part')
            with gzip.open(emnist/f'emnist-byclass-{split_name}-images-idx3-ubyte.gz','rb') as source, temp.open('wb') as target:
                shutil.copyfileobj(source,target,4*1024*1024)
            temp.replace(path)
        with path.open('rb') as f:
            magic,n,h,w = struct.unpack('>IIII',f.read(16))
        if (magic,n,h,w) != (2051,expected_n,28,28):
            raise ValueError('Incorrect EMNIST image header.')
        with gzip.open(emnist/f'emnist-byclass-{split_name}-labels-idx1-ubyte.gz','rb') as f:
            magic,m = struct.unpack('>II',f.read(8)); labels=np.frombuffer(f.read(),dtype=np.uint8)
        if magic != 2049 or m != n or len(labels) != n:
            raise ValueError('Incorrect EMNIST labels.')
        label_ids = np.array([class_id[chr(mapping[int(v)])] for v in labels], dtype=np.uint16)
        rows = np.zeros(n,dtype=RECORD)
        rows['file']=len(files); rows['offset']=16+np.arange(n,dtype=np.uint64)*784
        rows['width']=28; rows['height']=28; rows['label']=label_ids; rows['source']=1
        rows['split']=0 if split_name=='train' else 2
        if split_name=='train':
            for label in range(62):
                indices = np.flatnonzero(label_ids==label)
                chosen = rng.choice(indices,max(1,round(len(indices)*validation_fraction)),replace=False)
                rows['split'][chosen]=1
        files.append({'path':str(relative),'writer':None,'source':'EMNIST','split':'per_record'})
        chunks.append(rows)
    index=np.concatenate(chunks)
    for split in (0,1,2):
        labels=np.unique(index['label'][index['split']==split])
        if len(labels)!=7247:
            raise ValueError(f'Split {split} missing classes: {7247-len(labels)}')
    temp=output/'index.tmp.npy'; np.save(temp,index,allow_pickle=False);temp.replace(output/'index.npy')
    metadata={'format_version':1,'seed':seed,'validation_fraction':validation_fraction,'classes':classes,
              'index_sha256':digest(output/'index.npy'),'files':files,
              'casia_writers':{name:sorted(writer_sets[i]) for i,name in enumerate(('train','val','test'))},
              'split_counts':{name:int(np.count_nonzero(index['split']==i)) for i,name in enumerate(('train','val','test'))},
              'class_counts':{name:np.bincount(index['label'][index['split']==i],minlength=7247).tolist() for i,name in enumerate(('train','val','test'))},
              'excluded_empty_records':excluded_empty,'excluded_blank_records':excluded_blank,
              'emnist_validation':'Stratified sample split of the original training set; writer IDs are unavailable.',
              'emnist_orientation':'Transpose raw IDX images once before shared image preprocessing.'}
    (output/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':'complete','classes':len(classes),'splits':metadata['split_counts'],'seconds':round(time.monotonic()-start)},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',default='data')
    parser.add_argument('--output',default='data/processed')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--validation-fraction',type=float,default=0.1)
    args=parser.parse_args()
    if not 0<args.validation_fraction<0.5:
        parser.error('--validation-fraction must be between 0 and 0.5')
    prepare(args.data_root,args.output,args.seed,args.validation_fraction)
