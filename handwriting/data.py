import json
from collections import OrderedDict
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from .images import GlyphTransform

RECORD = np.dtype([('file', '<u2'), ('offset', '<u8'), ('width', '<u2'), ('height', '<u2'),
                   ('label', '<u2'), ('split', 'u1'), ('source', 'u1')])
SPLITS = {'train': 0, 'val': 1, 'test': 2}


class HandwritingDataset(Dataset):
    def __init__(self, root, split, size=128, training=False, limit=None, seed=42):
        self.root = Path(root)
        self.metadata = json.loads((self.root / 'metadata.json').read_text(encoding='utf-8-sig'))
        self.classes = self.metadata['classes']
        self.records = np.load(self.root / 'index.npy', mmap_mode='r', allow_pickle=False)
        self.indices = np.flatnonzero(self.records['split'] == SPLITS[split])
        if limit and len(self.indices) > limit:
            rng = np.random.default_rng(seed)
            labels = self.records['label'][self.indices]
            per_class = limit // len(self.classes)
            if per_class:
                selected = []
                order = np.argsort(labels, kind='stable')
                boundaries = np.flatnonzero(np.r_[True, labels[order][1:] != labels[order][:-1], True])
                for start, end in zip(boundaries[:-1], boundaries[1:]):
                    pool = self.indices[order[start:end]]
                    selected.extend(rng.choice(pool, min(per_class,len(pool)),replace=False).tolist())
                selected = np.asarray(selected,dtype=np.int64)
                remaining = np.setdiff1d(self.indices,selected,assume_unique=True)
                extra = rng.choice(remaining,limit-len(selected),replace=False)
                self.indices = np.sort(np.concatenate((selected,extra)))
            else:
                self.indices = np.sort(rng.choice(self.indices, limit, replace=False))
        self.transform = GlyphTransform(size, training)
        self._maps = OrderedDict()

    @property
    def labels(self):
        return np.asarray(self.records['label'][self.indices], dtype=np.int64)

    def __len__(self):
        return len(self.indices)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_maps'] = OrderedDict()  # workers open their own mmap objects
        state['records'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.records = np.load(self.root / 'index.npy', mmap_mode='r', allow_pickle=False)

    def raw_image(self, index):
        record = self.records[self.indices[index]]
        file_id = int(record['file'])
        if file_id not in self._maps:
            if len(self._maps) >= 16:
                self._maps.popitem(last=False)
            self._maps[file_id] = np.memmap(self.root / self.metadata['files'][file_id]['path'], mode='r', dtype=np.uint8)
        self._maps.move_to_end(file_id)
        a = self._maps[file_id]
        offset, w, h = int(record['offset']), int(record['width']), int(record['height'])
        image = a[offset:offset+w*h].reshape(h, w)
        if int(record['source']) == 1:
            image = 255 - image.T  # upright EMNIST, converted to known white-background polarity
        return ImageOps.expand(Image.fromarray(np.array(image, copy=True)), border=2, fill=255), int(record['label'])

    def __getitem__(self, index):
        image, label = self.raw_image(index)
        try:
            return self.transform(image), label
        except ValueError as exc:
            raise ValueError(f'Invalid glyph at dataset row {int(self.indices[index])}: {exc}') from exc
