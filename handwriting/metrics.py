import json
from collections import Counter
import numpy as np
import torch


@torch.inference_mode()
def evaluate(model, loader, device, classes, confusion_limit=100, progress=False):
    model.eval()
    counts = np.zeros(len(classes), dtype=np.int64)
    correct = np.zeros_like(counts)
    top5_correct = 0
    loss_sum = 0.0
    confusion = Counter()
    for batch_index, (images, target) in enumerate(loader):
        images, target = images.to(device), target.to(device)
        logits = model(images)
        loss_sum += float(torch.nn.functional.cross_entropy(logits, target)) * len(target)
        candidates = logits.topk(min(5, len(classes)), 1).indices
        predicted = candidates[:, 0]
        top5_correct += int((candidates == target[:, None]).any(1).sum())
        y, p = target.cpu().numpy(), predicted.cpu().numpy()
        counts += np.bincount(y, minlength=len(classes))
        correct += np.bincount(y[y == p], minlength=len(classes))
        confusion.update(zip(y[y != p].tolist(), p[y != p].tolist()))
        if progress and ((batch_index + 1) % 250 == 0 or batch_index + 1 == len(loader)):
            print(json.dumps({'event':'evaluation_progress','batch':batch_index+1,'batches':len(loader),
                              'samples':int(counts.sum())}),flush=True)
    total = int(counts.sum())
    if not total:
        raise ValueError('Empty evaluation dataset.')
    accuracies = np.divide(correct, counts, out=np.zeros(len(classes), dtype=float), where=counts > 0)
    groups = {}
    for name, indices in [('digits',range(10)),('uppercase',range(10,36)),('lowercase',range(36,62)),('chinese',range(62,len(classes)))]:
        idx = np.array(list(indices), dtype=int)
        n = int(counts[idx].sum())
        groups[name] = {'samples':n,'top1':float(correct[idx].sum()/n) if n else None}
    present = counts > 0
    return {'samples':total,'classes_evaluated':int(present.sum()),'loss':loss_sum/total,
            'top1':float(correct.sum()/total),'top5':top5_correct/total,
            'macro_top1':float(accuracies[present].mean()),'groups':groups,
            'worst_classes':[{'character':classes[i],'samples':int(counts[i]),'accuracy':float(accuracies[i])}
                             for i in np.flatnonzero(present)[np.argsort(accuracies[present])[:50]]],
            'confusions':[{'actual':classes[a],'predicted':classes[b],'count':n} for (a,b),n in confusion.most_common(confusion_limit)]}
