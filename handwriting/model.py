"""A handwriting-specific convolution/attention classifier, trained from scratch."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class DropPath(nn.Module):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate

    def forward(self, x):
        if not self.training or not self.rate:
            return x
        keep = 1 - self.rate
        mask = torch.rand((x.shape[0],) + (1,) * (x.ndim - 1), device=x.device) < keep
        return x * mask.to(x.dtype) / keep


class GRN(nn.Module):
    """Global Response Normalization from ConvNeXt V2 (NHWC)."""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        magnitude = torch.linalg.vector_norm(x, dim=(1, 2), keepdim=True)
        response = magnitude / (magnitude.mean(dim=-1, keepdim=True) + 1e-6)
        return x + self.gamma * (x * response) + self.beta


class StrokeBlock(nn.Module):
    def __init__(self, dim, drop_path):
        super().__init__()
        self.depthwise = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.expand = nn.Linear(dim, 4 * dim)
        self.grn = GRN(4 * dim)
        self.project = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        y = self.depthwise(x).permute(0, 2, 3, 1)
        y = self.project(self.grn(F.gelu(self.expand(self.norm(y)))))
        return x + self.drop_path(y.permute(0, 3, 1, 2))


class StructureBlock(nn.Module):
    """Depthwise positional encoding + attention over spatial tokens."""
    def __init__(self, dim, heads, drop_path):
        super().__init__()
        self.position = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.project = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 3), nn.GELU(), nn.Linear(dim * 3, dim))
        self.drop_path = DropPath(drop_path)
        self.heads = heads

    def forward(self, x):
        x = x + self.position(x)
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        qkv = self.qkv(self.norm1(tokens)).reshape(b, h * w, 3, self.heads, c // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        # Standard attention works on MPS without CUDA-specific fused kernels.
        attention = (q @ k.transpose(-2, -1)) * (c // self.heads) ** -0.5
        y = (attention.softmax(-1) @ v).transpose(1, 2).reshape(b, h * w, c)
        tokens = tokens + self.drop_path(self.project(y))
        tokens = tokens + self.drop_path(self.mlp(self.norm2(tokens)))
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class CosineHead(nn.Module):
    def __init__(self, dim, classes):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(classes, dim))
        self.log_scale = nn.Parameter(torch.tensor(math.log(16.0)))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        return F.linear(F.normalize(x, dim=-1), F.normalize(self.weight, dim=-1)) * self.log_scale.exp().clamp(1, 50)


class HandwritingNet(nn.Module):
    def __init__(self, num_classes, dims=(40, 80, 160, 320), depths=(2, 2, 6, 2),
                 attention_blocks=2, heads=8, drop_path=0.1, dropout=0.1, edges=True):
        super().__init__()
        if len(dims) != 4 or len(depths) != 4 or dims[-1] % heads:
            raise ValueError('Need four stages and a final dimension divisible by heads.')
        self.edges = edges
        kernel = torch.tensor([[[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
                               [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]]) / 8
        self.register_buffer('edge_kernel', kernel[:, None])
        self.stem = nn.Sequential(nn.Conv2d(3 if edges else 1, dims[0], 4, stride=2, padding=1), LayerNorm2d(dims[0]))
        rates = torch.linspace(0, drop_path, sum(depths)).tolist()
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        offset = 0
        for i, (dim, depth) in enumerate(zip(dims, depths)):
            self.stages.append(nn.Sequential(*(StrokeBlock(dim, rates[offset + j]) for j in range(depth))))
            offset += depth
            if i < 3:
                self.downsamples.append(nn.Sequential(LayerNorm2d(dim), nn.Conv2d(dim, dims[i + 1], 2, stride=2)))
        self.structure = nn.Sequential(*(StructureBlock(dims[-1], heads, drop_path) for _ in range(attention_blocks)))
        self.mid_projection = nn.Linear(dims[-2], dims[-1])
        self.pool_weights = nn.Parameter(torch.zeros(3))
        self.norm = nn.LayerNorm(dims[-1])
        self.dropout = nn.Dropout(dropout)
        self.classifier = CosineHead(dims[-1], num_classes)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 1 or min(x.shape[-2:]) < 32:
            raise ValueError('Expected grayscale images [batch, 1, height, width], size >=32.')
        if self.edges:
            gradient = F.conv2d(F.pad(x, (1, 1, 1, 1), mode='replicate'), self.edge_kernel)
            x = torch.cat((x, gradient), dim=1)
        x = self.stem(x)
        mid = None
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i == 2:
                mid = self.mid_projection(x.mean((2, 3)))
            if i < 3:
                x = self.downsamples[i](x)
        local = x.mean((2, 3))
        global_structure = self.structure(x).mean((2, 3))
        pooled = (torch.stack((mid, local, global_structure), dim=1) * self.pool_weights.softmax(0)[None, :, None]).sum(1)
        return self.classifier(self.dropout(self.norm(pooled)))


def build_model(config, num_classes):
    model = config.get('model', {})
    if model.get('name', 'handwriting_net') == 'handwriting_net':
        return HandwritingNet(num_classes, **{k: v for k, v in model.items() if k != 'name'})
    raise ValueError('Unsupported model: ' + str(model.get('name')))
