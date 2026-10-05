"""Official AdaPoinTr inference with portable, inference-only point operations.

The pretrained architecture and state dictionary are unchanged. Portable FPS
and gather avoid installing historical CUDA extensions into the base runtime.
All compatibility modules are scoped to a dedicated completion worker process.
"""
import importlib
from pathlib import Path
import sys
import types
import numpy as np
import torch
import yaml


def portable_ops():
    package = types.ModuleType('pointnet2_ops')
    ops = types.ModuleType('pointnet2_ops.pointnet2_utils')
    def fps(points, count):
        batch, number, _ = points.shape
        indices = torch.zeros((batch, count), dtype=torch.long, device=points.device)
        distance = torch.full((batch, number), float('inf'), device=points.device)
        farthest = torch.zeros(batch, dtype=torch.long, device=points.device)
        rows = torch.arange(batch, device=points.device)
        for i in range(count):
            indices[:, i] = farthest
            current = points[rows, farthest][:, None]
            distance = torch.minimum(distance, ((points-current)**2).sum(-1))
            farthest = distance.argmax(-1)
        return indices.int()
    def gather(features, indices):
        return torch.gather(features, 2, indices.long()[:, None].expand(-1, features.shape[1], -1))
    def three_nn(query, points):
        distance, indices = torch.cdist(query, points).topk(3, dim=-1, largest=False)
        return distance, indices.int()
    def interpolate(features, indices, weights):
        batch = torch.arange(features.shape[0], device=features.device)[:, None, None]
        grouped = features.transpose(1, 2)[batch, indices.long()]
        return (grouped*weights[..., None]).sum(2).transpose(1, 2).contiguous()
    ops.furthest_point_sample, ops.gather_operation = fps, gather
    ops.three_nn, ops.three_interpolate = three_nn, interpolate
    package.pointnet2_utils = ops
    sys.modules['pointnet2_ops'] = package
    sys.modules['pointnet2_ops.pointnet2_utils'] = ops
    chamfer = types.ModuleType('extensions.chamfer_dist')
    class Chamfer(torch.nn.Module):
        def forward(self, a, b):
            distances = torch.cdist(a, b)
            return (distances.min(1).values.mean()+distances.min(2).values.mean())/2
    chamfer.ChamferDistanceL1 = Chamfer
    extensions = types.ModuleType('extensions')
    extensions.__path__ = []
    sys.modules['extensions'], sys.modules['extensions.chamfer_dist'] = extensions, chamfer


class AttrDict(dict):
    def __getattr__(self, key):
        if key not in self:
            raise AttributeError(key)
        return self[key]


def attributes(value):
    if isinstance(value, dict):
        return AttrDict({k: attributes(v) for k, v in value.items()})
    if isinstance(value, list):
        return [attributes(x) for x in value]
    return value


class CompletionModel:
    def __init__(self, repo):
        root = Path(repo)/'scannet/script/thirdparty/PoinTr'
        portable_ops()
        for name in ['models', 'utils']:
            package = types.ModuleType(name)
            package.__path__ = [str(root/name)]
            sys.modules[name] = package
        # Registry decorators do not need training builders during inference.
        build = types.ModuleType('models.build')
        class Registry:
            def register_module(self):
                return lambda cls: cls
        build.MODELS = Registry()
        build.build_model_from_cfg = lambda *args, **kwargs: None
        sys.modules['models.build'] = build
        module = importlib.import_module('models.AdaPoinTr')
        cfg = yaml.safe_load((root/'cfgs/PCN_models/AdaPoinTr.yaml').read_text())
        self.model = module.AdaPoinTr(attributes(cfg['model']))
        checkpoint = torch.load(Path(repo)/'checkpoints/completion/AdaPoinTr_PCN.pth',
                                map_location='cpu', weights_only=False)
        state = checkpoint.get('base_model', checkpoint.get('model', checkpoint))
        state = {k.removeprefix('module.'): v for k, v in state.items()}
        self.model.load_state_dict(state, strict=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device).eval()

    def predict(self, points):
        center = points.mean(0)
        radius = float(np.max(np.linalg.norm(points-center, axis=1)))
        if radius < 1e-4:
            raise ValueError('Degenerate partial geometry')
        # ShapeNet/PCN uses Y-up; restore the original metric world frame later.
        rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
        canonical = (points-center) @ rotation.T/radius
        indices = np.linspace(0, len(canonical)-1, 2048).astype(int)
        tensor = torch.as_tensor(canonical[indices], dtype=torch.float32, device=self.device)[None]
        with torch.inference_mode():
            output = self.model(tensor)[-1][0].float().cpu().numpy()
        return (output*radius) @ rotation+center
