"""Official MGPC inference with measured camera coordinates and cached labels.

All learned tensors are loaded strictly from the official checkpoint. Only
PointNet sampling/indexing is implemented in PyTorch for CUDA-version portability;
no model layer is replaced or randomly left initialized.
"""
from argparse import Namespace
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types
import numpy as np


def normalized_camera_points(points, pose):
    """Normalize only the input, without GT extents or hidden-axis estimates."""
    camera = (np.asarray(points)-pose[:3, 3]) @ pose[:3, :3]
    center = camera.mean(0)
    # Every official demo has an input AABB diagonal of one, not radius one.
    radius = max(float(np.linalg.norm(np.ptp(camera, axis=0))), 1e-6)
    return (camera-center)/radius, center, radius


def world_points(points, center, radius, pose):
    return (np.asarray(points)*radius+center) @ pose[:3, :3].T+pose[:3, 3]


def portable_pointnet_ops():
    """Deterministic greedy FPS and exact tensor gathering, inference only."""
    import torch
    module = types.ModuleType('pointnet2_ops.pointnet2_utils')

    def fps(points, count):
        batch, length, _ = points.shape
        distances = torch.full((batch, length), 1e10, device=points.device)
        indices = torch.zeros((batch, count), dtype=torch.int64, device=points.device)
        farthest = torch.zeros(batch, dtype=torch.int64, device=points.device)
        batches = torch.arange(batch, device=points.device)
        for index in range(count):
            indices[:, index] = farthest
            center = points[batches, farthest, :][:, None, :]
            distances = torch.minimum(distances, torch.sum((points-center)**2, -1))
            farthest = distances.max(-1).indices
        return indices.to(torch.int32)

    def gather(features, indices):
        return torch.gather(features, 2, indices.long()[:, None, :].expand(-1, features.shape[1], -1))

    def group(features, indices):
        batch, groups, neighbours = indices.shape
        return gather(features, indices.reshape(batch, -1)).reshape(batch, features.shape[1], groups, neighbours)

    module.furthest_point_sample = fps
    module.gather_operation = gather
    module.grouping_operation = group
    return module


class JointShapeModel:
    def __init__(self, repo):
        import torch
        import clip
        from transformers import AutoModel, Dinov2Config, Dinov2Model
        from torchvision import transforms
        self.repo = Path(repo)
        weights = self.repo/'checkpoints/mgpc/ckpt_8192.pt'
        with torch.serialization.safe_globals([Namespace]):
            checkpoint = torch.load(weights, weights_only=True, map_location='cpu', mmap=True)
        state = checkpoint['state_dict']
        clip_prefix = 'text_encoder.model.clip_model.'
        clip_state = {key[len(clip_prefix):]: value for key, value in state.items() if key.startswith(clip_prefix)}
        vendor = self.repo/'scannet/script/thirdparty/MGPC'
        namespace = types.ModuleType('mgpc_vendor'); namespace.__path__ = [str(vendor)]
        sys.modules['mgpc_vendor'] = namespace
        pointnet = types.ModuleType('pointnet2_ops'); pointnet.__path__ = []
        sys.modules['pointnet2_ops'] = pointnet
        sys.modules['pointnet2_ops.pointnet2_utils'] = portable_pointnet_ops()
        metrics = types.ModuleType('utils.metrics')
        def training_only(*args, **kwargs):
            raise RuntimeError('This adapter supports inference, not upstream training loss')
        metrics.HyperCD = training_only
        if 'utils' not in sys.modules:
            package = types.ModuleType('utils'); package.__path__ = []
            sys.modules['utils'] = package
        sys.modules['utils.metrics'] = metrics
        configuration = json.loads((self.repo/'checkpoints/mgpc/dinov2_config.json').read_text())
        original_auto, original_clip = AutoModel.from_pretrained, clip.load
        try:
            AutoModel.from_pretrained = lambda *args, **kwargs: Dinov2Model(Dinov2Config(**configuration))
            clip.load = lambda *args, **kwargs: (clip.model.build_model(clip_state).float(), None)
            architecture = importlib.import_module('mgpc_vendor.models.MGPC')
            self.model = architecture.MGPC()
        finally:
            AutoModel.from_pretrained, clip.load = original_auto, original_clip
        self.model.load_state_dict(state, strict=True)
        self.model = self.model.eval().to('cuda', dtype=torch.float32)
        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        self.metadata = {'model': 'MGPC_official_8192', 'parameters_loaded_strictly': len(state),
            'partial_cloud_is_network_input': True, 'text_is_cached_instance_name': True,
            'pointnet_ops': 'portable_exact_indexing_and_greedy_FPS', 'qwen_api_calls': 0}

    def predict(self, observed, image, name, pose):
        import torch
        points, center, radius = normalized_camera_points(observed, pose)
        # Fixed reproducible coverage sampling; no GT or class-dependent scaling.
        if len(points) > 2048:
            index = np.linspace(0, len(points)-1, 2048, dtype=int)
            points = points[index]
        elif len(points) < 2048:
            points = points[np.arange(2048) % len(points)]
        partial = torch.from_numpy(points.astype(np.float32))[None].cuda()
        rgb = self.transform(image.convert('RGB'))[None].cuda()
        with torch.inference_mode():
            predicted, _, _ = self.model.predict(partial, rgb, [f'This is a/an "{name}"'])
        prediction = predicted[0].float().cpu().numpy()
        if prediction.shape != (8192, 3) or not np.all(np.isfinite(prediction)):
            raise ValueError('MGPC produced an invalid whole-object point set')
        return world_points(prediction, center, radius, pose), {
            **self.metadata, 'camera_centroid_m': center.tolist(), 'input_radius_m': radius,
            'input_points': 2048, 'output_points': 8192, 'identity': name}
