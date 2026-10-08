"""Official TripoSR weights with local DINO config and CPU marching cubes.

The network is unchanged. Foreground removal is supplied by registered instance
masks; the upstream rembg helper is intentionally unavailable in this worker.
"""
from pathlib import Path
import sys
import types
import numpy as np
import torch


class ImageShapeModel:
    def __init__(self, repo):
        root = Path(repo)
        sys.path.insert(0, str(root/'scannet/script/thirdparty/TripoSR'))
        from skimage.measure import marching_cubes
        portable = types.ModuleType('torchmcubes')
        def extract(level, threshold):
            vertices, faces, _, _ = marching_cubes(level.detach().cpu().numpy(), threshold)
            # Match torchmcubes' output ordering, which upstream reverses.
            return torch.from_numpy(vertices[:, ::-1].copy()), torch.from_numpy(faces.copy()).long()
        portable.marching_cubes = extract
        sys.modules['torchmcubes'] = portable
        masked_input = types.ModuleType('rembg')
        def unavailable(*args, **kwargs):
            raise RuntimeError('Use registered instance masks, not automatic background removal')
        masked_input.remove = masked_input.new_session = unavailable
        sys.modules['rembg'] = masked_input
        from tsr.models.tokenizers import image as tokenizer
        original_download = tokenizer.hf_hub_download
        def local_config(repo_id, filename, **kwargs):
            if repo_id == 'facebook/dino-vitb16' and filename == 'config.json':
                return str(root/'checkpoints/triposr/dino_config.json')
            return original_download(repo_id=repo_id, filename=filename, **kwargs)
        tokenizer.hf_hub_download = local_config
        from tsr.system import TSR
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = TSR.from_pretrained(str(root/'checkpoints/triposr'), 'config.yaml', 'model.ckpt')
        self.model.renderer.set_chunk_size(4096)
        self.model.to(self.device).eval()

    def predict(self, image):
        with torch.inference_mode():
            codes = self.model([image], device=self.device)
            mesh = self.model.extract_mesh(codes, has_vertex_color=False, resolution=128)[0]
        if len(mesh.faces) < 32:
            raise ValueError('The generative model returned a degenerate mesh')
        # Fixed sampling makes cached candidates reproducible and inspectable.
        import trimesh
        points, _ = trimesh.sample.sample_surface(mesh, 24000, seed=12)
        return np.asarray(points, float), {'vertices': len(mesh.vertices), 'faces': len(mesh.faces),
            'watertight': bool(mesh.is_watertight), 'surface_sampling_points': len(points)}


class DiffusionShapeModel:
    """Official mini Turbo model with sequential CPU offload; geometry only."""
    def __init__(self, repo):
        root = Path(repo)
        source = root/'scannet/script/thirdparty/Hunyuan3D/hy3dgen'
        # Import only the shape inference package, not optional texturing tools.
        for name, path in [('hy3dgen', source), ('hy3dgen.shapegen', source/'shapegen')]:
            package = types.ModuleType(name); package.__path__ = [str(path)]
            sys.modules[name] = package
        from hy3dgen.shapegen.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        self.model = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
            str(root/'checkpoints/hunyuan_mini/model.fp16.safetensors'),
            str(root/'checkpoints/hunyuan_mini/config.yaml'),
            device='cpu', dtype=torch.float16, use_safetensors=True)
        self.model.components = {name: getattr(self.model, name) for name in ['conditioner', 'model', 'vae']}
        self.model.enable_model_cpu_offload(device='cuda')
        self.model.device = torch.device('cuda')
        original_decode = self.model.vae.decode
        pipeline = self.model
        def staged_decode(*args, **kwargs):
            pipeline.conditioner.to('cpu'); pipeline.model.to('cpu')
            pipeline.vae.to('cuda')
            return original_decode(*args, **kwargs)
        self.model.vae.decode = staged_decode

    def predict(self, image):
        mesh = self.model(image=image, num_inference_steps=5, guidance_scale=0.,
            generator=torch.Generator(device='cuda').manual_seed(12),
            octree_resolution=192, num_chunks=4096, mc_algo='mc', enable_pbar=False)[0]
        import trimesh
        points, _ = trimesh.sample.sample_surface(mesh, 32000, seed=12)
        return np.asarray(points, float), {'vertices': len(mesh.vertices), 'faces': len(mesh.faces),
            'watertight': bool(mesh.is_watertight), 'surface_sampling_points': len(points)}
