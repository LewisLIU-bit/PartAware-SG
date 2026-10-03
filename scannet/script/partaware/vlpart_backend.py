"""Official VLPart inference with dynamic object-conditioned vocabularies."""
from pathlib import Path
import importlib.util
import sys
import tempfile
import numpy as np


class VLPartBackend:
    def __init__(self, repo, config, checkpoint, device='cuda', score=0.2, image_size=640):
        repo = Path(repo).expanduser().resolve()
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(checkpoint)
        sys.path.insert(0, str(repo))
        import torch
        import clip
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor
        from vlpart.config import add_vlpart_config
        self.torch = torch
        self.device = device
        self.clip, self.preprocess = clip.load('RN50', device=device, jit=False,
                                               download_root=str(repo.parents[3] / 'checkpoints/clip'))
        self.clip.eval()
        self.cache = {}
        # Load the official classifier-reset helper without generic module names.
        sys.path.insert(0, str(repo / 'demo'))
        spec = importlib.util.spec_from_file_location('_partaware_vlpart_predictor', repo / 'demo/predictor.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.reset_classifier = module.reset_cls_test
        cfg = get_cfg()
        add_vlpart_config(cfg)
        cfg.merge_from_file(str(Path(config).resolve()))
        cfg.MODEL.WEIGHTS = str(Path(checkpoint).resolve())
        cfg.MODEL.DEVICE = device
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score
        cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
        cfg.INPUT.MIN_SIZE_TEST = image_size
        cfg.INPUT.MAX_SIZE_TEST = max(image_size, int(image_size * 1.5))
        # Classifier buffers are replaced with actual CLIP text weights per frame.
        # No dataset metadata downloads or learned random features are used.
        with tempfile.TemporaryDirectory(prefix='partaware-classifier-') as directory:
            initial = Path(directory) / 'initial.npy'
            with torch.no_grad():
                weights = self.clip.encode_text(clip.tokenize(['a table leg']).to(device)).float().cpu().numpy()
            np.save(initial, weights)
            cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = str(initial)
            cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH_GROUP = [str(initial)]
            cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_INFERENCE_PATH = str(initial)
            cfg.MODEL.ROI_BOX_HEAD.USE_FED_LOSS = False
            cfg.MODEL.ROI_BOX_HEAD.USE_FED_LOSS_GROUP = [False]
            cfg.freeze()
            self.predictor = DefaultPredictor(cfg)

    def predict(self, image, classes):
        import clip
        from PIL import Image
        torch = self.torch
        if not classes:
            return []
        with torch.no_grad():
            missing = [name for name in classes if name not in self.cache]
            if missing:
                texts = clip.tokenize(['a ' + name.replace(':', ' ') for name in missing]).to(self.device)
                features = self.clip.encode_text(texts).float()
                for name, feature in zip(missing, features):
                    self.cache[name] = feature
            weights = torch.stack([self.cache[name] for name in classes], dim=1)
            self.reset_classifier(self.predictor.model, weights)
            prediction = self.predictor(image)['instances'].to('cpu')
            result = []
            for index in range(len(prediction)):
                cid = int(prediction.pred_classes[index])
                if not 0 <= cid < len(classes):
                    continue
                box = prediction.pred_boxes.tensor[index].numpy()
                x1, y1, x2, y2 = box.astype(int)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop_pixels = image[y1:y2, x1:x2, ::-1].copy()
                crop_mask = prediction.pred_masks[index].numpy()[y1:y2, x1:x2].astype(bool)
                crop_pixels[~crop_mask] = 127
                crop = Image.fromarray(crop_pixels)
                visual = self.clip.encode_image(self.preprocess(crop).unsqueeze(0).to(self.device)).float()[0]
                visual = visual / visual.norm().clamp_min(1e-9)
                result.append({'label': classes[cid], 'score': float(prediction.scores[index]),
                               'box': box, 'mask': prediction.pred_masks[index].numpy().astype(bool),
                               'feature': visual.cpu().numpy()})
        return result
