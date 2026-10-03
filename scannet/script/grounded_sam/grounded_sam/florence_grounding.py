"""Portable Florence evidence scoring without scene-specific reference images."""
import os
from pathlib import Path

from .joint_grounding import JointGrounding, normalized


class FlorenceGrounding(JointGrounding):
    """Reuse DINO boxes, Florence crop scoring, SAM masks, and DINO features."""

    def __init__(self, model_dir=None):
        self.model_dir = Path(model_dir or os.environ.get(
            'FLORENCE_MODEL_DIR', '~/models/vision/Florence-2-large-ft')).expanduser().resolve()
        if not (self.model_dir / 'config.json').is_file():
            raise FileNotFoundError(
                f'Florence model missing: {self.model_dir}. Set FLORENCE_MODEL_DIR '
                'or explicitly select --grounding_backend dino.')
        self.config = {'schema_version': 'florence_grounding_default_v1',
                       'model_dir': str(self.model_dir), 'dino_weight': 0.8,
                       'candidate_threshold': 0.25, 'nms_threshold': 0.5,
                       'pre_nms_threshold': 0.95, 'max_description_candidates': 128,
                       'mask_duplicate_iou': 0.85, 'min_saved_pixels': 16,
                       'reference_normalization': False,
                       'template': 'A photo of a {category}.',
                       'hard_semantic_veto': False,
                       'confusable_classes': {'amplifier': ['speaker'], 'speaker': ['amplifier'],
                                              'laptop': ['monitor'], 'monitor': ['laptop'],
                                              'cabinet': ['shelf'], 'shelf': ['cabinet']},
                       'descriptions': {'amplifier': 'An audio amplifier with an electronic case and controls on its front.',
                                        'speaker': 'A loudspeaker cabinet with circular speaker cones on its front.'}}
        self.weight = self.config['dino_weight']
        self.candidate_threshold = self.config['candidate_threshold']
        self.nms_threshold = self.config['nms_threshold']
        self.pre_nms_threshold = self.config['pre_nms_threshold']
        self.max_candidates = self.config['max_description_candidates']
        self.profiles, self.aliases = {}, {}
        self.proposal_routes, self.profile_queries = {}, {}
        self.shared_scorer = None
        self.last_details = []

    def _scorer_for_profile(self, profile):
        if self.shared_scorer is None:
            from qwen_tools.florence_worker import make_florence_scorer
            import torch
            self.shared_scorer = make_florence_scorer(
                self.model_dir, 'cuda' if torch.cuda.is_available() else 'cpu', 96)
        profile['scorer'] = self.shared_scorer
        return self.shared_scorer

    def infer(self, gsam, image, names, threshold, debug_dir=None):
        # Preserve input category names and the 256-dimensional detector features.
        # Confusable classes compete on the same crop; class names never imply identity.
        confusable = self.config['confusable_classes']
        self.profiles = {}
        for name in names:
            canonical = normalized(name)
            descriptions = {canonical: self.config['descriptions'].get(canonical, f'A photo of a {canonical}.'),
                            '__background__': 'A photo of the background.'}
            for other in confusable.get(canonical, []):
                descriptions[other] = self.config['descriptions'].get(other, f'A photo of a {other}.')
            self.profiles[canonical] = {
                'config': {'descriptions': descriptions, 'target_description_id': canonical,
                           'hard_negative_ids': [],
                           'description_source': 'default_category_templates'},
                'references': [], 'baseline': {key: 0.0 for key in descriptions},
                'model_dir': self.model_dir, 'scorer': self.shared_scorer}
        return super().infer(gsam, image, names, threshold, debug_dir)
