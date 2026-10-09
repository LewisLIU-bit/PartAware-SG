"""Keep crop scores and detector features aligned without truncating busy frames."""
import sys
import unittest
from unittest.mock import Mock
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'grounded_sam'))
from grounded_sam.joint_grounding import JointGrounding
from qwen_tools.description_scoring import FlorenceDescriptionScorer


class ScoringChecks(unittest.TestCase):
    def test_release_moves_a_shared_model_only_once(self):
        joint=JointGrounding.__new__(JointGrounding)
        scorer=SimpleNamespace(model=SimpleNamespace(to=Mock()))
        joint.profiles={str(i):{'scorer':scorer} for i in range(35)}
        gsam=SimpleNamespace(grounding_dino_model=SimpleNamespace(model=SimpleNamespace(to=Mock())),
            sam_predictor=SimpleNamespace(reset_image=Mock(),model=SimpleNamespace(to=Mock())))
        fake_torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False))
        joint._release(gsam,fake_torch)
        scorer.model.to.assert_called_once_with('cpu')

    def scorer(self):
        class Tokenizer:
            def __call__(self, text, **kwargs):
                return {'input_ids':torch.tensor([[1,2,3]]), 'attention_mask':torch.ones((1,3),dtype=torch.int64),
                        'special_tokens_mask':torch.tensor([[1,0,1]])}
            def convert_ids_to_tokens(self, ids): return [str(i) for i in ids]
        class Processor:
            tokenizer=Tokenizer()
            def __call__(self, text, images, **kwargs):
                return {'pixel_values':torch.tensor([np.asarray(image).mean()/255 for image in images],dtype=torch.float32)}
        def model(pixel_values, labels, **kwargs):
            logits=torch.zeros((*labels.shape,4));logits[:,:,2]=pixel_values[:,None]
            return SimpleNamespace(logits=logits)
        scorer=FlorenceDescriptionScorer.__new__(FlorenceDescriptionScorer)
        scorer.torch=torch;scorer.device='cpu';scorer.dtype=torch.float32
        scorer.max_tokens=96;scorer.processor=Processor();scorer.model=model
        return scorer

    def test_batch_preserves_individual_scores_and_order(self):
        scorer=self.scorer();images=[Image.new('RGB',(8,8),value) for value in ('black','white')]
        descriptions={'object':'an object'}
        separate=[scorer.score(image,descriptions) for image in images]
        batched=scorer.score_many(images,descriptions)
        self.assertEqual(separate,batched)
        self.assertNotEqual(batched[0]['object']['mean_log_likelihood'],batched[1]['object']['mean_log_likelihood'])

    def test_memory_bound_and_sequence_limit_do_not_truncate(self):
        scorer=self.scorer();image=Image.new('RGB',(8,8))
        with self.assertRaises(ValueError):scorer.score_many([image]*9,{'object':'an object'})
        scorer.max_tokens=2
        with self.assertRaises(ValueError):scorer.score(image,{'object':'an object'})
        self.assertEqual(scorer.score_many([],{}),[])

    def test_busy_frame_keeps_every_candidate_and_aligned_feature(self):
        count=164
        joint=JointGrounding.__new__(JointGrounding)
        joint.config={'description_candidate_policy':'bounded_all','description_score_batch':8}
        joint.candidate_threshold=.25;joint.nms_threshold=.5;joint.pre_nms_threshold=.95
        joint.max_candidates=128;joint.weight=.8;joint.proposal_routes={};joint.profile_queries={};joint.aliases={}
        scorer=SimpleNamespace(device='cpu',model=SimpleNamespace(to=lambda *a:None),
            score_many=lambda crops,descriptions:[{'object':{'mean_log_likelihood':-1.},
                '__background__':{'mean_log_likelihood':-3.}} for _ in crops])
        joint.profiles={'object':{'config':{'descriptions':{'object':'an object','__background__':'background'},
            'target_description_id':'object'},'baseline':{'object':0.,'__background__':0.}}}
        joint._release=lambda *a:None;joint._scorer_for_profile=lambda *a:scorer
        boxes=np.array([[4*i,0,4*i+3,20] for i in range(count)],float)
        features=torch.arange(count)[:,None].repeat(1,256)
        detector=SimpleNamespace(model=SimpleNamespace(to=lambda *a:None),device='cpu',
            predict_with_classes=lambda **kw:(SimpleNamespace(xyxy=boxes,confidence=np.full(count,.9)),features))
        def segment(image,proposals,**kwargs):
            masks=np.zeros((len(proposals),*image.shape[:2]),bool)
            for i in range(len(proposals)):masks[i,:20,4*i:4*i+3]=True
            return masks,[{} for _ in proposals]
        gsam=SimpleNamespace(grounding_dino_model=detector,
            sam_predictor=SimpleNamespace(model=SimpleNamespace(to=lambda *a:None)),segment_external_boxes=segment)
        _,masks,ids,scores,result=joint.infer(gsam,np.zeros((24,700,3),np.uint8),['object'],.4)
        self.assertEqual(len(masks),count);self.assertEqual(len(joint.last_details),count)
        np.testing.assert_array_equal(result[:,0],np.arange(count))


if __name__ == '__main__':unittest.main()
