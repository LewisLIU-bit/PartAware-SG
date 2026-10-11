"""Synthetic gradient/interface smoke check; not a dataset training result."""
import json
import torch
from evi_match import EviMatch,FEATURE_DIMENSIONS,verification_loss
torch.manual_seed(19)
model=EviMatch()
features={key:torch.randn(6,dim) for key,dim in FEATURE_DIMENSIONS.items()}
targets=dict(existence_known=torch.tensor([1,1,1,1,0,0],dtype=torch.bool),
    existence=torch.tensor([1.,1.,1.,1.,0.,0.]),quality_known=torch.ones(6,dtype=torch.bool),
    mask_IoU=torch.tensor([.8,.7,.9,.8,.1,.2]),instance_identity=torch.tensor([1,1,2,2,-1,-1]))
# Two distinct objects share a generic noun. All four text descriptions are
# compatible, although their instance identities differ.
positive=torch.zeros(6,6,dtype=torch.bool);positive[:4,:4]=True
targets.update(text_positive_mask=positive,text_comparison_known=torch.ones(6,6,dtype=torch.bool),
    compatibility_known=torch.tensor([1,1,1,1,1,0],dtype=torch.bool),
    compatible=torch.tensor([1.,1.,1.,1.,0.,0.]))
out=model(features);loss=verification_loss(out,targets)
assert torch.isfinite(loss['total'])
loss['total'].backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
changed={**targets,'existence':torch.tensor([1.,1.,1.,1.,1.,1.])}
assert torch.equal(verification_loss(out,changed)['existence'],loss['existence']), 'Unknown targets must not affect existence loss'
changed_features={**features,'text':torch.randn_like(features['text'])}
assert torch.equal(model(changed_features)['structure_embedding'],out['structure_embedding']), 'Structure branch must not read the target text'
assert positive[0,2] and targets['instance_identity'][0]!=targets['instance_identity'][2]
assert out['text_embedding'].shape==out['structure_embedding'].shape==(6,128)
unknown={**targets,'existence_known':torch.zeros(6,dtype=torch.bool),
    'quality_known':torch.zeros(6,dtype=torch.bool),'instance_identity':torch.full((6,),-1),
    'text_positive_mask':torch.zeros(6,6,dtype=torch.bool),
    'text_comparison_known':torch.zeros(6,6,dtype=torch.bool),
    'compatibility_known':torch.zeros(6,dtype=torch.bool)}
zero=verification_loss(model(features),unknown)['total']
assert torch.isfinite(zero) and zero.item()==0
print(json.dumps(dict(parameters=sum(p.numel() for p in model.parameters()),
    finite_forward_backward=True,unknown_occlusion_not_negative=True,
    distinct_instances_can_share_generic_text=True,target_text_excluded_from_structure=True,
    synthetic_only=True,dataset_training_completed=False)))
