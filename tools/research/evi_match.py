"""Candidate verifier prototype: no completion generator or trained checkpoint.

Real training records must bind actual ESAM queries, text embeddings, geometry,
predicted graph context and independent observation evidence by provenance.
Synthetic tensors are allowed only in the explicitly labelled smoke test.
"""
import torch
from torch import nn
from torch.nn import functional as F

FEATURE_DIMENSIONS=dict(esam_query=256,text=384,geometry=32,graph=64,evidence=12)


class EviMatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(sum(FEATURE_DIMENSIONS.values()),512),
            nn.LayerNorm(512),nn.GELU(),nn.Linear(512,256),nn.LayerNorm(256),nn.GELU())
        self.identity=nn.Linear(256,128)
        self.existence=nn.Linear(256,1)
        self.quality=nn.Linear(256,1)
        self.text_projection=nn.Linear(FEATURE_DIMENSIONS['text'],128)
        self.structure_projection=nn.Sequential(
            nn.Linear(sum(FEATURE_DIMENSIONS.values())-FEATURE_DIMENSIONS['text'],256),
            nn.LayerNorm(256),nn.GELU(),nn.Linear(256,128))

    def forward(self,features):
        values=[]
        for key,dim in FEATURE_DIMENSIONS.items():
            value=features[key]
            if value.ndim!=2 or value.shape[1]!=dim or not torch.isfinite(value).all():
                raise ValueError(f'Invalid actual feature block: {key}')
            values.append(value)
        if len({len(v) for v in values})!=1:raise ValueError('Feature row bindings differ')
        x=self.encoder(torch.cat(values,dim=1))
        # Text must not leak into its own geometric/graph comparison branch.
        text=F.normalize(self.text_projection(features['text']),dim=1)
        structure=F.normalize(self.structure_projection(torch.cat(
            [features[key] for key in FEATURE_DIMENSIONS if key!='text'],dim=1)),dim=1)
        return dict(identity=F.normalize(self.identity(x),dim=1),
            existence_logit=self.existence(x).squeeze(1),
            mask_quality=torch.sigmoid(self.quality(x).squeeze(1)),
            text_embedding=text,structure_embedding=structure,
            compatibility_logit=(text*structure).sum(1)/.07)


def multi_positive_loss(logits,positive,known):
    """Mean positive log probability, with explicit unknown comparison masking."""
    if positive.shape!=logits.shape or known.shape!=logits.shape:
        raise ValueError('Alignment target matrices must bind the same records')
    if (positive&~known).any():
        raise ValueError('Positive correspondence cannot be unknown')
    valid=positive.any(1)
    if not valid.any():return logits.sum()*0
    # Slice before logsumexp: all-unknown rows must not generate -inf or NaN.
    values=logits[valid];positive=positive[valid];known=known[valid]
    log_prob=values-torch.logsumexp(values.masked_fill(~known,float('-inf')),dim=1)[:,None]
    return -(log_prob.masked_fill(~positive,0).sum(1)/positive.sum(1)).mean()


def verification_loss(output,targets,temperature=.07):
    """Unknown/occluded existence labels are masked, never background negatives."""
    known=targets['existence_known'].bool()
    existence=(F.binary_cross_entropy_with_logits(output['existence_logit'][known],
        targets['existence'][known].float()) if known.any() else output['existence_logit'].sum()*0)
    quality_known=targets['quality_known'].bool()
    quality=(F.smooth_l1_loss(output['mask_quality'][quality_known],
        targets['mask_IoU'][quality_known]) if quality_known.any() else output['mask_quality'].sum()*0)
    identities=targets['instance_identity']
    embeddings=output['identity'];similarities=embeddings@embeddings.T/temperature
    diagonal=torch.eye(len(identities),dtype=torch.bool,device=identities.device)
    positive=(identities[:,None]==identities[None,:])&~diagonal&(identities[:,None]>=0)
    denominator=(identities[:,None]>=0)&(identities[None,:]>=0)&~diagonal
    contrastive=multi_positive_loss(similarities,positive,denominator)
    # Generic class text has multiple valid instances; do not use identity
    # equality as the semantic positive mask. Wrong names are not ghosts.
    logits=output['text_embedding']@output['structure_embedding'].T/temperature
    positive=targets['text_positive_mask'].bool()
    known=targets['text_comparison_known'].bool()
    text_alignment=(multi_positive_loss(logits,positive,known)+
        multi_positive_loss(logits.T,positive.T,known.T))/2
    compatible=targets['compatibility_known'].bool()
    compatibility=(F.binary_cross_entropy_with_logits(output['compatibility_logit'][compatible],
        targets['compatible'][compatible].float()) if compatible.any()
        else output['compatibility_logit'].sum()*0)
    return dict(total=existence+quality+contrastive+text_alignment+compatibility,
        existence=existence,quality=quality,contrastive=contrastive,
        text_alignment=text_alignment,compatibility=compatibility)
