"""Separate geometry and score provenance; no GT and no extra training.

GPT object-name confidence is not mask quality. For retained image-derived
geometry, use an IoU-weighted trained peer quality only where a peer actually
overlaps. This is an uncalibrated ranking ablation, not an existence test.
"""
import numpy as np
from mask_proposals import overlap

def rank(primary,peer,peer_scores,kept):
 a,b=np.asarray(primary,bool),np.asarray(peer,bool);sb=np.asarray(peer_scores,float)
 if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[1] or sb.shape!=(len(b),):raise ValueError('Common mask domain and peer scores required')
 if not np.isfinite(sb).all():raise ValueError('Finite mask quality required')
 agreement=overlap(a,b)
 native=(agreement*sb[None,:]).max(1) if len(b) else np.zeros(len(a))
 return np.r_[native,sb][np.asarray(kept,int)]

def consistency(primary,peer,kept,original_scores):
 """Soft symmetric corroboration penalty; no proposal is deleted or made zero."""
 a,b=np.asarray(primary,bool),np.asarray(peer,bool);agreement=overlap(a,b)
 ra=agreement.max(1) if len(b) else np.zeros(len(a))
 rb=agreement.max(0) if len(a) else np.zeros(len(b))
 support=np.r_[ra,rb][np.asarray(kept,int)]
 q=np.asarray(original_scores,float)
 if q.shape!=support.shape:raise ValueError('One score per retained proposal required')
 return q*(.5+.5*support)
