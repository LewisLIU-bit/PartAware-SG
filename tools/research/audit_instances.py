"""Independent GT diagnostics. Never import this module into construction."""
import numpy as np


def mask_diagnosis(predictions, truth, prediction_ids, GT_ids, minimum_points=100):
    p, g = np.asarray(predictions, bool), np.asarray(truth, bool)
    if p.ndim!=2 or g.ndim!=2 or p.shape[1]!=g.shape[1]:
        raise ValueError('Mask domains differ')
    if len(prediction_ids)!=len(p) or len(GT_ids)!=len(g):
        raise ValueError('ID metadata differs')
    inter=p.astype(np.int64)@g.astype(np.int64).T
    pc,gc=p.sum(1),g.sum(1)
    union=pc[:,None]+gc[None,:]-inter
    iou=np.divide(inter,union,out=np.zeros_like(union,float),where=union>0)
    rows=[]
    for j,gid in enumerate(GT_ids):
        if gc[j]<minimum_points:continue
        best=int(np.argmax(iou[:,j])) if len(p) else None
        score=float(iou[best,j]) if best is not None else 0.
        recall=float(inter[best,j]/gc[j]) if best is not None else 0.
        purity=float(inter[best,j]/pc[best]) if best is not None and pc[best] else 0.
        rows.append(dict(GT_id=gid,best_prediction=prediction_ids[best] if best is not None and score>0 else None,
            IoU=score,GT_coverage=recall,prediction_purity=purity,
            status='missed_at25' if score<=.25 else 'below50' if score<=.5 else 'matched_at50',
            cause='no_overlap' if score==0 else 'low_coverage' if recall<purity else 'contamination',
            overlapping_predictions=int(np.count_nonzero(iou[:,j]>.1))))
    mixtures=[]
    for i,pid in enumerate(prediction_ids):
        # Zero-overlap ties do not constitute mixed instances.
        hit=np.flatnonzero((iou[i]>.1)&(gc>=minimum_points))
        if len(hit)>1:mixtures.append(dict(prediction_id=pid,GT_ids=[GT_ids[j] for j in hit],
            IoUs=iou[i,hit].tolist()))
    return dict(GT_reserved_for_diagnostics=True,eligible_GT=len(rows),
        prediction_count=len(p),zero_domain_masks=[prediction_ids[i] for i in np.flatnonzero(pc==0)],
        below_minimum_prediction_masks=int(np.count_nonzero(pc<minimum_points)),
        missed_at25=sum(r['status']=='missed_at25' for r in rows),
        below50=sum(r['status']!='matched_at50' for r in rows),
        mixed_predictions=mixtures,GT=rows,
        empty_mask_is_not_ghost_proof=True)
