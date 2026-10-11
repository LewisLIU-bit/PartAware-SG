"""GT-assisted annotation-granularity AP; never construction or official AP.

Uniformly union surface-pure children of one annotation. Keep mixed/unknown
predictions as independent entities, including false positives. The prediction
envelope and its score come solely from saved children, never a GT box/score.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def entities(predictions, surfaces):
    assignments={str(r['prediction_id']):r for r in surfaces['per_prediction']}
    groups={}; consumed=set(); output=[]
    for p in predictions:
        row=assignments.get(str(p['id']),{})
        if row.get('pure_fragment') and row.get('dominant_gt_id') is not None:
            groups.setdefault(row['dominant_gt_id'],[]).append(p)
    for p in predictions:
        key=str(p['id'])
        if key in consumed:continue
        row=assignments.get(key,{})
        members=groups[row['dominant_gt_id']] if row.get('pure_fragment') else [p]
        consumed.update(str(m['id']) for m in members)
        ids=[str(m['id']) for m in members]
        boxes=np.asarray([m['bounds'] for m in members])
        output.append(dict(id='+'.join(ids),member_prediction_ids=ids,
            label=' + '.join(dict.fromkeys(m['label'] for m in members)),
            bounds=[boxes[:,0].min(0).tolist(),boxes[:,1].max(0).tolist()],
            confidence=min(float(m['confidence']) for m in members),
            group_assignment_gt_id=row.get('dominant_gt_id') if row.get('pure_fragment') else None))
    assert consumed=={str(p['id']) for p in predictions}
    assert sum(len(p['member_prediction_ids']) for p in output)==len(predictions)
    return output


def evaluate(predictions, truth, surfaces):
    from evaluate_hypersim import average_precision,box_overlap
    items=entities(predictions,surfaces)
    matrix=np.array([[box_overlap(np.asarray(g['bounds']),np.asarray(p['bounds']))[0]
                      for g in truth] for p in items]).reshape(len(items),len(truth))
    confidence=[p['confidence'] for p in items]
    result=dict(protocol='GAU_v1_annotation_assisted_pure_fragment_union',
        grouping_uses_ground_truth=True,construction_uses_ground_truth=False,
        official_benchmark_metric=False,purity_threshold=surfaces['dominant_assignment_purity'],
        distance_tolerance_m=surfaces['distance_tolerance_m'],
        bbox_source='union_of_saved_child_canonical_AABBs',confidence_rule='minimum_child_score',
        raw_prediction_count=len(predictions),statistical_entity_count=len(items),gt_count=len(truth),
        entities=items,one_to_one={},per_gt=[],
        limitation='Annotation-assisted grouping measures geometry at GT granularity. It is not independent instance-recognition AP or a predicted parent hierarchy. Mixed predictions are never split using GT. Unknown predictions retain FP liability; strict AP must also be reported.')
    for t in (.25,.5,.75):
        result[f'AP{int(t*100)}']=average_precision(matrix,confidence,t)
        score=np.where(matrix>=t,1+matrix/(min(matrix.shape)+1),-1e6)
        padded=np.concatenate([score,np.zeros((len(items),len(items)))],axis=1)
        ri,ci=linear_sum_assignment(-padded)
        matches=[dict(prediction_id=items[i]['id'],member_prediction_ids=items[i]['member_prediction_ids'],gt_id=truth[j]['id'],iou=float(matrix[i,j]))
                 for i,j in zip(ri,ci) if j<len(truth) and matrix[i,j]>=t]
        result['one_to_one'][str(t)]=dict(TP=len(matches),FP=len(items)-len(matches),FN=len(truth)-len(matches),matches=matches)
    for j,g in enumerate(truth):
        order=np.argsort(-matrix[:,j],kind='stable')[:3]
        result['per_gt'].append(dict(gt_id=g['id'],best_predictions=[dict(prediction_id=items[i]['id'],member_prediction_ids=items[i]['member_prediction_ids'],label=items[i]['label'],confidence=items[i]['confidence'],IoU=float(matrix[i,j])) for i in order]))
    return result
