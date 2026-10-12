"""Independent GT audit of saved 010 proposal screens; never builds candidates.

Uses exact per-mask histograms and the pinned evaluator's unchanged match/AP
functions. Large dense point-by-instance integer matrices are never created.
Counterfactuals are evaluation-only attribution and do not count as algorithms.
"""
import argparse, gc, hashlib, json, resource, sys, time
from pathlib import Path
from copy import deepcopy
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'scan200_work'))
from evaluate_scan200 import confidence_by_node, verify_evaluator
from scannet_utils.evaluate_semantic_instance import (
    scannet_eval, get_options, evaluate_matches, compute_averages)

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()

def compact_metrics(metrics):
    return {k: float(metrics[k]) for k in ('all_ap_25%', 'all_ap_50%', 'all_ap')}

def metrics_from_counts(counts, masses, scores, names, gt_ids, gt_mass):
    # These fields are exactly util_3d.Instance.to_dict's current defaults.
    truth = [dict(instance_id=int(k), label_id=3, vert_count=int(n),
                  med_dist=-1, dist_conf=0., matched_pred=[])
             for k, n in zip(gt_ids, gt_mass)]
    predictions = []
    for i in range(len(scores)):
        n = int(masses[i])
        if n < 100:
            continue
        pred = dict(filename=str(names[i]), pred_id=len(predictions), label_id=3,
                    vert_count=n, confidence=float(scores[i]),
                    void_intersection=int(n - counts[i].sum()))
        matched = []
        for g in np.flatnonzero(counts[i]):
            copied = {k: v for k, v in truth[g].items() if k != 'matched_pred'}
            copied['intersection'] = int(counts[i, g]); matched.append(copied)
            mirrored = pred.copy(); mirrored['intersection'] = int(counts[i, g])
            truth[g]['matched_pred'].append(mirrored)
        pred['matched_gt'] = matched; predictions.append(pred)
    matches = {0: dict(gt={'object': truth}, pred={'object': predictions})}
    opts = get_options()
    result = compute_averages(evaluate_matches(matches, ('object',), opts), opts, ('object',))
    # Count only nonmatched nonignored geometric FP entries, separately from
    # duplicate-match FP entries inside the unchanged evaluator.
    fp = {}
    for threshold in (.25, .5):
        values = []
        for p in predictions:
            qualifying = False; ignored = p['void_intersection']
            for g in p['matched_gt']:
                overlap = g['intersection'] / (g['vert_count'] + p['vert_count'] - g['intersection'])
                qualifying |= g['vert_count'] >= 100 and overlap > threshold
                if g['vert_count'] < 100: ignored += g['intersection']
            if not qualifying and ignored / p['vert_count'] <= threshold:
                values.append(dict(name=p['filename'], score=p['confidence'], points=p['vert_count']))
        fp[str(threshold)] = dict(nonmatched_nonignored=len(values),
            highest_score=sorted(values, key=lambda r: -r['score'])[:8])
    return compact_metrics(result), fp

def histogram(masks, gt, gt_ids):
    masses = np.empty(len(masks), np.int64)
    counts = np.empty((len(masks), len(gt_ids)), np.int64)
    for i, mask in enumerate(masks):
        masses[i] = int(mask.sum())
        counts[i] = np.bincount(gt[mask], minlength=4000)[gt_ids]
    return counts, masses

def iou_table(counts, masses, gt_mass):
    union = masses[:, None] + gt_mass[None, :] - counts
    return np.divide(counts, union, out=np.zeros_like(union, dtype=float), where=union > 0)

def geometry_summary(counts, masses, gt_mass, eligible):
    iou = iou_table(counts, masses, gt_mass)[:, eligible]
    best = iou.max(1) if len(iou) else np.zeros(0)
    active = masses >= 100
    return dict(predictions=len(masses), predictions_ge100=int(active.sum()),
        best_GT_IoU_le25=int(np.count_nonzero(active & (best <= .25))),
        best_GT_IoU_le50=int(np.count_nonzero(active & (best <= .5))),
        mixed_at_least_two_GT_IoU_gt10=int(np.count_nonzero(active & ((iou > .1).sum(1) >= 2))),
        GT_with_any_IoU_gt25=int(np.count_nonzero(iou.max(0) > .25)),
        GT_with_any_IoU_gt50=int(np.count_nonzero(iou.max(0) > .5)),
        GT_with_counted_prediction_IoU_gt25=int(np.count_nonzero(iou[active].max(0) > .25)),
        GT_with_counted_prediction_IoU_gt50=int(np.count_nonzero(iou[active].max(0) > .5)),
        any_prediction_counts_include_under100_masks=True,
        best_recall_is_not_one_to_one_AP_recall=True)

def equivalence_check():
    rng = np.random.default_rng(1701)
    gt = rng.choice([0, 3001, 3002, 3003, 3004], 1500)
    masks = rng.random((9, len(gt))) < rng.uniform(.1, .8, (9, 1))
    scores = rng.random(9); ids, mass = np.unique(gt[gt > 0], return_counts=True)
    counts, sizes = histogram(masks, gt, ids)
    expected = compact_metrics(scannet_eval([{str(i): dict(mask=m, label_id=3, conf=float(s))
        for i, (m, s) in enumerate(zip(masks, scores))}], [gt], None, (3,), ('object',), {3: 'object'}))
    result, _ = metrics_from_counts(counts, sizes, scores, list(range(len(scores))), ids, mass)
    assert result == expected, (result, expected)
    return dict(exact_matches_to_fixed_native_evaluator=True, seed=1701,
                synthetic_points=1500, synthetic_predictions=9)

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', default=str(HERE/'audit_010.json'))
    args = parser.parse_args(); started = time.time(); verify_evaluator()
    checks = equivalence_check()
    work = ROOT/'esam_hypersim_work'
    comparison = json.loads((work/'comparison.json').read_text())
    source = Path('/home/lewisliu/datasets/scannet-sg-processed/partaware_v19/hypersim/ai_001_010')
    base_target = Path('/home/lewisliu/datasets/scannet-sg-processed/research_cycle/sage_domain_transfer/hypersim/ai_001_010')
    receipt = json.loads((base_target/'candidate_receipt.json').read_text())
    image_construction = json.loads((ROOT/'v20_work/image_confirmed_construction.json').read_text())
    image_row = next(r for r in image_construction['results'] if r['scene'] == 'ai_001_010')
    image_evidence = json.loads((Path(image_row['masks_file']).parent/'image_evidence.json').read_text())
    confirmed = np.flatnonzero([r['confirmed'] for r in image_evidence])
    with np.load(work/'observed_domain.npz', allow_pickle=False) as z: xyz=z['xyz']
    with np.load(work/'evaluation_GT.npz', allow_pickle=False) as z: gt=z['gt']
    gt_ids, gt_mass = np.unique(gt[gt > 0], return_counts=True); eligible = gt_mass >= 100
    truth = {3001+r['id']:r for r in json.loads((work/'GT_receipt.json').read_text())['objects']}
    graph = json.loads((source/'topology_map.json').read_text()); keys=list(graph['object_nodes']['nodes'])
    assert keys == receipt['native_keys']
    cloud_name=graph['geometry_provenance']['source']; assert Path(cloud_name).name == cloud_name
    cloud_path=source/cloud_name; assert sha(cloud_path)==receipt['inputs']['source_cloud_sha256']
    cloud=o3d.io.read_point_cloud(str(cloud_path)); points=np.asarray(cloud.points)
    rgb=np.rint(np.asarray(cloud.colors)*255).astype(np.int64)
    raw_ids=rgb[:,0]+255*rgb[:,1]+255**2*rgb[:,2]
    distance, nearest=cKDTree(points).query(xyz, workers=2)
    labels=raw_ids[nearest]; labels[distance>.05]=-1
    native_owner=np.full(len(xyz), -1, np.int16)
    na=len(keys); counts_a=np.zeros((na,len(gt_ids)),np.int64); mass_a=np.zeros(na,np.int64)
    for i,key in enumerate(keys):
        mask=labels==int(key); native_owner[mask]=i; mass_a[i]=int(mask.sum())
        counts_a[i]=np.bincount(gt[mask],minlength=4000)[gt_ids]
    del cloud, points, rgb, raw_ids, distance, nearest, labels; gc.collect()
    confidences=confidence_by_node(source)
    sa=np.array([confidences.get(k,float(graph['object_nodes']['nodes'][k].get('confidence',0))) for k in keys])
    tracks=json.loads((source/'validated_object_tracks.json').read_text())
    build_sa=np.array([tracks.get(k,{}).get('confidence',graph['object_nodes']['nodes'][k].get('confidence',0)) for k in keys])
    assert np.array_equal(sa,build_sa), 'Builder score source differs from frozen baseline'
    assert sha(work/'ESAM_common_domain_masks.npz')==receipt['inputs']['peer_sha256']
    with np.load(work/'ESAM_common_domain_masks.npz',allow_pickle=False) as z: b=z['masks']; sb=z['scores']
    counts_b,mass_b=histogram(b,gt,gt_ids)
    intersection=np.zeros((na,len(b)),np.int64)
    for j,mask in enumerate(b):
        inter=np.bincount(native_owner[mask].astype(np.int64)+1,minlength=na+1)
        intersection[:,j]=inter[1:]
    union=mass_a[:,None]+mass_b[None,:]-intersection
    ab=np.divide(intersection,union,out=np.zeros_like(union,dtype=float),where=union>0)
    qa=.5*(sa+ab.max(1));qb=.5*(sb+ab.max(0))
    names_a=['native:'+k for k in keys]; names_b=['peer:'+str(i) for i in range(len(b))]
    def evaluate(c,m,s,n):return metrics_from_counts(c,m,s,n,gt_ids,gt_mass)
    native_metrics,native_fp=evaluate(counts_a,mass_a,sa,names_a)
    frozen=compact_metrics(next(r['point_metrics'] for r in comparison['results'] if r['profile']=='GPT v19 saved'))
    assert all(abs(native_metrics[k]-frozen[k])<1e-12 for k in frozen)
    result=dict(complete=False,evaluation_only=True,GT_in_construction=False,
        no_candidate_or_production_mutation=True,API_calls=0,training=False,
        counterfactuals_are_attribution_not_new_algorithm_attempts=True,
        scene='ai_001_010',domain_points=len(gt),point_eligible_GT=int(eligible.sum()),
        historical_full_box_GT=len(truth),historical_strict_box_metrics=dict(AP25=.665310717064037,AP50=.4354798656162385),
        new_strict_box_not_computed_because_screens_have_no_canonical_published_boxes=True,
        metric_protocol='pinned native OpenMMLab evaluator; .50-.90 mean; 100-point GT/pred criterion; fixed 5cm transfer',
        sparse_equivalence_check=checks,
        baseline=dict(metrics=native_metrics,geometry=geometry_summary(counts_a,mass_a,gt_mass,eligible),FP=native_fp),
        peer=dict(geometry=geometry_summary(counts_b,mass_b,gt_mass,eligible)),
        image_confirmed_peer=dict(count=int(len(confirmed)),geometry=geometry_summary(counts_b[confirmed],mass_b[confirmed],gt_mass,eligible)),
        inputs={name:sha(work/name) for name in ['comparison.json','observed_domain.npz','evaluation_GT.npz','ESAM_common_domain_masks.npz']},
        source_cloud_sha256=sha(cloud_path),graph_sha256=sha(source/'topology_map.json'),
        script_sha256=sha(__file__),variants=[])
    iou_a=iou_table(counts_a,mass_a,gt_mass);iou_b=iou_table(counts_b,mass_b,gt_mass)
    baseline_best=iou_a[:,eligible].max(0)
    baseline_counted_best=iou_a[mass_a>=100][:,eligible].max(0)
    baseline_gt_for_native=np.flatnonzero(eligible)[iou_a[:,eligible].argmax(1)]
    # Frozen identity pair for attribution: a native proposal's best eligible
    # GT. This does not introduce matching/GT into a construction function.
    correct=np.zeros(len(gt),bool)
    for i,g in enumerate(baseline_gt_for_native):correct|=(native_owner==i)&(gt==gt_ids[g])
    result['baseline']['correct_points_against_fixed_best_GT']=int(correct.sum())
    result['ranking']=dict(native_score_vs_best_IoU_spearman=float(spearmanr(sa[mass_a>=100],iou_a[:,eligible].max(1)[mass_a>=100]).statistic),
        SAGE_native_q_vs_best_IoU_spearman=float(spearmanr(qa[mass_a>=100],iou_a[:,eligible].max(1)[mass_a>=100]).statistic),
        peer_raw_score_vs_best_IoU_spearman=float(spearmanr(sb,iou_b[:,eligible].max(1)).statistic),
        peer_SAGE_q_vs_best_IoU_spearman=float(spearmanr(qb,iou_b[:,eligible].max(1)).statistic))
    result['counterfactuals']={'v19_geometry_SAGE_native_scores_only':evaluate(counts_a,mass_a,qa,names_a)[0],
        'v19_geometry_plus_all_raw_peers_no_NMS_no_ownership':evaluate(np.r_[counts_a,counts_b],np.r_[mass_a,mass_b],np.r_[sa,sb],names_a+names_b)[0]}
    print('FROZEN_AND_PEER_AUDIT',json.dumps(dict(baseline=result['baseline']['geometry'],peer=result['peer'],confirmed=result['image_confirmed_peer'],counterfactuals=result['counterfactuals'])),flush=True)
    variants=[dict(mode=r['mode'],path=Path(r['masks_file']),audit=r['audit'],peer_indices=np.arange(len(b)),sha256=r['masks_sha256']) for r in receipt['results']]
    variants.append(dict(mode='image_confirmed_core',path=Path(image_row['masks_file']),audit=image_row['audit'],peer_indices=confirmed,sha256=image_row['masks_sha256']))
    for v in variants:
        peer_idx=v['peer_indices']; ab_v=ab[:,peer_idx]
        qa_v=.5*(sa+ab_v.max(1));qb_v=.5*(sb[peer_idx]+ab_v.max(0))
        all_c=np.r_[counts_a,counts_b[peer_idx]];all_m=np.r_[mass_a,mass_b[peer_idx]]
        all_s=np.r_[qa_v,qb_v];all_names=names_a+[names_b[j] for j in peer_idx]
        kept=np.array(v['audit']['kept']);suppressed=np.array(v['audit']['suppressed'])
        dropped_small=np.setdiff1d(np.arange(len(all_m)),np.r_[kept,suppressed])
        before_metrics,before_fp=evaluate(all_c[kept],all_m[kept],all_s[kept],[all_names[i] for i in kept])
        native_kept=kept[kept<na]
        suppress_rows=[]
        for i in suppressed:
            g=int(iou_a[i,eligible].argmax()) if i<na else int(iou_b[peer_idx[i-na],eligible].argmax())
            eligible_g=np.flatnonzero(eligible)[g]
            suppress_rows.append(dict(name=all_names[i],points=int(all_m[i]),score=float(all_s[i]),
                best_GT_id=int(gt_ids[eligible_g]-3001),GT_label=truth[int(gt_ids[eligible_g])]['label'],
                best_GT_IoU=float(iou_table(all_c[i:i+1],all_m[i:i+1],gt_mass)[0,eligible_g])))
        assert sha(v['path'])==v['sha256']
        with np.load(v['path'],allow_pickle=False) as z:masks=z['masks'];scores=z['scores']
        assert np.allclose(scores,all_s[kept],rtol=0,atol=1e-12)
        counts,masses=histogram(masks,gt,gt_ids); final_metrics,final_fp=evaluate(counts,masses,scores,[all_names[i] for i in kept])
        ownership=np.full(len(gt),-1,np.int16)
        for i,mask in enumerate(masks):
            assert not np.any(ownership[mask]>=0);ownership[mask]=kept[i]
        moved=correct & (ownership!=native_owner)
        retained_native=kept<na
        original_scores=np.array([sa[k] if k<na else sb[peer_idx[k-na]] for k in kept])
        post_native_metrics=evaluate(counts[retained_native],masses[retained_native],sa[kept[retained_native]],
            [all_names[i] for i in kept[retained_native]])[0]
        cf=dict(v19_geometry_native_NMS_only_original_scores=evaluate(counts_a[native_kept],mass_a[native_kept],sa[native_kept],[names_a[i] for i in native_kept])[0],
            post_NMS_original_geometry_SAGE_scores=before_metrics,
            final_geometry_original_branch_scores=evaluate(counts,masses,original_scores,[all_names[i] for i in kept])[0],
            final_native_geometry_only_original_native_scores=post_native_metrics)
        final_iou=iou_table(counts,masses,gt_mass);final_best=final_iou[:,eligible].max(0)
        final_counted_best=final_iou[masses>=100][:,eligible].max(0)
        regressions=[]
        for eg,g in enumerate(np.flatnonzero(eligible)):
            bidx=int(iou_a[:,g].argmax());fidx=int(final_iou[:,g].argmax())
            if baseline_best[eg]-final_best[eg]>.05:
                native_output=np.flatnonzero(kept==bidx)
                native_iou=float(final_iou[native_output[0],g]) if len(native_output) else 0.
                native_retained_points=int(counts[native_output[0],g]) if len(native_output) else 0
                regressions.append(dict(GT_id=int(gt_ids[g]-3001),GT_label=truth[int(gt_ids[g])]['label'],
                    GT_points=int(gt_mass[g]),baseline_native=names_a[bidx],baseline_IoU=float(iou_a[bidx,g]),
                    final_best_name=all_names[kept[fidx]],final_best_IoU=float(final_iou[fidx,g]),
                    original_identity_IoU_after=float(native_iou),baseline_correct_points=int(counts_a[bidx,g]),
                    original_identity_correct_points_after=native_retained_points,
                    cause='native_NMS_suppression' if bidx in suppressed else 'point_ownership_competition',
                    final_best_purity=float(counts[fidx,g]/max(masses[fidx],1)),
                    final_best_GT_coverage=float(counts[fidx,g]/gt_mass[g])))
        row=dict(mode=v['mode'],saved_masks_sha256=v['sha256'],metrics=final_metrics,
            geometry=geometry_summary(counts,masses,gt_mass,eligible),FP=final_fp,
            native_NMS_suppressed=suppress_rows,native_or_peer_under100_skipped=[all_names[i] for i in dropped_small],
            original_geometry_after_NMS=geometry_summary(all_c[kept],all_m[kept],gt_mass,eligible),
            original_geometry_after_NMS_FP=before_fp,
            counterfactuals=cf,
            native_point_ownership=dict(native_points_original=int((native_owner>=0).sum()),
                baseline_correct_points=int(correct.sum()),correct_points_moved=int(moved.sum()),
                moved_to_peer=int(np.count_nonzero(moved&(ownership>=na))),
                moved_to_other_native=int(np.count_nonzero(moved&(ownership>=0)&(ownership<na))),
                moved_to_unowned=int(np.count_nonzero(moved&(ownership<0)))),
            GT_crossed_down_25=int(np.count_nonzero((baseline_best>.25)&(final_best<=.25))),
            GT_crossed_down_50=int(np.count_nonzero((baseline_best>.5)&(final_best<=.5))),
            GT_crossed_up_25=int(np.count_nonzero((baseline_best<=.25)&(final_best>.25))),
            GT_crossed_up_50=int(np.count_nonzero((baseline_best<=.5)&(final_best>.5))),
            GT_counted_prediction_crossed_down_25=int(np.count_nonzero((baseline_counted_best>.25)&(final_counted_best<=.25))),
            GT_counted_prediction_crossed_down_50=int(np.count_nonzero((baseline_counted_best>.5)&(final_counted_best<=.5))),
            native_kept_ge100_before_but_under100_after=int(np.count_nonzero(retained_native&(all_m[kept]>=100)&(masses<100))),
            GT_regressions_gt5pp=sorted(regressions,key=lambda r:r['final_best_IoU']-r['baseline_IoU']))
        result['variants'].append(row)
        print('VARIANT_AUDIT',v['mode'],json.dumps({k:row[k] for k in ['metrics','geometry','counterfactuals','native_point_ownership','GT_crossed_down_25','GT_crossed_down_50']}),flush=True)
        del masks,ownership,counts,masses;gc.collect()
        Path(args.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    result['complete']=True;result['elapsed_seconds']=time.time()-started
    result['peak_RSS_MiB']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    result['practical_limits']=['Best-IoU diagnostics are not one-to-one AP recall.',
        'Only repeated-development 010 is diagnosed; no full Hypersim benchmark claim.',
        'All candidates contain existing observed points, so no hidden-surface accuracy is demonstrated.',
        'Historical canonical box protocol is unchanged; saved screens have no new canonical publication.',
        'Attribution counterfactuals are not candidate build methods and cannot count toward the fifteen-attempt gate.']
    Path(args.output).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('AUDIT_COMPLETE',result['peak_RSS_MiB'],result['elapsed_seconds'],flush=True)

if __name__=='__main__':main()
