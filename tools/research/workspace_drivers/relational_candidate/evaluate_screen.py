"""Independent evaluation of saved relational masks; GT never enters builder."""
import gc,hashlib,importlib.util,json,resource,sys,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'scan200_work'))
import evaluate_scan200 as reference
spec=importlib.util.spec_from_file_location('independent_010_audit',ROOT/'v20_work/cross_domain_audit/audit_010.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def compact(m):return {k:float(m[k]) for k in ['all_ap_25%','all_ap_50%','all_ap']}

def main():
    started=time.time();reference.verify_evaluator();equivalence=audit.equivalence_check()
    construction=json.loads((HERE/'construction.json').read_text());assert construction['complete']
    historical_v19=json.loads((ROOT/'scan200_work/evaluation_results.json').read_text())
    signed_V=json.loads((ROOT/'v20_work/sage_verified_core_evaluation.json').read_text())
    hypersim=json.loads((ROOT/'esam_hypersim_work/comparison.json').read_text())
    domain_screen=json.loads((ROOT/'v20_work/010_sage_domain_screen.json').read_text())
    results=[]
    for row in construction['results']:
        if row['dataset']=='scannet':
            _,gt,truth,gt_hashes=reference.ground_truth(row['scene'])
            v19=compact(next(r['metrics'] for r in historical_v19['results'] if r['scene_id']==row['scene'] and r['profile']=='v19'))
            control=compact(next(r['metrics'] for r in signed_V['results'] if r['scene_id']==row['scene']))
        else:
            with np.load(ROOT/'esam_hypersim_work/evaluation_GT.npz',allow_pickle=False) as z:gt=z['gt']
            gt_hashes={'evaluation_GT_npz_sha256':sha(ROOT/'esam_hypersim_work/evaluation_GT.npz')}
            v19=compact(next(r['point_metrics'] for r in hypersim['results'] if r['profile']=='GPT v19 saved'))
            control=compact(next(r['metrics'] for r in domain_screen['results'] if r['mode']=='exclusive_core'))
        gt_ids,gt_mass=np.unique(gt[gt>0],return_counts=True);eligible=gt_mass>=100
        variants=[]
        for item in row['results']:
            path=Path(item['masks_file']);assert sha(path)==item['masks_sha256']
            with np.load(path,allow_pickle=False) as z:masks=z['masks'];scores=z['scores']
            counts,mass=audit.histogram(masks,gt,gt_ids)
            names=['native:'+row['results'][0]['native_keys'][k] if k<len(row['results'][0]['native_keys'])
                else 'peer:'+str(item['peer_indices'][k-len(row['results'][0]['native_keys'])]) for k in item['audit']['kept']]
            metrics,FP=audit.metrics_from_counts(counts,mass,scores,names,gt_ids,gt_mass)
            geometry=audit.geometry_summary(counts,mass,gt_mass,eligible)
            iou=audit.iou_table(counts,mass,gt_mass);best=np.zeros(len(gt_ids))
            if np.any(mass>=100):best=iou[mass>=100].max(0)
            variants.append(dict(mode=item['mode'],metrics=metrics,geometry=geometry,FP=FP,
                best_counted_GT_IoU={str(int(g)):float(v) for g,v in zip(gt_ids[eligible],best[eligible])},
                masks_sha256=item['masks_sha256']))
            del masks,counts,mass;gc.collect()
        control_reproduced=all(abs(variants[0]['metrics'][k]-control[k])<1e-12 for k in control)
        assert control_reproduced, ('Signed control failed to reproduce fixed screen/publication',row['scene'],variants[0]['metrics'],control)
        candidate=variants[1]['metrics'];delta_v19={k:100*(candidate[k]-v19[k]) for k in v19}
        delta_control={k:100*(candidate[k]-control[k]) for k in control}
        results.append(dict(scene=row['scene'],dataset=row['dataset'],v19_metrics=v19,signed_CORE_V_metrics=control,
            signed_control_exactly_reproduced=control_reproduced,variants=variants,
            delta_percentage_points_vs_v19=delta_v19,delta_percentage_points_vs_signed_CORE_V=delta_control,
            all_three_metrics_above_v19=all(v>0 for v in delta_v19.values()),
            all_three_metrics_no_worse_than_signed_CORE_V=all(v>=-1e-10 for v in delta_control.values()),
            cannot_link_pairs=row['cannot_link_pairs'],deferred_peer=row['peer_proposals']-row['admitted_peer'],GT_hashes=gt_hashes))
        print('RELATIONAL_EVALUATION',row['scene'],json.dumps(dict(control=control,candidate=candidate,delta_v19=delta_v19,delta_control=delta_control)),flush=True)
        del gt;gc.collect()
    complete=dict(complete=True,candidate_screen_only=True,full_pipeline_replay=False,
        GT_in_construction=False,API_calls=0,training=False,GPU=False,
        one_general_policy_all_four_domains=True,signed_control_exactly_reproduced_all=True,
        eligible_for_default_upgrade=False,strict_box_010_new_result=None,
        historical_strict_box_010_unchanged={'AP25':.665310717064037,'AP50':.4354798656162385,'GT':109},
        positive_screen_requires_full_published_cloud_graph_and_strictbox_replay=True,
        results=results,evaluator_sparse_equivalence_check=equivalence,
        construction_sha256=sha(HERE/'construction.json'),evaluator_script_sha256=sha(__file__),
        elapsed_seconds=time.time()-started,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        attempted_hypothesis_completed_on_four_domains=True,
        does_not_validate_new_five_scans_or_complete_eight_scan_upgrade=True)
    (HERE/'evaluation.json').write_text(json.dumps(complete,indent=2)+'\n')
    print('RELATIONAL_EVALUATION_COMPLETE',complete['elapsed_seconds'],complete['peak_RSS_MiB'],flush=True)
if __name__=='__main__':main()
