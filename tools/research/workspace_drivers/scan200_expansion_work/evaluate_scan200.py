"""Class-agnostic POINT-mask evaluation; no GT grouping or box surrogate.

The pinned OpenMMLab ScanNet evaluator is loaded verbatim. Output masks are
transferred to the official reconstructed vertices in the raw scan coordinates.
"""
import argparse,csv,hashlib,json,sys
from pathlib import Path
import numpy as np,open3d as o3d
from scipy.spatial import cKDTree

WORK=Path(__file__).resolve().parent
sys.path.insert(0,str(WORK/'reference/mmdet3d/evaluation/functional'))
from scannet_utils.evaluate_semantic_instance import scannet_eval,get_options
sys.path.insert(0,'/home/lewisliu/ScanNet/BenchmarkScripts/ScanNet200')
from scannet200_constants import VALID_CLASS_IDS_200
VALID=set(VALID_CLASS_IDS_200)-{1,3}  # raw-category wall/floor are stuff
TRANSFER_RADIUS=.05

def smooth_labels(labels,segments):
    """Author ESAM's fixed per-mesh-segment majority threshold (>0.5).

    Geometry regions only: no GT instance groups, semantic labels or IoUs.
    For mutually exclusive graph ownership, at most one label wins a segment.
    """
    _,group=np.unique(segments,return_inverse=True)
    count=np.bincount(group)
    output=np.full(labels.shape,-1,dtype=labels.dtype)
    for label in np.unique(labels):
        if label<0:continue
        positive=np.bincount(group[labels==label],minlength=len(count))
        wins=positive>count*.5
        output[wins[group]]=label
    return output

def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda:file.read(8*1024**2),b''):digest.update(block)
    return digest.hexdigest()

def ground_truth(scene):
    raw=Path('/home/lewisliu/datasets/scannet/scans')/scene
    mesh_path=raw/(scene+'_vh_clean_2.ply')
    mesh=o3d.io.read_triangle_mesh(str(mesh_path));vertices=np.asarray(mesh.vertices)
    segments_path=raw/(scene+'_vh_clean_2.0.010000.segs.json')
    aggregation_path=raw/(scene+'.aggregation.json')
    segments=np.array(json.loads(segments_path.read_text())['segIndices'])
    assert len(vertices)==len(segments) and np.isfinite(vertices).all()
    mapping_path=WORK/'reference/data/scannet200-mv/meta_data/scannetv2-labels.combined.tsv'
    with mapping_path.open() as file:mapping={r['raw_category']:int(r['id']) for r in csv.DictReader(file,delimiter='\t')}
    ids=np.zeros(len(vertices),dtype=np.int64);rows=[]
    for group in json.loads(aggregation_path.read_text())['segGroups']:
        category=mapping.get(group['label'],0)
        if category not in VALID:continue
        mask=np.isin(segments,group['segments'])
        assert not np.any(ids[mask]),'GT groups overlap'
        encoded=3000+int(group['objectId'])+1
        assert encoded<4000
        ids[mask]=encoded
        rows.append(dict(gt_id=int(group['objectId']),raw_label=group['label'],category_id=category,
                         encoded_id=encoded,vertices=int(mask.sum()),eligible=int(mask.sum())>=100))
    return vertices,ids,rows,dict(mesh_sha256=sha(mesh_path),segments_sha256=sha(segments_path),
        aggregation_sha256=sha(aggregation_path),mapping_sha256=sha(mapping_path))

def confidence_by_node(target):
    tracks_path=target/'validated_object_tracks.json'
    if not tracks_path.exists():tracks_path=target/'object_tracks.json'
    tracks=json.loads(tracks_path.read_text()) if tracks_path.exists() else {}
    scores={str(k):float(v['confidence']) for k,v in tracks.items()}
    observations={}
    for path in sorted((target/'refined_instance').glob('*_updated_instance.json')):
        for row in json.loads(path.read_text()):
            observations.setdefault(str(row['instance_id']),[]).append(float(row.get('confidence',0)))
    for key,values in observations.items():scores.setdefault(key,float(np.mean(values)))
    return scores

def result_for(scene,target):
    vertices,gt,gt_rows,gt_hashes=ground_truth(scene)
    graph_path=target/'topology_map.json';graph=json.loads(graph_path.read_text())
    geometry=graph.get('geometry_provenance',{}).get('source','instance_cloud_cleaned.ply')
    assert Path(geometry).name==geometry and geometry.endswith('.ply')
    cloud_path=target/geometry
    assert cloud_path.is_file(),'Evaluate the actual final published point cloud'
    cloud=o3d.io.read_point_cloud(str(cloud_path))
    points=np.asarray(cloud.points)
    colors=np.round(np.asarray(cloud.colors)*255).astype(np.int64)
    source_ids=colors[:,0]+255*colors[:,1]+255**2*colors[:,2]
    assert len(points) and np.isfinite(points).all()
    distance,index=cKDTree(points).query(vertices,workers=2)
    # Fixed physical guard prevents a distant object filling unobserved walls.
    transferred=source_ids[index].copy();transferred[distance>TRANSFER_RADIUS]=-1
    raw=Path('/home/lewisliu/datasets/scannet/scans')/scene
    segments=np.asarray(json.loads((raw/(scene+'_vh_clean_2.0.010000.segs.json')).read_text())['segIndices'])
    transferred=smooth_labels(transferred,segments)
    scores=confidence_by_node(target)
    pred={};pred_rows=[]
    for key,node in graph['object_nodes']['nodes'].items():
        mask=transferred==int(key)
        score=scores.get(str(key),float(node.get('confidence',0)))
        assert np.isfinite(score)
        pred[f'{scene}_{key}']=dict(mask=mask,label_id=3,conf=score)
        pred_rows.append(dict(node_id=str(key),label=node['name'],confidence=score,vertices=int(mask.sum())))
    metrics=scannet_eval([pred],[gt],None,(3,),('object',),{3:'object'})
    for row in gt_rows:
        mask=gt==row['encoded_id'];best=None
        for candidate,info in zip(pred_rows,pred.values()):
            intersection=int(np.count_nonzero(mask&info['mask']));union=row['vertices']+candidate['vertices']-intersection
            iou=intersection/union if union else 0
            if best is None or iou>best['point_IoU']:best=dict(node_id=candidate['node_id'],label=candidate['label'],point_IoU=iou)
        row['best_prediction']=best
    record=dict(scene_id=scene,processed_scene=str(target),metrics=metrics,
        predicted_objects=len(pred),eligible_GT_instances=sum(r['eligible'] for r in gt_rows),GT=gt_rows,predictions=pred_rows,
        graph_sha256=sha(graph_path),prediction_cloud=str(cloud_path),prediction_cloud_sha256=sha(cloud_path),GT_hashes=gt_hashes,
        protocol='class_agnostic_point_masks_full_official_reconstruction',GT_grouping=False,GT_in_construction=False,
        transfer=dict(method='nearest canonical instance point within fixed 5cm',radius_m=TRANSFER_RADIUS,
            within_radius_fraction=float(np.mean(distance<=TRANSFER_RADIUS)),GT_registration=False,
            segment_smoothing='author ESAM geometric regions, >0.5 majority; no instance GT'),
        scope='Validation pilot, not full 312-scan benchmark; source sampling/point transfer differ by native output representation.')
    return record,pred,gt

def esam_result_for(item):
    scene=item['scene'];target=Path(item['output'])
    vertices,gt,gt_rows,hashes=ground_truth(scene)
    with np.load(target/'predictions.npz') as output:
        masks=output['masks'];scores=output['scores']
    assert masks.dtype==np.bool_ and masks.shape==(len(scores),len(vertices))
    pred={f'{scene}_ESAM_{i}':dict(mask=m,label_id=3,conf=float(s)) for i,(m,s) in enumerate(zip(masks,scores))}
    metrics=scannet_eval([pred],[gt],None,(3,),('object',),{3:'object'})
    for row in gt_rows:
        mask=gt==row['encoded_id'];inter=(masks&mask).sum(1)
        unions=masks.sum(1)+row['vertices']-inter
        ious=np.divide(inter,unions,out=np.zeros(len(inter),dtype=float),where=unions>0)
        best=int(np.argmax(ious)) if len(ious) else None
        row['best_prediction']=dict(node_id=str(best),label='class-agnostic object',point_IoU=float(ious[best])) if best is not None else None
    record=dict(scene_id=scene,profile='ESAM',frames=item['frames'],processed_scene=str(target),
        metrics=metrics,predicted_objects=len(pred),eligible_GT_instances=sum(r['eligible'] for r in gt_rows),GT=gt_rows,
        GT_hashes=hashes,protocol='class_agnostic_point_masks_full_official_reconstruction',
        GT_grouping=False,GT_in_construction=False,prediction_sha256=sha(target/'predictions.npz'),
        checkpoint_sha256=item['checkpoint_sha256'],transfer='author full-frame 1NN + geometric segment majority >0.5')
    return record,pred,gt

def verify_evaluator():
    gt=np.repeat([3001,3002],200)
    perfect={f'perfect_{i}':dict(mask=gt==3000+i,label_id=3,conf=1.-i*.01) for i in [1,2]}
    score=scannet_eval([perfect],[gt],None,(3,),('object',),{3:'object'})
    assert np.isclose(score['all_ap'],1.) and np.isclose(score['all_ap_25%'],1.)
    missed=scannet_eval([{}],[gt],None,(3,),('object',),{3:'object'})
    assert missed['all_ap']==0.
    merged={'merged':dict(mask=np.ones(400,dtype=bool),label_id=3,conf=1.)}
    merge_score=scannet_eval([merged],[gt],None,(3,),('object',),{3:'object'})
    assert merge_score['all_ap_50%']==0.,'A merged mask at exactly 0.5 must not pass strict > threshold'
    print('OFFICIAL_EVALUATOR_VERIFIED',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true')
    parser.add_argument('--construction-record');parser.add_argument('--esam-record');args=parser.parse_args()
    verify_evaluator()
    if args.construction_record:
        construction=json.loads(Path(args.construction_record).read_text());assert construction['complete']
        results=[];combined={}
        for item in construction['results']:
            record,pred,gt=result_for(item['scene'],Path(item['output']))
            record['profile']=item['profile'];record['frames']=item['frames'];results.append(record)
            combined.setdefault(item['profile'],[]).append((pred,gt))
            (Path(item['output'])/'scannet200_evaluation.json').write_text(json.dumps(record,indent=2)+'\n')
            print('POINT_AP',item['scene'],item['profile'],record['metrics'],flush=True)
        if args.esam_record:
            baseline=json.loads(Path(args.esam_record).read_text());assert baseline['complete']
            for item in baseline['results']:
                record,pred,gt=esam_result_for(item);results.append(record)
                combined.setdefault('ESAM',[]).append((pred,gt))
                (Path(item['output'])/'scannet200_evaluation.json').write_text(json.dumps(record,indent=2)+'\n')
                print('POINT_AP',item['scene'],'ESAM',record['metrics'],flush=True)
        aggregate={profile:scannet_eval([x[0] for x in rows],[x[1] for x in rows],None,(3,),('object',),{3:'object'}) for profile,rows in combined.items()}
        output=dict(registration=construction['registration'],results=results,aggregate=aggregate,complete=True,
                    official_evaluator=json.loads((WORK/'scannet_evaluator_source.json').read_text()))
        (WORK/'evaluation_results.json').write_text(json.dumps(output,indent=2)+'\n')
