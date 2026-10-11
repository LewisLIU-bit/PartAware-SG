"""Independent post-publication visualization and annotation review.

This module never returns geometry to construction. The official evaluation
is performed only on the saved final result, after original-viewer rendering.
"""
import hashlib,json,sys
from pathlib import Path


def summarize(evaluation):
    strict=evaluation['one_to_one_bbox_geometry']
    matched={key:{str(r['prediction_id']) for r in value['matches']} for key,value in strict.items()}
    rows=[]
    for row in evaluation['object_error_diagnostic']['per_gt']:
        best=row['best_predictions'][:1]
        overlap=best[0]['IoU'] if best else 0.
        rows.append(dict(gt_id=row['gt_id'],official_label=row['label'],best_prediction=best[0] if best else None,
            strict_matches=row['one_to_one_matches'],
            diagnosis='未达到25%框匹配，需核查遗漏、碎片、错误归属或框尺度' if overlap<.25 else
                '仅达到25%，需核查实测覆盖及实例边界' if overlap<.5 else
                '达到50%，仍未达到75%' if overlap<.75 else '最佳候选达到75%；仍须核对严格一对一结果'))
    unmatched=[dict(prediction_id=r['id'],label=r['label'],confidence=r['confidence'],
        reason='25%严格一对一未分配；可能是错位、重复、分割粒度或GT范围差异，不能仅凭此断言物体不存在')
        for r in evaluation['prediction_objects'] if str(r['id']) not in matched['0.25']]
    aligned=evaluation.get('granularity_aligned_AP')
    if aligned:
        aligned_rows={r['gt_id']:r for r in aligned['per_gt']}
        for row in rows:
            row['granularity_aligned_best_predictions']=aligned_rows[row['gt_id']]['best_predictions']
            row['granularity_aligned_matches']={t:next((m for m in value['matches'] if m['gt_id']==row['gt_id']),None) for t,value in aligned['one_to_one'].items()}
    return dict(protocol='independent_observed_hypersim_geometry_review',construction_consumes_review=False,
        ground_truth_in_construction=False,graph_sha256=evaluation['graph_sha256'],
        AP25=evaluation['geometry_only_box_AP25'],AP50=evaluation['geometry_only_box_AP50'],AP75=evaluation['geometry_only_box_AP75'],
        strict_counts={key:{field:value[field] for field in ('TP','FP','FN')} for key,value in strict.items()},
        per_gt=rows,unmatched_predictions=unmatched,
        granularity_aligned_AP={k:aligned[k] for k in ('protocol','grouping_uses_ground_truth','AP25','AP50','AP75','statistical_entity_count')} if aligned else None)


def construct(context):
    if not context.manifest:return
    metadata=json.loads(context.manifest.read_text())
    if not metadata.get('ground_truth_source'):return
    saved=context.environment.copy()
    context.environment.update(XDG_SESSION_TYPE='x11',LIBGL_ALWAYS_SOFTWARE='true',OMP_NUM_THREADS='1')
    context.environment.pop('WAYLAND_DISPLAY',None);context.environment.pop('WAYLAND_SOCKET',None)
    try:
        graph=json.loads((context.scene/'topology_map.json').read_text())
        geometry=context.scene/graph.get('geometry_provenance',{}).get('source','instance_cloud_completed.ply')
        if not geometry.is_file():geometry=context.scene/'instance_cloud_cleaned.ply'
        for name,front in [('final_scene.png',[.15,-.85,.5]),('final_scene_reverse.png',[-.15,.85,.5])]:
            context.execute([sys.executable,str(context.repo/'script/visualize_map.py'),
                '--map_ply_path',str(geometry),
                '--topology_map_path',str(context.scene/'topology_map.json'),
                '--show_bboxes','--node_radius','.02','--screenshot',str(context.scene/name),
                '--view-front',*map(str,front)],'原版可视化后独立核对真值')
        evaluation=context.scene/'evaluation.json'
        context.execute([sys.executable,str(context.repo/'scannet/script/evaluate_hypersim.py'),
            '--manifest',str(context.manifest),'--processed-scene',str(context.scene),'--output',str(evaluation)],
            '正式发布后独立GT评价；不回写构建结果')
        report=summarize(json.loads(evaluation.read_text()))
        report['visualizations']=[dict(path=str(context.scene/name),sha256=hashlib.sha256((context.scene/name).read_bytes()).hexdigest())
            for name in ('final_scene.png','final_scene_reverse.png')]
        (context.scene/'scene_review_zh.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        context.event('可视化及逐物体真值核对完成',AP25=report['AP25'],AP50=report['AP50'],AP75=report['AP75'],
            strict_counts=report['strict_counts'],unmatched_predictions=len(report['unmatched_predictions']))
    finally:
        context.environment=saved
