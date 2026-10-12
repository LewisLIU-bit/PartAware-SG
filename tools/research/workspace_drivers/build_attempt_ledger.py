"""Conservative method-level ledger: parameter/replay variants count once."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT/'v20_work'
groups=[
 ('peer_geometry','跨源追加/并集/交集是否能恢复漏点','mask_proposals.py',['peer_trials.json'],'三扫描筛选退步；三种操作归并一次'),
 ('atomic_partition','互不重叠几何核心是否证明独立身份','atomic_partition.py',['partition_anchors.json','partition_conservative.json'],'拆碎whiteboard；两个阈值策略归并一次'),
 ('hard_mutual','双向可见覆盖硬门槛能否避免混合','mutual_association.py',['mutual_r1_evaluation.json'],'实际三扫描重跑有退步'),
 ('soft_mutual','保留资格、用双向覆盖软排序能否恢复遮挡','soft_mutual_association.py',['mutual_r2_evaluation.json'],'实际三扫描仍0011退步/0019不变'),
 ('nested_partition','嵌套实测候选能否恢复子物体','containment_partition.py',['partition_containment.json'],'三扫描筛选退步'),
 ('quality_ranking_family','观测/跨源质量代理能否改善排序','peer_quality_ranking.py',['quality_ranking_trial.json','verified_text_positive_ranked_peer_publication_trials.json','verified_text_positive_consistency_peer_publication_trials.json','verified_text_positive_retention_peer_publication_trials.json'],'支持比例、peer质量、软一致性、保留比例全部归并一次；无全场景收益'),
 ('weighted_ownership','跨源一致性与大小加权独占点归属','proposal_ensemble.py',['sage_exclusive_weighted_evaluation.json'],'三扫描实际发布：0019 AP50仍低于ESAM'),
 ('core_ownership','不竞争实测核心距离能否正确归属争议点','core_ownership.py',['sage_exclusive_core_evaluation.json'],'三扫描实际发布：0019 AP25/AP50未超过ESAM'),
 ('signed_visual_refinement','独立SAM可见正负投票能否修复几何','projection_verification.py',['verified_peer_trials.json'],'三扫描筛选不同时改善；未知不能当反证'),
 ('positive_text_refinement','独立原图正支持扩展且保持未知几何','source_refined_core.py',['sage_verified_core_evaluation.json'],'三扫描实际发布AP50/均值超过ESAM；0019 AP25仍未达'),
 ('local_subproposals','独立视角局部候选关联能否拆复合实例','build_local_subproposals.py',['verified_local_peer_publication_trials.json'],'三扫描筛选退步；几何一致不证明身份'),
 ('purity_nms','双向覆盖约束NMS能否保护整体候选','purity_core.py',['verified_text_positive_purity_peer_publication_trials.json'],'三扫描筛选退步'),
 ('image_confirmation','要求独立原图名称证据是否排除虚假peer','build_image_confirmed_trials.py',['image_confirmed_evaluation.json'],'四场景筛选退步，否决；无新增API/训练'),
 ('cached_positive_identity_ownership','独立原帧正身份支持是否改善跨源争议点归属','cue_candidate/cue_ownership.py',['cue_candidate/final_receipt.json'],'四域筛选；0011改善、0015退步、010仍远低v19，否决；不变q/NMS，独立于质量评分族'),
 ('cross_source_covisibility_relation','原帧共同可见分离身份与whole歧义能否阻止复合peer误合并','relational_candidate/relational_admission.py',['relational_candidate/final_receipt.json'],'四域同策略筛选；0011改善、0015/0019不变、010仍远低v19；否决全域替换，关系约束独立于逐点CUE和评分族')]
rows=[]
for ordinal,(key,hypothesis,module,files,outcome) in enumerate(groups,1):
 receipts=[]
 for name in files:
  p=WORK/name;data=json.loads(p.read_text());assert data.get('complete')
  receipts.append(dict(file=name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),full_pipeline=bool(data.get('full_geometry_replay') or data.get('full_pipeline_rerun') or data.get('full_pipeline_replay'))))
 source=WORK/'research'/module
 if not source.exists():source=WORK/module
 assert source.exists()
 rows.append(dict(ordinal=ordinal,id=key,hypothesis=hypothesis,receipts=receipts,outcome=outcome,
     representative_source=str(source.relative_to(WORK)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
     complete=True,current_eight_scan_and_010_validation=False))
result=dict(complete=True,minimum_distinct_attempts_before_training=15,conservatively_grouped_completed_hypotheses=len(rows),
    parameter_variants_and_repeat_jobs_count_separately=False,baseline_installations_count=False,
    expanded_eight_scan_and_010_acceptance_pending=True,minimum_attempt_count_met=len(rows)>=15,
    training_gate_passed=False,dataset_training_started=False,
    reason='Fifteen conservative method families completed. The quantity prerequisite is met; five new ScanNet comparisons and independent training/calibration data are still required, and 010 transfer regresses. No automatic training or claim of exhausted algorithm progress.',attempts=rows)
(WORK/'algorithm_attempt_ledger.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print('ALGORITHM_ATTEMPT_GATE',len(rows),'of minimum 15; training remains unstarted',flush=True)
