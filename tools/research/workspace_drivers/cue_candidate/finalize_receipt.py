"""Freeze compact CUE receipts after the independent screen and diagnosis."""
import hashlib
import json
from pathlib import Path
WORK = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


construction = json.loads((WORK/'construction.json').read_text())
evaluation = json.loads((WORK/'evaluation.json').read_text())
diagnosis = json.loads((WORK/'diagnosis.json').read_text())
tests = json.loads((WORK/'behavior_tests.json').read_text())
assert all(row['complete'] for row in [construction, evaluation, diagnosis, tests])
assert (WORK/'service.exit').read_text().strip() == '0'
assert sha(WORK/'build_cue_screen_executed.py') == construction['builder_sha256']
assert sha(WORK/'cue_ownership.py') == construction['component_sha256']
assert sha(WORK/'hypothesis.json') == construction['hypothesis_sha256']
assert sha(WORK/'construction.json') == evaluation['construction_sha256']
assert sha(WORK/'evaluate_cue_screen.py') == evaluation['evaluator_sha256']
rows = []
for build, evaluated, diagnosed in zip(construction['results'], evaluation['results'], diagnosis['results']):
    assert build['scene'] == evaluated['scene'] == diagnosed['scene']
    assert sha(build['masks_file']) == build['masks_sha256']
    assert sha(build['control_masks_file']) == build['control_masks_sha256']
    assert sha(build['evidence_file']) == build['evidence_sha256']
    rows.append(dict(scene=build['scene'], dataset=build['dataset'],
        AP25_AP50_mean_percent=[100*evaluated['cue_metrics'][key]
            for key in ['all_ap_25%', 'all_ap_50%', 'all_ap']],
        delta_vs_same_geometry_CORE_percentage_points=[evaluated['delta_vs_same_geometry_core_percentage_points'][key]
            for key in ['all_ap_25%', 'all_ap_50%', 'all_ap']],
        delta_vs_frozen_v19_percentage_points=[evaluated['delta_vs_frozen_v19_percentage_points'][key]
            for key in ['all_ap_25%', 'all_ap_50%', 'all_ap']]
            if evaluated['delta_vs_frozen_v19_percentage_points'] else None,
        modified_conflict_points=build['ownership_audit']['changed_conflict_owners'],
        changed_points_after_publication=build['changed_owners_after_publication'],
        eligible_GT_best_IoU_gains=diagnosed['eligible_GT_best_IoU_improvements'],
        eligible_GT_best_IoU_losses=diagnosed['eligible_GT_best_IoU_regressions']))
source_names = ['cue_ownership.py', 'build_cue_screen.py', 'build_cue_screen_executed.py',
    'evaluate_cue_screen.py', 'diagnose_cue_screen.py', 'test_cue_ownership.py', 'run_cue_service.sh']
receipt_names = ['hypothesis.json', 'construction.json', 'evaluation.json',
    'diagnosis.json', 'behavior_tests.json', 'service.exit']
receipt = dict(complete=True, method='CUE: Cached-view Unique-positive Evidence ownership',
    method_family_eligible_for_14=True, method_distinctness='Point ownership changes from independent original local mask identity; no score/NMS/threshold variant counting',
    adopted=False, decision='reject_current_cross_domain_replacement',
    full_pipeline_replay=False, new_Hypersim_strict_box_metrics=None,
    GT_in_construction=False, unknown_and_background_are_negative=False,
    new_API_calls=0, new_training=False, GPU=False,
    original_candidate_q_and_NMS_unchanged=True, CORE_control_reproduces_prior_registered_domain_metrics=True,
    original_ScanNet_geometric_majority_publication_unchanged=True,
    observed_candidate_union_preserved_before_publication=True,
    exclusive_point_ownership=True, physical_point_deletions=0, generated_points=0,
    builder_peak_RSS_GiB=max(row['process_peak_RSS_GiB'] for row in construction['results']),
    evaluator_peak_RSS_GiB=max(row['process_peak_RSS_GiB'] for row in evaluation['results']),
    resource_control=dict(OMP_OPENBLAS_MKL_threads=2, KDTree_workers=2,
        current_drivers_CPU_affinity_cores=2,
        executed_builder_exact_source_snapshot='build_cue_screen_executed.py',
        runtime_note='CPU affinity was additionally restricted externally during the executed build; the current driver includes that resource-only restriction. Ownership component and all inputs stayed unchanged.'),
    results=rows,
    failure_attribution=[
        'scene0015: one chair best IoU falls .771324 to .734694 when 70 correctly covered published points move from a peer candidate to a native partial candidate. Source positive support does not guarantee whole-object identity or ownership purity.',
        'scene0011: all three AP metrics gain, but a cabinet loses .009699 best IoU. AP gains do not imply every object improves.',
        'scene0019: the changed chair best IoU rises .782977 to .845694, improving mean AP; AP25 and AP50 remain unchanged and missing/merged-instance failures are not resolved.',
        'ai_001_010: all point AP metrics improve over the weak CORE screen, but remain far below v19. Ownership does not recover proposals suppressed by fixed NMS or repair all source identity mistakes.'
    ],
    limitations=['Four development domains, not a 312-scan benchmark or statistical significance test.',
        'Mask-space screening only; no production point-cloud/graph replay or new 010 strict-box result.',
        'Source masks may already undersegment different physical objects; two independent positive views can repeat that mistake.',
        'Frozen q/NMS can discard native objects before CUE arbitration.'],
    unchanged_shared_default_registry_reports_and_training_ledger=True,
    source_hashes={name: sha(WORK/name) for name in source_names},
    receipt_hashes={name: sha(WORK/name) for name in receipt_names},
    artifacts_retained='Compact compressed control/candidate masks plus per-frame evidence retained for independent attempt15 audit; temporary dense projection caches deleted only inside cue_candidate/results.')
# The counts in the frozen diagnosis are the authoritative attribution values.
kitchen = next(row for row in rows if row['dataset'] == 'hypersim')
receipt['failure_attribution'][-1] = ('ai_001_010: all point AP metrics improve over the weak CORE screen, '
    'but remain far below v19; '+str(kitchen['eligible_GT_best_IoU_losses'])+' eligible objects have lower '
    'best IoU while '+str(kitchen['eligible_GT_best_IoU_gains'])+' improve. Ownership does not recover '
    'proposals suppressed by fixed NMS or repair all source identity mistakes.')
(WORK/'final_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(dict(complete=True, source_hashes=receipt['source_hashes'], results=rows), indent=2), flush=True)
