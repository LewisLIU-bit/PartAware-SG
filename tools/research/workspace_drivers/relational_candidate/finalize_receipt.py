"""Compact completed-attempt receipt; no new screen or parameter variant."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
evaluation=json.loads((HERE/'evaluation.json').read_text());construction=json.loads((HERE/'construction.json').read_text())
assert evaluation['complete'] and construction['complete']
for name in ['screen.exit','build.exit','evaluation.exit']:assert (HERE/name).read_text().strip()=='0'
results=[]
for row in evaluation['results']:
    built=next(r for r in construction['results'] if r['scene']==row['scene'])
    c=next(r for r in row['variants'] if r['mode']=='signed_CORE_V_control')
    r=next(r for r in row['variants'] if r['mode']=='relational_CORE_V')
    identities=len(built['results'][0]['native_keys'])
    controls={int(k):v for k,v in c['best_counted_GT_IoU'].items()}
    candidate={int(k):v for k,v in r['best_counted_GT_IoU'].items()}
    suppressed=[]
    for variant in built['results']:
        suppressed.append(dict(mode=variant['mode'],native_nodes=[variant['native_keys'][k]
            for k in variant['audit']['suppressed'] if k<identities]))
    results.append(dict(scene=row['scene'],dataset=row['dataset'],metrics=r['metrics'],
        signed_CORE_V_control=c['metrics'],v19=row['v19_metrics'],
        delta_pp_signed_CORE_V=row['delta_percentage_points_vs_signed_CORE_V'],
        delta_pp_v19=row['delta_percentage_points_vs_v19'],
        peer_proposals=built['peer_proposals'],admitted_peer=built['admitted_peer'],
        deferred_peer_indices=[d['peer_index'] for d in built['deferrals']],
        cannot_link_pairs=built['cannot_link_pairs'],native_NMS_suppression=suppressed,
        counted_GT_IoU_improved_over5pp=sum(candidate[k]-v>.05 for k,v in controls.items()),
        counted_GT_IoU_regressed_over5pp=sum(v-candidate[k]>.05 for k,v in controls.items()),
        nonmatched_nonignored_FP50_control=c['FP']['0.5']['nonmatched_nonignored'],
        nonmatched_nonignored_FP50_candidate=r['FP']['0.5']['nonmatched_nonignored'],
        signed_inputs=built['signed_input_hashes'],source_cloud_sha256=built['source_cloud_sha256'],
        graph_sha256=built['graph_sha256'],tracks_sha256=built['tracks_sha256'],
        GT_hashes=row['GT_hashes']))
result=dict(complete=True,hypothesis_id='cross_source_covisibility_relation',
    conservative_distinct_attempt_candidate=15,distinct_completed_four_domain_screen=True,
    training_gate_does_not_automatically_pass=True,expanded_five_ours_results_and_independent_train_data_still_required=True,
    production_default_changed=False,global_upgrade_rejected=True,full_pipeline_replay=False,
    strict_box_010_new_result=None,API_calls=0,training=False,GPU=False,
    physical_points_deleted=0,generated_points=0,
    signed_controls_exactly_reproduced_all=True,behavior_and_invariance_tests_passed=16,
    build_seconds=construction['elapsed_seconds'],build_peak_RSS_MiB=construction['peak_RSS_MiB'],
    evaluation_seconds=evaluation['elapsed_seconds'],evaluation_peak_RSS_MiB=evaluation['peak_RSS_MiB'],
    outcome='Some0011 and010 gains over signed control, but0015 AP25 remains belowv19 and010 far belowv19;0019 unchanged.',
    failure_reasons=[
      'Relations only defer peer hypotheses covering>=80% of two native identities with repeated separated original masks; partial wrong peers survive.',
      '0019 has13 cannot-link pairs but no peer satisfies the composite test; merged original masks do not expose separate identities, so this candidate cannot recover them.',
      '010 defers only2 of75 peer proposals; unsafe source quality, four high-IoU native NMS removals and point competition remain.',
      'Any raw common whole mask marks the pair ambiguous, correctly protecting parts but also potentially hiding true separation behind an overbroad source mask.',
      'Consistent source oversegmentation cannot be ruled out without additional evidence; these relations are not perfect object-identity labels.',
      'Uniform CORE scoring still averages uncalibrated image-presence and geometric-quality sources; relational structure does not calibrate their scales.'],
    next_minimum_executable_problem=dict(
      priority='First establish an independent frozen2D entity-mask source for the source-merged0019 objects; a rank-only head cannot supply missing separate hypotheses.',
      executable_source_adapter='Feed author frozen CropFormer masks, after isolated weight/runtime verification, into the existing CDIS precomputed2D mask contract; cache once, keep input/depth/pose and evaluation fixed, permit all object names and no GT quantity constraints.',
      independent_source_reliability='Mechanistically necessary for the missing-candidate failure, but no new local accuracy or guaranteed advantage is claimed; the detector and complete four-domain pipeline must be measured.',
      task='Learn observed-mask quality for candidate adjudication with frozen backbones; keep unknown existence separate from low geometry quality.',
      optional_head='12→32→1,449 parameters, IoU quality regression; separate optional existence head only on fully labeled reliable observed cases.',
      possible_GT_free_input_features=['native score','peer score','cross-source Jaccard','both directional coverages',
         'original-mask dominance','mask-distribution entropy','independent visible view count','camera baseline',
         'cannot-link/composite support','common-raw-whole ambiguity','observed extent/point mass'],
      supervision='Only independent training scenes; targets from independent observed-domain GT audit; no010/eight development-validation GT training.',
      must_stop_treating_as_reliable_quality='Repeated hand tuning of score averages, point-count penalties and support proportions after their documented failures.',
      unresolved_source_merge='0019 lacks independent separate-mask evidence; a449-parameter rank head cannot create missing masks. A different frozen2D proposal source or genuinely supervised local identity head needs separate data/feature evidence.',
      not_training_started=True,not_guaranteed_to_win=True),
    results=results,
    receipt_hashes={name:sha(HERE/name) for name in ['hypothesis_registration.json','construction.json','evaluation.json','behavior_test_receipt.json','official_source_receipt.json']},
    module_hashes={name:sha(HERE/name) for name in ['relational_admission.py','stream_core.py','build_screen.py','evaluate_screen.py','test_relational_admission.py','register_hypothesis.py','finalize_receipt.py']},
    input_and_per_frame_depth_pose_original_mask_raw_cache_hashes_in='construction.json',
    source_paper_and_pinned_commit_in='official_source_receipt.json')
(HERE/'final_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
print('RELATIONAL_FINAL_RECEIPT_COMPLETE',flush=True)
