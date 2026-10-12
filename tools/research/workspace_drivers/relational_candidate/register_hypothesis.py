import hashlib,json
from pathlib import Path
from relational_admission import POLICY
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
items=json.loads((ROOT/'v20_work/four_domain_confirmation_registration.json').read_text())['items']
official=json.loads((HERE/'official_source_receipt.json').read_text())
record=dict(registered_before_screen=True,hypothesis_id='cross_source_covisibility_relation',
    proposed_distinct_attempt=15,completed=False,GT_in_construction=False,API_calls=0,training=False,GPU=False,
    hypothesis='A peer geometry covering multiple repeatedly separated original identities should remain a composite provisional hypothesis rather than replace those object identities.',
    distinction='Negative relation structure gates composite cross-source identity; no new scores, NMS threshold, point-support ownership, or local split generation.',
    existing_native_pair_conflict_logic_acknowledged=True,
    source_paper='MaskClustering CVPR2024, arXiv:2401.07745v2, sections3.2.1/3.2.3',
    source_implementation=official,author_pipeline_reproduced=False,policy=POLICY,
    risks=['Consistent 2D oversegmentation can falsely create a cannot-link relation when no raw whole mask is available.',
       'A single common raw whole mask marks the pair ambiguous; an overbroad raw mask can hide a true separation.',
       'Merged original masks such as stacked chairs may never provide separation evidence.',
       'Deferred peer geometry may lower true recall; deferral is not an existence rejection or physical point deletion.',
       'Instance graph part annotations and original whole masks are incomplete, so part/whole correctness is not guaranteed.',
       'This is a mask screen; a gain would require full point-cloud/graph publication and the original010 strictbox evaluation.'],
    items=items,registration_input_sha256=sha(ROOT/'v20_work/four_domain_confirmation_registration.json'),
    source_hashes={p.name:sha(p) for p in [HERE/'relational_admission.py',HERE/'stream_core.py']})
(HERE/'hypothesis_registration.json').write_text(json.dumps(record,indent=2)+'\n')
print('RELATIONAL_HYPOTHESIS_REGISTERED',flush=True)
