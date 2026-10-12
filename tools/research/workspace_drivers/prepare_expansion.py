"""Preregister the next five distinct official validation scans before AP."""
import ast,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OLD=ROOT/'scan200_work';WORK=ROOT/'scan200_expansion_work';WORK.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=json.loads((OLD/'preregistration.json').read_text());official=OLD/'scannetv2_val.txt'
assert sha(official)==old['official_splits']['val']['sha256']
ordered=sorted(s for s in official.read_text().splitlines() if s.endswith('_00'))
assert ordered[:3]==old['scenes']
selected=ordered[3:8];assert len(selected)==5 and not set(selected)&set(old['scenes'])
registration=dict(old,scenes=selected,existing_development_scans=old['scenes'],expanded_scans=ordered[:8],
 selection='Next five distinct _00 scan prefixes in fixed official validation lexicographic order, before images, annotations or AP; no selection by difficulty or performance.',
 benchmark_scope='Eight fixed scans, not complete 312-scan benchmark; five new scans begin as a preregistered transfer check and become development if used for changes.',
 acceptance='Per-scan AP25/AP50/mean, and ai_001_010 original strict-box and point protocols; no pooled-only or per-scene strategy selection.',
 training_gate='At least fifteen distinct, documented general algorithm hypotheses before optional training; repeat jobs and parameter variants do not count separately.')
path=WORK/'preregistration.json'
if path.exists():assert json.loads(path.read_text())==registration
else:path.write_text(json.dumps(registration,indent=2)+'\n')
for name in ['acquire_cache.py','run_scan200.py','prepare_esam.py','run_esam.py','evaluate_scan200.py',
 'esam_cpu_pointops.py','esam_cpu_compatibility.py','esam_weights.json','esam_import_compatibility.json','scannet_evaluator_source.json']:
 shutil.copyfile(OLD/name,WORK/name)
if not (WORK/'reference').exists():shutil.copytree(OLD/'reference',WORK/'reference')
# Preserve the official baseline parameters; only five-scan completion counts
# and global GPU mutual exclusion differ from the original three-scan driver.
p=WORK/'prepare_esam.py';text=p.read_text().replace('complete=len(rows)==3','complete=len(rows)==len(registration["scenes"])')
text=text.replace("with (WORK/'gpu.lock').open('a') as mutex:","with (WORK/'gpu.lock').open('a') as mutex, (Path.home()/'.cache/partaware-sg/gpu.lock').open('a') as shared:")
text=text.replace('fcntl.flock(mutex,fcntl.LOCK_EX)','fcntl.flock(mutex,fcntl.LOCK_EX)\n            fcntl.flock(shared,fcntl.LOCK_EX)')
p.write_text(text)
p=WORK/'run_esam.py';text=p.read_text().replace('complete=len(rows)==3','complete=len(rows)==len(registration["scenes"])');p.write_text(text)
# Reuse only inspected downloader/export functions, avoiding the old module's
# top-level three-scan registration and execution.
tree=ast.parse((OLD/'prepare_data.py').read_text());functions=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'sha','download','export'}],type_ignores=[])
helpers='import hashlib,json,struct,time,zlib\nfrom pathlib import Path\nimport cv2,numpy as np,requests\n\n'+ast.unparse(functions)+'\n'
(WORK/'download_helpers.py').write_text(helpers)
(WORK/'setup_receipt.json').write_text(json.dumps(dict(complete=True,preregistered=True,
 original_download_driver_sha256=sha(OLD/'prepare_data.py'),source_evaluator_sha256=sha(OLD/'evaluate_scan200.py'),
 copied_evaluator_sha256=sha(WORK/'evaluate_scan200.py'),registration_sha256=sha(path),
 new_data_GT_in_construction=False,algorithms_trained=False,semantic_API_config_reused='local config path only, never included'),indent=2)+'\n')
assert sha(WORK/'evaluate_scan200.py')==sha(OLD/'evaluate_scan200.py')
print('FIVE_ADDITIONAL_SCANS_PREREGISTERED',selected,flush=True)
