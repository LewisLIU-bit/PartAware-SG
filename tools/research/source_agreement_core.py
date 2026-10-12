"""SAGE-CORE reuses the existing publisher with independent core arbitration.

This isolated research callback leaves the production registry untouched.
Preserve the pre-arbitration geometry for causal audits after actual replays.
"""
import hashlib,json,shutil
from pathlib import Path
import source_agreement
from core_ownership import propose

def construct(context):
 old=source_agreement.propose
 input_path=Path(context.graph_geometry)
 shutil.copy2(input_path,context.scene/'instance_cloud_pre_sage.ply')
 for name in ['topology_map.json','validated_object_tracks.json']:
  shutil.copy2(context.scene/name,context.scene/('pre_sage_'+name))
 source_agreement.propose=propose
 try:source_agreement.construct(context)
 finally:source_agreement.propose=old
 audit_path=context.scene/'source_agreement_audit.json'
 audit=json.loads(audit_path.read_text())
 audit.update(algorithm='SAGE-CORE',pre_arbitration_geometry_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
              extra_algorithm_sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('core_ownership.py')]})
 # The input path can be overwritten by the publisher; hash the preserved copy.
 audit['pre_arbitration_geometry_sha256']=hashlib.sha256((context.scene/'instance_cloud_pre_sage.ply').read_bytes()).hexdigest()
 audit_path.write_text(json.dumps(audit,indent=2)+'\n')
