"""Research publisher for frozen text/box positive evidence plus CORE.

Unknown/negative SAM pixels cannot erase the official peer here. Only newly
observed mesh vertices corroborated by independent views augment proposals.
"""
import hashlib,json
from pathlib import Path
import source_agreement_core

def construct(context):
 receipt_path=Path(context.research_refinement_receipt)
 evidence=json.loads(receipt_path.read_text())
 if not evidence['complete'] or not evidence['positive_only'] or evidence['GT_in_construction']:raise ValueError('Signed positive-only independent evidence required')
 if hashlib.sha256(Path(context.research_peer).read_bytes()).hexdigest()!=evidence['output_sha256']:raise ValueError('Refined mask hash mismatch')
 source_agreement_core.construct(context)
 audit_path=context.scene/'source_agreement_audit.json';audit=json.loads(audit_path.read_text())
 audit.update(algorithm='SAGE-CORE-V',refinement_receipt=evidence,refinement_receipt_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),wrapper_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 audit_path.write_text(json.dumps(audit,indent=2)+'\n')
