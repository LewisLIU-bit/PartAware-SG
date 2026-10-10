"""Reject contact-only attachment between incompatible object identities."""
import re
import numpy as np
from .native_assembly import NativeMasks


def compositional_identity(body_name, source_name):
    body = set(re.findall(r'[a-z]+', body_name.lower()))
    source = set(re.findall(r'[a-z]+', source_name.lower()))
    return bool(body) and body < source


def acceptance(semantic, evidence):
    return semantic >= .8 or (semantic >= .65 and evidence.get('compositional_identity', False)) or (semantic >= .4 and evidence['native_whole_views'] >= 3
        and evidence['native_consensus'] >= .6 and evidence['native_separation_views'] == 0)


class Evidence:
    def __init__(self, views):
        self.native = NativeMasks(views)

    def check(self, body, source, body_node, source_node, frames):
        a, b = np.asarray(body_node['text_embedding']), np.asarray(source_node['text_embedding'])
        semantic = float(a @ b/max(np.linalg.norm(a)*np.linalg.norm(b), 1e-12))
        evidence = dict(native_whole_views=0, native_consensus=0., native_separation_views=0,
            compositional_identity=compositional_identity(body_node['name'], source_node['name']))
        if .4 <= semantic < .8 and self.native.available:
            evidence.update(self.native.measure(body[::max(1, len(body)//2048)],
                source[::max(1, len(source)//2048)], frames))
        return acceptance(semantic, evidence), dict(semantic_cosine=semantic, **evidence)
