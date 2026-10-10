"""Functional pipeline tree derived from the active code registry.

This is descriptive provenance, not a second execution engine. The runner still
owns the sequence of temporary graph, measured geometry, parts and publication.
"""


def process_tree(registry, recognition=None):
    def modules(*keys, omit=()):
        result = []
        for key in keys:
            values = getattr(registry, key, [])
            if not isinstance(values, (list, tuple)): values = [values]
            result.extend(value.__name__ for value in values if value is not None
                          and value.__name__.rsplit('.', 1)[-1] not in omit)
        return result

    return {
        'FOVEA': {'name': 'Fine Object and Visual Evidence Acquisition', 'children': {
            'semantics': {'source': recognition['provider'] if recognition else 'cached descriptions or fixed indoor vocabulary',
                          'model': recognition.get('requested_model') if recognition else None,
                          'shared_cache': recognition.get('shared_cache') if recognition else None,
                          'qwen_api_calls': 0},
            'observations': {'modules': modules('FRONTEND')},
            'features': {'object_visual_dimensions': 256, 'object_semantic_dimensions': 384,
                         'visual_part_dimensions': 1024, 'geometry_only_part_embedding': None}}},
        'MICA': {'name': 'Multiview Instance Consensus and Association', 'children': {
            'association': {'modules': modules('ASSOCIATION', 'FUSION', 'INSTANCE_REFINEMENT')},
            'whole_object': {'modules': modules('HIERARCHY_VALIDATION', 'WHOLE_OBJECT_VALIDATION', 'SURFACE_ASSEMBLY')},
            'contact_recovery': {'modules': [entry.__name__ for entry in getattr(registry, 'FINAL_GEOMETRY', [])
                                             if entry.__name__.rsplit('.',1)[-1] in ('contact_instances','visible_instances','residual_ownership')],
                                 'execution_after_part_fusion': True},
            'validation': {'modules': modules('OBJECT_VALIDATION', 'OBSERVED_VALIDATION', 'BACKGROUND_VALIDATION', 'IDENTITY_VALIDATION', 'SURFACE_VALIDATION', 'OWNERSHIP_VALIDATION')}}},
        'SHAPE': {'name': 'Scene-constrained Hypotheses Anchored to Physical Evidence', 'children': {
            'geometry': {'modules': modules('GEOMETRY_COMPONENTS')},
            'measured_recovery': {'modules': modules('PART_GEOMETRY', 'MEASURED_REFINEMENT', 'FINAL_GEOMETRY', omit=('contact_instances','visible_instances','residual_ownership'))},
            'assembly_ownership': {'modules': modules('AXIAL_VALIDATION')},
            'enclosure_continuity': {'modules': modules('BODY_CONTINUITY')},
            'evidence_boundary': {'generated_surfaces_are_measurements': False, 'ground_truth_in_construction': False}}},
        'GRAPH': {'name': 'Geometry Relations And Part Hierarchy', 'children': {
            'objects': {'backend': 'original ScanNet-SG graph binary'},
            'hierarchy': {'modules': modules('GRAPH_COMPONENTS')},
            'publication': {'modules': modules('BOX_FITTING', 'GEOMETRY_OUTPUT')},
            'evaluation': {'modules': modules('EVALUATION_DIAGNOSTICS', 'POST_PUBLICATION'), 'construction_dependency': False}}}
    }
