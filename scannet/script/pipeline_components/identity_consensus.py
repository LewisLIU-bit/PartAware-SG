"""Resolve conflicting names only when independent masks prove a duplicate surface."""

def contained_duplicate(containment, surface_coverage, evidence):
    return (containment >= .95 and surface_coverage >= .99
            and evidence['whole_mask_support_views'] >= 3
            and evidence['whole_mask_consensus'] >= .5
            and evidence['independent_separation_views'] == 0)
