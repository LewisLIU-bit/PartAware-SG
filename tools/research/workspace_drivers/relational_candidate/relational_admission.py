"""Cross-source composite-hypothesis abstention from original observations.

No GT, class filter, expected object count, learned or rescaled score. A peer
covering two repeatedly separated original identities is deferred. A common
raw whole mask makes the relation ambiguous, rather than a cannot-link edge.
"""
import numpy as np
from scipy.spatial.distance import pdist

POLICY = dict(minimum_visible_points=32, dominant_original_mask_fraction=.7,
              raw_whole_containment=.8, minimum_independent_views=2,
              minimum_camera_baseline_m=.15, peer_native_containment=.8,
              common_whole_view_makes_relation_ambiguous=True)

def native_frame_distribution(native_owner, labels, count, minimum_points=32, dominant_fraction=.7):
    """All visible points, including unlabeled ones, contribute to visibility.

    Dominance >.5 avoids ID-dependent argmax tie breaks. Local labels are
    opaque identifiers; the actual numeric ID has no influence on the rule.
    """
    owner=np.asarray(native_owner,int);labels=np.asarray(labels,int)
    if owner.shape!=labels.shape:raise ValueError('One original label per visible point required')
    good=owner>=0;visible=np.bincount(owner[good],minlength=count)
    local_ids,inverse=np.unique(labels,return_inverse=True)
    table=np.bincount(owner[good]*len(local_ids)+inverse[good],
        minlength=count*len(local_ids)).reshape(count,len(local_ids))
    table[:,local_ids==0]=0
    if len(local_ids):
        index=table.argmax(1);mass=table[np.arange(count),index]
        dominant=local_ids[index];valid=(visible>=minimum_points)&(mass>=dominant_fraction*visible)&(dominant!=0)
    else:dominant=np.zeros(count,int);valid=np.zeros(count,bool)
    return visible,dominant,valid

def frame_relations(native_owner, labels, raw_membership, count, part_pairs=()):
    """Separate labels alone are insufficient when a raw whole covers both.

    raw_membership rows are read-only cached original model masks at the
    projected pixels, before mutual exclusion overwrote overlapping parts.
    """
    visible,dominant,valid=native_frame_distribution(native_owner,labels,count,
        POLICY['minimum_visible_points'],POLICY['dominant_original_mask_fraction'])
    owner=np.asarray(native_owner,int);good=owner>=0
    contains=np.zeros((len(raw_membership),count),bool)
    for j,member in enumerate(raw_membership):
        hits=np.bincount(owner[good],weights=np.asarray(member,bool)[good],minlength=count)
        contains[j]=(visible>=POLICY['minimum_visible_points'])&(hits>=POLICY['raw_whole_containment']*visible)
    common=(contains.astype(np.int64).T@contains.astype(np.int64))>0
    jointly=valid[:,None]&valid[None,:]
    separate=jointly&(dominant[:,None]!=dominant[None,:])&~common
    for left,right in part_pairs:separate[left,right]=separate[right,left]=False;common[left,right]=common[right,left]=True
    np.fill_diagonal(separate,False);np.fill_diagonal(common,False)
    return separate,common,jointly

def cannot_link(separate_frames, whole_frames, camera_centers):
    separate_frames=np.asarray(separate_frames,bool);whole_frames=np.asarray(whole_frames,bool)
    centers=np.asarray(camera_centers,float)
    if separate_frames.shape!=whole_frames.shape or separate_frames.shape[0]!=len(centers):raise ValueError('Aligned view evidence required')
    count=separate_frames.shape[1];relation=np.zeros((count,count),bool);proof=[]
    for left in range(count):
        for right in range(left+1,count):
            indices=np.flatnonzero(separate_frames[:,left,right])
            common=int(whole_frames[:,left,right].sum())
            if len(indices)<POLICY['minimum_independent_views'] or common:continue
            baseline=float(pdist(centers[indices]).max(initial=0))
            if baseline+1e-12<POLICY['minimum_camera_baseline_m']:continue
            relation[left,right]=relation[right,left]=True
            proof.append(dict(left_index=left,right_index=right,separated_frames=indices.tolist(),
                camera_baseline_m=baseline,common_raw_whole_views=common))
    return relation,proof

def admit_peers(native_peer_intersection,native_mass,relations):
    intersection=np.asarray(native_peer_intersection,np.int64);mass=np.asarray(native_mass,np.int64)
    relations=np.asarray(relations,bool)
    if intersection.shape[0]!=len(mass) or relations.shape!=(len(mass),len(mass)):raise ValueError('Native identity domain mismatch')
    covered=(mass[:,None]>=100)&(intersection>=POLICY['peer_native_containment']*mass[:,None])
    admission=np.ones(intersection.shape[1],bool);deferrals=[]
    for peer in range(intersection.shape[1]):
        identities=np.flatnonzero(covered[:,peer]);edges=[]
        for offset,left in enumerate(identities):
            for right in identities[offset+1:]:
                if relations[left,right]:edges.append([int(left),int(right)])
        if edges:
            admission[peer]=False;deferrals.append(dict(peer_index=peer,cannot_link_pairs=edges,
                state='provisional_composite_geometry',physical_points_deleted=0))
    return admission,deferrals
