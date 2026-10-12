"""Meaningful identity, unknown, part and invariance tests, with no GT input."""
import json
from pathlib import Path
import numpy as np
import sys
from relational_admission import frame_relations,cannot_link,admit_peers,native_frame_distribution
from stream_core import propose as streaming_core

def run():
    owner=np.r_[np.zeros(40,int),np.ones(40,int)]
    labels=np.r_[np.full(40,17),np.full(40,999)]
    raw=np.array([owner==0,owner==1])
    sep,whole,_=frame_relations(owner,labels,raw,2)
    assert sep[0,1] and not whole[0,1]
    centers=np.array([[0.,0,0],[.25,0,0]])
    edges,proof=cannot_link(np.array([sep,sep]),np.array([whole,whole]),centers)
    keep,reason=admit_peers(np.array([[40,40],[40,0]]),np.array([40,40]),edges)
    # Standard evaluator >=100 is a proposal threshold; use >=100 masses
    # for the composite deferral test independently of frame-visible count.
    keep,reason=admit_peers(np.array([[120,120],[120,0]]),np.array([120,120]),edges)
    assert keep.tolist()==[False,True] and len(reason)==1
    # Unknown/unobserved and duplicated camera positions cannot veto.
    assert not cannot_link(np.array([sep]),np.array([whole]),centers[:1])[0].any()
    assert not cannot_link(np.array([sep,sep]),np.array([whole,whole]),np.zeros((2,3)))[0].any()
    _,_,valid=native_frame_distribution(owner,np.zeros(80,int),2);assert not valid.any()
    # An original whole covering both parts makes separation ambiguous.
    sep2,whole2,_=frame_relations(owner,labels,np.r_[raw,np.ones((1,80),bool)],2)
    assert not sep2[0,1] and whole2[0,1]
    assert not cannot_link(np.array([sep,sep]),np.array([whole2,whole]),centers)[0].any()
    # Explicit part-parent evidence prevents two-object interpretation.
    assert not frame_relations(owner,labels,raw,2,part_pairs=[(0,1)])[0].any()
    # Opaque local ID remap does not alter physical relationships.
    renamed=np.where(labels==17,2,1)
    assert np.array_equal(frame_relations(owner,renamed,raw,2)[0],sep)
    # Point/identity/view/peer permutation and rigid camera transforms.
    rng=np.random.default_rng(88);perm=rng.permutation(80)
    assert np.array_equal(frame_relations(owner[perm],labels[perm],raw[:,perm],2)[0],sep)
    assert np.array_equal(frame_relations(1-owner,labels,raw[::-1],2)[0],sep[::-1,::-1])
    rotation=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
    translated=centers@rotation.T+np.array([13.,-8.,4.])
    transformed=cannot_link(np.array([sep,sep]),np.array([whole,whole]),translated)[0]
    assert np.array_equal(transformed,edges)
    permuted=cannot_link(np.array([sep,sep])[::-1],np.array([whole,whole])[::-1],centers[::-1])[0]
    assert np.array_equal(permuted,edges)
    assert admit_peers(np.array([[120,120],[0,120]]),np.array([120,120]),edges)[0].tolist()==[True,False]
    root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root/'v20_work/research'))
    from core_ownership import propose as signed_core
    from projection_verification import project
    xyz=rng.normal(size=(900,3));own=np.repeat(np.arange(3),300)
    primary=np.array([own==i for i in range(3)]);native_scores=np.array([.71,.83,.67])
    peer=np.array([np.r_[np.ones(220,bool),np.zeros(680,bool)],
        np.r_[np.zeros(220,bool),np.ones(380,bool),np.zeros(300,bool)],rng.random(900)<.22])
    peer_scores=np.array([.89,.79,.61])
    expected,scores,signed_audit=signed_core(primary,native_scores,peer,peer_scores,points=xyz)
    actual,actual_scores,audit=streaming_core(own,native_scores,peer,peer_scores,xyz)
    assert np.array_equal(np.array([actual==i for i in range(len(actual_scores))]),expected)
    assert np.array_equal(scores,actual_scores) and audit['kept']==signed_audit['kept']
    perm=rng.permutation(len(own))
    changed,changed_scores,_=streaming_core(own[perm],native_scores,peer[:,perm],peer_scores,xyz[perm])
    assert np.array_equal(changed[np.argsort(perm)],actual) and np.array_equal(changed_scores,scores)
    moved,_,_=streaming_core(own,native_scores,peer,peer_scores,xyz@rotation.T+[13,-8,4])
    assert np.array_equal(moved,actual)
    # Physical visibility is invariant under the same world transform for
    # sensor pose and all points. Unobserved/occluded depth remains unknown.
    k=np.array([[100.,0,50],[0,100,50],[0,0,1]])
    pose=np.eye(4);depth=np.full((101,101),1000,np.uint16)
    points=np.column_stack([rng.uniform(-.4,.4,(30,2)),np.ones(30)])
    ix,uv=project(points,pose,k,k,depth,1000,101,101)
    transformed_pose=np.eye(4);transformed_pose[:3,:3]=rotation;transformed_pose[:3,3]=[13,-8,4]
    changed_ix,changed_uv=project(points@rotation.T+[13,-8,4],transformed_pose,k,k,depth,1000,101,101)
    assert np.array_equal(ix,changed_ix) and np.array_equal(uv,changed_uv)
    result=dict(passed=True,tests=['repeated_co_visible_separation','composite_peer_deferral',
        'unknown_kept_unknown','duplicate_view_not_independent','whole_part_ambiguity','explicit_part_parent',
        'local_ID_remap','point_order','native_order','peer_order','view_order','rigid_camera_transform',
        'stream_CORE_exact_signed_mask_score_equivalence','stream_CORE_point_order',
        'stream_CORE_rigid_transform','projection_rigid_transform'],
        GT_in_test=False,new_API_calls=0,training=False)
    Path(__file__).with_name('behavior_test_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print('RELATIONAL_BEHAVIOR_TESTS_PASSED',len(result['tests']),flush=True)
if __name__=='__main__':run()
