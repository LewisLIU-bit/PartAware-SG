"""Counterexamples for contact instances, ownership and measured boxes."""
import copy
import json
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from pipeline_components.contact_instances import (
    association, compact_component, consensus, independent_entities,
    projected_support, sample_cells)
from pipeline_components.boundary_ownership import sheet_evidence, outer_planes
from pipeline_components.front_continuity import carrier_support
from pipeline_components.face_boxes import fit
from pipeline_components.canonical_geometry import fit_box
from pipeline_components.support_layers import layer_cluster


class ContactEvidenceTests(unittest.TestCase):
    def test_touch_does_not_associate_two_instances(self):
        x,y=np.meshgrid(np.linspace(0,.1,20),np.linspace(0,.1,20))
        a=np.column_stack([x.ravel(),y.ravel(),np.zeros(x.size)])
        self.assertEqual(association(a,a+[.1,0,0]),0.)
        self.assertGreater(association(a,a+[.001,0,0]),0.)

    def test_detached_mask_leakage_is_removed_without_fabrication(self):
        rng=np.random.default_rng(15)
        body=rng.uniform(0,.05,(500,3));leak=rng.uniform(.3,.32,(100,3))
        original=np.concatenate([body,leak]);q=compact_component(original)
        self.assertIsNotNone(q);self.assertLess(q.max(),.06)
        self.assertTrue(all(any(np.array_equal(p,v) for v in original) for p in q))

    def test_repeated_detections_in_one_frame_are_not_independent_views(self):
        points=np.random.default_rng(15).uniform(0,.1,(100,3))
        group=dict(frames={'one'},points=[points]*8,evidence=[dict(frame_id='one')]*8)
        result,evidence=consensus(group)
        self.assertIsNone(result);self.assertFalse(evidence['accepted'])

    def test_zoom_duplicates_do_not_establish_multiplicity(self):
        left=dict(evidence=[]);right=dict(evidence=[])
        for fid in ('a','b','c'):
            left['evidence'].append(dict(frame_id=fid,native_index=0,is_zoom_proposal=False,segmentation_box=[0,0,4,4]))
            right['evidence'].append(dict(frame_id=fid,native_index=1,is_zoom_proposal=True,segmentation_box=[8,0,12,4]))
        self.assertFalse(independent_entities(left,right))
        for record in right['evidence']:record['is_zoom_proposal']=False
        self.assertTrue(independent_entities(left,right))

    def test_reprojection_accepts_sparse_sampling_but_not_occlusion(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory);(source/'sam3_cache').mkdir()
            mask=np.ones((1,16,16),np.uint8)
            for fid in ('a','b','c'):
                np.savez(source/'sam3_cache'/f'{fid}.npz',signature='signed',packed_masks=np.packbits(mask,axis=2))
            class Views:
                kc=np.array([[10.,0,8.],[0,10.,8.],[0,0,1.]])
                def project(self,points,fid):return np.arange(2),np.array([1.,.2]),None
                def get(self,fid):return np.eye(4),None,None
            native=types.SimpleNamespace(source=source,expected={fid:dict(signature='signed') for fid in ('a','b','c')},get=lambda fid:None)
            points=np.array([[0.,0.,1.],[.1,0.,1.]])
            group=dict(frames={'a','b','c'},evidence=[dict(frame_id=fid,native_index=0) for fid in ('a','b','c')])
            retained,votes=projected_support(points,group,Views(),native)
            np.testing.assert_array_equal(retained,points[:1]);np.testing.assert_array_equal(votes,[3,0])


class StructuralEvidenceTests(unittest.TestCase):
    @staticmethod
    def face(axis,level,a,b,count=40):
        x,y=np.meshgrid(np.linspace(a[0],b[0],count),np.linspace(a[1],b[1],count))
        q=np.empty((x.size,3));q[:,axis]=level
        other=[i for i in range(3) if i!=axis];q[:,other[0]]=x.ravel();q[:,other[1]]=y.ravel()
        return q

    def test_outer_sheet_does_not_remove_offset_furniture(self):
        wall=self.face(1,0.,[0,0],[2,2])
        plane=dict(normal=[0,1,0],offset=0.,basis=[[1,0],[0,0],[0,1]],lower=[0,0],upper=[2,2])
        self.assertIsNotNone(sheet_evidence(wall,[plane]))
        self.assertIsNone(sheet_evidence(wall+[0,.2,0],[plane]))
        self.assertIsNone(sheet_evidence(np.concatenate([wall,wall+[0,.05,0]]),[plane]))

    def test_only_outermost_parallel_sheet_is_room_boundary(self):
        planes=[dict(normal=np.array([0.,1.,0.]),offset=x) for x in (0.,.3)]
        result=outer_planes(planes);self.assertEqual(len(result),1);self.assertEqual(result[0]['offset'],.3)

    def test_a_carrier_gap_cannot_merge_fronts(self):
        a=self.face(1,0.,[0,0],[.3,.8]);b=a+[.6,0,0]
        carrier=self.face(2,.9,[0,-.1],[.9,.3],count=100)
        self.assertGreater(carrier_support(a,b,carrier),.9)
        disconnected=carrier[(carrier[:,0]<.31)|(carrier[:,0]>.59)]
        self.assertLess(carrier_support(a,b,disconnected),.9)

    def test_single_face_box_keeps_real_extents_and_direction(self):
        q=self.face(1,0.,[-.6,0],[.6,1.],count=45)
        matrix=Rotation.from_euler('z',.31).as_matrix();q=q@matrix.T
        before=q.copy();center,shape,evidence=fit(q,True,fit_box)
        self.assertTrue(evidence['accepted']);np.testing.assert_array_equal(q,before)
        box_rotation=Rotation.from_quat([shape['orientation'][k] for k in ('x','y','z','w')]).as_matrix()
        local=(q-center)@box_rotation
        extent=np.array([shape[k] for k in ('length','width','height')])
        self.assertTrue(np.all(np.abs(local)<=extent/2+1e-7))

    def test_open_layers_keep_measured_boards_but_reject_closed_front(self):
        boards=np.concatenate([self.face(2,z,[0,0],[.8,.3]) for z in (1.,1.3,1.6)])
        back=self.face(1,.3,[0,1],[.8,1.6],count=50)
        leak=self.face(1,.3,[.9,1],[1.4,1.6],count=40)
        q=np.concatenate([boards,back,leak]);retained,evidence=layer_cluster(q,np.array([[.4,-1.,1.4]]))
        self.assertIsNotNone(retained);self.assertEqual(evidence['layers'],3)
        self.assertLess(retained[:,0].max(),.83)
        closed=np.concatenate([boards,back,self.face(1,0.,[0,1],[.8,1.6],count=100)])
        self.assertIsNone(layer_cluster(closed,np.array([[.4,-1.,1.4]]))[0])

    def test_profile_detaches_all_v15_components(self):
        import pipeline_components as defaults
        from run_gpt_comparison import configure_profile
        registry=types.SimpleNamespace(**vars(defaults))
        for key in ('GEOMETRY_COMPONENTS','FINAL_GEOMETRY','MEASURED_REFINEMENT'):
            setattr(registry,key,list(getattr(defaults,key)))
        configure_profile('v15',registry);configure_profile('v15',registry)
        self.assertEqual(sum(m.__name__.endswith('.contact_instances') for m in registry.FINAL_GEOMETRY),1)
        self.assertEqual(len(registry.POST_PUBLICATION),1)
        configure_profile('v14',registry)
        self.assertEqual(registry.POST_PUBLICATION,[])
        self.assertFalse(any(m.__name__.endswith('.contact_instances') for m in registry.FINAL_GEOMETRY))


if __name__=='__main__':unittest.main()
