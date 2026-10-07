"""Counterexamples for measured structural and suspension recovery."""
import unittest
import numpy as np
import open3d as o3d
from pipeline_components.structural_surfaces import family,rear_boundary,storage_identity
from pipeline_components.suspension_geometry import line_member,ceiling_member_candidates,connected_body_points


class StructuralRecoveryTests(unittest.TestCase):
    def setUp(self):o3d.utility.random.seed(10)

    def test_floor_and_wall_labels_cannot_seed_storage(self):
        for name in ['wall','floor','ceiling','sink','microwave','stove']:
            self.assertIsNone(family(name))

    def test_cached_storage_and_support_families(self):
        self.assertEqual(family('wall cabinet'),'storage')
        self.assertEqual(family('shelves'),'storage')
        self.assertEqual(family('kitchen counter'),'support')

    def test_sheet_without_rear_is_rejected(self):
        y,z=np.meshgrid(np.linspace(-.5,.5,40),np.linspace(0,1,40))
        front=np.column_stack([np.ones(y.size),y.ravel(),z.ravel()])
        fit,_=rear_boundary(front,np.array([1.,0,0]),front,np.array([[2.,0,.5]]),'storage')
        self.assertIsNone(fit)

    def test_measured_parallel_rear_establishes_depth(self):
        y,z=np.meshgrid(np.linspace(-.5,.5,40),np.linspace(0,1,40))
        front=np.column_stack([np.ones(y.size),y.ravel(),z.ravel()]);rear=front.copy();rear[:,0]=.7
        fit,evidence=rear_boundary(front,np.array([1.,0,0]),rear,np.array([[2.,0,.5]]),'storage')
        self.assertIsNotNone(fit);self.assertAlmostEqual(evidence['measured_rear_depth_m'],.3,places=4)

    def test_wrongly_oriented_rear_cannot_establish_depth(self):
        y,z=np.meshgrid(np.linspace(-.5,.5,40),np.linspace(0,1,40));front=np.column_stack([np.ones(y.size),y.ravel(),z.ravel()])
        x,y=np.meshgrid(np.linspace(.4,.85,40),np.linspace(-.5,.5,40));ground=np.column_stack([x.ravel(),y.ravel(),np.zeros(x.size)])
        fit,_=rear_boundary(front,np.array([1.,0,0]),ground,np.array([[2.,0,.5]]),'storage')
        self.assertIsNone(fit)

    def storage_fixture(self):
        y,z=np.meshgrid(np.linspace(-.5,.5,60),np.linspace(0,1,60))
        face=np.column_stack([np.ones(y.size),y.ravel(),z.ravel()])
        x,z=np.meshgrid(np.linspace(.71,.94,30),np.linspace(0,1,30))
        sides=np.column_stack([x.ravel(),np.full(x.size,.5),z.ravel()])
        fit=(np.eye(3),1.,.3,np.array([1.,-.5,0]),np.array([1.,.5,1.]))
        tracks={'a':{'observations':[{'frame_id':str(i),'name':'cabinet'} for i in range(3)]}}
        return face,fit,[('a',1.)],tracks,np.concatenate([face,sides])

    def test_closed_cabinet_uses_secondary_cached_votes(self):
        name,evidence=storage_identity(*self.storage_fixture(),'shelf')
        self.assertEqual(name,'cabinet');self.assertTrue(evidence['accepted'])

    def test_flat_sheet_does_not_gain_cabinet_identity(self):
        face,fit,support,tracks,_=self.storage_fixture()
        name,_=storage_identity(face,fit,support,tracks,face,'shelf')
        self.assertEqual(name,'shelf')

    def test_open_facade_does_not_gain_closed_cabinet_identity(self):
        face,fit,support,tracks,body=self.storage_fixture()
        face=face[(face[:,2]<.02)|(abs(face[:,2]-.5)<.02)|(face[:,2]>.98)]
        name,_=storage_identity(face,fit,support,tracks,body,'shelves')
        self.assertEqual(name,'shelves')

    def test_repeated_same_frame_is_not_multiview_identity(self):
        face,fit,support,tracks,body=self.storage_fixture()
        for observation in tracks['a']['observations']:observation['frame_id']='one'
        name,_=storage_identity(face,fit,support,tracks,body,'shelf')
        self.assertEqual(name,'shelf')

    def test_storage_geometry_requires_cached_cabinet_category(self):
        face,fit,support,tracks,body=self.storage_fixture()
        for observation in tracks['a']['observations']:observation['name']='wall'
        name,_=storage_identity(face,fit,support,tracks,body,'shelf')
        self.assertEqual(name,'shelf')

    def test_vertical_member_is_accepted(self):
        z=np.linspace(0,.5,60);p=np.column_stack([np.sin(z)*.001,np.zeros(60),z])
        self.assertIsNotNone(line_member(p))

    def test_horizontal_member_and_wide_surface_are_rejected(self):
        z=np.linspace(0,.5,60);p=np.column_stack([z,np.zeros(60),np.zeros(60)])
        self.assertIsNone(line_member(p))
        x,z=np.meshgrid(np.linspace(-.2,.2,20),np.linspace(0,.5,20))
        self.assertIsNone(line_member(np.column_stack([x.ravel(),np.zeros(x.size),z.ravel()])))

    def test_connected_crossbar_does_not_join_two_suspension_members(self):
        z=np.linspace(.2,1.,120)
        rods=np.concatenate([np.column_stack([np.full(120,x),np.zeros(120),z]) for x in [-.4,.4]])
        bar=np.column_stack([np.linspace(-.4,.4,160),np.zeros(160),np.full(160,.2)])
        self.assertEqual(len(ceiling_member_candidates(np.concatenate([rods,bar]),1.02)),2)

    def test_ceiling_surface_cannot_become_a_member(self):
        x,y=np.meshgrid(np.linspace(-.3,.3,30),np.linspace(-.3,.3,30))
        ceiling=np.column_stack([x.ravel(),y.ravel(),np.ones(x.size)])
        self.assertEqual(ceiling_member_candidates(ceiling,1.02),[])

    def test_aligned_endpoints_cannot_prove_missing_shaft(self):
        z=np.r_[np.linspace(0,.03,40),np.linspace(.7,.73,40)]
        self.assertIsNone(line_member(np.column_stack([np.zeros(80),np.zeros(80),z])))

    def test_body_growth_cannot_jump_to_disconnected_roof_mounts(self):
        body=np.column_stack([np.linspace(0,.2,80),np.zeros(80),np.zeros(80)])
        sides=body.copy();sides[:,2]=.01
        roof=body.copy();roof[:,2]=.7
        got=connected_body_points(body,np.concatenate([sides,roof]))
        np.testing.assert_array_equal(got,sides)


if __name__=='__main__':unittest.main()
