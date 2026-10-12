"""Occlusion is unknown; only independent observed support changes geometry."""
import numpy as np
from projection_verification import refine,project
def run():
 before=np.array([True,True,False,False]);visible=np.array([0,2,2,1]);positive=np.array([0,0,2,1])
 assert np.array_equal(refine(before,positive,visible),[True,False,True,False])
 xyz=np.array([[0,0,1],[0,0,2]]);pose=np.eye(4);k=np.eye(3);depth=np.ones((2,2))
 indices,_=project(xyz,pose,k,k,depth,1,2,2);assert indices.tolist()==[0]
 rotated=np.array([[0,-1,0],[1,0,0],[0,0,1]])
 changed=pose.copy();changed[:3,:3]=rotated;changed[:3,3]=[2,3,4]
 indices2,_=project(xyz@rotated.T+[2,3,4],changed,k,k,depth,1,2,2);assert np.array_equal(indices,indices2)
 print('PROJECTION_INDEPENDENT_EVIDENCE_AND_OCCLUSION_CONTRACTS_PASS')
if __name__=='__main__':run()
