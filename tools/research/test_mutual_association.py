"""A partial container must not match a larger object through containment alone."""
from types import SimpleNamespace
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scannet/script'))
from mutual_association import visibility_score,mutual_support
grid=np.linspace(-.2,.2,15)
xx,yy=np.meshgrid(grid,grid)
points=np.c_[xx.ravel(),yy.ravel(),np.ones(xx.size)]
mask=np.zeros((128,128),bool);mask[42:87,42:87]=True
frame=dict(pose=np.eye(4),kd=np.array([[100,0,64],[0,100,64],[0,0,1.]]),
    kc=np.array([[100,0,64],[0,100,64],[0,0,1.]]),depth=np.full((128,128),1000,np.uint16),scale=1000.)
ob=SimpleNamespace(record={},points=points,mask=mask,color=np.ones((3,8))/8)
track=SimpleNamespace(points=points.copy(),color=np.ones((3,8))/8)
full=visibility_score(ob,track,frame)
assert full is not None and full['mutual_visible_surface']==1
hidden=points.copy();hidden[:,2]=2
track.points=np.r_[points,hidden]
occluded=visibility_score(ob,track,frame)
assert occluded is not None and occluded['mutual_visible_surface']==1
# A small fragment of the same observed surface is insufficient whole identity.
fragment=points[0]+np.c_[np.linspace(0,.002,20),np.zeros(20),np.zeros(20)]
small=SimpleNamespace(record={},points=fragment,mask=mask,color=np.ones((3,8))/8)
assert visibility_score(small,SimpleNamespace(points=points,color=np.ones((3,8))/8),frame) is None
for x,y in [(0,.8),(.2,.9),(.9,.2),(1,1)]:
 assert mutual_support(x,y)==mutual_support(y,x)
# World translation changes no evidence when the camera is translated as well.
offset=np.array([8.,-3.,2.]);translated=dict(frame);translated['pose']=frame['pose'].copy();translated['pose'][:3,3]=offset
shift=visibility_score(SimpleNamespace(record={},points=points+offset,mask=mask,color=np.ones((3,8))/8),
 SimpleNamespace(points=points+offset,color=np.ones((3,8))/8),translated)
assert shift is not None and shift['mutual_visible_surface']==full['mutual_visible_surface']
print('MUTUAL_ASSOCIATION_TEST_PASSED: symmetry, containment rejection, occlusion unknown, world translation')
