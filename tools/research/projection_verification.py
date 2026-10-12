"""Depth-visible mesh projection and conservative independent-view consensus."""
import numpy as np

def project(points,pose,kd,kc,depth,scale,height,width):
 camera=(np.asarray(points)-pose[:3,3])@pose[:3,:3];z=camera[:,2];safe=np.maximum(z,1e-8)
 uv=np.rint(camera[:,:2]/safe[:,None]*[kc[0,0],kc[1,1]]+[kc[0,2],kc[1,2]]).astype(int)
 duv=np.rint(camera[:,:2]/safe[:,None]*[kd[0,0],kd[1,1]]+[kd[0,2],kd[1,2]]).astype(int)
 valid=(z>0)&(uv[:,0]>=0)&(uv[:,1]>=0)&(uv[:,0]<width)&(uv[:,1]<height)
 valid&=(duv[:,0]>=0)&(duv[:,1]>=0)&(duv[:,0]<depth.shape[1])&(duv[:,1]<depth.shape[0])
 ix=np.flatnonzero(valid);measured=depth[duv[ix,1],duv[ix,0]]/scale;tolerance=.025+.01*measured
 accept=(measured>0)&(np.abs(z[ix]-measured)<=.3*tolerance)
 return ix[accept],uv[ix][accept]

def refine(original,positive,visible):
 original=np.asarray(original,bool);positive=np.asarray(positive,int);visible=np.asarray(visible,int)
 if positive.shape!=original.shape or visible.shape!=original.shape or np.any(positive<0) or np.any(positive>visible):raise ValueError('Independent-view evidence counts are inconsistent')
 added=(positive>=2)&(positive/np.maximum(visible,1)>=2/3)
 rejected=(visible>=2)&(positive==0)
 return (original&~rejected)|added
