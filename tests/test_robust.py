import unittest
import cv2
import numpy as np
from single_shot_calib.config import DEFAULTS
from single_shot_calib.robust import fit_pattern
from single_shot_calib.solver import project_camera,solve_robust


def synthetic_board(cols,rows,square,rng):
    """Dense LiDAR-like points on a 2.5 m board plane with checker reflectivity."""
    sq=(cols+1,rows+1);uv=rng.uniform([-.6*sq[0]*square,-.6*sq[1]*square],[.6*sq[0]*square,.6*sq[1]*square],(40000,2))
    a=uv[:,0]/square+sq[0]/2;b=uv[:,1]/square+sq[1]/2;inside=(a>0)&(a<sq[0])&(b>0)&(b<sq[1])
    dark=((np.floor(a)+np.floor(b))%2==0)&inside  # corner squares are black
    intensity=np.where(dark,10.0,np.where(inside,120.0,60.0))+rng.normal(0,5,len(uv))
    e1,e2,n=np.array([0,1.,0]),np.array([0,0,1.]),np.array([-1.,0,0]);centre=np.array([2.5,.1,-.2])
    return centre+uv[:,:1]*e1+uv[:,1:]*e2,intensity,n,centre,e1,e2


class RobustTest(unittest.TestCase):
    def test_pattern_fit_recovers_corners(self):
        cols,rows,square=4,6,.06;rng=np.random.default_rng(0)
        pts,inten,n,c,e1,e2=synthetic_board(cols,rows,square,rng)
        fit=fit_pattern(pts,inten,n,(cols,rows),DEFAULTS["robust"])
        truth=np.array([c+(i-(cols+1)/2)*square*e1+(j-(rows+1)/2)*square*e2 for j in range(1,rows+1) for i in range(1,cols+1)])
        # The symmetric pattern may be returned in any of its equivalent orderings: compare as sets.
        err=np.linalg.norm(fit["corners_xyz"][:,None]-truth[None],axis=2).min(1).max()
        self.assertLess(err,0.004)
        self.assertAlmostEqual(fit["square_size_m"],square,delta=.002)
        self.assertGreater(fit["cell_coverage"],.95)

    def test_symmetric_board_needs_orientation_hint(self):
        cols,rows=4,6
        obj=np.array([[.06*i,.06*j,0.] for j in range(rows) for i in range(cols)]);obj-=obj.mean(0)
        board_R=cv2.Rodrigues(np.array([.1,-.2,.05]))[0];xyz=obj@board_R.T+np.array([2.4,.1,-.3])
        true_R=np.array([[0,-1.,0],[0,0,-1],[1,0,0]]);true_t=np.array([.05,-.02,.01])
        K=np.array([[700.,0,640.],[0,700.,360.],[0,0,1.]]);D=np.zeros(5)
        pixels=project_camera(xyz,cv2.Rodrigues(true_R)[0],true_t,K,D,"pinhole")
        normal=board_R[:,2]
        cfg=dict(DEFAULTS["robust"]);opt={"maximum_iterations":300}
        best,sols,notes=solve_robust(xyz,np.arange(len(xyz)),pixels,(cols,rows),K,D,"pinhole",opt,cfg,normal)
        self.assertTrue(any("symmetric board" in n for n in notes))
        self.assertEqual(sum(s["facing"] for s in sols),2)  # the two mirrored orderings are rejected
        hint=cv2.Rodrigues(np.array([.2,.1,-.15]))[0]@true_R  # rough (~15 deg) mounting knowledge
        cfg["approx_rotation_lidar_to_camera"]=hint.tolist()
        best,_,notes=solve_robust(xyz,np.arange(len(xyz)),pixels,(cols,rows),K,D,"pinhole",opt,cfg,normal)
        np.testing.assert_allclose(best["R"],true_R,atol=1e-5)
        np.testing.assert_allclose(best["t"],true_t,atol=1e-5)
        self.assertFalse(any("symmetric board" in n for n in notes))


if __name__=="__main__":unittest.main()
