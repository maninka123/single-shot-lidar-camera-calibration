import unittest
import cv2
import numpy as np
from single_shot_calib.solver import project_camera,solve


class SolverTest(unittest.TestCase):
    def test_recovers_synthetic_transform(self):
        cols,rows=6,9
        xyz=np.array([[.04*i,.04*j,0.0] for j in range(rows) for i in range(cols)],float)
        true_rvec=np.array([.08,-.12,.04]);true_t=np.array([-.1,-.15,2.2])
        K=np.array([[700.,0,640.],[0,700.,360.],[0,0,1.]])
        pixels=project_camera(xyz,true_rvec,true_t,K,np.zeros(5),"pinhole")
        result=solve(xyz,np.arange(len(xyz)),pixels,(cols,rows),K,np.zeros(5),"pinhole",
                     {"maximum_iterations":300,"ftol":1e-10,"xtol":1e-10,"gtol":1e-10})
        rms,_,rotation,translation,*_=result
        self.assertLess(rms,1e-5)
        np.testing.assert_allclose(rotation,cv2.Rodrigues(true_rvec)[0],atol=1e-5)
        np.testing.assert_allclose(translation,true_t,atol=1e-5)


if __name__=="__main__":unittest.main()
