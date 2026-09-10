from __future__ import annotations
import cv2
import numpy as np
from scipy.optimize import least_squares


def project_camera(points,rvec,tvec,K,D,model):
    obj=np.asarray(points,np.float64).reshape(1,-1,3)
    if model=="fisheye":return cv2.fisheye.projectPoints(obj,rvec,tvec,K,D.reshape(4,1))[0].reshape(-1,2)
    return cv2.projectPoints(obj,rvec,tvec,K,D)[0].reshape(-1,2)


def normalized_camera_points(points,K,D,model):
    p=np.asarray(points,np.float64).reshape(-1,1,2)
    return (cv2.fisheye.undistortPoints(p,K,D.reshape(4,1)) if model=="fisheye" else cv2.undistortPoints(p,K,D)).reshape(-1,2)


def solve(lidar_xyz,lattice_indices,camera_grid,board,K,D,model,opt_cfg):
    cols,rows=map(int,board);base=camera_grid.reshape(rows,cols,2);solutions=[]
    variants=[("identity",base),("flip_x",base[:,::-1]),("flip_y",base[::-1]),("flip_xy",base[::-1,::-1])]
    for name,array in variants:
        image_points=array.reshape(-1,2)[lattice_indices]
        normalized=normalized_camera_points(image_points,K,D,model)
        ok,rvec,tvec=cv2.solvePnP(lidar_xyz,normalized,np.eye(3),None,flags=cv2.SOLVEPNP_EPNP)
        if not ok:continue
        def residual(x):return (project_camera(lidar_xyz,x[:3],x[3:],K,D,model)-image_points).ravel()
        x0=np.r_[rvec.ravel(),tvec.ravel()]
        initial_errors=np.linalg.norm(residual(x0).reshape(-1,2),axis=1)
        # The paper uses Levenberg-Marquardt after E-PnP.  Residuals are kept
        # in distorted camera pixels so the reported error is directly useful.
        fit=least_squares(residual,x0,method="lm",max_nfev=int(opt_cfg["maximum_iterations"]),
                          ftol=float(opt_cfg.get("ftol",1e-10)),xtol=float(opt_cfg.get("xtol",1e-10)),
                          gtol=float(opt_cfg.get("gtol",1e-10)))
        rotation=cv2.Rodrigues(fit.x[:3])[0];camera_xyz=lidar_xyz@rotation.T+fit.x[3:]
        if np.mean(camera_xyz[:,2]>0)<.9:continue
        errors=np.linalg.norm(residual(fit.x).reshape(-1,2),axis=1)
        solutions.append((float(np.sqrt(np.mean(errors**2))),name,rotation,fit.x[3:],image_points,errors,
                          float(np.sqrt(np.mean(initial_errors**2)))))
    if not solutions:raise RuntimeError("E-PnP did not produce a physically valid camera pose")
    return min(solutions,key=lambda x:x[0])
